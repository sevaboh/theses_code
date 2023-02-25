#define NEED_OPENCL
#include <stdio.h>
#include <stdlib.h>
#include <vector>
//#include <windows.h>
#include "../sarpok3d/include/sarpok3d.h"
#define AK 20 // number of terms in approximation series
#define MAXAK 40 // maximal number of terms in approximation series
int BS=10; // block size
int NB=5; // number of block
int N;  // number of nodes
int double_ext=0;
int device = 0;
double A = 0.85;
double approx_eps = 1e-5;
double eps = 1e-7;
extern double Gamma(double);
extern double log2(double);
extern void pcr_solver(cl_mem A, cl_mem B, cl_mem C, cl_mem R, cl_mem X, int , cl_command_queue *q,int ver,cl_mem A2, cl_mem B2, cl_mem C2, cl_mem R2);
extern int pcr_small_systems_init(cl_context cxGPUContext,int id);
#define GetTickCount _GetTickCount
////////////////////////////////////////////////////////
// approximations of g(x,r)=(x-r+1)^a-(x-r)^a
////////////////////////////////////////////////////////
// ia approximation gives a_AK<approx_eps for i>ia_mini(r)
double ia_mini(double r,int nterms)
{
	double ak = 1;
	for (int i = 1;i<nterms;i++)
		ak *= (A - i + 1) / i;
	return pow(abs(ak*(pow(1 - r, (double)nterms) - pow(-r, (double)nterms)) / approx_eps), 1.0 / (nterms - A));
}
class ia_approximation
{
public:
	double min;
	int nterms;
	double c[MAXAK];
	double calc(double x)
	{
		double ret=0.0;
		for (int i=0;i<nterms;i++)
			ret+=c[i]*pow(x,A-(double)i);
		return ret;
	}
	ia_approximation(double rmin,double rmax,double *F,double cmin=-1,int nt=AK) 
	// approximate S=sum_r=rmin..rmax(g(x,r)*F[r]):
	// approximate g(x,r) by sum_k=0...inf(a*(a-1)*...*(a-k+1)/k!*x^(a-k)*((1-r)^k-(-r)^k))=sum_k=0...inf(K(r)_k*x^(a-k))
	// then S is approximated by sum_k=0...inf(sum_r=rmin..rmax(K(r)_k*F[r])*x^(a-k))
	{
		double ak=1,krk;
		c[0]=0;
		for (int i=1;i<nt;i++)
		{
			ak*=(A-i+1)/i;
			krk=0.0;
			for (int j=rmin;j<rmax;j++)
				krk+=F[j]*(pow(1.0-(double)j,(double)i)-pow(-(double)j,(double)i));
			c[i]=ak*krk;
		}
		for (int i = nt;i < MAXAK;i++)
			c[i] = 0.0;
		if (cmin != -1)
			min = cmin;
		else
			min = ceil(ia_mini(rmax,nt));
		nterms = nt;
	}
	ia_approximation(char *m)
	{
		memcpy(this,m,sizeof(ia_approximation));
	}
};
// ibk approximation gives a_AK<approx_eps for b-ibk_i2(r,b)<i<b+ibk_i2(r,b)
double ibk_i2(double r, double b,int nterms)
{
	double ak = 1;
	for (int i = 1;i<nterms;i++)
		ak *= (A - i + 1) / i;
	if (b==r)
		return pow(abs(approx_eps / ak), 1.0 / nterms);
	return pow(abs(approx_eps / (ak*(pow(1 + b - r, A - nterms) - pow(b - r, A - nterms)))),1.0/nterms);
}
class ibk_approximation
{
public:
	double min,max;
	double b;
	int nterms;
	int old; // 1 if newer overlaping approximation exists in the set
	double c[MAXAK];
	double calc(double x)
	{
		double ret=0.0,xbk=1;
		for (int i=0;i<nterms;i++)
		{
			ret+=c[i]*xbk;
			xbk*=(x-b);
		}
		return ret;
	}
	ibk_approximation(double rmin,double rmax,double *F,double _b,double i2=-1,int nt=AK) 
	// approximate S=sum_r=rmin..rmax(g(x,r)*F[r]):
	// approximate g(x,r) by sum_k=0...inf(a*(a-1)*...*(a-k+1)/k!*(x-b)^k*((1+b-r)^(a-k)-(b-r)^(a-k)))=sum_k=0...inf(K(r)_k*(x-b)^k)
	// then S is approximated by sum_k=0...inf(sum_r=rmin..rmax(K(r)_k*F[r])*(x-b)^k)
	{
		double ak=1,krk;
		b=_b;
		for (int i=0;i<nt;i++)
		{
			if (i!=0) ak*=(A-i+1)/i;
			krk = 0.0;
			for (int j=rmin;j<rmax;j++)
				krk+=F[j]*(pow(1.0+b-(double)j,A-(double)i)-pow(b-(double)j,A-(double)i));
			c[i]=ak*krk;
		}
		for (int i = nt;i < MAXAK;i++)
			c[i] = 0.0;
		if (i2 != -1)
			min = i2;
		else
			min = b - floor(ibk_i2(rmax, b,nt));
		max=b;
		nterms = nt;
		old = 0;
	}
	ibk_approximation(ibk_approximation *i)
	{
		min = i->min;
		max = i->max;
		b = i->b;
		nterms = i->nterms;
		for (int j = 0;j < MAXAK;j++)
			c[j] = i->c[j];
		old = i->old;
	}
	ibk_approximation(char *m)
	{
		memcpy(this,m,sizeof(ibk_approximation));
	}
};
double *F; // function values
double *BVA; // approximated values of F=sum_r=1...x-1(g(x,r)*F[r])
double *BVK; // values of F
// calculate values of F for block
void calc_v(int block,double *F,double *V)
{
	for (int i = 0;i<BS;i++)
		V[i] = 0.0;
	if (A == 1.0) return;
	for (int i=block*BS+1;i<(block+1)*BS+1;i++)
	{
		double v=0.0;
		for (int j=1;j<=i-1;j++)			
			v+=(pow((double)i-(double)j+1,A)-pow((double)i-(double)j,A))*F[j];
		V[i-(block*BS+1)]=v;
	}
}
// current approximation
std::vector<ia_approximation *> ias; // "long" approximations
std::vector<ibk_approximation *> ibks;	// "short" approximations
// calculate approximated values of F for block
// approximations for previous blocks must be built so function must be called in loop from 0 to NB-1
void calc_va(int block,double *F,double *V)
{
	ibk_approximation *bibk;
	ia_approximation *bia;
	int nadd=0, nsplit=0, ndel=0,nnew=0;
	long long t1, t2;
	double bm1 = -1;
	int rmax = ((block + 1)*BS + 1);
	t1 = GetTickCount();
	for (int i=0;i<BS;i++)
		V[i]=0.0;
	if (A == 1.0) return;
	// calculate F for r=1,...,block*BS+1 using current approximation
	for (int i=block*BS+1;i<(block+1)*BS+1;i++)
	{
		for (int j = 0;j<ias.size();j++)
			if (i >= ias[j]->min)
				V[i - (block*BS + 1)] += ias[j]->calc(i);
		for (int j = 0;j<ibks.size();j++)
			if ((i >= ibks[j]->min)&& (i <= ibks[j]->max))
				V[i - (block*BS + 1)] += ibks[j]->calc(i);
	}
	// add F for r=block*BS+1,...,(block+1)*BS
	for (int i=block*BS+1;i<(block+1)*BS+1;i++)
	{
		double v=0.0;
		for (int j=block*BS+1;j<=i-1;j++)
			v+=(pow((double)i-(double)j+1,A)-pow((double)i-(double)j,A))*F[j];
		V[i-(block*BS+1)]+=v;
	}
	t1 = GetTickCount() - t1;
	t2 = GetTickCount();
	// 1) remove ibks with max<rmax
	for (int j = 0;j < ibks.size();j++)
		if (ibks[j]->max < rmax)
		{
			delete ibks[j];
			ibks.erase(ibks.begin() + j);
			j--;
			ndel++;
		}
	// 2) append current block approximation to ibks with min>rmax
	int ss = ibks.size();
	for (int j = 0;j < ss;j++)
		if (ibks[j]->min >= rmax)
		if (ibks[j]->old==0)
		{
			ibk_approximation *i2;
			int spl = 0;
			bibk = new ibk_approximation(block*BS + 1, (block + 1)*BS + 1, F, ibks[j]->b);
			if (bibk->min > ibks[j]->min)
			{
				int a;
				// increase number of terms 
				for (a = AK;a < MAXAK;a++)
					if ((ibks[j]->b - ibk_i2(rmax, ibks[j]->b, a)) < ibks[j]->min)
						break;
				if (a == MAXAK)
					spl = 1;
				else
				{
					delete bibk;
					bibk = new ibk_approximation(block*BS + 1, (block + 1)*BS + 1, F, ibks[j]->b, -1.0, a);
				}
				if (spl)
					i2 = new ibk_approximation(ibks[j]);
			}
			for (int i = 0;i<bibk->nterms;i++)
				ibks[j]->c[i] += bibk->c[i];
			if (ibks[j]->nterms < bibk->nterms)
				ibks[j]->nterms = bibk->nterms;
			nadd++;
			// split ibks into two parts if current block ibk min is bigger that ibks min
			if (spl)
				if (bibk->min > ibks[j]->min)
				{
					i2->max = ceil(bibk->min);
					i2->old = 1;
					ibks[j]->min = i2->max + 1;
					ibks.push_back(i2);
					delete bibk;
					if (ibks[j]->min > ibks[j]->max)
					{
						i2->max = ibks[j]->max;
						delete ibks[j];
						ibks.erase(ibks.begin() + j);
						j--;
						ss--;
						ndel++;
					}
					// create serie of approximations for i2->min,i2->max
					double rr = i2->min;
					double bb0 = rr, bb = rr;
					do
					{
						// find bj>rj:bj-i2(r,bj,k)/2=rj+1
						bb = rr + 1;
						do
						{
							bb0 = bb;
							bb = rr + ibk_i2(i2->min, bb, AK) + 1;
						} while (abs(bb - bb0) > eps);
						// build ibk approximation 
						bibk = new ibk_approximation(block*BS + 1, (block + 1)*BS + 1, F, floor(bb), rr);
						ibks.push_back(bibk);
						nnew++;
						// move r to bj+i2(r,bj,k)
						rr = floor(bb) + 1;
						if (floor(bb) > i2->max)
							bibk->max = i2->max;
					} 
					while (floor(bb) < i2->max);
					nsplit++;
				}
		}
	// 3) build block j approximations for ibks with min<rmax and max>rmax and for splitted ibks with min>rmax
	for (int j = 0;j < ibks.size();j++)
		if (ibks[j]->max >= rmax)
			if (ibks[j]->min < rmax)
				if (ibks[j]->max>bm1)
				if (ibks[j]->old==0)
				{
					bm1 = ibks[j]->max;
					ibks[j]->old = 1;
				}
	if (bm1!=-1)
	{
		for (int j = 0;j < ibks.size();j++)
			if (ibks[j]->max < bm1)
				ibks[j]->old = 1;
		double rr = rmax;
		double bb0=rr,bb = rr;
		do
		{
			// find bj>rj:bj-i2(r,bj,k)/2=rj+1
			bb = rr+1;
			do
			{
				bb0 = bb;
				bb = rr + ibk_i2(rmax, bb,AK)+1;
			} while (abs(bb - bb0)>eps);
			// build ibk approximation 
			bibk = new ibk_approximation(block*BS + 1, (block + 1)*BS + 1, F, floor(bb),rr);
			ibks.push_back(bibk);
			nnew++;
			// move r to bj+i2(r,bj,k)
			rr = floor(bb)+1;
			if (floor(bb) > bm1)
				bibk->max = bm1;
		} 
		while (floor(bb) < bm1);
	}
	// 4) build ia approximation and add it to last ia approximation (with biggest min) if current ia.min< lastia.min
	bia=new ia_approximation(block*BS+1,(block+1)*BS+1,F);
	if (ias.size())
		if (bia->min < ias[ias.size() - 1]->min)
		{ 
			for (int i = 0;i<bia->nterms;i++)
				ias[ias.size()-1]->c[i] += bia->c[i];
			if (ias[ias.size() - 1]->nterms < bia->nterms)
				ias[ias.size() - 1]->nterms = bia->nterms;
			nadd++;
			delete bia;
			bia = NULL;
		}
	// 5) build ibk approximation serie if current ia.min>lastia.min, set current ia minimum as ibks maximum and add it to ias
	if (bia)
	{
		bm1 = rmax;
		for (int j = 0;j < ibks.size();j++)
			if ((ibks[j]->max+1)>bm1)
				bm1 = ibks[j]->max+1;
		double rr = bm1;
		double bb0 = rr, bb = rr;
		do
		{
			// find bj>rj:bj-i2(r,bj,k)=rj+1
			bb = rr+1;
			do
			{
				bb0 = bb;
				bb = rr + ibk_i2(rmax, bb,AK)+1;
			} while (abs(bb-bb0)>eps);
			// build ibk approximation 
			bibk = new ibk_approximation(block*BS + 1, (block + 1)*BS + 1, F, floor(bb), rr);
			ibks.push_back(bibk);
			nnew++;
			// move r to bj+i2(r,bj,k)
			rr = floor(bb)+1;
		} while (floor(bb) < ceil(bia->min));
		bia->min = rr;
		ias.push_back(bia);
		nnew++;
	}
	t2 = GetTickCount() - t2;
}
void test_gF()
{
	double sumabs = 0.0;
	double sumsize = 0.0;
	// fill function values
	for (int i=0;i<N;i++)
		F[i]=sin((i+1)/50.0)*log((double)(i+1));
	// do check
	for (int i=0;i<NB;i++)
	{
		double abserr = 0.0,maxrelerr=0.0;
		long long t1, t2;
		t1 = GetTickCount();
		//calc_v(i,F,BVK);
		t1 = GetTickCount() - t1;
		t2 = GetTickCount();
		calc_va(i,F,BVA);
		t2 = GetTickCount() - t2;
		printf("block %d i %d v %g a %g\n",i, i*BS +  1, BVK[0], BVA[0]);
		printf("block %d i %d v %g a %g\n",i, i*BS + (BS - 1) + 1, BVK[(BS - 1)], BVA[(BS - 1)]);
		for (int j = 0;j < BS;j++)
		{
			double rel;
			abserr += (BVA[j] - BVK[j])*(BVA[j] - BVK[j]);
			if (BVK[j])
			{
				rel = (BVA[j] - BVK[j]) / BVK[j];
				if (rel < 0) rel = -rel;
				if (rel > maxrelerr) maxrelerr = rel;
			}
			//printf("%d %g %g\n", i*BS + j + 1, BVK[j], BVA[j]);
		}
		sumabs += abserr;
		sumsize += ibks.size()*sizeof(ibk_approximation) + ias.size()*sizeof(ia_approximation);
		if ((i%10)==0)
		{
			double send_est;
			double d1 = (AK - 1) / (AK -A);
			double nk = 1;
			for (int i = 1;i < AK;i++)
				nk *= (A - i + 1) / i;
			double g1 = pow(abs(nk / approx_eps)*AK, 1.0 / (AK - A));
			double g2 = pow(abs(approx_eps / nk), 1.0 / AK);
			double i1max = pow(g1, 1.0 / (1.0 - d1)) - BS;
			if (i1max > i*BS) i1max = i*BS;
			double m1max = ((1.0 / log(d1))*log(1.0 - (1.0 - d1)*log(i1max) / log(g1)) - 1);
			double m2max = log(i*BS - 1.0) / log((1 + g2) / (1 - g2));
			send_est = AK*i*BS*m1max*(1.0 + 2.0*m2max);
			printf("block %d errs: abs %g maxrel %g t1 %lld tapp %lld sends %g sends est %g\n", i, abserr, maxrelerr, t1, t2, BS*sumsize,send_est);
		}
	}
	{
		double send_est;
		double d1 = (AK - 1) / (AK - A);
		double nk = 1;
		for (int i = 1;i < AK;i++)
			nk *= (A + i - 1) / i;
		double g1 = pow(abs(nk / approx_eps)*AK, 1.0 / ( AK - A));
		double g2 = pow(abs(approx_eps / nk), 1.0 / AK);
		double i1max = pow(g1, 1.0 / (1.0 - d1)) - BS;
		if (i1max > N) i1max = N;
		double m1max = ((1.0 / log(d1))*log(1.0 - (1.0 - d1)*log( i1max) / log(g1)) - 1);
		double m2max = log(N - 1.0) / log((1 + g2) / (1 - g2));
		send_est = AK*N*m1max*(1.0 + 2.0*m2max);
		printf("total err %g 2d sends est %g sends %g n1 (1cpn) %g n1(4cpn) %g n1(8cpn) %g n2 %g\n", sumabs, send_est,BS*sumsize, (double)(BS*BS*(0.5 * 1 * (1 + 1) + 1 * (NB - 2 * 1))*sizeof(double)), (double)(BS*BS*(0.5 * 4 * (4 + 1) + 4 * (NB - 2 * 4))*sizeof(double)), (double)(BS*BS*(0.5 * 8 * (8 + 1) + 8 * (NB - 2 * 8))*sizeof(double)), (double)(BS*N*sizeof(double)*NB*(NB - 1) / 2));
	}
}
////////////////////////////////////////////
// space-fractional solver 1d /////////////
////////////////////////////////////////////
class space_fract_solver {
public:
	double *U; // pressure
	double *V; // velocity
	double *C; // concentration
	double *Al; // alpha coefficients
	double *Bt; // beta coefficients
	double *Om; // right part
	std::vector<double*> oldC; // for time-fractional derivatives in C-equation
	double delta, s1;
	double h1;
	double delta2,delta3,tg,g3;
	double Cm,kmm;

	double tau;
	double w;
	double alpha, beta,gamma;
	double d;
	double H0;
	double sigma;
	void init()
	{
		U=new double[N+2];
		V=new double[N+2];
		C=new double[N+2];
		Al=new double[N+2];
		Bt=new double[N+2];
		Om=new double[N+2];

		tau = (1.0 / N);
		w = 0.25e-3;
		d = 0.06e-2;
		H0 = 0.95;
		sigma = 0.2;
		Cm = 1;
		kmm = 0.01;

		h1 = 1.0 / N;
		delta = pow(h1, 1.0 - alpha) / Gamma(2.0 - alpha);
		delta2 = pow(h1, 1.0 - beta) / Gamma(2.0 - beta);
		delta3 = 1.0 / (pow(tau, gamma)*Gamma(2.0 - gamma));
		g3 = Gamma(2.0 - gamma);
		tg = pow(tau, 1.0 - gamma);
		s1 = 2.0 + h1*h1 / (tau*delta);		
		for (int i = 0;i < N + 1;i++)
		{
			U[i] = H0;
			C[i] = 1.0;
		}
		C[0] = 0.0;
		U[0] = 1.0;
	}
	// U equation alpha coeffients
	void al1()
	{
		Al[1] = 0;
		for (int i = 1;i < N;i++)
			Al[i + 1] = 1.0 / (s1 - Al[i]);
	}
	// U equation right part
	void Om1(int approx)
	{
		for (int i = 1;i < N;i++)
			F[i] = U[i + 1] - 2.0*U[i] + U[i - 1];
		for (int i = 0;i < NB;i++)
		{
			if (approx == 0)
				calc_v(i, F, BVK);
			else
				calc_va(i, F, BVK);
			for (int j = 0;j < BS;j++)
				Om[1 + i*BS + j] = -(h1*h1 / delta)*((U[1 + i*BS + j] / tau) + delta*BVK[j] / (h1*h1));
		}
		if (approx)
		{
			for (int i = 0;i < ias.size();i++)
				delete ias[i];
			ias.clear();
			for (int i = 0;i < ibks.size();i++)
				delete ibks[i];
			ibks.clear();
		}
	}
	// U equation beta coeffients
	void bt1()
	{
		Bt[1] = 1;
		for (int i = 1;i < N;i++)
			Bt[i + 1] = Al[i + 1] * (Bt[i] - Om[i]);
	}
	// calc U
	void U1(int approx)
	{
		A = 1 - alpha;
		al1();
		Om1(approx);
		bt1();
		U[N] = H0;
		U[0] = 1.0;
		for (int i = N - 1;i >= 1;i--)
			U[i] = Al[i + 1] * U[i + 1] + Bt[i + 1];
	}
	// calc V
	void V1(int approx)
	{
		V[0] = 0;
		V[1] = -(w / (h1*Gamma(2.0 - alpha)))*(U[1] - U[0]);
		for (int i = 1;i < N;i++)
			F[i] = U[i] - U[i - 1];
		for (int i = 0;i < NB;i++)
		{
			if (approx == 0)
				calc_v(i, F, BVK);
			else
				calc_va(i, F, BVK);
			for (int j = 0;j < BS;j++)
				if ((i != 0) || (j != 0))
					V[1 + i*BS + j] = -(w / (h1*Gamma(2.0 - alpha)))*(U[1 + i*BS + j] - U[i*BS + j] + BVK[j]);
		}
		if (approx)
		{
			for (int i = 0;i < ias.size();i++)
				delete ias[i];
			ias.clear();
			for (int i = 0;i < ibks.size();i++)
				delete ibks[i];
			ibks.clear();
		}
	}
	// C equation alpha coefficient
	void al2()
	{
		Al[1] = 0;
		for (int i = 1;i < N;i++)
			Al[i + 1] = ((delta2*d / (h1*h1)) - (V[i] / (2.0*h1))) / (((sigma *delta3) + (2.0*delta2*d / (h1*h1)) + ((V[i + 1] - V[i - 1]) / (2.0*h1))+kmm) - Al[i] * ((delta2*d / (h1*h1)) + (V[i] / (2.0*h1))));
	}
	// C equation right part
	void Om2(int approx)
	{
		for (int i = 1;i < N;i++)
			F[i] = C[i + 1] - 2.0*C[i] + C[i - 1];
		for (int i = 0;i < NB;i++)
		{
			if (approx == 0)
				calc_v(i, F, BVK);
			else
				calc_va(i, F, BVK);
			for (int j = 0;j < BS;j++)
			{
				Om[1 + i*BS + j] = -(sigma *delta3)*C[1 + i*BS + j] - d*delta2*BVK[j] / (h1*h1);
				// time-fractional derivative
				double time_sum = 0;
				if (oldC.size()>=2)
				for (int t = 0;t < oldC.size() - 1;t++)
					time_sum += (pow((double)(oldC.size() - t ), 1.0 - gamma) - pow((double)(oldC.size() - t-1), 1.0 - gamma))*(oldC[t + 1][1 + i*BS + j] - oldC[t][1 + i*BS + j]);
				Om[1 + i*BS + j] += tg*(sigma/g3)*time_sum/tau;
				Om[1 + i*BS + j] -= Cm*kmm;
			}
		}
		if (approx)
		{
			for (int i = 0;i < ias.size();i++)
				delete ias[i];
			ias.clear();
			for (int i = 0;i < ibks.size();i++)
				delete ibks[i];
			ibks.clear();
		}
	}
	void bt2()
	{
		Bt[1] = 0;
		for (int i = 1;i < N;i++)
			Bt[i + 1] = (Al[i + 1] / ((delta2*d / (h1*h1)) - (V[i] / (2.0*h1)))) * (((delta2*d / (h1*h1)) + (V[i] / (2.0*h1)))*Bt[i] - Om[i]);
	}
	// calc C
	void C1(int approx)
	{
		A = 1 - beta;
		al2();
		Om2(approx);
		bt2();
		C[N] = 1.0;
		C[0] = 0.0;
		for (int i = N - 1;i >= 1;i--)
			C[i] = Al[i + 1] * C[i + 1] + Bt[i + 1];
		// save C for time-fractional derivative calculations
		double *cs = new double[N + 2];
		memcpy(cs, C, (N + 2)*sizeof(double));
		oldC.push_back(cs);
	}
	~space_fract_solver()
	{
		for (int i = 0;i < oldC.size();i++)
			delete oldC[i];
	}
};
void solve(double t,double save_tau)
{
	space_fract_solver s1, s2,s3;
	int nsave = 1;
	FILE *fi1, *fi2;
	fi1 = fopen("log.txt", "wt");
	fi2 = fopen("results.txt", "wt");
	s1.alpha = s2.alpha = s3.alpha=0.8;
	s1.beta = s2.beta = s2.beta = 0.9;
	s1.gamma = 1.0;
	s2.gamma = 0.9;
	s3.gamma = 0.8;
	s1.init();
	s2.init();
	s3.init();
	s2.kmm = 0;
	for (double tt = 0;tt < t;tt += s1.tau)
	{
		double err1=0.0,err2=0.0,err3=0.0;
		long long i1, i2, i3;
		i1 = GetTickCount();
		s1.U1(1);
		s1.V1(1);
		s1.C1(1);
		i1 = GetTickCount()-i1;
		i2 = GetTickCount();
		s2.U1(1);
		s2.V1(1);
		s2.C1(1);
		i2 = GetTickCount() - i2;
		i3 = GetTickCount();
		s3.U1(1);
		s3.V1(1);
		s3.C1(1);
		i3 = GetTickCount() - i3;
		// calculate errors
		for (int i = 0;i < N + 1;i++)
		{
			err1 += (s2.U[i] - s1.U[i])*(s2.U[i] - s1.U[i]);
			err2 += (s2.V[i] - s1.V[i])*(s2.V[i] - s1.V[i]);
			err3 += (s2.C[i] - s1.C[i])*(s2.C[i] - s1.C[i]);
		}
		fprintf(fi1, "t %g time0 %lld time1 %lld time2 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, i3, err1, err2, err3);
		printf("t %g time0 %lld time1 %lld time2 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, i3, err1, err2, err3);
		fflush(stdout);
		// save result
		if (tt > nsave*save_tau)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g f %g %g %g a %g %g %g c %g %g %g\n", tt,(double)i/N, s1.U[i], s1.V[i], s1.C[i], s2.U[i], s2.V[i], s2.C[i],s3.U[i],s3.V[i],s3.C[i]);
			nsave++;
		}
	}
}
OpenCL_program *prg;
OpenCL_commandqueue *queue;
OpenCL_prg *prog;
OpenCL_kernel *kUA,*kUB,*kUC,*kUr,*kV,*kCA,*kCB,*kCC,*kCr;
OpenCL_buffer *bA,*bB,*bC,*br,*bU,*bV,*bCn,*bS,*bprevC;
OpenCL_buffer *bA2,*bB2,*bC2,*br2;
int prevC_size, prevC_alloced;
int padded_size;
double *zp = NULL;
double *op = NULL;
int Br_DIV=10;
int OPT_LS=16;
int OPTIMIZED_RIGHTPART=1;
char *fr1_opencl_text = "\n\
#pragma OPENCL EXTENSION %s : enable \n\
#define Br_DIV %d\n\
#define OPT_LS %d\n\
#define delta (*((__global double *)(class+%d)))\n\
#define s1 (*((__global double *)(class+%d)))\n\
#define h1 (*((__global double *)(class+%d)))\n\
#define delta2 (*((__global double *)(class+%d)))\n\
#define delta3 (*((__global double *)(class+%d)))\n\
#define tg (*((__global double *)(class+%d)))\n\
#define g3 (*((__global double *)(class+%d)))\n\
#define tau (*((__global double *)(class+%d)))\n\
#define w (*((__global double *)(class+%d)))\n\
#define alpha (*((__global double *)(class+%d)))\n\
#define beta (*((__global double *)(class+%d)))\n\
#define gamma (*((__global double *)(class+%d)))\n\
#define d (*((__global double *)(class+%d)))\n\
#define H0 (*((__global double *)(class+%d)))\n\
#define sigma (*((__global double *)(class+%d)))\n\
#define Cm (*((__global double *)(class+%d)))\n\
#define kmm (*((__global double *)(class+%d)))\n\
__kernel void UA(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {A[i]=0.0; return;}\n\
	if (i==N) {A[i]=0.0; return;}\n\
	A[i]=-1.0;\n\
	// save C for time-fractional \n\
	if (prevC_size!=-1)\n\
	prevC[prevC_size*(N+2)+i]=C[i];\n\
}\n\
__kernel void UB(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) { B[i]=0.0; return;}\n\
	if (i==N) { B[i]=0.0;return;}\n\
	B[i]=-1.0;\n\
}\n\
__kernel void UC(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {mC[i]=1.0;return;}\n\
	if (i==N) {mC[i]=1.0;return;}\n\
	mC[i]=s1;\n\
}\n\
__kernel void URightPart(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int b=get_global_id(0);\n\
	int n=get_global_size(0);\n\
	int j,k;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j&1) \n\
			i+=n-1-b;\n\
		else \n\
			i+=b;\n\
		if (i>N) continue;\n\
		if (i == 0)\n\
			r[i] = 1.0;\n\
		else\n\
		{\n\
			if (i == N)\n\
				r[i] = H0;\n\
			else\n\
			{\n\
				double sum = 0.0;\n\
				double A = 1.0 - alpha;\n\
				if (A!=1.0)\n\
				for (k = 1;k <= i - 1;k++)\n\
					sum += (pow((double)i - (double)k + 1, A) - pow((double)i - (double)k, A))*(U[k + 1] - 2.0*U[k] + U[k - 1]);\n\
				r[i] = (h1*h1 / delta)*((U[i] / tau)+ delta*sum / (h1*h1));\n\
			}\n\
		}\n\
	}\n\
}\n\
__kernel void URightPart_optimized(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int b=get_global_id(0);\n\
	int lid=get_local_id(0);\n\
	int n=get_global_size(0);\n\
	__local double Us[OPT_LS+2];\n\
	int ids[Br_DIV]; // element indices \n\
	double sums[Br_DIV]; \n\
	double Al = 1.0 - alpha;\n\
	int j,k,bl;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j&1) \n\
			i+=n-1-b;\n\
		else \n\
			i+=b;\n\
		ids[j]=i;\n\
		if (i!=0)\n\
		{ \n\
			if (i<N)\n\
				sums[j]=0.0;\n\
			else\n\
				if (i==N)\n\
					r[i]=H0;\n\
		}\n\
		else \n\
			r[i]=1.0;\n\
	} \n\
	if (Al!=1.0)\n\
	for (bl=0;bl<((N/OPT_LS)+1);bl++) \n\
	{ \n\
		int kk=1+bl*OPT_LS+lid;\n\
		if (kk<N)\n\
		{\n\
			Us[lid+1]=U[kk];\n\
			if (lid==0)\n\
				Us[0]=U[kk-1];\n\
			if (lid==(OPT_LS-1))\n\
				Us[OPT_LS+1]=U[kk+1];\n\
			if (kk==(N-1))\n\
				Us[lid+2]=U[kk+1];\n\
		}\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		for (k=1+bl*OPT_LS;k<1+(bl+1)*OPT_LS;k++) \n\
		{	\n\
			int kl=k-bl*OPT_LS; \n\
			if (k>N) break; \n\
			for (j = 0;j < Br_DIV;j++)\n\
				if ((ids[j]>0)&&(ids[j]<N))\n\
					if (k<ids[j])\n\
						sums[j] += (pow((double)ids[j] - (double)k + 1, Al) - pow((double)ids[j] - (double)k, Al))*(Us[kl+1] - 2.0*Us[kl] + Us[kl-1]);\n\
		}	\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
	for (j = 0;j < Br_DIV;j++)\n\
		if ((ids[j]>0)&&(ids[j]<N))\n\
		r[ids[j]] = (h1*h1 / delta)*((U[ids[j]] / tau)+ delta*sums[j] / (h1*h1));\n\
}\n\
__kernel void CA(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {A[i]=0.0;return;}\n\
	if (i==N) {A[i]=0.0;return;}\n\
	A[i]=-((delta2*d / (h1*h1)) + (V[i] / (2.0*h1)));\n\
}\n\
__kernel void CB(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {B[i]=0.0;return;}\n\
	if (i==N) {B[i]=0.0;return;}\n\
	B[i]=-((delta2*d / (h1*h1)) - (V[i] / (2.0*h1)));\n\
}\n\
__kernel void CC(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {mC[i]=1.0;return;}\n\
	if (i==N) {mC[i]=1.0;return;}\n\
	mC[i]=((sigma *delta3) + (2.0*delta2*d / (h1*h1)) + ((V[i + 1] - V[i - 1]) / (2.0*h1))+kmm);\n\
}\n\
__kernel void CRightPart(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size)\n\
{\n\
	int b = get_global_id(0);\n\
	int n = get_global_size(0);\n\
	int j, k;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j & 1) \n\
			i += n - 1 - b;\n\
		else \n\
			i += b;\n\
		if (i>N) continue;\n\
		if (i == 0)\n\
			r[i] = 0.0;\n\
		else\n\
		{\n\
			if (i == N)\n\
				r[i] = 1.0;\n\
			else\n\
			{\n\
				double sum = 0.0;\n\
				double A = 1.0 - beta;\n\
				if (A != 1.0)\n\
				for (k = 1;k <= i - 1;k++)\n\
					sum += (pow((double)i - (double)k + 1, A) - pow((double)i - (double)k, A))*(C[k + 1] - 2.0*C[k] + C[k - 1]);\n\
				r[i] = (sigma *delta3)*C[i] + d*delta2*sum / (h1*h1);\n\
				// time-fractional derivative \n\
				double time_sum = 0;\n\
				if (prevC_size >= 2)\n\
				for (int t = 0;t < prevC_size - 1;t++)\n\
					time_sum += (pow((double)(prevC_size - t), 1.0 - gamma) - pow((double)(prevC_size - t - 1), 1.0 - gamma))*(prevC[(t + 1)*(N+2)+i] - prevC[t*(N+2)+i]);\n\
				r[i] -= tg*(sigma / g3)*time_sum / tau;\n\
				r[i] += Cm*kmm;\n\
			}\n\
		}\n\
	}\n\
}\n\
__kernel void CRightPart_optimized(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size)\n\
{\n\
	int b=get_global_id(0);\n\
	int lid=get_local_id(0);\n\
	int n=get_global_size(0);\n\
	__local double Us[OPT_LS+2];\n\
	int ids[Br_DIV]; // element indices \n\
	double sums[Br_DIV]; \n\
	double Al = 1.0 - beta;\n\
	int j,k,bl;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j&1) \n\
			i+=n-1-b;\n\
		else \n\
			i+=b;\n\
		ids[j]=i;\n\
		if (i!=0)\n\
		{ \n\
			if (i<N)\n\
				sums[j]=0.0;\n\
			else\n\
				if (i==N)\n\
					r[i]=1.0;\n\
		}\n\
		else \n\
			r[i]=0.0;\n\
	} \n\
	if (Al!=1.0)\n\
	for (bl=0;bl<((N/OPT_LS)+1);bl++) \n\
	{ \n\
		int kk=1+bl*OPT_LS+lid;\n\
		if (kk<N)\n\
		{\n\
			Us[lid+1]=C[kk];\n\
			if (lid==0)\n\
				Us[0]=C[kk-1];\n\
			if (lid==(OPT_LS-1))\n\
				Us[OPT_LS+1]=C[kk+1];\n\
			if (kk==(N-1))\n\
				Us[lid+2]=C[kk+1];\n\
		}\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		for (k=1+bl*OPT_LS;k<1+(bl+1)*OPT_LS;k++) \n\
		{	\n\
			int kl=k-bl*OPT_LS; \n\
			if (k>N) break; \n\
			for (j = 0;j < Br_DIV;j++)\n\
				if ((ids[j]>0)&&(ids[j]<N))\n\
					if (k<ids[j])\n\
						sums[j] += (pow((double)ids[j] - (double)k + 1, Al) - pow((double)ids[j] - (double)k, Al))*(Us[kl+1] - 2.0*Us[kl] + Us[kl-1]);\n\
		}	\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
	for (j = 0;j < Br_DIV;j++)\n\
		if ((ids[j]>0)&&(ids[j]<N))\n\
		{\n\
			r[ids[j]] = (sigma *delta3)*C[ids[j]] + d*delta2*sums[j] / (h1*h1);\n\
			// time-fractional derivative \n\
			double time_sum = 0;\n\
			if (prevC_size >= 2)\n\
			for (int t = 0;t < prevC_size - 1;t++)\n\
				time_sum += (pow((double)(prevC_size - t), 1.0 - gamma) - pow((double)(prevC_size - t - 1), 1.0 - gamma))*(prevC[(t + 1)*(N+2)+ids[j]] - prevC[t*(N+2)+ids[j]]);\n\
			r[ids[j]] -= tg*(sigma / g3)*time_sum / tau;\n\
			r[ids[j]] += Cm*kmm;\n\
		}\n\
}\n\
__kernel void V(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,double G)\n\
{\n\
	int b=get_global_id(0);\n\
	int n=get_global_size(0);\n\
	int j,k;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j&1) \n\
			i+=n-1-b;\n\
		else \n\
			i+=b;\n\
		if (i>N) continue;\n\
		if (i == 0)\n\
			V[i] = 0;\n\
		else\n\
		{\n\
			if (i == 1)\n\
				V[i] = -(w / (h1*G))*(U[1] - U[0]);\n\
			else\n\
			{\n\
				double sum = 0.0;\n\
				double A = 1.0 - alpha;\n\
				if (A!=1.0)\n\
				for (k = 1;k <= i - 1;k++)\n\
					sum += (pow((double)i - (double)k + 1, A) - pow((double)i - (double)k, A))*(U[k] - U[k - 1]);\n\
				V[i] = -(w / (h1*G))*(U[i]-U[i-1] + sum);\n\
			}\n\
		}\n\
	}\n\
}\n\
__kernel void V_optimized(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,double G)\n\
{\n\
	int b=get_global_id(0);\n\
	int lid=get_local_id(0);\n\
	int n=get_global_size(0);\n\
	__local double Us[OPT_LS+1];\n\
	int ids[Br_DIV]; // element indices \n\
	double sums[Br_DIV]; \n\
	double Al = 1.0 - alpha;\n\
	int j,k,bl;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j&1) \n\
			i+=n-1-b;\n\
		else \n\
			i+=b;\n\
		ids[j]=i;\n\
		if (i!=0)\n\
		{ \n\
			if (i!=1)\n\
				sums[j]=0.0;\n\
			else\n\
				V[i] = -(w / (h1*G))*(U[1] - U[0]);\n\
		}\n\
		else \n\
			V[i]=0.0;\n\
	} \n\
	if (Al!=1.0)\n\
	for (bl=0;bl<((N/OPT_LS)+1);bl++) \n\
	{ \n\
		int kk=1+bl*OPT_LS+lid;\n\
		if (kk<N)\n\
		{\n\
			Us[lid+1]=U[kk];\n\
			if (lid==0)\n\
				Us[0]=U[kk-1];\n\
		}\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		for (k=1+bl*OPT_LS;k<1+(bl+1)*OPT_LS;k++) \n\
		{	\n\
			int kl=k-bl*OPT_LS; \n\
			if (k>N) break; \n\
			for (j = 0;j < Br_DIV;j++)\n\
				if ((ids[j]>1)&&(ids[j]<=N))\n\
					if (k<ids[j])\n\
					sums[j] += (pow((double)ids[j] - (double)k + 1, Al) - pow((double)ids[j] - (double)k, Al))*(Us[kl] - Us[kl-1]);\n\
		}	\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
	for (j = 0;j < Br_DIV;j++)\n\
		if ((ids[j]>1)&&(ids[j]<=N))\n\
			V[ids[j]] = -(w / (h1*G))*(U[ids[j]]-U[ids[j]-1] + sums[j]);\n\
}\n";
void fr_ocl_init(space_fract_solver *s)
{
	prg=new OpenCL_program(0);
	queue=prg->create_queue(device,0);
	{
		char *text=new char[strlen(fr1_opencl_text)*2];
		sprintf(text, fr1_opencl_text, ((double_ext == 0) ? "cl_amd_fp64" : "cl_khr_fp64"), Br_DIV, OPT_LS
										, ((char *)&s->delta) - (char *)s
										, ((char *)&s->s1) - (char *)s
										, ((char *)&s->h1) - (char *)s
										, ((char *)&s->delta2) - (char *)s
										, ((char *)&s->delta3) - (char *)s
										, ((char *)&s->tg) - (char *)s
										, ((char *)&s->g3) - (char *)s
										, ((char *)&s->tau) - (char *)s
										, ((char *)&s->w) - (char *)s
										, ((char *)&s->alpha) - (char *)s
										, ((char *)&s->beta) - (char *)s
										, ((char *)&s->gamma) - (char *)s
										, ((char *)&s->d) - (char *)s
										, ((char *)&s->H0) - (char *)s
										, ((char *)&s->sigma) - (char *)s
										, ((char *)&s->Cm) - (char *)s
										, ((char *)&s->kmm) - (char *)s);
			prog=prg->create_program(text);
		delete [] text;
	}
	pcr_small_systems_init(prog->hContext[0],0);
    kUA=prg->create_kernel(prog,"UA"); 
	kUB=prg->create_kernel(prog,"UB");
	kUC=prg->create_kernel(prog,"UC");
    kCA=prg->create_kernel(prog,"CA"); 
	kCB=prg->create_kernel(prog,"CB");
	kCC=prg->create_kernel(prog,"CC");
	if (OPTIMIZED_RIGHTPART)
	{
		kUr=prg->create_kernel(prog,"URightPart_optimized");
		kCr=prg->create_kernel(prog,"CRightPart_optimized");
		kV=prg->create_kernel(prog,"V_optimized");
	}
	else
	{
		kUr=prg->create_kernel(prog,"URightPart");
		kCr=prg->create_kernel(prog,"CRightPart");
		kV=prg->create_kernel(prog,"V");
	}
	bS=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(space_fract_solver),(void *)s);
	padded_size = pow(2.0,log2(N + 2)+1);
	zp = new double[padded_size];
	op = new double[padded_size];
	for (int i = 0;i < padded_size;i++)
	{
		zp[i] = 0.0;
		op[i] = 1.0;
	}
	bU=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),(void *)s->U);
	bCn=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),(void *)s->C);
    bV=prg->create_buffer(CL_MEM_READ_WRITE, padded_size*sizeof(double),NULL);
    bA=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bB=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bC=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),op);
    br=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bA2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bB2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bC2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),op);
    br2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
	prevC_size = -1;
	prevC_alloced = 10;
	bprevC = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
}
void fr_ocl_get_results(space_fract_solver *s)
{
	queue->EnqueueBuffer(bU, s->U, 0, (N + 2)*sizeof(double));
	queue->EnqueueBuffer(bV, s->V, 0, (N + 2)*sizeof(double));
	queue->EnqueueBuffer(bCn, s->C, 0, (N + 2)*sizeof(double));
	queue->Finish();
}
void set_ABCr_args(OpenCL_kernel *k)
{
	int err;
	int i = N;
	err=k->SetBufferArg(bA,0);
	err|=k->SetBufferArg(bB,1);
	err|=k->SetBufferArg(bC,2);
	err|=k->SetBufferArg(br,3);
	err|=k->SetBufferArg(bU,4);
	err|=k->SetBufferArg(bV,5);
	err|=k->SetBufferArg(bCn,6);
	err|=k->SetBufferArg(bS,7);
	err |= k->SetArg(8, sizeof(int), &i);
	if (err) SERROR("Error: Failed to set kernels args");
}
void fr_ocl_calc_tri(OpenCL_kernel *kA,OpenCL_kernel *kB,OpenCL_kernel *kC,OpenCL_kernel *kr,OpenCL_buffer *bRes,FILE *fi)
{
	size_t nth,lsize=1;
	long long t1,t2,t3,t4;
	// matrix diagonals A,B,C are calculated one thread per element
	nth=N+1;
	set_ABCr_args(kA);
	if (kA == kUA) // do save C
	{
		kA->SetArg(10,sizeof(int),&prevC_size);
		prevC_size++;
		if (prevC_size == prevC_alloced) // do realloc
		{
			prevC_alloced *= 2;
			double *b = new double[prevC_alloced*(N+2)];
			queue->EnqueueBuffer(bprevC, b);
			delete bprevC;
			bprevC = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
			queue->EnqueueWriteBuffer(bprevC, b, 0, prevC_alloced*(N + 2)*sizeof(double));
		}
		kA->SetBufferArg(bprevC, 9);
	}
	t1=GetTickCount();
	queue->ExecuteKernel(kA,1,&nth,&lsize);
	set_ABCr_args(kB);
	queue->ExecuteKernel(kB,1,&nth,&lsize);
	set_ABCr_args(kC);
	queue->ExecuteKernel(kC,1,&nth,&lsize);
	queue->Finish();
	t2=GetTickCount();
	// right part vector is calculated Br_DIV elements per thread
	nth=((N+2)/Br_DIV)+1;
	lsize=OPT_LS;
	nth=((nth/lsize)+1)*lsize;
	set_ABCr_args(kr);
	if (kr == kCr) // do save C
	{
		kr->SetBufferArg(bprevC, 9);
		kr->SetArg(10, sizeof(int), &prevC_size);
	}
	queue->ExecuteKernel(kr,1,&nth,&lsize);
	queue->Finish();
	t3=GetTickCount();
	// solve tridiagonal by NVIDIA version of parallel cyclic reducion 
	pcr_solver(bA->buffer,bC->buffer,bB->buffer,br->buffer,bRes->buffer, padded_size,&queue->hCmdQueue,1,bA2->buffer,bC2->buffer,bB2->buffer,br2->buffer);
	queue->Finish();
	t4=GetTickCount();
	fprintf(fi,"fillM %lld fill rp %lld solve %lld\n",t2-t1,t3-t2,t4-t3);
}
void fr_ocl_calc_V(space_fract_solver *s,FILE *fi)
{
	size_t nth,lsize;
	long long t1,t2;
	double G = Gamma(2.0 - s->alpha);
	// V vector is calculated Br_DIV elements per thread
	t1=GetTickCount();
	nth=((N+2)/Br_DIV)+1;
	lsize=OPT_LS;
	nth=((nth/lsize)+1)*lsize;
	set_ABCr_args(kV);
	kV->SetArg(9, sizeof(double), &G);
	queue->ExecuteKernel(kV,1,&nth,&lsize);
	queue->Finish();
	t2=GetTickCount();
	fprintf(fi,"calc V %lld\n",t2-t1);
}
void solve_ocl(double t,double save_tau)
{
	space_fract_solver s1, s2,s3;
	int nsave = 1;
	int step = 0;
	FILE *fi1, *fi2;
	fi1 = fopen("log.txt", "wt");
	fi2 = fopen("results.txt", "wt");
	s1.gamma = s2.gamma = s3.gamma=0.7;
	s1.init();
	s2.init();
	s3.init();
	fr_ocl_init(&s1);
	for (double tt = 0;tt < t;tt += s1.tau)
	{
		double err1=0.0,err2=0.0;
		long long i1, i2, i3,i11;
		// GPU
		i1 = GetTickCount();
		fr_ocl_calc_tri(kUA,kUB,kUC,kUr,bU,fi1);
		fr_ocl_calc_V(&s1,fi1);
		fr_ocl_calc_tri(kCA,kCB,kCC,kCr,bCn,fi1);
		i1 = GetTickCount()-i1;
		i11= GetTickCount();
		fr_ocl_get_results(&s1);
		i11 = GetTickCount()-i11;
		// CPU-simple
		i2 = GetTickCount();
		s2.U1(0);
		s2.V1(0);
		s2.C1(0);
		i2 = GetTickCount() - i2;
		// CPU-approx
		i3 = GetTickCount();
		s3.U1(1);
		s3.V1(1);
		s3.C1(1);
		i3 = GetTickCount() - i3;
		// calculate errors
		for (int i = 0;i < N + 1;i++)
		{
			err1 += (s2.U[i] - s1.U[i])*(s2.U[i] - s1.U[i]);
			err1 += (s2.V[i] - s1.V[i])*(s2.V[i] - s1.V[i]);
			err1 += (s2.C[i] - s1.C[i])*(s2.C[i] - s1.C[i]);
			err2 += (s2.U[i] - s3.U[i])*(s2.U[i] - s3.U[i]);
			err2 += (s2.V[i] - s3.V[i])*(s2.V[i] - s3.V[i]);
			err2 += (s2.C[i] - s3.C[i])*(s2.C[i] - s3.C[i]);
		}
		fprintf(fi1, "t %g ocl %lld(%lld) cpu-simple %lld cpu-approx %lld e(gpu-cpu) %g e(appr-cpu) %g\n", tt, i1, i11, i2, i3, err1, err2);
		printf("t %g ocl %lld(%lld) cpu-simple %lld cpu-approx %lld e(gpu-cpu) %g e(appr-cpu) %g\n", tt, i1, i11, i2, i3, err1, err2);
		fflush(stdout);
		// save result
		if (tt > nsave*save_tau)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g f %g %g %g a %g %g %g c %g %g %g\n", tt,(double)i/N, s1.U[i], s1.V[i], s1.C[i], s2.U[i], s2.V[i], s2.C[i],s3.U[i],s3.V[i],s3.C[i]);
			nsave++;
		}
	}
}

////////////////////////////////////////////////////
// fractional consolidation solver 1d /////////////
///////////////////////////////////////////////////
class cons_fract_solver {
public:
	double *U; // pressure
	double *V; // velocity
	double *C; // concentration
	double *Al; // alpha coefficients
	double *Bt; // beta coefficients
	double *Om; // right part
	std::vector<double*> oldC; // for time-fractional derivatives in C-equation
	std::vector<double*> oldH; // for time-fractional derivatives in H-equation
	double tau,h,d,u,v;
	double alpha,ba,ca;
	double sigma;
	double Cv,mu;
	double uA,uS;
	int bottom_cond; // 0 - H,C=0, 1 - d{H,C}/dn=0
	int model; 
	cons_fract_solver()
	{
		U = V = C = Al = Bt = Om = NULL;
	}
	void init(double _tau=0.0)
	{
		clear();
		U=new double[N+2];
		V=new double[N+2];
		C=new double[N+2];
		Al=new double[N+2];
		Bt=new double[N+2];
		Om=new double[N+2];

		h = 1.0 / N;
		if (_tau == 0)
			tau = 0.1*(1.0 / N);
		else
			tau = _tau;
		sigma = 0.3;
		d=0.032;
		u=0.016;
		v=0.011;
		Cv=0.052;
		mu=0.0362;
		bottom_cond=1;
		model = 2;

		ba=alpha/(1.0-alpha);
		ca=(1.0/alpha)*(1.0-exp(-ba*tau));
		uA=Cv/(ca*h*h);
		if (model==1)
			uS = ((1.0 / tau) + 2 * uA);
		else
			uS = (((1.0 + 0.6*alpha) / tau) + 2 * uA);

		for (int i = 0;i < N + 1;i++)
		{
			U[i] = 1.0;
			C[i] = 0.0;
		}
		C[0] = 1.0;
		U[0] = 0.0;
		U[N]=0.0;
	}
	// U equation alpha coeffients
	void al1()
	{
		Al[1] = 0;
		for (int i = 1;i < N;i++)
			Al[i + 1] = uA / (uS - uA*Al[i]);
	}
	// U equation right part
	void Om1(int approx)
	{
		for (int i = 1;i < N;i++)
		{
			if (model==1)
				Om[i] = (mu/(ca*h*h))*(C[i+1]-2.0*C[i]+C[i-1])-U[i]/tau;
			else
				Om[i] = (mu / (ca*h*h))*(C[i + 1] - 2.0*C[i] + C[i - 1]) - (1.0 + 0.6*alpha)*U[i] / tau;
			// time-fractional derivative
			double time_sum = 0;
			if (oldH.size()>=2)
			for (int t = 0;t < oldH.size() - 1;t++)
				time_sum += exp(-ba*tau*(double)(oldH.size() - t))*(oldH[t + 1][i] - oldH[t][i]);
			if (model==1)
				Om[i] += time_sum/tau;
			else
				Om[i] += 0.6*alpha*time_sum / tau;
		}
	}
	// U equation beta coeffients
	void bt1()
	{
		Bt[1] = 0;
		for (int i = 1;i < N;i++)
			Bt[i + 1] = Al[i + 1] * (Bt[i] - (Om[i]/uA));
	}
	// calc U
	void U1(int approx)
	{
		al1();
		Om1(approx);
		bt1();
		if (bottom_cond==0)
			U[N] = 0.0;
		else
			U[N]=Bt[N]/(1.0-Al[N]);
		U[0] = 0.0;
		for (int i = N - 1;i >= 1;i--)
			U[i] = Al[i + 1] * U[i + 1] + Bt[i + 1];
		// save H for time-fractional derivative calculations
		double *cs = new double[N + 2];
		memcpy(cs, U, (N + 2)*sizeof(double));
		oldH.push_back(cs);
	}
	// calc V
	void V1(int approx)
	{
		V[0] = 0.0;
		V[N] = 0.0;
		for (int i = 1;i < N;i++)
			V[i]=(1.0/(2.0*h))*(u*(U[i+1]-U[i-1])-v*(C[i+1]-C[i-1]));
	}
	// C equation alpha coefficient
	void al2()
	{
		Al[1] = 0;
		for (int i = 1;i < N;i++)
		{
			double R=1.0+(h/(2.0*d))*fabs(V[i]);
			double Xi=d/R;
			double Vp=0.5*(V[i]+fabs(V[i]));
			double Vm=0.5*(V[i]-fabs(V[i]));
			double B=(1.0/h)*((Xi/h)+Vp);
			double A=(1.0/h)*((Xi/h)-Vm);
			double S=(sigma*ca/tau)+A+B;
			Al[i + 1] = B/(S-A*Al[i]);
		}
	}
	// C equation right part
	void Om2(int approx)
	{
		for (int i = 1;i < N;i++)
		{
			Om[i] = -C[i]/tau;
			// time-fractional derivative
			double time_sum = 0;
			if (oldC.size()>=2)
			for (int t = 0;t < oldC.size() - 1;t++)
				time_sum += exp(-ba*tau*(double)(oldC.size() - t))*(oldC[t + 1][i] - oldC[t][i]);
			Om[i] += time_sum/tau;
			Om[i] *= sigma*ca;
		}
	}
	void bt2()
	{
		Bt[1] = 1;
		for (int i = 1;i < N;i++)
		{
			double R=1.0+(h/(2.0*d))*fabs(V[i]);
			double Xi=d/R;
			double Vp=0.5*(V[i]+fabs(V[i]));
			double Vm=0.5*(V[i]-fabs(V[i]));
			double B=(1.0/h)*((Xi/h)+Vp);
			double A=(1.0/h)*((Xi/h)-Vm);
			Bt[i + 1] = (Al[i + 1] / B)*(A*Bt[i] - Om[i]);
		}
	}
	// calc C
	void C1(int approx)
	{
		al2();
		Om2(approx);
		bt2();
		if (bottom_cond==0)
			C[N] = 0.0;
		else
			C[N]=Bt[N]/(1.0-Al[N]);
		C[0] = 1.0;
		for (int i = N - 1;i >= 1;i--)
			C[i] = Al[i + 1] * C[i + 1] + Bt[i + 1];
		// save C for time-fractional derivative calculations
		double *cs = new double[N + 2];
		memcpy(cs, C, (N + 2)*sizeof(double));
		oldC.push_back(cs);
	}
	void clear()
	{
		if (U) delete[] U;
		if (V) delete[] V;
		if (C) delete[] C;
		if (Al) delete[] Al;
		if (Bt) delete[] Bt;
		if (Om) delete[] Om;
		for (int i = 0;i < oldC.size();i++)
			delete oldC[i];
		for (int i = 0;i < oldH.size();i++)
			delete oldH[i];
		oldC.clear();
		oldH.clear();
	}
	~cons_fract_solver()
	{
		clear();
	}
};
void cons_solve(double t,double save_tau)
{
	cons_fract_solver s1,s2,s3,s4,s5;
	int nsave = 1;
	FILE *fi2;
	fi2 = fopen("results.txt", "wt");
	s1.alpha=0.8;
	s1.init();
	s2.alpha=0.8;
	s2.init();
	s3.alpha = 0.9999999;
	s3.init();
	s4.alpha = 0.7;
	s4.init();
	s5.alpha = 0.6;
	s5.init();
	// set tau
	do
	{
		double e;
		s2.tau = s1.tau / 2;
		s1.init(s1.tau);
		s2.init(s2.tau);
		for (double tt = 0;tt < 0.01;tt += s1.tau)
		{
			s1.V1(1);
			s1.C1(1);
			s1.U1(1);
		}
		for (double tt = 0;tt < 0.01;tt += s2.tau)
		{
			s2.V1(1);
			s2.C1(1);
			s2.U1(1);
		}
		e = 0;
		for (int i = 1;i < N;i++)
		{
			e += (s1.U[i] - s2.U[i])*(s1.U[i] - s2.U[i]);
			e += (s1.V[i] - s2.V[i])*(s1.V[i] - s2.V[i]);
			e += (s1.C[i] - s2.C[i])*(s1.C[i] - s2.C[i]);
		}
		if (e > 1e-5)
			s1.tau /= 2;
		else
			break;
		printf("%g %g\n",e,s1.tau);
		if (s1.tau<0.00001)
		    break;
	} while (1);
	s2.tau = s3.tau = s1.tau;
	s2.alpha = 0.9;
	s1.init(s1.tau);
	s2.init(s2.tau);
	s3.init(s3.tau);
	s4.init(s3.tau);
	s5.init(s3.tau);
	for (double tt = 0;tt < t;tt += s1.tau)
	{
		long long i1;
		i1 = GetTickCount();
		s1.V1(1);
		s1.C1(1);
		s1.U1(1);
		s2.V1(1);
		s2.C1(1);
		s2.U1(1);
		s3.V1(1);
		s3.C1(1);
		s3.U1(1);
		s4.V1(1);
		s4.C1(1);
		s4.U1(1);
		s5.V1(1);
		s5.C1(1);
		s5.U1(1);
		i1 = GetTickCount()-i1;
		printf("t %g time %lld\n", tt, i1);
		fflush(stdout);
		// save result
		if (tt > nsave*save_tau)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", tt,(double)i/N, s1.U[i], s1.V[i], s1.C[i], s2.U[i], s2.V[i], s2.C[i], s3.U[i], s3.V[i], s3.C[i], s4.U[i], s4.V[i], s4.C[i], s5.U[i], s5.V[i], s5.C[i]);
			nsave++;
		}
	}
}
// consolidation OCL
OpenCL_buffer *bprevH;
char *fr_cons_opencl_text = "\n\
#pragma OPENCL EXTENSION %s : enable \n\
#define Br_DIV %d\n\
#define OPT_LS %d\n\
#define tau (*((__global double *)(class+%d)))\n\
#define h (*((__global double *)(class+%d)))\n\
#define d (*((__global double *)(class+%d)))\n\
#define u (*((__global double *)(class+%d)))\n\
#define v (*((__global double *)(class+%d)))\n\
#define alpha (*((__global double *)(class+%d)))\n\
#define ba (*((__global double *)(class+%d)))\n\
#define ca (*((__global double *)(class+%d)))\n\
#define sigma (*((__global double *)(class+%d)))\n\
#define Cv (*((__global double *)(class+%d)))\n\
#define mu (*((__global double *)(class+%d)))\n\
#define uA (*((__global double *)(class+%d)))\n\
#define uS (*((__global double *)(class+%d)))\n\
__kernel void UA(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevH,int prevC_size)\n\
{\n\
	int i=get_global_id(0);\n\
	// save H \n\
	if (prevC_size>=0)\n\
		prevH[prevC_size*(N+2)+i]=U[i];\n\
	if (i==0) {A[i]=0.0; return;}\n\
	if (i==N) {A[i]=0.0; return;}\n\
	A[i]=-uA;\n\
}\n\
__kernel void UB(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) { B[i]=0.0; return;}\n\
	if (i==N) { B[i]=0.0;return;}\n\
	B[i]=-uA;\n\
}\n\
__kernel void UC(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {mC[i]=1.0;return;}\n\
	if (i==N) {mC[i]=1.0;return;}\n\
	mC[i]=uS;\n\
}\n\
__kernel void URightPart(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevH,int prevC_size)\n\
{\n\
	int b=get_global_id(0);\n\
	int n=get_global_size(0);\n\
	int j,k;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j&1) \n\
			i+=n-1-b;\n\
		else \n\
			i+=b;\n\
		if (i>N) continue;\n\
		if (i == 0)\n\
			r[i] = 0.0;\n\
		else\n\
		{\n\
			if (i == N)\n\
				r[i] = 0.0;\n\
			else\n\
			{\n\
				r[i] = -((mu/(ca*h*h))*(C[i+1]-2.0*C[i]+C[i-1])-(1.0+0.6*alpha)*U[i]/tau);\n\
				double time_sum = 0;\n\
				if (prevC_size >= 1)\n\
				for (int t = 0;t < prevC_size ;t++)\n\
					time_sum += exp(-ba*tau*(double)(prevC_size - t))*(prevH[(t + 1)*(N+2)+i] - prevH[t*(N+2)+i]);\n\
				r[i] -= 0.6*alpha*time_sum/tau;\n\
			}\n\
		}\n\
	}\n\
}\n\
__kernel void CA(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size)\n\
{\n\
	int i=get_global_id(0);\n\
	// save C \n\
	if (prevC_size>=0)\n\
		prevC[prevC_size*(N+2)+i]=C[i];\n\
	if (i==0) {A[i]=0.0;return;}\n\
	if (i==N) {A[i]=0.0;return;}\n\
	double R=1.0+(h/(2.0*d))*fabs(V[i]);\n\
	double Xi=d/R;\n\
	double Vp=0.5*(V[i]+fabs(V[i]));\n\
	double Vm=0.5*(V[i]-fabs(V[i]));\n\
	A[i]=-((1.0/h)*((Xi/h)-Vm));\n\
}\n\
__kernel void CB(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {B[i]=0.0;return;}\n\
	if (i==N) {B[i]=0.0;return;}\n\
	double R=1.0+(h/(2.0*d))*fabs(V[i]);\n\
	double Xi=d/R;\n\
	double Vp=0.5*(V[i]+fabs(V[i]));\n\
	double Vm=0.5*(V[i]-fabs(V[i]));\n\
	B[i]=-((1.0/h)*((Xi/h)+Vp));\n\
}\n\
__kernel void CC(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {mC[i]=1.0;return;}\n\
	if (i==N) {mC[i]=1.0;return;}\n\
	double R=1.0+(h/(2.0*d))*fabs(V[i]);\n\
	double Xi=d/R;\n\
	double Vp=0.5*(V[i]+fabs(V[i]));\n\
	double Vm=0.5*(V[i]-fabs(V[i]));\n\
	double cB=(1.0/h)*((Xi/h)+Vp);\n\
	double cA=(1.0/h)*((Xi/h)-Vm);\n\
	mC[i]=(sigma*ca/tau)+cA+cB;\n\
}\n\
__kernel void CRightPart(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size)\n\
{\n\
	int b = get_global_id(0);\n\
	int n = get_global_size(0);\n\
	int j, k;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j & 1) \n\
			i += n - 1 - b;\n\
		else \n\
			i += b;\n\
		if (i>N) continue;\n\
		if (i == 0)\n\
			r[i] = 1.0;\n\
		else\n\
		{\n\
			if (i == N)\n\
				r[i] = 0.0;\n\
			else\n\
			{\n\
				r[i] = C[i]/tau;\n\
				double time_sum = 0;\n\
				if (prevC_size>=1)\n\
				for (int t = 0;t < prevC_size ;t++)\n\
					time_sum += exp(-ba*tau*(double)(prevC_size - t))*(prevC[(t + 1)*(N+2)+i] - prevC[t*(N+2)+i]);\n\
				r[i] -= time_sum/tau;\n\
				r[i] *= sigma*ca;\n\
			}\n\
		}\n\
	}\n\
}\n\
__kernel void V(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int b=get_global_id(0);\n\
	int n=get_global_size(0);\n\
	int j,k;\n\
	for (j = 0;j < Br_DIV;j++)\n\
	{\n\
		int i = j*n;\n\
		if (j&1) \n\
			i+=n-1-b;\n\
		else \n\
			i+=b;\n\
		if (i>N) continue;\n\
		if (i == 0)\n\
			V[i] = 0;\n\
		else\n\
		{\n\
			if (i==N)\n\
				V[i]=0.0;\n\
			else\n\
				V[i]=(1.0/(2.0*h))*(u*(U[i+1]-U[i-1])-v*(C[i+1]-C[i-1]));\n\
		}\n\
	}\n\
}\n";
void fr_cons_ocl_init(cons_fract_solver *s)
{
	prg=new OpenCL_program(0);
	queue=prg->create_queue(device,0);
	{
		char *text=new char[strlen(fr_cons_opencl_text)*2];
		sprintf(text, fr_cons_opencl_text, ((double_ext == 0) ? "cl_amd_fp64" : "cl_khr_fp64"), Br_DIV, OPT_LS
										, ((char *)&s->tau) - (char *)s
										, ((char *)&s->h) - (char *)s
										, ((char *)&s->d) - (char *)s
										, ((char *)&s->u) - (char *)s
										, ((char *)&s->v) - (char *)s
										, ((char *)&s->alpha) - (char *)s
										, ((char *)&s->ba) - (char *)s
										, ((char *)&s->ca) - (char *)s
										, ((char *)&s->sigma) - (char *)s
										, ((char *)&s->Cv) - (char *)s
										, ((char *)&s->mu) - (char *)s
										, ((char *)&s->uA) - (char *)s
										, ((char *)&s->uS) - (char *)s);
			prog=prg->create_program(text);
		delete [] text;
	}
	pcr_small_systems_init(prog->hContext[0],0);
    kUA=prg->create_kernel(prog,"UA"); 
	kUB=prg->create_kernel(prog,"UB");
	kUC=prg->create_kernel(prog,"UC");
    kCA=prg->create_kernel(prog,"CA"); 
	kCB=prg->create_kernel(prog,"CB");
	kCC=prg->create_kernel(prog,"CC");
	kUr=prg->create_kernel(prog,"URightPart");
	kCr=prg->create_kernel(prog,"CRightPart");
	kV=prg->create_kernel(prog,"V");
	bS=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(space_fract_solver),(void *)s);
	padded_size = pow(2.0,log2(N + 2)+1);
	zp = new double[padded_size];
	op = new double[padded_size];
	for (int i = 0;i < padded_size;i++)
	{
		zp[i] = 0.0;
		op[i] = 1.0;
	}
	bU=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),(void *)s->U);
	bCn=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),(void *)s->C);
    bV=prg->create_buffer(CL_MEM_READ_WRITE, padded_size*sizeof(double),NULL);
    bA=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bB=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bC=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),op);
    br=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bA2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bB2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
    bC2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),op);
    br2=prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double),zp);
	prevC_size = -2;
	prevC_alloced = 10;
	bprevC = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
	bprevH = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
}
void fr_cons_ocl_get_results(cons_fract_solver *s)
{
	queue->EnqueueBuffer(bU, s->U, 0, (N + 2)*sizeof(double));
	queue->EnqueueBuffer(bV, s->V, 0, (N + 2)*sizeof(double));
	queue->EnqueueBuffer(bCn, s->C, 0, (N + 2)*sizeof(double));
	queue->Finish();
}
void fr_cons_ocl_calc_tri(OpenCL_kernel *kA,OpenCL_kernel *kB,OpenCL_kernel *kC,OpenCL_kernel *kr,OpenCL_buffer *bRes,FILE *fi)
{
	size_t nth,lsize=1;
	long long t1,t2,t3,t4;
	// matrix diagonals A,B,C are calculated one thread per element
	nth=N+1;
	set_ABCr_args(kA);
	if ((kA == kCA)||(kA == kUA)) // do save C
	{
		if (kA == kCA)
			prevC_size++;
		if (prevC_size == prevC_alloced) // do realloc
		{
			prevC_alloced *= 2;
			double *b = new double[prevC_alloced*(N+2)];
			double *bH = new double[prevC_alloced*(N+2)];
			queue->EnqueueBuffer(bprevC, b);
			queue->EnqueueBuffer(bprevH, bH);
			delete bprevC;
			delete bprevH;
			bprevC = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
			bprevH = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
			queue->EnqueueWriteBuffer(bprevC, b, 0, prevC_alloced*(N + 2)*sizeof(double));
			queue->EnqueueWriteBuffer(bprevH, bH, 0, prevC_alloced*(N + 2)*sizeof(double));
		}
		kA->SetArg(10,sizeof(int),&prevC_size);
		if (kA == kCA)
			kA->SetBufferArg(bprevC, 9);
		else
			kA->SetBufferArg(bprevH, 9);
	}
	t1=GetTickCount();
	queue->ExecuteKernel(kA,1,&nth,&lsize);
	set_ABCr_args(kB);
	queue->ExecuteKernel(kB,1,&nth,&lsize);
	set_ABCr_args(kC);
	queue->ExecuteKernel(kC,1,&nth,&lsize);
	queue->Finish();
	t2=GetTickCount();
	// right part vector is calculated Br_DIV elements per thread
	nth=((N+2)/Br_DIV)+1;
	lsize=OPT_LS;
	nth=((nth/lsize)+1)*lsize;
	set_ABCr_args(kr);
	kr->SetArg(10, sizeof(int), &prevC_size);
	if (kr == kCr)
		kr->SetBufferArg(bprevC, 9);
	if (kr == kUr)
		kr->SetBufferArg(bprevH, 9);
	queue->ExecuteKernel(kr,1,&nth,&lsize);
	queue->Finish();
	t3=GetTickCount();
	// solve tridiagonal by NVIDIA version of parallel cyclic reducion 
	pcr_solver(bA->buffer,bC->buffer,bB->buffer,br->buffer,bRes->buffer, padded_size,&queue->hCmdQueue,1,bA2->buffer,bC2->buffer,bB2->buffer,br2->buffer);
	queue->Finish();
	t4=GetTickCount();
	fprintf(fi,"fillM %lld fill rp %lld solve %lld\n",t2-t1,t3-t2,t4-t3);
}
void fr_cons_ocl_calc_V(FILE *fi)
{
	size_t nth,lsize;
	long long t1,t2;
	// V vector is calculated Br_DIV elements per thread
	t1=GetTickCount();
	nth=((N+2)/Br_DIV)+1;
	lsize=OPT_LS;
	nth=((nth/lsize)+1)*lsize;
	set_ABCr_args(kV);
	queue->ExecuteKernel(kV,1,&nth,&lsize);
	queue->Finish();
	t2=GetTickCount();
	fprintf(fi,"calc V %lld\n",t2-t1);
}
void solve_cons_ocl(double t,double save_tau)
{
	cons_fract_solver s1, s2;
	int nsave = 1;
	int step = 0;
	FILE *fi1, *fi2;
	fi1 = fopen("log.txt", "wt");
	fi2 = fopen("results.txt", "wt");
	s1.alpha=0.85;
	s1.init();
	s2.alpha=0.85;
	s2.init();
	// set tau
	do
	{
		double e;
		s2.tau = s1.tau / 2;
		s1.init(s1.tau);
		s2.init(s2.tau);
		for (double tt = 0;tt < 0.01;tt += s1.tau)
		{
			s1.V1(1);
			s1.C1(1);
			s1.U1(1);
		}
		for (double tt = 0;tt < 0.01;tt += s2.tau)
		{
			s2.V1(1);
			s2.C1(1);
			s2.U1(1);
		}
		e = 0;
		for (int i = 1;i < N;i++)
		{
			e += (s1.U[i] - s2.U[i])*(s1.U[i] - s2.U[i]);
			e += (s1.V[i] - s2.V[i])*(s1.V[i] - s2.V[i]);
			e += (s1.C[i] - s2.C[i])*(s1.C[i] - s2.C[i]);
		}
		if (e > 1e-5)
			s1.tau /= 2;
		else
			break;
		printf("%g %g\n",e,s1.tau);
		if (s1.tau<0.00001)
		    break;
	} while (1);
	s2.tau = s1.tau;
	s1.init(s1.tau);
	s2.init(s2.tau);
	fr_cons_ocl_init(&s1);
	for (double tt = 0;tt < t;tt += s1.tau)
	{
		double err1=0.0;
		long long i1, i2,i11;
		// GPU
		i1 = GetTickCount();
		fr_cons_ocl_calc_V(fi1);
		fr_cons_ocl_calc_tri(kCA,kCB,kCC,kCr,bCn,fi1);
		fr_cons_ocl_calc_tri(kUA,kUB,kUC,kUr,bU,fi1);
		i1 = GetTickCount()-i1;
		i11= GetTickCount();
		fr_cons_ocl_get_results(&s1);
		i11 = GetTickCount()-i11;
		// CPU
		i2 = GetTickCount();
		s2.V1(0);
		s2.C1(0);
		s2.U1(0);
		i2 = GetTickCount() - i2;
		// calculate errors
		for (int i = 0;i < N + 1;i++)
		{
			err1 += (s2.U[i] - s1.U[i])*(s2.U[i] - s1.U[i]);
			err1 += (s2.V[i] - s1.V[i])*(s2.V[i] - s1.V[i]);
			err1 += (s2.C[i] - s1.C[i])*(s2.C[i] - s1.C[i]);
		}
		fprintf(fi1, "t %g ocl %lld(%lld) cpu %lld e(gpu-cpu) %g\n", tt, i1, i11, i2,err1);
		printf("t %g ocl %lld(%lld) cpu %lld e(gpu-cpu) %g\n", tt, i1, i11, i2,err1);
		fflush(stdout);
		// save result
		if (tt > nsave*save_tau)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g %g %g %g %g %g %g\n", tt,(double)i/N, s1.U[i], s1.V[i], s1.C[i], s2.U[i], s2.V[i], s2.C[i]);
			nsave++;
		}
	}
}
////////////////////////////////////////////////////////////////
// 1d diffusion with generalized Caputo derivative /////////////
////////////////////////////////////////////////////////////////
// generalized Caputo derivative
int func_in_kernel = 2;
double func_power = 0.5;
double global_eps = 1e-5;
int integr_max_niter = 10000;
double g(double t)
{
	if (func_in_kernel == 1)
		return sqrt(t);
	if (func_in_kernel == 2)
		return t*t;
	if (func_in_kernel == 3)
		return pow(t, func_power);
	return t;
}
// should return kf(n)(t) given kf(n-1)(t)=der_nm1
// for n=0 return f(t)
double inv_g_der(double t, int n, double der_nm1)
{
	if (func_in_kernel == 1) // sqrt(t) -> f=t^2 -> df/dt= 2t, d2f/dt=2, d(n>=3)f/dt=0
	{
		if (n == 0)
			return t*t;
		if (n == 1)
			return 2 * t*(der_nm1 / (t*t));
		if (n == 2)
			return 2 * (der_nm1 / (2 * t));
		return 0.0;
	}
	if (func_in_kernel == 2) // t^2 -> f=sqrt(t) -> df/dt=0.5*t(-0.5), d(n+1)f/dt=(d(n)f/dt)*(0.5-n)/t
	{
		if (n == 0)
			return sqrt(t);
		if (n == 1)
			return (0.5 / sqrt(t))*(der_nm1 / sqrt(t));
		return der_nm1*(0.5 - n + 1) / t;
	}
	if (func_in_kernel == 3) // t^k -> f=t^(1/k) -> df/dt=(1/k)*t((1/k)-1), d(n+1)f/dt=(d(n)f/dt)*((1/k)-n)/t
	{
		if (n == 0)
			return pow(t, 1.0 / func_power);
		if (n == 1)
			return ((1.0 / func_power) * pow(t, (1.0 / func_power) - 1))*(der_nm1 / pow(t, 1.0 / func_power));
		return der_nm1*((1.0 / func_power) - n + 1) / t;
	}
	return ((n == 0) ? t : ((n == 1) ? (der_nm1 / t) : 0));
}
class gen_diff_solver {
public:
	double *C; // concentration
	double *Al; // alpha coefficients
	double *Bt; // beta coefficients
	double *Om; // right part
	std::vector<double*> oldC; // for time-fractional derivatives in C-equation
	double tau, h, d;
	double alpha;
	double sigma, Da;
	int tstep;
	// using taylor series on gti-g(x) in g(ti)
	// for right-part - series on g(x)-gti in g(t1)
	std::vector< std::vector< std::vector<double> > > row2_precalc;
	std::vector<double> kb_cache_pts;
	std::vector< std::vector< std::vector<double> > > kb_cache;
	double _kb_row(double t0, double t1, double gtj, double alpha, int &niter)
	{
		double sum = 0.0, v = 0.0, v2, v3, v4;
		double i = 0;
		niter = 0;
		if ((alpha == 1.0) && (g(t1) == gtj))
			return 1.0;
		if ((alpha == 1.0) && (g(t1) != gtj))
			return 0.0;
		if (g(t1) <= gtj)
		{
			v = inv_g_der(gtj, 1, inv_g_der(gtj, 0, 1));
			v *= pow(gtj, 1 - alpha);
			// ((a-g(d))^(n-b+1)-(a-g(c))^(n-b+1))/(n-b+1)
			v3 = pow(fabs(1.0 - g(t1) / gtj), 1 - alpha);
			v4 = pow(fabs(1.0 - g(t0) / gtj), 1 - alpha);
			v2 = -(v3 - v4) / (1 - alpha);
			sum += v2*v;
			while (fabs(v*v2) > global_eps)
			{
				i += 1.0;
				//(((-1)^n/n!)*f(n+1)(a))*a^(n-b+1)
				v = inv_g_der(gtj, i + 1.0, v);
				v *= -gtj / i;
				v3 *= fabs(1.0 - g(t1) / gtj);
				v4 *= fabs(1.0 - g(t0) / gtj);
				v2 = -(v3 - v4) / (i - alpha + 1.0);
				sum += v2*v;
				niter++;
				if (niter > integr_max_niter)
					break;
			}
		}
		else
		{
			double gh = g(t1) - gtj;
			double gl = g(t0) - gtj;
			double g1 = g(t1);
			// f'(g(t1))/gl^(1-a)
			v = inv_g_der(g1, 1, inv_g_der(g1, 0, 1));
			v *= pow(gl, 1 - alpha);
			// I0=1/(1-a) * (gh^(1-a) - gl^(1-a))
			v2 = ((pow(gh, 1 - alpha) / pow(gl, 1 - alpha)) - 1.0) / (1 - alpha); // I0/gl^(1-a)
			sum += v2*v;
			while (fabs(v*v2) > global_eps)
			{
				i += 1.0;
				// gh/(n+1-a) * f(n+1)(g(t1))/gl^(1-a)
				v = inv_g_der(g1, i + 1.0, v);
				v /= (1 - alpha + i) / gh;
				// v2 - Bi, v4 - Ci
				if (i == 1.0)
					v4 = (gl - gh) / gh;
				else
					v4 *= (gl - gh)*((i - 1.0) / i)*((-alpha + i) / (gh*(i - 1.0)));
				v2 = -(v2 + v4);
				sum += v2*v;
				niter++;
				if (niter > integr_max_niter)
					break;
			}
		}
		return sum;
	}
	// using newton binomial on gtj and taylor series of f'(x) in g(t1)
	double _kb_row2(double t0, double t1, double gtj, int idx, int idx2, double alpha, int &niter, int *outer_niter = NULL)
	{
		double sum = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4, v5, v6, v7;
		double gt0 = g(t0), gt1 = g(t1);
		std::vector<double> *cache;
		double i = 0, m;
		int mode = 0;
		int found;
		niter = 0;
		if (outer_niter)
			outer_niter[0] = 0;
		if ((alpha == 1.0) && (g(t1) == gtj))
			return 1.0;
		if ((alpha == 1.0) && (g(t1) != gtj))
			return 0.0;
		while (row2_precalc.size()<(idx + 1))
			row2_precalc.push_back(std::vector< std::vector<double> >());
		while (row2_precalc[idx].size()<(idx2 + 1))
			row2_precalc[idx].push_back(std::vector<double>());
		cache = &row2_precalc[idx][idx2];
		if (g(t1) > gtj)
			mode = 1;
		// (a,n)*(-1)^n
		v1 = 1.0;
		if (mode == 0)
		{
			v2 = pow(gtj, -alpha)*gt1;
			v6 = gt0 / gt1;
		}
		else
		{
			v2 = pow(gt0, 1 - alpha);
			v6 = pow(gt1, 1 - alpha) / pow(gt0, 1 - alpha);
		}
		do
		{
			// (a,n)*(-1)^n
			if (i != 0.0)
			{
				v1 *= -(-alpha - i + 1.0) / i;
				// (a^(-b-n)*g(t1)^(n+1)
				if (mode == 0)
				{
					v2 *= gt1 / gtj;
					v6 *= gt0 / gt1;
				}
				else
				{
					v2 *= gtj / gt0;
					v6 *= gt0 / gt1;
				}
			}
			// integrate(f'(x)x^n,x,g(c),g(d))/g(t1)^(n+1)
			found = 0;
			if (cache->size()>(int)i)
			{
				v3 = (*cache)[(int)i];
				found = 1;
			}
			if (found == 0)
			{
				if ((i == 0.0) && (mode == 0))
					v3 = (t1 - t0) / gt1;
				else
				{
					v3 = 0.0;
					m = 0.0;
					// f(m+1)(gd)/m!
					v4 = inv_g_der(gt1, 1, inv_g_der(gt1, 0, 0));
					// I(n,m)=integrate(x^n*(x-g(t1))^m,x,g(t0),g(t1));
					// I(n,0)=(1/n+1)*(g(t1)^(n+1)-g(t0)^n+1)
					if (mode == 0)
						v5 = (1.0 / (i + 1.0))*(1.0 - v6);
					else
						v5 = -(1.0 / (-alpha - i + 1.0))*(1.0 - v6);
					v3 += v4*v5;
					v7 = 1;
					while (fabs(v4*v5) > (fabs(v3)*global_eps))
					{
						m += 1.0;
						// (f(m+1)(gd)/m!)
						v4 = inv_g_der(gt1, m + 1.0, v4);
						v4 /= m;
						if (mode == 0)
							v4 /= ((i + m + 1.0) / (gt1*m));
						else
							v4 /= ((-alpha - i + m + 1.0) / (gt1*m));

						if (m == 1.0)
						{
							if (mode == 0)
								v7 = (1.0 / (gt1))*(v6*(gt0 - gt1));
							else
								v7 = (1.0 / (gt1))*(gt0 - gt1);
						}
						else
						{
							if (mode == 0)
								v7 *= (gt0 - gt1)*((m - 1.0) / m)*((i + m) / (gt1*(m - 1.0)));
							else
								v7 *= (gt0 - gt1)*((m - 1.0) / m)*((-alpha - i + m) / (gt1*(m - 1.0)));
						}
						// I(n,m)=-(I(n,m-1)+(1/(g(t1)*m))*(g(t0)^(n+1)*(g(t0)-g(t1))^m))/((n+m+1)/(g(t1)*m)
						v5 = -(v5 + v7);
						v3 += v4*v5;
						niter++;
						if (niter>integr_max_niter)
							break;
					}
				}
				while (cache->size()<(i + 1.0))
					cache->push_back(0);
				(*cache)[(int)i] = v3;
			}
			sum += v1*v2*v3;
			i += 1.0;
			niter++;
			if (outer_niter)
				outer_niter[0]++;
			if (niter>integr_max_niter)
				break;
		} while (fabs(v1*v2*v3) > global_eps);
		return sum;
	}
	//b(t0,t1,a) = int((g(x_a)-g(tau))^(-alpha),tau=x_t0,x_t1)
	// use newton binomial near t0==0
	// use taylor series near t1==a
	// move usage boundary to ensure minimal number of iteration done by both algorithms
	int a3k;
	double last_points;
	int last_i1;
	int last_a;
	std::vector< std::vector<double> > *last_vi1;
	double* last_vi1a;
	int last_vi1a_size;
	double kb(double alpha, int t0, int t1, int a, double dt)
	{
		std::vector< std::vector<double> > *vi1;
		double* vi1a = NULL;
		int vi1a_size;
		int niter, niter2;
		double v;
		if (alpha == 1.0)
		{
			if (t1 == a)
				return 1.0;
			else
				return 0.0;
		}
		// try to find in cache
		int i1 = -1;
		if (last_points == dt)
		{
			i1 = last_i1;
			vi1 = last_vi1;
			if (last_a == a)
			{
				vi1a = last_vi1a;
				vi1a_size = last_vi1a_size;
			}
		}
		else
		{
			for (int i = 0;i < kb_cache_pts.size();i++)
				if (kb_cache_pts[i] == dt)
				{
					i1 = i;
					break;
				}
			if (i1 == -1)
			{
				kb_cache_pts.push_back(dt);
				kb_cache.push_back(std::vector< std::vector<double> >());
				i1 = kb_cache_pts.size() - 1;
			}
			last_i1 = i1;
			last_points = dt;
			vi1 = last_vi1 = &kb_cache[i1];
		}
		if (vi1a == NULL)
		{
			if (vi1->size()>a)
			{
				last_vi1a = vi1a = &((*vi1)[a][0]);
				last_vi1a_size = vi1a_size = (*vi1)[a].size();
				last_a = a;
			}
		}
		if (vi1a)
			if (vi1a_size>t0)
			{
				v = vi1a[t0];
				if (v != -1)
					return v;
			}
		// calculate
		if ((t0 <= a3k) && (t1 != a) && (t0 != a))
		{
			v = _kb_row2(dt*t0, dt*t1, g(dt*a), i1, t0, alpha, niter);
			if (t0 == a3k)
			{
				double v2;
				v2 = _kb_row(dt*t0, dt*t1, g(dt*a), alpha, niter2);
				if (niter2 > niter)
					a3k++;
				else
					if (a3k != 1)
						a3k--;
			}
		}
		else
			v = _kb_row(dt*t0, dt*t1, g(dt*a), alpha, niter2);
		// put in cache
		while (kb_cache[i1].size() <= a) kb_cache[i1].push_back(std::vector<double>());
		while (kb_cache[i1][a].size() <= t0) kb_cache[i1][a].push_back(-1);
		kb_cache[i1][a][t0] = v;
		return v;
	}
	gen_diff_solver()
	{
		C = Al = Bt = Om = NULL;
	}
	void init(double aa = 0.8, double _tau = 0.0)
	{
		clear();
		int padded_size = pow(2.0, log2(N + 2) + 1);
		C = new double[padded_size];
		Al = new double[N + 2];
		Bt = new double[N + 2];
		Om = new double[N + 2];

		h = 1.0 / N;
		if (_tau == 0)
			tau = 0.1*(1.0 / N);
		else
			tau = _tau;
		sigma = 0.3;
		d = 0.032;
		alpha = aa;
		tstep = 0;
		oldC.clear();
		row2_precalc.clear();
		kb_cache_pts.clear();
		kb_cache.clear();
		a3k = 1;
		last_points = NULL;
		if (alpha != 1.0)
			Da = 1.0 / Gamma(1.0 - alpha);
		else
			Da = 1.0;
		for (int i = 1;i < N + 1;i++)
			C[i] = 0.0;
		C[0] = 1.0;
	}
	double A_(int i)
	{
		return d / (h * h);
	}
	double B(int i)
	{
		return d / (h * h);
	}
	double R(int i)
	{
		return (2.0*d / (h*h)) + sigma*Da*kb(alpha, tstep, tstep + 1, tstep + 1, tau) / tau;
	}
	// alpha coeffients
	void al1()
	{
		Al[1] = 0;
		for (int i = 1;i < N;i++)
			Al[i + 1] = B(i) / (R(i) - A_(i)*Al[i]);
	}
	// right part
	void Om1()
	{
		for (int i = 1;i < N;i++)
		{
			Om[i] = -(sigma*Da*kb(alpha, tstep, tstep + 1, tstep + 1, tau) / tau)*C[i];
			// time-fractional derivative
			double time_sum = 0;
			double kv, diff;
			if ((tstep - 1) >= 1)
				for (int t = 0;t < oldC.size() - 1;t++)
				{
					kv = kb(alpha, t, t + 1, tstep + 1, tau);
					diff = (oldC[t + 1][i] - oldC[t][i]);
					time_sum += kv*diff / tau;
				}
			if ((tstep - 1) >= 0)
			{
				kv = kb(alpha, tstep - 1, tstep, tstep + 1, tau);
				diff = (C[i] - oldC[tstep - 1][i]);
				time_sum += kv*diff / tau;
			}
			Om[i] += sigma*Da*time_sum;
		}
	}
	// beta coeffients
	void bt1()
	{
		Bt[1] = 1.0;
		for (int i = 1;i < N;i++)
			Bt[i + 1] = (Al[i + 1] / B(i)) * (A_(i)*Bt[i] - Om[i]);
	}
	// calc C
	void C1()
	{
		al1();
		Om1();
		bt1();
		C[N] = 0.0;
		C[0] = 1.0;
		for (int i = N - 1;i >= 1;i--)
			C[i] = Al[i + 1] * C[i + 1] + Bt[i + 1];
		// save H for time-fractional derivative calculations
		double *cs = new double[N + 2];
		memcpy(cs, C, (N + 2)*sizeof(double));
		oldC.push_back(cs);
		tstep++;
	}
	void clear()
	{
		if (C) delete[] C;
		if (Al) delete[] Al;
		if (Bt) delete[] Bt;
		if (Om) delete[] Om;
		for (int i = 0;i < oldC.size();i++)
			delete oldC[i];
		oldC.clear();
	}
	~gen_diff_solver()
	{
		clear();
	}
};
void gen_diff_solve(double t, double save_tau)
{
	gen_diff_solver s1, s2, s3;
	int nsave = 1;
	FILE *fi2;
	fi2 = fopen("results.txt", "wt");
	s1.init(0.9);
	s2.init(0.9);
	s3.init(0.7);
	// set tau
	do
	{
		double e;
		s2.tau = s1.tau / 2;
		s1.init(0.9, s1.tau);
		s2.init(0.9, s2.tau);
		for (double tt = 0;tt < 0.01;tt += s1.tau)
			s1.C1();
		for (double tt = 0;tt < 0.01;tt += s2.tau)
			s2.C1();
		e = 0;
		for (int i = 1;i < N;i++)
			e += (s1.C[i] - s2.C[i])*(s1.C[i] - s2.C[i]);
		if (e > 1e-5)
			s1.tau /= 2;
		else
			break;
		printf("%g %g\n", e, s1.tau);
		if (s1.tau<0.00001)
			break;
	} while (1);
	s1.init(0.9, s1.tau);
	s2.init(0.8, s1.tau);
	s3.init(0.7, s1.tau);
	for (double tt = 0;tt < t;tt += s1.tau)
	{
		long long i1;
		i1 = GetTickCount();
		s1.C1();
		s2.C1();
		s3.C1();
		i1 = GetTickCount() - i1;
		printf("t %g time %lld\n", tt, i1);
		fflush(stdout);
		// save result
		if (tt > nsave*save_tau)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g %g %g %g\n", tt, (double)i / N, s1.C[i], s2.C[i], s3.C[i]);
			nsave++;
		}
	}
}
// consolidation OCL
OpenCL_buffer *b_kb, *b_r2;
OpenCL_kernel *k_kb, *k_sum2;
int s2_rl,r00,r10;
std::vector<int> devices;
std::vector<int> n_contexts;
typedef struct {
	int id,r0,r1,s2_rl;
	OpenCL_program *prg;
	OpenCL_commandqueue *queue;
	OpenCL_prg *prog;
	OpenCL_kernel *kCr,*k_saveC;
	OpenCL_kernel *k_kb, *k_sum2;
	OpenCL_buffer *b_kb, *b_r2;
	OpenCL_buffer *bA,*bB,*bC, *br,*bU,*bV,*bCn, *bS, *bprevC;
} additional_context;
std::vector<additional_context> contexts;
double *cpu_kb;
char *gen_diff_opencl_text = "\n\
#pragma OPENCL EXTENSION %s : enable \n\
#define tau (*((__global double *)(class+%d)))\n\
#define h (*((__global double *)(class+%d)))\n\
#define d (*((__global double *)(class+%d)))\n\
#define alpha (*((__global double *)(class+%d)))\n\
#define sigma (*((__global double *)(class+%d)))\n\
#define Da (*((__global double *)(class+%d)))\n\
#define func_in_kernel %d\n\
#define func_power %g\n\
#define global_eps %g\n\
#define integr_max_niter %d\n\
#define GS %d\n\
double g(double t) \n\
{ \n\
	if (func_in_kernel == 1) \n\
		return sqrt(t); \n\
	if (func_in_kernel == 2) \n\
		return t*t;\n\
	if (func_in_kernel == 3)\n\
		return pow(t,func_power);\n\
	return t;\n\
}\n\
double inv_g_der(double t, int n, double der_nm1)\n\
{\n\
	if (func_in_kernel == 1) // sqrt(t) -> f=t^2 -> df/dt= 2t, d2f/dt=2, d(n>=3)f/dt=0 \n\
	{ \n\
		if (n == 0) \n\
			return t*t; \n\
		if (n == 1) \n\
			return 2 * t*(der_nm1 / (t*t)); \n\
		if (n == 2) \n\
			return 2 * (der_nm1 / (2 * t)); \n\
		return 0.0; \n\
	} \n\
	if (func_in_kernel == 2) // t^2 -> f=sqrt(t) -> df/dt=0.5*t(-0.5), d(n+1)f/dt=(d(n)f/dt)*(0.5-n)/t \n\
	{ \n\
		if (n == 0) \n\
			return sqrt(t); \n\
		if (n == 1) \n\
			return (0.5 / sqrt(t))*(der_nm1 / sqrt(t)); \n\
		return der_nm1*(0.5 - n + 1) / t; \n\
	} \n\
	if (func_in_kernel == 3) // t^k -> f=t^(1/k) -> df/dt=(1/k)*t((1/k)-1), d(n+1)f/dt=(d(n)f/dt)*((1/k)-n)/t \n\
	{ \n\
		if (n == 0) \n\
			return pow(t,1.0/func_power); \n\
		if (n == 1) \n\
			return ((1.0/func_power) * pow(t,(1.0/func_power)-1))*(der_nm1 / pow(t,1.0/func_power)); \n\
		return der_nm1*((1.0/func_power) - n + 1) / t; \n\
	} \n\
	return ((n == 0) ? t : ((n == 1) ? (der_nm1 / t) : 0)); \n\
} \n\
double _kb_row(double t0, double t1, double gtj,double _alpha,int *niter) \n\
{ \n\
	double sum = 0.0, v = 0.0, v2, v3, v4; \n\
	double i = 0; \n\
	niter[0] = 0; \n\
	if ((_alpha == 1.0)&&(g(t1)==gtj)) \n\
		return 1.0; \n\
	if ((_alpha==1.0)&&(g(t1)!=gtj)) \n\
		return 0.0; \n\
	if (g(t1) <= gtj) \n\
	{ \n\
		v = inv_g_der(gtj, 1, inv_g_der(gtj, 0, 1)); \n\
		v *= pow(gtj, 1 - _alpha); \n\
		v3 = pow(fabs(1.0 - g(t1) / gtj), 1 - _alpha); \n\
		v4 = pow(fabs(1.0 - g(t0) / gtj), 1 - _alpha); \n\
		v2 = -(v3 - v4) / (1 - _alpha); \n\
		sum += v2*v; \n\
		while (fabs(v*v2) > global_eps) \n\
		{ \n\
			i += 1.0; \n\
			v = inv_g_der(gtj, i + 1.0, v); \n\
			v *= -gtj / i; \n\
			v3 *= fabs(1.0 - g(t1) / gtj);  \n\
			v4 *= fabs(1.0 - g(t0) / gtj); \n\
			v2 = -(v3 - v4) / (i - _alpha + 1.0);\n\
			sum += v2*v;\n\
			niter[0]++;\n\
			if (niter[0] > integr_max_niter)\n\
				break;\n\
		}\n\
	}\n\
	else\n\
	{\n\
		double gh = g(t1) - gtj;\n\
		double gl = g(t0) - gtj;\n\
		double g1 = g(t1);\n\
		v = inv_g_der(g1, 1, inv_g_der(g1, 0, 1));\n\
		v *= pow(gl, 1 - _alpha);\n\
		v2 =( (pow(gh, 1 - _alpha) / pow(gl, 1 - _alpha)) - 1.0) / (1 - _alpha); // I0/gl^(1-a)\n\
		sum += v2*v;\n\
		while (fabs(v*v2) > global_eps)\n\
		{\n\
			i += 1.0;\n\
			v = inv_g_der(g1, i + 1.0, v);\n\
			v /= (1 - _alpha + i) / gh;\n\
			if (i == 1.0)\n\
				v4 = (gl - gh)/gh;\n\
			else\n\
				v4 *= (gl - gh)*((i - 1.0) / i)*((-_alpha + i) / (gh*(i - 1.0)));\n\
			v2 = -(v2 + v4);\n\
			sum += v2*v;\n\
			niter[0]++;\n\
			if (niter[0] > integr_max_niter)\n\
				break;\n\
		}\n\
	}\n\
	return sum;\n\
}\n\
double _kb_row2(double t0, double t1, double gtj, double _alpha,int *niter)\n\
{\n\
	double sum = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4, v5, v6, v7;\n\
	double gt0 = g(t0), gt1 = g(t1);\n\
	double i = 0, m;\n\
	int mode=0; \n\
	niter[0] = 0;\n\
	if ((_alpha == 1.0)&&(g(t1)==gtj))\n\
		return 1.0;\n\
	if ((_alpha==1.0)&&(g(t1)!=gtj))\n\
		return 0.0;\n\
	if (g(t1) > gtj)\n\
		mode = 1;\n\
	v1 = 1.0;\n\
	if (mode == 0)\n\
	{\n\
		v2 = pow(gtj, -_alpha)*gt1;\n\
		v6 = gt0 / gt1;\n\
	}\n\
	else\n\
	{\n\
		v2 = pow(gt0, 1-_alpha);\n\
		v6 = pow(gt1,1-_alpha) / pow(gt0,1-_alpha);\n\
	}\n\
	do\n\
	{\n\
		if (i != 0.0)\n\
		{\n\
			v1 *= -(-_alpha - i + 1.0) / i;\n\
			if (mode == 0)\n\
			{\n\
				v2 *= gt1 / gtj;\n\
				v6 *= gt0 / gt1;\n\
			}\n\
			else\n\
			{\n\
				v2 *= gtj / gt0;\n\
				v6 *= gt0 / gt1;\n\
			}\n\
		}\n\
		{\n\
			if ((i == 0.0)&&(mode==0))\n\
				v3 = (t1 - t0) / gt1;\n\
			else\n\
			{\n\
				v3 = 0.0;\n\
				m = 0.0;\n\
				v4 = inv_g_der(gt1, 1, inv_g_der(gt1, 0, 0));\n\
				if (mode==0)\n\
					v5 = (1.0 / (i + 1.0))*(1.0 - v6);\n\
				else\n\
					v5 = -(1.0 / (-_alpha-i + 1.0))*(1.0 - v6);\n\
				v3 += v4*v5;\n\
				v7 = 1;\n\
				while (fabs(v4*v5) > (fabs(v3)*global_eps))\n\
				{\n\
					m+=1.0;\n\
					v4 = inv_g_der(gt1, m + 1.0, v4);\n\
					v4 /= m;\n\
					if (mode==0)\n\
						v4 /= ((i + m + 1.0) / (gt1*m));\n\
					else\n\
						v4 /= ((-_alpha-i + m + 1.0) / (gt1*m));\n\
					if (m == 1.0)\n\
					{\n\
						if (mode == 0)\n\
							v7 = (1.0 / (gt1))*(v6*(gt0 - gt1));\n\
						else\n\
							v7 = (1.0 / (gt1))*(gt0 - gt1);\n\
					}\n\
					else\n\
					{\n\
						if (mode==0)\n\
							v7 *= (gt0 - gt1)*((m - 1.0) / m)*((i + m) / (gt1*(m - 1.0)));\n\
						else\n\
							v7 *= (gt0 - gt1)*((m - 1.0) / m)*((-_alpha-i + m) / (gt1*(m - 1.0)));\n\
					}\n\
					v5 = -(v5 + v7);\n\
					v3 += v4*v5;\n\
					niter[0]++;\n\
					if (niter[0]>integr_max_niter)\n\
						break;\n\
				}\n\
			}\n\
		}\n\
		sum += v1*v2*v3;\n\
		i+=1.0;\n\
		niter[0]++;\n\
		if (niter[0]>integr_max_niter)\n\
			break;\n\
	} while (fabs(v1*v2*v3) > global_eps);\n\
	return sum;\n\
}\n\
double kb(double _alpha, int t0,int t1,int a,double dt,int *a3k)\n\
{\n\
	double v;\n\
	int niter,niter2;\n\
	if (_alpha == 1.0)\n\
	{\n\
		if (t1==a)\n\
			return 1.0;\n\
		else\n\
			return 0.0;\n\
	}\n\
	if ((t0 <= a3k[0])&&(t1!=a)&&(t0!=a))\n\
	{\n\
		v = _kb_row2(dt*t0,dt*t1,g(dt*a),_alpha,&niter);\n\
		if (t0 == a3k[0])\n\
		{\n\
			double v2;\n\
			v2 = _kb_row(dt*t0, dt*t1, g(dt*a),_alpha,&niter2);\n\
			if (niter2 > niter)\n\
				a3k[0]++;\n\
			else\n\
				if (a3k[0] != 1)\n\
					a3k[0]--;\n\
		}\n\
	}\n\
	else\n\
		v = _kb_row(dt*t0, dt*t1, g(dt*a), _alpha, &niter2);\n\
	return v;\n\
}\n\
__kernel void calc_kb(__global double *kbs,int tstep,__global char *class)\n\
{\n\
	int i=get_global_id(0);\n\
	int a3k=10;\n\
	kbs[i]=kb(alpha, i, i+1, tstep+1, tau,&a3k);\n\
}\n\
__kernel void UA(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevH,int prevC_size)\n\
{\n\
	int i=get_global_id(0);\n\
	// save H \n\
	if (prevC_size>=0)\n\
		prevH[prevC_size*(N+2)+i]=C[i];\n\
	if (i==0) {A[i]=0.0; return;}\n\
	if (i==N) {A[i]=0.0; return;}\n\
	A[i]=-d/(h*h);\n\
}\n\
__kernel void UB(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) { B[i]=0.0; return;}\n\
	if (i==N) { B[i]=0.0;return;}\n\
	B[i]=-d/(h*h);\n\
}\n\
__kernel void UC(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,int tstep,__global double *kbs)\n\
{\n\
	int i=get_global_id(0);\n\
	if (i==0) {mC[i]=1.0;return;}\n\
	if (i==N) {mC[i]=1.0;return;}\n\
	mC[i]=(2.0*d/(h*h))+sigma*Da*kbs[tstep]/tau;\n\
}\n\
__kernel void URightPart(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size,int tstep,__global double *kbs,int r0,int r1)\n\
{\n\
	int i=get_global_id(0)+r0;\n\
	double time_sum = 0,sum;\n\
	double diff;\n\
	sum = (sigma*Da*kbs[tstep]/tau)*C[i];\n\
	for (int t = 0;t < prevC_size;t++)\n\
	{\n\
		diff = (prevC[(t + 1)*(N+2)+i] - prevC[t*(N+2)+i]);\n\
		time_sum += kbs[t]*diff/tau;\n\
	}\n\
	if (prevC_size>=0)\n\
	{\n\
		diff = (C[i] - prevC[prevC_size*(N+2)+i]);\n\
		time_sum += kbs[tstep-1]*diff/tau;\n\
	}\n\
	if (i<r1)\n\
	{\n\
		if (i==0) {r[i]=1.0;return;}\n\
		if (i==N) {r[i]=0.0;return;}\n\
		r[i] = sum-sigma*Da*time_sum;\n\
	}\n\
}\n\
__kernel void URightPart_opt(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size,int tstep,__global double *kbs,int r0,int r1)\n\
{\n\
	int l=get_local_id(0);\n\
	int g=get_global_id(0)+r0;\n\
	__local double kk[GS];\n\
	double sum = 0,time_sum=0;\n\
	double diff;\n\
	int i;\n\
	if (g<N) sum = (sigma*Da*kbs[tstep]/tau)*C[g];\n\
	for (int t = 0;t < prevC_size;t+=GS)\n\
	{\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		if ((t+l)<prevC_size) kk[l]=kbs[t+l];\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		if (g<N)\n\
		for (int t2=0;t2<GS;t2++)\n\
		if ((t+t2)<prevC_size)\n\
		{\n\
			diff = (prevC[((t+t2)+ 1)*(N+2)+g] - prevC[(t+t2)*(N+2)+g]);\
			time_sum += kk[t2]*diff/tau;\n\
		}\n\
	}\
	if (g<N)\n\
	if (prevC_size>=0)\n\
	{\n\
		diff = (C[g] - prevC[prevC_size*(N+2)+g]);\
		time_sum += kbs[tstep-1]*diff/tau;\
	}\n\
	if (g<r1)\n\
	{\n\
		if (g<N)\n\
			r[g] = sum-sigma*Da*time_sum;\
		if (g==0) r[g]=1.0;\n\
		if (g==N) r[g]=0.0;\n\
	}\n\
}\n\
__kernel void URightPart_opt2(__global double *A,__global double *B,__global double *mC,__global double *r,__global double *U,__global double *V,__global double *C,__global char *class,int N,__global double *prevC,int prevC_size,int tstep,__global double *kbs,__global double *rs,int rl,int r0,int r1)\n\
{\n\
	int l=get_local_id(0);\n\
	int gl=get_global_id(0);\n\
	__local double kk[GS];\n\
	double sum = 0,time_sum=0;\n\
	double diff;\n\
	int i;\n\
	int g,bl;\n\
	bl=gl/rl;\n\
	g=r0+gl-bl*rl;\n\
	int i1=(bl+1)*GS;\n\
	if (i1>=prevC_size) i1=prevC_size;\n\
	for (int t = bl*GS;t < i1;t+=GS)\n\
	{\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		if ((t+l)<prevC_size) kk[l]=kbs[t+l];\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		if (g<N)\n\
		for (int t2=0;t2<GS;t2++)\n\
		if ((t+t2)<prevC_size)\n\
		{\n\
			diff = (prevC[((t+t2)+ 1)*(N+2)+g] - prevC[(t+t2)*(N+2)+g]);\
			time_sum += kk[t2]*diff/tau;\n\
		}\n\
	}\
	if (g<N)\n\
	if (bl==0)\n\
	{\n\
		if (prevC_size>=0)\n\
		{\n\
			diff = (C[g] - prevC[prevC_size*(N+2)+g]);\
			time_sum += kbs[tstep-1]*diff/tau;\
		}\n\
		sum=(sigma*Da*kbs[tstep]/tau)*C[g];\n\
	}\n\
	if (g<r1)\n\
		if (g<N)\n\
			rs[gl] = sum-sigma*Da*time_sum;\
}\n\
__kernel void URightPart_opt2_sum(__global double *r,__global char *class,int rl,__global double *rs,int prevC_size,int N,int r0,int r1)\n\
{\n\
	int g=get_global_id(0)+r0;\n\
	double sum=0.0;\n\
	if (prevC_size>=1)\n\
		for (int i=0;i<prevC_size;i+=GS)\n\
			sum+=rs[(i/GS)*rl+(g-r0)];\n\
	else\n\
		sum=rs[(g-r0)];\n\
	if (g<r1)\n\
	{\n\
		r[g]=sum;\n\
		if (g==0) r[g]=1.0;\n\
		if (g==N) r[g]=0.0;\n\
	}\n\
}\n\
__kernel void save_C(__global double *C,int N,__global double *prevH,int prevC_size,int r0)\n\
{\n\
	int i=get_global_id(0)+r0;\n\
	if (prevC_size>=0)\n\
		prevH[prevC_size*(N+2)+i]=C[i];\n\
}\n\
\n";
void gen_diff_ocl_init(gen_diff_solver *s)
{
	char *text = new char[strlen(gen_diff_opencl_text) * 2];
	sprintf(text, gen_diff_opencl_text, ((double_ext == 0) ? "cl_amd_fp64" : "cl_khr_fp64")
			, ((char *)&s->tau) - (char *)s
			, ((char *)&s->h) - (char *)s
			, ((char *)&s->d) - (char *)s
			, ((char *)&s->alpha) - (char *)s
			, ((char *)&s->sigma) - (char *)s
			, ((char *)&s->Da) - (char *)s
			, func_in_kernel
			, func_power
			, global_eps
			, integr_max_niter
			, OPT_LS);
	prg = new OpenCL_program(1);
	queue = prg->create_queue(device, 0);
	prog = prg->create_program(text);
	pcr_small_systems_init(prog->hContext[0], 0);
	kCA = prg->create_kernel(prog, "UA");
	kCB = prg->create_kernel(prog, "UB");
	kCC = prg->create_kernel(prog, "UC");
	if ((OPTIMIZED_RIGHTPART & 3) == 2)
	{
		kCr = prg->create_kernel(prog, "URightPart_opt2");
		k_sum2 = prg->create_kernel(prog, "URightPart_opt2_sum");
	}
	if ((OPTIMIZED_RIGHTPART & 3) == 1)
		kCr = prg->create_kernel(prog, "URightPart_opt");
	if ((OPTIMIZED_RIGHTPART & 3) == 0)
		kCr = prg->create_kernel(prog, "URightPart");
	k_kb = prg->create_kernel(prog, "calc_kb");
	bS = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(gen_diff_solver), (void *)s);
	padded_size = pow(2.0, log2(N + 2) + 1);
	zp = new double[padded_size];
	op = new double[padded_size];
	for (int i = 0;i < padded_size;i++)
	{
		zp[i] = 0.0;
		op[i] = 1.0;
	}
	bU = prg->create_buffer(CL_MEM_READ_WRITE, padded_size*sizeof(double), NULL);
	bCn = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), (void *)s->C);
	bV = prg->create_buffer(CL_MEM_READ_WRITE, padded_size*sizeof(double), NULL);
	bA = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
	bB = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
	bC = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), op);
	br = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
	bA2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
	bB2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
	bC2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), op);
	br2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
	prevC_size = -2;
	prevC_alloced = 10;
	bprevC = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
	b_kb = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*sizeof(double), NULL);
	r00 = 0;
	r10 = N + 1;
	s2_rl = (((N + 2) / OPT_LS) + 1)*OPT_LS;
	b_r2 = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*s2_rl*sizeof(double), NULL);
	if (OPTIMIZED_RIGHTPART & 4)
		cpu_kb = new double[prevC_alloced];
	if (OPTIMIZED_RIGHTPART & 8)
	{
		int id = 1;
		for (int i = 0;i < devices.size();i++)
			for (int j = 0;j < n_contexts[i];j++)
			{
				additional_context a;
				a.id = id++;
				a.prg = new OpenCL_program(1);
				a.queue = a.prg->create_queue(devices[i], 0);
				a.prog = a.prg->create_program(text);
				if ((OPTIMIZED_RIGHTPART & 3) == 2)
				{
					a.kCr = a.prg->create_kernel(a.prog, "URightPart_opt2");
					a.k_sum2 = a.prg->create_kernel(a.prog, "URightPart_opt2_sum");
				}
				if ((OPTIMIZED_RIGHTPART & 3) == 1)
					a.kCr = a.prg->create_kernel(a.prog, "URightPart_opt");
				if ((OPTIMIZED_RIGHTPART & 3) == 0)
					a.kCr = a.prg->create_kernel(a.prog, "URightPart");
				a.k_kb = a.prg->create_kernel(a.prog, "calc_kb");
				a.k_saveC = a.prg->create_kernel(a.prog, "save_C");
				a.b_kb = a.prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*sizeof(double), NULL);
				a.b_r2 = a.prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*s2_rl*sizeof(double), NULL);
				a.bC = a.prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), op);
				a.br = a.prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
				a.bprevC = a.prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
				a.bS = a.prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(gen_diff_solver), (void *)s);
				a.bU = a.prg->create_buffer(CL_MEM_READ_WRITE, padded_size*sizeof(double), NULL);
				a.bCn = a.prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), (void *)s->C);
				a.bV = a.prg->create_buffer(CL_MEM_READ_WRITE, padded_size*sizeof(double), NULL);
				a.bA = a.prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
				a.bB = a.prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, padded_size*sizeof(double), zp);
				contexts.push_back(a);
			}
		r00 = 0;
		r10 = (N+2)/(contexts.size()+1);
		s2_rl = ((r10 / OPT_LS) + 1)*OPT_LS;
		for (int i = 0;i < contexts.size();i++)
		{
			contexts[i].r0 = r10*contexts[i].id;
			contexts[i].r1 = r10*(contexts[i].id+1);
			if (contexts[i].id == contexts.size())
				contexts[i].r1 = N + 1;
			contexts[i].s2_rl = (((contexts[i].r1 - contexts[i].r0)/ OPT_LS) + 1)*OPT_LS;
		}
	}
	delete[] text;
}
void gen_diff_ocl_get_results(gen_diff_solver *s)
{
	queue->EnqueueBuffer(bCn, s->C, 0, (N + 2)*sizeof(double));
	queue->Finish();
}
void gen_diff_ocl_calc_tri(OpenCL_kernel *kA, OpenCL_kernel *kB, OpenCL_kernel *kC, OpenCL_kernel *kr, OpenCL_buffer *bRes, FILE *fi, int tstep, gen_diff_solver *s)
{
	size_t nth, lsize = 1;
	int r0=r00, r1=r10;
	long long t1, t2, t3, t4, t5;
	int df = 0;
	if (OPTIMIZED_RIGHTPART & 4)
		df = 3;
	prevC_size++;
	if (prevC_size == (prevC_alloced - df)) // do realloc
	{
		prevC_alloced *= 2;
		double *b = new double[prevC_alloced*(N + 2)];
		queue->EnqueueBuffer(bprevC, b);
		delete bprevC;
		delete b_kb;
		bprevC = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
		queue->EnqueueWriteBuffer(bprevC, b, 0, prevC_alloced*(N + 2)*sizeof(double));
		b_kb = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*sizeof(double), NULL);
		if (OPTIMIZED_RIGHTPART & 4)
		{
			double *bb = new double[prevC_alloced / 2];
			memcpy(bb, cpu_kb, (prevC_alloced / 2)*sizeof(double));
			delete[] cpu_kb;
			cpu_kb = new double[prevC_alloced];
			memcpy(cpu_kb, bb, (prevC_alloced / 2)*sizeof(double));
			delete[] bb;
		}
		if ((OPTIMIZED_RIGHTPART & 3) == 2)
		{
			delete b_r2;
			b_r2 = prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*s2_rl*sizeof(double), NULL);
		}
		if (OPTIMIZED_RIGHTPART & 8)
		{
			for (int i = 0;i < contexts.size();i++)
			{
				double *b = new double[prevC_alloced*(N + 2)];
				contexts[i].queue->EnqueueBuffer(contexts[i].bprevC, b);
				delete contexts[i].bprevC;
				delete contexts[i].b_kb;
				contexts[i].bprevC = contexts[i].prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*(N + 2)*sizeof(double), NULL);
				contexts[i].queue->EnqueueWriteBuffer(contexts[i].bprevC, b, 0, prevC_alloced*(N + 2)*sizeof(double));
				contexts[i].b_kb = contexts[i].prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*sizeof(double), NULL);
				if ((OPTIMIZED_RIGHTPART & 3) == 2)
				{
					delete contexts[i].b_r2;
					contexts[i].b_r2 = contexts[i].prg->create_buffer(CL_MEM_READ_WRITE, prevC_alloced*contexts[i].s2_rl*sizeof(double), NULL);
				}
			}
		}
	}
	// calculate integrals values
	if ((OPTIMIZED_RIGHTPART & 4) == 0)
	{
		k_kb->SetBufferArg(b_kb, 0);
		k_kb->SetArg(1, sizeof(int), &tstep);
		k_kb->SetBufferArg(bS, 2);
		nth = tstep + 1;
		queue->ExecuteKernel(k_kb, 1, &nth, &lsize);
	}
	else
	{
		if (tstep == 0)
			cpu_kb[0] = s->kb(s->alpha, 0, 1, 1, s->tau);
		queue->EnqueueWriteBuffer(b_kb, cpu_kb, 0, (tstep + 1)*sizeof(double));
	}
	// matrix diagonals A,B,C are calculated one thread per element
	r0 = r00;
	r1 = r10;
	nth = N + 1;
	set_ABCr_args(kA);
	kA->SetArg(10, sizeof(int), &prevC_size);
	kA->SetBufferArg(bprevC, 9);
	t1 = GetTickCount();
	queue->ExecuteKernel(kA, 1, &nth, &lsize);
	set_ABCr_args(kB);
	queue->ExecuteKernel(kB, 1, &nth, &lsize);
	set_ABCr_args(kC);
	kC->SetArg(9, sizeof(int), &tstep);
	kC->SetBufferArg(b_kb, 10);
	queue->ExecuteKernel(kC, 1, &nth, &lsize);
	t2 = GetTickCount();
	// right part vector is calculated Br_DIV elements per thread
	nth= (r1 - r0);
	set_ABCr_args(kr);
	kr->SetArg(10, sizeof(int), &prevC_size);
	kr->SetBufferArg(bprevC, 9);
	kr->SetArg(11, sizeof(int), &tstep);
	kr->SetBufferArg(b_kb, 12);
	if (((OPTIMIZED_RIGHTPART & 3) == 0)|| ((OPTIMIZED_RIGHTPART & 3) == 1))
	{
		kr->SetArg(13, sizeof(int), &r0);
		kr->SetArg(14, sizeof(int), &r1);
	}
	if ((OPTIMIZED_RIGHTPART & 3) == 1)
	{
		lsize = OPT_LS;
		nth = ((nth / lsize) + 1)*lsize;
	}
	if ((OPTIMIZED_RIGHTPART & 3) == 2)
	{
		lsize = OPT_LS;
		nth = s2_rl*(((prevC_size<0 ? 0 : prevC_size) / lsize) + 1);
		kr->SetBufferArg(b_r2, 13);
		kr->SetArg(14, sizeof(int), &s2_rl);
		kr->SetArg(15, sizeof(int), &r0);
		kr->SetArg(16, sizeof(int), &r1);
	}
	queue->ExecuteKernel(kr, 1, &nth, &lsize);
	if ((OPTIMIZED_RIGHTPART & 3) == 2)
	{
		k_sum2->SetBufferArg(br, 0);
		k_sum2->SetBufferArg(bS, 1);
		k_sum2->SetArg(2, sizeof(int), &s2_rl);
		k_sum2->SetBufferArg(b_r2, 3);
		k_sum2->SetArg(4, sizeof(int), &prevC_size);
		k_sum2->SetArg(5, sizeof(int), &N);
		k_sum2->SetArg(6, sizeof(int), &r0);
		k_sum2->SetArg(7, sizeof(int), &r1);
		nth = r1-r0;
		lsize = 1;
		queue->ExecuteKernel(k_sum2, 1, &nth, &lsize);
	}
	if (OPTIMIZED_RIGHTPART & 8)
	{
		for (int i = 0;i < contexts.size();i++)
		{
			// send parts of C to other contexts
			contexts[i].queue->EnqueueWriteBuffer(contexts[i].bCn, s->C+ contexts[i].r0, contexts[i].r0*sizeof(double), (contexts[i].r1 - contexts[i].r0)*sizeof(double));
			// saveC to prevC
			contexts[i].k_saveC->SetBufferArg(contexts[i].bCn, 0);
			contexts[i].k_saveC->SetArg(1, sizeof(int), &N);
			contexts[i].k_saveC->SetBufferArg(contexts[i].bprevC, 2);
			contexts[i].k_saveC->SetArg(3, sizeof(int), &prevC_size);
			contexts[i].k_saveC->SetArg(4, sizeof(int), &contexts[i].r0);
			nth = contexts[i].r1 - contexts[i].r0;
			lsize = 1;
			contexts[i].queue->ExecuteKernel(contexts[i].k_saveC, 1, &nth, &lsize);
		}
		// run right part calculations
		for (int i = 0;i < contexts.size();i++)
		{
			// calculate integrals values
			if ((OPTIMIZED_RIGHTPART & 4) == 0)
			{
				contexts[i].k_kb->SetBufferArg(contexts[i].b_kb, 0);
				contexts[i].k_kb->SetArg(1, sizeof(int), &tstep);
				contexts[i].k_kb->SetBufferArg(contexts[i].bS, 2);
				nth = tstep + 1;
				lsize = 1;
				contexts[i].queue->ExecuteKernel(contexts[i].k_kb, 1, &nth, &lsize);
			}
			else
				contexts[i].queue->EnqueueWriteBuffer(contexts[i].b_kb, cpu_kb, 0, (tstep + 1)*sizeof(double));
			r0 = contexts[i].r0;
			r1 = contexts[i].r1;
			nth = r1 - r0;
			contexts[i].kCr->SetBufferArg(contexts[i].bA, 0);
			contexts[i].kCr->SetBufferArg(contexts[i].bB, 1);
			contexts[i].kCr->SetBufferArg(contexts[i].bC, 2);
			contexts[i].kCr->SetBufferArg(contexts[i].br, 3);
			contexts[i].kCr->SetBufferArg(contexts[i].bU, 4);
			contexts[i].kCr->SetBufferArg(contexts[i].bV, 5);
			contexts[i].kCr->SetBufferArg(contexts[i].bCn, 6);
			contexts[i].kCr->SetBufferArg(contexts[i].bS, 7);
			contexts[i].kCr->SetArg(8, sizeof(int), &N);
			contexts[i].kCr->SetArg(10, sizeof(int), &prevC_size);
			contexts[i].kCr->SetBufferArg(contexts[i].bprevC, 9);
			contexts[i].kCr->SetArg(11, sizeof(int), &tstep);
			contexts[i].kCr->SetBufferArg(contexts[i].b_kb, 12);
			if (((OPTIMIZED_RIGHTPART & 3) == 0) || ((OPTIMIZED_RIGHTPART & 3) == 1))
			{
				contexts[i].kCr->SetArg(13, sizeof(int), &r0);
				contexts[i].kCr->SetArg(14, sizeof(int), &r1);
			}
			if ((OPTIMIZED_RIGHTPART & 3) == 1)
			{
				lsize = OPT_LS;
				nth = ((nth / lsize) + 1)*lsize;
			}
			if ((OPTIMIZED_RIGHTPART & 3) == 2)
			{
				lsize = OPT_LS;
				nth = contexts[i].s2_rl*(((prevC_size < 0 ? 0 : prevC_size) / lsize) + 1);
				contexts[i].kCr->SetBufferArg(contexts[i].b_r2, 13);
				contexts[i].kCr->SetArg(14, sizeof(int), &contexts[i].s2_rl);
				contexts[i].kCr->SetArg(15, sizeof(int), &r0);
				contexts[i].kCr->SetArg(16, sizeof(int), &r1);
			}
			contexts[i].queue->ExecuteKernel(contexts[i].kCr, 1, &nth, &lsize);
			if ((OPTIMIZED_RIGHTPART & 3) == 2)
			{
				contexts[i].k_sum2->SetBufferArg(contexts[i].br, 0);
				contexts[i].k_sum2->SetBufferArg(contexts[i].bS, 1);
				contexts[i].k_sum2->SetArg(2, sizeof(int), &s2_rl);
				contexts[i].k_sum2->SetBufferArg(contexts[i].b_r2, 3);
				contexts[i].k_sum2->SetArg(4, sizeof(int), &prevC_size);
				contexts[i].k_sum2->SetArg(5, sizeof(int), &N);
				contexts[i].k_sum2->SetArg(6, sizeof(int), &r0);
				contexts[i].k_sum2->SetArg(7, sizeof(int), &r1);
				nth = r1 - r0;
				lsize = 1;
				contexts[i].queue->ExecuteKernel(contexts[i].k_sum2, 1, &nth, &lsize);
			}
		}
	}
	t3 = GetTickCount();
	if (OPTIMIZED_RIGHTPART & 4)
	{
		for (int i = 0;i < tstep + 2;i++)
			cpu_kb[i] = s->kb(s->alpha, i, i + 1, tstep + 2, s->tau);
	}
	if (OPTIMIZED_RIGHTPART & 8)
	{
		// get right part parts from contexts and write them to main context memory
		for (int i = 0;i < contexts.size();i++)
		    contexts[i].queue->EnqueueBuffer(contexts[i].br, s->C+ contexts[i].r0, contexts[i].r0*sizeof(double), (contexts[i].r1 - contexts[i].r0)*sizeof(double));
		for (int i = 0;i < contexts.size();i++)
		    contexts[i].queue->Finish();
		for (int i = 0;i < contexts.size();i++)
    		    queue->EnqueueWriteBuffer(br, s->C + contexts[i].r0, contexts[i].r0*sizeof(double), (contexts[i].r1 - contexts[i].r0)*sizeof(double));
		queue->Finish();
	}
	else
	    queue->Finish();
	t4 = GetTickCount();
	// solve tridiagonal by NVIDIA version of parallel cyclic reducion 
	pcr_solver(bA->buffer, bC->buffer, bB->buffer, br->buffer, bRes->buffer, padded_size, &queue->hCmdQueue, 1, bA2->buffer, bC2->buffer, bB2->buffer, br2->buffer);
	// get C from main context
	if (OPTIMIZED_RIGHTPART & 8)
	    gen_diff_ocl_get_results(s);
	t5 = GetTickCount();
	fprintf(fi, "fillM %lld fill rp %lld solve %lld kb %lld\n", t2 - t1, t3 - t2, t5 - t4, t4 - t3);
}
void gen_diff_solve_ocl(double t, double save_tau, int nocpu,double ftau)
{
	gen_diff_solver s1, s2;
	int nsave = 1;
	int step = 0;
	FILE *fi1, *fi2;
	fi1 = fopen("log.txt", "wt");
	fi2 = fopen("results.txt", "wt");
	if (ftau==0)
	{
	s1.init(0.85, 0);
	s2.init(0.85, 0);
	// set tau
	do
	{
		double e;
		s2.tau = s1.tau / 2;
		s1.init(0.85, s1.tau);
		s2.init(0.85, s2.tau);
		for (double tt = 0;tt < 0.01;tt += s1.tau)
			s1.C1();
		for (double tt = 0;tt < 0.01;tt += s2.tau)
			s2.C1();
		e = 0;
		for (int i = 1;i < N;i++)
			e += (s1.C[i] - s2.C[i])*(s1.C[i] - s2.C[i]);
		if (e > 1e-5)
			s1.tau /= 2;
		else
			break;
		printf("%g %g\n", e, s1.tau);
		if (s1.tau<0.00001)
			break;
	} while (1);
	s1.init(0.85, s1.tau);
	s2.init(0.85, s1.tau);
	}
	else
	{
	s1.init(0.85, ftau);
	s2.init(0.85, ftau);
	}
	gen_diff_ocl_init(&s1);
	int tstep = 0;
	for (double tt = 0;tt < t;tt += s1.tau, tstep++)
	{
		double err1 = 0.0;
		long long i1, i2, i11;
		// GPU
		i1 = GetTickCount();
		gen_diff_ocl_calc_tri(kCA, kCB, kCC, kCr, bCn, fi1, tstep, &s1);
		i1 = GetTickCount() - i1;
		i11 = GetTickCount();
		gen_diff_ocl_get_results(&s1);
		i11 = GetTickCount() - i11;
		// CPU
		i2 = GetTickCount();
		if (nocpu==0)s2.C1();
		i2 = GetTickCount() - i2;
		// calculate errors
		for (int i = 0;i < N + 1;i++)
			err1 += (s2.C[i] - s1.C[i])*(s2.C[i] - s1.C[i]);
		fprintf(fi1, "t %g ocl %lld(%lld) cpu %lld e(gpu-cpu) %g\n", tt, i1, i11, i2, err1);
		printf("t %g ocl %lld(%lld) cpu %lld e(gpu-cpu) %g\n", tt, i1, i11, i2, err1);
		fflush(stdout);
		// save result
		if (tt > nsave*save_tau)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g %g %g\n", tt, (double)i / N, s1.C[i], s2.C[i]);
			nsave++;
		}
	}
}
/// main/////////////////
int main(int argc, char **argv)
{
	double Tm = 81.0;
	double Sm = 10.0;
	double ftau=0;
	int nocpu=0;
	int eq = 0;
	int ocl = 0;
	int first_dev = 1;
	if (argc == 1)
	{
		printf("cons : 0 - space_fract, 1 - cons_fract, 2 - generalized Caputo diffusion\n");
		printf("ocl : 0 - no OpenCL, 1 - use OpenCL\n");
		printf("Tm : end time\n");
		printf("Sm : save time\n");
		printf("BS : gird block size\n");
		printf("NB : number of grid blocks\n");
		printf("Br_DIV : number of elements per OpenCL thread\n");
		printf("OPT_LS : thread group size\n");
		printf("OPT : for space_fract and generalized Caputo (for cons==2: bit 1,2 - 0: no local, 1: local, 2: enhanced scalability, bit 3 - kb on cpu, bit 4 - multicontext\n");
		printf("D : OpenCL extention for doubles - 0 - cl_amd_fp64, 1 - cl_khr_fp64\n");
		printf("dev : index of OpenCL device (additional context on device for multicontext)\n");
	}
	for (int i = 1;i<argc;i += 2)
	{
		if (strcmp(argv[i], "Tm") == 0)
			Tm = atof(argv[i + 1]);
		if (strcmp(argv[i], "Sm") == 0)
			Sm = atof(argv[i + 1]);
		if (strcmp(argv[i], "BS") == 0)
			BS = atoi(argv[i + 1]);
		if (strcmp(argv[i], "NB") == 0)
			NB = atoi(argv[i + 1]);
		if (strcmp(argv[i], "Br_DIV") == 0)
			Br_DIV = atoi(argv[i + 1]);
		if (strcmp(argv[i], "OPT_LS") == 0)
			OPT_LS = atoi(argv[i + 1]);
		if (strcmp(argv[i], "OPT") == 0)
			OPTIMIZED_RIGHTPART = atoi(argv[i + 1]);
		if (strcmp(argv[i], "D") == 0)
			double_ext = atoi(argv[i + 1]);
		if (strcmp(argv[i], "nocpu") == 0)
			nocpu = atoi(argv[i + 1]);
		if (strcmp(argv[i], "ftau") == 0)
			ftau = atof(argv[i + 1]);
		if (strcmp(argv[i], "dev") == 0)
		{
			if (first_dev)
			{
				device = atoi(argv[i + 1]);
				first_dev = 0;
			}
			else
			if (eq==2)
				if (OPTIMIZED_RIGHTPART & 8)
				{
					int d= atoi(argv[i + 1]);
					int found = 0;
					for (int i = 0;i < devices.size();i++)
						if (devices[i] == d)
						{
							n_contexts[i]++;
							found = 1;
							break;
						}
					if (found == 0)
					{
						devices.push_back(d);
						n_contexts.push_back(1);
					}
				}
		}
		if (strcmp(argv[i], "cons") == 0)
			eq = atoi(argv[i + 1]);
		if (strcmp(argv[i], "ocl") == 0)
			ocl = atoi(argv[i + 1]);
	}
	N = (BS*NB);
	printf("N %d (%d*%d) ocl:(br_div %d, opt %d, opt_ls %d, D %d,device %d) tend %g tsave %g eq %d ocl %d\n", N, BS, NB, Br_DIV, OPTIMIZED_RIGHTPART, OPT_LS, double_ext, device, Tm, Sm, eq, ocl);fflush(stdout);
	if (eq == 0)
	{
		F = new double[N + 2];
		BVA = new double[BS];
		BVK = new double[BS];
		if (ocl == 0)
			solve(Tm, Sm);
		else
			solve_ocl(Tm, Sm);
		delete[] F;
		delete[] BVA;
		delete[] BVK;
	}
	if (eq == 1)
	{
		if (ocl == 0)
			cons_solve(Tm, Sm);
		else
			solve_cons_ocl(Tm, Sm);
	}
	if (eq == 2)
	{
		if (ocl == 0)
			gen_diff_solve(Tm, Sm);
		else
			gen_diff_solve_ocl(Tm, Sm,nocpu,ftau);
	}
	return 0;
}
