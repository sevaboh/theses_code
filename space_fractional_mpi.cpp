//#define T_DEBUG
//#define unix
#define OCL
#ifdef OCL
#define NEED_OPENCL
#endif
#define USE_MPI
#define OMPI_IMPORTS
#include <stdio.h>
#include <vector>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#ifdef unix
#include <dlfcn.h>
#include <sys/times.h>
#include <unistd.h>
#include <dirent.h>
unsigned int GetTickCount()
{
   struct tms t;
   long long time=times(&t);
   int clk_tck=sysconf(_SC_CLK_TCK);
   return (unsigned int)(((long long)(time*(1000.0/clk_tck)))%0xFFFFFFFF);    
}
#else
#include <windows.h>
#include "../sarpok3d/include/sarpok3d.h"
#endif
#define M_PI 3.1415926535897932384
#define abs(x) (((x)>0.0)?(x):(-(x)))
#define AK 10 // number of terms in approximation series
#define MAXAK 20 // maximal number of terms in approximation series
#define BS 120 // block size on x
#define BS2 120 // block size on y
#define NB 1 // number of block
#define N (BS*NB) // number of rows
#define M (BS2*NB) // number of columns (M must be < N)
double A = 0.85;
double approx_eps = 1e-7;
double eps = 1e-09;
extern double Gamma(double);
int rank;
void system_debug_out(const char *str, int i, int type, const char *file, int line)
{
	printf("%s %d %s %d\n", file, line, str, i);
}
void save_and_convert(char *name,int id,double a,double b,double k,double (*d)[M+2]);
/////////////////////////////////////////
/// opencl parameters ///////////
///////////////////////////////////
int device = 0;
int double_ext = 0; // 0- amd, 1 - khr
////////////////////////////////////////////////////////
// approximations of g(x,r)=(x-r+1)^a-(x-r)^a
////////////////////////////////////////////////////////
// ia approximation gives a_AK<approx_eps for i>ia_mini(r)
// if Ts!=NULL - varying time step - r1=-Ts[r+a],r2=-Ts[r+b]
double ia_mini(double r, int nterms, double a, double b,double *Ts)
{
	double ak = 1;
	int an = -ceil(fabs(a))*a / fabs(a);
	int bn = -ceil(fabs(b))*b / fabs(b);
	if (a == 0.0) an = 0;
	if (b == 0.0) bn = 0;
	for (int i = 1;i<nterms;i++)
		ak *= (A - i + 1) / i;
	if (Ts)
		return pow(abs(ak*(pow(-(Ts[(int)r] + fabs(a)*(Ts[(int)r + an] - Ts[(int)r])), (double)nterms) - pow(-(Ts[(int)r] + fabs(b)*(Ts[(int)r + bn] - Ts[(int)r])), (double)nterms)) / approx_eps), 1.0 / (nterms - A));
	return pow(abs(ak*(pow(a - r, (double)nterms) - pow(b - r, (double)nterms)) / approx_eps), 1.0 / (nterms - A));
}
class ia_approximation
{
public:
	double a,b;
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
	ia_approximation() { a = 1.0, b = 0.0; };
	ia_approximation(double rmin, double rmax, double *F, double *Ts,double cmin = -1, int nt = AK, double _a = 1.0, double _b = 0.0) :a(_a), b(_b)
		// approximate S=sum_r=rmin..rmax(g(x,r)*F[r]):
		// approximate g(x,r) by sum_k=0...inf(a*(a-1)*...*(a-k+1)/k!*x^(a-k)*((1-r)^k-(-r)^k))=sum_k=0...inf(K(r)_k*x^(a-k))
		// then S is approximated by sum_k=0...inf(sum_r=rmin..rmax(K(r)_k*F[r])*x^(a-k))
		// Ts!=NULL - varying time step
	{
		int an = -ceil(fabs(a))*a / fabs(a);
		int bn = -ceil(fabs(b))*b / fabs(b);
		if (a == 0.0) an = 0;
		if (b == 0.0) bn = 0;
		double ak = 1, krk;
		c[0] = 0;
		for (int i = 1;i<nt;i++)
		{
			ak *= (A - i + 1) / i;
			krk = 0.0;
			if (Ts==NULL)
				for (int j = rmin;j<rmax;j++)
					krk += F[j] * (pow(a - (double)j, (double)i) - pow(b - (double)j, (double)i));
			else
				for (int j = rmin;j<rmax;j++)
					krk += F[j] * (pow(-(Ts[j] + fabs(a)*(Ts[j + an] - Ts[j])), (double)i) - pow(-(Ts[j] + fabs(b)*(Ts[j + bn] - Ts[j])), (double)i));
			c[i] = ak*krk;
		}
		for (int i = nt;i < MAXAK;i++)
			c[i] = 0.0;
		if (cmin != -1)
			min = cmin;
		else
		{
			if (Ts==NULL)
				min = ceil(ia_mini(rmax, nt, a, b, Ts));
			else
				min = ia_mini(rmax, nt, a, b, Ts);
		}
		nterms = nt;
	}
	static int size()
	{
	    return ((MAXAK+1)*sizeof(double)+sizeof(int));
	}
	char *serialize()
	{
	    char *ret=new char[((MAXAK+1)*sizeof(double)+sizeof(int))];
	    double *d=(double *)(ret+sizeof(int));
	    ((int *)ret)[0]=nterms;
	    d[0]=min;
	    for (int i=0;i<MAXAK;i++)
		d[i+1]=c[i];
	    return ret;
	}
	ia_approximation(char *m,double _a=1.0,double _b=0.0):a(_a),b(_b)
	{
	    double *d=(double *)(m+sizeof(int));
	    nterms=((int *)m)[0];
	    min=d[0];
	    for (int i=0;i<MAXAK;i++)
		c[i]=d[i+1];
	}
};

// ibk approximation gives a_AK<approx_eps for b-ibk_i2(r,b)<i<b+ibk_i2(r,b)
// if Ts!=NULL - varying time step - r1=-Ts[r+a],r2=-Ts[r+b]
double ibk_i2(double r, double b,int nterms,double a,double c,double *Ts,double step=-1.0)
{
	double ak = 1;
	int an = -ceil(fabs(a))*a / fabs(a);
	int cn = -ceil(fabs(c))*c / fabs(c);
	if (a == 0.0) an = 0;
	if (c == 0.0) cn = 0;
	for (int i = 1;i<nterms;i++)
		ak *= (A - i + 1) / i;
	if (fabs(b-((Ts==NULL)?r:Ts[(int)r]))<1e-10)
		return pow(abs(approx_eps / ak), 1.0 / nterms);
	if (Ts)
	{
		double v1=b - (Ts[(int)r] + fabs(a)*(Ts[(int)r + an] - Ts[(int)r]));
		double v2=b - (Ts[(int)r] + fabs(c)*(Ts[(int)r + cn] - Ts[(int)r]));
		if (v1<0.0) v1=0.0; else v1=pow(v1, A - nterms);
		if (v2<0.0) v2=0.0; else v2=pow(v2, A - nterms);	
		if ((v1!=0.0) && (v2!=0.0))
			return pow(abs(approx_eps / (ak*( v1- v2))), 1.0 / nterms);
		else
			return pow(abs(approx_eps / ak), 1.0 / nterms);
	}
	if (step==-1)
		return pow(abs(approx_eps / (ak*(pow(a + b - r, A - nterms) - pow(c + b - r, A - nterms)))), 1.0 / nterms);
	else
		return pow(abs(approx_eps / (ak*(pow(a*step + b - r, A - nterms) - pow(c*step + b - r, A - nterms)))), 1.0 / nterms);
}
class ibk_approximation
{
public:
	double a,_b;
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
	ibk_approximation() { a = 1.0;_b = 0.0; };
	ibk_approximation(double rmin, double rmax, double *F, double *Ts, double ___b, double i2 = -1, int nt = AK, double _a = 1.0, double __b = 0.0) :a(_a), _b(__b)
		// approximate S=sum_r=rmin..rmax(g(x,r)*F[r]):
		// approximate g(x,r) by sum_k=0...inf(a*(a-1)*...*(a-k+1)/k!*(x-b)^k*((1+b-r)^(a-k)-(b-r)^(a-k)))=sum_k=0...inf(K(r)_k*(x-b)^k)
		// then S is approximated by sum_k=0...inf(sum_r=rmin..rmax(K(r)_k*F[r])*(x-b)^k)
		// Ts!=NULL - varying time step
	{
		int an = -ceil(fabs(a))*a / fabs(a);
		int bn = -ceil(fabs(_b))*_b / fabs(_b);
		if (a == 0.0) an = 0;
		if (_b == 0.0) bn = 0;
		double ak = 1, krk;
		b = ___b;
		for (int i = 0;i<nt;i++)
		{
			if (i != 0) ak *= (A - i + 1) / i;
			krk = 0.0;
			if (Ts==NULL)
				for (int j = rmin;j<rmax;j++)
					krk += F[j] * (pow(a + b - (double)j, A - (double)i) - pow(_b + b - (double)j, A - (double)i));
			else
				for (int j = rmin;j<rmax;j++)
					krk += F[j] * (pow(b - (Ts[j] + fabs(a)*(Ts[j + an] - Ts[j])), A - (double)i) - pow(b - (Ts[j] + fabs(_b)*(Ts[j + bn] - Ts[j])), A - (double)i));
			c[i] = ak*krk;
		}
		for (int i = nt;i < MAXAK;i++)
			c[i] = 0.0;
		if (i2 != -1)
			min = i2;
		else
		{
			if (Ts==NULL)
				min = b - floor(ibk_i2(rmax, b, nt, a, _b, Ts));
			else
				min = b - ibk_i2(rmax, b, nt, a, _b, Ts);
		}
		max = b;
		nterms = nt;
		old = 0;
	}
	ibk_approximation(ibk_approximation *i,double _a=1.0,double __b=0.0):a(_a),_b(__b)
	{
		min = i->min;
		max = i->max;
		b = i->b;
		nterms = i->nterms;
		for (int j = 0;j < MAXAK;j++)
			c[j] = i->c[j];
		old = i->old;
	}
	static int size() 
	{
	    return ((MAXAK+3)*sizeof(double)+2*sizeof(int));
	}
	char *serialize()
	{
	    char *ret=new char[((MAXAK+3)*sizeof(double)+2*sizeof(int))];
	    double *d=(double *)(ret+2*sizeof(int));
	    ((int *)ret)[0]=nterms;
	    ((int *)ret)[1]=old;
	    d[0]=min;
	    d[1]=max;
	    d[2]=b;
	    for (int i=0;i<MAXAK;i++)
		d[i+3]=c[i];
	    return ret;
	}
	ibk_approximation(char *m,double _a=1.0,double __b=0.0):a(_a),_b(__b)
	{
	    double *d=(double *)(m+2*sizeof(int));
	    nterms=((int *)m)[0];
	    old=((int *)m)[1];
	    min=d[0];
	    max=d[1];
	    b=d[2];
	    for (int i=0;i<MAXAK;i++)
		c[i]=d[i+3];
	}
};
// calculate values of F for block
void calc_v(int bs,int block,double *F,double *V,double a=1.0,double b=0.0)
{
	for (int i = 0;i<bs;i++)
		V[i] = 0.0;
	if (A == 0.0) return;
	for (int i=block*bs+1;i<(block+1)*bs+1;i++)
	{
		double v=0.0;
		for (int j=1;j<=i-1;j++)
			v+=(pow((double)i-(double)j+a,A)-pow((double)i-(double)j+b,A))*F[j];
		V[i-(block*bs+1)]=v;
	}
}
// calculate values with variable time steps of F for block
void calc_v_vt(int bs,int block,double *F,double *V,double *Ts,double a=1.0,double b=0.0)
{
	int an=-ceil(fabs(a))*a/fabs(a);
	int bn=-ceil(fabs(b))*b/fabs(b);
	if (a == 0.0) an = 0;
	if (b == 0.0) bn = 0;
	for (int i = 0;i<bs;i++)
		V[i] = 0.0;
	if (A == 0.0) return;
	for (int i=block*bs+1;i<(block+1)*bs+1;i++)
	{
		double v=0.0;
		for (int j=1;j<=i-1;j++)
			v+=(pow(-(Ts[j]+fabs(a)*(Ts[j+an]-Ts[j]))+Ts[i],A)-pow(-(Ts[j]+fabs(b)*(Ts[j+bn]-Ts[j]))+Ts[i],A))*F[j];
		V[i-(block*bs+1)]=v;
	}
}
// calculate approximated values of F for block
// approximations for previous blocks must be built so function must be called in loop from 0 to NB-1
// ias - "long" approximations
// ibks -  "short" approximations
// if Ts!=0 - varying time step
void calc_va(int bs, int block, double *F, double *V, double *Ts, std::vector<ia_approximation> &ias, std::vector<ibk_approximation> &ibks, double _a = 1.0, double _b = 0.0)
{
	ibk_approximation *bibk;
	ia_approximation *bia;
	int an = -ceil(fabs(_a))*_a / fabs(_a);
	int bn = -ceil(fabs(_b))*_b / fabs(_b);
	if (_a == 0.0) an = 0;
	if (_b == 0.0) bn = 0;
	int nadd = 0, nsplit = 0, ndel = 0, nnew = 0;
	long long t1, t2;
	double bm1 = -1;
	double rmax = ((block + 1)*bs + 1);
	int rmaxidx = rmax;
	if (Ts) rmax = Ts[(int)rmax];
	t1 = GetTickCount();
	for (int i = 0;i<bs;i++)
		V[i] = 0.0;
	if (A == 0.0) return;
	// calculate F for r=1,...,block*bs+1 using current approximation
	for (int i = block*bs + 1;i<(block + 1)*bs + 1;i++)
	{
		double ii = i;
		if (Ts) ii = Ts[i];
		for (int j = 0;j < ias.size();j++)
			if (ii >= ias[j].min)
				V[i - (block*bs + 1)] += ias[j].calc(ii);
		for (int j = 0;j < ibks.size();j++)
			if ((ii >= ibks[j].min) && (ii <= ibks[j].max))
				V[i - (block*bs + 1)] += ibks[j].calc(ii);
	}
	// add F for r=block*bs+1,...,(block+1)*bs
	for (int i = block*bs + 1;i<(block + 1)*bs + 1;i++)
	{
		double v = 0.0;
		if (Ts==NULL)
			for (int j = block*bs + 1;j <= i - 1;j++)
				v += (pow((double)i - (double)j + _a, A) - pow((double)i - (double)j + _b, A))*F[j];
		else
			for (int j = block*bs + 1;j <= i - 1;j++)
				v += (pow(-(Ts[j] + fabs(_a)*(Ts[j + an] - Ts[j])) + Ts[i], A) - pow(-(Ts[j] + fabs(_b)*(Ts[j + bn] - Ts[j])) + Ts[i], A))*F[j];
		V[i - (block*bs + 1)] += v;
	}
	t1 = GetTickCount() - t1;
	t2 = GetTickCount();
	// 1) remove ibks with max<rmax
	for (int j = 0;j < ibks.size();j++)
		if (ibks[j].max < rmax)
		{
			ibks.erase(ibks.begin() + j);
			j--;
			ndel++;
		}
	// 2) append current block approximation to ibks with min>rmax
	int ss = ibks.size();
	for (int j = 0;j < ss;j++)
		if (ibks[j].min >= rmax)
			if (ibks[j].old == 0)
			{
				ibk_approximation *i2;
				int spl = 0;
				bibk = new ibk_approximation(block*bs + 1, (block + 1)*bs + 1, F,Ts, ibks[j].b, -1.0, AK, _a, _b);
				if (bibk->min > ibks[j].min)
				{
					int a;
					// increase number of terms 
					for (a = AK;a < MAXAK;a++)
						if ((ibks[j].b - ibk_i2(rmaxidx, ibks[j].b, a, _a, _b,Ts)) < ibks[j].min)
							break;
					if (a == MAXAK)
						spl = 1;
					else
					{
						delete bibk;
						bibk = new ibk_approximation(block*bs + 1, (block + 1)*bs + 1, F,Ts, ibks[j].b, -1.0, a, _a, _b);
					}
					if (spl)
						i2 = new ibk_approximation(&ibks[j], _a, _b);
				}
				for (int i = 0;i<bibk->nterms;i++)
					ibks[j].c[i] += bibk->c[i];
				if (ibks[j].nterms < bibk->nterms)
					ibks[j].nterms = bibk->nterms;
				nadd++;
				// split ibks into two parts if current block ibk min is bigger that ibks min
				if (spl)
				{
					if (bibk->min > ibks[j].min)
					{
						if (Ts == NULL) // integer i and r
						{
							i2->max = ceil(bibk->min);
							ibks[j].min = i2->max + 1;
						}
						else // float i and r
						{
							i2->max = bibk->min;
							ibks[j].min = i2->max + 1e-15;
						}
						i2->old = 1;
						if (ibks[j].min > ibks[j].max)
						{
							i2->max = ibks[j].max;
							ibks.erase(ibks.begin() + j);
							j--;
							ss--;
							ndel++;
						}
						ibks.push_back(i2[0]);
						delete bibk;
						// create serie of approximations for i2->min,i2->max
						double rr = i2->min;
						double bb0 = rr, bb = rr;
						do
						{
							// find bj>rj:bj-i2(r,bj,k)/2=rj+1
							bb = rr + ((Ts==NULL)?1: (Ts[rmaxidx + 1] - Ts[rmaxidx]));
							do
							{
								bb0 = bb;
								bb = rr + ibk_i2(i2->min, bb, AK, _a, _b,NULL, ((Ts == NULL) ? -1.0 : (Ts[rmaxidx + 1] - Ts[rmaxidx]))) + ((Ts == NULL) ? 1 : 1e-15);
							} while ((bb - bb0) > eps);
							// build ibk approximation
							bibk = new ibk_approximation(block*bs + 1, (block + 1)*bs + 1, F,Ts, ((Ts==NULL)?floor(bb):bb), rr, AK, _a, _b);
							// move r to bj+i2(r,bj,k)
							if (Ts == NULL)
							{
								rr = floor(bb) + 1;
								if (floor(bb) > i2->max)
									bibk->max = i2->max;
							}
							else
							{
								rr = bb + 1e-15;
								if (bb > i2->max)
									bibk->max = i2->max;
							}
							ibks.push_back(bibk[0]);
							nnew++;
							delete bibk;
						} while (((Ts==NULL)?floor(bb):bb) < i2->max);
						nsplit++;
					}
					delete i2;
				}
				else
					delete bibk;
			}
	// 3) build block j approximations for ibks with min<rmax and max>rmax and for splitted ibks with min>rmax
	for (int j = 0;j < ibks.size();j++)
		if (ibks[j].max >= rmax)
			if (ibks[j].min < rmax)
				if (ibks[j].max>bm1)
					if (ibks[j].old == 0)
					{
						bm1 = ibks[j].max;
						ibks[j].old = 1;
					}
	if (bm1 != -1)
	{
		for (int j = 0;j < ibks.size();j++)
			if (ibks[j].max < bm1)
				ibks[j].old = 1;
		double rr = rmax;
		double bb0 = rr, bb = rr;
		do
		{
			// find bj>rj:bj-i2(r,bj,k)/2=rj+1
			bb = rr + ((Ts == NULL) ? 1 : (Ts[rmaxidx+1]-Ts[rmaxidx]));
			do
			{
				bb0 = bb;
				bb = rr + ibk_i2(rmaxidx, bb, AK, _a, _b,Ts) + ((Ts == NULL) ? 1 : 1e-15);
			} while ((bb - bb0)>eps);
			// build ibk approximation 
			bibk = new ibk_approximation(block*bs + 1, (block + 1)*bs + 1, F,Ts, ((Ts==NULL)?floor(bb):bb), rr, AK, _a, _b);
			// move r to bj+i2(r,bj,k)
			if (Ts == NULL)
			{
				rr = floor(bb) + 1;
				if (floor(bb) > bm1)
					bibk->max = bm1;
			}
			else
			{
				rr = bb + 1e-15;
				if (bb > bm1)
					bibk->max = bm1;
			}
			ibks.push_back(bibk[0]);
			nnew++;
			delete bibk;
		} while (((Ts==NULL)?floor(bb):bb) < bm1);
	}
	// 4) build ia approximation and add it to last ia approximation (with biggest min) if current ia.min< lastia.min
	bia = new ia_approximation(block*bs + 1, (block + 1)*bs + 1, F, Ts,-1, AK, _a, _b);
	if (ias.size())
		if (bia->min < ias[ias.size() - 1].min)
		{
			for (int i = 0;i<bia->nterms;i++)
				ias[ias.size() - 1].c[i] += bia->c[i];
			if (ias[ias.size() - 1].nterms < bia->nterms)
				ias[ias.size() - 1].nterms = bia->nterms;
			nadd++;
			delete bia;
			bia = NULL;
		}
	// 5) build ibk approximation serie if current ia.min>lastia.min, set current ia minimum as ibks maximum and add it to ias
	if (bia)
	{
		bm1 = rmax;
		for (int j = 0;j < ibks.size();j++)
			if ((ibks[j].max + ((Ts == NULL) ? 1 : 1e-15))>bm1)
				bm1 = ibks[j].max + ((Ts==NULL)?1:1e-15);
		double rr = bm1;
		double bb0 = rr, bb = rr;
		do
		{
			// find bj>rj:bj-i2(r,bj,k)=rj+1
			bb = rr + ((Ts == NULL) ? 1 : (Ts[rmaxidx + 1] - Ts[rmaxidx]));
			do
			{
				bb0 = bb;
				bb = rr + ibk_i2(rmaxidx, bb, AK, _a, _b,Ts) + ((Ts==NULL)?1:1e-15);
			} while ((bb - bb0)>eps);
			// build ibk approximation 
			bibk = new ibk_approximation(block*bs + 1, (block + 1)*bs + 1, F,Ts, ((Ts==NULL)?floor(bb):bb), rr, AK, _a, _b);
			ibks.push_back(bibk[0]);
			delete bibk;
			nnew++;
			// move r to bj+i2(r,bj,k)
			if (Ts == NULL)
				rr = floor(bb) + 1;
			else
				rr = bb + 1e-15;
		} while (((Ts==NULL)?floor(bb):bb) < ((Ts==NULL)?ceil(bia->min):bia->min));
		bia->min = rr;
		ias.push_back(bia[0]);
		delete bia;
		nnew++;
	}
	t2 = GetTickCount() - t2;
}
///////////////////////////////////////////
// space-fractional solver 2d - 
// dh/dt=d_(1+alpha)h/x+d_(1+alpha)h/y
// v/(x,y)=-kd_alpha_h/(x,y)
// dc/dt=d_(1+beta)c/x+d_(1+beta)c/y-vx*c-vy*c
///////////////////////////////////////////
char *fr2_opencl_text = "\n\
#pragma OPENCL EXTENSION %s : enable \n\
#define delta (*((__global double *)(class+%d)))\n\
#define s1 (*((__global double *)(class+%d)))\n\
#define h1 (*((__global double *)(class+%d)))\n\
#define delta2 (*((__global double *)(class+%d)))\n\
#define delta_m (*((__global double *)(class+%d)))\n\
#define s1_m (*((__global double *)(class+%d)))\n\
#define h2 (*((__global double *)(class+%d)))\n\
#define delta2_m (*((__global double *)(class+%d)))\n\
#define tau (*((__global double *)(class+%d)))\n\
#define w (*((__global double *)(class+%d)))\n\
#define alpha (*((__global double *)(class+%d)))\n\
#define beta (*((__global double *)(class+%d)))\n\
#define d (*((__global double *)(class+%d)))\n\
#define H0 (*((__global double *)(class+%d)))\n\
#define sigma (*((__global double *)(class+%d)))\n\
#define IA_SIZE %d\n\
#define ia_min(n,k) (*((__global double *)(ias+(alloced_approx_per_row*(k)+(n))*(IA_SIZE)+%d)))\n\
#define ia_nterms(n,k) (*((__global int *)(ias+(alloced_approx_per_row*(k)+(n))*(IA_SIZE)+%d)))\n\
#define ia_c(n,k) ((__global double *)(ias+(alloced_approx_per_row*(k)+(n))*(IA_SIZE)+%d))\n\
#define IBK_SIZE %d\n\
#define ibk_min(n,k) (*((__global double *)(ibks+(alloced_approx_per_row*(k)+(n))*(IBK_SIZE)+%d)))\n\
#define ibk_max(n,k) (*((__global double *)(ibks+(alloced_approx_per_row*(k)+(n))*(IBK_SIZE)+%d)))\n\
#define ibk_b(n,k) (*((__global double *)(ibks+(alloced_approx_per_row*(k)+(n))*(IBK_SIZE)+%d)))\n\
#define ibk_nterms(n,k) (*((__global int *)(ibks+(alloced_approx_per_row*(k)+(n))*(IBK_SIZE)+%d)))\n\
#define ibk_old(n,k) (*((__global int *)(ibks+(alloced_approx_per_row*(k)+(n))*(IBK_SIZE)+%d)))\n\
#define ibk_c(n,k) ((__global double *)(ibks+(alloced_approx_per_row*(k)+(n))*(IBK_SIZE)+%d))\n\
\n\
#define approx_eps (%g)\n\
#define AK (%d)\n\
#define MAXAK (%d)\n\
#define BS (%d)\n\
#define BS2 (%d)\n\
#define eps (%g)\n\
#define G2a (%g)\n\
#define abs(x) (((x)>0)?(x):(-(x)))\n\
double ia_mini(double r,int nterms,double A)\n\
{\n\
	double ak = 1;\n\
	for (int i = 1;i<nterms;i++)\n\
		ak *= (A - i + 1) / i;\n\
	return pow(abs(ak*(pow(1 - r, (double)nterms) - pow(-r, (double)nterms)) / approx_eps), 1.0 / (nterms - A));\n\
}\n\
double ia_calc(double x,__global char *ias,int n,int k,int alloced_approx_per_row,double A)\n\
{\n\
	double ret = 0.0;\n\
	for (int i = 0;i<ia_nterms(n,k);i++)\n\
		ret += ia_c(n,k)[i] * pow(x, A - (double)i);\n\
	return ret;\n\
}\n\
void new_ia_approximation(double rmin, double rmax,int k,int alloced_approx_per_row, __global double *F,__global char *ias,int n,double A,int N, double cmin, int nt)\n\
{\n\
	double ak = 1, krk;\n\
	ia_c(n,k)[0] = 0;\n\
	for (int i = 1;i<nt;i++)\n\
	{\n\
		ak *= (A - i + 1) / i;\n\
		krk = 0.0;\n\
		for (int j = rmin;j<rmax;j++)\n\
			krk += F[j*(N+2)+k] * (pow(1.0 - (double)j, (double)i) - pow(-(double)j, (double)i));\n\
		ia_c(n,k)[i] = ak*krk;\n\
	}\n\
	for (int i = nt;i < MAXAK;i++)\n\
		ia_c(n,k)[i] = 0.0;\n\
	if (cmin != -1)\n\
		ia_min(n,k) = cmin;\n\
	else\n\
		ia_min(n,k) = ceil(ia_mini(rmax, nt,A));\n\
	ia_nterms(n,k) = nt;\n\
}\n\
void delete_ia(int n,int k,int alloced_approx_per_row,__global char *ias,__global int *Nias)\n\
{\n\
	for (int i=(alloced_approx_per_row*k+n+1)*IA_SIZE;i<(alloced_approx_per_row*k+Nias[k])*IA_SIZE;i++)\n\
		ias[i-IA_SIZE]=ias[i];\n\
	Nias[k]--;\n\
}\n\
void delete_ibk(int n,int k,int alloced_approx_per_row,__global char *ibks,__global int *Nibks)\n\
{\n\
	for (int i=(alloced_approx_per_row*k+n+1)*IBK_SIZE;i<(alloced_approx_per_row*k+Nibks[k])*IBK_SIZE;i++)\n\
		ibks[i-IBK_SIZE]=ibks[i];\n\
	Nibks[k]--;\n\
}\n\
double ibk_i2(double r, double b, int nterms,double A)\n\
{\n\
	double ak = 1;\n\
	for (int i = 1;i<nterms;i++)\n\
		ak *= (A - i + 1) / i;\n\
	if (b == r)\n\
		return pow(abs(approx_eps / ak), 1.0 / nterms);\n\
	return pow(abs(approx_eps / (ak*(pow(1 + b - r, A - nterms) - pow(b - r, A - nterms)))), 1.0 / nterms);\n\
}\n\
double ibk_calc(double x,__global char *ibks,int n,int k,int alloced_approx_per_row)\n\
{\n\
	double ret = 0.0, xbk = 1;\n\
	for (int i = 0;i<ibk_nterms(n,k);i++)\n\
	{\n\
		ret += ibk_c(n,k)[i] * xbk;\n\
		xbk *= (x - ibk_b(n,k));\n\
	}\n\
	return ret;\n\
}\n\
void new_ibk_approximation(double rmin, double rmax, int k,int alloced_approx_per_row,__global double *F, double _b,__global char *ibks,int n,double A,int N, double i2, int nt)\n\
{\n\
	double ak = 1, krk;\n\
	ibk_b(n,k) = _b;\n\
	for (int i = 0;i<nt;i++)\n\
	{\n\
		if (i != 0) ak *= (A - i + 1) / i;\n\
		krk = 0.0;\n\
		for (int j = rmin;j<rmax;j++)\n\
			krk += F[j*(N+2)+k] * (pow(1.0 + ibk_b(n,k) - (double)j, A - (double)i) - pow(ibk_b(n,k) - (double)j, A - (double)i));\n\
		ibk_c(n,k)[i] = ak*krk;\n\
	}\n\
	for (int i = nt;i < MAXAK;i++)\n\
		ibk_c(n,k)[i] = 0.0;\n\
	if (i2 != -1)\n\
		ibk_min(n,k) = i2;\n\
	else\n\
		ibk_min(n,k) = ibk_b(n,k) - floor(ibk_i2(rmax, ibk_b(n,k), nt,A));\n\
	ibk_max(n,k) = ibk_b(n,k);\n\
	ibk_nterms(n,k) = nt;\n\
	ibk_old(n,k) = 0;\n\
}\n\
void copy_ibk_approximation(__global char *ibks,int n,int i,int k,int alloced_approx_per_row)\n\
{\n\
	ibk_min(n,k) = ibk_min(i,k);\n\
	ibk_max(n,k) = ibk_max(i,k);\n\
	ibk_b(n,k) = ibk_b(i,k);\n\
	ibk_nterms(n,k) = ibk_nterms(i,k);\n\
	for (int j = 0;j < MAXAK;j++)\n\
		ibk_c(n,k)[j] = ibk_c(i,k)[j];\n\
	ibk_old(n,k) = ibk_old(i,k);\n\
}\n\
void calc_va(int bs,int block,int k,int alloced_approx_per_row, __global double *F, __global double *BVK,__global int *Nibks,__global int *Nias,__global char *ias,__global char *ibks,double A,int N)\n\
{\n\
	double bm1 = -1;\n\
	int bibk;\n\
	int rmax = ((block + 1)*bs + 1);\n\
	for (int i = 0;i<bs;i++)\n\
		BVK[i*(N+2)+k] = 0.0;\n\
	if (A == 1.0) return;\n\
	// calculate F for r=1,...,block*BS+1 using current approximation\n\
	for (int i = block*bs + 1;i<(block + 1)*bs + 1;i++)\n\
	{\n\
		for (int j = 0;j<Nias[k];j++)\n\
			if (i >= ia_min(j,k))\n\
				BVK[(i - (block*bs + 1))*(N+2)+k] += ia_calc(i,ias,j,k,alloced_approx_per_row,A);\n\
		for (int j = 0;j<Nibks[k];j++)\n\
			if ((i >= ibk_min(j,k)) && (i <= ibk_max(j,k)))\n\
				BVK[(i - (block*bs + 1))*(N+2)+k] += ibk_calc(i,ibks,j,k,alloced_approx_per_row);\n\
	}\n\
	// add F for r=block*BS+1,...,(block+1)*BS\n\
	for (int i = block*bs + 1;i<(block + 1)*bs + 1;i++)\n\
	{\n\
		double v = 0.0;\n\
		for (int j = block*bs + 1;j <= i - 1;j++)\n\
			v += (pow((double)i - (double)j + 1, A) - pow((double)i - (double)j, A))*F[j*(N+2)+k];\n\
		BVK[(i - (block*bs + 1))*(N+2)+k] += v;\n\
	}\n\
	// 1) remove ibks with max<rmax\n\
	for (int j = 0;j < Nibks[k];j++)\n\
		if (ibk_max(j,k) < rmax)\n\
		{\n\
			delete_ibk(j,k,alloced_approx_per_row,ibks,Nibks);\n\
			j--;\n\
		}\n\
	// 2) append current block approximation to ibks with min>rmax\n\
	int ss = Nibks[k];\n\
	for (int j = 0;j < ss;j++)\n\
		if (ibk_min(j,k) >= rmax)\n\
			if (ibk_old(j,k) == 0)\n\
			{\n\
				int i2=Nibks[k]+1;\n\
				int spl = 0;\n\
				int bibk=Nibks[k];\n\
				new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, ibk_b(j,k),ibks,bibk,A,N,-1,AK);\n\
				if (ibk_min(bibk,k) > ibk_min(j,k))\n\
				{\n\
					int a;\n\
					// increase number of terms \n\
					for (a = AK;a < MAXAK;a++)\n\
						if ((ibk_b(j,k) - ibk_i2(rmax, ibk_b(j,k), a,A)) < ibk_min(j,k))\n\
							break;\n\
					if (a == MAXAK)\n\
						spl = 1;\n\
					else\n\
						new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1, k, alloced_approx_per_row, F, ibk_b(j, k), ibks, bibk, A, N,-1.0,a);\n\
					if (spl)\n\
						copy_ibk_approximation(ibks,i2,j,k,alloced_approx_per_row);\n\
				}\n\
				for (int i = 0;i<ibk_nterms(bibk,k);i++)\n\
					ibk_c(j,k)[i] += ibk_c(bibk,k)[i];\n\
				if (ibk_nterms(j,k) < ibk_nterms(bibk,k))\n\
					ibk_nterms(j,k) = ibk_nterms(bibk,k);\n\
				// split ibks into two parts if current block ibk min is bigger that ibks min\n\
				if (spl)\n\
					if (ibk_min(bibk,k) > ibk_min(j,k))\n\
					{\n\
						ibk_max(i2,k) = ceil(ibk_min(bibk,k));\n\
						ibk_old(i2,k) = 1;\n\
						ibk_min(j,k) = ibk_max(i2,k) + 1;\n\
						copy_ibk_approximation(ibks,Nibks[k],i2,k,alloced_approx_per_row);\n\
						i2=Nibks[k];\n\
						Nibks[k]++;\n\
						if (ibk_min(j,k) > ibk_max(j,k))\n\
						{\n\
							ibk_max(i2,k) = ibk_max(j,k);\n\
							delete_ibk(j,k,alloced_approx_per_row,ibks,Nibks);\n\
							j--;\n\
							ss--;\n\
						}\n\
						// create serie of approximations for i2->min,i2->max\n\
						double rr = ibk_min(i2,k);\n\
						double bb0 = rr, bb = rr;\n\
						do\n\
						{\n\
							// find bj>rj:bj-i2(r,bj,k)/2=rj+1\n\
							bb = rr + 1;\n\
							do\n\
							{\n\
								bb0 = bb;\n\
								bb = rr + ibk_i2(ibk_min(i2,k), bb, AK,A) + 1;\n\
							} while (abs(bb - bb0) > eps);\n\
							// build ibk approximation \n\
							bibk=Nibks[k];\n\
							new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, floor(bb),ibks,bibk,A,N,rr,AK);\n\
							Nibks[k]++;\n\
							// move r to bj+i2(r,bj,k)\n\
							rr = floor(bb) + 1;\n\
							if (floor(bb) > ibk_max(i2,k))\n\
								ibk_max(bibk,k) = ibk_max(i2,k);\n\
						} while (floor(bb) < ibk_max(i2,k));\n\
					}\n\
			}\n\
	// 3) build block j approximations for ibks with min<rmax and max>rmax and for splitted ibks with min>rmax\n\
	for (int j = 0;j < Nibks[k];j++)\n\
		if (ibk_max(j,k) >= rmax)\n\
			if (ibk_min(j,k) < rmax)\n\
				if (ibk_max(j,k)>bm1)\n\
					if (ibk_old(j,k) == 0)\n\
					{\n\
						bm1 = ibk_max(j,k);\n\
						ibk_old(j,k) = 1;\n\
					}\n\
	if (bm1 != -1)\n\
	{\n\
		for (int j = 0;j < Nibks[k];j++)\n\
			if (ibk_max(j,k) < bm1)\n\
				ibk_old(j,k) = 1;\n\
		double rr = rmax;\n\
		double bb0 = rr, bb = rr;\n\
		do\n\
		{\n\
			// find bj>rj:bj-i2(r,bj,k)/2=rj+1\n\
			bb = rr + 1;\n\
			do\n\
			{\n\
				bb0 = bb;\n\
				bb = rr + ibk_i2(rmax, bb, AK,A) + 1;\n\
			} while (abs(bb - bb0)>eps);\n\
			// build ibk approximation \n\
			bibk=Nibks[k];\n\
			new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, floor(bb),ibks,bibk,A,N,rr,AK);\n\
			Nibks[k]++;\n\
			// move r to bj+i2(r,bj,k)\n\
			rr = floor(bb) + 1;\n\
			if (floor(bb) > bm1)\n\
				ibk_max(bibk,k) = bm1;\n\
		} while (floor(bb) < bm1);\n\
	}\n\
	// 4) build ia approximation and add it to last ia approximation (with biggest min) if current ia.min< lastia.min\n\
	int bia=Nias[k];\n\
	new_ia_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F,ias,bia,A,N,-1,AK);\n\
	if (Nias[k])\n\
		if (ia_min(bia,k) < ia_min(Nias[k]-1,k))\n\
		{\n\
			for (int i = 0;i<ia_nterms(bia,k);i++)\n\
				ia_c(Nias[k]-1,k)[i] += ia_c(bia,k)[i];\n\
			if (ia_nterms(Nias[k]- 1,k) < ia_nterms(bia,k))\n\
				ia_nterms(Nias[k]-1,k) = ia_nterms(bia,k);\n\
			bia = -1;\n\
		}\n\
	// 5) build ibk approximation serie if current ia.min>lastia.min, set current ia minimum as ibks maximum and add it to ias\n\
	if (bia!=-1)\n\
	{\n\
		bm1 = rmax;\n\
		for (int j = 0;j < Nibks[k];j++)\n\
			if ((ibk_max(j,k) + 1)>bm1)\n\
				bm1 = ibk_max(j,k) + 1;\n\
		double rr = bm1;\n\
		double bb0 = rr, bb = rr;\n\
		do\n\
		{\n\
			// find bj>rj:bj-i2(r,bj,k)=rj+1\n\
			bb = rr + 1;\n\
			do\n\
			{\n\
				bb0 = bb;\n\
				bb = rr + ibk_i2(rmax, bb, AK,A) + 1;\n\
			} while (abs(bb - bb0)>eps);\n\
			// build ibk approximation \n\
			bibk=Nibks[k];\n\
			new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, floor(bb),ibks,bibk,A,N,rr,AK);\n\
			Nibks[k]++;\n\
			// move r to bj+i2(r,bj,k)\n\
			rr = floor(bb) + 1;\n\
		} while (floor(bb) < ceil(ia_min(bia,k)));\n\
		ia_min(bia,k) = rr;\n\
		Nias[k]++;\n\
	}\n\
}\n";
char *fr2_opencl_text2 = "\n\
__kernel void UAl_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int i=get_global_id(0)+y0;\n\
	Al[1*(N+2)+i] = 0;\n\
	for (int j = x0;j < x1;j++)\n\
		Al[(j + 1)*(N+2)+i] = 1.0 / (s1 - Al[j*(N+2)+i]);\n\
}\n\
__kernel void UBt_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	Bt[1*(N+2)+j] = 1;\n\
	for (int i = x0;i < x1;i++)\n\
		Bt[(i + 1)*(N+2)+j] = Al[(i + 1)*(N+2)+j] * (Bt[i*(N+2)+j] - Om[i*(N+2)+j]);\n\
}\n\
__kernel void UOm_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int b0,int b1,__global int *Nibk,__global int *Nias,__global char *ibks,__global char *ias,__global double *F,__global double *BVK, int alloced_approx_per_row,double A,int M)\n\
{\n\
	int k=get_global_id(0)+y0;\n\
	for (int i = x0;i < x1;i++)\n\
		F[i*(N+2)+k] = U[(i + 1)*(N+2)+k] - 2.0*U[i*(N+2)+k] + U[(i - 1)*(N+2)+k];\n\
	for (int i = b0;i < b1;i++)\n\
	{\n\
		calc_va(BS,i,k,alloced_approx_per_row, F, BVK,Nibk,Nias,ias,ibks,A,N);\n\
		for (int j = 0;j < BS;j++)\n\
			Om[(1 + i*BS + j)*(N+2)+k] = -(h1*h1 / delta)*((U[(1 + i*BS + j)*(N+2)+k] / tau) + delta*BVK[j*(N+2)+k] / (h1*h1));\n\
	}\n\
}\n\
__kernel void U_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int M)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	U[M*(N+2)+j] = H0;\n\
	U[0*(N+2)+j] = 1.0;\n\
	for (int i = x1 - 1;i >= x0;i--)\n\
		U[i*(N+2)+j] = Al[(i + 1)*(N+2)+j] * U[(i + 1)*(N+2)+j] + Bt[(i + 1)*(N+2)+j];\n\
}\n\
__kernel void UAl_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	Al[j*(N+2)+1] = 0;\n\
	for (int i = x0;i < x1;i++)\n\
		Al[j*(N+2)+i + 1] = 1.0 / (s1_m - Al[j*(N+2)+i]);\n\
}\n\
__kernel void UBt_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	Bt[j*(N+2)+1] = 1;\n\
	for (int i = x0;i < x1;i++)\n\
		Bt[j*(N + 2) + i + 1] = Al[j*(N + 2) + i + 1] * (Bt[j*(N + 2) + i] - Om[j*(N + 2) + i]);\n\
}\n\
__kernel void UOm_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int b0,int b1,__global int *Nibk,__global int *Nias,__global char *ibks,__global char *ias,__global double *F,__global double *BVK, int alloced_approx_per_row, double A,int M)\n\
{\n\
	int k=get_global_id(0)+y0;\n\
	for (int i = x0;i < x1;i++)\n\
		F[i*(M+2)+k] = U[k*(N+2)+i + 1] - 2.0*U[k*(N+2)+i] + U[k*(N+2)+i - 1];\n\
	for (int i = b0;i < b1;i++)\n\
	{\n\
		calc_va(BS2,i,k,alloced_approx_per_row, F, BVK,Nibk,Nias,ias,ibks,A,M);\n\
		for (int j = 0;j < BS2;j++)\n\
			Om[k*(N+2)+1 + i*BS2 + j] = -(h2*h2 / delta_m)*((U[k*(N+2)+1 + i*BS2 + j] / tau) + delta_m*BVK[j*(M+2)+k] / (h2*h2));\n\
	}\n\
}\n\
__kernel void U_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	U[j*(N+2)+N] = H0;\n\
	U[j*(N+2)+0] = 1.0;\n\
	for (int i = x1 - 1;i >= x0;i--)\n\
		U[j*(N+2)+i] = Al[j*(N+2)+i + 1] * U[j*(N+2)+i + 1] + Bt[j*(N+2)+i + 1];\n\
}\n\
__kernel void V_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int b0,int b1,__global int *Nibk,__global int *Nias,__global char *ibks,__global char *ias, __global double *F, __global double *BVK, int alloced_approx_per_row, double A,int M)\n\
{\n\
	int k=get_global_id(0)+y0;\n\
	V_h[0*(N+2)+k] = 0;\n\
	V_h[1*(N+2)+k] = -(w / (h1*G2a))*(U[1*(N+2)+k] - U[0*(N+2)+k]);\n\
	for (int i = x0;i < x1;i++)\n\
		F[i*(N+2)+k] = U[i*(N+2)+k] - U[(i - 1)*(N+2)+k];\n\
	for (int i = b0;i < b1;i++)\n\
	{\n\
		calc_va(BS,i,k,alloced_approx_per_row, F, BVK,Nibk,Nias,ias,ibks,A,N);\n\
		for (int j = 0;j < BS;j++)\n\
			if ((i != 0) || (j != 0))\n\
				V_h[(1 + i*BS + j)*(N+2)+k] = -(w / (h1*G2a))*(U[(1 + i*BS + j)*(N+2)+k] - U[(i*BS + j)*(N+2)+k] + BVK[j*(N+2)+k]);\n\
	}\n\
}\n\
__kernel void V_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int b0,int b1,__global int *Nibk,__global int *Nias,__global char *ibks,__global char *ias, __global double *F, __global double *BVK, int alloced_approx_per_row, double A,int M)\n\
{\n\
	int k=get_global_id(0)+y0;\n\
	V_v[k*(N+2)+0] = 0;\n\
	V_v[k*(N+2)+1] = -(w / (h1*G2a))*(U[k*(N+2)+1] - U[k*(N+2)+0]);\n\
	for (int i = x0;i < x1;i++)\n\
		F[i*(M+2)+k] = U[k*(N+2)+i] - U[k*(N+2)+i - 1];\n\
	for (int i = b0;i < b1;i++)\n\
	{\n\
		calc_va(BS2,i,k,alloced_approx_per_row, F, BVK,Nibk,Nias,ias,ibks,A,M);\n\
		for (int j = 0;j < BS2;j++)\n\
			if ((i != 0) || (j != 0))\n\
				V_v[k*(N+2)+1 + i*BS2 + j] = -(w / (h2*G2a))*(U[k*(N+2)+1 + i*BS2 + j] - U[k*(N+2)+i*BS2 + j] + BVK[j*(M+2)+k]);\n\
	}\n\
}\n\
__kernel void CAl_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	Al[1*(N+2)+j] = 0;\n\
	for (int i = x0;i < x1;i++)\n\
		Al[(i + 1)*(N+2)+j] = ((delta2*d / (h1*h1)) - (V_h[i*(N+2)+j] / (2.0*h1))) / (((sigma / tau) + (2.0*delta2*d / (h1*h1)) + ((V_h[(i + 1)*(N+2)+j] - V_h[(i - 1)*(N+2)+j]) / (2.0*h1))) - Al[i*(N+2)+j] * ((delta2*d / (h1*h1)) + (V_h[i*(N+2)+j] / (2.0*h1))));\n\
}\n\
__kernel void CBt_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	Bt[1*(N+2)+j] = 0;\n\
	for (int i = x0;i < x1;i++)\n\
		Bt[(i + 1)*(N + 2) + j] = (Al[(i + 1)*(N + 2) + j] / ((delta2*d / (h1*h1)) - (V_h[i*(N + 2) + j] / (2.0*h1)))) * (((delta2*d / (h1*h1)) + (V_h[i*(N + 2) + j] / (2.0*h1)))*Bt[i*(N + 2) + j] - Om[i*(N + 2) + j]);\n\
}\n\
__kernel void COm_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int b0,int b1,__global int *Nibk,__global int *Nias,__global char *ibks,__global char *ias, __global double *F, __global double *BVK, int alloced_approx_per_row, double A,int M)\n\
{\n\
	int k=get_global_id(0)+y0;\n\
	for (int i = x0;i < x1;i++)\n\
		F[i*(N+2)+k] = C[(i + 1)*(N+2)+k] - 2.0*C[i*(N+2)+k] + C[(i - 1)*(N+2)+k];\n\
	for (int i = b0;i < b1;i++)\n\
	{\n\
		calc_va(BS,i,k,alloced_approx_per_row, F, BVK,Nibk,Nias,ias,ibks,A,N);\n\
		for (int j = 0;j < BS;j++)\n\
			Om[(1 + i*BS + j)*(N+2)+k] = -(sigma / tau)*C[(1 + i*BS + j)*(N+2)+k] - d*delta2*BVK[j*(N+2)+k] / (h1*h1);\n\
	}\n\
}\n\
__kernel void C_h(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int M)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	C[M*(N+2)+j] = 1.0;\n\
	C[0 * (N + 2) + j] = 0.0;\n\
	for (int i = x1 - 1;i >= x0;i--)\n\
		C[i*(N + 2) + j] = Al[(i + 1)*(N + 2) + j] * C[(i + 1)*(N + 2) + j] + Bt[(i + 1)*(N + 2) + j];\n\
}\n\
__kernel void CAl_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	Al[j*(N+2)+1] = 0;\n\
	for (int i = x0;i < x1;i++)\n\
		Al[j*(N + 2) + i + 1] = ((delta2_m*d / (h2*h2)) - (V_v[j*(N + 2) + i] / (2.0*h2))) / (((sigma / tau) + (2.0*delta2_m*d / (h2*h2)) + ((V_v[j*(N + 2) + i + 1] - V_v[j*(N + 2) + i - 1]) / (2.0*h2))) - Al[j*(N + 2) + i] * ((delta2_m*d / (h2*h2)) + (V_v[j*(N + 2) + i] / (2.0*h2))));\n\
}\n\
__kernel void CBt_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	Bt[j*(N+2)+1] = 0;\n\
	for (int i = x0;i < x1;i++)\n\
		Bt[j*(N + 2) + i + 1] = (Al[j*(N + 2) + i + 1] / ((delta2_m*d / (h2*h2)) - (V_v[j*(N + 2) + i] / (2.0*h2)))) * (((delta2_m*d / (h2*h2)) + (V_v[j*(N + 2) + i] / (2.0*h2)))*Bt[j*(N + 2) + i] - Om[j*(N + 2) + i]);\n\
}\n\
__kernel void COm_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1,int b0,int b1,__global int *Nibk,__global int *Nias,__global char *ibks,__global char *ias, __global double *F, __global double *BVK, int alloced_approx_per_row, double A,int M)\n\
{\n\
	int k=get_global_id(0)+y0;\n\
	for (int i = x0;i < x1;i++)\n\
		F[i*(M+2)+k] = C[k*(N+2)+i + 1] - 2.0*C[k*(N+2)+i] + C[k*(N+2)+i - 1];\n\
	for (int i = b0;i < b1;i++)\n\
	{\n\
		calc_va(BS2,i,k,alloced_approx_per_row, F, BVK,Nibk,Nias,ias,ibks,A,M);\n\
		for (int j = 0;j < BS2;j++)\n\
			Om[k*(N+2)+1 + i*BS2 + j] = -(sigma / tau)*C[k*(N+2)+1 + i*BS2 + j] - d*delta2_m*BVK[j*(M+2)+k] / (h1*h1);\n\
	}\n\
}\n\
__kernel void C_v(__global double *Al,__global double *Bt,__global double *Om,__global double *U,__global double *V_v,__global double *V_h,__global double *C,__global char *class,int N,int x0,int x1,int y0,int y1)\n\
{\n\
	int j=get_global_id(0)+y0;\n\
	C[j*(N+2)+N] = 1.0;\n\
	C[j*(N+2)+0] = 0.0;\n\
	for (int i = x1 - 1;i >= x0;i--)\n\
		C[j*(N+2)+i] = Al[j*(N+2)+i + 1] * C[j*(N+2)+i + 1] + Bt[j*(N+2)+i + 1];\n\
}\n\
";
class space_fract_solver_2d {
public:
	double U[N + 2][M+2]; // pressure/temperature
	double V_h[N + 2][M+2]; // velocity
	double V_v[N + 2][M+2]; // velocity
	double C[N + 2][M+2]; // concentration
	double Al[N + 2][M+2]; // alpha coefficients
	double Bt[N + 2][M+2]; // beta coefficients
	double Om[N + 2][M+2]; // right part
	double alpha, beta;
	double tau;
	int use_ocl;
	int do_not_clear_approx;

	virtual void init(int ocl=0,double _a=0.8,double _b=0.9)=0;
	virtual void al1_h(int i0=1,int i1=N,int j0=1,int j1=M)=0;
	virtual void al1_v(int i0=1,int i1=M,int j0=1,int j1=N)=0;
	virtual void Om1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB)=0;
	virtual void Om1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB)=0;
	virtual void bt1_h(int i0=1,int i1=N,int j0=1,int j1=M)=0;
	virtual void bt1_v(int i0=1,int i1=M,int j0=1,int j1=N)=0;
	virtual void U1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB,int only_f=0)=0;
	virtual void U1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB,int only_f=0)=0;
	virtual void V1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB)=0;
	virtual void V1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB)=0;
	virtual void al2_h(int i0=1,int i1=N,int j0=1,int j1=M)=0;
	virtual void al2_v(int i0=1,int i1=M,int j0=1,int j1=N)=0;
	virtual void Om2_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB)=0;
	virtual void Om2_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB)=0;
	virtual void bt2_h(int i0=1,int i1=N,int j0=1,int j1=M)=0;
	virtual void bt2_v(int i0=1,int i1=M,int j0=1,int j1=N)=0;
	virtual void C1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB,int only_f=0)=0;
	virtual void C1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB,int only_f=0)=0;

	virtual void fr_ocl_get(double(*B)[M + 2], int i0, int i1, int j0, int j1) { }
	virtual void fr_ocl_put(double(*B)[M + 2], int i0, int i1, int j0, int j1) { }

	virtual void exchange_approx(int r, int np, int block, std::vector<int> &j0s, std::vector<int> &j1s, int xy, int poffmult = 1, int rbmode = 0, int debug = 0) {}
	virtual void fr_ocl_clear_approximations() {}
	virtual void clear_approximations() {}
	// saving for time-fractional derivative
	virtual void save_U() {}
	virtual void save_C() {}
};
class space_fract_solver_HVC_2d:public space_fract_solver_2d {
public:
	double delta,delta_m,s1,s1_m;
	double h1,h2;
	double delta2,delta2_m;

	double w;
	double d;
	double H0;
	double sigma;

	typedef struct
	{
		std::vector<ia_approximation> ias; // "long" approximations
		std::vector<ibk_approximation> ibks;	// "short" approximations
	} approximation;
	approximation app[N]; // approximation for each row 

	double F[N + 2]; // function values
	double BVA[BS]; // approximated values of F=sum_r=1...x-1(g(x,r)*F[r])
	double BVK[BS]; // values of F
	std::vector<ia_approximation> ias;
	std::vector<ibk_approximation> ibks;

	int n_ibk[N + 2]; // number of ibk approximations per row
	int n_ias[N + 2]; // number of ia approximations per row
	char *row_ibks; // storage for approximation in one row
	char *row_ias;
#ifdef OCL
	OpenCL_program *prg;
	OpenCL_commandqueue *queue;
	OpenCL_prg *prog;
	OpenCL_kernel *kUAl_h, *kUBt_h, *kUOm_h, *kU_h;
	OpenCL_kernel *kUAl_v, *kUBt_v, *kUOm_v, *kU_v;
	OpenCL_kernel *kCAl_h, *kCBt_h, *kCOm_h, *kC_h;
	OpenCL_kernel *kCAl_v, *kCBt_v, *kCOm_v, *kC_v;
	OpenCL_kernel *kV_h, *kV_v;
	OpenCL_buffer *bAl, *bBt, *bOm, *bU, *bV_h, *bV_v, *bC, *bS, *b_ibk, *b_ias, *b_nibk, *b_nias, *b_F, *b_BVK;
#endif
	int alloced_approx_per_row;

	void init(int ocl = 0,double _alpha=0.8,double _beta=0.9)
	{
		tau = (0.1 / N);
		w = 0.25e-3;
		alpha = _alpha;
		beta = _beta;
		d = 0.06e-2;
		H0 = 0.95;
		sigma = 0.2;
		do_not_clear_approx = 0;
		use_ocl = ocl;

		h1 = 1.0 / N;
		h2 = 1.0 / M;
		delta = pow(h1, 1.0 - alpha) / Gamma(2.0 - alpha);
		delta2 = pow(h1, 1.0 - beta) / Gamma(2.0 - beta);
		s1 = 2.0 + h1*h1 / (tau*delta);
		delta_m = pow(h2, 1.0 - alpha) / Gamma(2.0 - alpha);
		delta2_m = pow(h2, 1.0 - beta) / Gamma(2.0 - beta);
		s1_m = 2.0 + h2*h2 / (tau*delta_m);
		for (int i = 0;i < N + 1;i++)
			for (int j = 0;j < M + 1;j++)
			{
				U[i][j] = H0;
				C[i][j] = 1.0;
				V_v[i][j] = 0.0;
				V_h[i][j] = 0.0;
			}
		for (int i = 0;i < M + 1;i++)
		{
			C[0][i] = 0.0;
			U[0][i] = 1.0;
		}
		for (int i = 0;i < N + 1;i++)
		{
			C[i][0] = 0.0;
			U[i][0] = 1.0;
		}
		// initialize OpenCL
		if (use_ocl)
		{
#ifdef OCL
			prg = new OpenCL_program(0);
			queue = prg->create_queue(device, 0);
			{
				char *text = new char[(strlen(fr2_opencl_text) + strlen(fr2_opencl_text2)) * 2];
				ia_approximation ia;
				ibk_approximation ibk;
				sprintf(text, fr2_opencl_text, ((double_ext == 0) ? "cl_amd_fp64" : "cl_khr_fp64")
					, ((char *)&delta) - (char *)this
					, ((char *)&s1) - (char *)this
					, ((char *)&h1) - (char *)this
					, ((char *)&delta2) - (char *)this
					, ((char *)&delta_m) - (char *)this
					, ((char *)&s1_m) - (char *)this
					, ((char *)&h2) - (char *)this
					, ((char *)&delta2_m) - (char *)this
					, ((char *)&tau) - (char *)this
					, ((char *)&w) - (char *)this
					, ((char *)&alpha) - (char *)this
					, ((char *)&beta) - (char *)this
					, ((char *)&d) - (char *)this
					, ((char *)&H0) - (char *)this
					, ((char *)&sigma) - (char *)this
					, sizeof(ia_approximation)
					, ((char *)&ia.min) - (char *)&ia
					, ((char *)&ia.nterms) - (char *)&ia
					, ((char *)&ia.c[0]) - (char *)&ia
					, sizeof(ibk_approximation)
					, ((char *)&ibk.min) - (char *)&ibk
					, ((char *)&ibk.max) - (char *)&ibk
					, ((char *)&ibk.b) - (char *)&ibk
					, ((char *)&ibk.nterms) - (char *)&ibk
					, ((char *)&ibk.old) - (char *)&ibk
					, ((char *)&ibk.c[0]) - (char *)&ibk
					, approx_eps
					, AK
					, MAXAK
					, BS
					, BS2
					, eps
					, Gamma(2.0 - alpha));
				strcat(text, fr2_opencl_text2);
				prog = prg->create_program(text);
				delete[] text;
			}
			kUAl_h = prg->create_kernel(prog, "UAl_h");
			kUAl_v = prg->create_kernel(prog, "UAl_v");
			kUBt_h = prg->create_kernel(prog, "UBt_h");
			kUBt_v = prg->create_kernel(prog, "UBt_v");
			kUOm_h = prg->create_kernel(prog, "UOm_h");
			kUOm_v = prg->create_kernel(prog, "UOm_v");
			kU_h = prg->create_kernel(prog, "U_h");
			kU_v = prg->create_kernel(prog, "U_v");
			kV_h = prg->create_kernel(prog, "V_h");
			kV_v = prg->create_kernel(prog, "V_v");
			kCAl_h = prg->create_kernel(prog, "CAl_h");
			kCAl_v = prg->create_kernel(prog, "CAl_v");
			kCBt_h = prg->create_kernel(prog, "CBt_h");
			kCBt_v = prg->create_kernel(prog, "CBt_v");
			kCOm_h = prg->create_kernel(prog, "COm_h");
			kCOm_v = prg->create_kernel(prog, "COm_v");
			kC_h = prg->create_kernel(prog, "C_h");
			kC_v = prg->create_kernel(prog, "C_v");
			bS = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(space_fract_solver_HVC_2d), (void *)this);
			bAl = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			bBt = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			bOm = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			bU = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			bV_v = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			bV_h = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			bC = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			b_F = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			b_BVK = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*(M + 2)*sizeof(double), NULL);
			alloced_approx_per_row = 4 * (N + 2);
			row_ibks = new char[alloced_approx_per_row*sizeof(ibk_approximation)];
			row_ias = new char[alloced_approx_per_row*sizeof(ia_approximation)];
			b_nibk = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*sizeof(int), NULL);
			b_nias = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*sizeof(int), NULL);
			b_ibk = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*alloced_approx_per_row*sizeof(ibk_approximation), NULL);
			b_ias = prg->create_buffer(CL_MEM_READ_WRITE, (N + 2)*alloced_approx_per_row*sizeof(ia_approximation), NULL);
			for (int i = 0;i < N + 2;i++)
			{
				queue->EnqueueWriteBuffer(bU, U[i], i*(M + 2)*sizeof(double), (M + 2)*sizeof(double));
				queue->EnqueueWriteBuffer(bC, C[i], i*(M + 2)*sizeof(double), (M + 2)*sizeof(double));
				queue->EnqueueWriteBuffer(bV_v, V_v[i], i*(M + 2)*sizeof(double), (M + 2)*sizeof(double));
				queue->EnqueueWriteBuffer(bV_h, V_h[i], i*(M + 2)*sizeof(double), (M + 2)*sizeof(double));
			}
			clear_approximations();
#endif
		}
	}
#ifdef OCL
	OpenCL_buffer *fr_ocl_buffer(double(*B)[M + 2])
	{
		if (B == Al) return bAl;
		if (B == Bt) return bBt;
		if (B == Om) return bOm;
		if (B == U) return bU;
		if (B == C) return bC;
		if (B == V_v) return bV_v;
		if (B == V_h) return bV_h;
		return NULL;
	}
#endif
	void fr_ocl_get(double(*B)[M + 2], int i0, int i1, int j0, int j1)
	{
#ifdef OCL
		OpenCL_buffer *b = fr_ocl_buffer(B);
		for (int i = i0;i < i1;i++)
			queue->EnqueueBuffer(b, B[i] + j0, (i*(M + 2) + j0)*sizeof(double), (j1 - j0)*sizeof(double));
		queue->Finish();
#endif
	}
	void fr_ocl_put(double(*B)[M + 2], int i0, int i1, int j0, int j1)
	{
#ifdef OCL
		OpenCL_buffer *b = fr_ocl_buffer(B);
		for (int i = i0;i < i1;i++)
			queue->EnqueueWriteBuffer(b, B[i] + j0, (i*(M + 2) + j0)*sizeof(double), (j1 - j0)*sizeof(double));
		queue->Finish();
#endif
	}
	void fr_ocl_check_and_resize_approximations()
	{
#ifdef OCL
		// check
		int max_used_approx_per_row = 0;
		queue->EnqueueBuffer(b_nibk, n_ibk);
		queue->EnqueueBuffer(b_nias, n_ias);
		for (int i = 0;i < N + 1;i++)
		{
			if (n_ibk[i] > max_used_approx_per_row) max_used_approx_per_row = n_ibk[i];
			if (n_ias[i] > max_used_approx_per_row) max_used_approx_per_row = n_ias[i];
		}
		// resize
		if (max_used_approx_per_row * 2 > alloced_approx_per_row)
		{
			char *newa1 = new char[2 * (N + 2)*alloced_approx_per_row*sizeof(ibk_approximation)];
			char *newa2 = new char[2 * (N + 2)*alloced_approx_per_row*sizeof(ia_approximation)];
			// get old from gpu
			queue->EnqueueBuffer(b_ibk, newa1, 0, (N + 2)*alloced_approx_per_row*sizeof(ibk_approximation));
			queue->EnqueueBuffer(b_ias, newa2, 0, (N + 2)*alloced_approx_per_row*sizeof(ia_approximation));
			// shift data
			for (int i = N + 1;i >= 1;i--)
			{
				memcpy(newa1 + (i * 2 * alloced_approx_per_row*sizeof(ibk_approximation)), newa1 + (i * alloced_approx_per_row*sizeof(ibk_approximation)), alloced_approx_per_row*sizeof(ibk_approximation));
				memcpy(newa2 + (i * 2 * alloced_approx_per_row*sizeof(ia_approximation)), newa1 + (i * alloced_approx_per_row*sizeof(ia_approximation)), alloced_approx_per_row*sizeof(ia_approximation));
			}
			// delete old gpu buffers
			delete b_ibk;
			delete b_ias;
			// pu new data
			alloced_approx_per_row *= 2;
			b_ibk = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (N + 2)*alloced_approx_per_row*sizeof(ibk_approximation), newa1);
			b_ias = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (N + 2)*alloced_approx_per_row*sizeof(ia_approximation), newa2);
			// clean up
			queue->Finish();
			delete[] newa1;
			delete[] newa2;
			// realloc row approximations storage
			delete[] row_ias;
			delete[] row_ibks;
			row_ibks = new char[alloced_approx_per_row*sizeof(ibk_approximation)];
			row_ias = new char[alloced_approx_per_row*sizeof(ia_approximation)];
		}
#endif
	}
#ifdef OCL
	void fr_ocl_set_args(OpenCL_kernel *k, int i0, int i1, int j0, int j1, int b0 = -1, int b1 = -1)
	{
		int err;
		int i = M;
		err = k->SetBufferArg(bAl, 0);
		err |= k->SetBufferArg(bBt, 1);
		err |= k->SetBufferArg(bOm, 2);
		err |= k->SetBufferArg(bU, 3);
		err |= k->SetBufferArg(bV_v, 4);
		err |= k->SetBufferArg(bV_h, 5);
		err |= k->SetBufferArg(bC, 6);
		err |= k->SetBufferArg(bS, 7);
		err |= k->SetArg(8, sizeof(int), &i);
		err |= k->SetArg(9, sizeof(int), &i0);
		err |= k->SetArg(10, sizeof(int), &i1);
		err |= k->SetArg(11, sizeof(int), &j0);
		err |= k->SetArg(12, sizeof(int), &j1);
		if (b0 != -1)
		{
			err |= k->SetArg(13, sizeof(int), &b0);
			err |= k->SetArg(14, sizeof(int), &b1);
			err |= k->SetBufferArg(b_nibk, 15);
			err |= k->SetBufferArg(b_nias, 16);
			err |= k->SetBufferArg(b_ibk, 17);
			err |= k->SetBufferArg(b_ias, 18);
			err |= k->SetBufferArg(b_F, 19);
			err |= k->SetBufferArg(b_BVK, 20);
			err |= k->SetArg(21, sizeof(int), &alloced_approx_per_row);
			err |= k->SetArg(22, sizeof(double), &A);
			i = N;
			err |= k->SetArg(23, sizeof(int), &i);
		}
		if (err) SERROR("Error: Failed to set kernels args");
	}
	void fr_ocl_call(OpenCL_kernel *k, int i0, int i1, int j0, int j1, int b0 = -1, int b1 = -1)
	{
		size_t nth, lsize;
		nth = j1 - j0;
		lsize = 1;
		fr_ocl_set_args(k, i0, i1, j0, j1, b0, b1);
		if ((k == kU_h) || (k == kC_h))
		{
			int i = N;
			k->SetArg(13, sizeof(int), &i);
		}
		queue->ExecuteKernel(k, 1, &nth, &lsize);
		queue->Finish();
	}
#endif
	void exchange_approx(int r, int np, int block, std::vector<int> &j0s, std::vector<int> &j1s, int xy, int poffmult = 1, int rbmode = 0, int debug = 0);
	void fr_ocl_clear_approximations();
	void clear_approximations();
	// U equation alpha coeffients
	void al1_h(int i0 = 1, int i1 = N, int j0 = 1, int j1 = M)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kUAl_h, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Al[1][j] = 0;
			for (int i = i0;i < i1;i++)
				Al[i + 1][j] = 1.0 / (s1 - Al[i][j]);
		}
	}
	void al1_v(int i0 = 1, int i1 = M, int j0 = 1, int j1 = N)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kUAl_v, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Al[j][1] = 0;
			for (int i = i0;i < i1;i++)
				Al[j][i + 1] = 1.0 / (s1_m - Al[j][i]);
		}
	}
	// U equation right part
	void Om1_h(int approx, int i0 = 1, int i1 = N, int j0 = 1, int j1 = M, int b0 = 0, int b1 = NB)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_check_and_resize_approximations();
			fr_ocl_call(kUOm_h, i0, i1, j0, j1, b0, b1);
			if (do_not_clear_approx == 0)
				fr_ocl_clear_approximations();
			return;
		}
#endif
		for (int k = j0;k<j1;k++)
		{
			if (approx)
			{
				ias = app[k].ias;
				ibks = app[k].ibks;
			}
			for (int i = i0;i < i1;i++)
				F[i] = U[i + 1][k] - 2.0*U[i][k] + U[i - 1][k];
			for (int i = b0;i < b1;i++)
			{
				if (approx == 0)
					calc_v(BS,i, F, BVK);
				else
					calc_va(BS, i, F, BVK,NULL,ias,ibks);
				for (int j = 0;j < BS;j++)
					Om[1 + i*BS + j][k] = -(h1*h1 / delta)*((U[1 + i*BS + j][k] / tau) + delta*BVK[j] / (h1*h1));
			}
			if (approx)
			{
				if (do_not_clear_approx == 0)
				{
					ias.clear();
					ibks.clear();
				}
				else
				{
					app[k].ias = ias;
					app[k].ibks = ibks;
				}
			}
		}
	}
	void Om1_v(int approx, int i0 = 1, int i1 = M, int j0 = 1, int j1 = N, int b0 = 0, int b1 = NB)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_check_and_resize_approximations();
			fr_ocl_call(kUOm_v, i0, i1, j0, j1, b0, b1);
			if (do_not_clear_approx == 0)
				fr_ocl_clear_approximations();
			return;
		}
#endif
		for (int k = j0;k<j1;k++)
		{
			if (approx)
			{
				ias = app[k].ias;
				ibks = app[k].ibks;
			}
			for (int i = i0;i < i1;i++)
				F[i] = U[k][i + 1] - 2.0*U[k][i] + U[k][i - 1];
			for (int i = b0;i < b1;i++)
			{
				if (approx == 0)
					calc_v(BS2, i, F, BVK);
				else
					calc_va(BS2, i, F, BVK, NULL, ias, ibks);
				for (int j = 0;j < BS2;j++)
					Om[k][1 + i*BS2 + j] = -(h2*h2 / delta_m)*((U[k][1 + i*BS2 + j] / tau) + delta_m*BVK[j] / (h2*h2));
			}
			if (approx)
			{
				if (do_not_clear_approx == 0)
				{
					ias.clear();
					ibks.clear();
				}
				else
				{
					app[k].ias = ias;
					app[k].ibks = ibks;
				}
			}
		}
	}
	// U equation beta coeffients
	void bt1_h(int i0 = 1, int i1 = N, int j0 = 1, int j1 = M)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kUBt_h, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Bt[1][j] = 1;
			for (int i = i0;i < i1;i++)
				Bt[i + 1][j] = Al[i + 1][j] * (Bt[i][j] - Om[i][j]);
		}
	}
	void bt1_v(int i0 = 1, int i1 = M, int j0 = 1, int j1 = N)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kUBt_v, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Bt[j][1] = 1;
			for (int i = i0;i < i1;i++)
				Bt[j][i + 1] = Al[j][i + 1] * (Bt[j][i] - Om[j][i]);
		}
	}
	// calc U
	void U1_h(int approx, int i0 = 1, int i1 = N, int j0 = 1, int j1 = M, int b0 = 0, int b1 = NB, int only_f = 0)
	{
		if (only_f == 0)
		{
			A = 1 - alpha;
			al1_h(i0, i1, j0, j1);
			Om1_h(approx, i0, i1, j0, j1, b0, b1);
			bt1_h(i0, i1, j0, j1);
		}
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kU_h, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			U[N][j] = H0;
			U[0][j] = 1.0;
			for (int i = i1 - 1;i >= i0;i--)
				U[i][j] = Al[i + 1][j] * U[i + 1][j] + Bt[i + 1][j];
		}
	}
	void U1_v(int approx, int i0 = 1, int i1 = M, int j0 = 1, int j1 = N, int b0 = 0, int b1 = NB, int only_f = 0)
	{
		if (only_f == 0)
		{
			A = 1 - alpha;
			al1_v(i0, i1, j0, j1);
			Om1_v(approx, i0, i1, j0, j1, b0, b1);
			bt1_v(i0, i1, j0, j1);
		}
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kU_v, i0, i1, j0, j1);
			return;
		}
#endif		
		for (int j = j0;j<j1;j++)
		{
			U[j][M] = H0;
			U[j][0] = 1.0;
			for (int i = i1 - 1;i >= i0;i--)
				U[j][i] = Al[j][i + 1] * U[j][i + 1] + Bt[j][i + 1];
		}
	}
	// calc V
	void V1_h(int approx, int i0 = 1, int i1 = N, int j0 = 1, int j1 = M, int b0 = 0, int b1 = NB)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_check_and_resize_approximations();
			fr_ocl_call(kV_h, i0, i1, j0, j1, b0, b1);
			if (do_not_clear_approx == 0)
				fr_ocl_clear_approximations();
			return;
		}
#endif
		for (int k = j0;k<j1;k++)
		{
			if (approx)
			{
				ias = app[k].ias;
				ibks = app[k].ibks;
			}
			V_h[0][k] = 0;
			V_h[1][k] = -(w / (h1*Gamma(2.0 - alpha)))*(U[1][k] - U[0][k]);
			for (int i = i0;i < i1;i++)
				F[i] = U[i][k] - U[i - 1][k];
			for (int i = b0;i < b1;i++)
			{
				if (approx == 0)
					calc_v(BS, i, F, BVK);
				else
					calc_va(BS, i, F, BVK, NULL, ias, ibks);
				for (int j = 0;j < BS;j++)
					if ((i != 0) || (j != 0))
						V_h[1 + i*BS + j][k] = -(w / (h1*Gamma(2.0 - alpha)))*(U[1 + i*BS + j][k] - U[i*BS + j][k] + BVK[j]);
			}
			if (approx)
			{
				if (do_not_clear_approx == 0)
				{
					ias.clear();
					ibks.clear();
				}
				else
				{
					app[k].ias = ias;
					app[k].ibks = ibks;
				}
			}
		}
	}
	void V1_v(int approx, int i0 = 1, int i1 = M, int j0 = 1, int j1 = N, int b0 = 0, int b1 = NB)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_check_and_resize_approximations();
			fr_ocl_call(kV_v, i0, i1, j0, j1, b0, b1);
			if (do_not_clear_approx == 0)
				fr_ocl_clear_approximations();
			return;
		}
#endif
		for (int k = j0;k<j1;k++)
		{
			if (approx)
			{
				ias = app[k].ias;
				ibks = app[k].ibks;
			}
			V_v[k][0] = 0;
			V_v[k][1] = -(w / (h2*Gamma(2.0 - alpha)))*(U[k][1] - U[k][0]);
			for (int i = i0;i < i1;i++)
				F[i] = U[k][i] - U[k][i - 1];
			for (int i = b0;i < b1;i++)
			{
				if (approx == 0)
					calc_v(BS2, i, F, BVK);
				else
					calc_va(BS2, i, F, BVK, NULL, ias, ibks);
				for (int j = 0;j < BS2;j++)
					if ((i != 0) || (j != 0))
						V_v[k][1 + i*BS2 + j] = -(w / (h2*Gamma(2.0 - alpha)))*(U[k][1 + i*BS2 + j] - U[k][i*BS2 + j] + BVK[j]);
			}
			if (approx)
			{
				if (do_not_clear_approx == 0)
				{
					ias.clear();
					ibks.clear();
				}
				else
				{
					app[k].ias = ias;
					app[k].ibks = ibks;
				}
			}
		}
	}
	// C equation alpha coefficient
	void al2_h(int i0 = 1, int i1 = N, int j0 = 1, int j1 = M)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kCAl_h, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Al[1][j] = 0;
			for (int i = i0;i < i1;i++)
				Al[i + 1][j] = ((delta2*d / (h1*h1)) - (V_h[i][j] / (2.0*h1))) / (((sigma / tau) + (2.0*delta2*d / (h1*h1)) + ((V_h[i + 1][j] - V_h[i - 1][j]) / (2.0*h1))) - Al[i][j] * ((delta2*d / (h1*h1)) + (V_h[i][j] / (2.0*h1))));
		}
	}
	void al2_v(int i0 = 1, int i1 = M, int j0 = 1, int j1 = N)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kCAl_v, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Al[j][1] = 0;
			for (int i = i0;i < i1;i++)
				Al[j][i + 1] = ((delta2_m*d / (h2*h2)) - (V_v[j][i] / (2.0*h2))) / (((sigma / tau) + (2.0*delta2_m*d / (h2*h2)) + ((V_v[j][i + 1] - V_v[j][i - 1]) / (2.0*h2))) - Al[j][i] * ((delta2_m*d / (h2*h2)) + (V_v[j][i] / (2.0*h2))));
		}
	}
	// C equation right part
	void Om2_h(int approx, int i0 = 1, int i1 = N, int j0 = 1, int j1 = M, int b0 = 0, int b1 = NB)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_check_and_resize_approximations();
			fr_ocl_call(kCOm_h, i0, i1, j0, j1, b0, b1);
			if (do_not_clear_approx == 0)
				fr_ocl_clear_approximations();
			return;
		}
#endif
		for (int k = j0;k<j1;k++)
		{
			if (approx)
			{
				ias = app[k].ias;
				ibks = app[k].ibks;
			}
			for (int i = i0;i < i1;i++)
				F[i] = C[i + 1][k] - 2.0*C[i][k] + C[i - 1][k];
			for (int i = b0;i < b1;i++)
			{
				if (approx == 0)
					calc_v(BS, i, F, BVK);
				else
					calc_va(BS, i, F, BVK, NULL, ias, ibks);
				for (int j = 0;j < BS;j++)
					Om[1 + i*BS + j][k] = -(sigma / tau)*C[1 + i*BS + j][k] - d*delta2*BVK[j] / (h1*h1);
			}
			if (approx)
			{
				if (do_not_clear_approx == 0)
				{
					ias.clear();
					ibks.clear();
				}
				else
				{
					app[k].ias = ias;
					app[k].ibks = ibks;
				}
			}
		}
	}
	void Om2_v(int approx, int i0 = 1, int i1 = M, int j0 = 1, int j1 = N, int b0 = 0, int b1 = NB)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_check_and_resize_approximations();
			fr_ocl_call(kCOm_v, i0, i1, j0, j1, b0, b1);
			if (do_not_clear_approx == 0)
				fr_ocl_clear_approximations();
			return;
		}
#endif
		for (int k = j0;k<j1;k++)
		{
			if (approx)
			{
				ias = app[k].ias;
				ibks = app[k].ibks;
			}
			for (int i = i0;i < i1;i++)
				F[i] = C[k][i + 1] - 2.0*C[k][i] + C[k][i - 1];
			for (int i = b0;i < b1;i++)
			{
				if (approx == 0)
					calc_v(BS2, i, F, BVK);
				else
					calc_va(BS2, i, F, BVK, NULL, ias, ibks);
				for (int j = 0;j < BS2;j++)
					Om[k][1 + i*BS2 + j] = -(sigma / tau)*C[k][1 + i*BS2 + j] - d*delta2_m*BVK[j] / (h2*h2);
			}
			if (approx)
			{
				if (do_not_clear_approx == 0)
				{
					ias.clear();
					ibks.clear();
				}
				else
				{
					app[k].ias = ias;
					app[k].ibks = ibks;
				}
			}
		}
	}
	void bt2_h(int i0 = 1, int i1 = N, int j0 = 1, int j1 =M)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kCBt_h, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Bt[1][j] = 0;
			for (int i = i0;i < i1;i++)
				Bt[i + 1][j] = (Al[i + 1][j] / ((delta2*d / (h1*h1)) - (V_h[i][j] / (2.0*h1)))) * (((delta2*d / (h1*h1)) + (V_h[i][j] / (2.0*h1)))*Bt[i][j] - Om[i][j]);
		}
	}
	void bt2_v(int i0 = 1, int i1 = M, int j0 = 1, int j1 = N)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kCBt_v, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			Bt[j][1] = 0;
			for (int i = i0;i < i1;i++)
				Bt[j][i + 1] = (Al[j][i + 1] / ((delta2_m*d / (h2*h2)) - (V_v[j][i] / (2.0*h2)))) * (((delta2_m*d / (h2*h2)) + (V_v[j][i] / (2.0*h2)))*Bt[j][i] - Om[j][i]);
		}
	}
	// calc C
	void C1_h(int approx, int i0 = 1, int i1 = N, int j0 = 1, int j1 = M, int b0 = 0, int b1 = NB, int only_f = 0)
	{
		if (only_f == 0)
		{
			A = 1 - beta;
			al2_h(i0, i1, j0, j1);
			Om2_h(approx, i0, i1, j0, j1, b0, b1);
			bt2_h(i0, i1, j0, j1);
		}
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kC_h, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			C[N][j] = 1.0;
			C[0][j] = 0.0;
			for (int i = i1 - 1;i >= i0;i--)
				C[i][j] = Al[i + 1][j] * C[i + 1][j] + Bt[i + 1][j];
		}
	}
	void C1_v(int approx, int i0 = 1, int i1 = M, int j0 = 1, int j1 = N, int b0 = 0, int b1 = NB, int only_f = 0)
	{
		if (only_f == 0)
		{
			A = 1 - beta;
			al2_v(i0, i1, j0, j1);
			Om2_v(approx, i0, i1, j0, j1, b0, b1);
			bt2_v(i0, i1, j0, j1);
		}
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kC_v, i0, i1, j0, j1);
			return;
		}
#endif
		for (int j = j0;j<j1;j++)
		{
			C[j][M] = 1.0;
			C[j][0] = 0.0;
			for (int i = i1 - 1;i >= i0;i--)
				C[j][i] = Al[j][i + 1] * C[j][i + 1] + Bt[j][i + 1];
		}
	}
};
////////////////////////////////////////////////////////////////////////////////
///////////// non-isothermal problem with stationary filtration field //////////
////////////// BS must be equal to 1
////////////////////////////////////////////////////////////////////////////////
class space_fract_solver_T_2d:public space_fract_solver_2d {
public:
	double** Ts[3];
	double** Cs[3];
	std::vector<double> Ts1[N + 2][M + 2];
	std::vector<double> Ts2[N + 2][M + 2];
	std::vector<double> Cs1[N + 2][M + 2];
	std::vector<double> Cs2[N + 2][M + 2];
	std::vector<double> time_points; // time steps
	int approx_builtT1[N + 2][M + 2];
	int approx_builtT2[N + 2][M + 2];
	int approx_builtC1[N + 2][M + 2];
	int approx_builtC2[N + 2][M + 2];
	int approx_clearedT1[N + 2][M + 2];
	int approx_clearedT2[N + 2][M + 2];
	int approx_clearedC1[N + 2][M + 2];
	int approx_clearedC2[N + 2][M + 2];
	int tcall,ccall;

	double sigma;
	double h1,h2,fi0,a,kappa;
	double d;
	double k1;
	double bC,bT;
	double fC,fT;
	double H0;
	double dt;
	double lambda,mu,omega;
	double tb2,ta2;
	double tpa,tpb;
	double T; // current time value
	int varying_time_points;

	int do_not_clear_approx;
	struct
	{
		std::vector<ia_approximation> ias; // "long" approximations
		std::vector<ibk_approximation> ibks;	// "short" approximations
	} TappH[N+2][M+2],CappH[N+2][M+2],TappV[N+2][M+2],CappV[N+2][M+2];
	
	double sqv(int i,int j)
	{
		double f=i*h1;
		double p=(j-0.5)*h2;
		double s1=sin(0.5*M_PI*p);
	   double s2=exp(0.5*M_PI*f)*cos(0.5*M_PI*p)+k1;
	   return (4*a/(M_PI*M_PI))*((1/(exp(M_PI*f)*s1*s1))+(1/(s2*s2)));
	}
	void clear_Ts_Cs()
	{
		for (int i=0;i<3;i++)
		if (Ts[i])
		{
			for (int j=0;j<N+2;j++)
				delete [] Ts[i][j];
			delete [] Ts[i];
		}
		for (int i=0;i<3;i++)
		if (Cs[i])
		{
			for (int j=0;j<N+2;j++)
				delete [] Cs[i][j];
			delete [] Cs[i];
		}
	}
	void copy(space_fract_solver_T_2d *s2)
	{
		clear_Ts_Cs();
		this[0]=s2[0];
		for (int j=0;j<3;j++)
		{
			if (Ts[j])
			{
				Ts[j] = new double *[N + 2];
				for (int i = 0;i<N + 2;i++)
					Ts[j][i] = new double[M + 2];
				for (int i = 0;i <N + 2;i++)
					memcpy(Ts[j][i], s2->Ts[j][i], (M + 2)*sizeof(double));
			}
			if (Cs[j])
			{
				Cs[j] = new double *[N + 2];
				for (int i = 0;i<N + 2;i++)
					Cs[j][i] = new double[M + 2];
				for (int i = 0;i <N + 2;i++)
					memcpy(Cs[j][i], s2->Cs[j][i], (M + 2)*sizeof(double));
			}
		}		
	}
	~space_fract_solver_T_2d()
	{
		clear_Ts_Cs();
	}
	void reset_tau(double _tau)
	{
	        tau=_tau;
		fC = (sigma/(2.0*Gamma(2.0 - alpha)));
		fT = 1.0/(2.0*Gamma(2.0 - beta));
		bC = (sigma/(pow(2.0, 1.0 - alpha) *pow(tau,alpha)* Gamma(2.0 - alpha)));
		bT = 1.0/(pow(2.0, 1.0 - beta) *pow(tau,beta)* Gamma(2.0 - beta));
		tb2=pow(2.0/tau,beta);
		ta2=pow(2.0/tau,alpha);
		tpa=pow(tau,1.0-alpha);
		tpb=pow(tau,1.0-beta);
	}
	void init(int ocl=0,double _alpha=0.8,double _beta=0.9)
	{
		use_ocl=ocl;
		tcall = ccall= 0;
		tau = 0.000005;//0.000001/8.0;

		sigma = 0.2;
		alpha = _alpha;
		beta = _beta;
		fi0 = 5.0;
		d = 0.022;
		a=2025.0;
		kappa=2.22;
		k1=5.73;
		//dt = 0.0;
		dt = 0.0005;
		//dt = 0.005;
		//dt = 0.076;
		//lambda = 0.0066;
		lambda = 0.066;
		mu =6.5;
		omega=0.3;
		H0 = 0.05;
		do_not_clear_approx=0;
		varying_time_points=0;
		T=0.0;
		Ts[0] = Ts[1] = Ts[2] = Cs[0] = Cs[1] = Cs[2] = NULL;

		h1 = 2.0*fi0/(2.0*N+1.0);
		h2 = 1.0 / M;
		
		reset_tau(tau);		
		
		for (int i = 0;i <= N + 1;i++)
		for (int j = 0;j <= M + 1;j++)
		{
			C[i][j] = 0.0;
			U[i][j] = 0.0;
			V_v[i][j]=sqv(i,j);
			Ts1[i][j].push_back(0.0);
			Ts2[i][j].push_back(0.0);
			Cs1[i][j].push_back(0.0);
			Cs2[i][j].push_back(0.0);
			approx_builtT1[i][j] = 0;
			approx_builtT2[i][j] = 0;
			approx_builtC1[i][j] = 0;
			approx_builtC2[i][j] = 0;
			approx_clearedT1[i][j] = 0;
			approx_clearedT2[i][j] = 0;
			approx_clearedC1[i][j] = 0;
			approx_clearedC2[i][j] = 0;
		}
		for (int i = 0;i <= M + 1;i++)
		{
			C[0][i] = 1.0;
			U[0][i] = 1.0;
		}
		time_points.push_back(0.0);
		time_points.push_back(0.0);
	}
	// start next time step for varying time steps
	void init_time_step(double t)
	{
		if (varying_time_points)
		{
			tau=t;
			if (time_points.size()==2)
			    time_points[0]=-tau;
			T+=tau;			
			time_points.push_back(T);
			bC = (sigma/(pow(2.0, 1.0 - alpha) *pow(tau,alpha)* Gamma(2.0 - alpha)));
			bT = 1.0/(pow(2.0, 1.0 - beta) *pow(tau,beta)* Gamma(2.0 - beta));
			tb2=pow(2.0/tau,beta);
			ta2=pow(2.0/tau,alpha);
			tpa=1.0;
			tpb=1.0;
		}
	}
	// save U in (0,1,2) rotation
	void save_U()
	{
		int rot = 1;
		double **uc = new double *[N + 2];
		for (int i = 0;i<N + 2;i++)
			uc[i] = new double[M + 2];
		for (int i = 0;i <N + 2;i++)
			memcpy(uc[i], &U[i][0], (M + 2)*sizeof(double));
		if (!Ts[0]) { Ts[0] = uc;rot=0; }
		if (rot) if (!Ts[1]) { Ts[1] = uc;rot=0; }
		if (rot) if (!Ts[2]) { Ts[2] = uc;rot=0; }
		if (rot)
		{
			for (int i = 0;i < N + 2;i++)
				delete[] Ts[0][i];
			delete[] Ts[0];
			Ts[0] = Ts[1];
			Ts[1] = Ts[2];
			Ts[2] = uc;
		}
		if ((tcall&1)==0)
		if (A != 0.0)
			if (Ts[1])
					for (int i = 1;i < N;i++)
						for (int k = 1;k<M;k++)
							Ts1[i][k].push_back((Ts[1][i][k] - Ts[0][i][k]) / tau);
		if ((tcall&1)==1)
		if (A != 0.0)
			if (Ts[1])//1
				for (int k = 1;k<N;k++)
					for (int i = 1;i < M;i++)
						Ts2[k][i].push_back((Ts[1][k][i] - Ts[0][k][i]) / tau); //1
		tcall++;
		if ((tcall%20)==0)
			saved_cleanup();
	}
	// save U in (0,1,2) rotation
	void save_C()
	{
		int rot = 1;
		double **uc = new double *[N + 2];
		for (int i = 0;i<N + 2;i++)
			uc[i] = new double[M + 2];
		for (int i = 0;i <N + 2;i++)
			memcpy(uc[i], &C[i][0], (M + 2)*sizeof(double));
		if (!Cs[0]) { Cs[0] = uc;rot=0; }
		if (rot) if (!Cs[1]) { Cs[1] = uc;rot=0; }
		if (rot) if (!Cs[2]) { Cs[2] = uc;rot = 0; }
		if (rot)
		{
			for (int i = 0;i < N + 2;i++)
				delete[] Cs[0][i];
			delete[] Cs[0];
			Cs[0] = Cs[1];
			Cs[1] = Cs[2];
			Cs[2] = uc;
		}
		if ((ccall & 1) == 0)
			if (A != 0.0)
				if (Cs[1])
					for (int i = 1;i < N;i++)
						for (int k = 1;k<M;k++)
							Cs1[i][k].push_back((Cs[1][i][k] - Cs[0][i][k]) / tau);
		if ((ccall & 1) == 1)
			if (A != 0.0)
			if (Cs[1]) //1
				for (int k = 1;k<N;k++)
					for (int i = 1;i < M;i++)
						Cs2[k][i].push_back((Cs[1][k][i] - Cs[0][k][i]) / tau); //1
		ccall++;
	}
	// removes from Ts[1,2]/Cs[1,2] values that are already put into approximation
	void saved_cleanup()
	{
		for (int i=0;i<N+2;i++)
			for (int j=0;j<M+2;j++)
			{
				std::vector<double> n1,n2,n3,n4;
				if (approx_builtT1[i][j])
				{
					n1.push_back(0.0);
					for (int k=approx_builtT1[i][j]+1;k<Ts1[i][j].size();k++)
						n1.push_back(Ts1[i][j][k]);
					Ts1[i][j]=n1;
					approx_clearedT1[i][j]+=approx_builtT1[i][j];
					approx_builtT1[i][j]=0;
				}
				if (approx_builtT2[i][j])
				{
					n2.push_back(0.0);
					for (int k=approx_builtT2[i][j]+1;k<Ts2[i][j].size();k++)
						n2.push_back(Ts2[i][j][k]);
					Ts2[i][j]=n2;
					approx_clearedT2[i][j]+=approx_builtT2[i][j];
					approx_builtT2[i][j]=0;
				}
				if (approx_builtC1[i][j])
				{
					n3.push_back(0.0);
					for (int k=approx_builtC1[i][j]+1;k<Cs1[i][j].size();k++)
						n3.push_back(Cs1[i][j][k]);
					Cs1[i][j]=n3;
					approx_clearedC1[i][j]+=approx_builtC1[i][j];
					approx_builtC1[i][j]=0;
				}
				if (approx_builtC2[i][j])
				{
					n4.push_back(0.0);
					for (int k=approx_builtC2[i][j]+1;k<Cs2[i][j].size();k++)
						n4.push_back(Cs2[i][j][k]);
					Cs2[i][j]=n4;
					approx_clearedC2[i][j]+=approx_builtC2[i][j];
					approx_builtC2[i][j]=0;
				}
			}
	}
	// T equation alpha coeffients
	void al1_h(int i0=1,int i1=N,int j0=1,int j1=M)
	{
		if (varying_time_points)
			init_time_step(tau);
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			Al[1][j] = 0;
			for (int i = i0;i < i1;i++)
			{
				double R=(V_v[i][j]/h1)*((lambda/h1)-0.5*mu);
				double G=(V_v[i][j]/h1)*((lambda/h1)+0.5*mu);
				Al[i + 1][j] = R / ((bT+R+G) - G*Al[i][j]);
			}
		}
	}
	void al1_v(int i0=1,int i1=M,int j0=1,int j1=N)
	{
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
#ifdef Tc1		
			Al[j][1] = 0;
#else
			Al[j][1]=1;
#endif						
			for (int i = i0;i < i1;i++)
			{
				double _M=lambda*V_v[j][i]/(h2*h2);
				Al[j][i + 1] = _M / ((bT+2.0*_M) - _M*Al[j][i]);
			}
		}
	}
	// T equation right part
	void Om1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB)
	{
#pragma omp parallel for
		for (int k=j0;k<j1;k++)
			for (int i = i0;i < i1;i++)
			{
				double V=0.0;
				if (Ts1[i][k].size()!=1)
				{
					if (varying_time_points)
					{
						if (approx == 0)
							calc_v_vt(1,Ts1[i][k].size() - 2, &Ts1[i][k][0], &V, &time_points[0],0.5, -0.5);
						else
						{
							for (int a = approx_builtT1[i][k];a <= Ts1[i][k].size() - 2;a++)
								calc_va(1, a + approx_clearedT1[i][k], &Ts1[i][k][0] - approx_clearedT1[i][k], &V, &time_points[0], TappH[i][k].ias, TappH[i][k].ibks, 0.5, -0.5);
							approx_builtT1[i][k] = Ts1[i][k].size() - 1;
						}
					}
					else
					{
						if (approx == 0)
							calc_v(1,Ts1[i][k].size() - 2, &Ts1[i][k][0], &V, 0.5, -0.5);
						else
						{
							for (int a = approx_builtT1[i][k];a <= Ts1[i][k].size() - 2;a++)
								calc_va(1, a+approx_clearedT1[i][k], &Ts1[i][k][0]-approx_clearedT1[i][k], &V, NULL, TappH[i][k].ias, TappH[i][k].ibks, 0.5, -0.5);
							approx_builtT1[i][k] = Ts1[i][k].size() - 1;
						}
					}
				}
				Om[i][k] = fT*(tpb*V- tb2*U[i][k]);
			}
	}
	void Om1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB)
	{
#pragma omp parallel for
		for (int k=j0;k<j1;k++)
			for (int i = i0;i < i1;i++)
			{
				double V=0.0;
				if (Ts2[k][i].size()!=1)
				{
					if (varying_time_points)
					{
						if (approx == 0)
							calc_v_vt(1, Ts2[k][i].size() - 2, &Ts2[k][i][0], &V,  &time_points[0],1.0, 0.5);
						else
						{
							for (int a = approx_builtT2[k][i];a <= Ts2[k][i].size() - 2;a++)
								calc_va(1, a + approx_clearedT2[k][i], &Ts2[k][i][0] - approx_clearedT2[k][i], &V, &time_points[0], TappV[k][i].ias, TappV[k][i].ibks, 1.0, 0.5);
							approx_builtT2[k][i] = Ts2[k][i].size() - 1;
						}
					}
					else
					{
						if (approx == 0)
							calc_v(1, Ts2[k][i].size() - 2, &Ts2[k][i][0], &V, 1.0, 0.5);
						else
						{
							for (int a = approx_builtT2[k][i];a <= Ts2[k][i].size() - 2;a++)
								calc_va(1, a+approx_clearedT2[k][i], &Ts2[k][i][0]-approx_clearedT2[k][i], &V,NULL, TappV[k][i].ias, TappV[k][i].ibks, 1.0, 0.5);
							approx_builtT2[k][i] = Ts2[k][i].size() - 1;
						}
					}
				}
				Om[k][i] = 2.0*fT*(tpb*V -0.5*tb2*U[k][i]);
			}
	}
	// T equation beta coeffients
	void bt1_h(int i0=1,int i1=N,int j0=1,int j1=M)
	{
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			Bt[1][j] = 1;
			for (int i = i0;i < i1;i++)
			{
				double R=(V_v[i][j]/h1)*((lambda/h1)-0.5*mu);
				double G=(V_v[i][j]/h1)*((lambda/h1)+0.5*mu);
				Bt[i + 1][j] = (Al[i + 1][j]/R) * (G*Bt[i][j] - Om[i][j]);
			}
		}
	}
	void bt1_v(int i0=1,int i1=M,int j0=1,int j1=N)
	{
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
#ifdef Tc1		
			Bt[j][1] = omega;
#else
			Bt[j][1]=0.0;
#endif
			for (int i = i0;i < i1;i++)
			{
				double _M=lambda*V_v[j][i]/(h2*h2);
				Bt[j][i + 1] = Al[j][i + 1] * (Bt[j][i] - Om[j][i]/_M);
			}
		}
	}
	// calc T
	void U1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB,int only_f=0)
	{
		if (only_f==0)
		{
			A = 1-beta;
			al1_h(i0,i1,j0,j1);
			Om1_h(approx,i0,i1,j0,j1,b0,b1);
			bt1_h(i0,i1,j0,j1);
		}		
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			U[N][j] = Bt[N][j]/(1.0-Al[N][j]);
			U[0][j] = 1.0;
			for (int i = i1-1;i >= i0;i--)
				U[i][j] = Al[i + 1][j] * U[i + 1][j] + Bt[i + 1][j];
		}	
	}
	void U1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB,int only_f=0)
	{
		if (only_f==0)
		{
			A = 1-beta;
			al1_v(i0,i1,j0,j1);
			Om1_v(approx,i0,i1,j0,j1,b0,b1);
			bt1_v(i0,i1,j0,j1);
		}
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
#ifdef Tc1
			U[j][M] = omega;
			U[j][0] = omega;
#else
			U[j][M] = Bt[j][M]/(1.0-Al[j][M]);
#endif
			for (int i = i1-1;i >= i0;i--)
				U[j][i] = Al[j][i + 1] * U[j][i + 1] + Bt[j][i + 1];
		}
	}
	// V funcs are empty
	void V1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB)
	{
	}
	void V1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB)
	{
	}
	// C equation alpha coefficient
	void al2_h(int i0=1,int i1=N,int j0=1,int j1=M)
	{
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			Al[1][j] = 0;
			for (int i = i0;i < i1;i++)
			{
				double A=(V_v[i][j]/h1)*((d/h1)-0.5);
				double S=(V_v[i][j]/h1)*((d/h1)+0.5);
				Al[i + 1][j] = A / ((bC+A+S) - S*Al[i][j]);
			}
		}
	}
	void al2_v(int i0=1,int i1=M,int j0=1,int j1=N)
	{
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			Al[j][1] = 1;
			for (int i = i0;i < i1;i++)
			{
				double P=d*V_v[j][i]/(h2*h2);
				Al[j][i + 1] = P / ((bC+2.0*P) - P*Al[j][i]);
			}
		}
	}
	// C equation right part
	void Om2_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB)
	{
#pragma omp parallel for
		for (int k=j0;k<j1;k++)
			for (int i = i0;i < i1;i++)
			{
				double V=0.0;
				if (Cs1[i][k].size()!=1)
				{
					if (varying_time_points)
					{
						if (approx == 0)
							calc_v_vt(1, Cs1[i][k].size() - 2, &Cs1[i][k][0], &V, &time_points[0], 0.5, -0.5);
						else
						{
							for (int a = approx_builtC1[i][k];a <= Cs1[i][k].size() - 2;a++)
								calc_va(1, a + approx_clearedC1[i][k], &Cs1[i][k][0] - approx_clearedC1[i][k], &V, &time_points[0], CappH[i][k].ias, CappH[i][k].ibks, 0.5, -0.5);
							approx_builtC1[i][k] = Cs1[i][k].size() - 1;
						}
					}
					else
					{
						if (approx == 0)
							calc_v(1, Cs1[i][k].size() - 2, &Cs1[i][k][0], &V, 0.5, -0.5);
						else
						{
							for (int a = approx_builtC1[i][k];a <= Cs1[i][k].size() - 2;a++)
								calc_va(1,a+approx_clearedC1[i][k], &Cs1[i][k][0]-approx_clearedC1[i][k], &V,NULL, CappH[i][k].ias, CappH[i][k].ibks, 0.5, -0.5);
							approx_builtC1[i][k] = Cs1[i][k].size() - 1;
						}
					}
				}
				Om[i][k] = fC*(tpa*V-ta2*C[i][k])-(dt*V_v[i][k]/(h1*h1))*(U[i-1][k]-2.0*U[i][k]+U[i+1][k]);
			}
	}
	void Om2_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB)
	{
#pragma omp parallel for
		for (int k=j0;k<j1;k++)
			for (int i = i0;i < i1;i++)
			{
				double V=0.0;
				if (Cs2[k][i].size()!=1)
				{
					if (varying_time_points)
					{
						if (approx == 0)
							calc_v_vt(1, Cs2[k][i].size() - 2, &Cs2[k][i][0], &V, &time_points[0], 1.0, 0.5);
						else
						{
							for (int a = approx_builtC2[k][i];a <= Cs2[k][i].size() - 2;a++)
								calc_va(1, a + approx_clearedC2[k][i], &Cs2[k][i][0] - approx_clearedC2[k][i], &V, &time_points[0], CappV[k][i].ias, CappV[k][i].ibks, 1.0, 0.5);
							approx_builtC2[k][i] = Cs2[k][i].size() - 1;
						}
					}
					else
					{
						if (approx == 0)
							calc_v(1, Cs2[k][i].size() - 2, &Cs2[k][i][0], &V, 1.0, 0.5);
						else
						{
							for (int a = approx_builtC2[k][i];a <= Cs2[k][i].size() - 2;a++)
								calc_va(1, a+approx_clearedC2[k][i], &Cs2[k][i][0]-approx_clearedC2[k][i], &V,NULL, CappV[k][i].ias, CappV[k][i].ibks, 1.0, 0.5);
							approx_builtC2[k][i] = Cs2[k][i].size() - 1;
						}
					}
				}
				Om[k][i] = 2.0*fC*(tpa*V-0.5*ta2*C[k][i])-(dt*V_v[k][i]/(h2*h2))*(U[k][i-1]-2.0*U[k][i]+U[k][i+1]);
			}
	}
	void bt2_h(int i0=1,int i1=N,int j0=1,int j1=M)
	{
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			Bt[1][j] = 1;
			for (int i = i0;i < i1;i++)
			{
				double A=(V_v[i][j]/h1)*((d/h1)-0.5);
				double S=(V_v[i][j]/h1)*((d/h1)+0.5);
				Bt[i + 1][j] = (Al[i + 1][j]/A) * (S*Bt[i][j] - Om[i][j]);
			}
		}
	}
	void bt2_v(int i0=1,int i1=M,int j0=1,int j1=N)
	{
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			Bt[j][1] = 0;
			for (int i = i0;i < i1;i++)
			{
				double P=d*V_v[j][i]/(h2*h2);
				Bt[j][i + 1] = Al[j][i + 1] * (Bt[j][i] - Om[j][i]/P);
			}
		}
	}
	// calc C
	void C1_h(int approx,int i0=1,int i1=N,int j0=1,int j1=M,int b0=0,int b1=NB,int only_f=0)
	{
		if (only_f==0)
		{
			A = 1 - alpha;
			al2_h(i0,i1,j0,j1);
			Om2_h(approx,i0,i1,j0,j1,b0,b1);
			bt2_h(i0,i1,j0,j1);
		}
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			C[N][j] = Bt[N][j]/(1.0-Al[N][j]);
			C[0][j] = 1.0;
			for (int i = i1-1;i >= i0;i--)
				C[i][j] = Al[i + 1][j] * C[i + 1][j] + Bt[i + 1][j];
		}
	}
	void C1_v(int approx,int i0=1,int i1=M,int j0=1,int j1=N,int b0=0,int b1=NB,int only_f=0)
	{
		if (only_f==0)
		{
			A = 1 - alpha;
			al2_v(i0,i1,j0,j1);
			Om2_v(approx,i0,i1,j0,j1,b0,b1);
			bt2_v(i0,i1,j0,j1);
		}
#pragma omp parallel for
		for (int j=j0;j<j1;j++)
		{
			C[j][M] = Bt[j][M]/(1.0-Al[j][M]);
			for (int i = i1-1;i >= i0;i--)
				C[j][i] = Al[j][i + 1] * C[j][i + 1] + Bt[j][i + 1];
		}
#ifdef T_DEBUG
		double cmax[MAXAK],cmin[MAXAK],csum[MAXAK];
		int c1max=0,c2max=0;
		int f[MAXAK],cn[MAXAK];
		for (int i=0;i<MAXAK;i++)
		{
		f[i]=1;
		csum[i]=0.0;
		cn[i]=0.0;
		}
		for (int i=0;i<N+2;i++)
		for (int j=0;j<M+2;j++)
		{
		  if(TappH[i][j].ias.size()>c1max) c1max=TappH[i][j].ias.size();
		  if(TappV[i][j].ias.size()>c1max) c1max=TappV[i][j].ias.size();
		  if(CappH[i][j].ias.size()>c1max) c1max=CappH[i][j].ias.size();
		  if(CappV[i][j].ias.size()>c1max) c1max=CappV[i][j].ias.size();
		  if(TappH[i][j].ibks.size()>c2max) c2max=TappH[i][j].ibks.size();
		  if(TappV[i][j].ibks.size()>c2max) c2max=TappV[i][j].ibks.size();
		  if(CappH[i][j].ibks.size()>c2max) c2max=CappH[i][j].ibks.size();
		  if(CappV[i][j].ibks.size()>c2max) c2max=CappV[i][j].ibks.size();
#define stat(v) for(int k=0;k<v.size();k++)\
		    for (int k1=0;k1<v[k].nterms;k1++) {\
		    if (f[k1] || (v[k].c[k1]< cmin[k1])) cmin[k1]=v[k].c[k1];\
		    if (f[k1] || (v[k].c[k1]> cmax[k1])) cmax[k1]=v[k].c[k1];\
		    csum[k1]+=v[k].c[k1];\
		    cn[k1]++;\
		    f[k1]=0;\
		    }
		   stat(TappH[i][j].ias);
		   stat(TappV[i][j].ias);
		   stat(CappH[i][j].ias);
		   stat(CappV[i][j].ias);
		   stat(TappH[i][j].ibks);
		   stat(TappV[i][j].ibks);
		   stat(CappH[i][j].ibks);
		   stat(CappV[i][j].ibks);
#undef stat
		}
		printf("ias max %d ibks max %d\n",c1max,c2max);
		for(int k=0;k<MAXAK;k++)
		if(cn[k])
		    printf("%d min %g avg %g max %g n %d\n",k,cmin[k],csum[k]/cn[k],cmax[k],cn[k]);
#endif
	}
	void solve_step(int approx)
	{
		U1_h(approx);
		//save_U();
		U1_v(approx);
		save_U();
		V1_h(approx);
		V1_v(approx);
		C1_h(approx);
		//save_C();
        	C1_v(approx);
		save_C();
	}
};
// solves s1 with step tau, s2 - two steps with step tau/2, saves initial s1 into s3 and returns ||s1-s2||
double solve_T_half_step_and_check(space_fract_solver_T_2d *s1,space_fract_solver_T_2d *s2,space_fract_solver_T_2d *s3,int approx)
{
	double err1,err2;
	s3->copy(s1);
	// calc s1 - 1 step tau
	s1->solve_step(approx);
	// calc s2 - 2 step tau/2
	s2->tau = s1->tau/2;
	s2->solve_step(approx);
	s2->solve_step(approx);
	// calc err=||s1-s2||
	err1=err2=0.0;
	for (int i = 1;i < N ;i++)
	for (int j = 1;j < M ;j++)
	{
		err1 += (s2->U[i][j] - s1->U[i][j])*(s2->U[i][j] - s1->U[i][j]);
		err2 += (s2->C[i][j] - s1->C[i][j])*(s2->C[i][j] - s1->C[i][j]);
	}
	return err1+err2;
}
void solve_T_var_time_step(double t,double save_tau,double eps=1e-4,double _a=0.8,double _b=0.9,int compare_approx=0,int approx=0,int compare_fixed=1)
{
	long long i2,i3;
	space_fract_solver_T_2d *s1,*s2,*s3,*s4,*s5;
	int nsave = 1;
	FILE *fi1;
	fi1 = fopen("log.txt", "wt");
	// main solver
	s1=new space_fract_solver_T_2d();
	s1->init(0,_a,_b);
	s1->varying_time_points=1;
	// half time step solver
	s2=new space_fract_solver_T_2d();
	s2->init(0,_a,_b);
	s2->varying_time_points=1;
	// solver for saving
	s3=new space_fract_solver_T_2d();
	s3->init(0,_a,_b);
	s3->varying_time_points=1;
	// solver with equal time-steps for comparison
	s4=new space_fract_solver_T_2d();
	s4->init(0,_a,_b);
	//s4->reset_tau(s1->tau*2);
	// solver for comparing with approximated varying time step
	if (compare_approx)
	{
		s5 = new space_fract_solver_T_2d();
		s5->init(0, _a, _b);
		s5->varying_time_points = 1;
	}
	i2=GetTickCount();
	while (s1->T<t)
	{
		double err;
		long long i1;
		i1 = GetTickCount();
		// do initial check
		err=solve_T_half_step_and_check(s1,s2,s3,approx);
		printf("1 %g\n",err);
		// if err>eps - decrease step until err>eps, otherwise - increase step
		if (err>eps)
		{
			while (err>eps)
			{
				//restore s1,s2
				s1->copy(s3);
				s2->copy(s3);
				// decrease tau
				s1->tau/=2.0;
				// do check
				err=solve_T_half_step_and_check(s1,s2,s3,approx);
				printf("2 %g %g\n",s1->tau,err);
			}
		}
		else
		{
			while (err<=eps)
			{
				//restore s1,s2
				s1->copy(s3);
				s2->copy(s3);
				// increase tau
				s1->tau*=2.0;
				// do check
				err=solve_T_half_step_and_check(s1,s2,s3,approx);
				printf("3 %g %g\n",s1->tau,err);
				if (s1->tau>1e5)
					break;
			}
		}
		if (s1->T >= nsave*save_tau)
		{
			double stau=s1->tau;
			s1->copy(s3);
			s2->copy(s3);
			s1->tau=nsave*save_tau-s1->T;
			if (s1->tau>1e-20)
				solve_T_half_step_and_check(s1,s2,s3,approx);
			else
				s1->T+=s1->tau;
			s1->tau=stau;
			printf("4 %g %g\n",s1->tau,err);
		}
		if (compare_approx)
		{
			s5->tau = s1->tau;
			if (s1->T >= nsave*save_tau)
				s5->tau = nsave*save_tau - s3->T;
			s5->solve_step(1);
			s5->tau = s1->tau;
		}
		i1 = GetTickCount()-i1;
		s1->T+=s1->tau;
		fprintf(fi1, "t %g time0 %lld tau %g\n", s1->T, i1, s1->tau);
		printf("t %7.7f tau %9.9f ", s1->T, s1->tau);
		for (int i = 0;i < N;i+=4)
			printf("%1.1f ", s1->C[i][M / 2]);
		printf("\n");
		fflush(stdout);
		if (compare_approx)
		{
			double err3 = 0.0, err4 = 0.0;
			for (int i = 1;i < N;i++)
				for (int j = 1;j < M;j++)
				{
					err3 += (s1->U[i][j] - s5->U[i][j])*(s1->U[i][j] - s5->U[i][j]);
					err4 += (s1->C[i][j] - s5->C[i][j])*(s1->C[i][j] - s5->C[i][j]);
				}
			printf("err3 %g err4 %g\n", err3, err4);
			fflush(stdout);
		}
		// comparison and result saving
		if (s1->T >= nsave*save_tau-1e-10)
		{
			i3=GetTickCount()-i2; // varying time step solving time
		        double err1=0.0,err2=0.0;
			if (compare_fixed)
			{
			    i2=GetTickCount();
			    for (double tt = 0;tt <= save_tau-1e-10 ;tt += s4->tau)
			    {
				printf(".");
				s4->solve_step(approx);
			    }
			    printf("\n");
			    i2=GetTickCount()-i2; // fixed time step solving time
			    // compare
			    for (int i = 1;i < N ;i++)
			    for (int j = 1;j < M ;j++)
			    {
			        double e1=(s4->U[i][j] - s1->U[i][j])*(s4->U[i][j] - s1->U[i][j]);
			        double e2=(s4->C[i][j] - s1->C[i][j])*(s4->C[i][j] - s1->C[i][j]);
				err1 +=e1 ;
				err2 += e2;
			    }
			}    
			fprintf(fi1, "T %g var_t time %lld fixed t time %lld err1 %g err2 %g\n", s1->T, i3, i2, err1, err2);
			printf("T %g var_t time %lld fixed t time %lld err1 %g err2 %g\n", s1->T, i3, i2, err1, err2);
			fflush(stdout);
			i2=GetTickCount();
			// save 
			//save_and_convert("Uva",nsave,s1->alpha,s1->beta,s1->dt,s1->U);
			//save_and_convert("Cva",nsave, s1->alpha, s1->beta, s1->dt, s1->C);
			if (compare_fixed)
			{
			    save_and_convert("Ufa2",nsave,s4->alpha,s4->beta,s4->dt,s4->U);
			    save_and_convert("Cfa2",nsave, s4->alpha, s4->beta, s4->dt, s4->C);
			}
			nsave++;
		}
	}
}
////////////////////////////////////////////////////////////////////////////////////////
//////////////////// parallel algorithms and solving procedures /////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
void save_and_convert(char *name,int id,double a,double b,double k,double (*d)[M+2])
{
	char str[1024], tiff_name[1024], jpg_name[1024],txt_name[1024];
	char cmdline[1024];
	FILE *fi;
	sprintf(str, "%s_%g_%g_%g_%d.raw", name, a,b,k,id);
	sprintf(tiff_name, "%s_%g_%g_%g_%d.tiff", name, a,b,k,id);
	sprintf(jpg_name, "%s_%g_%g_%g_%d.jpg", name, a,b,k,id);
	sprintf(txt_name, "%s_%g_%g_%g_%d.txt", name, a,b,k,id);
	fi = fopen(str, "wb");
	for (int i = 0;i<N + 2;i++)
		fwrite(d[i], sizeof(double), M + 2, fi);
	fclose(fi);
	fi=fopen(txt_name,"wt");
	for (int i = 0;i<N + 2;i++)
		fprintf(fi,"%d %lg\n",i,d[i][M/2]);
	fclose(fi);
#ifndef unix
	{
		PROCESS_INFORMATION processInformation;
		STARTUPINFO startupInfo;
		memset(&processInformation, 0, sizeof(processInformation));
		memset(&startupInfo, 0, sizeof(startupInfo));
		startupInfo.cb = sizeof(startupInfo);
		sprintf(cmdline, "tiff/raw2tiff.exe -d double -w %d -l %d %s %s", M + 2, N + 2, str, tiff_name);
		// run convertor
		if (CreateProcess(NULL, cmdline, NULL, NULL, FALSE, NORMAL_PRIORITY_CLASS, NULL, NULL, &startupInfo, &processInformation))
		{
			// wait for convertor to finish
			WaitForSingleObject(processInformation.hProcess, INFINITE);
			CloseHandle(processInformation.hProcess);
			CloseHandle(processInformation.hThread);
			{
				PROCESS_INFORMATION processInformation;
				STARTUPINFO startupInfo;
				memset(&processInformation, 0, sizeof(processInformation));
				memset(&startupInfo, 0, sizeof(startupInfo));
				startupInfo.cb = sizeof(startupInfo);
				sprintf(cmdline, "tiff/convert.exe %s %s", tiff_name, jpg_name);
				// run convertor
				if (CreateProcess(NULL, cmdline, NULL, NULL, FALSE, NORMAL_PRIORITY_CLASS, NULL, NULL, &startupInfo, &processInformation))
				{
					// wait for convertor to finish
					WaitForSingleObject(processInformation.hProcess, INFINITE);
					CloseHandle(processInformation.hProcess);
					CloseHandle(processInformation.hThread);
				}
			}
		}
	}
#endif
}
void solve_2d(double t,double save_tau,int ocl=0,int problem=0,double _a=0.8,double _b=0.9,int approx=0,int solve=3)
{
	space_fract_solver_2d *s1, *s2;
	int nsave = 1;
	FILE *fi1;
	fi1 = fopen("log.txt", "wt");
	if (problem==0)
	{
		s1=new space_fract_solver_HVC_2d();
		s2=new space_fract_solver_HVC_2d();
	}
	else
	{
		s1=new space_fract_solver_T_2d();
		s2=new space_fract_solver_T_2d();
	}
	s1->init(ocl,_a,_b);
	s2->init(0,_a,_b);
	for (double tt = 0;tt < t;tt += s1->tau)
	{
		double err1=0.0,err2=0.0,err3=0.0;
		long long i1, i2;
		i1 = GetTickCount();
		if (solve&1)
		{
		    s1->U1_h(approx);
		    s1->save_U();
		    s1->U1_v(approx);
		    s1->save_U();
		    s1->V1_h(approx);
		    s1->V1_v(approx);
		    s1->C1_h(approx);
		    s1->save_C();
    		s1->C1_v(approx);
		    s1->save_C();
		}
		i1 = GetTickCount()-i1;
		i2 = GetTickCount();
		if (solve&2)
		{
		    s2->U1_h(1);
		    s2->save_U();
		    s2->U1_v(1);
		    s2->save_U();
		    s2->V1_h(1);
		    s2->V1_v(1);
		    s2->C1_h(1);
		    s2->save_C();
		    s2->C1_v(1);
		    s2->save_C();
		}
		i2 = GetTickCount() - i2;
		// calculate errors
		if (ocl)
		{
			s1->fr_ocl_get(s1->U,0,N+2,0,M+2);
			s1->fr_ocl_get(s1->C,0,N+2,0,M+2);
			s1->fr_ocl_get(s1->V_v,0,N+2,0,M+2);
			s1->fr_ocl_get(s1->V_h,0,N+2,0,M+2);
			s1->fr_ocl_get(s1->Al,0,N+2,0,M+2);
			s1->fr_ocl_get(s1->Bt,0,N+2,0,M+2);
			s1->fr_ocl_get(s1->Om,0,N+2,0,M+2);
		}
		if (solve==3)
		for (int i = 1;i < N ;i++)
		for (int j = 1;j < M ;j++)
		{
			err1 += (s2->U[i][j] - s1->U[i][j])*(s2->U[i][j] - s1->U[i][j]);
			err2 += (s2->V_h[i][j] - s1->V_h[i][j])*(s2->V_h[i][j] - s1->V_h[i][j]);
			err2 += (s2->V_v[i][j] - s1->V_v[i][j])*(s2->V_v[i][j] - s1->V_v[i][j]);
			err3 += (s2->C[i][j] - s1->C[i][j])*(s2->C[i][j] - s1->C[i][j]);
		}
		fprintf(fi1, "t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, err1, err2, err3);
		printf("t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, err1, err2, err3);
		fflush(stdout);
		// save result
		if ((tt+ s1->tau) >= nsave*save_tau)
		{
			save_and_convert("U",nsave,s1->alpha,s1->beta,((problem==0)?0.0:(((space_fract_solver_T_2d*)s1)->dt)),s1->U);
			save_and_convert("Vh",nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_h);
			save_and_convert("Vv",nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_v);
			save_and_convert("C",nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->C);
			nsave++;
		}
	}
}
///////////////////////////////////////
///////// 1d-parallel 2d solver /////// 
///////////////////////////////////////
int nsends=0;
unsigned int time_send=0;
unsigned int time_time;
// dir 0 - row to col, dir 1 - col to row
void transpose(space_fract_solver_2d *solver, int r, int np, double(*U)[M + 2],int i0,int i1, int j0, int j1, int dir, int b0 = 0, int b1 = 0, int poff = 0, int ulx = 0, int uly = 0, int debug = 0)
{
	static MPI_Request *s = NULL;
	MPI_Status st;
	int nr = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	time_time = GetTickCount();
	if ((b1 == 0) || (b1>np)) b1 = np;
	if (s == NULL) s = new MPI_Request[2 * N*np];
	if (debug) printf("transpose %d %d %d %d %d %d %d %d %d ulx %d uly %d\n", r, i0,i1,j0, j1, dir, b0, b1, poff, ulx, uly);
	if (solver->use_ocl)
	{
		int ii0;
		int ii1;
		if (dir == 1)
		{
			ii0 = (int)((N / (double)np)*b0) + 1;
			ii1 = (int)((N / (double)np)*b1) + 1;
			if (b1 == np)
				ii1 = N + 1;
		}
		else
		{
			ii0 = (int)((M / (double)np)*b0) + 1;
			ii1 = (int)((M / (double)np)*b1) + 1;
			if (b1 == np)
				ii1 = M + 1;
		}
		if (dir == 0)
			solver->fr_ocl_get(U, j0, j1 + ((j1 == N) ? 1 : 0), ii0, ii1);
		else
			solver->fr_ocl_get(U, ii0, ii1, j0, j1 + ((j1 == M) ? 1 : 0));
	}
	for (int rr = b0;rr<b1;rr++)
		if ((rr + poff) != r)
			if ((rr + poff)<np)
			{
				int rj0;
				int rj1;
				if (dir == 1)
				{
					rj1 = (int)((N / (double)np)*(rr + 1)) + 1;
					rj0 = (int)((N / (double)np)*rr) + 1;
					if (rr == (np - 1))
						rj1 = N + 1;
				}
				else
				{
					rj1 = (int)((M / (double)np)*(rr + 1)) + 1;
					rj0 = (int)((M / (double)np)*rr) + 1;
					if (rr == (np - 1))
						rj1 = M + 1;
				}
				if (dir == 0)
					for (int j = j0;j<j1 + ((j1 == N) ? 1 : 0);j++)
					{
						MPI_Isend(U[j] + rj0, (rj1 - rj0)*sizeof(double), MPI_BYTE, rr + poff, 0, MPI_COMM_WORLD, &s[nr++]);
						nsends += (rj1 - rj0)*sizeof(double);
						if (debug)
							for (int k = rj0;k < rj1;k++)
								printf("%d ts %d f[%d][%d]=%g\n", r, rr + poff, j, k, U[j][k]);
					}
				else
					for (int j = rj0;j<rj1;j++)
					{
						MPI_Isend(U[j] + j0, (j1 - j0 + ((j1 == M) ? 1 : 0))*sizeof(double), MPI_BYTE, rr + poff, 0, MPI_COMM_WORLD, &s[nr++]);
						nsends += (j1 - j0 + ((j1 == N) ? 1 : 0))*sizeof(double);
						if (debug)
							for (int k = j0;k < j1 + ((j1 == N) ? 1 : 0);k++)
								printf("%d ts %d f[%d][%d]=%g\n", r, rr + poff, j, k, U[j][k]);
					}
			}
	if (dir == 0)
	{
		if (i1 == M) i1++;
	}
	else
	{
		if (i1 == N) i1++;
	}
	i0 = ulx + i0 - uly;
	i1 = ulx + i1 - uly;
	for (int rr = b0;rr<b1;rr++)
		if ((rr + poff) != r)
			if ((rr + poff)<np)
			{
				int rj0;
				int rj1;
				if (dir == 0)
				{
					rj0 = (int)((N / (double)np)*rr) + 1;
					rj1 = (int)((N / (double)np)*(rr + 1)) + 1;
					if (rr == (np - 1))
						rj1 = N + 1;
				}
				else
				{
					rj0 = (int)((M / (double)np)*rr) + 1;
					rj1 = (int)((M / (double)np)*(rr + 1)) + 1;
					if (rr == (np - 1))
						rj1 = M + 1;
				}
				rj0 = uly + rj0 - ulx;
				rj1 = uly + rj1 - ulx;
				if (dir == 0)
					for (int j = rj0;j < rj1;j++)
					{
						if (debug == 0)
							MPI_Irecv(U[j] + i0, (i1 - i0)*sizeof(double), MPI_BYTE, rr + poff, 0, MPI_COMM_WORLD, &s[nr++]);
						else
						{
							MPI_Recv(U[j] + i0, (i1 - i0)*sizeof(double), MPI_BYTE, rr + poff, 0, MPI_COMM_WORLD, &st);
							for (int k = i0;k < i1;k++)
								printf("%d tr %d f[%d][%d]=%g\n", r, rr + poff, j, k, U[j][k]);
						}
					}
				else
					for (int j = i0;j < i1;j++)
					{
						if (debug == 0)
							MPI_Irecv(U[j] + rj0, (rj1 - rj0)*sizeof(double), MPI_BYTE, rr + poff, 0, MPI_COMM_WORLD, &s[nr++]);
						else
						{
							MPI_Recv(U[j] + rj0, (rj1 - rj0)*sizeof(double), MPI_BYTE, rr + poff, 0, MPI_COMM_WORLD, &st);
							for (int k = rj0;k < rj1;k++)
								printf("%d tr %d f[%d][%d]=%g\n", r, rr + poff, j, k, U[j][k]);
						}
					}
			}
	for (int i = 0;i < nr;i++)
		MPI_Wait(&s[i], &st);
	if (solver->use_ocl)
	{
		int ii0;
		int ii1;
		if (dir == 0)
		{
			ii0 = (int)((N / (double)np)*b0) + 1;
			ii1 = (int)((N / (double)np)*b1) + 1;
			if (b1 == np)
				ii1 = N + 1;
		}
		else
		{
			ii0 = (int)((M / (double)np)*b0) + 1;
			ii1 = (int)((M / (double)np)*b1) + 1;
			if (b1 == np)
				ii1 = M + 1;
		}
		ii0 = uly + ii0 - ulx;
		ii1 = uly + ii1 - ulx;
		if (dir == 0)
			solver->fr_ocl_put(U, ii0, ii1, i0, i1);
		else
			solver->fr_ocl_put(U, i0, i1, ii0, ii1);
	}
	time_send += GetTickCount() - time_time;
}
// each process owns [j0,j1] block of columns, does vertical calculations, 
// then processes do transposition of matrix ownership so that each process owns [j0,j1] block of rows,
// then each process does horizontal calculations
void solve_2d_par1(double t, double save_tau, int approx, int ocl,int problem=0)
{
	space_fract_solver_2d *s1, *s2;
	int nsave = 1;
	int np;
	int r;
	int j0, j1,i0,i1;
	FILE *fi1;
	MPI_Status mpi_status;
	MPI_Comm_rank(MPI_COMM_WORLD, &r);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	if (problem==0)
		s1 = new space_fract_solver_HVC_2d();
	else
		s1 = new space_fract_solver_T_2d();
	if (r == 0)
	{
		fi1 = fopen("log.txt", "wt");
		if (problem==0)
			s2 = new space_fract_solver_HVC_2d();
		else
			s2 = new space_fract_solver_T_2d();
	}
	s1->init(ocl);
	if (r == 0)
		s2->init();
	j0 = (int)((N / (double)np)*r) + 1;
	j1 = (int)((N / (double)np)*(r + 1)) + 1;
	if (r == (np - 1))
		j1 = N;
	i0 = (int)((M / (double)np)*r) + 1;
	i1 = (int)((M / (double)np)*(r + 1)) + 1;
	if (r == (np - 1))
		i1 = M;
	printf("rank %d np %d j0 %d j1 %d N %d i0 %d i1 %d M %d\n", r, np, j0, j1, N,i0,i1,M);fflush(stdout);
	for (double tt = 0;tt < t;tt += s1->tau)
	{
		double err1 = 0.0, err2 = 0.0, err3 = 0.0;
		long long _i1, i2;
		nsends = 0;
		time_send = 0;
		_i1 = GetTickCount();
		transpose(s1, r, np, s1->U,i0,i1, j0, j1, 0);
		s1->U1_h(approx, 1, N, i0, i1);
		transpose(s1, r, np, s1->U, j0, j1,  i0, i1, 1);
		s1->save_U();
		s1->U1_v(approx, 1, M, j0, j1);
		transpose(s1, r, np, s1->U, i0, i1, j0, j1, 0);
		s1->save_U();
		s1->V1_h(approx, 1, N, i0, i1);
		transpose(s1, r, np, s1->V_h, j0, j1, i0, i1, 1);
		transpose(s1, r, np, s1->U, j0, j1, i0, i1, 1);
		s1->V1_v(approx, 1, M, j0, j1);
		transpose(s1, r, np, s1->C, i0, i1, j0, j1, 0);
		if (tt!=0)
			s1->save_C();
		s1->C1_h(approx, 1, N, i0, i1);
		transpose(s1, r, np, s1->C, j0, j1, i0, i1, 1);
		s1->save_C();
		s1->C1_v(approx, 1, M, j0, j1);
		_i1 = GetTickCount() - _i1;
		if (ocl)
		{
			s1->fr_ocl_get(s1->U, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->C, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->V_v, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->V_h, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Al, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Bt, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Om, 0, N + 2, 0, M + 2);
		}
		printf("%d nsends %d tsends %d\n", r, nsends, time_send);fflush(stdout);
		if (r == 0)
		{
			i2 = GetTickCount();
			s2->U1_h(0);
			s2->save_U();
			s2->U1_v(0);
			s2->save_U();
			s2->V1_h(0);
			s2->V1_v(0);
			s2->C1_h(0);
			s2->save_C();
			s2->C1_v(0);
			s2->save_C();
			i2 = GetTickCount() - i2;
			// gather all
			for (int rr = 1;rr<np;rr++)
			{
				int rj0 = (int)((N / (double)np)*rr) + 1;
				int rj1 = (int)((N / (double)np)*(rr + 1)) + 1;
				if (rr == (np - 1))
					rj1 = N + 1;
				for (int j = rj0;j<rj1;j++)
				{
					MPI_Recv(s1->U[j], (M + 2)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
					MPI_Recv(s1->V_h[j], (M + 2)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
					MPI_Recv(s1->V_v[j], (M + 2)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
					MPI_Recv(s1->C[j], (M + 2)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
				}
			}
			// calculate errors			
			for (int i = 1;i < N ;i++)
				for (int j = 1;j < M ;j++)
				{
					err1 += (s2->U[i][j] - s1->U[i][j])*(s2->U[i][j] - s1->U[i][j]);
					err2 += (s2->V_h[i][j] - s1->V_h[i][j])*(s2->V_h[i][j] - s1->V_h[i][j]);
					err2 += (s2->V_v[i][j] - s1->V_v[i][j])*(s2->V_v[i][j] - s1->V_v[i][j]);
					err3 += (s2->C[i][j] - s1->C[i][j])*(s2->C[i][j] - s1->C[i][j]);
				}
			fprintf(fi1, "t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, _i1, i2, err1, err2, err3);
			printf("t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, _i1, i2, err1, err2, err3);
			fflush(stdout);
			// save result
			if ((tt + s1->tau) >= nsave*save_tau)
			{
				save_and_convert("U", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->U);
				save_and_convert("Vh", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_h);
				save_and_convert("Vv", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_v);
				save_and_convert("C", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->C);
				nsave++;
			}
		}
		else // send all to r0
		{
			for (int j = j0;j<j1 + ((j1 == N) ? 1 : 0);j++)
			{
				MPI_Send(s1->U[j], (M + 2)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
				MPI_Send(s1->V_h[j], (M + 2)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
				MPI_Send(s1->V_v[j], (M + 2)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
				MPI_Send(s1->C[j], (M + 2)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
			}
		}
	}
	if (r == 0)
	{
		delete s2;
		fclose(fi1);
	}
	delete s1;
}
///////////////////////////////////////
///////// 2d-parallel 2d solver /////// 
///////////////////////////////////////
void space_fract_solver_HVC_2d::exchange_approx(int r, int np, int block, std::vector<int> &j0s, std::vector<int> &j1s, int xy, int poffmult, int rbmode, int debug)
{
	static MPI_Request s1, s2;
	MPI_Status st;
	int nr = 0;
	char *to_send, *to_recv;
	int s_to_send = 0, s_to_recv = 0;
	int b;
	int rr = (np + r + (1 - 2 * xy)*poffmult) % np, rb;
	int r_to_send;
	MPI_Barrier(MPI_COMM_WORLD);
	time_time = GetTickCount();
	if (rbmode == 0)
	{
		if (xy == 0)
		{
			b = (block + r) % np;
			rb = (block + rr) % np;
		}
		else
		{
			b = (np + block - r) % np;
			rb = (np + block + poffmult - r) % np;
		}
	}
	else
	{
		rb = (np - block - poffmult + r) % np;
		b = (np - block + r) % np;
	}
	// get from GPU
	if (use_ocl)
	{
#ifdef OCL
		// get number of approximations from GPU
		queue->EnqueueBuffer(b_nibk, n_ibk + j0s[b], sizeof(int)*j0s[b], (j1s[b] - j0s[b])*sizeof(int));
		queue->EnqueueBuffer(b_nias, n_ias + j0s[b], sizeof(int)*j0s[b], (j1s[b] - j0s[b])*sizeof(int));
		for (int j = j0s[b];j<j1s[b];j++)
		{
			// clear approximations
			app[j].ias.clear();
			app[j].ibks.clear();
			// get serialized approximations from GPU
			queue->EnqueueBuffer(b_ibk, row_ibks, j*alloced_approx_per_row*sizeof(ibk_approximation), n_ibk[j] * sizeof(ibk_approximation));
			queue->EnqueueBuffer(b_ias, row_ias, j*alloced_approx_per_row* sizeof(ia_approximation), n_ias[j] * sizeof(ia_approximation));
			// unpack
			char *to_unpack = row_ias;
			for (int i = 0;i < n_ias[j];i++, to_unpack += sizeof(ia_approximation))
			{
				ia_approximation *ia = new ia_approximation();
				ia->nterms = ((int *)(to_unpack + ((char *)&(ia->nterms) - (char *)ia)))[0];
				ia->min = ((double *)(to_unpack + ((char *)&(ia->min) - (char *)ia)))[0];
				for (int i = 0;i < MAXAK;i++)
					ia->c[i] = ((double *)(to_unpack + ((char *)&(ia->c[i]) - (char *)ia)))[0];
				app[j].ias.push_back(ia[0]);
			}
			to_unpack = row_ibks;
			for (int i = 0;i < n_ibk[j];i++, to_unpack += sizeof(ibk_approximation))
			{
				ibk_approximation *ibk = new ibk_approximation();
				ibk->nterms = ((int *)(to_unpack + ((char *)&(ibk->nterms) - (char *)ibk)))[0];
				ibk->old = ((int *)(to_unpack + ((char *)&(ibk->old) - (char *)ibk)))[0];
				ibk->min = ((double *)(to_unpack + ((char *)&(ibk->min) - (char *)ibk)))[0];
				ibk->max = ((double *)(to_unpack + ((char *)&(ibk->max) - (char *)ibk)))[0];
				ibk->b = ((double *)(to_unpack + ((char *)&(ibk->b) - (char *)ibk)))[0];
				for (int i = 0;i<MAXAK;i++)
					ibk->c[i] = ((double *)(to_unpack + ((char *)&(ibk->c[i]) - (char *)ibk)))[0];
				app[j].ibks.push_back(ibk[0]);
			}
		}
#endif
	}
	// calculate size of approximations for block of rows
	for (int j = j0s[b];j<j1s[b];j++)
		s_to_send += app[j].ias.size()*ia_approximation::size() + app[j].ibks.size()*ibk_approximation::size();
	to_send = new char[s_to_send + 2 * (j1s[b] - j0s[b])*sizeof(int)];
	s_to_send = 0;
	// pack number of ias and ibks
	for (int j = j0s[b];j<j1s[b];j++)
	{
		*(int *)(to_send + s_to_send) = app[j].ias.size();
		*(int *)(to_send + s_to_send + sizeof(int)) = app[j].ibks.size();
		s_to_send += 2 * sizeof(int);
	}
	// pack approximations
	for (int j = j0s[b];j<j1s[b];j++)
	{
		char *c;
		for (int i = 0;i<app[j].ias.size();i++)
		{
			memcpy(to_send + s_to_send, c = app[j].ias[i].serialize(), ia_approximation::size());
			delete[] c;
			s_to_send += ia_approximation::size();
		}
		for (int i = 0;i<app[j].ibks.size();i++)
		{
			memcpy(to_send + s_to_send, c = app[j].ibks[i].serialize(), ibk_approximation::size());
			delete[] c;
			s_to_send += ibk_approximation::size();
		}
	}
	nsends += s_to_send;
	r_to_send = (np + r + (-1 + 2 * xy)*poffmult) % np;
	if ((rbmode == 1) && (xy == 0))
	{
		int sr = rr;
		rr = r_to_send;
		r_to_send = sr;
	}
	if (debug) printf("exchange approx %d %d %d %d to s %d to r %d b %d rb %d ns %d\n", r, block, xy, poffmult, r_to_send, rr, b, rb, s_to_send);
	// do send
	MPI_Isend((void *)&s_to_send, sizeof(int), MPI_BYTE, r_to_send, 0, MPI_COMM_WORLD, &s1);
	MPI_Isend(to_send, s_to_send, MPI_BYTE, r_to_send, 0, MPI_COMM_WORLD, &s2);
	// receive
	MPI_Recv((void *)&s_to_recv, sizeof(int), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &st);
	to_recv = new char[s_to_recv];
	MPI_Recv(to_recv, s_to_recv, MPI_BYTE, rr, 0, MPI_COMM_WORLD, &st);
	// clear approximations
	for (int j = j0s[rb];j<j1s[rb];j++)
	{
		app[j].ias.clear();
		app[j].ibks.clear();
	}
	// unpack
	s_to_recv = 2 * sizeof(int)*(j1s[rb] - j0s[rb]);
	char *to_unpack = to_recv + s_to_recv;
	for (int j = j0s[rb];j<j1s[rb];j++)
	{
		for (int i = 0;i<((int *)to_recv)[2 * (j - j0s[rb]) + 0];i++, to_unpack += ia_approximation::size())
			app[j].ias.push_back((new ia_approximation(to_unpack))[0]);
		for (int i = 0;i<((int *)to_recv)[2 * (j - j0s[rb]) + 1];i++, to_unpack += ibk_approximation::size())
			app[j].ibks.push_back((new ibk_approximation(to_unpack))[0]);
	}
	// put into GPU memory
	if (use_ocl)
	{
#ifdef OCL
		for (int j = j0s[rb];j < j1s[rb];j++)
		{
			n_ibk[j] = app[j].ibks.size();
			n_ias[j] = app[j].ias.size();
		}
		// put number of approximations
		queue->EnqueueWriteBuffer(b_nibk, n_ibk + j0s[rb], sizeof(int)*j0s[rb], (j1s[rb] - j0s[rb])*sizeof(int));
		queue->EnqueueWriteBuffer(b_nias, n_ias + j0s[rb], sizeof(int)*j0s[rb], (j1s[rb] - j0s[rb])*sizeof(int));
		for (int j = j0s[rb];j<j1s[rb];j++)
		{
			// pack and put into GPU memory
			char *to_pack;
			to_pack = row_ias;
			for (int i = 0;i<app[j].ias.size();i++)
			{
				memcpy(to_pack, &app[j].ias[i], sizeof(ia_approximation));
				to_pack += sizeof(ia_approximation);
			}
			queue->EnqueueWriteBuffer(b_ias, row_ias, j*alloced_approx_per_row* sizeof(ia_approximation), n_ias[j] * sizeof(ia_approximation));
			to_pack = row_ibks;
			for (int i = 0;i<app[j].ibks.size();i++)
			{
				memcpy(to_pack, &app[j].ibks[i], sizeof(ibk_approximation));
				to_pack += sizeof(ibk_approximation);
			}
			queue->EnqueueWriteBuffer(b_ibk, row_ibks, j*alloced_approx_per_row*sizeof(ibk_approximation), n_ibk[j] * sizeof(ibk_approximation));
		}
#endif
	}
	// wait and cleanup
	MPI_Wait(&s1, &st);
	MPI_Wait(&s2, &st);
	delete[] to_send;
	delete[] to_recv;
	time_send += GetTickCount() - time_time;
}
void space_fract_solver_HVC_2d::fr_ocl_clear_approximations()
{
#ifdef OCL
	if (use_ocl)
	{
		for (int i = 0;i < N + 2;i++)
		{
			n_ibk[i] = 0;
			n_ias[i] = 0;
		}
		queue->EnqueueWriteBuffer(b_nibk, n_ibk);
		queue->EnqueueWriteBuffer(b_nias, n_ias);
	}
#endif
}
void space_fract_solver_HVC_2d::clear_approximations()
{
	for (int i = 0;i<N;i++)
	{
		app[i].ias.clear();
		app[i].ibks.clear();
	}
	fr_ocl_clear_approximations();
}
// pdir=1 - forward direction or -1 - backward direction
// if xy=0 - horizontal process
// if xy=1 - vertical process
void exchange_data(space_fract_solver_2d *solver, int r, int np, int block, double(*U)[M + 2], std::vector<int> &i0s, std::vector<int> &i1s, std::vector<int> &j0s, std::vector<int> &j1s, int pdir, int xy, int offset = 0, int poffmult = 1, int rbmode = 0, int debug = 0)
{
	static MPI_Request s1, s2;
	MPI_Status st;
	int nr = 0;
	char *to_send, *to_recv;
	int s_to_send = 0, s_to_recv = 0;
	int p_to_send, p_to_recv;
	MPI_Barrier(MPI_COMM_WORLD);
	time_time = GetTickCount();
	// calculate size to send
	if (rbmode == 0)
	{
		if (xy == 1)
			s_to_send = sizeof(double)*(j1s[(poffmult*(block / poffmult) + r) % np] - j0s[(poffmult*(block / poffmult) + r) % np]);
		else
			s_to_send = sizeof(double)*(i1s[(np + poffmult*(block / poffmult) - r) % np] - i0s[(np + poffmult*(block / poffmult) - r) % np]);
	}
	else
	{
		if (xy==1)
			s_to_send = sizeof(double)*(j1s[(np - poffmult*(block / poffmult) + r) % np] - j0s[(np - poffmult*(block / poffmult) + r) % np]);
		else
			s_to_send = sizeof(double)*(i1s[(np - poffmult*(block / poffmult) + r) % np] - i0s[(np - poffmult*(block / poffmult) + r) % np]);
	}
	to_send = new char[s_to_send];
	// do send
	if (rbmode == 0)
	{
		if (xy == 0)
		{
			p_to_send = (np + r + pdir*poffmult) % np;
			p_to_recv = (np + r - pdir*poffmult) % np;
		}
		else
		{
			p_to_send = (np + r - pdir*poffmult) % np;
			p_to_recv = (np + r + pdir*poffmult) % np;
		}
	}
	else
	{
		p_to_send = (np + r + pdir*poffmult) % np;
		p_to_recv = (np + r - pdir*poffmult) % np;
	}
	s_to_send = 0;
	// pack data
	if (rbmode == 0)
	{
		if (xy == 0)
		{
			if (solver->use_ocl)
				solver->fr_ocl_get(U, i0s[(np + poffmult*(block / poffmult) - r) % np], i1s[(np + poffmult*(block / poffmult) - r) % np], (pdir == 1) ? j1s[block] - offset : j0s[block] + offset, ((pdir == 1) ? j1s[block] - offset : j0s[block] + offset) + 1);
			for (int j = i0s[(np + poffmult*(block / poffmult) - r) % np];j < i1s[(np + poffmult*(block / poffmult) - r) % np];j++, s_to_send += sizeof(double))
				((double *)(&to_send[s_to_send]))[0] = U[j][(pdir == 1) ? j1s[block] - offset : j0s[block] + offset];
		}
		else
		{
			if (solver->use_ocl)
				solver->fr_ocl_get(U, (pdir == 1) ? i1s[block] - offset : i0s[block] + offset, ((pdir == 1) ? i1s[block] - offset : i0s[block] + offset) + 1, j0s[(poffmult*(block / poffmult) + r) % np], j1s[(poffmult*(block / poffmult) + r) % np]);
			for (int j = j0s[(poffmult*(block / poffmult) + r) % np];j < j1s[(poffmult*(block / poffmult) + r) % np];j++, s_to_send += sizeof(double))
				((double *)(&to_send[s_to_send]))[0] = U[(pdir == 1) ? i1s[block] - offset : i0s[block] + offset][j];
		}
	}
	else
	{
		if (xy == 0)
		{
			if (solver->use_ocl)
				solver->fr_ocl_get(U, i0s[(np - poffmult*(block / poffmult) + r) % np], i1s[(np - poffmult*(block / poffmult) + r) % np], (pdir == 1) ? j1s[block] - offset : j0s[block] + offset, ((pdir == 1) ? j1s[block] - offset : j0s[block] + offset) + 1);
			for (int j = i0s[(np - poffmult*(block / poffmult) + r) % np];j < i1s[(np - poffmult*(block / poffmult) + r) % np];j++, s_to_send += sizeof(double))
			{
				if (debug) printf("%d exs %d [%d][%d]=%g\n", r, p_to_send, j, (pdir == 1) ? j1s[block] - offset : j0s[block] + offset, U[j][(pdir == 1) ? j1s[block] - offset : j0s[block] + offset]);
				((double *)(&to_send[s_to_send]))[0] = U[j][(pdir == 1) ? j1s[block] - offset : j0s[block] + offset];
			}
		}
		else
		{
			if (solver->use_ocl)
				solver->fr_ocl_get(U, (pdir == 1) ? i1s[block] - offset : i0s[block] + offset, ((pdir == 1) ? i1s[block] - offset : i0s[block] + offset) + 1, j0s[(np - poffmult*(block / poffmult) + r) % np], j1s[(np - poffmult*(block / poffmult) + r) % np]);
			for (int j = j0s[(np - poffmult*(block / poffmult) + r) % np];j < j1s[(np - poffmult*(block / poffmult) + r) % np];j++, s_to_send += sizeof(double))
			{
				if (debug) printf("%d exs %d [%d][%d]=%g\n", r, p_to_send, (pdir == 1) ? i1s[block] - offset : i0s[block] + offset, j, U[(pdir == 1) ? j1s[block] - offset : j0s[block] + offset][j]);
				((double *)(&to_send[s_to_send]))[0] = U[(pdir == 1) ? i1s[block] - offset : i0s[block] + offset][j];
			}
		}
	}
	nsends += s_to_send;
	if (debug) printf("exchange data %d %d %d %d %d to s %d to r %d\n", r, block, xy, offset, poffmult, p_to_send, p_to_recv);
	MPI_Isend((void *)&s_to_send, sizeof(int), MPI_BYTE, p_to_send, 0, MPI_COMM_WORLD, &s1);
	MPI_Isend(to_send, s_to_send, MPI_BYTE, p_to_send, 0, MPI_COMM_WORLD, &s2);
	// receive
	MPI_Recv((void *)&s_to_recv, sizeof(int), MPI_BYTE, p_to_recv, 0, MPI_COMM_WORLD, &st);
	to_recv = new char[s_to_recv];
	MPI_Recv(to_recv, s_to_recv, MPI_BYTE, p_to_recv, 0, MPI_COMM_WORLD, &st);
	// unpack
	s_to_recv = 0;
	if (rbmode == 0)
	{
		if (xy == 0)
		{
			for (int j = i0s[(np + poffmult*(block / poffmult) - p_to_recv) % np];j < i1s[(np + poffmult*(block / poffmult) - p_to_recv) % np];j++, s_to_recv += sizeof(double))
				U[j][(pdir == 1) ? j1s[block] - offset : j0s[block] + offset] = ((double *)(&to_recv[s_to_recv]))[0];
			if (solver->use_ocl)
				solver->fr_ocl_put(U, i0s[(np + poffmult*(block / poffmult) - p_to_recv) % np], i1s[(np + poffmult*(block / poffmult) - p_to_recv) % np], (pdir == 1) ? j1s[block] - offset : j0s[block] + offset, ((pdir == 1) ? j1s[block] - offset : j0s[block] + offset) + 1);
		}
		else
		{
			for (int j = j0s[(poffmult*(block / poffmult) + p_to_recv) % np];j < j1s[(poffmult*(block / poffmult) + p_to_recv) % np];j++, s_to_recv += sizeof(double))
				U[(pdir == 1) ? i1s[block] - offset : i0s[block] + offset][j] = ((double *)(&to_recv[s_to_recv]))[0];
			if (solver->use_ocl)
				solver->fr_ocl_put(U, (pdir == 1) ? i1s[block] - offset : i0s[block] + offset, ((pdir == 1) ? i1s[block] - offset : i0s[block] + offset) + 1, j0s[(poffmult*(block / poffmult) + p_to_recv) % np], j1s[(poffmult*(block / poffmult) + p_to_recv) % np]);
		}
	}
	else
	{
		if (xy == 0)
		{
			for (int j = i0s[(np - poffmult*(block / poffmult) + p_to_recv) % np];j < i1s[(np - poffmult*(block / poffmult) + p_to_recv) % np];j++, s_to_recv += sizeof(double))
			{
				U[j][(pdir == 1) ? j1s[block] - offset : j0s[block] + offset] = ((double *)(&to_recv[s_to_recv]))[0];
				if (debug) printf("%d exr %d [%d][%d]=%g\n", r, p_to_recv, j, (pdir == 1) ? j1s[block] - offset : j0s[block] + offset, U[j][(pdir == 1) ? j1s[block] - offset : j0s[block] + offset]);
			}
			if (solver->use_ocl)
				solver->fr_ocl_put(U, i0s[(np - poffmult*(block / poffmult) + p_to_recv) % np], i1s[(np - poffmult*(block / poffmult) + p_to_recv) % np], (pdir == 1) ? j1s[block] - offset : j0s[block] + offset, ((pdir == 1) ? j1s[block] - offset : j0s[block] + offset) + 1);
		}
		else
		{
			for (int j = j0s[(np - poffmult*(block / poffmult) + p_to_recv) % np];j < j1s[(np - poffmult*(block / poffmult) + p_to_recv) % np];j++, s_to_recv += sizeof(double))
			{
				U[(pdir == 1) ? i1s[block] - offset : i0s[block] + offset][j] = ((double *)(&to_recv[s_to_recv]))[0];
				if (debug) printf("%d exr %d [%d][%d]=%g\n", r, p_to_recv, (pdir == 1) ? i1s[block] - offset : i0s[block] + offset, j, U[(pdir == 1) ? j1s[block] - offset : j0s[block] + offset][j]);
			}
			if (solver->use_ocl)
				solver->fr_ocl_put(U, (pdir == 1) ? i1s[block] - offset : i0s[block] + offset, ((pdir == 1) ? i1s[block] - offset : i0s[block] + offset) + 1, j0s[(np - poffmult*(block / poffmult) + p_to_recv) % np], j1s[(np - poffmult*(block / poffmult) + p_to_recv) % np]);
		}
	}
	// wait and cleanup
	MPI_Wait(&s1, &st);
	MPI_Wait(&s2, &st);
	delete[] to_send;
	delete[] to_recv;
	time_send += GetTickCount() - time_time;
}
void solve_2d_par2(double t, double save_tau, int compare, int ocl,int problem=0,int approx=1)
{
	space_fract_solver_2d *s1, *s2;
	int nsave = 1;
	int np;
	int r;
	std::vector<int> j0s, j1s,i0s,i1s;
	int j0, j1,i0,i1;
	FILE *fi1;
	MPI_Status mpi_status;
	MPI_Comm_rank(MPI_COMM_WORLD, &r);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	if (NB != np) // number of blocks must be equal to number of processes
		return;
	if (problem==0)
		s1 = new space_fract_solver_HVC_2d();
	else
		s1 = new space_fract_solver_T_2d();
	if ((r == 0) && compare)
	{
		fi1 = fopen("log.txt", "wt");
		if (problem == 0)
			s2 = new space_fract_solver_HVC_2d();
		else
			s2 = new space_fract_solver_T_2d();
	}
	s1->init(ocl);
	s1->do_not_clear_approx = 1;
	if ((r == 0) && compare)
		s2->init();
	// calc block coordinates
	for (int i = 0;i<np;i++)
	{
		j0 = (int)(BS2*i) + 1;
		j1 = (int)(BS2*(i + 1)) + 1;
		if (i == (np - 1))
			j1 = M;
		j0s.push_back(j0);
		j1s.push_back(j1);
	}
	for (int i = 0;i<np;i++)
	{
		i0 = (int)(BS*i) + 1;
		i1 = (int)(BS*(i + 1)) + 1;
		if (i == (np - 1))
			i1 = N;
		i0s.push_back(i0);
		i1s.push_back(i1);
	}
	printf("rank %d np %d N %d M %d s0 %d s1 %d i0 %d i1 %d\n", r, np, N,M, j0s[r], j1s[r],i0s[r],i1s[r]);fflush(stdout);
	for (double tt = 0;tt < t;tt += s1->tau)
	{
		double err1 = 0.0, err2 = 0.0, err3 = 0.0;
		long long i1, i2;
		nsends = 0;
		time_send = 0;
		i1 = GetTickCount();
		// do calculations
		A = 1 - s1->alpha;
		// U h
		for (int b = 0;b<NB;b++)
		{
			s1->al1_h(i0s[b], i1s[b], j0s[(r + b) % np], j1s[(r + b) % np]);
			exchange_data(s1, r, np, b, s1->Al, i0s,i1s,j0s, j1s, 1, 1);
		}
		for (int b = 0;b<NB;b++)
		{
			if (b != (NB - 1)) exchange_data(s1, r, np, b + 1, s1->U, i0s, i1s, j0s, j1s, -1, 1);
			exchange_data(s1, r, np, b, s1->U, i0s, i1s, j0s, j1s, 1, 1, 1);
			s1->Om1_h(approx, i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np], b, b + 1);
			if (b != (NB - 1))
				s1->exchange_approx(r, np, b, j0s, j1s, 0);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b++)
		{
			s1->bt1_h(i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np]);
			exchange_data(s1, r, np, b, s1->Bt, i0s, i1s, j0s, j1s, 1, 1);
		}
		for (int b = NB - 1;b >= 0;b--)
		{
			s1->U1_h(approx, i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np], 0, NB, 1);
			exchange_data(s1, r, np, b, s1->U, i0s, i1s, j0s, j1s, -1, 1);
		}
		s1->save_U();
		// U v
		for (int b = 0;b<NB;b++)
		{
			s1->al1_v(j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np]);
			exchange_data(s1, r, np, b, s1->Al, i0s, i1s, j0s, j1s, 1, 0);
		}
		for (int b = 0;b<NB;b++)
		{
			if (b != (NB - 1)) exchange_data(s1, r, np, b + 1, s1->U, i0s, i1s, j0s, j1s, -1, 0, 0);
			exchange_data(s1, r, np, b, s1->U, i0s, i1s, j0s, j1s, 1, 0, 1);
			s1->Om1_v(approx, j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np], b, b + 1);
			if (b != (NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 1);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b++)
		{
			s1->bt1_v(j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np]);
			exchange_data(s1, r, np, b, s1->Bt, i0s, i1s, j0s, j1s, 1, 0);
		}
		for (int b = NB - 1;b >= 0;b--)
		{
			s1->U1_v(1, j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np], 0, NB, 1);
			exchange_data(s1, r, np, b, s1->U, i0s, i1s, j0s, j1s, -1, 0);
		}
		s1->save_U();		
		// V h
		for (int b = 0;b<NB;b++)
		{
			if (b != (NB - 1)) exchange_data(s1, r, np, b + 1, s1->U, i0s, i1s, j0s, j1s, -1, 1);
			exchange_data(s1, r, np, b, s1->U, i0s, i1s, j0s, j1s, 1, 1, 1);
			s1->V1_h(approx, i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np], b, b + 1);
			if (b != (NB - 1))
				s1->exchange_approx(r, np, b, j0s, j1s, 0);
		}
		s1->clear_approximations();
		// V v
		for (int b = 0;b<NB;b++)
		{
			if (b != (NB - 1)) exchange_data(s1, r, np, b + 1, s1->U, i0s, i1s, j0s, j1s, -1, 0, 0);
			exchange_data(s1, r, np, b, s1->U, i0s, i1s, j0s, j1s, 1, 0, 1);
			s1->V1_v(approx, j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np], b, b + 1);
			if (b != (NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 1);
		}
		s1->clear_approximations();
		A = 1 - s1->beta;
		// C h
		for (int b = 0;b<NB;b++)
		{
			s1->al2_h(i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np]);
			exchange_data(s1, r, np, b, s1->Al, i0s, i1s, j0s, j1s, 1, 1);
		}
		for (int b = 0;b<NB;b++)
		{
			if (b != (NB - 1)) exchange_data(s1, r, np, b + 1, s1->C, i0s, i1s, j0s, j1s, -1, 1);
			exchange_data(s1, r, np, b, s1->C, i0s, i1s, j0s, j1s, 1, 1, 1);
			s1->Om2_h(approx, i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np], b, b + 1);
			if (b != (NB - 1))
				s1->exchange_approx(r, np, b, j0s, j1s, 0);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b++)
		{
			s1->bt2_h(i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np]);
			exchange_data(s1, r, np, b, s1->Bt, i0s, i1s, j0s, j1s, 1, 1);
		}
		for (int b = NB - 1;b >= 0;b--)
		{
			s1->C1_h(1, i0s[b], i1s[b], j0s[(b + r) % np], j1s[(b + r) % np], 0, NB, 1);
			exchange_data(s1, r, np, b, s1->C, i0s, i1s, j0s, j1s, -1, 1);
		}
		s1->save_C();
		// U v
		for (int b = 0;b<NB;b++)
		{
			s1->al2_v(j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np]);
			exchange_data(s1, r, np, b, s1->Al, i0s, i1s, j0s, j1s, 1, 0);
		}
		for (int b = 0;b<NB;b++)
		{
			if (b != (NB - 1)) exchange_data(s1, r, np, b + 1, s1->C, i0s, i1s, j0s, j1s, -1, 0, 0);
			exchange_data(s1, r, np, b, s1->C, i0s, i1s, j0s, j1s, 1, 0, 1);
			s1->Om2_v(approx, j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np], b, b + 1);
			if (b != (NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 1);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b++)
		{
			s1->bt2_v(j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np]);
			exchange_data(s1, r, np, b, s1->Bt, i0s, i1s, j0s, j1s, 1, 0);
		}
		for (int b = NB - 1;b >= 0;b--)
		{
			s1->C1_v(1, j0s[b], j1s[b], i0s[(np + b - r) % np], i1s[(np + b - r) % np], 0, NB, 1);
			exchange_data(s1, r, np, b, s1->C, i0s, i1s, j0s, j1s, -1, 0);
		}
		s1->save_C();
		i1 = GetTickCount() - i1;
		if (ocl)
		{
			s1->fr_ocl_get(s1->U, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->C, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->V_v, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->V_h, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Al, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Bt, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Om, 0, N + 2, 0, M + 2);
		}
		printf("%d nsends %d time_send %d\n", r, nsends, time_send);fflush(stdout);
		if (r == 0)
		{
			i2 = GetTickCount();
			if (compare)
			{
				s2->U1_h(0);
				s2->save_U();
				s2->U1_v(0);
				s2->save_U();
				s2->V1_h(0);
				s2->V1_v(0);
				s2->C1_h(0);
				s2->save_C();
				s2->C1_v(0);
				s2->save_C();
			}
			i2 = GetTickCount() - i2;
			// gather all
			for (int rr = 1;rr<np;rr++)
			{
				for (int j = 0;j<NB;j++) // blocks
					for (int jj = i0s[j] - ((j == 0) ? 1 : 0);jj<i1s[j] + ((j == (NB - 1)) ? 1 : 0);jj++) // rows
					{
						int c0 = j0s[(rr + j) % NB]; // columns
						int c1 = j1s[(rr + j) % NB];
						if (c0 == 1) c0 = 0;
						if (c1 == M) c1 = M + 1;
						MPI_Recv(s1->Al[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->Bt[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->Om[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->U[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->V_h[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->V_v[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->C[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
					}
			}
			// calculate errors
			if (compare)
				for (int i = 1;i < N;i++)
					for (int j = 1;j < M;j++)
					{
						err1 += (s2->U[i][j] - s1->U[i][j])*(s2->U[i][j] - s1->U[i][j]);
						err2 += (s2->V_h[i][j] - s1->V_h[i][j])*(s2->V_h[i][j] - s1->V_h[i][j]);
						err2 += (s2->V_v[i][j] - s1->V_v[i][j])*(s2->V_v[i][j] - s1->V_v[i][j]);
						err3 += (s2->C[i][j] - s1->C[i][j])*(s2->C[i][j] - s1->C[i][j]);
					}
			if (compare)
				fprintf(fi1, "t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, err1, err2, err3);
			printf("t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, err1, err2, err3);
			fflush(stdout);
			// save result
			if ((tt + s1->tau) >= nsave*save_tau)
			{
				save_and_convert("U", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->U);
				save_and_convert("Vh", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_h);
				save_and_convert("Vv", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_v);
				save_and_convert("C", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->C);
				nsave++;
			}
		}
		else // send all to r0
		{
			for (int j = 0;j<NB;j++) //blocks
				for (int jj = i0s[j] - ((j == 0) ? 1 : 0);jj<i1s[j] + ((j == (NB - 1)) ? 1 : 0);jj++) // rows
				{
					int c0 = j0s[(r + j) % NB];
					int c1 = j1s[(r + j) % NB];
					if (c0 == 1) c0 = 0;
					if (c1 == M) c1 = M + 1;
					MPI_Send(s1->Al[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->Bt[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->Om[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->U[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->V_h[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->V_v[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->C[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
				}
		}
	}
	if ((r == 0) && compare)
	{
		delete s2;
		fclose(fi1);
	}
	delete s1;
}
void solve_2d_par3(double t, double save_tau, int K, int compare, int ocl,int problem=0,int approx=1)
{
	space_fract_solver_2d *s1, *s2;
	int np;
	int nsave = 1;
	int r;
	std::vector<int> j0s, j1s,i0s,i1s;
	int j0, j1,i0,i1;
	FILE *fi1;
	MPI_Status mpi_status;
	MPI_Comm_rank(MPI_COMM_WORLD, &r);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	if (NB != np) // number of blocks must be equal to number of processes
		return;
	// optimal blocking parameter
	if (K == 0) K = sqrt((double)NB);
	if (K == 0) K = 1;
	if (problem==0)
		s1 = new space_fract_solver_HVC_2d();
	else
		s1 = new space_fract_solver_T_2d();
	if ((r == 0) && compare)
	{
		fi1 = fopen("log.txt", "wt");
		if (problem==0)
			s2 = new space_fract_solver_HVC_2d();
		else
			s2 = new space_fract_solver_T_2d();
	}
	s1->init(ocl);
	s1->do_not_clear_approx = 1;
	if ((r == 0) && compare)
		s2->init();
	// calc block coordinates
	for (int i = 0;i<np;i++)
	{
		i0 = (int)(BS*i) + 1;
		i1 = (int)(BS*(i + 1)) + 1;
		if (i == (np - 1))
			i1 = N;
		i0s.push_back(i0);
		i1s.push_back(i1);
	}
	for (int i = 0;i<np;i++)
	{
		j0 = (int)(BS2*i) + 1;
		j1 = (int)(BS2*(i + 1)) + 1;
		if (i == (np - 1))
			j1 = M;
		j0s.push_back(j0);
		j1s.push_back(j1);
	}
	rank = r;
	printf("rank %d np %d N %d M %d s0 %d s1 %d i0 %d i1 %d\n", r, np, N, M, j0s[r], j1s[r], i0s[r], i1s[r]);fflush(stdout);
	for (double tt = 0;tt < t;tt += s1->tau)
	{
		double err1 = 0.0, err2 = 0.0, err3 = 0.0;
		long long i1, i2;
		nsends = 0;
		time_send = 0;
		i1 = GetTickCount();
		// do calculations
		A = 1 - s1->alpha;
		// U h
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->al1_h(i0s[b + o], i1s[b + o], j0s[(np + r - b) % np], j1s[(np + r - b) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Al,i0s,i1s, j0s, j1s, 1, 1, 0, K, 1);
		}
		for (int b = 0;b<NB;b += K)
		{
			if ((b + K - 1)<NB)
			{
				if ((b + K)<NB) exchange_data(s1, r, np, b + K, s1->U, i0s, i1s, j0s, j1s, -1, 1, 0, K, 1);
				exchange_data(s1, r, np, b + K - 1, s1->U, i0s, i1s, j0s, j1s, 1, 1, 1, K, 1);
			}
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->Om1_h(approx, i0s[b + o], i1s[b + o], j0s[(np - b + r) % np], j1s[(np - b + r) % np], b + o, b + o + 1);
			if ((b + K - 1)<(NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 0, K, 1);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->bt1_h(i0s[b + o], i1s[b + o], j0s[(np - b + r) % np], j1s[(np - b + r) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Bt, i0s, i1s, j0s, j1s, 1, 1, 0, K, 1);
		}
		for (int b = NB - 1;b >= 0;b -= K)
		{
			for (int o = 0;o <K;o++)
				if (((b - o) >= 0) && ((b - (K - 1) + r) >= 0))
					s1->U1_h(1, i0s[b - o], i1s[b - o], j0s[(np - b + (K - 1) + r) % np], j1s[(np - b + (K - 1) + r) % np], 0, NB, 1);
			if ((b - (K - 1))>0)
				exchange_data(s1, r, np, b - (K - 1), s1->U, i0s, i1s, j0s, j1s, -1, 1, 0, K, 1);
		}
		for (int b = 0;b<NB;b++)
			if ((b%K) == 0)
				transpose(s1, r, np, s1->U, i0s[(np - b + r) % np], i1s[(np - b + r) % np], j0s[(np - b + r) % np], j1s[(np - b + r) % np], 1, b, b + K, ((r / K)*K) - b, j0s[b], i0s[(np - b + (r / K)*K) % np]);
		s1->save_U();
		// U v			
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->al1_v(j0s[b + o], j1s[b + o], i0s[(np - b + r) % np], i1s[(np - b + r) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Al, i0s, i1s, j0s, j1s, 1, 0, 0, K, 1);
		}
		for (int b = 0;b<NB;b += K)
		{
			if ((b + K - 1)<NB)
			{
				if ((b + K)<NB) exchange_data(s1, r, np, b + K, s1->U, i0s, i1s, j0s, j1s, -1, 0, 0, K, 1);
				exchange_data(s1, r, np, b + K - 1, s1->U, i0s, i1s, j0s, j1s, 1, 0, 1, K, 1);
			}
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->Om1_v(approx, j0s[b + o], j1s[b + o], i0s[(np - b + r) % np], i1s[(np - b + r) % np], b + o, b + o + 1);
			if ((b + K - 1)<(NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 1, K, 1);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->bt1_v(j0s[b + o], j1s[b + o], i0s[(np - b + r) % np], i1s[(np - b + r) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Bt, i0s,i1s,j0s, j1s, 1, 0, 0, K, 1);
		}
		for (int b = NB - 1;b >= 0;b -= K)
		{
			for (int o = 0;o <K;o++)
				if (((b - o) >= 0) && ((b - (K - 1) + r) >= 0))
					s1->U1_v(1, j0s[b - o], j1s[b - o], i0s[(np - b + (K - 1) + r) % np], i1s[(np - b + (K - 1) + r) % np], 0, NB, 1);
			if ((b - (K - 1))>0)
				exchange_data(s1, r, np, b - (K - 1), s1->U, i0s, i1s, j0s, j1s, -1, 0, 0, K, 1);
		}
		s1->save_U();
		// V h
		for (int b = 0;b<NB;b++)
			if ((b%K) == 0)
				transpose(s1, r, np, s1->U, j0s[(np - b + r) % np], j1s[(np - b + r) % np], i0s[(np - b + r) % np], i1s[(np - b + r) % np], 0, b, b + K, ((r / K)*K) - b, i0s[b], j0s[(np - b + (r / K)*K) % np]);
		for (int b = 0;b<NB;b += K)
		{
			if ((b + K - 1)<NB)
			{
				if ((b + K)<NB) exchange_data(s1, r, np, b + K, s1->U, i0s, i1s, j0s, j1s, -1, 1, 0, K, 1);
				exchange_data(s1, r, np, b + K - 1, s1->U, i0s, i1s, j0s, j1s, 1, 1, 1, K, 1);
			}
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->V1_h(approx, i0s[b + o], i1s[b + o], j0s[(np - b + r) % np], j1s[(np - b + r) % np], b + o, b + o + 1);
			if ((b + K - 1)<(NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 0, K, 1);
		}
		s1->clear_approximations();
		// V v
		for (int b = 0;b<NB;b++)
			if ((b%K) == 0)
				transpose(s1, r, np, s1->U, i0s[(np - b + r) % np], i1s[(np - b + r) % np], j0s[(np - b + r) % np], j1s[(np - b + r) % np], 1, b, b + K, ((r / K)*K) - b, j0s[b], i0s[(np - b + (r / K)*K) % np]);
		for (int b = 0;b<NB;b += K)
		{
			if ((b + K - 1)<NB)
			{
				if ((b + K)<NB) exchange_data(s1, r, np, b + K, s1->U, i0s, i1s, j0s, j1s, -1, 0, 0, K, 1);
				exchange_data(s1, r, np, b + K - 1, s1->U, i0s, i1s, j0s, j1s, 1, 0, 1, K, 1);
			}
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->V1_v(approx, j0s[b + o], j1s[b + o], i0s[(np - b + r) % np], i1s[(np - b + r) % np], b + o, b + o + 1);
			if ((b + K - 1)<(NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 1, K, 1);
		}
		s1->clear_approximations();
		A = 1 - s1->beta;
		// C h
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->al2_h(i0s[b + o], i1s[b + o], j0s[(np - b + r) % np], j1s[(np - b + r) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Al, i0s, i1s, j0s, j1s, 1, 1, 0, K, 1);
		}
		for (int b = 0;b<NB;b += K)
		{
			if ((b + K - 1)<NB)
			{
				if ((b + K)<NB) exchange_data(s1, r, np, b + K, s1->C, i0s, i1s, j0s, j1s, -1, 1, 0, K, 1);
				exchange_data(s1, r, np, b + K - 1, s1->C, i0s, i1s, j0s, j1s, 1, 1, 1, K, 1);
			}
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->Om2_h(approx, i0s[b + o], i1s[b + o], j0s[(np - b + r) % np], j1s[(np - b + r) % np], b + o, b + o + 1);
			if ((b + K - 1)<(NB - 1))
				s1->exchange_approx(r, np, b, j0s, j1s, 0, K, 1);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->bt2_h(i0s[b + o], i1s[b + o], j0s[(np - b + r) % np], j1s[(np - b + r) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Bt, i0s, i1s, j0s, j1s, 1, 1, 0, K, 1);
		}
		for (int b = NB - 1;b >= 0;b -= K)
		{
			for (int o = 0;o <K;o++)
				if (((b - o) >= 0) && ((b - (K - 1) + r) >= 0))
					s1->C1_h(1, i0s[b - o], i1s[b - o], j0s[(np - b + (K - 1) + r) % np], j1s[(np - b + (K - 1) + r) % np], 0, NB, 1);
			if ((b - (K - 1))>0)
				exchange_data(s1, r, np, b - (K - 1), s1->C, i0s, i1s, j0s, j1s, -1, 1, 0, K, 1);
		}
		s1->save_C();
		// C v			
		for (int b = 0;b<NB;b++)
			if ((b%K) == 0)
				transpose(s1, r, np, s1->C, i0s[(np - b + r) % np], i1s[(np - b + r) % np],j0s[(np - b + r) % np], j1s[(np - b + r) % np], 1, b, b + K, ((r / K)*K) - b, j0s[b], i0s[(np - b + (r / K)*K) % np]);
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->al2_v(j0s[b + o], j1s[b + o], i0s[(np - b + r) % np], i1s[(np - b + r) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Al, i0s, i1s, j0s, j1s, 1, 0, 0, K, 1);
		}
		for (int b = 0;b<NB;b += K)
		{
			if ((b + K - 1)<NB)
			{
				if ((b + K)<NB)  exchange_data(s1, r, np, b + K, s1->C, i0s, i1s, j0s, j1s, -1, 0, 0, K, 1);
				exchange_data(s1, r, np, b + K - 1, s1->C, i0s, i1s, j0s, j1s, 1, 0, 1, K, 1);
			}
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->Om2_v(approx, j0s[b + o], j1s[b + o], i0s[(np - b + r) % np], i1s[(np - b + r) % np], b + o, b + o + 1);
			if ((b + K - 1)<(NB - 1))
				s1->exchange_approx(r, np, b, i0s, i1s, 1, K, 1);
		}
		s1->clear_approximations();
		for (int b = 0;b<NB;b += K)
		{
			for (int o = 0;o < K;o++)
				if ((b + o) < NB)
					s1->bt2_v(j0s[b + o], j1s[b + o], i0s[(np - b + r) % np], i1s[(np - b + r) % np]);
			if ((b + K - 1)<(NB - 1))
				exchange_data(s1, r, np, b + K - 1, s1->Bt, i0s, i1s, j0s, j1s, 1, 0, 0, K, 1);
		}
		for (int b = NB - 1;b >= 0;b -= K)
		{
			for (int o = 0;o <K;o++)
				if (((b - o) >= 0) && ((b - (K - 1) + r) >= 0))
					s1->C1_v(1, j0s[b - o], j1s[b - o], i0s[(np - b + (K - 1) + r) % np], i1s[(np - b + (K - 1) + r) % np], 0, NB, 1);
			if ((b - (K - 1))>0)
				exchange_data(s1, r, np, b - (K - 1), s1->C, i0s, i1s, j0s, j1s, -1, 0, 0, K, 1);
		}
		for (int b = 0;b<NB;b++)
			if ((b%K) == 0)
			{
				transpose(s1, r, np, s1->U, j0s[(np - b + r) % np], j1s[(np - b + r) % np] ,i0s[(np - b + r) % np], i1s[(np - b + r) % np], 0, b, b + K, ((r / K)*K) - b, i0s[b], j0s[(np - b + (r / K)*K) % np]);
				transpose(s1, r, np, s1->C, j0s[(np - b + r) % np], j1s[(np - b + r) % np], i0s[(np - b + r) % np], i1s[(np - b + r) % np], 0, b, b + K, ((r / K)*K) - b, i0s[b], j0s[(np - b + (r / K)*K) % np]);
			}
		s1->save_C();
		i1 = GetTickCount() - i1;
		if (ocl)
		{
			s1->fr_ocl_get(s1->U, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->C, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->V_v, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->V_h, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Al, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Bt, 0, N + 2, 0, M + 2);
			s1->fr_ocl_get(s1->Om, 0, N + 2, 0, M + 2);
		}
		printf("%d nsends %d time_send %d\n", r, nsends, time_send);fflush(stdout);
		if (r == 0)
		{
			i2 = GetTickCount();
			if (compare)
			{
				s2->U1_h(0);
				s2->save_U();
				s2->U1_v(0);
				s2->save_U();
				s2->V1_h(0);
				s2->V1_v(0);
				s2->C1_h(0);
				s2->save_C();
				s2->C1_v(0);
				s2->save_C();
			}
			i2 = GetTickCount() - i2;
			// gather all
			for (int rr = 1;rr<np;rr++)
			{
				for (int j = 0;j<NB;j++) //blocks
					for (int jj = i0s[j] - ((j == 0) ? 1 : 0);jj<i1s[j] + ((j == (NB - 1)) ? 1 : 0);jj++) // rows
					{
						int c0 = j0s[(np - K*(j / K) + rr) % NB];
						int c1 = j1s[(np - K*(j / K) + rr) % NB];
						if (c0 == 1) c0 = 0;
						if (c1 == M) c1 = M + 1;
						MPI_Recv(s1->Al[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->Bt[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->Om[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->U[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->V_h[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->V_v[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
						MPI_Recv(s1->C[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, rr, 0, MPI_COMM_WORLD, &mpi_status);
					}
			}
			// calculate errors
			if (compare)
				for (int i = 1;i < N;i++)
					for (int j = 1;j < M;j++)
					{
						err1 += (s2->U[i][j] - s1->U[i][j])*(s2->U[i][j] - s1->U[i][j]);
						err2 += (s2->V_h[i][j] - s1->V_h[i][j])*(s2->V_h[i][j] - s1->V_h[i][j]);
						err2 += (s2->V_v[i][j] - s1->V_v[i][j])*(s2->V_v[i][j] - s1->V_v[i][j]);
						err3 += (s2->C[i][j] - s1->C[i][j])*(s2->C[i][j] - s1->C[i][j]);
					}
			if (compare)
				fprintf(fi1, "t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, err1, err2, err3);
			printf("t %g time0 %lld time1 %lld e1 %g e2 %g e3 %g\n", tt, i1, i2, err1, err2, err3);
			fflush(stdout);
			// save result
			if ((tt + s1->tau) >= nsave*save_tau)
			{
				save_and_convert("U", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->U);
				save_and_convert("Vh", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_h);
				save_and_convert("Vv", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->V_v);
				save_and_convert("C", nsave, s1->alpha, s1->beta, ((problem == 0) ? 0.0 : (((space_fract_solver_T_2d*)s1)->dt)), s1->C);
				nsave++;
			}
		}
		else // send all to r0
		{
			for (int j = 0;j<NB;j++) //blocks
				for (int jj = i0s[j] - ((j == 0) ? 1 : 0);jj<i1s[j] + ((j == (NB - 1)) ? 1 : 0);jj++) // rows
				{
					int c0 = j0s[(np - K*(j / K) + r) % NB];
					int c1 = j1s[(np - K*(j / K) + r) % NB];
					if (c0 == 1) c0 = 0;
					if (c1 == M) c1 = M + 1;
					MPI_Send(s1->Al[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->Bt[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->Om[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->U[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->V_h[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->V_v[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
					MPI_Send(s1->C[jj] + c0, (c1 - c0)*sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
				}
		}
	}
	if ((r == 0) && compare)
	{
		delete s2;
		fclose(fi1);
	}
	delete s1;
}
int main(int argc,char **argv)
{
	double a=0.85,b=0.9;
	int approx=0,solve=3;
	if (argc>=2) a=atof(argv[1]);
	if (argc>=3) b=atof(argv[2]);
	if (argc>=4) approx=atoi(argv[3]);
	if (argc>=5) solve=atoi(argv[4]);

	//MPI_Init(&argc,&argv);
	//solve_2d_par1(0.4 / N,0.4/N,0,0);
	//solve_2d_par1(0.4/ N,0.4/N,1,0);
	//solve_2d_par1(0.4 / N, 0.4 / N, 1, 1);
	//solve_2d_par2(0.4 / N, 0.4 / N, 1, 0);
	//solve_2d_par2(0.4 / N, 0.4 / N, 1, 1);
	//solve_2d_par3(0.4 / N, 0.4 / N, 1, 1, 0);
	//solve_2d_par3(0.4 / N, 0.4 / N, 2, 1, 0);
	//solve_2d_par3(0.4 / N, 0.4 / N, 4, 1, 0);
	//solve_2d_par3(0.4 / N, 0.4 / N, 1, 1, 1);
	//solve_2d_par3(0.4 / N, 0.4 / N, 2, 1, 1);
	//solve_2d_par3(0.4 / N, 0.4 / N, 4, 1, 1);
	
	//solve_2d_par1(0.01, 0.01, 1, 0, 1);
	//solve_2d_par2(0.01 , 0.01, 1, 0, 1);
	//solve_2d_par3(0.01, 0.01, 1, 1, 0, 1);
	//solve_2d_par3(0.01, 0.01, 2, 1, 0, 1);
	//solve_2d_par3(0.01, 0.01, 4, 1, 0, 1);
	//solve_2d_par1(0.05, 0.05, 0, 0, 1);
	//solve_2d_par2(0.05 , 0.05, 1, 0, 1,0);
	//solve_2d_par3(0.01, 0.01, 1, 1, 0, 1,1);
	//solve_2d_par3(0.05, 0.05, 2, 1, 0, 1,0);
	//solve_2d_par3(0.01, 0.01, 4, 1, 0, 1,1);
	//MPI_Finalize();
	//solve_2d(0.1, 0.1, 1, 0);
	//solve_2d(1.0, 0.1, 0, 1,a,b,approx,solve); 
	solve_2d(0.0011, 0.0001, 0, 1,a,b,0,1);
	//solve_T_var_time_step(0.0011 , 0.0001, 0.01, a, b,0,0,0);
	//solve_2d(0.0001, 0.00001, 0, 1, 0.95, 1, 1, 1);
	return 0;
}
