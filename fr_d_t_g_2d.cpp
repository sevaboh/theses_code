// 2D diffusion equation with mass exchange (Caputo type and non-linear, Caputo-Fabrizio, type) for the cases of
// [psi-]Caputo derivative
// k-derivative
// psi-derivative
// local approximation of Caputo derivative (for Caputo-type mass exchange, without OpenCL parallelization)
// With
// Caputo derivative approximation with two sets of series
// psi-derivative approximation with recursive subdivision cubature and two series expansion algorithms
// Fractional derivatives approximation using fixed memory principle
// OpenCL code
// OpenMP parallelization
// PSO parameters fitting (with additional steepest descent and PSO-over-PSO optimization)
#define _USE_MATH_DEFINES
//#define OCL
#ifdef OCL
#define NEED_OPENCL
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef WIN32
#include <Windows.h>
unsigned __int64 cpu_counter()
{
	LARGE_INTEGER ret;
	QueryPerformanceCounter(&ret);
	return ret.QuadPart;
}
#define finite _finite
#define __attribute__(x)
#else
#include <unistd.h>
#include <sys/times.h>
#define __int64 long long
#ifndef NO_ASM
#define rdtscll(val) do { \
     unsigned int __a,__d; \
     asm volatile("rdtsc" : "=a" (__a), "=d" (__d)); \
     (val) = ((unsigned long)__a) | (((unsigned long)__d)<<32); \
} while(0)
#endif
unsigned __int64 cpu_counter()
{
#ifndef NO_ASM
    unsigned __int64 l;
	rdtscll(l);
    return l;
#else
    return clock();
#endif
}
unsigned int GetTickCount()
{
   struct tms t;
   long long time=times(&t);
   int clk_tck=sysconf(_SC_CLK_TCK);
   return (unsigned int)(((long long)(time*(1000.0/clk_tck)))%0xFFFFFFFF);    
}
#endif
#ifdef OCL
#include "../include/sarpok3d.h"
#endif
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////

// Note that the functions Gamma and LogGamma are mutually dependent.
double Gamma
(
    double x    // We require x > 0
);

double LogGamma
(
    double x    // x must be positive
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}

    if (x < 12.0)
    {
        return log(fabs(Gamma(x)));
    }

	// Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    static const double c[8] =
    {
		 1.0/12.0,
		-1.0/360.0,
		1.0/1260.0,
		-1.0/1680.0,
		1.0/1188.0,
		-691.0/360360.0,
		1.0/156.0,
		-3617.0/122400.0
    };
    double z = 1.0/(x*x);
    double sum = c[7];
    for (int i=6; i >= 0; i--)
    {
        sum *= z;
        sum += c[i];
    }
    double series = sum/x;

    static const double halfLogTwoPi = 0.91893853320467274178032973640562;
    double logGamma = (x - 0.5)*log(x) - x + halfLogTwoPi + series;    
	return logGamma;
}
double Gamma
(
    double x    // We require x > 0
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}

    // Split the function domain into three intervals:
    // (0, 0.001), [0.001, 12), and (12, infinity)

    ///////////////////////////////////////////////////////////////////////////
    // First interval: (0, 0.001)
	//
	// For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
	// So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
	// The relative error over this interval is less than 6e-7.

	const double gamma = 0.577215664901532860606512090; // Euler's gamma constant

    if (x < 0.001)
        return 1.0/(x*(1.0 + gamma*x));

    ///////////////////////////////////////////////////////////////////////////
    // Second interval: [0.001, 12)
    
	if (x < 12.0)
    {
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.
		
		double y = x;
        int n = 0;
        bool arg_was_less_than_one = (y < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below
        if (arg_was_less_than_one)
        {
            y += 1.0;
        }
        else
        {
            n = static_cast<int> (floor(y)) - 1;  // will use n later
            y -= n;
        }

        // numerator coefficients for approximation over the interval (1,2)
        static const double p[] =
        {
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        };

        // denominator coefficients for approximation over the interval (1,2)
        static const double q[] =
        {
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        };

        double num = 0.0;
        double den = 1.0;
        int i;

        double z = y - 1;
        for (i = 0; i < 8; i++)
        {
            num = (num + p[i])*z;
            den = den*z + q[i];
        }
        double result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= (y-1.0);
        }
        else
        {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for (i = 0; i < n; i++)
                result *= y++;
        }

		return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Third interval: [12, infinity)

    if (x > 171.624)
    {
		// Correct answer too large to display. Force +infinity.
		double temp = DBL_MAX;
		return temp*2.0;
    }

    return exp(LogGamma(x));
}

////////////////////////////////////////////////////////
// approximations of g(x,r)=(x-r+1)^a-(x-r)^a
////////////////////////////////////////////////////////

#define AK 10 // number of terms in approximation series
#define MAXAK 20 // maximal number of terms in approximation series
double approx_eps = 1e-7;
double eps = 1e-09;
// ia approximation gives a_AK<approx_eps for i>ia_mini(r)
// if Ts!=NULL - varying time step - r1=-Ts[r+a],r2=-Ts[r+b]
double ia_mini(double A,double r, int nterms, double a, double b,double *Ts)
{
	double ak = 1;
	int an = -ceil(fabs(a))*a / fabs(a);
	int bn = -ceil(fabs(b))*b / fabs(b);
	if (a == 0.0) an = 0;
	if (b == 0.0) bn = 0;
	for (int i = 1;i<nterms;i++)
		ak *= (A - i + 1) / i;
	if (Ts)
		return pow(fabs(ak*(pow(-(Ts[(int)r] + fabs(a)*(Ts[(int)r + an] - Ts[(int)r])), (double)nterms) - pow(-(Ts[(int)r] + fabs(b)*(Ts[(int)r + bn] - Ts[(int)r])), (double)nterms)) / approx_eps), 1.0 / (nterms - A));
	return pow(fabs(ak*(pow(a - r, (double)nterms) - pow(b - r, (double)nterms)) / approx_eps), 1.0 / (nterms - A));
}
class ia_approximation
{
public:
	double a,b;
	double min;
	int nterms;
	double c[MAXAK];
	double calc(double A,double x)
	{
		double ret=0.0;
		for (int i=0;i<nterms;i++)
			ret+=c[i]*pow(x,A-(double)i);
		return ret;
	}
	ia_approximation() { a = 1.0, b = 0.0; };
	ia_approximation(double A,double rmin, double rmax, double *F, double *Ts,double cmin = -1, int nt = AK, double _a = 1.0, double _b = 0.0) :a(_a), b(_b)
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
				min = ceil(ia_mini(A,rmax, nt, a, b, Ts));
			else
				min = ia_mini(A,rmax, nt, a, b, Ts);
		}
		if (!finite(min)) min=rmax;
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
double ibk_i2(double A,double r, double b,int nterms,double a,double c,double *Ts,double step=-1.0)
{
	double ak = 1;
	int an = -ceil(fabs(a))*a / fabs(a);
	int cn = -ceil(fabs(c))*c / fabs(c);
	if (a == 0.0) an = 0;
	if (c == 0.0) cn = 0;
	for (int i = 1;i<nterms;i++)
		ak *= (A - i + 1) / i;
	if (fabs(b-((Ts==NULL)?r:Ts[(int)r]))<1e-10)
		return pow(fabs(approx_eps / ak), 1.0 / nterms);
	if (Ts)
	{
		double v1=b - (Ts[(int)r] + fabs(a)*(Ts[(int)r + an] - Ts[(int)r]));
		double v2=b - (Ts[(int)r] + fabs(c)*(Ts[(int)r + cn] - Ts[(int)r]));
		if (v1<0.0) v1=0.0; else v1=pow(v1, A - nterms);
		if (v2<0.0) v2=0.0; else v2=pow(v2, A - nterms);	
		if ((v1!=0.0) && (v2!=0.0))
			return pow(fabs(approx_eps / (ak*( v1- v2))), 1.0 / nterms);
		else
			return pow(fabs(approx_eps / ak), 1.0 / nterms);
	}
	if (step==-1)
	{
		double p1=pow((a + b - r), A)/pow((a + b - r),nterms);
		double p2=pow((c+b - r), A)/pow((c+b - r),nterms);
		return pow(fabs(approx_eps / (ak*(p1-p2))), 1.0 / nterms);
	}
	else
		return pow(fabs(approx_eps / (ak*(pow(a*step + b - r, A - nterms) - pow(c*step + b - r, A - nterms)))), 1.0 / nterms);
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
	ibk_approximation(double A,double rmin, double rmax, double *F, double *Ts, double ___b, double i2 = -1, int nt = AK, double _a = 1.0, double __b = 0.0) :a(_a), _b(__b)
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
				min = b - floor(ibk_i2(A,rmax, b, nt, a, _b, Ts));
			else
				min = b - ibk_i2(A,rmax, b, nt, a, _b, Ts);
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
// calculate approximated values of F for block
// approximations for previous blocks must be built so function must be called in loop from 0 to NB-1
// ias - "long" approximations
// ibks -  "short" approximations
// if Ts!=0 - varying time step
void calc_va(double A,int bs, int block, double *F, double *V, double *Ts, std::vector<ia_approximation> &ias, std::vector<ibk_approximation> &ibks, double _a = 1.0, double _b = 0.0)
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
				V[i - (block*bs + 1)] += ias[j].calc(A,ii);
		for (int j = 0;j < ibks.size();j++)
			if ((ii >= ibks[j].min) && (ii <= ibks[j].max))
				V[i - (block*bs + 1)] += ibks[j].calc(ii);
	}
	// add F for r=block*bs+1,...,(block+1)*bs
	for (int i = block*bs + 1;i<(block + 1)*bs + 1;i++)
	{
		double v = 0.0;
		if (Ts==NULL)
			for (int j = block*bs + 1;j <= i;j++)
				v += (pow((double)i - (double)j + _a, A) - pow((double)i - (double)j + _b, A))*F[j];
		else
			for (int j = block*bs + 1;j <= i;j++)
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
				bibk = new ibk_approximation(A,block*bs + 1, (block + 1)*bs + 1, F,Ts, ibks[j].b, -1.0, AK, _a, _b);
				if (bibk->min > ibks[j].min)
				{
					int a;
					// increase number of terms 
					for (a = AK;a < MAXAK;a++)
						if ((ibks[j].b - ibk_i2(A,rmaxidx, ibks[j].b, a, _a, _b,Ts)) < ibks[j].min)
							break;
					if (a == MAXAK)
						spl = 1;
					else
					{
						delete bibk;
						bibk = new ibk_approximation(A,block*bs + 1, (block + 1)*bs + 1, F,Ts, ibks[j].b, -1.0, a, _a, _b);
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
								bb = rr + ibk_i2(A,i2->min, bb, AK, _a, _b,NULL, ((Ts == NULL) ? -1.0 : (Ts[rmaxidx + 1] - Ts[rmaxidx]))) + ((Ts == NULL) ? 1 : 1e-15);
							} while ((bb - bb0) > eps);
							if (!finite(bb))
							{
							     bb=bb0;
								if (!finite(bb))
								    bb=rr;
							}								
							// build ibk approximation
							bibk = new ibk_approximation(A,block*bs + 1, (block + 1)*bs + 1, F,Ts, ((Ts==NULL)?floor(bb):bb), rr, AK, _a, _b);
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
				bb = rr + ibk_i2(A,rmaxidx, bb, AK, _a, _b,Ts) + ((Ts == NULL) ? 1 : 1e-15);
			} while ((bb - bb0)>eps);
			if (!finite(bb))
			{
			     bb=bb0;
			     if (!finite(bb))
			        bb=rr;
			}								
			// build ibk approximation 
			bibk = new ibk_approximation(A,block*bs + 1, (block + 1)*bs + 1, F,Ts, ((Ts==NULL)?floor(bb):bb), rr, AK, _a, _b);
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
	bia = new ia_approximation(A,block*bs + 1, (block + 1)*bs + 1, F, Ts,-1, AK, _a, _b);
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
				bb = rr + ibk_i2(A,rmaxidx, bb, AK, _a, _b,Ts) + ((Ts==NULL)?1:1e-15);
			} while ((bb - bb0)>eps);
			if (!finite(bb))
			{
			     bb=bb0;
			     if (!finite(bb))
			        bb=rr;
			}								
			// build ibk approximation 
			bibk = new ibk_approximation(A,block*bs + 1, (block + 1)*bs + 1, F,Ts, ((Ts==NULL)?floor(bb):bb), rr, AK, _a, _b);
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
/////////////////////////////////////////////////////////////
//////////////// opencl /////////////////////////////////////
/////////////////////////////////////////////////////////////
int lBS=32;
char *frt_input_opencl_text = "\n\
#pragma OPENCL EXTENSION %s : enable \n\
#define tau %g\n\
#define L (*((__global double *)(class+%d)))\n\
#define dL %g\n\
#define sL (*((__global double *)(class+%d)))\n\
#define sdL %g\n\
#define alpha %g\n\
#define beta2 %g\n\
#define alpha2 %g\n\
#define func_power %g\n\
#define Da %g\n\
#define Da2 %g\n\
#define sigma %g\n\
#define lambda (*((__global double *)(class+%d)))\n\
#define C0 (*((__global double *)(class+%d)))\n\
#define C1 (*((__global double *)(class+%d)))\n\
#define Q (*((__global double *)(class+%d)))\n\
#define H (*((__global double *)(class+%d)))\n\
#define Lx (*((__global double *)(class+%d)))\n\
#define dLx %g\n\
#define k_derivative (*((__global int *)(class+%d)))\n\
#define d_value %g\n\
#define k_der_k (*((__global double *)(class+%d)))\n\
#define testing_k %g\n\
#define func_in_kernel %d\n\
#define global_eps %g\n\
#define dt_eps %g\n\
#define integr_max_niter %d\n\
#define int_alg %d\n\
#define massN_a %g\n\
#define massN_b %g\n\
#define testing %d\n\
#define dt_alg %d\n\
#define massN %d\n\
#define IA_SIZE %d\n\
#define ia_row(k) (((__global char *)(ias+(alloced_approx_per_row*(k))*(IA_SIZE))))\n\
#define ia_min(n) (*((__global double *)(ias+(n)*(IA_SIZE)+%d)))\n\
#define ia_nterms(n) (*((__global int *)(ias+(n)*(IA_SIZE)+%d)))\n\
#define ia_c(n) ((__global double *)(ias+(n)*(IA_SIZE)+%d))\n\
#define ia_a(n) (*(__global double *)(ias+(n)*(IA_SIZE)+%d))\n\
#define ia_b(n) (*(__global double *)(ias+(n)*(IA_SIZE)+%d))\n\
#define IBK_SIZE %d\n\
#define ibk_row(k) (((__global char *)(ibks+(alloced_approx_per_row*(k))*(IBK_SIZE))))\n\
#define ibk_min(n) (*((__global double *)(ibks+(n)*(IBK_SIZE)+%d)))\n\
#define ibk_max(n) (*((__global double *)(ibks+(n)*(IBK_SIZE)+%d)))\n\
#define ibk_b(n) (*((__global double *)(ibks+(n)*(IBK_SIZE)+%d)))\n\
#define ibk_nterms(n) (*((__global int *)(ibks+(n)*(IBK_SIZE)+%d)))\n\
#define ibk_old(n) (*((__global int *)(ibks+(n)*(IBK_SIZE)+%d)))\n\
#define ibk_c(n) ((__global double *)(ibks+(n)*(IBK_SIZE)+%d))\n\
#define ibk_a(n) (*(__global double *)(ibks+(n)*(IBK_SIZE)+%d))\n\
#define ibk__b(n) (*(__global double *)(ibks+(n)*(IBK_SIZE)+%d))\n\
\n\
#define approx_eps (%g)\n\
#define AK (%d)\n\
#define MAXAK (%d)\n\
#define BS (%d)\n\
#define BS2 (%d)\n\
#define eps (%g)\n\
#define G2a (%g)\n\
#define NN (%d)\n\
#define NULL 0 \n\
#define kb_stack_size 100\n\
#define lBS (%d)\n\
";
char *frt_approx_opencl_text = "\n\
#define abs(x) (((x)>0)?(x):(-(x)))\n\
double ia_mini(double r,int nterms,double A,double a,double b)\n\
{\n\
	double ak = 1;\n\
	for (int i = 1;i<nterms;i++)\n\
		ak *= (A - i + 1) / i;\n\
	return powr(abs(ak*(pown(a - r, nterms) - pown(b-r, nterms)) / approx_eps), 1.0 / (nterms - A));\n\
}\n\
double ia_calc(double x,__global char *ias,int n,int k,int alloced_approx_per_row,double A)\n\
{\n\
	double ret = 0.0;\n\
	int nt=ia_nterms(n);\n\
	__global double *c=ia_c(n);\
	double p0=powr(x, A);\n\
	for (int i = 0;i<nt;i++)\n\
	{\n\
		ret += c[i] * p0;\n\
		p0/=x;\n\
	}\n\
	return ret;\n\
}\n\
void new_ia_approximation(double rmin, double rmax,int k,int alloced_approx_per_row, __global double *F,__global char *ias,int n,double A,int N, double cmin, int nt,double a,double b)\n\
{\n\
	double ak;\n\
	double cc[MAXAK];\n\
	__global double *c=ia_c(n);\n\
	for (int i = 0;i<MAXAK;i++)\n\
	    cc[i]=-0.0;\n\
	for (int j = rmin;j<rmax;j++)\n\
	{\n\
		ak=1.0;\n\
		double f=F[j];\n\
		double p1= a - (double)j;\n\
		double p2= b -(double)j;\n\
		for (int i = 1;i<nt;i++)\n\
		{\n\
			ak *= (A - i + 1) / i;\n\
			cc[i]+=ak*f*(p1-p2);\n\
			p1*=a - (double)j;\n\
			p2*=b - (double)j;\n\
		}\n\
	}\n\
	for (int i = 0;i < MAXAK;i++)\n\
	    c[i]=cc[i];\n\
	if (cmin != -1)\n\
		ia_min(n) = cmin;\n\
	else\n\
		ia_min(n) = ceil(ia_mini(rmax, nt,A,a,b));\n\
	if (!isfinite(ia_min(n))) ia_min(n)=rmax;\n\
	ia_nterms(n) = nt;\n\
	ia_a(n)=a;\n\
	ia_b(n)=b;\n\
}\n\
void delete_ia(int n,int k,int alloced_approx_per_row,__global char *ias,__global int *Nias)\n\
{\n\
	int nias=Nias[k];\n\
	for (int i=(n+1)*IA_SIZE;i<nias*IA_SIZE;i++)\n\
		ias[i-IA_SIZE]=ias[i];\n\
	Nias[k]--;\n\
}\n\
void delete_ibk(int n,int k,int alloced_approx_per_row,__global char *ibks,__global int *Nibks)\n\
{\n\
	int nibks=Nibks[k];\n\
	for (int i=(n+1)*IBK_SIZE;i<nibks*IBK_SIZE;i++)\n\
		ibks[i-IBK_SIZE]=ibks[i];\n\
	Nibks[k]--;\n\
}\n\
double ibk_i2(double r, double b, int nterms,double A,double a,double _b)\n\
{\n\
	double ak = 1;\n\
	for (int i = 1;i<nterms;i++)\n\
		ak *= (A - i + 1) / i;\n\
	if (b == r)\n\
		return powr(abs(approx_eps / ak), 1.0 / nterms);\n\
	double p1=powr((a + b - r), A)/pown((a + b - r),nterms);\n\
	double p2=powr((_b+b - r), A)/pown((_b+b - r),nterms);\n\
	return powr(abs(approx_eps / (ak*(p1-p2))), 1.0 / nterms);\n\
}\n\
double ibk_calc(double x,__global char *ibks,int n,int k,int alloced_approx_per_row)\n\
{\n\
	double ret = 0.0, xbk = 1;\n\
	__global double *c=ibk_c(n);\
	double b=ibk_b(n);\
	int nt=ibk_nterms(n);\
	for (int i = 0;i<nt;i++)\n\
	{\n\
		ret += c[i] * xbk;\n\
		xbk *= (x - b);\n\
	}\n\
	return ret;\n\
}\n\
void new_ibk_approximation(double rmin, double rmax, int k,int alloced_approx_per_row,__global double *F, double _b,__global char *ibks,int n,double A,int N, double i2, int nt,double a,double __b)\n\
{\n\
	double ak;\n\
	double cc[MAXAK];\n\
	__global double *c=ibk_c(n);\n\
	ibk_b(n) = _b;\n\
	for (int i = 0;i<MAXAK;i++)\n\
	    cc[i]=-0.0;\n\
	for (int j = rmin;j<rmax;j++)\n\
	{\n\
		ak=1;\n\
		double f=F[j];\n\
		double p1=powr(a + _b - (double)j, (double)A);\n\
		double p2=powr(__b+_b - (double)j, (double)A);\n\
		if (!isfinite(p1)) p1=0.0;\n\
		if (!isfinite(p2)) p2=0.0;\n\
		for (int i = 0;i<nt;i++)\n\
		{\n\
			if (i != 0) ak *= (A - i + 1) / i;\n\
			cc[i]+=ak*f*(p1-p2);\n\
			p1/= a + _b - (double)j;\n\
			p2/=__b+_b - (double)j;\n\
		}\n\
	}\n\
	for (int i = 0;i < MAXAK;i++)\n\
	    c[i]=cc[i];\n\
	if (i2 != -1)\n\
		ibk_min(n) = i2;\n\
	else\n\
		ibk_min(n) = _b - floor(ibk_i2(rmax, _b, nt,A,a,__b));\n\
	ibk_max(n) = _b;\n\
	ibk_nterms(n) = nt;\n\
	ibk_old(n) = 0;\n\
	ibk_a(n)=a;\n\
	ibk__b(n)=__b;\n\
}\n\
void copy_ibk_approximation(__global char *ibks,int n,int i,int k,int alloced_approx_per_row)\n\
{\n\
	__global double *cn=ibk_c(n);\
	__global double *ci=ibk_c(i);\
	ibk_min(n) = ibk_min(i);\n\
	ibk_max(n) = ibk_max(i);\n\
	ibk_b(n) = ibk_b(i);\n\
	ibk_a(n)=ibk_a(i);\n\
	ibk__b(n)=ibk__b(i);\n\
	ibk_nterms(n) = ibk_nterms(i);\n\
	for (int j = 0;j < MAXAK;j++)\n\
		cn[j] = ci[j];\n\
	ibk_old(n) = ibk_old(i);\n\
}\n\
void calc_va(int bs,int block,int k,int alloced_approx_per_row, __global double *F, double *BVK,__global int *Nibks,__global int *Nias,__global char *ias,__global char *ibks,double A,int N,double _a,double _b)\n\
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
		double v=0.0;\n\
		int nias=Nias[k];\n\
		int nibks=Nibks[k];\n\
		for (int j = 0;j<nias;j++)\n\
			if (i >= ia_min(j))\n\
				v += ia_calc(i,ias,j,k,alloced_approx_per_row,A);\n\
		for (int j = 0;j<nibks;j++)\n\
			if ((i >= ibk_min(j)) && (i <= ibk_max(j)))\n\
				v += ibk_calc(i,ibks,j,k,alloced_approx_per_row);\n\
		BVK[(i - (block*bs + 1))*(N+2)+k]+=v;\n\
	}\n\
	// add F for r=block*BS+1,...,(block+1)*BS\n\
	for (int i = block*bs + 1;i<(block + 1)*bs + 1;i++)\n\
	{\n\
		double v = 0.0;\n\
		for (int j = block*bs + 1;j <= i;j++)\n\
			v += (powr((double)i - (double)j + _a, A) - powr((double)i - (double)j+_b, A))*F[j];\n\
		BVK[(i - (block*bs + 1))*(N+2)+k]+=v;\n\
	}\n\
	// 1) remove ibks with max<rmax\n\
	for (int j = 0;j < Nibks[k];j++)\n\
		if (ibk_max(j) < rmax)\n\
		{\n\
			delete_ibk(j,k,alloced_approx_per_row,ibks,Nibks);\n\
			j--;\n\
		}\n\
	// 2) append current block approximation to ibks with min>rmax\n\
	int ss = Nibks[k];\n\
	for (int j = 0;j < ss;j++)\n\
		if (ibk_min(j) >= rmax)\n\
			if (ibk_old(j) == 0)\n\
			{\n\
				int i2=Nibks[k]+1;\n\
				int spl = 0;\n\
				int bibk=Nibks[k];\n\
				new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, ibk_b(j),ibks,bibk,A,N,-1,AK,_a,_b);\n\
				if (ibk_min(bibk) > ibk_min(j))\n\
				{\n\
					int a;\n\
					// increase number of terms \n\
					for (a = AK;a < MAXAK;a++)\n\
						if ((ibk_b(j) - ibk_i2(rmax, ibk_b(j), a,A,_a,_b)) < ibk_min(j))\n\
							break;\n\
					if (a == MAXAK)\n\
						spl = 1;\n\
					else\n\
						new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1, k, alloced_approx_per_row, F, ibk_b(j), ibks, bibk, A, N,-1.0,a,_a,_b);\n\
					if (spl)\n\
						copy_ibk_approximation(ibks,i2,j,k,alloced_approx_per_row);\n\
				}\n\
				int bibknt=ibk_nterms(bibk);\n\
				for (int i = 0;i<bibknt;i++)\n\
					ibk_c(j)[i] += ibk_c(bibk)[i];\n\
				if (ibk_nterms(j) < ibk_nterms(bibk))\n\
					ibk_nterms(j) = ibk_nterms(bibk);\n\
				// split ibks into two parts if current block ibk min is bigger that ibks min\n\
				if (spl)\n\
					if (ibk_min(bibk) > ibk_min(j))\n\
					{\n\
						ibk_max(i2) = ceil(ibk_min(bibk));\n\
						ibk_old(i2) = 1;\n\
						ibk_min(j) = ibk_max(i2) + 1;\n\
						copy_ibk_approximation(ibks,Nibks[k],i2,k,alloced_approx_per_row);\n\
						i2=Nibks[k];\n\
						Nibks[k]++;\n\
						if (ibk_min(j) > ibk_max(j))\n\
						{\n\
							ibk_max(i2) = ibk_max(j);\n\
							delete_ibk(j,k,alloced_approx_per_row,ibks,Nibks);\n\
							j--;\n\
							ss--;\n\
						}\n\
						// create serie of approximations for i2->min,i2->max\n\
						double rr = ibk_min(i2);\n\
						double bb0 = rr, bb = rr;\n\
						double i2min=ibk_min(i2);\n\
						double i2max = ibk_max(i2);\n\
						do\n\
						{\n\
							// find bj>rj:bj-i2(r,bj,k)/2=rj+1\n\
							bb = rr + 1;\n\
							do\n\
							{\n\
								bb0 = bb;\n\
								bb = rr + ibk_i2(i2min, bb, AK,A,_a,_b) + 1;\n\
							} while (abs(bb - bb0) > eps);\n\
							if (!isfinite(bb)) bb=bb0;\n\
							// build ibk approximation \n\
							bibk=Nibks[k];\n\
							new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, floor(bb),ibks,bibk,A,N,rr,AK,_a,_b);\n\
							Nibks[k]++;\n\
							// move r to bj+i2(r,bj,k)\n\
							rr = floor(bb) + 1;\n\
							if (floor(bb) > i2max)\n\
								ibk_max(bibk) = i2max;\n\
						} while (floor(bb) < i2max);\n\
					}\n\
			}\n\
	// 3) build block j approximations for ibks with min<rmax and max>rmax and for splitted ibks with min>rmax\n\
	int nibks=Nibks[k];\n\
	for (int j = 0;j < nibks;j++)\n\
		if (ibk_max(j) >= rmax)\n\
			if (ibk_min(j) < rmax)\n\
				if (ibk_max(j)>bm1)\n\
					if (ibk_old(j) == 0)\n\
					{\n\
						bm1 = ibk_max(j);\n\
						ibk_old(j) = 1;\n\
					}\n\
	if (bm1 != -1)\n\
	{\n\
		int nibks=Nibks[k];\n\
		for (int j = 0;j < nibks;j++)\n\
			if (ibk_max(j) < bm1)\n\
				ibk_old(j) = 1;\n\
		double rr = rmax;\n\
		double bb0 = rr, bb = rr;\n\
		do\n\
		{\n\
			// find bj>rj:bj-i2(r,bj,k)/2=rj+1\n\
			bb = rr + 1;\n\
			do\n\
			{\n\
				bb0 = bb;\n\
				bb = rr + ibk_i2(rmax, bb, AK,A,_a,_b) + 1;\n\
			} while (abs(bb - bb0)>eps);\n\
			if (!isfinite(bb)) bb=bb0;\n\
			// build ibk approximation \n\
			bibk=Nibks[k];\n\
			new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, floor(bb),ibks,bibk,A,N,rr,AK,_a,_b);\n\
			Nibks[k]++;\n\
			// move r to bj+i2(r,bj,k)\n\
			rr = floor(bb) + 1;\n\
			if (floor(bb) > bm1)\n\
				ibk_max(bibk) = bm1;\n\
		} while (floor(bb) < bm1);\n\
	}\n\
	// 4) build ia approximation and add it to last ia approximation (with biggest min) if current ia.min< lastia.min\n\
	int bia=Nias[k];\n\
	new_ia_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F,ias,bia,A,N,-1,AK,_a,_b);\n\
	if (Nias[k])\n\
		if (ia_min(bia) < ia_min(Nias[k]-1))\n\
		{\n\
			int ia_nt=ia_nterms(bia);\n\
			for (int i = 0;i<ia_nt;i++)\n\
				ia_c(Nias[k]-1)[i] += ia_c(bia)[i];\n\
			if (ia_nterms(Nias[k]- 1) < ia_nterms(bia))\n\
				ia_nterms(Nias[k]-1) = ia_nterms(bia);\n\
			bia = -1;\n\
		}\n\
	// 5) build ibk approximation serie if current ia.min>lastia.min, set current ia minimum as ibks maximum and add it to ias\n\
	if (bia!=-1)\n\
	{\n\
		bm1 = rmax;\n\
		int nibks=Nibks[k];\n\
		for (int j = 0;j < nibks;j++)\n\
			if ((ibk_max(j) + 1)>bm1)\n\
				bm1 = ibk_max(j) + 1;\n\
		double rr = bm1;\n\
		double bb0 = rr, bb = rr;\n\
		double imin=ia_min(bia);\n\
		do\n\
		{\n\
			// find bj>rj:bj-i2(r,bj,k)=rj+1\n\
			bb = rr + 1;\n\
			do\n\
			{\n\
				bb0 = bb;\n\
				bb = rr + ibk_i2(rmax, bb, AK,A,_a,_b) + 1;\n\
			} while (abs(bb - bb0)>eps);\n\
			if (!isfinite(bb)) \n\
			{\n\
			    bb=bb0;\n\
			    if (!isfinite(bb)) bb=rr;\n\
			}\n\
			// build ibk approximation \n\
			bibk=Nibks[k];\n\
			new_ibk_approximation(block*bs + 1, (block + 1)*bs + 1,k,alloced_approx_per_row,F, floor(bb),ibks,bibk,A,N,rr,AK,_a,_b);\n\
			Nibks[k]++;\n\
			// move r to bj+i2(r,bj,k)\n\
			rr = floor(bb) + 1;\n\
		} while (floor(bb) < ceil(imin));\n\
		ia_min(bia) = rr;\n\
		Nias[k]++;\n\
	}\n\
}\n";
// integrals calculation (only t^gamma)
char *ints_opencl_text = "\n\
double g(double t) \n\
{ \n\
	if (func_in_kernel == 1) \n\
		return sqrt(t); \n\
	if (func_in_kernel == 2) \n\
		return t*t;\n\
	if (func_in_kernel == 3)\n\
		return powr(t,func_power);\n\
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
			return powr(t,1.0/func_power); \n\
		if (n == 1) \n\
			return ((1.0/func_power) * powr(t,(1.0/func_power)-1))*(der_nm1 / powr(t,1.0/func_power)); \n\
		return der_nm1*((1.0/func_power) - n + 1) / t; \n\
	} \n\
	return ((n == 0) ? t : ((n == 1) ? (der_nm1 / t) : 0)); \n\
} \n\
//int((g(t_j+ja)-g(tau))^(-alpha),tau=t0,...,t1)\n\
// n steps of 1-order numerical integration\n\
double  __kb(int n,double t0,double t1,double gtj,double alpha_)\n\
{\n\
	double ret=0.0;\n\
	double step=(t1-t0)/n;\n\
	for (int i = 0;i < n;i++)\n\
	{\n\
		double p1 = gtj - g(t0 + (i+0.5)*step);\n\
		p1 = pow(p1, -alpha_);\n\
		ret += step*p1;\n\
	}\n\
	return ret;\n\
}\n\
double  ___kb(int n,double t0,double t1,double gtj,double alpha_)\n\
{\n\
	double f=pow(gtj-g(t0),-alpha_),f1,f2,f3,f4;\n\
	double sum=0.0,v;\n\
	double t=t0;\n\
	double step=(t1-t0)/n;\n\
	do\n\
	{\n\
		f1 = pow(gtj - g(t + 0.5*step), -alpha_);\n\
		f3 = pow(gtj - g(t + step/3.0), -alpha_);\n\
		f4 = pow(gtj - g(t + 2.0*step/3.0), -alpha_);\n\
		f2 = pow(gtj - g(t + step), -alpha_);\n\
		v=(f+3*f3+3*f4+f2)*step/8.0;\n\
		if (!isfinite(v))\n\
			break;\n\
		sum += v;\n\
		t += step;	\n\
		if ((f1 / f) < (f2 / f1))\n\
			step /= (f2/f1)/(f1/f);\n\
		while ((t + step) > t1)\n\
			step /= 2.0;\n\
		f = f2;\n\
	}\n\
	while ((t1-t)>global_eps*global_eps*global_eps);\n\
	return sum;\n\
}\n\
//int((g(t_j+ja)-g(tau))^(-alpha),tau=t0,...,t1)\n\
// using recursive subdivision\n\
double _kb(int n, double t0, double t1, double gtj,int sing,double alpha_)\n\
{\n\
	double t0stack[kb_stack_size];\n\
	double t1stack[kb_stack_size];\n\
	int sptr=0;\n\
	int ssize=1;\n\
	double v1, v2,ret=0.0;\n\
	t0stack[0]=t0;\n\
	t1stack[0]=t1;\n\
	while (sptr!=ssize)\n\
	{ \n\
		t0=t0stack[sptr];\n\
		t1=t1stack[sptr];\n\
		sptr++;\n\
		v1 = __kb(n, t0, t1, gtj,alpha_);\n\
		v2 = __kb(n*2, t0, t1, gtj,alpha_);\n\
		if ((t1 - t0) < global_eps) \n\
		{\n\
			if (sing)\n\
				ret+=___kb(n*16,t0,t1,gtj,alpha_);\n\
			else\n\
				ret+=0.5*(v1+v2);\n\
		}\n\
		if (fabs(v2 - v1)>global_eps)\n\
		{\n\
			if (ssize>=98)\n\
				ret+=v2;\n\
			else\n\
			{\n\
				t0stack[ssize]=t0;\n\
				t1stack[ssize]=0.5*(t0 + t1);\n\
				ssize++;\n\
				t0stack[ssize]=0.5*(t0 + t1);\n\
				t1stack[ssize]=t1;\n\
				ssize++;\n\
			}\n\
		}\n\
		else\n\
			ret+=v2;\n\
	}\n\
	return ret;\n\
}\n\
double _kb_row(double t0, double t1, double gtj,int *niter,double alpha_) \n\
{ \n\
	double sum = 0.0, v = 0.0, v2, v3, v4; \n\
	double i = 0; \n\
	niter[0] = 0; \n\
	if ((alpha_ == 1.0)&&(g(t1)==gtj)) \n\
		return 1.0; \n\
	if ((alpha_==1.0)&&(g(t1)!=gtj)) \n\
		return 0.0; \n\
	if (g(t1) <= gtj) \n\
	{ \n\
		v = inv_g_der(gtj, 1, inv_g_der(gtj, 0, 1)); \n\
		v *= pow(gtj, 1 - alpha_); \n\
		v3 = pow(fabs(1.0 - g(t1) / gtj), 1 - alpha_); \n\
		v4 = pow(fabs(1.0 - g(t0) / gtj), 1 - alpha_); \n\
		v2 = -(v3 - v4) / (1 - alpha_); \n\
		sum += v2*v; \n\
		while (fabs(v*v2) > global_eps) \n\
		{ \n\
			i += 1.0; \n\
			v = inv_g_der(gtj, i + 1.0, v); \n\
			v *= -gtj / i; \n\
			v3 *= fabs(1.0 - g(t1) / gtj);  \n\
			v4 *= fabs(1.0 - g(t0) / gtj); \n\
			v2 = -(v3 - v4) / (i - alpha_ + 1.0);\n\
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
		v *= pow(gl, 1 - alpha_);\n\
		v2 =( (pow(gh, 1 - alpha_) / pow(gl, 1 - alpha_)) - 1.0) / (1 - alpha_); // I0/gl^(1-a)\n\
		sum += v2*v;\n\
		while (fabs(v*v2) > global_eps)\n\
		{\n\
			i += 1.0;\n\
			v = inv_g_der(g1, i + 1.0, v);\n\
			v /= (1 - alpha_ + i) / gh;\n\
			if (i == 1.0)\n\
				v4 = (gl - gh)/gh;\n\
			else\n\
				v4 *= (gl - gh)*((i - 1.0) / i)*((-alpha_ + i) / (gh*(i - 1.0)));\n\
			v2 = -(v2 + v4);\n\
			sum += v2*v;\n\
			niter[0]++;\n\
			if (niter[0] > integr_max_niter)\n\
				break;\n\
		}\n\
	}\n\
	return sum;\n\
}\n\
double _kb_row2(double t0, double t1, double gtj,int *niter,__global double *v3precalc,double alpha_)\n\
{\n\
	double sum = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4, v5, v6, v7;\n\
	double gt0 = g(t0), gt1 = g(t1);\n\
	double i = 0, m,iter;\n\
	int mode=0; \n\
	niter[0] = 0;\n\
	if ((alpha_ == 1.0)&&(g(t1)==gtj))\n\
		return 1.0;\n\
	if ((alpha_==1.0)&&(g(t1)!=gtj))\n\
		return 0.0;\n\
	if (g(t1) > gtj)\n\
		mode = 1;\n\
	v1 = 1.0;\n\
	if (mode == 0)\n\
	{\n\
		v2 = pow(gtj, -alpha_)*gt1;\n\
		v6 = gt0 / gt1;\n\
	}\n\
	else\n\
	{\n\
		v2 = pow(gt0, 1-alpha_);\n\
		v6 = pow(gt1,1-alpha_) / pow(gt0,1-alpha_);\n\
	}\n\
	do\n\
	{\n\
		if (i != 0.0)\n\
		{\n\
			v1 *= -(-alpha_ - i + 1.0) / i;\n\
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
					v5 = -(1.0 / (-alpha_-i + 1.0))*(1.0 - v6);\n\
				v3 += v4*v5;\n\
				v7 = 1;\n\
				iter=0;\n\
				while (fabs(v4*v5) > (fabs(v3)*global_eps))\n\
				{\n\
					m+=1.0;\n\
					v4 = inv_g_der(gt1, m + 1.0, v4);\n\
					v4 /= m;\n\
					if (mode==0)\n\
						v4 /= ((i + m + 1.0) / (gt1*m));\n\
					else\n\
						v4 /= ((-alpha_-i + m + 1.0) / (gt1*m));\n\
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
							v7 *= (gt0 - gt1)*((m - 1.0) / m)*((-alpha_-i + m) / (gt1*(m - 1.0)));\n\
					}\n\
					v5 = -(v5 + v7);\n\
					v3 += v4*v5;\n\
					iter+=1.0;\n\
					if (dt_eps>=1)\n\
					    if (iter>dt_eps)\n\
						break;\n\
					if (iter>integr_max_niter)\n\
						break;\n\
				}\n\
			}\n\
			if (dt_eps>=1)\n\
				if (v3precalc)\n\
					if (niter[0]<dt_eps)\n\
					if (niter[0]>=0)\n\
						v3precalc[niter[0]]=v3;\n\
		}\n\
		sum += v1*v2*v3;\n\
		i+=1.0;\n\
		niter[0]++;\n\
		if (niter[0]>integr_max_niter)\n\
			break;\n\
		if (dt_eps>=1) // fixed number of terms\n\
		{\n\
			if (niter[0]<dt_eps)\n\
			{\n\
			    if (dt_alg==3)\n\
					continue;\n\
			}\n\
			else\n\
				break;\n\
		}\n\
	} while (fabs(v1*v2*v3) > global_eps);\n\
	return sum;\n\
}\n\
double kb(double t0,double t1,double a,__global double *v3precalc,double alpha_,double Da_)\n\
{\n\
	double v;\n\
	int niter,niter2;\n\
	if (alpha_ == 1.0)\n\
	{\n\
		if (t1==a)\n\
			return 1.0;\n\
		else\n\
			return 0.0;\n\
	}\n\
	if (int_alg == 0)\n\
		v = Da_*_kb(1, tau*t0, tau*t1, g(tau*a), t1==a,alpha_);\n\
	if (int_alg == 1)\n\
		v = Da_*_kb_row(tau*t0, tau*t1, g(tau*a),&niter,alpha_);\n\
	if (int_alg == 2)\n\
		v = Da_*_kb_row2(tau*t0, tau*t1, g(tau*a), &niter2,v3precalc,alpha_);\n\
	if (int_alg == 6)\n\
	{\n\
		if ((a - t1) <= 1)\n\
		{\n\
			v = Da_*_kb_row(tau*t0, tau*t1, g(tau*a),&niter,alpha_);\n\
			if (dt_alg==3) \n\
				_kb_row2(tau*t0, tau*t1, g(tau*a), &niter2,v3precalc,alpha_);\n\
		}\n\
		else\n\
			v = Da_*_kb_row2(tau*t0, tau*t1, g(tau*a), &niter2,v3precalc,alpha_);\n\
	}\n\
	return v;\n\
}\n\
// calc row with coefs - inner integrals for t0,t1 \n\
double _kb_row2_fixed_coefs(double gtj, __global double *_coefs,int n,double alpha_)\n\
{\n\
	double sum = 0.0, v1;\n\
	__global double *c=_coefs;\n\
	double i = 0;\n\
	int ii;\n\
	v1 = powr(gtj, -alpha_);\n\
	for (ii=1;ii<=n;ii++)\n\
	{\n\
		sum += v1*(c++)[0];\n\
		v1 *= -(-alpha_ - ii + 1.0) / (ii*gtj);\n\
		if (!isfinite(v1)) break;\n\
	}\n\
	return sum;\n\
}\n\
__kernel void calc_kb(__global double *kbs,__global double *kbs2,int tstep,__global double *v3precalc,__global double *v3precalc2,int precalc_size)\n\
{\n\
	int i=get_global_id(0);\n\
	if ((dt_alg!=3)||((tstep-i)<=1))\n\
	{\n\
		// 0.5 0 [0.5] \n\
		kbs[5*i+0]=kb(i, i+0.5, tstep+0.5, NULL,alpha,Da);\n\
		// 1.0 0.5 [0.5] \n\
		kbs[5*i+1]=kb(i+0.5, i+1.0, tstep+1.0, NULL,alpha,Da);\n\
		// 0.5 0 1.0 \n\
		kbs[5*i+2]=kb(i, i+1.0, tstep+0.5, NULL,alpha,Da);\n\
		// 1.0 0 1.0 \n\
		kbs[5*i+3]=kb(i, i+1.0, tstep+1.0, NULL,alpha,Da);\n\
		// 1.0 0 [0.5] \n\
		kbs[5*i+4]=kb(i, i+0.5, tstep+1.0, NULL,alpha,Da);\n\
	}\n\
	\
	// v3precalc\n\
	if (i==(tstep))\n\
	if (dt_alg==3)\n\
		if ((int_alg==2)||(int_alg==6))\n\
			if (dt_eps>=1)\n\
				if (v3precalc!=NULL)\n\
				{\n\
					if (tstep!=0)\n\
						kb(i-1.0, i, tstep+0.5, v3precalc+2*(precalc_size-1)*(int)dt_eps,alpha,Da);\n\
					kb(i, i+1.0, tstep+1.0, v3precalc+(2*precalc_size+1)*(int)dt_eps,alpha,Da);\n\
				};\n\
	{\n\
		if ((dt_alg!=3)||((tstep-i)<=1))\n\
		{\n\
			// 0.5 0 [0.5] \n\
			kbs2[5*i+0]=kb(i, i+0.5, tstep+0.5, NULL,alpha2,Da2);\n\
			// 1.0 0.5 [0.5] \n\
			kbs2[5*i+1]=kb(i+0.5, i+1.0, tstep+1.0, NULL,alpha2,Da2);\n\
			// 0.5 0 1.0 \n\
			kbs2[5*i+2]=kb(i, i+1.0, tstep+0.5, NULL,alpha2,Da2);\n\
			// 1.0 0 1.0 \n\
			kbs2[5*i+3]=kb(i, i+1.0, tstep+1.0, NULL,alpha2,Da2);\n\
			// 1.0 0 [0.5] \n\
			kbs2[5*i+4]=kb(i, i+0.5, tstep+1.0, NULL,alpha2,Da2);\n\
		}\n\
		\
		// v3precalc\n\
		if (i==(tstep))\n\
		if (dt_alg==3)\n\
			if ((int_alg==2)||(int_alg==6))\n\
				if (dt_eps>=1)\n\
					if (v3precalc2!=NULL)\n\
					{\n\
						if (tstep!=0)\n\
							kb(i-1.0, i, tstep+0.5, v3precalc2+2*(precalc_size-1)*(int)dt_eps,alpha2,Da2);\n\
						kb(i, i+1.0, tstep+1.0, v3precalc2+(2*precalc_size+1)*(int)dt_eps,alpha2,Da2);\n\
					};\n\
	}\n\
}\n\
";
// solver - only fixed D, no fixed F
// one row - one thread
char *fr_d_t_g_row_opencl_text = "\n\
double testing_solutionC(int i,int j,double t)\n\
{\n\
	double f = j*sdL;\n\
	double p = (i - 0.5)*dLx;\n\
	double T=t*tau;\n\
	return f*f*p*p+T*T;\n\
}\n\
double testing_solutionN(int i,int j,double t)\n\
{\n\
	double f = j*sdL;\n\
	double p = (i - 0.5)*dLx;\n\
	double T=t*tau;\n\
	return f+p-T*T;\n\
}\n\
double testing_rpN(int i,int j,double t)\n\
{\n\
	double T=t*tau;\n\
	return -2.0*T-massN_b*(sigma*testing_solutionC(i,j,t)-massN_a*testing_solutionN(i,j,t));\n\
}\n\
double testing_rpC(int i,int j,double t,__global double *V2,int N)\n\
{\n\
	double f = j*sdL;\n\
	double p = (i - 0.5)*dLx;\n\
	double T=t*tau;\n\
	double Dv=d_value;\n\
	return sigma*testing_k*pow(T,2.0-alpha*func_power)-2.0*T-2.0*V2[i*(N+2)+j]*(Dv*(f*f+p*p)-f*p*p);\n\
}\n\
double A_(int i,int j,int N,int m3d_c,__global double *V2)\n\
{\n\
	if (m3d_c==0)\n\
		return (V2[i*(N+2)+j]/dL)*((d_value/dL)+0.5);\n\
	if (m3d_c==1)\n\
		return (V2[i*(N+2)+j]/(dLx*dLx))*d_value;\n\
	return 0.0;\n\
}\n\
double B(int i,int j,int N,int m3d_c,__global double *V2)\n\
{\n\
	if (m3d_c==0)\n\
		return (V2[i*(N+2)+j]/dL)*((d_value/dL)-0.5);\n\
	if (m3d_c==1)\n\
		return (V2[i*(N+2)+j]/(dLx*dLx))*d_value;\n\
	return 0.0;\n\
}\n\
double R(int i,int j,int N,int m3d_c,__global double *V2,__global double *kbs,__global double *kbs2,int tstep)\n\
{\n\
	if (m3d_c==0)\n\
		return (sigma/tau)*(kbs[5*tstep+0]+beta2*kbs2[5*tstep+0])+A_(i,j,N,m3d_c,V2)+B(i,j,N,m3d_c,V2);\n\
	if (m3d_c==1)\n\
		return (sigma/tau)*(kbs[5*tstep+1]+beta2*kbs2[5*tstep+1])+A_(i,j,N,m3d_c,V2)+B(i,j,N,m3d_c,V2);\n\
	return 0.0;\n\
}\n\
// alpha coefficients\n\
__kernel void Al(__global double *Al,__global char *class,int N,int _M,int m3d_c,__global double *V2,__global double *kbs,__global double *kbs2,int tstep)\n\
{\n\
	int r=get_global_id(0)+1;\n\
	if (m3d_c==1)\n\
	{\n\
		Al[1*(N+2)+r] = 1;\n\
		if (testing)\n\
			Al[1*(N+2)+r] =0;\n\
		for (int i = 1;i < _M;i++)\n\
			Al[(i + 1)*(N+2)+r] = B(i,r,N,m3d_c,V2) / (R(i,r,N,m3d_c,V2,kbs,kbs2,tstep) - A_(i,r,N,m3d_c,V2)*Al[i*(N+2)+r]);\n\
	}\n\
	if (m3d_c==0)\n\
	{\n\
		Al[r*(N+2)+1]=0;\n\
		for (int i = 1;i < N;i++)\n\
			Al[r*(N+2)+i+1] = B(r,i,N,m3d_c,V2) / (R(r,i,N,m3d_c,V2,kbs,kbs2,tstep)- A_(r,i,N,m3d_c,V2) *Al[r*(N+2)+i]);\n\
	}\n\
}\n\
// beta coeffients\n\
__kernel void Bt(__global double *Al,__global double *Bt,__global double *Om,__global char *class,int N,int _M,int m3d_c,__global double *V2,__global double *kbs,__global double *kbs2,int tstep)\n\
{\n\
	int r=get_global_id(0)+1;\n\
	if (m3d_c==1)	\n\
		Bt[1*(N+2)+r]=0.0;\n\
	if (m3d_c==0)		\n\
		Bt[r*(N+2)+1]=C1;\n\
	if (testing)\n\
	{\n\
		double v;\n\
		if (m3d_c == 1)\n\
			v = testing_solutionC(0, r,tstep+1.0);\n\
		if (m3d_c == 0)\n\
			v = testing_solutionC(r,0,tstep+1);\n\
		if (m3d_c==1)\n\
			Bt[1*(N+2)+r]=v;\n\
		if (m3d_c==0)\n\
			Bt[r*(N+2)+1]=v;\n\
	}\n\
	if (m3d_c==1)	\n\
		for (int i = 1;i < _M;i++)\n\
		{\n\
			if ((Al[(i + 1)*(N+2)+r]==0.0)&&(B(i,r,N,m3d_c,V2)==0.0))\n\
				Bt[(i + 1)*(N+2)+r] = (A_(i,r,N,m3d_c,V2)*Bt[i*(N+2)+r] - Om[i*(N+2)+r])/ (R(i,r,N,m3d_c,V2,kbs,kbs2,tstep) - A_(i,r,N,m3d_c,V2)*Al[i*(N+2)+r]);\n\
			else			\n\
				Bt[(i + 1)*(N+2)+r] = (Al[(i + 1)*(N+2)+r] / B(i,r,N,m3d_c,V2)) * (A_(i,r,N,m3d_c,V2)*Bt[i*(N+2)+r] - Om[i*(N+2)+r]);\n\
		}\n\
	if (m3d_c==0)	\n\
		for (int i = 1;i < N;i++)\n\
		{\n\
			if ((Al[r*(N+2)+i + 1]==0.0)&&(B(r,i,N,m3d_c,V2)==0.0))\n\
				Bt[r*(N+2)+i + 1] = (A_(r,i,N,m3d_c,V2)*Bt[r*(N+2)+i] - Om[r*(N+2)+i])/ (R(r,i,N,m3d_c,V2,kbs,kbs2,tstep) - A_(r,i,N,m3d_c,V2)*Al[r*(N+2)+i]);\n\
			else			\n\
				Bt[r*(N+2)+i + 1] = (Al[r*(N+2)+i + 1] / B(r,i,N,m3d_c,V2)) * (A_(r,i,N,m3d_c,V2)*Bt[r*(N+2)+i] - Om[r*(N+2)+i]);\n\
		}\n\
}\n\
__kernel void Om(__global double *Om,__global double *U,__global double *oldU,__global double *N_,__global double *prevN_,__global double *Htmp1,__global double *Htmp2,__global double *Htmp12,__global double *Htmp22,__global double *v3precalc,__global double *v3precalc2,__global char *class,int N,int _M,int m3d_c,__global double *V2,__global double *kbs,__global double *kbs2,int tstep,int old_size,\
__global int *approx_built2,__global int *approx_cleared2,\
int H42_size,int H42_alloced_per_row,__global double *H42,\
__global int *Nibk1,__global int *Nias1,__global char *ibks1,__global char *ias1,\
__global int *Nibk2,__global int *Nias2,__global char *ibks2,__global char *ias2,\
__global int *Nibk12,__global int *Nias12,__global char *ibks12,__global char *ias12,\
__global int *Nibk22,__global int *Nias22,__global char *ibks22,__global char *ias22,\
int alloced_approx_per_row)\n\
{\n\
	int rr=get_global_id(0)+1;\n\
	int ll=get_local_id(0);\n\
	__local double kl[lBS],kl2[lBS];\n\
	int r,i;\n\
	double k = 1.0;\n\
	double ret=0.0;\n\
	if (m3d_c == 0)\n\
	{\n\
		r=rr%(N+2);\n\
		i=rr/(N+2);\n\
	}\n\
	else\n\
	{\n\
		i=rr%(N+2);\n\
		r=rr/(N+2);\n\
	}\n\
	if (k_derivative)\n\
		k = pow(2.0, alpha - 1.0);\n\
	if (m3d_c == 0)\n\
		ret= -(sigma / tau)*(kbs[5*tstep+0]+beta2*kbs2[5*tstep+0])*U[i*(N+2)+r];\n\
	if (m3d_c == 1)\n\
		ret = -(sigma / tau)*(kbs[5*tstep+1]+beta2*kbs2[5*tstep+1])*U[r*(N+2)+i];\n\
	if (massN)\n\
	{\n\
		if (m3d_c == 0)\n\
			ret += (1.0 / tau)*(N_[i*(N+2)+r]-prevN_[i*(N+2)+r]);\n\
		if (m3d_c == 1)\n\
			ret += (1.0 / tau)*(N_[r*(N+2)+i]-prevN_[r*(N+2)+i]);\n\
	}\n\
	if (testing)\n\
	{\n\
		if (m3d_c == 0)\n\
			ret-=0.5*testing_rpC(i,r,tstep+0.5,V2,N);\n\
		if (m3d_c == 1)\n\
			ret-=0.5*testing_rpC(r,i,tstep+1.0,V2,N);\n\
	}\n\
	if ((alpha != 1.0)&&((beta2==0.0)||(alpha2!=1.0)))\n\
	{\n\
		double time_sum = 0;\n\
		int id;\n\
		if (m3d_c==0)\n\
			id=i*(N+2)+r;\n\
		if (m3d_c==1)\n\
			id=r*(N+2)+i;\n\
		if ((dt_alg == 0)||(dt_alg==2)||(dt_alg==3)) // full step integration\n\
			if (old_size >= 2)\n\
			{\n\
				double kv=1.0, diff=1.0,kv2=1.0;\n\
				if (old_size >= 3)\n\
				{\n\
					if (dt_alg!=3)\n\
					for (int t = ((old_size - 3)/2)*2;t >=0 ;t -= 2*lBS)\n\
					{\n\
						int br=0,br2=0;\n\
						barrier(CLK_LOCAL_MEM_FENCE);\n\
						if ((t-2*ll)>=0) \n\
						{\n\
						    if (m3d_c == 0)\n\
							kl[ll] = kbs[5*((t / 2)-ll)+2];\n\
						    if (m3d_c == 1)\n\
							kl[ll] = kbs[5*((t / 2)-ll)+3];\n\
						}\n\
						if (beta2!=0.0)\n\
						if ((t-2*ll)>=0) \n\
						{\n\
						    if (m3d_c == 0)\n\
							kl2[ll] = kbs2[5*((t / 2)-ll)+2];\n\
						    if (m3d_c == 1)\n\
							kl2[ll] = kbs2[5*((t / 2)-ll)+3];\n\
						}\n\
						barrier(CLK_LOCAL_MEM_FENCE);\n\
						for (int tt=0;tt<lBS;tt++)\n\
						if ((t-2*tt)>=0)\n\
						{\n\
						    kv = kl[tt];\n\
						    if (dt_alg==2) // restricted summing\n\
							if (fabs(kv)<dt_eps)\n\
								br++;\n\
						    if (beta2!=0.0)\n\
						    {\n\
							kv2 = kl2[tt];\n\
							if (dt_alg==2) // restricted summing\n\
							    if (fabs(kv2)<dt_eps)\n\
								br2++;\n\
							kv+=beta2*kv2;\n\
						    }\n\
						    if (rr<(N+2)*(_M+2))\n\
						    {\n\
							if (m3d_c == 0)\n\
							    diff = oldU[(t-2*tt + 2)*(N+2)*(_M+2)+i*(N+2)+r] - oldU[(t-2*tt)*(N+2)*(_M+2)+i*(N+2)+r];\n\
							if (m3d_c == 1)\n\
							    diff = oldU[(t-2*tt + 2)*(N+2)*(_M+2)+r*(N+2)+i] - oldU[(t-2*tt)*(N+2)*(_M+2)+r*(N+2)+i];\n\
							time_sum += kv*diff;\n\
						    }\n\
						}\n\
						else\n\
						{\n\
						    br++;\n\
						    br2++;\n\
						}\n\
						if ((br==lBS)&&(br2==lBS))\n\
						    break;\n\
					}\n\
					if (dt_alg==3)// integration by series expansion\n\
					{\n\
						int t=((old_size - 3)/2)*2;\n\
						__global double *Htmp,*Htmp_2,*Htmp0,*pr0;\n\
						int pr_idx;\n\
						int o1,o2,de=(int)dt_eps;\n\
						double mult;\n\
						double gtj;\n\
						if (m3d_c==0)\n\
						{\n\
							pr_idx=0;\n\
							gtj=g((tstep+0.5)*tau);\n\
							Htmp=Htmp1;\n\
							Htmp_2=Htmp12;\n\
						}\n\
						if (m3d_c==1)\n\
						{\n\
							pr_idx=1;\n\
							gtj=g((tstep+1.0)*tau);\n\
							Htmp=Htmp2;\n\
							Htmp_2=Htmp22;\n\
						}\n\
						diff = oldU[(t + 2)*(N+2)*(_M+2)+id] - oldU[t*(N+2)*(_M+2)+id];\n\
						// update Sn\n\
						mult = g(((t/2)+1)*tau);\n\
						pr_idx+=2*(t/2);\
						o1=pr_idx*(int)dt_eps;\n\
						o2=id*(int)dt_eps;\n\
						Htmp0=&Htmp[o2];\n\
						pr0=&v3precalc[o1];\n\
						barrier(CLK_LOCAL_MEM_FENCE);\n\
						kl[ll]=pr0[ll];\n\
						barrier(CLK_LOCAL_MEM_FENCE);\n\
						if (rr<(N+2)*(_M+2)) \n\
						for (int j=0;j<de;j++,Htmp0++)\n\
						{\n\
							double v=mult*kl[j]*diff;\n\
							Htmp0[0]+=v;\n\
							mult *= g(((t/2)+1)*tau);\n\
						}\n\
						if (rr<(N+2)*(_M+2)) \n\
						    time_sum=Da*_kb_row2_fixed_coefs(gtj,Htmp+id*(int)dt_eps,dt_eps,alpha);\n\
						if (beta2!=0.0)\
						{\n\
							mult = g(((t/2)+1)*tau);\n\
							Htmp0=&Htmp_2[o2];\n\
							pr0=&v3precalc2[o1];\n\
							barrier(CLK_LOCAL_MEM_FENCE);\n\
							kl2[ll]=pr0[ll];\n\
							barrier(CLK_LOCAL_MEM_FENCE);\n\
							if (rr<(N+2)*(_M+2)) \n\
							for (int j=0;j<de;j++,Htmp0++)\n\
							{\n\
								double v=mult*kl2[j]*diff;\n\
								Htmp0[0]+=v;\n\
								mult *= g(((t/2)+1)*tau);\n\
							}\n\
							if (rr<(N+2)*(_M+2)) \n\
							    time_sum+=beta2*Da2*_kb_row2_fixed_coefs(gtj,Htmp+id*(int)dt_eps,dt_eps,alpha2);\n\
						}\n\
					}\n\
				}\n\
				if (m3d_c == 1)\n\
				{\n\
					id = r*(N+2)+i;\n\
					kv = (kbs[5*((old_size- 1) / 2)+4]+beta2*kbs2[5*((old_size- 1) / 2)+4]);\n\
					diff = oldU[(old_size - 2)*(N+2)*(_M+2)+id] - oldU[(old_size-1)*(N+2)*(_M+2)+id];\n\
					time_sum += kv*diff;\n\
				}\n\
			}\n\
		if (dt_alg==4)\n\
		{\n\
			double kv,diff;\n\
			double A=1.0-alpha;\n\
			double A2=1.0-alpha2;\n\
			if (H42_size!=1)\n\
			{\n\
				double ts;\n\
				for (int a = approx_built2[id];a <= H42_size - 2;a++)\n\
				{\n\
					__global char *ias=ias2;\n\
					__global char *ibks=ibks2;\n\
					if (m3d_c==1)\n\
					{\n\
						calc_va(1,a+approx_cleared2[id],0,alloced_approx_per_row, &H42[id*H42_alloced_per_row]-approx_cleared2[id],	&time_sum,\
							&Nibk2[id],&Nias2[id],ia_row(id),ibk_row(id),A,N,2.0,1.0);\n\
						if (beta2!=0.0) \n\
						{\n\
							ias=ias22;\n\
							ibks=ibks22;\n\
							calc_va(1,a+approx_cleared2[id],0,alloced_approx_per_row, &H42[id*H42_alloced_per_row]-approx_cleared2[id],	&ts,\
								&Nibk22[id],&Nias22[id],ia_row(id),ibk_row(id),A2,N,2.0,1.0);\n\
						}\n\
					}\n\
					if (m3d_c==0)\n\
					{\n\
						ias=ias1;\n\
						ibks=ibks1;\n\
						calc_va(1,a+approx_cleared2[id],0,alloced_approx_per_row, &H42[id*H42_alloced_per_row]-approx_cleared2[id],	&time_sum,\
							&Nibk1[id],&Nias1[id],ia_row(id),ibk_row(id),A,N,1.5,0.5);\n\
						if (beta2!=0.0) \n\
						{\n\
							ias=ias12;\n\
							ibks=ibks12;\n\
							calc_va(1,a+approx_cleared2[id],0,alloced_approx_per_row, &H42[id*H42_alloced_per_row]-approx_cleared2[id],	&ts,\
								&Nibk12[id],&Nias12[id],ia_row(id),ibk_row(id),A2,N,1.5,0.5);\n\
						}\n\
					}\n\
				}\n\
				if (m3d_c==1)\n\
					approx_built2[id] = H42_size - 1;\n\
				time_sum*=Da*(pow(tau,A)/A);\n\
				time_sum+=beta2*Da2*ts*(pow(tau,A2)/A2);\n\
			}\n\
			if (m3d_c==1)\n\
			{\n\
				kv = kbs[5*((old_size- 1) / 2)+4];\n\
				diff = oldU[(old_size - 2)*(N+2)*(_M+2)+id] - oldU[(old_size-1)*(N+2)*(_M+2)+id];\n\
				time_sum += kv*diff;\n\
			}\n\
		}\n\
		ret += (sigma / tau)*time_sum;\n\
	}\n\
	if (rr<(N+2)*(_M+2)) Om[rr]=ret;\n\
}\n\
__kernel void U(__global double *Al,__global double *Bt,__global double *U,__global char *class,int N,int _M,int m3d_c,int tstep)\n\
{\n\
	int r=get_global_id(0)+1;\n\
	if (m3d_c == 1)\n\
		U[_M*(N+2)+r] = Bt[_M*(N+2)+r] / (1.0 - Al[_M*(N+2)+r]);\n\
	if (m3d_c == 0)\n\
		U[r*(N+2)+N] = Bt[r*(N+2)+N] / (1.0 - Al[r*(N+2)+N]);\n\
	if (testing)\n\
	{\n\
		if (m3d_c == 1)\n\
			U[_M*(N+2)+r]  = testing_solutionC(_M, r,tstep+1);\n\
		if (m3d_c == 0)\n\
			U[r*(N+2)+N]  = testing_solutionC(r,N,tstep+1);\n\
	}\n\
	if (m3d_c == 1)\n\
	{\n\
		for (int i = _M - 1;i >= 0;i--)\n\
			U[i*(N+2)+r] = Al[(i + 1)*(N+2)+r] * U[(i + 1)*(N+2)+r] + Bt[(i + 1)*(N+2)+r];\n\
		U[(_M+1)*(N+2)+r]=U[_M*(N+2)+r];\n\
	}\n\
	if (m3d_c == 0)\n\
	{\n\
		for (int i = N - 1;i >= 0;i--)\n\
			U[r*(N+2)+i] = Al[r*(N+2)+(i + 1)] * U[r*(N+2)+(i + 1)] + Bt[r*(N+2)+(i + 1)];\n\
		U[r*(N+2)+(_M+1)]=U[r*(N+2)+_M];\n\
	}\n\
}\n\
__kernel void N1(__global double *U,__global double *N_,__global double *prevN,__global char *class,int N,int _M,int m3d_c,int tstep,double ps)\n\
{\n\
	int r=get_global_id(0);\n\
	if (massN)\n\
	{\n\
		if (m3d_c == 1)\n\
			for (int i = 0;i <=_M ;i++)\n\
			{\n\
				prevN[i*(N+2)+r]=N_[i*(N+2)+r];\n\
				N_[i*(N+2)+r]=(1.0/(2.0+tau*massN_a*massN_b))*(2*prevN[i*(N+2)+r]+tau*sigma*massN_b*U[i*(N+2)+r]);\n\
				if (testing)\n\
					N_[i*(N+2)+r]+=(1.0/(2.0+tau*massN_a*massN_b))*(tau*testing_rpN(i,r,tstep+ps-0.5));\n\
			}\n\
		if (m3d_c == 0)\n\
			for (int i = 0;i <= N;i++)\n\
			{\n\
				prevN[r*(N+2)+i]=N_[r*(N+2)+i];\n\
				N_[r*(N+2)+i]=(1.0/(2.0+tau*massN_a*massN_b))*(2*prevN[r*(N+2)+i]+tau*sigma*massN_b*U[r*(N+2)+i]);\n\
				if (testing)\n\
					N_[r*(N+2)+i]+=(1.0/(2.0+tau*massN_a*massN_b))*(tau*testing_rpN(r,i,tstep+ps-0.5));\n\
			}\n\
	}\n\
}\
// auxiliary \n\
__kernel void save2(__global double *U,__global double *oldU,int old_size,__global double *H42,int h42_size,int h42_per_row,int N,int _M)\n\
{\n\
	int r=get_global_id(0);\n\
	for (int i = 0;i <=_M ;i++)\n\
		oldU[old_size*(N+2)*(_M+2)+i*(N+2)+r]=U[i*(N+2)+r];\n\
	if (dt_alg==4)\n\
		if (old_size>=2)\n\
			for (int i = 0;i <=_M ;i++)\n\
				H42[(i*(N+2)+r)*h42_per_row+h42_size]=U[i*(N+2)+r]-oldU[i*(N+2)+r+(old_size-2)*(N+2)*(_M+2)];\n\
}\n\
__kernel void save1(__global double *U,__global double *oldU,int old_size,int N,int _M)\n\
{\n\
	int r=get_global_id(0);\n\
	for (int i = 0;i <= N ;i++)\n\
		oldU[old_size*(N+2)*(_M+2)+r*(N+2)+i]=U[r*(N+2)+i];\n\
}\n\
__kernel void cleanup(__global int *approx_built2,__global int *approx_cleared2,__global double *H42,int h42_size,int h42_per_row,int N,int _M)\n\
{\n\
	int r=get_global_id(0);\n\
	for (int i = 0;i <=N ;i++)\n\
	{\n\
		if (approx_built2[r*(N+2)+i])\n\
		{\n\
			int j=0;\n\
			H42[(r*(N+2)+i)*h42_per_row+(j++)]=0.0;\n\
			for (int k=approx_built2[r*(N+2)+i]+1;k<h42_size;k++)\n\
				H42[(r*(N+2)+i)*h42_per_row+(j++)]=H42[(r*(N+2)+i)*h42_per_row+k];\n\
			approx_cleared2[r*(N+2)+i]+=approx_built2[r*(N+2)+i];\n\
			approx_built2[r*(N+2)+i]=0;\n\
		}\n\
	}\n\
}\n\
";

//////////////////////////////////////
/////// solver ///////////////////////
//////////////////////////////////////

int sN;  // number of nodes
int _M; // x,y nodes for 3D mode
double int_err=1e-5; // integration eps
int int_alg = 0;
int dt_alg = 0; // 0 - full step, 1 - half-step, 2 - full step restricted, 3 - full step with series expansion, 4 - series expansion for g(t)=t, 5 - local approximation for C0=0
double dt_eps=1e-8; // restriction for dt_alg=2 and number of terms in series for dt_alg=3
int massN=0; // 1 - add mass exchange equation
double massN_a=0.1,massN_b=0.1;
double init_eps=1e-5;
int init_max_iter=1000;
double init_tau_m=1000;
double __L=22;
double __H=1.0;
double _C1=1.0;
double _S=0.0; // source in right part
double __Lx=3.0;
double _sigma=0.2; // 0.8;
int testing=0; // 1 - model problem with right part
#define idx(i,j) ((i)+(j)*(sN+2))
int mode1d=0;
int debug_level=1;
double func_power2=0.0;
int use_ocl=0;
int device=0;
int double_ext=0;
int integr_max_niter=10000;
class row {
public:
	double *A;
	int x,c;
	row()
	{
		A=NULL;
		x=0;
		c=-1;
	}
	void set(double *_A,int _x,int _c)
	{
		A=_A;
		x=_x;
		c=_c;
	}
	__attribute__((always_inline)) double& operator [] (int i)
	{
		if (c == 0)
			return A[idx(i, x)];
		if (c == 1)
			return A[idx(x, i)];
		return A[i];
	}
};
////////////////////////////////////////////
class TG_solver {
public:
	int perf_count[20];
	double perf_count_d[20];
	std::vector<double> kb_precalc[8];
	std::vector<double> kb_precalc2[8];
	double *b_C; // concentration in solute
	double *b_N,*b_prevN; // concentration in non-soluble phase
	double *V2;
	row C;
	double *Al; // alpha coefficients
	double *Bt; // beta coefficients
	double *Om; // right part
	std::vector<double*> oldC; // for time-fractional derivatives
	std::vector<double> *Htmp1,*Htmp2; // for series coefficients (alpha)
	std::vector<double> *Htmp12,*Htmp22; // for series coefficients (alpha2)
	// series 1
#ifdef OCL
	cl_int
#else		
	int 
#endif
	*approx_built2,*approx_cleared2;
	std::vector<double> *H42;
	typedef struct
	{
		std::vector<ia_approximation> ias; // "long" approximations
		std::vector<ibk_approximation> ibks;	// "short" approximations
	} apps;
	apps *Happ1,*Happ2,*Happ12,*Happ22;
	// steps
	double tau; // time step length 
	int tstep; // current time step
	double L,dL,sL,sdL; // domain length and space variable step length
	int N;
	// H equation and common
	double alpha; // alpha - time derivative power
	double func_power;
	double Da,Da2;
	double k; // averaged filtration coefficient
	double sigma; // average soil porosity
	double lambda; // diffusion coefficient multiplier
	double C0; // initial condition for C
	double C1; // upper boundary condition
	double Q; 
	double H;
	int m3d_x,m3d_c; // current row
	double Lx,dLx; // length and step in x
	double ka[6];
	 // k-Caputo derivative
	int k_derivative;
	double d_value;
	double k_der_k;
	// two derivatives in left part - DaC+b2*Da2C
	double beta2,alpha2;
	// fixed points for initial conditions
	double *fixedV;
	double nfixedV;
	double *fixed_mask;
	
	double _C1,_S,func_power2;

	double testing_k;

	int old_step;
	int a3k;
#ifdef OCL
	std::vector<int> n_ibk; // number of ibk approximations per row
	std::vector<int> n_ias; // number of ia approximations per row
	char *row_ibks; // storage for approximation in one row
	char *row_ias;
	OpenCL_program *prg;
	OpenCL_commandqueue *queue;
	OpenCL_prg *prog;
	OpenCL_kernel *kAl, *kBt, *kOm, *kU,*kN1,*kcalc_kb,*ksave1,*ksave2,*kcleanup;
	OpenCL_buffer *bS,*bAl, *bBt, *bOm, *bU, *bN, *bprev_N,*boldU;
	OpenCL_buffer *bkbs,*bV2,*bv3precalc,*bkbs2,*bv3precalc2;
	OpenCL_buffer *bHtmp1,*bHtmp2,*bHtmp12,*bHtmp22;
	OpenCL_buffer *b_ibk1, *b_ias1, *b_nibk1, *b_nias1;
	OpenCL_buffer *b_ab2,*b_ac2,*b_ibk2, *b_ias2, *b_nibk2, *b_nias2,*b_H42;
	OpenCL_buffer *b_ibk12, *b_ias12, *b_nibk12, *b_nias12;
	OpenCL_buffer *b_ibk22, *b_ias22, *b_nibk22, *b_nias22;
	int old_size;
	int H42_per_row,H42_size;
	int alloced_approx_per_row;
	int kbs_alloced;
	cl_int *an_ibk,*an_ias;
	virtual void fr_ocl_call(OpenCL_kernel *k,double ps=0.0,int precalc_size=0.0)=0;
	virtual void init_opencl()=0;
	virtual void fr_ocl_check_and_resize()=0;
	virtual void fr_ocl_get(double *B)=0;
#endif
	// square of velocity
	__attribute__((always_inline)) 	double sqv(int i,int j)
	{
		return V2[idx(i, j)];
	}
	double X(int i,int j)
	{
		double f = i*sdL;
		double p = (j - 0.5)*dLx;
		return H*exp(0.5*M_PI*f / Q)*sin(0.5*M_PI*p / Q)+(p/k);
	}
	double Y(int i,int j)
	{
		double f = i*sdL;
		double p = (j - 0.5)*dLx;
		return H*exp(0.5*M_PI*f / Q)*cos(0.5*M_PI*p / Q)+(f/k);
	}
	// replaces X=X[0],Y=X[1] by fi,psi
	void FPforXY(double *_X)
	{
	    double x=_X[0],y=_X[1];
	    double bd=1e300;
	    int bi=-1,bj=-1;
	    for (int i=0;i<sN;i++)
		for (int j=0;j<_M;j++)
		{
		    double d=(x-X(i,j))*(x-X(i,j))+(y-Y(i,j))*(y-Y(i,j));
		    if (d<bd)
		    {
			bd=d;
			bi=i;
			bj=j;
		    }
		}
   	    _X[0]=bi*sdL;
	    _X[1]=(bj-0.5)*dLx;
	    if (mode1d)
	        _X[1]=(_M-1.5)*dLx;
	}
	virtual double testing_rpC(int i,int j,double t)=0;
	double testing_rpN(int i,int j,double t)
	{
		double T=t*tau;
		return -2.0*T-massN_b*(sigma*testing_solutionC(i,j,t)-massN_a*testing_solutionN(i,j,t));
	}
	double testing_solutionC(int i,int j,double t)
	{
		double f = i*sdL;
		double p = (j - 0.5)*dLx;
		double T=t*tau;
		return f*f*p*p+T*T;
	}
	double testing_solutionN(int i,int j,double t)
	{
		double f = i*sdL;
		double p = (j - 0.5)*dLx;
		double T=t*tau;
		return f+p-T*T;
	}
	double testing_check()
	{
		double err1=0.0,err2=0.0;
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_get(b_C);
			fr_ocl_get(b_N);
			fr_ocl_get(b_prevN);
		}
#endif		
	    for (int i=0;i<=sN;i++)
			for (int j=0;j<=_M;j++)
			{
				if ((i==0)&&(j==0)) continue;
				if ((i==0)&&(j==_M)) continue;
				if ((i==sN)&&(j==0)) continue;
				if ((i==sN)&&(j==_M)) continue;
				double d1=testing_solutionC(i,j,tstep)-b_C[idx(i,j)];
				double d2=testing_solutionN(i,j,tstep)-b_N[idx(i,j)];
				err1+=d1*d1;
				err2+=d2*d2;
			}
		return (err1+err2)/(2.0*sN*_M);
	}
	void initial(double *fixed_initial=NULL,int nfixed_initial=0)
	{
		// initial conditions
		for (int i = 0;i < sN + 1;i++)
			for (int j = 0;j < _M + 1;j++)
			{
			    b_C[idx(i, j)] =( (i!=0)?C0:C1);
				if (testing)
					b_C[idx(i, j)]=testing_solutionC(i,j,0);
			    if (massN)
			    {
					b_N[idx(i, j)] =0.0;
					b_prevN[idx(i, j)] =0.0;
					if (testing)
						b_N[idx(i, j)] =testing_solutionN(i,j,0);
			    }
			}
		if ((fixed_initial)&&(nfixed_initial)) // fixed initial values - triples (x,y,C)
		{
		    double sa=alpha;
		    double sb2=beta2;
		    double sda=Da;
		    double sda2=Da2;
		    double *sc= new double[(sN + 2)*(_M + 2)];
		    double err;
		    int iter=0;
		    double *init_trans=new double[3*nfixed_initial];
		    fixed_mask = new double[(sN + 2)*(_M + 2)];
		    for (int i = 0;i <= sN + 1;i++)
			for (int j = 0;j <= _M + 1;j++)
			    fixed_mask[idx(i,j)]=1e300;
		    for (int i=0;i<nfixed_initial;i++)
		    {
			memcpy(init_trans+3*i,fixed_initial+3*i,3*sizeof(double));
			FPforXY(init_trans+3*i);
			init_trans[3*i+0]=(int)(init_trans[3*i+0]/sdL);
			init_trans[3*i+1]=(int)(0.5+(init_trans[3*i+1]/dLx));
			if (init_trans[3*i+0]==0) init_trans[3*i+0]++;
			if (init_trans[3*i+0]==sN) init_trans[3*i+0]--;
			if (init_trans[3*i+1]==0) init_trans[3*i+1]++;
			if (init_trans[3*i+1]==_M) init_trans[3*i+1]--;
			fixed_mask[idx((int)init_trans[3*i+0],(int)init_trans[3*i+1])]=init_trans[3*i+2];
			if (debug_level>=3) printf("initial %d %d %g\n",(int)init_trans[3*i+0],(int)init_trans[3*i+1],init_trans[3*i+2]);
		    }
		    fixedV=init_trans;
		    nfixedV=nfixed_initial;
		    alpha=1.0;
			beta2=0.0;
		    Da=1.0;
		    Da2=1.0;
		    tau*=init_tau_m;
		    do
		    {
				memcpy(sc,b_C,(sN+2)*(_M+2)*sizeof(double));
				calc_step();
				err=0.0;
				for (int i=0;i<(sN+2)*(_M+2);i++)
					err+=(sc[i]-b_C[i])*(sc[i]-b_C[i]);
				if ((iter++)>init_max_iter) break;
				if (!finite(b_C[idx(sN/2,_M/2)]))
						break;
				if (debug_level>=3) printf("%d %g\n",iter,err);
		    }
		    while (err>init_eps);
		    if (debug_level>=2) printf("steady state initial: %d %g\n",iter,err);
		    if (err>init_eps) b_C[0]=1e300;
		    tau/=init_tau_m;
		    fixedV=NULL;
		    nfixedV=0;
		    alpha=sa;
		    Da=sda;
			Da2=sda2;
			beta2=sb2;
		    tstep=0;
		    delete [] fixed_mask;
		}
		else
		{
			fixedV=NULL;
			nfixedV=0;
		}
		N=sN;
		double *hs = new double[(N + 2)*(_M + 2)];
		memcpy(hs, b_C, (N + 2)*(_M + 2)*sizeof(double));
		oldC.push_back(hs);
	}
	TG_solver(double tau_m,double a,double kdk,int k_der=1,double dv=0.1,double _kk=-1,double l=5,double fp=1.0,double _L=-1,double _H=-1,double *fixed_initial=NULL,int nfixed_initial=0,double _Lx=-1,int noinit=0,double __C1=1e300,double __S=1e300,double _func_power2=0,double _b2=0.0,double _a2=0.0):k_der_k(kdk),k_derivative(k_der),d_value(dv)
	{
		ka[0] = 1.005e-3;
		ka[1] = 1.056e-2;
		ka[2] = -7.43e-2;
		ka[3] = 1.705e-1;
		ka[4] = -1.67e-1;
		ka[5] = 5.94e-2;
		func_power=fp;
		beta2=_b2;
		alpha2=_a2;
		if ((beta2!=0.0)&&(dt_alg==4))
			dt_alg=0;
		b_C = new double[(sN + 2)*(_M + 2)];
		if (massN)
		{
	    	    b_N = new double[(sN + 2)*(_M + 2)];
	    	    b_prevN = new double[(sN + 2)*(_M + 2)];
	    }
		else
		{
			b_N=b_prevN=NULL;
		}
		V2 = new double[(sN + 2)*(_M + 2)];
		Al=new double[sN+_M+2];
		Bt=new double[sN+ _M  + 2];
		Om=new double[sN+ _M + 2];		

		N=sN;
		
		old_step=-1;
		a3k=2;

		if (__C1!=1e300)
		    _C1=__C1;
		else
			_C1=1;
		if (__S!=1e300)
		    _S=__S;
		else
			_S=0.0;
		func_power2=_func_power2;
		L = __L;
		L=22.0; // length
		H = __H;
		H=1 / sqrt(5.0);//1.0; // depth
		C0 = 0.0; // initial C
		// averaged saturated filtration coefficient
		double cmid = 0.5;
		k = ka[0] + ka[1] * (cmid)+ka[2]*(cmid*cmid) + ka[3] * (cmid*cmid*cmid) + ka[4] * (cmid*cmid*cmid*cmid) + ka[5] * (cmid*cmid*cmid*cmid*cmid);
		k = 2.0*sqrt(5.0) / (M_PI*2.5);
		if (_kk>0) k=_kk;
		Q = k*(0.5*L-H);
		if (_L>0.0) L=_L;
		if (_H>0.0) H=_H;
		alpha=a;
		if (k_derivative)
			alpha = (k_der_k + alpha - 1) / k_der_k;
		if (testing)
			testing_k=Gamma(1.0+(2.0/func_power))/Gamma(1.0-alpha+(2.0/func_power));
		sigma=_sigma;
		C1=_C1;//35;
		lambda=l;

		sL = L;
		tau = tau_m;
		tstep=0;

		Lx = __Lx;
		Lx=5.0;
		if (_Lx>=0.0) Lx=_Lx;
		dLx = Q / _M;
		if ((Lx==0)&&(fixed_initial)) // choose Lx to have maxX>maxXinit
		{
			double maxx=0;
			double xlx;
			for (int i=0;i<nfixed_initial;i++)
				if (fixed_initial[3*i+0]>maxx) maxx=fixed_initial[3*i+0];
			do
			{
				Lx+=0.1;
				sdL=Lx;
				xlx=X(1,1);
			}
			while (xlx<1.5*maxx);				
			dL=2*Lx/(2.0*N+1.0);
			sdL = dL;
		}
		else	
		    if (Lx==0)
				Lx=3.0; // fi0
		dL=2*Lx/(2.0*N+1.0);
		sdL = dL;
		if (alpha == 1.0)
			Da = 1.0;
		else
			Da=1.0/Gamma(1.0-alpha);
		if (alpha2 == 1.0)
			Da2 = 1.0;
		else
			Da2=1.0/Gamma(1.0-alpha2);
		// calculate velocities	
		for (int ii = 0;ii <= sN;ii++)
	    	    for (int jj = 0;jj <= _M;jj++)
		    {
			double f = ii*sdL;
			double p = (jj - 0.5)*dLx;
			double a = 0.5*M_PI*H / Q;
			double s1 = a*exp(0.5*M_PI*f / Q)*sin(0.5*M_PI*p / Q);
			double s2 = a*exp(0.5*M_PI*f / Q)*cos(0.5*M_PI*p / Q) + (1 / k);
			V2[idx(ii,jj)]=(1 / (s1*s1)) + (1 / (s2*s2));
			if (k_derivative)
				V2[idx(ii, jj)] *= pow(k_der_k, 1.0 - alpha);
			if (dt_alg==5) // local approximation of Caputo derivative
				V2[idx(ii, jj)] /= alpha+beta2*alpha2;
		    }
		update();
		// calculate initial distribution
		if (noinit==0) initial(fixed_initial,nfixed_initial);
	}
	void update()
	{
		if (dt_alg==3)
		{
			if ((int_alg!=2)&&(int_alg!=6))
				int_alg=6;
			Htmp1=new std::vector<double>[(sN + 2)*(_M + 2)];
			Htmp2=new std::vector<double>[(sN + 2)*(_M + 2)];
			for (int i=0;i<(sN + 2)*(_M + 2);i++)
				for (int j=0;j<dt_eps;j++)
				{
					Htmp1[i].push_back(0.0);
					Htmp2[i].push_back(0.0);
				}
			if (beta2!=0.0)
			{
				Htmp12=new std::vector<double>[(sN + 2)*(_M + 2)];
				Htmp22=new std::vector<double>[(sN + 2)*(_M + 2)];
				for (int i=0;i<(sN + 2)*(_M + 2);i++)
					for (int j=0;j<dt_eps;j++)
					{
						Htmp12[i].push_back(0.0);
						Htmp22[i].push_back(0.0);
					}
			}
		}
		if (dt_alg==4)
		{
			approx_built2=new int[(sN + 2)*(_M + 2)];
			memset(approx_built2,0,sizeof(int)*(sN + 2)*(_M + 2));
			approx_cleared2=new int[(sN + 2)*(_M + 2)];
			memset(approx_cleared2,0,sizeof(int)*(sN + 2)*(_M + 2));
			H42=new std::vector<double>[(sN + 2)*(_M + 2)];
			for (int i=0;i<(sN + 2)*(_M + 2);i++)
				H42[i].push_back(0.0);
			Happ1=new apps[(sN + 2)*(_M + 2)];
			Happ2=new apps[(sN + 2)*(_M + 2)];
			Happ12=new apps[(sN + 2)*(_M + 2)];
			Happ22=new apps[(sN + 2)*(_M + 2)];
		}
		if (use_ocl)
		{
			delete [] Al;
			delete [] Bt;
			delete [] Om;
			Al=new double[(sN + 2)*(_M + 2)];
			Bt=new double[(sN + 2)*(_M + 2)];
			Om=new double[(sN + 2)*(_M + 2)];
		}
	}
	// removes from U[1,2] values that are already put into approximation
	void saved_cleanup()
	{
		for (int i=0;i<(sN + 2)*(_M + 2);i++)
		{
			std::vector<double> n2;
			if (approx_built2[i])
			{
				n2.push_back(0.0);
				for (int k=approx_built2[i]+1;k<H42[i].size();k++)
					n2.push_back(H42[i][k]);
				H42[i]=n2;
				approx_cleared2[i]+=approx_built2[i];
				approx_built2[i]=0;
			}
		}
	}
	// set working row and its saved values vector
	void get_F_oldF(row *U_,row *N_,row *prevN_,std::vector<double*> **old)
	{
		U_[0].set(b_C,m3d_x,m3d_c);
		if (massN)
		{
	    	    N_[0].set(b_N,m3d_x,m3d_c);		    
	    	    prevN_[0].set(b_prevN,m3d_x,m3d_c);		    
	    	}
		old[0]=&oldC;
	}
	// alpha coefficients
	virtual void al1()=0;
	// right part
	virtual void Om1()=0;
	// beta coeffients
	virtual void bt1()=0;
	void N1(double ps)
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kN1,ps);
			return;
		}
#endif
	    if (massN)
	    for (int i = 0;i < sN+1 ;i++)
		for (int j = ((mode1d==0)?0:_M-1);j < _M+1 ;j++)
		{
		    b_prevN[idx(i,j)]=b_N[idx(i,j)];
		    b_N[idx(i,j)]=(1.0/(2.0+tau*massN_a*massN_b))*(2*b_prevN[idx(i,j)]+tau*sigma*massN_b*b_C[idx(i,j)]);
			if (testing)
				b_N[idx(i,j)]+=(1.0/(2.0+tau*massN_a*massN_b))*(tau*testing_rpN(i,j,tstep+ps-0.5));
		}	    
	}
	// calc F
	void U1()	
	{
		row U_,N_,prevN_;
		std::vector<double*> *old;
		get_F_oldF(&U_,&N_,&prevN_,&old);
		al1();
		Om1();
		bt1();
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kU);
			return;
		}
#endif
		U_[N] = Bt[N] / (1.0 - Al[N]);
		if (testing)
		{
			double v;
			if (m3d_c == 0)
				v = testing_solutionC(N, m3d_x,tstep+1);
			if (m3d_c == 1)
				v = testing_solutionC(m3d_x,N,tstep+1);
			U_[N]=v;
		}
		for (int i = N - 1;i >= 0;i--)
			U_[i] = Al[i + 1] * U_[i + 1] + Bt[i + 1];
		U_[N + 1] = U_[N];
	}
	void calc_step()
	{
		perf_count[0]=0;
		perf_count[1] = 0;
		perf_count[2] = 0;
		perf_count[3] = 0;
		perf_count_d[0] = 0;
		perf_count_d[1] = 0;
#ifdef OCL
		if (use_ocl)
		{
			if (tstep==0)
				init_opencl();
			fr_ocl_check_and_resize();
			fr_ocl_call(kcalc_kb,0.0,tstep);
		}
#endif
		N1(0.5);
#ifdef OCL
		if (use_ocl)
		{
			m3d_c=0;
			fr_ocl_call(kOm);
		}
#endif		
		for (int j = ((mode1d==0)?1:_M-1);j < _M ;j++)
		{
			m3d_x=j;
			m3d_c=0;
			C.set(b_C,m3d_x,m3d_c);
			dL = sdL;
			N=sN;
			U1();
			if (use_ocl) break;
			if (debug_level==4)			
			if (use_ocl==0)
				for (int i=0;i<sN+1;i++)
					printf("%d %d %g %g %g %g %g %g %g\n",i,j,Al[i],Bt[i],Om[i],b_C[j*(sN+2)+i],(b_N?b_N[j*(sN+2)+i]:0),(b_prevN?b_prevN[j*(sN+2)+i]:0),V2[j*(sN+2)+i]);	
		}
#ifdef OCL
		if (debug_level==4)			
		if (use_ocl)
		{
			fr_ocl_get(Om);
			fr_ocl_get(Al);
			fr_ocl_get(Bt);
			fr_ocl_get(V2);
			fr_ocl_get(b_C);
			fr_ocl_get(b_N);
			fr_ocl_get(b_prevN);
			for (int j=0;j<_M+1;j++)
				for (int i=0;i<sN+1;i++)
					printf("%d %d %g %g %g %g %g %g %g\n",i,j,Al[j*(sN+2)+i],Bt[j*(sN+2)+i],Om[j*(sN+2)+i],b_C[j*(sN+2)+i],(b_N?b_N[j*(sN+2)+i]:0),(b_prevN?b_prevN[j*(sN+2)+i]:0),V2[j*(sN+2)+i]);
		}
#endif
		// save H for time-fractional derivative calculations
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(ksave1);
			old_size++;
		}
		else
		{
#endif
		if (alpha!=1.0)
		{
		    double *hs = new double[(sN + 2)*(_M + 2)];
		    memcpy(hs, b_C, (sN + 2)*(_M + 2)*sizeof(double));
		    oldC.push_back(hs);
		}
#ifdef OCL
		}
#endif		
		N1(1.0);
#ifdef OCL
		if (use_ocl)
		{
			m3d_c=1;
			fr_ocl_call(kOm);
		}
#endif		
		if (mode1d==0)
		for (int i = 1;i < sN ;i++)
		{
			m3d_x=i;
			m3d_c=1;
			C.set(b_C,m3d_x,m3d_c);
			dL = dLx;
			N=_M;
			U1();
			if (use_ocl) break;
			if (debug_level==4)			
			if (use_ocl==0)
				for (int j=0;j<_M+1;j++)
					printf("%d %d %g %g %g %g %g %g %g\n",i,j,Al[j],Bt[j],Om[j],b_C[j*(sN+2)+i],(b_N?b_N[j*(sN+2)+i]:0),(b_prevN?b_prevN[j*(sN+2)+i]:0),V2[j*(sN+2)+i]);		
		}
		// save H for time-fractional derivative calculations
#ifdef OCL
		if (debug_level==4)			
		if (use_ocl)
		{
			fr_ocl_get(Om);
			fr_ocl_get(Al);
			fr_ocl_get(Bt);
			fr_ocl_get(V2);
			fr_ocl_get(b_C);
			fr_ocl_get(b_N);
			fr_ocl_get(b_prevN);
			for (int i=1;i<sN;i++)
				for (int j=0;j<_M+1;j++)
					printf("%d %d %g %g %g %g %g %g %g\n",i,j,Al[j*(sN+2)+i],Bt[j*(sN+2)+i],Om[j*(sN+2)+i],b_C[j*(sN+2)+i],(b_N?b_N[j*(sN+2)+i]:0),(b_prevN?b_prevN[j*(sN+2)+i]:0),V2[j*(sN+2)+i]);
		}
#endif
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(ksave2);
			old_size++;
			H42_size++;
			if (debug_level==4)			
			{
				fr_ocl_get(b_C);
				fr_ocl_get(b_N);
				fr_ocl_get(b_prevN);
			}
		}
		else
		{
#endif
		if (alpha!=1.0)
		{
		    double *hs = new double[(sN + 2)*(_M + 2)];
		    memcpy(hs, b_C, (sN + 2)*(_M + 2)*sizeof(double));
		    oldC.push_back(hs);
			if (dt_alg==4)
				if (oldC.size()>=3)
					for (int i=0;i<(sN + 2)*(_M + 2);i++)
						H42[i].push_back((b_C[i] - oldC[oldC.size()-3][i]));
		}
#ifdef OCL
		}
#endif		
		tstep++;
		if (dt_alg==4)
			if ((tstep%20)==0)
			{
#ifdef OCL
				if (use_ocl)
				{
					fr_ocl_call(kcleanup);
					H42_size=2;
				}
				else
#endif
				saved_cleanup();
			}
#ifdef OCL
		if (use_ocl)
			queue->Finish();
#endif		
		if (debug_level==4)
		{
			fflush(stdout);
			system("clear");
		}
	}
	~TG_solver()
	{
		for (int i = 0;i < oldC.size();i++)
			delete oldC[i];
		if (dt_alg==3)
		{
			for (int i=0;i<(sN + 2)*(_M + 2);i++)
			{
				Htmp1[i].clear();
				Htmp2[i].clear();
			}
			delete [] Htmp1;
			delete [] Htmp2;
			if (beta2!=0.0)
			{
				for (int i=0;i<(sN + 2)*(_M + 2);i++)
				{
					Htmp12[i].clear();
					Htmp22[i].clear();
				}
				delete [] Htmp12;
				delete [] Htmp22;
			}
		}
		if (dt_alg==4)
		{
			for (int i=0;i<(sN + 2)*(_M + 2);i++)
				H42[i].clear();
			delete [] H42;
			delete [] approx_built2;
			delete [] approx_cleared2;
			for (int i=0;i<(sN + 2)*(_M + 2);i++)
			{
				Happ1[i].ias.clear();
				Happ2[i].ias.clear();
				Happ1[i].ibks.clear();
				Happ2[i].ibks.clear();
				Happ12[i].ias.clear();
				Happ22[i].ias.clear();
				Happ12[i].ibks.clear();
				Happ22[i].ibks.clear();
			}
			delete [] Happ1;
			delete [] Happ2;
			delete [] Happ12;
			delete [] Happ22;
		}
		delete[] b_C;
		delete[] V2;
		delete [] Al;
		delete [] Bt;
		delete [] Om;
	}
};
template <int func,int fixed_d,int kform> class TG_solver_:public TG_solver {
public:
	// non-linear filtration coefficient 
	__attribute__((always_inline)) double  KK(int i)
	{
		if (kform == 0)
			return ka[0];
		else
		{
		    double c = C[i] / C1;
		    return ka[0]+ka[1]*c+ka[2]*c*c+ka[3]*c*c*c+ka[4]*c*c*c*c+ka[5]*c*c*c*c*c;
		}
	}
	// non-linear diffusion coefficient
	__attribute__((always_inline)) 	double  D(int i)
	{
		if (fixed_d == 0)
		{
			double v = 0.0, _k;
			if (m3d_c == 0)
				v = sqrt(sqv(i, m3d_x));
			if (m3d_c == 1)
				v = sqrt(sqv(m3d_x, i));
			_k = KK(i) / k;
			return lambda*v*_k;
		}
		else
			return d_value;
	}
	double testing_rpC(int i,int j,double t)
	{
		double f = i*sdL;
		double p = (j - 0.5)*dLx;
		double T=t*tau;
		double Dv;
		int sx=m3d_x;
		if (m3d_c==0)
		{
			m3d_x=j;
			Dv=D(i);
			m3d_x=sx;
		}
		if (m3d_c==1)
		{
			m3d_x=i;
			Dv=D(j);
			m3d_x=sx;
		}
		return sigma*testing_k*pow(T,2.0-alpha*func_power)-2.0*T-2.0*sqv(i,j)*(Dv*(f*f+p*p)-f*p*p);
	}
	__attribute__((always_inline)) 	double  g(double t)
	{
		if (func==1)
			return sqrt(t);
		if (func==2)
			return t*t;
		if (func == 3)
			return pow(t,func_power);
		if (func==4)
			return exp(t*func_power2)*pow(t,func_power);
		return t;
	}
	// should return kf(n)(t) given kf(n-1)(t)=der_nm1
	// for n=0 return f(t)
	__attribute__((always_inline)) 	double  inv_g_der(double t,int n,double der_nm1)
	{
		if (func==1) // sqrt(t) -> f=t^2 -> df/dt= 2t, d2f/dt=2, d(n>=3)f/dt=0
		{
			if (n==0)
				return t*t;
			if (n==1)
				return 2*t*(der_nm1/(t*t));
			if (n==2)
				return 2*(der_nm1/(2*t));
			return 0.0;
		}
		if (func==2) // t^2 -> f=sqrt(t) -> df/dt=0.5*t(-0.5), d(n+1)f/dt=(d(n)f/dt)*(0.5-n)/t
		{
			if (n==0)
				return sqrt(t);
			if (n==1)
				return (0.5/sqrt(t))*(der_nm1/sqrt(t));
			return der_nm1*(0.5-n+1)/t;
		}
		if (func == 3) // t^k -> f=t^(1/k) -> df/dt=(1/k)*t((1/k)-1), d(n+1)f/dt=(d(n)f/dt)*((1/k)-n)/t
		{
			if (n == 0)
			    return pow(t,1.0/func_power);
			if (n == 1)
			    return ((1.0/func_power) * pow(t,(1.0/func_power)-1))*(der_nm1 / pow(t,1.0/func_power));
			return der_nm1*((1.0/func_power) - n + 1) / t;
		}
		if (func==4)
		{
			printf("no inverse derivative for func4\n");
			exit(1);
		}
		return ((n==0)?t:((n==1)?(der_nm1/t):0));
	}
	//int((g(t_j+ja)-g(tau))^(-alpha),tau=t0,...,t1)
	// n steps of 1-order numerical integration
	double  __kb(int n,double t0,double t1,double gtj,double alpha)
	{
		double ret=0.0;
		double step=(t1-t0)/n;
		for (int i = 0;i < n;i++)
		{
			double p1 = gtj - g(t0 + (i+0.5)*step);
			p1 = pow(p1, -alpha);
			perf_count[3]++;
			ret += step*p1;
		}
		return ret;
	}
	double  ___kb(int n,double t0,double t1,double gtj,double alpha)
	{
		double f=pow(gtj-g(t0),-alpha),f1,f2,f3,f4;
		double sum=0.0,v;
		double t=t0;
		double step=(t1-t0)/n;
		do
		{
			f1 = pow(gtj - g(t + 0.5*step), -alpha);
			f3 = pow(gtj - g(t + step/3.0), -alpha);
			f4 = pow(gtj - g(t + 2.0*step/3.0), -alpha);
			f2 = pow(gtj - g(t + step), -alpha);
			v=(f+3*f3+3*f4+f2)*step/8.0;
			if (!finite(v))
				break;
			sum += v;
			t += step;			
			perf_count[3]+=4;
			if ((f1 / f) < (f2 / f1))
				step /= (f2/f1)/(f1/f);
			while ((t + step) > t1)
				step /= 2.0;
			f = f2;
		}
		while ((t1-t)>int_err*int_err*int_err);
		return sum;
	}
	//int((g(t_j+ja)-g(tau))^(-alpha),tau=t0,...,t1)
	// using recursive subdivision
	double _kb(int n, double t0, double t1, double gtj,int sing,double alpha)
	{
		double v1, v2;
		v1 = __kb(n, t0, t1, gtj,alpha);
		v2 = __kb(n*2, t0, t1, gtj,alpha);
		if ((t1 - t0) < int_err) 
		{
			if (sing)
				return ___kb(n*16,t0,t1,gtj,alpha);
			return 0.5*(v1+v2);
		}
		if (fabs(v2 - v1)>int_err)
		{
			v1 = _kb(n, t0, 0.5*(t0 + t1), gtj,sing,alpha);
			v2 = _kb(n, 0.5*(t0 + t1), t1,gtj,sing,alpha);
			return v1 + v2;
		}
		else
			return v2;
	}
	// using taylor series on gtj-g(x) in g(t1)
	double _kb_row(double t0, double t1, double gtj,double alpha)
	{
		double sum = 0.0, v = 0.0, v2, v3, v4;
		int i = 0;
		v = inv_g_der(gtj, 1, inv_g_der(gtj,0,1));
		v *= pow(gtj, 1 - alpha);
		// ((a-g(d))^(n-b+1)-(a-g(c))^(n-b+1))/(n-b+1)
		v3 = pow(1.0 - g(t1) / gtj, 1 - alpha);
		v4 = pow(1.0 - g(t0) / gtj, 1 - alpha);
		v2 = -(v3 - v4 ) / (1 - alpha);
		sum += v2*v;
		while (fabs(v*v2)>int_err)
		{
			i++;
			//(((-1)^n/n!)*f(n+1)(a))*a^(n-b+1)
			v = inv_g_der(gtj, i + 1, v);
			v *= -gtj / i;
			v3 *= 1.0 - g(t1) / gtj;
			v4 *= 1.0 - g(t0) / gtj;
			v2 = -(v3-v4) / (i - alpha + 1);
			sum += v2*v;
			perf_count[3]++;
		}
		return sum;
	}
	// using newton binomial on gtj and taylor series of f'(x) in g(t1)
	std::vector< std::vector<double> > row2_precalc;
	std::vector< std::vector<double> > row2_precalc2;
	double _kb_row2(double t0, double t1, double gtj,int idx,double alpha,std::vector< std::vector<double> >*row2_precalc)
	{
		double sum = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4, v5,v6,v7;
		double gt0 = g(t0), gt1 = g(t1);
		int i = 0, m,found;
		// (a,n)*(-1)^n
		v1 = 1;
		v2 = pow(gtj, -alpha) * gt1;
		v6 = gt0 / gt1; 
		do
		{
			// (a,n)*(-1)^n
			if (i != 0) v1 *= -(-alpha - i + 1) / i;
			// (a^(-b-n)*g(t1)^(n+1)
			if (i!=0) v2 *=gt1/gtj;
			if (i != 0) v6 *= gt0 / gt1;
			// integrate(f'(x)x^n,x,g(c),g(d))/g(t1)^(n+1)
			found=0;
			if (row2_precalc->size()>idx)
				if (row2_precalc[0][idx].size()>i)
				{
					v3=row2_precalc[0][idx][i];
					found=1;
				}
			if (found==0)
			{
				if (i == 0)
					v3 = (t1-t0)/gt1;
				else
				{
					v3 = 0.0;
					m = 0;
					// f(m+1)(gd)/m!
					v4 = inv_g_der(gt1, 1, inv_g_der(gt1,0,0));
					// I(n,m)=integrate(x^n*(x-g(t1))^m,x,g(t0),g(t1));
					// I(n,0)=(1/n+1)*(g(t1)^(n+1)-g(t0)^n+1)
					v5 = (1.0 / (i + 1.0))*(1-v6);
					v3 += v4*v5;
					v7 = 1;
					while (fabs(v4*v5) > (v3*int_err))
					{
						m++;
						// (f(m+1)(gd)/m!)
						v4 = inv_g_der(gt1, m + 1, v4);
						v4 /= m;
						v4 /= ((i + m + 1) / (gt1*m));
						if (m == 1)
							v7 = (1 / (gt1*m))*(v6*(gt0 - gt1));
						else
							v7 *= (gt0 - gt1)*((m - 1.0) / m)*((i + m ) / (gt1*(m-1)));
						// I(n,m)=-(I(n,m-1)+(1/(g(t1)*m))*(g(t0)^(n+1)*(g(t0)-g(t1))^m))/((n+m+1)/(g(t1)*m)
						v5 = -(v5 + v7);
						v3 += v4*v5;
						perf_count[3]++;
					}
				}
				while (row2_precalc->size()<(idx+1))
					row2_precalc->push_back(std::vector<double>());
				while (row2_precalc[0][idx].size()<(i+1))
					row2_precalc[0][idx].push_back(0);
				row2_precalc[0][idx][i]=v3;
			}
			sum += v1*v2*v3;
			perf_count[3]++;
			i++;
			if (i>integr_max_niter)
			    break;
			if (dt_eps>=1) // fixed number of terms
				if (i<dt_eps)
					continue;
				else
					break;
		} while (fabs(v1*v2*v3) > int_err);
		return sum;
	}
	// calc row with coefs - inner integrals for t0,t1 
	double _kb_row2_fixed_coefs(double gtj, double *_coefs,int n,double alpha)
	{
		double sum = 0.0, v1;
		double *c=_coefs;
		double i = 0;
		int ii = 0;
		v1 = pow(gtj, -alpha);
		do
		{
			sum += v1*(c++)[0];
			i+=1.0;
			ii++;
			v1 *= -(-alpha - i + 1.0) / (i*gtj);
			if (!finite(v1)) break;
		} while (ii!=n);
		return sum;
	}
	//b(j,s,ja,sa) = int((g(t_j+ja)-g(tau))^(-alpha),tau=t_s+sa,...,ts+1/2+sa)
	double kb( int s, double ja, double sa,double l=0.5,int a=0)
	{
		double alpha=this->alpha;
		double Da=this->Da;
		if (dt_alg==5) // local approximation of Caputo derivative
			return 1.0;
		if (a==1)
		{
			alpha=this->alpha2;
			Da=this->Da2;
		}
		if (alpha == 1.0)
			if ((s+sa+l)==(tstep+ja))
				return 1.0;
			else
				return 0.0;
		if (alpha==0.0)
			return 0.0;
		int idx=0; // bit 0 - ja==0.5 or 1.0, bit 1 - sa==0.0 or 0.5, bit 2 - l==0.5 or 1.0
		if (ja==1.0) idx+=1; 
		if (sa==0.5) idx+=2;
		if (l==1.0) idx+=4;
		double v;
		std::vector<double> *precalc=&kb_precalc[idx];
		std::vector< std::vector<double> >*r2_precalc=&row2_precalc;
		if (a==1)
		{
			precalc=&kb_precalc2[idx];
			r2_precalc=&row2_precalc2;
		}
		int j=tstep;
		if ((j!=old_step)||(precalc->size()<=s)) // do precalculations
		{
			int t1=GetTickCount();
			unsigned __int64 c1, c2;
			c1 = cpu_counter();
			if (a==0)
			for (int i=0;i<8;i++)
				kb_precalc[i].clear();
			if (a==1)
			for (int i=0;i<8;i++)
				kb_precalc2[i].clear();
			int ii=0;
			if (dt_alg==2) // fill with 0 for restrictive summing
			for (int i=0;i<=j;i++)
			    for (int _idx=0;_idx<8;_idx++)
			    {
				if (a==0)
					kb_precalc[_idx].push_back(0.0);
				if (a==1)
					kb_precalc2[_idx].push_back(0.0);
			    }
			// compute integrals values - from j down to 0 for restrictive summing, from 0 to j for other algorithms
			for (int i=((dt_alg==2)?j:0);((dt_alg==2)?(i>=0):(i<=j));((dt_alg==2)?(i--):(i++)))
			{
				double maxv=0;
				for (double _ja=0.5;_ja<=1.0;_ja+=0.5)
					for (double _sa=0.0;_sa<=0.5;_sa+=0.5)
						for (double _l=0.5;_l<=1.0;_l+=0.5)
						{
							int _idx=0;
							if (_ja==1.0) _idx+=1; 
							if (_sa==0.5) _idx+=2;
							if (_l==1.0) _idx+=4;
							v=0;
							if ((dt_alg==4)&&(i!=j)) goto save; // only i==j for two-series expansion summing
							if ((dt_alg==3)&&((j-i)>1)) goto save; // only i==j,j-1 for single-series expansion summing
							if (tau*(i + _sa + _l) <= tau*(j + _ja))
							{
								if (int_alg == 0)
									v = Da*_kb(1, tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), (tau*(i + _sa + _l)) == (tau*(j + _ja)),alpha);
								if (int_alg == 1)
									v = Da*_kb_row(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)),alpha);
								if (int_alg == 2)
									v = Da*_kb_row2(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), ii,alpha,r2_precalc);
								if (int_alg == 6)
									if (((tau*(j + _ja)) - tau*(i + _sa + _l)) <= tau)
									{
										v = Da*_kb_row(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)),alpha);										
										if (dt_alg==3) // to genetate v3precalc
											_kb_row2(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), ii,alpha,r2_precalc);
									}
									else
										v = Da*_kb_row2(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), ii,alpha,r2_precalc);
								if (int_alg == 3)
								{
									if (((tau*(j + _ja)) - tau*(i + _sa + _l)) < a3k*tau)
										v = Da*_kb_row(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)),alpha);
									else
									{
										int niter = perf_count[3];
										v = Da*_kb_row2(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), ii,alpha,r2_precalc);
										niter = perf_count[3] - niter;
										if ((j - i ) == a3k)
										{
											int niter2 = perf_count[3];
											double v2 = Da*_kb_row(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)),alpha);
											niter2 = perf_count[3] - niter2;
											if (niter2 > niter)
											{
												if (a3k != 2)
													a3k--;
											}
											else
												a3k++;
										}
									}
								}
								if (int_alg == 4)
								{
									if ((tau*(i + _sa) <= a3k*tau) && (tau*(i + _sa + _l) != tau*(j + _ja)))
									{
										int niter = perf_count[3];
										v = Da*_kb_row2(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), ii,alpha,r2_precalc);
										niter = perf_count[3] - niter;
										if (tau*(i + _sa) == a3k*tau)
										{
											int niter2= perf_count[3];
											double v2 = Da*_kb_row(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)),alpha);
											niter2 = perf_count[3] - niter2;
											if (niter2 > niter)
												a3k++;
											else
												if (a3k != 1)
													a3k--;
										}
									}
									else
										v = Da*_kb_row(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)),alpha);
								}
								if (int_alg == 5) // testing
								{
									double v2,v3;
									int niter,niter2,niter3;
									niter = perf_count[3];
									v = Da*_kb(1, tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), (tau*(i + _sa + _l)) == (tau*(j + _ja)),alpha);
									niter = perf_count[3] - niter;
									niter2 = perf_count[3];
									v2 = Da*_kb_row(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)),alpha);
									niter2 = perf_count[3] - niter2;
									niter3 = perf_count[3];
									v3 = Da*_kb_row2(tau*(i + _sa), tau*(i + _sa + _l), g(tau*(j + _ja)), ii,alpha,r2_precalc);
									niter3 = perf_count[3] - niter3;
									perf_count_d[0] += (v - v2)*(v - v2);
									perf_count_d[1] += (v - v3)*(v - v3);
									FILE *fi = fopen("log3.txt", "at");
									fprintf(fi, "%g %g %g a1 %g %d a2 %g %d a3 %g %d\n", tau*(i + _sa), tau*(i + _sa + _l), tau*(j + _ja), v, niter, v2, niter2, v3, niter3);
									fclose(fi);
								}
							}
save:							
							if (dt_alg!=2)
							{
							    if (a==0)
								kb_precalc[_idx].push_back(v);
							    if (a==1)
								kb_precalc2[_idx].push_back(v);
							}
							else
							{
							    if (a==0)
								kb_precalc[_idx][i]=v;
							    if (a==1)
								kb_precalc2[_idx][i]=v;
							}
							if (v!=0.0) if (fabs(v)>maxv) maxv=v;
							ii++;
						}
			    // for restrictive summing calculate only integrals with |I|>dt_eps
			    if (dt_alg==2)
				if (maxv<dt_eps)
				    break;
			}
			old_step=j;
			c2 = cpu_counter();
			perf_count[0] += GetTickCount() - t1;
			perf_count[2] += c2-c1;
		}
		v=precalc[0][s];
		return v;
	}
	// c0 - dL, c1 - dLx
	// AHi-1 - RHi + BHi+1 = Psi
	__attribute__((always_inline))  double  A_(int i)
	{
		if (m3d_c==0)
			return (sqv(i,m3d_x)/dL)*((0.5*(D(i-1)+D(i))/dL)+0.5);
		if (m3d_c==1)
			return (sqv(m3d_x,i)/(dL*dL))*0.5*(D(i-1)+D(i));
		return 0.0;
	}
	__attribute__((always_inline)) 	double  B(int i)
	{
		if (m3d_c==0)
			return (sqv(i,m3d_x)/dL)*((0.5*(D(i+1)+D(i))/dL)-0.5);
		if (m3d_c==1)
			return (sqv(m3d_x,i)/(dL*dL))*0.5*(D(i+1)+D(i));
		return 0.0;
	}
	__attribute__((always_inline)) 	double R(int i)
	{
		if (m3d_c==0)
			return (sigma/tau)*(kb(tstep,0.5,0.0)+((dt_alg!=5)?beta2*kb(tstep,0.5,0.0,0.5,1):0))+A_(i)+B(i);
		if (m3d_c==1)
			return (sigma/tau)*(kb(tstep,1.0,0.5)+((dt_alg!=5)?beta2*kb(tstep,1.0,0.5,0.5,1):0))+A_(i)+B(i);
		return 0.0;
	}
	// alpha coefficients
	void al1()
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kAl);
			return;
		}
#endif
		if (m3d_c==0)
			Al[1] = 0;
		if (m3d_c==1)
		{			
			Al[1]=1;
			if (testing)
				Al[1]=0;
		}
		if (fixedV==NULL)
		    for (int i = 1;i < N;i++)
			Al[i + 1] = B(i) / (R(i) - A_(i)*Al[i]);
		else
		{
		    for (int i = 1;i < N;i++)
		    {
			int found=0;
			if (m3d_c==0)
			    if (fixed_mask[idx(i,m3d_x)]!=1e300)
				    found=1;
			if (m3d_c==1)
			    if (fixed_mask[idx(m3d_x,i)]!=1e300)
				    found=1;
			if (found==0)
			    Al[i + 1] = B(i) / (R(i) - A_(i)*Al[i]);
			else
			    Al[i+1]=0;
		    }
		}
	}
	// right part
	void Om1()
	{
#ifdef OCL
		if (use_ocl)
			return;
#endif
		row U_,N_,prevN_;
		double k = 1.0;
		if (k_derivative)
			k = pow(2.0, alpha - 1.0);
		unsigned long t1 = GetTickCount();
		std::vector<double*> *old;
		get_F_oldF(&U_,&N_,&prevN_, &old);
		if (m3d_c == 0)
			for (int i = 1;i < N;i++)
				Om[i] = -(sigma / tau)*(kb(tstep, 0.5, 0.0)+((dt_alg!=5)?(beta2*kb(tstep,0.5,0.0,0.5,1)):((tau / sigma)*0.5*(1-alpha+beta2*(1-alpha2))/(alpha+beta2*alpha2))))*U_[i];
		if (m3d_c == 1)
			for (int i = 1;i < N;i++)
				Om[i] = -(sigma / tau)*(kb(tstep, 1.0, 0.5)+((dt_alg!=5)?(beta2*kb(tstep,1.0,0.5,0.5,1)):((old[0][old->size()-1][idx(m3d_x, i)]/U_[i])*(tau / sigma)*0.5*(1-alpha+beta2*(1-alpha2))/(alpha+beta2*alpha2))))*U_[i];
		for (int i = 1;i < N;i++)
		    Om[i]+=_S*(mode1d?1.0:0.5);
		if (massN)
		    for (int i = 1;i < N;i++)
				Om[i] += (1.0 / tau)*(N_[i]-prevN_[i]);
		if (testing)
		for (int i = 1;i < N;i++)
		{
			double rp;
			if (m3d_c==0)
				rp=testing_rpC(i,m3d_x,tstep+0.5);
			if (m3d_c==1)
				rp=testing_rpC(m3d_x,i,tstep+1.0);
			Om[i]-=0.5*rp;
		}
		if (((alpha != 1.0)||((beta2!=0.0)&&(alpha2!=1.0)))&&(dt_alg!=5))
#pragma omp parallel for
			for (int i = 1;i < N;i++)
			{
				int id;
				double time_sum = 0;
				if (m3d_c == 0)
					id = idx(i, m3d_x);
				if (m3d_c == 1)
					id = idx(m3d_x, i);
				if ((dt_alg == 0)||(dt_alg==2)||(dt_alg==3)) // full step integration
					if (old->size() >= 2)
					{
						double kv=1, diff=1,kv2=1;
						if (old->size() >= 3)
						{
							if (dt_alg!=3)
							for (int t = ((old->size() - 3)/2)*2;t >=0 ;t -= 2)
							{
								int br=0;
								if (m3d_c == 0)
									kv = kb(t / 2, 0.5, 0.0, 1.0);
								if (m3d_c == 1)
									kv = kb(t / 2, 1.0, 0.0, 1.0);
								if (dt_alg==2) // restricted summing
									if (fabs(kv)<dt_eps)
										br++;
								if (beta2!=0.0)
								{
								    if (m3d_c == 0)
									kv2 = kb(t / 2, 0.5, 0.0, 1.0,1);
								    if (m3d_c == 1)
									kv2 = kb(t / 2, 1.0, 0.0, 1.0,1);
								    if (dt_alg==2) // restricted summing
									if (fabs(kv2)<dt_eps)
										br++;
								    kv+=kv2*beta2;
								}
								if ((beta2==0)&& br)
									break;
								if ((beta2!=0)&& (br==2))
									break;
								diff = (old[0][t + 2][id] - old[0][t][id]);
								time_sum += kv*diff;
							}
							if (dt_alg==3)// integration by series expansion
							{
								int t=((old->size() - 3)/2)*2;
								std::vector<double> *Htmp,*Htmp_2;
								int pr_idx;
								double mult;
								double gtj;
								if (m3d_c==0)
								{
									pr_idx=1;
									gtj=g((tstep+0.5)*tau);
									Htmp=Htmp1;
									Htmp_2=Htmp12;
								}
								if (m3d_c==1)
								{
									pr_idx=5;
									gtj=g((tstep+1.0)*tau);
									Htmp=Htmp2;
									Htmp_2=Htmp22;
								}
								diff = (old[0][t + 2][id] - old[0][t][id]);
								// update Sn
								mult = g(((t/2)+1)*tau);
								pr_idx+=8*(t/2);
								for (int j=0;j<Htmp[id].size();j++)
								if (row2_precalc[pr_idx].size()>j)
								{
									Htmp[id][j]+=mult*row2_precalc[pr_idx][j]*diff;
									mult *= g(((t/2)+1)*tau);
								}
								time_sum=Da*_kb_row2_fixed_coefs(gtj,(double *)&Htmp[id][0],Htmp[id].size(),alpha);
								if (beta2!=0.0)
								{
									mult = g(((t/2)+1)*tau);
									for (int j=0;j<Htmp_2[id].size();j++)
									if (row2_precalc2[pr_idx].size()>j)
									{
										Htmp_2[id][j]+=mult*row2_precalc2[pr_idx][j]*diff;
										mult *= g(((t/2)+1)*tau);
									}
									time_sum+=beta2*Da2*_kb_row2_fixed_coefs(gtj,(double *)&Htmp_2[id][0],Htmp_2[id].size(),alpha2);
								}
							}
						}
						if (m3d_c == 1)
						{
							id = idx(m3d_x, i);
							kv = kb((old->size() - 1) / 2, 1.0, 0.0, 0.5)+beta2*kb((old->size() - 1) / 2, 1.0, 0.0, 0.5,1);
							diff = (old[0][old->size() - 2][id] - old[0][old->size() - 1][id]);
							time_sum += kv*diff;
						}
					}
				if (dt_alg==4)
				{
					if (H42[id].size()!=1)
					{
						double ts;
						for (int a = approx_built2[id];a <= H42[id].size() - 2;a++)
						{
							if (m3d_c==1)
							{
								calc_va(1.0-alpha,1,a+approx_cleared2[id],&H42[id][0]-approx_cleared2[id], &time_sum,NULL, Happ2[id].ias, Happ2[id].ibks, 2.0, 1.0);
								if (beta2!=0.0)
									calc_va(1.0-alpha2,1,a+approx_cleared2[id],&H42[id][0]-approx_cleared2[id], &ts,NULL, Happ22[id].ias, Happ22[id].ibks, 2.0, 1.0);
							}
							if (m3d_c==0)
							{
								calc_va(1.0-alpha,1,a+approx_cleared2[id], &H42[id][0]-approx_cleared2[id], &time_sum,NULL, Happ1[id].ias, Happ1[id].ibks, 1.5, 0.5);
								if (beta2!=0.0)
									calc_va(1.0-alpha2,1,a+approx_cleared2[id], &H42[id][0]-approx_cleared2[id], &ts,NULL, Happ12[id].ias, Happ12[id].ibks, 1.5, 0.5);			
							}
						}
						if (m3d_c==1)
							approx_built2[id] = H42[id].size() - 1;
						time_sum*=Da*(pow(tau,1.0-alpha)/(1.0-alpha));
						time_sum+=beta2*Da2*ts*(pow(tau,1.0-alpha2)/(1.0-alpha2));
					}
					if (m3d_c==1)
					{
						double kv = kb((old->size() - 1) / 2, 1.0, 0.0, 0.5)+beta2*kb((old->size() - 1) / 2, 1.0, 0.0, 0.5,1);
						double diff = (old[0][old->size() - 2][id] - old[0][old->size() - 1][id]);
						time_sum += kv*diff;
					}
				}
				if (dt_alg == 1) // half-step integration
					if (old->size() >= 2)
					{
						double kv, diff;
						if (m3d_c == 0)
							for (int t = 0;t < old->size() - 2;t += 2)
							{
								kv = kb(t / 2, 0.5, 0.0, 1.0)+beta2*kb(t / 2, 0.5, 0.0, 1.0,1);
								diff = 0.5*(old[0][t + 2][id] - old[0][t][id]);
								time_sum += kv*diff;
							}
						if (m3d_c == 1)
							for (int t = 0;t < old->size() - 1;t ++)
							{
								kv = kb(t / 2, 1.0, 0.5*(t%1))+beta2*kb(t / 2, 1.0, 0.5*(t%1),0.5,1);
								diff = (old[0][t + 1][id] - old[0][t][id]);
								time_sum += kv*diff;
							}
					}
				Om[i] += (sigma / tau)*time_sum;
			}
		perf_count[1] += GetTickCount() - t1;
	}
	// beta coeffients
	void bt1()
	{
#ifdef OCL
		if (use_ocl)
		{
			fr_ocl_call(kBt);
			return;
		}
#endif	
		if (m3d_c==0)	
			Bt[1]=C1;
		if (m3d_c==1)		
			Bt[1]=0.0;
		if (testing)
		{
			double v;
			if (m3d_c == 0)
				v = testing_solutionC(0, m3d_x,tstep+1.0);
			if (m3d_c == 1)
				v = testing_solutionC(m3d_x,0,tstep+1);
			Bt[1]=v;
		}
		if (fixedV==NULL)
		    for (int i = 1;i < N;i++)
		    {
			if ((Al[i + 1]==0.0)&&(B(i)==0.0))
				Bt[i + 1] = (A_(i)*Bt[i] - Om[i])/ (R(i) - A_(i)*Al[i]);
			else			
				Bt[i + 1] = (Al[i + 1] / B(i)) * (A_(i)*Bt[i] - Om[i]);
		    }
		else
		{
		    for (int i = 1;i < N;i++)
		    {
			int found=0;
			double val=0.0;
			if (m3d_c==0)
			    if (fixed_mask[idx(i,m3d_x)]!=1e300)
			    {
				    val=fixed_mask[idx(i,m3d_x)];
				    found=1;
			    }
			if (m3d_c==1)
			    if (fixed_mask[idx(m3d_x,i)]!=1e300)
			    {
				    val=fixed_mask[idx(m3d_x,i)];
				    found=1;
			    }
			if (found==0)
		        {
		    	    if ((Al[i + 1]==0.0)&&(B(i)==0.0))
				Bt[i + 1] = (A_(i)*Bt[i] - Om[i])/ (R(i) - A_(i)*Al[i]);
			    else			
				Bt[i + 1] = (Al[i + 1] / B(i)) * (A_(i)*Bt[i] - Om[i]);
			}
			else
			    Bt[i+1]=val;
		    }
		}
	}
	TG_solver_(double tau_m,double a,double kdk,int k_der=1,double dv=0.1,double _kk=-1,double l=5,double fp=1.0,double _L=-1,double _H=-1,double *fixed_initial=NULL,int nfixed_initial=0,double _Lx=-1,int noinit=0,double __C1=1e300,double __S=1e300,double _func_power2=0,double _b2=0.0,double _a2=0.0):TG_solver(tau_m,a,kdk,k_der,dv,_kk,l,fp,_L,_H,fixed_initial,nfixed_initial,_Lx,noinit,__C1,__S,_func_power2,_b2,_a2)
	{
		if (func==0)
			func_power=1;
		if (func==1)
			func_power=0.5;
		if (func==2)
			func_power=2;
	}
	// opencl code
#ifdef OCL
	// initialize OpenCL
	void init_opencl()
	{
		int iv;
		prg = new OpenCL_program(1);
		queue = prg->create_queue(device, 0);
		if (dt_alg==3) lBS=(int)dt_eps;
		{
			char *text = new char[(strlen(frt_input_opencl_text)+strlen(frt_approx_opencl_text) + strlen(ints_opencl_text) + strlen(fr_d_t_g_row_opencl_text)) * 2];
			ia_approximation ia;
			ibk_approximation ibk;
			sprintf(text, frt_input_opencl_text, ((double_ext == 0) ? "cl_amd_fp64" : "cl_khr_fp64")
				,tau
				, ((char *)&L) - (char *)this
				, dL
				, ((char *)&sL) - (char *)this
				, sdL
				, alpha
				, beta2
				, alpha2
				, func_power
				, Da
				, Da2
				, sigma
				, ((char *)&lambda) - (char *)this
				, ((char *)&C0) - (char *)this
				, ((char *)&C1) - (char *)this
				, ((char *)&Q) - (char *)this
				, ((char *)&H) - (char *)this
				, ((char *)&Lx) - (char *)this
				, dLx
				, ((char *)&k_derivative) - (char *)this
				, d_value
				, ((char *)&k_der_k) - (char *)this
				, testing_k
				, func
				, int_err
				, dt_eps
				, integr_max_niter
				, int_alg
				, massN_a
				, massN_b
				, testing
				, dt_alg
				, massN	
				, sizeof(ia_approximation)
				, ((char *)&ia.min) - (char *)&ia
				, ((char *)&ia.nterms) - (char *)&ia
				, ((char *)&ia.c[0]) - (char *)&ia
				, ((char *)&ia.a) - (char *)&ia
				, ((char *)&ia.b) - (char *)&ia
				, sizeof(ibk_approximation)
				, ((char *)&ibk.min) - (char *)&ibk
				, ((char *)&ibk.max) - (char *)&ibk
				, ((char *)&ibk.b) - (char *)&ibk
				, ((char *)&ibk.nterms) - (char *)&ibk
				, ((char *)&ibk.old) - (char *)&ibk
				, ((char *)&ibk.c[0]) - (char *)&ibk
				, ((char *)&ibk.a) - (char *)&ibk
				, ((char *)&ibk._b) - (char *)&ibk
				, approx_eps
				, AK
				, MAXAK
				, 1
				, 1
				, eps
				, Gamma(2.0 - alpha)
				,N
				,lBS);
			strcat(text, frt_approx_opencl_text);
			strcat(text, ints_opencl_text);
			strcat(text, fr_d_t_g_row_opencl_text);
			//printf("%s\n",text);fflush(stdout);
			prog = prg->create_program(text);
			delete[] text;
			//printf("ocl code compiled\n");fflush(stdout);
		}
		kAl = prg->create_kernel(prog, "Al");
		kBt = prg->create_kernel(prog, "Bt");
		kOm = prg->create_kernel(prog, "Om");
		kU = prg->create_kernel(prog, "U");
		kN1 = prg->create_kernel(prog, "N1");
		kcalc_kb = prg->create_kernel(prog, "calc_kb");
		ksave1 = prg->create_kernel(prog, "save1");
		ksave2 = prg->create_kernel(prog, "save2");
		kcleanup = prg->create_kernel(prog, "cleanup");
		bS = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(TG_solver_<func,fixed_d,kform>), (void *)this);
		bAl = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 2)*(_M + 2)*sizeof(double), Al);
		bBt = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 2)*(_M + 2)*sizeof(double), Bt);
		bOm = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 2)*(_M + 2)*sizeof(double), Om);
		bU = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 2)*(_M + 2)*sizeof(double), b_C);
		if (massN)
		{
			bN = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 2)*(_M + 2)*sizeof(double), b_N);
			bprev_N = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 2)*(_M + 2)*sizeof(double), b_prevN);
		}
		else
		{
			bN = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 2)*(_M + 2)*sizeof(double), NULL);
			bprev_N = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 2)*(_M + 2)*sizeof(double),NULL);
		}

		old_size=1;
		kbs_alloced=40; // for one time step
		boldU = prg->create_buffer(CL_MEM_READ_WRITE, 2*kbs_alloced*(sN + 2)*(_M + 2)*sizeof(double), NULL);
		queue->EnqueueWriteBuffer(boldU, oldC[0], 0, (sN + 2)*(_M + 2)*sizeof(double));
		
		bkbs = prg->create_buffer(CL_MEM_READ_WRITE, 5*kbs_alloced*sizeof(double), NULL);
		bkbs2 = prg->create_buffer(CL_MEM_READ_WRITE, 5*kbs_alloced*sizeof(double), NULL);
		bV2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 2)*(_M + 2)*sizeof(double), V2);
		
		if (dt_alg==3)
			iv=2*kbs_alloced*dt_eps;
		else
			iv=1;
		double *zv=new double[iv];
		memset(zv,0,iv*sizeof(double));
		bv3precalc = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, iv*sizeof(double), zv);
		bv3precalc2 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, iv*sizeof(double), zv);
		delete [] zv;
		
		if (dt_alg==3)
			iv=2*dt_eps*(sN + 3)*(_M+3);
		else
			iv=1;
		zv=new double[iv];
		memset(zv,0,iv*sizeof(double));
		bHtmp1 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, iv*sizeof(double), zv);
		bHtmp2 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, iv*sizeof(double), zv);
		bHtmp12 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, iv*sizeof(double), zv);
		bHtmp22 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, iv*sizeof(double), zv);
		delete [] zv;
		
		alloced_approx_per_row = 40;
		b_ibk1 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), NULL);
		b_ias1 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), NULL);
		b_ibk2 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), NULL);
		b_ias2 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), NULL);		
		b_ibk12 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), NULL);
		b_ias12 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), NULL);
		b_ibk22 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), NULL);
		b_ias22 = prg->create_buffer(CL_MEM_READ_WRITE, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), NULL);
		
		cl_int *ziv=new cl_int[ (sN + 3)*(_M+3)];
		memset(ziv,0, (sN + 3)*(_M+3)*sizeof(cl_int));
		b_nibk1 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		b_nias1 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		b_nibk2 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		b_nias2 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		b_nibk12 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		b_nias12 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		b_nibk22 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		b_nias22 = prg->create_buffer(CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), ziv);
		delete [] ziv;
		an_ibk=new cl_int[(N+3)*(_M+3)];
		an_ias=new cl_int[(N+3)*(_M+3)];
		
		if (dt_alg==4)
		{
			b_ab2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), approx_built2);
			b_ac2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*sizeof(cl_int), approx_cleared2);
		}
		else
		{
			b_ab2 = prg->create_buffer(CL_MEM_READ_WRITE, sizeof(int), NULL);
			b_ac2 = prg->create_buffer(CL_MEM_READ_WRITE, sizeof(int), NULL);
		}
		
		H42_per_row=40;
		H42_size=1;
		zv=new double[(sN + 3)*(_M+3)*H42_per_row];
		memset(zv,0,(sN + 3)*(_M+3)*H42_per_row*sizeof(double));
		b_H42 = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*H42_per_row*sizeof(double), zv);
		delete [] zv;
		//printf("ocl initialization completed\n");fflush(stdout);
	}
	OpenCL_buffer *fr_ocl_buffer(double *B)
	{
		if (B == Al) return bAl;
		if (B == Bt) return bBt;
		if (B == Om) return bOm;
		if (B == V2) return bV2;
		if (B == b_C) return bU;
		if (B == b_N) return bN;
		if (B == b_prevN) return bprev_N;
		return NULL;
	}
	void fr_ocl_get(double *B)
	{
		//printf("ocl get ");fflush(stdout);
		queue->EnqueueBuffer(fr_ocl_buffer(B), B);
		//printf(".");fflush(stdout);
		queue->Finish();
		//printf(".\n");fflush(stdout);
	}
	void fr_ocl_check_and_resize()
	{
		// integrals
		if (tstep>=(kbs_alloced-2))
		{
			//printf("ocl kbs resized ");fflush(stdout);			
			double *k=new double[5*kbs_alloced*2];
			double *oU=new double [ 2*2*kbs_alloced*(sN + 2)*(_M + 2)];
			queue->EnqueueBuffer(bkbs, k, 0,  5*kbs_alloced*sizeof(double));
			queue->EnqueueBuffer(boldU, oU, 0,  2*kbs_alloced*(sN + 2)*(_M + 2)*sizeof(double));
			delete bkbs;
			delete boldU;
			bkbs = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, 5*2*kbs_alloced*sizeof(double), k);
			boldU = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, 2*2*kbs_alloced*(sN + 2)*(_M + 2)*sizeof(double), oU);
			if (dt_alg==3)
			{
				double *v3=new double[(int)dt_eps*2*kbs_alloced*2];
				queue->EnqueueBuffer(bv3precalc, v3, 0,  (int)dt_eps*2*kbs_alloced*sizeof(double));
				delete bv3precalc;
				bv3precalc = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, (int)dt_eps*2*kbs_alloced*2*sizeof(double), v3);
				queue->Finish();
				delete [] v3;
			}
			{
				queue->EnqueueBuffer(bkbs2, k, 0,  5*kbs_alloced*sizeof(double));
				delete bkbs2;
				bkbs2 = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, 5*2*kbs_alloced*sizeof(double), k);
				if (dt_alg==3)
				{
					double *v3=new double[(int)dt_eps*2*kbs_alloced*2];
					queue->EnqueueBuffer(bv3precalc2, v3, 0,  (int)dt_eps*2*kbs_alloced*sizeof(double));
					delete bv3precalc2;
					bv3precalc2 = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, (int)dt_eps*2*kbs_alloced*2*sizeof(double), v3);
					delete [] v3;
				}
			}
			kbs_alloced*=2;		
			queue->Finish();
			delete [] k;
			delete [] oU;
			//printf(" %d\n",kbs_alloced);fflush(stdout);
		}
		// H42
		if (dt_alg==4)
			if (tstep>=(H42_per_row)-2)
			{
				//printf("ocl H42 resized ");fflush(stdout);
				char *newa2 = new char[2* (sN + 3)*(_M+3)*H42_per_row*sizeof(double)];
				queue->EnqueueBuffer(b_H42, newa2, 0,(sN + 3)*(_M+3)*H42_per_row*sizeof(double));
				for (int i = (sN + 3)*(_M+3)-1;i >= 1;i--)
					memcpy(newa2 + (i * 2 * H42_per_row*sizeof(double)), newa2 + (i * H42_per_row*sizeof(double)), H42_per_row*sizeof(double));
				delete b_H42;
				H42_per_row *= 2;
				b_H42 = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*H42_per_row*sizeof(double), newa2);
				queue->Finish();
				delete[] newa2;
				//printf(" %d\n",H42_per_row);fflush(stdout);
			}
		// approximations
		if (dt_alg==4)
		{
			// check
			//printf("ocl napprox check ");fflush(stdout);
			int max_used_approx_per_row = 0;
			queue->EnqueueBuffer(b_nibk1, an_ibk,0,(N+3)*(_M+3)*sizeof(cl_int));
			queue->EnqueueBuffer(b_nias1, an_ias,0,(N+3)*(_M+3)*sizeof(cl_int));
			for (int i = 0;i < (sN + 3)*(_M+3);i++)
			{
				if (fabs(an_ibk[i]) > max_used_approx_per_row) max_used_approx_per_row = an_ibk[i];
				if (fabs(an_ias[i]) > max_used_approx_per_row) max_used_approx_per_row = an_ias[i];
			}
			queue->EnqueueBuffer(b_nibk2, an_ibk,0,(N+3)*(_M+3)*sizeof(cl_int));
			queue->EnqueueBuffer(b_nias2, an_ias,0,(N+3)*(_M+3)*sizeof(cl_int));
			for (int i = 0;i < (sN + 3)*(_M+3);i++)
			{
				if (fabs(an_ibk[i]) > max_used_approx_per_row) max_used_approx_per_row = an_ibk[i];
				if (fabs(an_ias[i]) > max_used_approx_per_row) max_used_approx_per_row = an_ias[i];
			}
			//printf(" - %d\n",max_used_approx_per_row);fflush(stdout);
			// resize
			if (max_used_approx_per_row * 2 > alloced_approx_per_row)
			{
				//printf("ocl approximations resizing ");fflush(stdout);
				char *newa1 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation)];
				char *newa2 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation)];
				char *newa3 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation)];
				char *newa4 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation)];
				char *newa12 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation)];
				char *newa22 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation)];
				char *newa32 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation)];
				char *newa42 = new char[2 * (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation)];
				// get old from gpu
				queue->EnqueueBuffer(b_ibk1, newa1, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation));
				queue->EnqueueBuffer(b_ias1, newa2, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation));
				queue->EnqueueBuffer(b_ibk2, newa3, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation));
				queue->EnqueueBuffer(b_ias2, newa4, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation));
				queue->EnqueueBuffer(b_ibk12, newa12, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation));
				queue->EnqueueBuffer(b_ias12, newa22, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation));
				queue->EnqueueBuffer(b_ibk22, newa32, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation));
				queue->EnqueueBuffer(b_ias22, newa42, 0, (N + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation));
				// shift data
				for (int i = (sN + 3)*(_M+3)-1;i >= 1;i--)
				{
					memcpy(newa1 + (i * 2 * alloced_approx_per_row*sizeof(ibk_approximation)), newa1 + (i * alloced_approx_per_row*sizeof(ibk_approximation)), alloced_approx_per_row*sizeof(ibk_approximation));
					memcpy(newa2 + (i * 2 * alloced_approx_per_row*sizeof(ia_approximation)), newa2 + (i * alloced_approx_per_row*sizeof(ia_approximation)), alloced_approx_per_row*sizeof(ia_approximation));
					memcpy(newa3 + (i * 2 * alloced_approx_per_row*sizeof(ibk_approximation)), newa3 + (i * alloced_approx_per_row*sizeof(ibk_approximation)), alloced_approx_per_row*sizeof(ibk_approximation));
					memcpy(newa4 + (i * 2 * alloced_approx_per_row*sizeof(ia_approximation)), newa4 + (i * alloced_approx_per_row*sizeof(ia_approximation)), alloced_approx_per_row*sizeof(ia_approximation));
					memcpy(newa12 + (i * 2 * alloced_approx_per_row*sizeof(ibk_approximation)), newa12 + (i * alloced_approx_per_row*sizeof(ibk_approximation)), alloced_approx_per_row*sizeof(ibk_approximation));
					memcpy(newa22 + (i * 2 * alloced_approx_per_row*sizeof(ia_approximation)), newa22 + (i * alloced_approx_per_row*sizeof(ia_approximation)), alloced_approx_per_row*sizeof(ia_approximation));
					memcpy(newa32 + (i * 2 * alloced_approx_per_row*sizeof(ibk_approximation)), newa32 + (i * alloced_approx_per_row*sizeof(ibk_approximation)), alloced_approx_per_row*sizeof(ibk_approximation));
					memcpy(newa42 + (i * 2 * alloced_approx_per_row*sizeof(ia_approximation)), newa42 + (i * alloced_approx_per_row*sizeof(ia_approximation)), alloced_approx_per_row*sizeof(ia_approximation));
				}
				// delete old gpu buffers
				delete b_ibk1;
				delete b_ias1;
				delete b_ibk2;
				delete b_ias2;
				delete b_ibk12;
				delete b_ias12;
				delete b_ibk22;
				delete b_ias22;
				// put new data
				alloced_approx_per_row *= 2;
				b_ibk1 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), newa1);
				b_ias1 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), newa2);
				b_ibk2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), newa3);
				b_ias2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), newa4);
				b_ibk12 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), newa12);
				b_ias12 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), newa22);
				b_ibk22 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ibk_approximation), newa32);
				b_ias22 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sN + 3)*(_M+3)*alloced_approx_per_row*sizeof(ia_approximation), newa42);
				// clean up
				queue->Finish();
				delete[] newa1;
				delete[] newa2;
				delete[] newa3;
				delete[] newa4;
				delete[] newa12;
				delete[] newa22;
				delete[] newa32;
				delete[] newa42;
				//printf(" %d\n",alloced_approx_per_row);fflush(stdout);				
			}
		}
	}
	void fr_ocl_set_args(OpenCL_kernel *k,double ps,int tstep,int precalc_size)
	{
		int err=0;
		int ar=0;
		int iv;
		if ((k==kAl)||(k==kBt))
			err |= k->SetBufferArg(bAl, ar++);
		if (k==kBt)
		{
			err |= k->SetBufferArg(bBt, ar++);
			err |= k->SetBufferArg(bOm, ar++);
		}
		if (k==kOm)
		{
			err |= k->SetBufferArg(bOm, ar++);
			err |= k->SetBufferArg(bU, ar++);
			err |= k->SetBufferArg(boldU, ar++);
			err |= k->SetBufferArg(bN, ar++);
			err |= k->SetBufferArg(bprev_N, ar++);
			err |= k->SetBufferArg(bHtmp1, ar++);
			err |= k->SetBufferArg(bHtmp2, ar++);
			err |= k->SetBufferArg(bHtmp12, ar++);
			err |= k->SetBufferArg(bHtmp22, ar++);
			err |= k->SetBufferArg(bv3precalc, ar++);
			err |= k->SetBufferArg(bv3precalc2, ar++);
		}
		if (k==kU)
		{
			err |= k->SetBufferArg(bAl, ar++);
			err |= k->SetBufferArg(bBt, ar++);
			err |= k->SetBufferArg(bU, ar++);
		}
		if (k==kN1)
		{
			err |= k->SetBufferArg(bU, ar++);
			err |= k->SetBufferArg(bN, ar++);
			err |= k->SetBufferArg(bprev_N, ar++);
		}
		if ((k!=kcalc_kb)&&(k!=ksave1)&&(k!=ksave2)&&(k!=kcleanup))
		{
			err |= k->SetBufferArg(bS, ar++);
			err |= k->SetArg(ar++, sizeof(int), &sN);
			err |= k->SetArg(ar++, sizeof(int), &_M);
			err |= k->SetArg(ar++, sizeof(int), &m3d_c);
			if ((k!=kU)&&(k!=kN1))
			{
				err |= k->SetBufferArg(bV2, ar++);
				err |= k->SetBufferArg(bkbs, ar++);			
				err |= k->SetBufferArg(bkbs2, ar++);			
			}
			err |= k->SetArg(ar++, sizeof(int), &tstep);
		}
		if (k==kN1)
			err |= k->SetArg(ar++, sizeof(double), &ps);
		if (k==kOm)
		{
			err |= k->SetArg(ar++, sizeof(int), &old_size);
			err |= k->SetBufferArg(b_ab2, ar++);
			err |= k->SetBufferArg(b_ac2, ar++);
			err |= k->SetArg(ar++, sizeof(int), &H42_size);
			err |= k->SetArg(ar++, sizeof(int), &H42_per_row);
			err |= k->SetBufferArg(b_H42, ar++);
			err |= k->SetBufferArg(b_nibk1, ar++);
			err |= k->SetBufferArg(b_nias1, ar++);
			err |= k->SetBufferArg(b_ibk1, ar++);
			err |= k->SetBufferArg(b_ias1, ar++);
			err |= k->SetBufferArg(b_nibk2, ar++);
			err |= k->SetBufferArg(b_nias2, ar++);
			err |= k->SetBufferArg(b_ibk2, ar++);
			err |= k->SetBufferArg(b_ias2, ar++);
			err |= k->SetBufferArg(b_nibk12, ar++);
			err |= k->SetBufferArg(b_nias12, ar++);
			err |= k->SetBufferArg(b_ibk12, ar++);
			err |= k->SetBufferArg(b_ias12, ar++);
			err |= k->SetBufferArg(b_nibk22, ar++);
			err |= k->SetBufferArg(b_nias22, ar++);
			err |= k->SetBufferArg(b_ibk22, ar++);
			err |= k->SetBufferArg(b_ias22, ar++);
			err |= k->SetArg(ar++, sizeof(int), &alloced_approx_per_row);
		}
		if (k==kcalc_kb)
		{
			err |= k->SetBufferArg(bkbs, ar++);			
			err |= k->SetBufferArg(bkbs2, ar++);			
			err |= k->SetArg(ar++, sizeof(int), &tstep);
			err |= k->SetBufferArg(bv3precalc, ar++);
			err |= k->SetBufferArg(bv3precalc2, ar++);
			err |= k->SetArg(ar++, sizeof(int), &precalc_size);
		}
		if ((k==ksave1)||(k==ksave2))
		{
			err |= k->SetBufferArg(bU, ar++);
			err |= k->SetBufferArg(boldU, ar++);
			err |= k->SetArg(ar++, sizeof(int), &old_size);
			if (k==ksave2)
			{
				err |= k->SetBufferArg(b_H42, ar++);
				err |= k->SetArg(ar++, sizeof(int), &H42_size);
				err |= k->SetArg(ar++, sizeof(int), &H42_per_row);
			}
			err |= k->SetArg(ar++, sizeof(int), &sN);
			err |= k->SetArg(ar++, sizeof(int), &_M);
		}
		if (k==kcleanup)
		{
			err |= k->SetBufferArg(b_ab2, ar++);
			err |= k->SetBufferArg(b_ac2, ar++);
			err |= k->SetBufferArg(b_H42, ar++);
			err |= k->SetArg(ar++, sizeof(int), &H42_size);
			err |= k->SetArg(ar++, sizeof(int), &H42_per_row);
			err |= k->SetArg(ar++, sizeof(int), &sN);
			err |= k->SetArg(ar++, sizeof(int), &_M);
		}
		if (err) SERROR("Error: Failed to set kernels args");
	}
	void fr_ocl_call(OpenCL_kernel *k,double ps=0.0,int precalc_size=0.0)
	{
		size_t nth, lsize=1;
		if (m3d_c==1)
			nth=sN;
		if (m3d_c==0)
			nth=_M;
		if ((k==kN1)||(k==ksave1)||(k==ksave2)||(k==kcleanup))
			nth++;
		if (k==kU)
			nth--;
		if (k==kcalc_kb)
			nth=tstep+1;
		if (k==kOm)
		{
			nth=(sN+2)*(_M+2);
			lsize=lBS;
			if (dt_alg==3)
			    lsize=(int)dt_eps;
			if ((nth%lsize)!=0)
			    nth=((nth/lsize)+1)*lsize;
		}
		//int i1=GetTickCount();
		//printf("ocl call %p %d %d.",k,nth,lsize);fflush(stdout);
		fr_ocl_set_args(k,ps,tstep,precalc_size);
		//printf(".");fflush(stdout);
		queue->ExecuteKernel(k, 1, &nth, &lsize);
		//printf(".");fflush(stdout);
		queue->Finish();
		//printf(". %d\n",GetTickCount()-i1);fflush(stdout);
	}
#endif
};
//////////////////////////////////////////////////////////
///////// simple solve ///////////////////////////////////
//////////////////////////////////////////////////////////
int conv_check=0;
void solve(double t,double save_tau,double out_tau,double tau_m,int test)
{
	int nsave = 1,nout=1;
	FILE *fi1,*fi2;
	fi1 = fopen("log1.txt", "at");
	fi2 = fopen("log2.txt", "at");
	if (conv_check)
	do
	{
		double err;
		double div=1;
		TG_solver_<1,1,1> s0(tau_m,1.0,0,0,1);
		TG_solver_<1,1,1> s1(tau_m/2,1.0,0,0,1);
		for (int i = 0;i < div;i++)
			s0.calc_step();
		for (int i = 0;i < 2*div;i++)
			s1.calc_step();
		err=0.0;
		for (int i = 0;i < sN + 1;i++)
			for (int j = 0;j < _M + 1;j++)
				err+=(s0.b_C[idx(i,j)]-s1.b_C[idx(i,j)])*(s0.b_C[idx(i,j)]-s1.b_C[idx(i,j)]);
		err/=(sN*_M);
		if (err > int_err*10)
		{
			tau_m /= 2.0;
			div*=2;
		}
		else
			break;
		printf("tau %g err %g\n", tau_m, err);
	}
	while (true);
	std::vector<TG_solver *> s;
	std::vector<int> dta;
	std::vector<int> ina;
	std::vector<int> uocl;
	std::vector<double> dte;
	int def_dta=dt_alg;
	int def_ina=int_alg;
	int def_ocl=use_ocl;
	double def_dteps=dt_eps;
	double sum_diff2=0, sum_s02=0;
	if (test==0) // diff g(x) for changeable alpha / fixed k and D no massN, single derivative
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 1.0, 0, 0, 1.0));
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.9, 0, 0, 1.0));
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 1.0));
	    s.push_back(new TG_solver_<1,1,1>(tau_m, 0.9, 0, 0, 1.0));
	    s.push_back(new TG_solver_<1,1,1>(tau_m, 0.8, 0, 0, 1.0));
	    s.push_back(new TG_solver_<2,1,1>(tau_m, 0.9, 0, 0, 1.0));
	    s.push_back(new TG_solver_<2,1,1>(tau_m, 0.8, 0, 0, 1.0));

	    s.push_back(new TG_solver_<1,1,1>(tau_m, 0.8, 0, 0, 1.0));
    	    s.push_back(new TG_solver_<2,1,1>(tau_m, 0.8, 0, 0, 1.0));
	    s.push_back(new TG_solver_<0,0,1>(tau_m, 0.9, 0, 0, 1.0));
	    s.push_back(new TG_solver_<0,0,1>(tau_m, 0.8, 0, 0, 1.0));
	
	    s.push_back(new TG_solver_<1,0,1>(tau_m, 1.0, 0, 0, 1.0));
	    s.push_back(new TG_solver_<1,0,1>(tau_m, 0.9, 0, 0, 1.0));
	    s.push_back(new TG_solver_<2,0,1>(tau_m, 0.9, 0, 0, 1.0));

	    s.push_back(new TG_solver_<3,0,1>(tau_m, 0.9, 0, 0, 1.0));
	    s.push_back(new TG_solver_<1,0,1>(tau_m, 0.8, 0, 0, 1.0));
	    s.push_back(new TG_solver_<2,0,1>(tau_m, 0.8, 0, 0, 1.0));
	    s.push_back(new TG_solver_<3,0,1>(tau_m, 0.8, 0, 0, 1.0));
	}
	if (test==1) // g(x)=x, no massN, diff beta2/alpha2
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,__Lx,0,1e300,1e300,0,0.6,1.0));
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,__Lx,0,1e300,1e300,0,0.0,0.0));
	}
	// g(x)=x, no massn,two derivatives / opencl tests
	double beta2=0.2;
	if (test==2) // diff int_algs
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(1);
	    dte.push_back(1e-8);
	    uocl.push_back(0);
    	s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(1);
	    dte.push_back(1e-8);
	    uocl.push_back(1);

	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(6);
	    dte.push_back(1e-8);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(6);
	    dte.push_back(1e-8);
	    uocl.push_back(1);
	}
	if (test==3) // restricted sums
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(1);
	    dte.push_back(1e-8);
	    uocl.push_back(0);
	
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.1);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.1);
	    uocl.push_back(1);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.01);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.01);
	    uocl.push_back(1);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.005);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.005);
	    uocl.push_back(1);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.001);
    	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(2);
	    ina.push_back(1);
	    dte.push_back(0.001);
	    uocl.push_back(1);
	}
	if (test==4) // one series expantion
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(1);
	    dte.push_back(1e-8);
	    uocl.push_back(0);
	
    	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(25);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(25);
	    uocl.push_back(1);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(50);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(50);
	    uocl.push_back(1);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(100);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(100);
	    uocl.push_back(1);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(125);
	    uocl.push_back(0);
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(3);
	    ina.push_back(6);
	    dte.push_back(125);
	    uocl.push_back(1);
	}
	if (test==5) // two series expansion
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(1);
	    dte.push_back(1e-8);
	    uocl.push_back(0);

	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(4);
	    ina.push_back(1);
	    dte.push_back(0.1);
	    uocl.push_back(0);	
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(4);
	    ina.push_back(1);
	    dte.push_back(0.1);
	    uocl.push_back(1);	
	}
	if (test==6)
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(1);
	    dte.push_back(1e-8);
	    uocl.push_back(0);

    	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(4);
	    ina.push_back(1);
	    dte.push_back(0.01);
	    uocl.push_back(0);	
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(4);
	    ina.push_back(1);
	    dte.push_back(0.01);
	    uocl.push_back(1);	
	}
	if (test==7)
	{
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(0);
	    ina.push_back(1);
	    dte.push_back(1e-8);
	    uocl.push_back(0);

    	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
	    dta.push_back(4);
	    ina.push_back(1);
	    dte.push_back(0.001);
	    uocl.push_back(0);	
	    s.push_back(new TG_solver_<0,1,1>(tau_m, 0.8, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,-1,0,1e300,1e300,0,beta2,0.6));
    	    dta.push_back(4);
	    ina.push_back(1);
	    dte.push_back(0.001);
	    uocl.push_back(1);	
	}
	if (test==8) // local approximation comparing to full summation
	{
	    dta.push_back(1);
		dt_alg=1;
		s.push_back(new TG_solver_<0,1,1>(tau_m, massN_a, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,__Lx,0,1e300,1e300,0,0.1,massN_b));
	    dta.push_back(5);
		dt_alg=5;
		s.push_back(new TG_solver_<0,1,1>(tau_m, massN_a, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,__Lx,0,1e300,1e300,0,0.1,massN_b));
		/*double b=0.8;
		double vs[8]={0.999999,0.99,0.98,0.96,0.92,0.84,0.68,0.36};
		for (int i=0;i<8;i++)
			for (int j=0;j<8;j++)
				s.push_back(new TG_solver_<0,1,1>(tau_m, vs[i], 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,__Lx,0,1e300,1e300,0,b,vs[j]));
		double bs[11]={0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
		for (int i=0;i<11;i++)
			s.push_back(new TG_solver_<0,1,1>(tau_m, 0.92, 0, 0, 0.1,-1,5,1.0,-1,-1,NULL,0,__Lx,0,1e300,1e300,0,bs[i],0.92));*/
	}
	printf("int_alg %d dt_alg %d ocl %d int_err %g int_max_niter %d approx_eps %g eps %g\n",int_alg,dt_alg,use_ocl,int_err,integr_max_niter,approx_eps,eps);
	if (s.size())
	for (double tt = 0;tt < t;tt += tau_m)
	{
		if (tt==0)
		    for (int i = 0;i < s.size();i++)
		    {
				printf("%d - s %g a %g+%g*%g l %g k %g kdk %g f 1(fp %g) kf 1 k_der %d fd 1 L %g H %g Q %g Lx %g C1 %g S %g ",i,s[i]->sigma,s[i]->alpha,s[i]->beta2,s[i]->alpha2,s[i]->lambda,s[i]->k,s[i]->k_der_k,s[i]->func_power,s[i]->k_derivative,s[i]->L,s[i]->H,s[i]->Q,s[i]->Lx,_C1,_S);
				printf("N %d a* %g b* %g ",massN, massN_a,massN_b);
				if (dta.size()>i)
					printf("dt_alg %d ",dta[i]);
				else
					printf("dt_alg %d ",def_dta);
				if (ina.size()>i)
					printf("int_alg %d ",ina[i]);
				else
					printf("int_alg %d ",def_ina);
				if (dte.size()>i)
					printf("dt_eps %g ",dte[i]);
				else
					printf("dt_eps %g ",def_dteps);
				if (uocl.size()>i)
					printf("ocl %d ",uocl[i]);
				else
					printf("ocl %d ",def_ocl);
				printf("\n");
		    }
		if (tt >= nout*out_tau)
		{
			printf("T %g ",tt);
			fprintf(fi2,"T %g ",tt);
		}
		for (int i = 0;i < s.size();i++)
		{
			int t1=GetTickCount();
			// set integration algorithm and summation algorithm
			if (dta.size()>i)
				dt_alg=dta[i];
			else
				dt_alg=def_dta;
			if (ina.size()>i)
				int_alg=ina[i];
			else
				int_alg=def_ina;
			if (dte.size()>i)
				dt_eps=dte[i];
			else
				dt_eps=def_dteps;
			if (dt_alg==4)
			    approx_eps=dt_eps;
			if (uocl.size()>i)
				use_ocl=uocl[i];
			else
				use_ocl=def_ocl;			
			if (tt==0)
				s[i]->update();
			// solve
			s[i]->calc_step();
			if (tt >= nout*out_tau)
			{
#ifdef OCL
				if (use_ocl)
				{
					s[i]->fr_ocl_get(s[i]->b_C);
					s[i]->fr_ocl_get(s[i]->b_N);
					s[i]->fr_ocl_get(s[i]->b_prevN);
				}
#endif
				//printf("total time %d ints %d Om %d ints hr %d niters %d err0 %g err1 %g ", GetTickCount() - t1, s[i]->perf_count[0], s[i]->perf_count[1], s[i]->perf_count[2], s[i]->perf_count[3], s[i]->perf_count_d[0], s[i]->perf_count_d[1]);
				printf("t %d ", GetTickCount() - t1);
				printf(" V %g ",s[i]->b_C[idx(sN / 4, _M / 2)]);fflush(stdout);
				//fprintf(fi2, "T %g V %g total time %d ints %d Om %d ints hr %d niters %d err0 %g err1 %g ", tt, s[i]->b_C[idx(sN / 4, _M / 2)], GetTickCount() - t1, s[i]->perf_count[0], s[i]->perf_count[1], s[i]->perf_count[2], s[i]->perf_count[3], s[i]->perf_count_d[0], s[i]->perf_count_d[1]);
				fprintf(fi2, "V %g t %d ", s[i]->b_C[idx(sN / 4, _M / 2)], GetTickCount() - t1);
				if (testing)
				{
					double err=s[i]->testing_check();
					printf("e %g ",err);
					fprintf(fi2,"e %g ",err);				
				}
			}
		}
		if (tt >= nout*out_tau)
		{
			// compare with first
			printf(" diffs(avgsq,maxabs,global rel for[0,T]) ");
			fprintf(fi2," diffs(avgsq,maxabs,global rel for[0,T]) ");
			for (int i = 1;i < s.size();i++)
			{
				double err=0.0;
				double max=0.0;
				for (int _i=0;_i<=sN;_i++)
					for (int j=0;j<=_M;j++)
					{
						double diff=s[0]->b_C[idx(_i,j)]-s[i]->b_C[idx(_i,j)];
						err+=diff*diff;
						sum_diff2+=diff*diff;
						sum_s02+=s[0]->b_C[idx(_i,j)]*s[0]->b_C[idx(_i,j)];
						if (fabs(diff)>max) max=fabs(diff);
					}
				double rel1=sqrt(sum_diff2)/sqrt(sum_s02);
				err/=sN*_M;
				printf("%g %g %g ",err,max,rel1);
				fprintf(fi2,"%g %g %g ",err,max,rel1);
			}
			printf("\n");
			fprintf(fi2,"\n");
			nout++;
		}
		// save result
		if (tt > nsave*save_tau)
		{
			if (testing==0)
			for (int i = 0;i < sN + 1;i++)
			{
				fprintf(fi1, "%g %g ",tt, (double)i*s[0]->sdL);
				for (int j = 0;j < s.size();j++)
				{
					fprintf(fi1, "%g %g %g ", s[j]->b_C[idx(i, _M *1/ 4)],s[j]->b_C[idx(i, _M *2/ 4)],s[j]->b_C[idx(i, _M *3/ 4)]);
					if (massN)
						fprintf(fi1, "%g %g %g ", s[j]->b_N[idx(i, _M *1/ 4)],s[j]->b_N[idx(i, _M *2/ 4)],s[j]->b_N[idx(i, _M *3/ 4)]);
				}
				fprintf(fi1, "\n");
			}
			else
			for (int i = 0;i < sN + 1;i++)
			for (int k = 0;k < _M+1;k++)
			{
				fprintf(fi1, "%g %g %g ",tt, (double)i*s[0]->sdL,(double)k*s[0]->dLx);
				for (int j = 0;j < s.size();j++)
					fprintf(fi1, "%g %g %g %g ", s[j]->b_C[idx(i, k)],s[j]->testing_solutionC(i,k,s[j]->tstep),s[j]->b_N[idx(i, k)],s[j]->testing_solutionN(i,k,s[j]->tstep));
				fprintf(fi1, "\n");
			}
			fflush(fi1);
			nsave++;
		}
		printf(" T %g V[0.25,0.5] ", tt);
		for (int j = 0;j < s.size();j++)
			printf("%g ", s[j]->b_C[idx(sN/4, _M / 2)]);
		printf("\n");
	}
}

////////////////////////////////////////////////////////////////
////////////////////////// PSO optimization /////////////////////
////////////////////////////////////////////////////////////////

double bounds[10][2]={{0.8,1.0}, // alpha
		     {1e-8,2.0}, // lamdba/D for fixed D
		     {1.0,5.0}, // k
		     {0.5,2.0}, // k-der k
		     {0.1,1.9}, // g-der power1
		     {0.1,1.9}, // g-der power2
		     {1,10}, // L
		     {0.1,3}, // H
		     {0,100}, // C1
		     {-100,100} // S
		     };

int init_print=1;
char *filename=NULL;
int fixed_initial_values=0;
double *fixed_init=NULL;
double *testing_T=NULL,*testing_Z=NULL,*testing_F=NULL;
int ntesting=0;
int nfixed_init=0;
int fit_alg=0; // 0 -pso,1 - multiple steepest descent, 2 - pso + steepest descent
int st_desc_maxiter=100;
int max_repeat=1000;
double _Lx=3.0;
double st_desc_eps=1e-9;
double st_desc_diff=1e-5;
double st_desc_div=10000.0;
double st_desc_init_step=1.0;
int goal_func=0; // 0 -sum sq, 1 - sum rel
TG_solver *init_solver(double *p,int to_fit, int f,double fp,int kf,int k_der,int fd,double tau_m)
{
	double a=bounds[0][1];
	double lambda=bounds[1][1];
	double k=bounds[2][1];
	double kdk=bounds[3][1];
	double dv=0.1;
	double _L=bounds[6][1],_H=bounds[7][1];
	double _C1,_S,func_power2;
	int offs=1;
	int pv_offs=0;
	func_power2=bounds[5][1];
	_C1=bounds[8][1];
	_S=bounds[9][1];
	if (to_fit&1) a=p[offs++];
	if (to_fit&2) lambda=p[offs++]; 
	if (to_fit&4) k=p[offs++]; 
	if (to_fit&8) kdk=p[offs++]; 
	if (to_fit&16) fp=p[offs++]; 
	if (to_fit&32) _L=p[offs++]; 
	if (to_fit&64) _H=p[offs++]; 
	if (to_fit&128) _C1=p[offs++]; 
	if (to_fit&256) func_power2=p[offs++]; 
	if (to_fit&512) _S=p[offs++]; 
	if (fd==1) dv=lambda;
	if (a>1.0) a=1.0;
	if (a<0.0) a=0.0;
	TG_solver *ret;
	if (kf==0)
	{
	    if (fd==0)
	    {
		if (f==1)
		    ret=new TG_solver_<1,0,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==2)
		    ret=new TG_solver_<2,0,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==3)
		    ret=new TG_solver_<3,0,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==4)
		    ret=new TG_solver_<4,0,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
	    }
	    else
	    {
		if (f==1)
		    ret=new TG_solver_<1,1,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==2)
		    ret=new TG_solver_<2,1,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==3)
		    ret=new TG_solver_<3,1,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==4)
		    ret=new TG_solver_<4,1,0>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
	    }
	}
	else
	{
	    if (fd==0)
	    {
		if (f==1)
		    ret=new TG_solver_<1,0,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==2)
		    ret=new TG_solver_<2,0,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==3)
		    ret=new TG_solver_<3,0,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==4)
		    ret=new TG_solver_<4,0,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
	    }
	    else
	    {
		if (f==1)
		    ret=new TG_solver_<1,1,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==2)
		    ret=new TG_solver_<2,1,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==3)
		    ret=new TG_solver_<3,1,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
		if (f==4)
		    ret=new TG_solver_<4,1,1>(tau_m,a,kdk,k_der,dv,k,lambda,fp,_L,_H,fixed_init,nfixed_init,_Lx,1,_C1,_S,func_power2);
	    }
	}
	if (init_print==1)
	{
	    printf("a %g l %g k %g kdk %g f %d(fp %g) kf %d k_der %d fd %d L %g H %g Q %g Lx %g C1 %g S %g\n",ret->alpha,ret->lambda,ret->k,ret->k_der_k,f,ret->func_power,kf,ret->k_derivative,fd,ret->L,ret->H,ret->Q,ret->Lx,_C1,_S);
	    if (debug_level==0) init_print=0;
	}
	return ret;
}
void clear_solver(TG_solver *ss,int to_fit)
{
	delete ss;
}
double solve_and_test(double *p,int to_fit,int f,double fp,int kf,int k_der,int fd,double tau_m,
	FILE *fi, double *values_T,double *values_Z,double *values_F,int nvalues,double t)
{
	double err=0.0;	
	double *vs=new double [nvalues];
	double *values_Z_trans=new double[2*nvalues];
	int diff;
	TG_solver *ss=init_solver(p,to_fit,f,fp,kf,k_der,fd,tau_m);
	for (int i=0;i<nvalues;i++)
	{
	    if (debug_level>=3) printf("val at %g (%g,%g) ",values_T[i],values_Z[2*i+0],values_Z[2*i+1]);
	    values_Z_trans[2*i+0]=values_Z[2*i+0];
	    values_Z_trans[2*i+1]=values_Z[2*i+1];
	    ss->FPforXY(values_Z_trans+2*i);
	    if (debug_level>=3) printf("-> fi_psi(%g,%g)i_j(%d,%d)X,Y(%g,%g) = %g\n",values_Z_trans[2*i+0],values_Z_trans[2*i+1],(int)(values_Z_trans[2*i+0]/ss->sdL),(int)(0.5+(values_Z_trans[2*i+1]/ss->dLx)),ss->X((int)(values_Z_trans[2*i+0]/ss->sdL),(int)(0.5+(values_Z_trans[2*i+1]/ss->dLx))),ss->Y((int)(values_Z_trans[2*i+0]/ss->sdL),(int)(0.5+(values_Z_trans[2*i+1]/ss->dLx))),values_F[i]);
	    if (debug_level>=3) fprintf(fi,"-> fi_psi(%g,%g)i_j(%d,%d)X,Y(%g,%g) = %g\n",values_Z_trans[2*i+0],values_Z_trans[2*i+1],(int)(values_Z_trans[2*i+0]/ss->sdL),(int)(0.5+(values_Z_trans[2*i+1]/ss->dLx)),ss->X((int)(values_Z_trans[2*i+0]/ss->sdL),(int)(0.5+(values_Z_trans[2*i+1]/ss->dLx))),ss->Y((int)(values_Z_trans[2*i+0]/ss->sdL),(int)(0.5+(values_Z_trans[2*i+1]/ss->dLx))),values_F[i]);
	}
	values_Z=values_Z_trans;
	diff=1;
	for (int i=0;i<nvalues;i++)
	{
	    int found=0;
	    for (int j=0;j<i;j++)
		if (values_Z_trans[2*i+0]==values_Z_trans[2*j+0])
		    if (values_Z_trans[2*i+1]==values_Z_trans[2*j+1])
		    if (values_T[i]==values_T[j])
		    {
			found=1;
			break;
		    }
	    if (found==0)
		diff++;
	    if ((int)(values_Z_trans[2*i+0]/ss->sdL)==0)
		diff=-10000;
	}
	if (diff>(nvalues/2))	    
	{
	ss->initial(fixed_init,nfixed_init);
	if (ss->b_C[0]==1e300)
	{
	    err=1e300;
	    if (debug_level>=2) printf("initial failed\n");
	    goto cl;
	}
	for (double tt = 0;tt <= t;tt += ss->tau)
	{
		// save old values
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
			{
				for (int ii=0;ii<=sN-1;ii++)
				for (int j=0;j<=_M-1;j++)
					if (((ss->sdL*ii)<=values_Z[2*i+0])&&((ss->sdL*(ii+1))>values_Z[2*i+0]))
					if (((ss->dLx*(j-0.5))<=values_Z[2*i+1])&&((ss->dLx*(j+0.5))>values_Z[2*i+1]))
					{
						double k2x=(values_Z[2*i+0]-(ss->sdL*ii))/ss->sdL;
						double k2y=(values_Z[2*i+1]-(ss->dLx*(j-0.5)))/ss->dLx;
						double v1,v2;
						v1=(1-k2x)*ss->b_C[idx(ii,j)]+k2x*ss->b_C[idx(ii+1,j)];
						v2=(1-k2y)*ss->b_C[idx(ii,j)]+k2y*ss->b_C[idx(ii,j+1)];
						vs[i]=(v1+v2)/2.0;
						break;
					}
			}
		// solve
		ss->calc_step();
		// add to err
		double midv=0;
		if (nvalues)
		{
		    for (int i=0;i<nvalues;i++)
			midv+=values_F[i];
    		    midv/=nvalues;
    		}
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
			{
				double k1=(values_T[i]-tt)/ss->tau;
				for (int ii=0;ii<=sN-1;ii++)
				for (int j=0;j<=_M-1;j++)
					if (((ss->sdL*ii)<=values_Z[2*i+0])&&((ss->sdL*(ii+1))>values_Z[2*i+0]))
					if (((ss->dLx*(j-0.5))<=values_Z[2*i+1])&&((ss->dLx*(j+0.5))>values_Z[2*i+1]))
					{
						double k2x=(values_Z[2*i+0]-(ss->sdL*ii))/ss->sdL;
						double k2y=(values_Z[2*i+1]-(ss->dLx*(j-0.5)))/ss->dLx;
						double v1,v2,v;
						v1=(1-k2x)*ss->b_C[idx(ii,j)]+k2x*ss->b_C[idx(ii+1,j)];
						v2=(1-k2y)*ss->b_C[idx(ii,j)]+k2y*ss->b_C[idx(ii,j+1)];
						v=(v1+v2)/2.0;
						v=(1-k1)*vs[i]+k1*v;
						if (goal_func==0)
						    err+=(values_F[i]-v)*(values_F[i]-v);
						if ((goal_func==1)||(goal_func==3))
						{
						    double r=1;
						    if (values_F[i]!=midv)
							r=fabs((values_F[i]-v)/(values_F[i]-midv));
						    else
							if (v==midv)
							    r=0;
						    if (goal_func==1)
							err+=r;
						    else
							if (r>err)
							    err=r;
						}
						if (goal_func==2)
						    if (((values_F[i]-v)*(values_F[i]-v))>err)
							err=(values_F[i]-v)*(values_F[i]-v);
						
						break;
					}
			}
		if (!finite(ss->b_C[idx(sN/2,_M/2)]))
		{
		     err=1e300;
    	    	     if (debug_level>=2) printf("solution failed (%g %g)\n",tt,t);
		     break;		     
		}
	}	
	}
	else
	{
    	    if (debug_level>=2) printf("grid failed (%d %d)\n",diff,nvalues/2);
	    err=1e300;
	}
cl:
	clear_solver(ss,to_fit);
	delete [] values_Z;
	delete [] vs;
	if (!finite(err))
	{
    	     if (debug_level>=2) printf("final solution failed\n");
	     err = 1e300;
	}
	return err;
}
double rnd(int i,int idx,int size)
{
	if (fixed_initial_values==0)
		return ((rand() % 10000) / (10000.0 - 1.0));
	else
	{
		if (idx>(2<<size))	
			return ((rand() % 10000) / (10000.0 - 1.0));
		return idx&(1<<i);
	}
}
void init_particle(double *particle, int to_fit,int idx,int size)
{
	int s = 0;	
	if (to_fit & 1) particle[1+s++] = bounds[0][0] + (bounds[0][1]-bounds[0][0])*rnd(s,idx,size);
	if (to_fit & 2)	particle[1+s++] = bounds[1][0] + (bounds[1][1]-bounds[1][0])*rnd(s,idx,size);
	if (to_fit & 4)	particle[1+s++] = bounds[2][0] + (bounds[2][1]-bounds[2][0])*rnd(s,idx,size);
	if (to_fit & 8)	particle[1+s++] = bounds[3][0] + (bounds[3][1]-bounds[3][0])*rnd(s,idx,size);
	if (to_fit & 16) particle[1+s++] = bounds[4][0] + (bounds[4][1]-bounds[4][0])*rnd(s,idx,size);
	if (to_fit & 32) particle[1+s++] = bounds[6][0] + (bounds[6][1]-bounds[6][0])*rnd(s,idx,size);
	if (to_fit & 64) particle[1+s++] = bounds[7][0] + (bounds[7][1]-bounds[7][0])*rnd(s,idx,size);
	if (to_fit & 128) particle[1+s++] = bounds[8][0] + (bounds[8][1]-bounds[8][0])*rnd(s,idx,size);
	if (to_fit & 256) particle[1+s++] = bounds[5][0] + (bounds[5][1]-bounds[5][0])*rnd(s,idx,size);
	if (to_fit & 512) particle[1+s++] = bounds[9][0] + (bounds[9][1]-bounds[9][0])*rnd(s,idx,size);
	for (int i = 0;i < s;i++)
		particle[i + 1 + 2 * s] = particle[i + 1];
	for (int i = 0;i<s;i++)
		particle[i + 1 + s] = 0.0;
}
void save_images(TG_solver *ss)
{
	double *valsC;
	char str[1024],str2[1024];
	valsC=new double[(sN-2)*(_M-2)];
	for (int j=1;j<sN-1;j++)
		for (int k=1;k<_M-1;k++)
			valsC[(j-1)*(_M-2)+k-1]=ss->b_C[idx(j,k)];
	sprintf(str,"valsC_%d.raw",ss->tstep);
	FILE *fi=fopen(str,"wb");
	fwrite(valsC,(sN-2)*(_M-2),sizeof(double),fi);
	fclose(fi);
	sprintf(str2,"raw2tiff -w %d -l %d -d double %s %s.tif",_M-2,sN-2,str,str);
	system(str2);
	sprintf(str2,"rm %s",str);
	system(str2);

	for (int j=1;j<sN-1;j++)
		for (int k=1;k<_M-1;k++)
			valsC[(j-1)*(_M-2)+k-1]=ss->V2[idx(j,k)];
	sprintf(str,"v2_%d.raw",ss->tstep);
	fi=fopen(str,"wb");
	fwrite(valsC,(sN-2)*(_M-2),sizeof(double),fi);
	fclose(fi);
	sprintf(str2,"raw2tiff -w %d -l %d -d double %s %s.tif",_M-2,sN-2,str,str);
	system(str2);
	sprintf(str2,"rm %s",str);
	system(str2);

	delete [] valsC;
}
double steepest_descent(double *p,int to_fit,int f,double fp,int kf,int k_der,int fd,double tau_m,
	FILE *fi, double *values_T,double *values_Z,double *values_F,int nvalues,double t)
{
	double vs[10],vnew,sp[10];
	double div=st_desc_div;
	double step=st_desc_init_step;
	int iter=0;
	int s=0;
	int bounds_nums[]={0,1,2,3,4,6,7,8,5,9};
	vs[0]=solve_and_test(p,to_fit,f,fp,kf,k_der,fd,tau_m,fi,values_T,values_Z,values_F,nvalues,t);
	if (vs[0]==1e300) return 1e300;
	do
	{
	    s=0;
	    for (int i=0;i<=10;i++)
		if (to_fit & (1<<i))
		{
		    double st=step*(bounds[bounds_nums[i]][1]-bounds[bounds_nums[i]][0])/div;
		    p[1+s]+=st;
		    vs[i]=solve_and_test(p,to_fit,f,fp,kf,k_der,fd,tau_m,fi,values_T,values_Z,values_F,nvalues,t);
		    vs[i]=(vs[i]-vs[0])/st; // dF/dpi
		    p[1+s]-=st;
		    s++;
		}
	    // x+=step*dF/dpi
	    s=0;
	    for (int i=0;i<=10;i++)
		if (to_fit & (1<<i))
		{
		    sp[1+s]=p[1+s];
		    p[1+s]-=step*vs[i];
		    if (p[1+s]<bounds[bounds_nums[i]][0])
			p[1+s]=bounds[bounds_nums[i]][0];
		    if (p[1+s]>bounds[bounds_nums[i]][1])
			p[1+s]=bounds[bounds_nums[i]][1];
		    s++;
		}
	    vnew=solve_and_test(p,to_fit,f,fp,kf,k_der,fd,tau_m,fi,values_T,values_Z,values_F,nvalues,t);
	    s=0;
	    for (int i=0;i<=10;i++)
		if (to_fit & (1<<i))
		    printf ("%g %g ",p[1+(s++)],vs[i]);
	    printf("%d vnew %g v0 %g step %g\n",iter,vnew,vs[0],step);
	    if (vnew>=vs[0])
	    {
		// move back and decrease step
		s=0;
		for (int i=0;i<=9;i++)
		    if (to_fit & (1<<i))
		    {
			p[1+s]=sp[1+s];
			s++;
		    }
		step/=2.0;
		if (step<st_desc_eps)
		    break;
	    }
	    else
	    {
		if (fabs(vnew-vs[0])<st_desc_diff)
		    return vnew;
		vs[0]=vnew;
	    }
	    if ((iter++)>st_desc_maxiter)
		break;
	}
	while (1);		
	return vs[0];
}
double fit_and_solve(double t,double save_tau,double out_tau,
	double *values_T,double *values_Z,double *values_F,int nvalues,
	int to_fit,int f,double fp,int kf,int k_der,int fd,double tau_m,
	int pso_n,double pso_o,double pso_fi_p,double pso_fi_g,double pso_eps,int pso_max_iter,int mindeb=0)
{
	int size=0;
	double **particles;
	int best;
	double *best_p;
	int iter=0;
	char fn[2048]="log.txt",rfn[2048]="res.txt";
	if (filename)
	{
	    sprintf(fn,"log_%s.txt",filename);
	    sprintf(rfn,"res_%s.txt",filename);
	}
	FILE *fi1 = fopen(fn, "at");
	// number of variables to optimize
	if (to_fit&1) size++;
	if (to_fit&2) size++;
	if (to_fit&4) size++;
	if (to_fit&8) size++;
	if (to_fit&16) size++;
	if (to_fit&32) size++;
	if (to_fit&64) size++;
	if (to_fit&128) size++;
	if (to_fit&256) size++;
	if (to_fit&512) size++;
	if (size==0) return -1;
	best_p=new double[size+1];
	if (mindeb==0)
	{
		for (int i=0;i<=9;i++)
		{
			fprintf(fi1,"#bounds[%d]={%g,%g}\n",i,bounds[i][0],bounds[i][1]);
			printf("bounds[%d]={%g,%g}\n",i,bounds[i][0],bounds[i][1]);
		}
		printf("pso params: fit_alg %d n %d o %g fi_p %g fi_g %g eps %g miter %d\n",fit_alg,pso_n,pso_o,pso_fi_p,pso_fi_g,pso_eps,pso_max_iter);
		printf("int_alg %d dt_alg %d ocl %d int_err %g int_max_niter %d approx_eps %g eps %g\n",int_alg,dt_alg,use_ocl,int_err,integr_max_niter,approx_eps,eps);
	}
	particles=new double *[pso_n];
	for (int i=0;i<pso_n;i++)
		particles[i]=new double[3*size+2]; // particles[0] contains f value, particles[1+size].. contain velocities, particles[1+2*size]... contain particle's best known position, particles[1+3*size] contains best known f value
	// initialize
	int nit=0;
	do
	{
	    init_particle(particles[0], to_fit,0,size);
	    if ((fit_alg==0)||(fit_alg==2))
	        particles[0][0]=particles[0][1+3*size]=solve_and_test(particles[0],to_fit,f,fp,kf,k_der,fd,tau_m,fi1,values_T,values_Z,values_F,nvalues,t);
	    if ((fit_alg==1)||(fit_alg==2))
	        particles[0][0]=particles[0][1+3*size]=steepest_descent(particles[0],to_fit,f,fp,kf,k_der,fd,tau_m,fi1,values_T,values_Z,values_F,nvalues,t);    
		if (mindeb==0) { printf(".");fflush(stdout); }
	    if ((nit++)==max_repeat)
		break;
	}
	while (particles[0][0]>=1e10);
	if (mindeb==0)
	{
		fprintf(fi1, "#initial 0 - %g\n", particles[0][0]);
		printf("initial 0 - %g\n", particles[0][0]);
	}
#pragma omp parallel for
	for (int i=1;i<pso_n;i++)
	{
		int nit=0;
		do
		{
		    init_particle(particles[i], to_fit,i,size);
		    if ((fit_alg==0)||(fit_alg==2))
		        particles[i][0]=particles[i][1+3*size]=solve_and_test(particles[i],to_fit,f,fp,kf,k_der,fd,tau_m,fi1,values_T,values_Z,values_F,nvalues,t);
		    if ((fit_alg==1)||(fit_alg==2))
		        particles[i][0]=particles[i][1+3*size]=steepest_descent(particles[i],to_fit,f,fp,kf,k_der,fd,tau_m,fi1,values_T,values_Z,values_F,nvalues,t);    
		    if (mindeb==0) { printf(".");fflush(stdout); }
		    if ((nit++)==max_repeat)
			break;
		}
		while (particles[i][0]>=1e10);
		if (mindeb==0)
		{
			fprintf(fi1, "#initial %d - %g\n",i, particles[i][0]);
			printf("initial %d - %g\n",i, particles[i][0]);
		}
	}
	// save best known position
	best=0;
	for (int i=1;i<pso_n;i++)
	    if (particles[i][0]<particles[best][0])
		best=i;
	for (int j=0;j<=size;j++)
		best_p[j]=particles[best][j];
	if (mindeb==0)
	{
		fprintf(fi1, "#initial best: ");
		printf("initial best: ");
		for (int j = 0;j <= size;j++)
		{
			fprintf(fi1, "%2.2g ", best_p[j]);
			printf("%2.2g ", best_p[j]);
		}
		fprintf(fi1, "\n");
		printf("\n");
		fflush(stdout);
	}
	// process
	if ((fit_alg==0)||(fit_alg==2))
	do
	{
#pragma omp parallel for
		for (int i=0;i<pso_n;i++)
		{
			// update velocity
			for (int j=0;j<size;j++)
			{
				double rp=(rand()%10000)/10000.0;
				double rg=(rand()%10000)/10000.0;
				particles[i][j+1+size]=pso_o*particles[i][j+1+size]+pso_fi_p*rp*(particles[i][j+1+2*size]-particles[i][j+1])+pso_fi_g*rg*(best_p[j+1]-particles[i][j+1]);
			}
			// update position
			for (int j=0;j<size;j++)
			{
				particles[i][1+j]+=particles[i][j+1+size];
				if (particles[i][1+j]<1e-10)
					particles[i][1+j]=1e-10;
			}
			// calc f value
			particles[i][0]=solve_and_test(particles[i],to_fit,f,fp,kf,k_der,fd,tau_m,fi1,values_T,values_Z,values_F,nvalues,t);
			if (fit_alg==2)
			    particles[i][0]=steepest_descent(particles[i],to_fit,f,fp,kf,k_der,fd,tau_m,fi1,values_T,values_Z,values_F,nvalues,t);    
			// update particle best
			if (particles[i][0]<particles[i][1+3*size])
			{
				for (int j=0;j<size;j++)
					particles[i][j+1+2*size]=particles[i][j+1];
				particles[i][1+3*size]=particles[i][0];
			}
			fflush(stdout);
		}
		// update swarm best
		for (int i=0;i<pso_n;i++)
		    if (particles[i][0]<best_p[0])
		    {
			for (int j=0;j<size;j++)
		    	    best_p[j+1]=particles[i][j+1];
			best_p[0]=particles[i][0];
		    }
		// check best-worst difference
		double max = 0.0;
		double avg=0.0;
		for (int i = 0;i < pso_n;i++)
		{
			if (max < particles[i][0])
			max = particles[i][0];
			avg+= particles[i][0];
		}
		avg/=pso_n;
		if (mindeb==0)
		{
			fprintf(fi1, "%d avg %g best: ", iter,avg);
			printf("%d avg %g best: ", iter,avg);
			for (int j = 0;j <= size;j++)
			{
				fprintf(fi1, "%g ", best_p[j]);
				printf("%g ", best_p[j]);
			}
			fprintf(fi1, "\n");
			printf("\n");
		}
		if ((max - best_p[0]) < pso_eps)
			break;
		iter++;
	}
	while ((iter<pso_max_iter)&&(best_p[0]>pso_eps));
	if (mindeb)
	{
		double max = 0.0;
		double avg=0.0;
		for (int i = 0;i < pso_n;i++)
		{
			if (max < particles[i][0])
			max = particles[i][0];
			avg+= particles[i][0];
		}
		avg/=pso_n;
		fprintf(fi1,"avg %g best: ",avg);
		printf("avg %g best: ",avg);
		for (int j = 0;j <= size;j++)
		{
			fprintf(fi1, "%g ", best_p[j]);
			printf("%g ", best_p[j]);
		}
		fprintf(fi1, "\n");
		printf("\n");
	}
	// solve with best parameters values
	TG_solver *ss;
	init_print=1;
	ss=init_solver(best_p,to_fit,f,fp,kf,k_der,fd,tau_m);
	ss->initial(fixed_init,nfixed_init);	
	int nsave = 0;
	int nout=1;
	int d,oldd=-1;
	int logn=0,maxlogn=0;
	double err=0.0,srel_err=0.0;
	int nerr=0;	
	
	nvalues=ntesting;
	values_Z=testing_Z;
	values_T=testing_T;
	values_F=testing_F;	
	
	double *vs=new double [nvalues];
	FILE *fi2;
	fi2 = fopen(rfn, "at");
	if (nvalues)
    	    if (t<values_T[nvalues-1])
		t=values_T[nvalues-1];
	double *values_Z_trans=new double[2*nvalues];
	for (int i=0;i<nvalues;i++)
	{
	    values_Z_trans[2*i+0]=values_Z[2*i+0];
	    values_Z_trans[2*i+1]=values_Z[2*i+1];
	    ss->FPforXY(values_Z_trans+2*i);
	}
	values_Z=values_Z_trans;
	for (double tt = 0;tt <= t;tt += ss->tau)
	{
		// save result
		if (mindeb==0)
		if (tt ==0)
		{
			for (int i = 0;i < sN + 1;i++)
			    for (int j = 0;j < _M + 1;j++)
				fprintf(fi2, "%g %d %d %g %g %g\n", tt,i,j,ss->X(i,j),ss->Y(i,j),ss->b_C[idx(i, j)]);
			fflush(fi2);
			save_images(ss);
			nsave++;
		}
		// save old 
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
				for (int ii=0;ii<=sN-1;ii++)
				for (int j=0;j<=_M-1;j++)
					if (((ss->sdL*ii)<=values_Z[2*i+0])&&((ss->sdL*(ii+1))>values_Z[2*i+0]))
					if (((ss->dLx*(j-0.5))<=values_Z[2*i+1])&&((ss->dLx*(j+0.5))>values_Z[2*i+1]))
					{
						double k2x=(values_Z[2*i+0]-(ss->sdL*ii))/ss->sdL;
						double k2y=(values_Z[2*i+1]-(ss->dLx*(j-0.5)))/ss->dLx;
						double v1,v2;
						v1=(1-k2x)*ss->b_C[idx(ii,j)]+k2x*ss->b_C[idx(ii+1,j)];
						v2=(1-k2y)*ss->b_C[idx(ii,j)]+k2y*ss->b_C[idx(ii,j+1)];
						vs[i]=(v1+v2)/2.0;
						break;
					}
		// solve	
		ss->calc_step();
		// add to err
		int is_out = 0;
		logn=0;
		if (logn>maxlogn) maxlogn=logn;
		for (int ii=0;ii<=sN-1;ii++)
		{
			for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
			{
				double k1=(values_T[i]-tt)/ss->tau;
				for (int j=0;j<=_M-1;j++)
					if (((ss->sdL*ii)<=values_Z[2*i+0])&&((ss->sdL*(ii+1))>values_Z[2*i+0]))
					if (((ss->dLx*(j-0.5))<=values_Z[2*i+1])&&((ss->dLx*(j+0.5))>values_Z[2*i+1]))
					{
						double k2x=(values_Z[2*i+0]-(ss->sdL*ii))/ss->sdL;
						double k2y=(values_Z[2*i+1]-(ss->dLx*(j-0.5)))/ss->dLx;
						double v1,v2,v;
						v1=(1-k2x)*ss->b_C[idx(ii,j)]+k2x*ss->b_C[idx(ii+1,j)];
						v2=(1-k2y)*ss->b_C[idx(ii,j)]+k2y*ss->b_C[idx(ii,j+1)];
						v=(v1+v2)/2.0;
						v=(1-k1)*vs[i]+k1*v;
						err+=(values_F[i]-v)*(values_F[i]-v);
						if ((values_F[i]==0)&&(v!=0.0))
							srel_err+=1.0;
						if (values_F[i]!=0)
						{
							if (fabs((values_F[i]-v)/values_F[i])<1)
								srel_err+=fabs((values_F[i]-v)/values_F[i]);
							else 
								srel_err+=1.0;
						}
						nerr++;
						if (mindeb==0)
						{
							fprintf(fi1, "%g ", values_T[i]);
							fprintf(fi1, "%g %g %g %g\n", values_Z[2*i+0], values_Z[2*i+1], values_F[i], v);
						}
						logn++;
						is_out = 1;
					}		
			}
		}
		if (mindeb==0)
		{
			if (is_out)
				fprintf(fi1,"sum_err %g avg_err %g avg_rel_err %g\n",err,(nerr?err/nerr:0),(nerr?srel_err/nerr:0));
			// save result
			if (tt > nsave*save_tau)
			{
				for (int i = 0;i < sN + 1;i++)
					for (int j = 0;j < _M + 1;j++)
					fprintf(fi2, "%g %d %d %g %g %g\n", tt,i,j,ss->X(i,j),ss->Y(i,j),ss->b_C[idx(i, j)]);
				save_images(ss);
				fflush(fi2);
				nsave++;
			}
		}
		if (!finite(ss->b_C[0]))
		{
		    err=1e300;
		    break;
		}
	}
	delete [] values_Z_trans;
	fprintf(fi1,"#checking err: %g\n",err);
	printf("checking err: %g\n",err);
	clear_solver(ss,to_fit);
	fclose(fi1);
	return best_p[0];
}
void pso_over_fit_and_solve(double t,double save_tau,double out_tau,
	double *values_T,double *values_Z,double *values_F,int nvalues,
	int to_fit,int f,double fp,int kf,int k_der,int fd,double tau_m,
	int pso_n,double pso_o,double pso_fi_p,double pso_fi_g,double pso_eps,int pso_max_iter)
{
	int size=4; // fitting pso_n,o,fi_p,fi_g
	double **particles;
	int best;
	double *best_p;
	int iter=0;
	char fn[2048]="log.txt";
	if (filename)
	    sprintf(fn,"log_%s.txt",filename);
	FILE *fi1 = fopen(fn, "at");
	best_p=new double[size+1];
	particles=new double *[pso_n];
	for (int i=0;i<pso_n;i++)
		particles[i]=new double[3*size+2]; 
	// initialize
	for (int i=0;i<pso_n;i++)
	{
		particles[i][1+0]=(int)(10+50*rnd(0,0,4));
		particles[i][1+1]=rnd(0,0,4);
		particles[i][1+2]=rnd(0,0,4);
		particles[i][1+3]=rnd(0,0,4);
		for (int j = 0;j < size;j++)
			particles[i][j + 1 + 2 * size] = particles[i][j + 1];
		for (int j = 0;j<size;j++)
			particles[i][j + 1 + size] = 0.0;
		particles[i][0]=particles[i][1+3*size]=fit_and_solve(t,save_tau,out_tau,values_T,values_Z,values_F,nvalues,
												to_fit,f,fp,kf,k_der,fd,tau_m,
												particles[i][1],particles[i][2],particles[i][3],particles[i][4],pso_eps,pso_max_iter,1);
		fprintf(fi1, "#initial %d - %g\n",i, particles[i][0]);
		printf("initial %d - %g\n",i, particles[i][0]);
	}
	// save best known position
	best=0;
	for (int i=0;i<pso_n;i++)
	    if (particles[i][0]<particles[best][0])
		best=i;
	for (int j=0;j<=size;j++)
		best_p[j]=particles[best][j];
	fprintf(fi1, "#initial best: ");
	printf("initial best: ");
	for (int j = 0;j <= size;j++)
	{
		fprintf(fi1, "%2.2g ", best_p[j]);
		printf("%2.2g ", best_p[j]);
	}
	fprintf(fi1, "\n");
	printf("\n");
	fflush(stdout);
	// process
	do
	{
		for (int i=0;i<pso_n;i++)
		{
			// update velocity
			for (int j=0;j<size;j++)
			{
				double rp=(rand()%10000)/10000.0;
				double rg=(rand()%10000)/10000.0;
				particles[i][j+1+size]=pso_o*particles[i][j+1+size]+pso_fi_p*rp*(particles[i][j+1+2*size]-particles[i][j+1])+pso_fi_g*rg*(best_p[j+1]-particles[i][j+1]);
			}
			// update position
			for (int j=0;j<size;j++)
			{
				particles[i][1+j]+=particles[i][j+1+size];
				if (particles[i][1+j]<1e-10)
					particles[i][1+j]=1e-10;
			}
			// calc f value
			particles[i][0]=fit_and_solve(t,save_tau,out_tau,values_T,values_Z,values_F,nvalues,
												to_fit,f,fp,kf,k_der,fd,tau_m,
												particles[i][1],particles[i][2],particles[i][3],particles[i][4],pso_eps,pso_max_iter,1);
			// update particle best
			if (particles[i][0]<particles[i][1+3*size])
			{
				for (int j=0;j<size;j++)
					particles[i][j+1+2*size]=particles[i][j+1];
				particles[i][1+3*size]=particles[i][0];
			}
			fflush(stdout);
		}
		// update swarm best
		for (int i=0;i<pso_n;i++)
		    if (particles[i][0]<best_p[0])
		    {
				for (int j=0;j<size;j++)
		    	    best_p[j+1]=particles[i][j+1];
				best_p[0]=particles[i][0];
		    }
		// check best-worst difference
		double max = 0.0;
		double avg=0.0;
		for (int i = 0;i < pso_n;i++)
		{
			if (max < particles[i][0])
			max = particles[i][0];
			avg+= particles[i][0];
		}
		avg/=pso_n;
		fprintf(fi1, "%d avg %g best: ", iter,avg);
		printf("%d avg %g best: ", iter,avg);
		for (int j = 0;j <= size;j++)
		{
			fprintf(fi1, "%g ", best_p[j]);
			printf("%g ", best_p[j]);
		}
		fprintf(fi1, "\n");
		printf("\n");
		if ((max - best_p[0]) < pso_eps)
			break;
		iter++;
	}
	while ((iter<pso_max_iter)&&(best_p[0]>pso_eps));
}
/// main/////////////////
int main(int argc,char **argv)
{
	double Tm = 100; // ending time
	double Sm = 5; // results saving interval
	double Om = 5; // log output interval
	int f=1; // g-derivative func - 0 - t, 1 - sqrt(t), 2 - t^2 , 3 - t^func_power, 4 - exp(t*fp2)*t^fp
	double fp=0.5;
 	int kf=1; // filtration k form - 1 - k(C), 0 - fixed
	int k_der=0; // use of k-derivative
	int fd=0; // use of fixed D value
	double tau_m=0.1;
	int _pso_n=10;
	int pso2=0;
	double inverse=0.0; // input=inverse-input if inverse!=0.0
	double _pso_o=0.4;
	double _pso_fi_p=0.4;
	double _pso_fi_g=0.4;
	double _pso_eps=1e-10;
	int _pso_max_iter=40;
	double *values_T=NULL,*values_Z=NULL,*values_F=NULL;
	int nvalues=0;
	int to_fit=1;
	// fixed parameters
	int_err=1e-6;
	int_alg = 0;
	sN=20;
	_M =30;
	// parameters parsing
	for (int i=1;i<argc;i+=2)
	{
		if (strcmp(argv[i],"fit")==0)
			to_fit=atoi(argv[i+1]);
		if (strcmp(argv[i],"Tm")==0)
			Tm=atof(argv[i+1]);
		if (strcmp(argv[i],"Sm")==0)
			Sm=atof(argv[i+1]);
		if (strcmp(argv[i],"Om")==0)
			Om=atof(argv[i+1]);
		if (strcmp(argv[i],"N")==0)
			sN=atoi(argv[i+1]);
		if (strcmp(argv[i],"Tau")==0)
			tau_m=atof(argv[i+1]);
		if (strcmp(argv[i],"_M")==0)
			_M=atoi(argv[i+1]);
		if (strcmp(argv[i],"gfunc")==0)
		{
			f=atoi(argv[i+1]);
			if (f==0) fp=1;
			if (f==1) fp=0.5;
			if (f==2) fp==2;
		}
		if (strcmp(argv[i],"gfunc_power")==0)
		{
			fp=atof(argv[i+1]);
			f=3;
		}
		if (strcmp(argv[i],"kC")==0)
			kf=atoi(argv[i+1]);
		if (strcmp(argv[i],"kder")==0)
			k_der=atoi(argv[i+1]);
		if (strcmp(argv[i],"fixedD")==0)
			fd=atoi(argv[i+1]);
		if (strcmp(argv[i],"nvalues")==0)
		{
			nvalues=atoi(argv[i+1]);
			values_T=new double[nvalues];
			values_Z=new double[2*nvalues]; 
			values_F=new double[nvalues];
			for (int j=0;j<nvalues;j++)
			{
				values_T[j]=atof(argv[i+2+4*j+0]);
				values_Z[2*j+0]=atof(argv[i+2+4*j+1]);
				values_Z[2*j+1]=atof(argv[i+2+4*j+2]);
				values_F[j]=atof(argv[i+2+4*j+3]);
			}
		}
		if (strcmp(argv[i],"ninit")==0)
		{
			nfixed_init=atoi(argv[i+1]);
			fixed_init=new double[3*nfixed_init];
			for (int j=0;j<nfixed_init;j++)
			{
				fixed_init[3*j+0]=atof(argv[i+2+3*j+0]);
				fixed_init[3*j+1]=atof(argv[i+2+3*j+1]);
				fixed_init[3*j+2]=atof(argv[i+2+3*j+2]);
			}
		}
		if (strcmp(argv[i],"values_file")==0)
		{
			FILE *fi=fopen(argv[i+1],"rt");
			char str[1024];
			if (fi)
			{
			    nvalues=0;
			    while(fgets(str,1024,fi)) nvalues++;
			    values_T=new double[nvalues];
			    values_Z=new double[2*nvalues]; 
			    values_F=new double[nvalues];
			    fseek(fi,0,SEEK_SET);
			    nvalues=0;
			    while(fscanf(fi,"%lg %lg %lg %lg\n",values_T+nvalues,values_Z+2*nvalues+0,values_Z+2*nvalues+1,values_F+nvalues)==4) 
				nvalues++;
			    if (inverse!=0.0)
			    for(int i=0;i<nvalues;i++)
				values_F[i]=inverse-values_F[i];
			    fclose(fi);
			}
		}
		if (strcmp(argv[i],"testing_file")==0)
		{
			FILE *fi=fopen(argv[i+1],"rt");
			char str[1024];
			if (fi)
			{
			    ntesting=0;
			    while(fgets(str,1024,fi)) ntesting++;
			    testing_T=new double[ntesting];
			    testing_Z=new double[2*ntesting]; 
			    testing_F=new double[ntesting];
			    fseek(fi,0,SEEK_SET);
			    ntesting=0;
			    while(fscanf(fi,"%lg %lg %lg %lg\n",testing_T+ntesting,testing_Z+2*ntesting+0,testing_Z+2*ntesting+1,testing_F+ntesting)==4) 
				ntesting++;
			    if (inverse!=0.0)
			    for(int i=0;i<ntesting;i++)
				testing_F[i]=inverse-testing_F[i];
			    fclose(fi);
			}
		}
		if (strcmp(argv[i],"init_file")==0)
		{
			FILE *fi=fopen(argv[i+1],"rt");
			char str[1024];
			if (fi)
			{
			    nfixed_init=0;
			    while(fgets(str,1024,fi)) nfixed_init++;
			    fixed_init=new double[3*nfixed_init];
			    fseek(fi,0,SEEK_SET);
			    nfixed_init=0;
			    while(fscanf(fi,"%lg %lg %lg\n",fixed_init+3*nfixed_init+0,fixed_init+3*nfixed_init+1,fixed_init+3*nfixed_init+2)==3) 
				nfixed_init++;
			    if (inverse!=0.0)
			    for(int i=0;i<nfixed_init;i++)
				fixed_init[3*i+2]=inverse-fixed_init[3*i+2];
			    fclose(fi);
			}
		}
		if (strstr(argv[i],"bounds")!=NULL)
		{
		    char *str=argv[i];
		    while ((str[0]!='_')&&(str[0])) str++;
		    if (str[0]=='_')
		    {
				str++;
				char *str2=str;
				while ((str2[0]!='_')&&(str2[0])) str2++;
				if (str2[0]=='_')
				{
					str2[0]=0;
					str2++;
					int ii=atoi(str);
					int j=atoi(str2);
					double v=atof(argv[i+1]);
					if ((j>=0)&&(j<=1))
					if ((ii>=0)&&(ii<=9))
						bounds[ii][j]=v;
				}
		    }
		}
		if (strstr(argv[i],"pso_n")!=NULL)
		    _pso_n=atoi(argv[i+1]);
		if (strstr(argv[i],"pso_o")!=NULL)
		    _pso_o=atof(argv[i+1]);
		if (strstr(argv[i],"pso_fi_p")!=NULL)
		    _pso_fi_p=atof(argv[i+1]);
		if (strstr(argv[i],"pso_fi_g")!=NULL)
		    _pso_fi_g=atof(argv[i+1]);
		if (strstr(argv[i],"pso_eps")!=NULL)
		    _pso_eps=atof(argv[i+1]);
		if (strstr(argv[i],"pso_max_iter")!=NULL)
		    _pso_max_iter=atoi(argv[i+1]);
		if (strstr(argv[i],"init_max_iter")!=NULL)
		    init_max_iter=atoi(argv[i+1]);
		if (strstr(argv[i],"init_eps")!=NULL)
		    init_eps=atof(argv[i+1]);
		if (strstr(argv[i],"init_tau_m")!=NULL)
		    init_tau_m=atof(argv[i+1]);
		if (strstr(argv[i],"filename")!=NULL)
		    filename=argv[i+1];
		if (strstr(argv[i],"Lx")!=NULL)
		    __Lx=_Lx=atof(argv[i+1]);
		if (strstr(argv[i],"_L")!=NULL)
		    __L=atof(argv[i+1]);
		if (strstr(argv[i],"_H")!=NULL)
		    __H=atof(argv[i+1]);
		if (strstr(argv[i],"conv_check")!=NULL)
		    conv_check=atoi(argv[i+1]);
		if (strstr(argv[i],"fixed_initial_values")!=NULL)
		    fixed_initial_values=atoi(argv[i+1]);
		if (strstr(argv[i],"fit_alg")!=NULL)
		    fit_alg=atoi(argv[i+1]);
		if (strstr(argv[i],"mode1d")!=NULL)
		    mode1d=atoi(argv[i+1]);
		if (strstr(argv[i],"st_desc_maxiter")!=NULL)
		    st_desc_maxiter=atoi(argv[i+1]);
		if (strstr(argv[i],"st_desc_eps")!=NULL)
		    st_desc_eps=atof(argv[i+1]);
		if (strstr(argv[i],"st_desc_diff")!=NULL)
		    st_desc_diff=atof(argv[i+1]);
		if (strstr(argv[i],"st_desc_div")!=NULL)
		    st_desc_div=atof(argv[i+1]);
		if (strstr(argv[i],"st_desc_init_step")!=NULL)
		    st_desc_init_step=atof(argv[i+1]);
		if (strstr(argv[i],"debug_level")!=NULL)
		    debug_level=atoi(argv[i+1]);
		if (strstr(argv[i],"_massN")!=NULL)
		    massN=atoi(argv[i+1]);
		if (strstr(argv[i],"massN_a")!=NULL)
		    massN_a=atof(argv[i+1]);
		if (strstr(argv[i],"massN_b")!=NULL)
		    massN_b=atof(argv[i+1]);
		if (strstr(argv[i],"sigma")!=NULL)
		    _sigma=atof(argv[i+1]);
		if (strstr(argv[i],"int_alg")!=NULL)
		    int_alg=atoi(argv[i+1]);
		if (strstr(argv[i],"dt_alg")!=NULL)
		    dt_alg=atoi(argv[i+1]);
		if (strstr(argv[i],"dt_eps")!=NULL)
		    dt_eps=atof(argv[i+1]);
		if (strstr(argv[i],"s1_app_eps")!=NULL)
		    approx_eps=atof(argv[i+1]);
		if (strstr(argv[i],"s1_eps")!=NULL)
		    eps=atof(argv[i+1]);
		if (strstr(argv[i],"testing")!=NULL)
		{
		    testing=atoi(argv[i+1]);
			if (testing)
				massN=1;
		}
		if (strstr(argv[i],"ocl")!=NULL)
		    use_ocl=atoi(argv[i+1]);
		if (strstr(argv[i],"device")!=NULL)
		    device=atoi(argv[i+1]);
		if (strstr(argv[i],"double_ext")!=NULL)
		    double_ext=atoi(argv[i+1]);
		if (strstr(argv[i],"integr_max_niter")!=NULL)
		    integr_max_niter=atoi(argv[i+1]);
		if (strstr(argv[i],"approx_eps")!=NULL)
		    approx_eps=atof(argv[i+1]);
		if (strstr(argv[i],"int_err")!=NULL)
		    int_err=atof(argv[i+1]);
		if (strstr(argv[i],"max_repeat")!=NULL)
		    max_repeat=atoi(argv[i+1]);
		if (strstr(argv[i],"goal_func")!=NULL)
		    goal_func=atoi(argv[i+1]);
		if (strstr(argv[i],"inverse")!=NULL)
		    inverse=atof(argv[i+1]);
		if (strstr(argv[i],"pso2")!=NULL)
		    pso2=1;
		if (strstr(argv[i],"lBS")!=NULL)
		    lBS=atoi(argv[i+1]);
#ifndef WIN32
		if (strstr(argv[i],"newrand")!=NULL)
		    srand(time(NULL));
#endif
	}
	printf("(%d,%d) tend %g tsave %g tout %g\n",sN,_M,Tm,Sm,Om);
	fflush(stdout);		
	if (to_fit<=0)
		solve(Tm,Sm,Om,tau_m,-to_fit);
	else
	{
		if (pso2==0)
			fit_and_solve(Tm,Sm,Om,values_T,values_Z,values_F,nvalues,to_fit,f,fp,kf,k_der,fd,tau_m,_pso_n,_pso_o,_pso_fi_p,_pso_fi_g,_pso_eps,_pso_max_iter);
		else
		pso_over_fit_and_solve(Tm,Sm,Om,values_T,values_Z,values_F,nvalues,to_fit,f,fp,kf,k_der,fd,tau_m,_pso_n,_pso_o,_pso_fi_p,_pso_fi_g,_pso_eps,_pso_max_iter);
	}
	return 0;
}
