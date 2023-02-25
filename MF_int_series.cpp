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
#include <algorithm>
#include <unistd.h>
#include <sys/times.h>
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
unsigned int GetTickCount()
{
   struct tms t;
   long long time=times(&t);
   int clk_tck=sysconf(_SC_CLK_TCK);
   return (unsigned int)(((long long)(time*(1000.0/clk_tck)))%0xFFFFFFFF);    
}
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
		return LogGamma(x+1.0)-log(x);
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
		return (1.0/x)*Gamma(x+1.0);
	//	std::stringstream os;
        //os << "Invalid input argument " << x <<  ". Argument must be positive.";
        //throw std::invalid_argument( os.str() ); 
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
bool AbsGreat(double x, double y) {
    return fabs(x) > fabs(y);
}
bool GreaterPair(std::pair<double,double> x, std::pair<double,double> y) {
    return x.first > y.first;
}
double sum_sort_pyramid(std::vector<double> values)
{
	if (values.size()==0) return 0.0;
	std::sort(values.begin(),values.end(),AbsGreat);
	int old_s=0;
	int old_e=0;
	do
	{
		old_s=old_e;
		old_e=values.size();
		for (int i=old_s;i<(old_e-(old_e-old_s)%2);i+=2)
			values.push_back(values[i]+values[i+1]);
		if ((old_e-old_s)%2)
			values[values.size()-1]+=values[old_e-1];
	}
	while (values.size()-old_e>1);
	return values[values.size()-1];
}
// integrate by linear approximation and recursive subdivision
double integrate(double (*func)(double a,double b,double z,double r),double a,double b,double z,double r0,double r1,double eps,int sd=1,double prev=0.0)
{
	double v1=0.5*(r1-r0)*(func(a,b,z,r0)+func(a,b,z,r1));
	double v=0.0;
	for (int i=1;i<=sd;i++)
		v+=func(a,b,z,r0+(i/((double)(sd+1)))*(r1-r0));
	v*=(r1-r0);
	if (finite(v1)) v+=v1;
	v/=(double)(sd+1);
	if (sd<1000) // recursive subdivition
	{
		if (sd==1)
			prev=v1;
		if (fabs(v-prev)<eps)
			return v;
		else
			return integrate(func,a,b,z,r0,r1,eps,sd*2.0,v);
	}
	return v;
}
#define PI 3.14159265358979323846
double K(double a,double b,double z,double r)
{
	double k1=(1.0/(PI*a))*pow(r,(1-b)/a)*exp(-pow(r,1.0/a));
	double k2=r*sin(PI*(1-b))-z*sin(PI*(1-b+a));
	double k3=r*r-2*r*z*cos(PI*a)+z*z;
	return k1*k2/k3;
}
double mittag_leffler(double a,double b,double z,double eps)
{
	int k0=log(eps*(1-z))/log(z);
	int kmin=1+(int)((1-b)/a);
	if ((z>=1.0)||(z<0.0)) k0=-2.0*log(eps);
	if (k0<kmin) k0=kmin;
	double res=0.0;
	double zz=1.0;
	if (z>0)
	{
	// do simple sum for z>0
	for (int i=0;i<k0;i++)
	{
		double v=zz/Gamma(b+a*i);
		if (finite(v))
			res+=v;
		else
			break;
		zz*=z;
	}
	}
	else
	{
		double v1,v2;
		// integration
		if (((fabs(z)>PI*a)&&(b<(1+a)))&&((a>0)&&(a<1)))
		{
			double r1=1.0;
			v1=2.0*fabs(z);
			if (v1>r1) r1=v1;
			v2=pow(-log(PI*eps/6.0),a);
			if (v2>r1) r1=v2;
			res=integrate(K,a,b,z,0.0,r1,eps);
		}
		else
		{
			zz=1.0;
			std::vector<double> values;
			// for z<0 sum + and - elements and save to array
			for (int i=0;i<k0;i+=2)
			{
				v1=zz/Gamma(b+a*i);
				zz*=z;
				v2=zz/Gamma(b+a*(i+1));
				zz*=z;
				if (finite(v1+v2))
					values.push_back(v1+v2);
				else
					break;
			}
			// sum array with sorting and pyramidal summing
			res=sum_sort_pyramid(values);
		}
	}
	return res;
}
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// int(Eb,1(-a/(1-a) (tj-t)^a),t=ts,ts1) 
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

int F_noquadrature=0;
int F_nosimple=0;
int F_notimes=0;
int no_prints=0;
int F_func=0;
int F_nosorting=1;

// int(Eb,1(-a/(1-a) (tj-t)^a),t=ts,ts1) through recursive subdivision quadrature
double MF_f(double a,double b,double z,double r)
{
	return mittag_leffler(b,1,-(a/(1-a))*pow(z-r,a),1e-10);
}
double MF_int_quadrature(double tj,double ts1,double ts,double a,double b,double eps=1e-6)
{
	int t1=GetTickCount();
	double ret=integrate(MF_f,a,b,tj,ts,ts1,eps);
	if ((no_prints==0)&&(F_notimes==0)) printf(" q %d ",GetTickCount()-t1);
	return ret;
}
// int(Eb,1(-a/(1-a) (tj-t)^a),t=ts,ts1) through basic series representation
double last_kterm;
int max_kiter;
#define sign(x) (((x)>0.0)?1.0:-1.0)
double MF_int_ksum(int l,double tj,double a,double b,int maxkiter,double eps,int maxl)
{
	static std::vector< double* > logs_cache,sign_cache;
	static std::vector<double> lg_cache;
	static std::vector<double> ll_cache;
	static double sa=-1;
	static double sb=-1;
	static int smaxl=0;
	std::vector<double> kterms;
	double kterm,k,ret=0.0;
	int zero_in_row=0;
	double lfact=LogGamma(l+1.0);
	double la=log(a/(1.0-a));
	double ltj=log(tj);
	double leps=log(eps);
	double k1,k2,k3,k4,s4;
	double s=pow(-1.0,l);
	if (smaxl!=maxl)
	{
		ll_cache.resize(0);
		ll_cache.push_back(0.0);
		for (int i=1;i<maxl;i++)
			ll_cache.push_back(log((double)i));
		smaxl=maxl;
	}
	if (sa!=a)
	{
		for (int i=0;i<logs_cache.size();i++)
		{
			delete [] logs_cache[i];
			delete [] sign_cache[i];
		}
		logs_cache.resize(0);
		sign_cache.resize(0);
		for (int i=0;i<maxkiter;i++)
		{
			logs_cache.push_back(new double[maxl]);
			sign_cache.push_back(new double[maxl]);
			for (int j=0;j<maxl;j++)
			{
				logs_cache[i][j]=0.0;
				sign_cache[i][j]=2.0;
			}
		}
		sa=a;
	}
	if (sb!=b)
	{
		lg_cache.resize(0);
		for (int i=0;i<maxkiter;i++)
			lg_cache.push_back(-1.0);
		sb=b;
	}
	for (k=0;k<maxkiter;k++)
	{
		k1=k*la;
		if (lg_cache[k]<0)
		{
			k2=LogGamma(b*k+1.0);
			lg_cache[k]=k2;
		}
		else
			k2=lg_cache[k];
		k3=(a*k-l)*ltj;
		kterm=k1+k3-k2;
		k4=0;
		s4=1;
		if (sign_cache[k][l]==2)
		{
			for (int i=1;i<=l;i++)
			{
				k4+=log(fabs(a*k-i+1))-ll_cache[i];
				s4*=sign(a*k-i+1);
			}
			logs_cache[k][l]=k4;
			sign_cache[k][l]=s4;
		}
		else
		{
			k4=logs_cache[k][l];
			s4=sign_cache[k][l];
		}
		kterm+=k4;
		kterm=exp(kterm);
		kterm*=s*s4/(l+1.0);
		if (log(fabs(kterm))<leps-lfact)
			zero_in_row++;
		else
			zero_in_row=0;
		if (zero_in_row==10)
			break;
		if (!finite(kterm))
			break;
		if (F_nosorting)
			ret+=kterm;
		else
			kterms.push_back(kterm);
		s*=-1.0;
	}
	if (max_kiter<k) max_kiter=k;
	if (F_nosorting==0)
		return sum_sort_pyramid(kterms);
	return ret;
}
double MF_int(double tj,double ts1,double ts,double a,double b,int maxl=1000,int maxkiter=1000,double eps=1e-10,int *realmaxl=NULL)
{
	int t1=GetTickCount();
	double ret=0.0;
	double lterm,lk,ksum;
	int l;
	last_kterm=0.0;
	for (l=0;l<maxl;l++)
	{
		lk=pow(ts1,(l+1.0))-pow(ts,(l+1.0));
		ksum=MF_int_ksum(l,tj,a,b,maxkiter,eps,maxl);
		if (finite(ksum)) last_kterm=fabs(ksum);
		lterm=lk*ksum;
		if (finite(lterm))
			ret+=lterm;
	}
	if (realmaxl) realmaxl[0]=l;
	if ((no_prints==0)&&(F_notimes==0)) printf("R %d ",GetTickCount()-t1);
	return ret;
}
// int(Ea,1(-a/(1-a) (tj-t)^a),t=ts,ts1) using the approximation of integral MLF representation
double Fv(double a,double v)
{
	double p=-a/(1.0-a);
	double k1=1.0/(PI*a);
	double k2=p*sin(PI*a);
	double k3=v*v-2*v*p*cos(PI*a)+p*p;
	return k1*k2/k3;
}
double F2(double a,double v,double tj,double ts1,double ts)
{
	return pow(v,-1.0/a)*(exp(-pow(v,1.0/a)*(tj-ts))-exp(-pow(v,1.0/a)*(tj-ts1)));
}
double MF_intFv(double tj,double ts1,double ts,double a,int maxl=1000,double eps=1e-10)
{
	int t1=GetTickCount();
	double ret=0.0;
	double lterm;
	double C=1.0;
	int l;
	double Fm,Fd;
	double wk,vk;
	double wk1,wk2,wk3,mC1,mC2;
	// find maxC for with integrated function become < eps
	do
	{
		double ks=ts*pow(C,1.0/a);
		double ks1=ts1*pow(C,1.0/a);
		double k2=log(fabs(Fv(a,C)))-(1.0/a)*log(C)-pow(C,1.0/a)*tj;
		double v=fabs(exp(ks+k2)-exp(ks1+k2));
		if (v>eps)
			C*=1.5;
		else 
			break;
	}
	while (1);
	Fm=-a*cos(PI*a)/(1.0-a); // maximum of Fv
	Fd=5*(sqrt(3.0)/3)*sqrt((a*a/((1.0-a)*(1.0-a)))-Fm*Fm); // n * distance between maximum and inflection points
	mC1=Fm+Fd;
	mC2=Fm-Fd;
	if (mC2<0.0) mC2=0.0;
	if (mC1<0.0) mC1=0.0;
	if (mC2==0) mC2=mC1/2.0;
	if (mC1==0) { mC1=C/3.0; mC2=2.0*C/3.0; }
	// 1/3 of points from 0 to Fm-Fd, 1/3 between Fm-Fd and Fm+Fd, 1/3 from Fm+Fd to max C
	// equal steps in the first two thirds, linearly increasing step in the last third
	wk1=mC2/(maxl/3);
	wk2=(mC1-mC2)/(maxl/3);
	wk3=2.0*(C-mC2)/(maxl/3);
	for (l=0;l<maxl;l++)
	{
		if (l<(maxl/3))   { wk=wk1; vk=l*wk+0.5*wk; }
	else {  if (l<2*(maxl/3)) { wk=wk2; vk=mC2+(l-(maxl/3))*wk+0.5*wk; }
	else {  		    wk=wk2+(wk3-wk2)*(double)(l-2*(maxl/3))/(maxl/3); vk=mC1+(l-2*(maxl/3))*wk+0.5*wk; }}
		double ks=ts*pow(vk,1.0/a);
		double ks1=ts1*pow(vk,1.0/a);
		double k2=log(fabs(Fv(a,vk)))-(1.0/a)*log(vk)-pow(vk,1.0/a)*tj;
		lterm=wk*(exp(ks+k2)-exp(ks1+k2));
		if (Fv(a,vk)<0) lterm=-lterm;
		if (!finite(lterm))
			break;
		ret+=lterm;
	}
	if ((no_prints==0)&&(F_notimes==0)) printf("Fv(%g) %d ",C,GetTickCount()-t1);
	return ret;
}
/////////////////////
double MF_int_choice(double tj,double ts1,double ts,double a,double b,int maxl=1000,int maxkiter=1000,double eps=1e-10)
{
	double maxz=(a/(1-a))*pow(tj,a);
	double limit0=1.0;
	if ((maxz<limit0)||(b!=a))
		return MF_int(tj,ts1,ts,a,b,maxl,maxkiter,eps);
	return MF_intFv(tj,ts1,ts,a,maxl,1e-10);
}
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// int(f*Eb,1(-a/(1-a) (tj-t)^a),t=0,tj) 
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
double ppow=0.1;
double F(double t)
{
	if (F_func==0) return sin(t);
	if (F_func==1) return exp(t);
	if (F_func==2) return exp(-t);
	if (F_func==3) return pow(t,ppow-1)*ppow;
	if (F_func==6) return t;
	return 0.0;
}
double IF(double t)
{
	if (F_func==0) return 1-cos(t);
	if (F_func==1) return exp(t)-1;
	if (F_func==2) return 1-exp(-t);
	if (F_func==3) return pow(t,ppow);
	return 0.0;
}
// through recursive subdivision quadrature
double MF_fF(double a,double b,double z,double r)
{
	return F(r)*mittag_leffler(b,1,-(a/(1-a))*pow(z-r,a),1e-10);
}
double MF_intF_quadrature(double tj,double a,double b,double eps=1e-10)
{
	double ret;
	if (F_noquadrature) return 0.0;
	int t1=GetTickCount();
	ret=integrate(MF_fF,a,b,tj,0,tj,eps)/(1-a);
	if (F_notimes==0) printf("q %d ",GetTickCount()-t1);
	return ret;
}
// through series representation
double MF_intF(int n,double tau,double a,double b,int maxl=1000,int maxkiter=1000,double eps=1e-10,double (*FF)(double)=F,std::vector<double> *oLc=NULL, int on=0,int mn=0,int only_append=0,int is_Fvalue=0,double Fvalue=0)
{
	int t1=GetTickCount();
	double ret=0.0;
	int l;
	std::vector<double> Lc,*lLc=&Lc;
	static std::vector<double> Kc;
	static int old_n=-1;
	int fill_kc=0;
	if ((maxl!=Kc.size())||(n!=old_n))
	{
		Kc.resize(0);
		old_n=n;
		fill_kc=1;
	}
	if (only_append==0)
	for (l=0;l<maxl;l++)
	{
		Lc.push_back(0.0);		
		if (fill_kc==1)
			Kc.push_back(MF_int_ksum(l,n*tau,a,b,maxkiter,eps,maxl));
	}
	if (oLc!=NULL) lLc=oLc;
	for (int t=on;t<n-mn;t++)
	    for (l=0;l<maxl;l++)
			lLc[0][l]+=(pow((t+1)*tau,(l+1.0))-pow(t*tau,l+1.0))*((is_Fvalue==0)?FF(t*tau+tau*0.5):Fvalue);
	if (only_append)
		return 0.0;
	if (F_nosorting)
		for (l=0;l<maxl;l++)
			if ((finite(lLc[0][l]))&&(finite(Kc[l])))
				ret+=lLc[0][l]*Kc[l];
	else
	{
		for (l=0;l<maxl;l++)
			if ((finite(lLc[0][l]))&&(finite(Kc[l])))
				Lc[l]=lLc[0][l]*Kc[l];
			else
				Lc[l]=0.0;
		ret=sum_sort_pyramid(Lc);
	}
	if (F_notimes==0) printf("SR(%d) %d ",maxl,GetTickCount()-t1);
	return ret/(1.0-a);
}
// integral-weighted sum
double MF_intF_simple(int n,double tau,double a,double b,int quad=0,int maxl=1000,int maxkiter=1000,double eps=1e-10,double (*FF)(double)=F,int mn=0,double *Fvs=NULL)
{
	if (F_nosimple) return 0;
	static std::vector<double> I_cache;
	static int squad=-1;
	static double sa=-1;
	static double sb=-1;
	int t1=GetTickCount();
	double ret=0.0;
	no_prints=1;
	if ((squad!=quad)||(sa!=a)||(sb!=b))
	{
		I_cache.resize(0);
		squad=quad;
		sa=a;
		sb=b;
	}
	for (int tt=0;tt<n-mn;tt++)
	{
		double t=tt*tau;
		double I;
		int found=0;
		if (I_cache.size()>(n-tt))
			if (I_cache[n-tt]>=0)
			{
				found=1;
				I=I_cache[n-tt];
			}
		if (found==0)
		{
			if (quad==0) // series or integral
				I=MF_int_choice(n*tau,tt*tau+tau,tt*tau,a,b,maxl,maxkiter);
			if (quad==1) // quadrature
				I=MF_int_quadrature(n*tau,tt*tau+tau,tt*tau,a,b,1e-6);
			while (I_cache.size()<=(n-tt)) I_cache.push_back(-1.0);
			I_cache[n-tt]=I;
		}
		if (quad==2) // accuracy testing
		{
			double e1=0.0,e2=0.0;
			double t1=clock();
			int realmaxl=maxl;
			double I2=MF_int(n*tau,tt*tau+tau,tt*tau,a,b,maxl,maxkiter,eps,&realmaxl);
			t1=clock()-t1;
			double t2=clock();
			double I3=MF_intFv(n*tau,tt*tau+tau,tt*tau,a,maxl,eps);
			t2=clock()-t2;
			double opt_n1=1,opt_n2=1;
			double Cest;
			I=MF_int_quadrature(n*tau,tt*tau+tau,tt*tau,a,b,1e-10);
			// lower accuracy estimates for series
			{
				double r1=last_kterm*((tt*tau+tau)*exp(tt*tau+tau)-(tt*tau)*exp(tt*tau));
				double r2=(pow(tt*tau+tau,realmaxl+2)*exp(tt*tau+tau)-pow(tt*tau,realmaxl+2)*exp(tt*tau))/(realmaxl+1);
				e1=last_kterm*r2;
				double a=fabs(I-I2)*r1*log(tt*tau+tau);		
				if (a<0)
					opt_n1=-(1.0/log(tt*tau+tau))*(log(-1.0/a)-log(log(-1.0/a)));
			}
			// upper accuracy estimates for integral
			{
				double Fm=-a*cos(PI*a)/(1.0-a); // maximum of Fv
				double Fd=5*(sqrt(3.0)/3)*sqrt((a*a/((1.0-a)*(1.0-a)))-Fm*Fm); // n * distance between maximum and inflection points
				double cc=-a/(1.0-a);
				double aa=(1.0/(PI*a))*cc*sin(PI*a);
				double bb=cc*cos(PI*a);
				// find maxC for with integrated function become < eps
				Cest=1.0;
				do
				{
					double ts=tt*tau,ts1=tt*tau+tau;
					double tj=n*tau;
					double ks=ts*pow(Cest,1.0/a);
					double ks1=ts1*pow(Cest,1.0/a);
					double k2=log(fabs(Fv(a,Cest)))-(1.0/a)*log(Cest)-pow(Cest,1.0/a)*tj;
					double v=fabs(exp(ks+k2)-exp(ks1+k2));
					if (v>eps)
						Cest*=1.5;
					else 
						break;
				}
				while (1);
				double eps2=fabs(aa*F2(a,Cest,n*tau,tt*tau+tau,tt*tau)*(atan2(bb-Cest,pow(cc*cc-bb*bb,0.5))+PI*0.5)/pow(cc*cc-bb*bb,0.5));
				double mC1=Fm+Fd;
				double mC2=Fm-Fd;
				if (mC2<0.0) mC2=0.0;
				if (mC1<0.0) mC1=0.0;
				if (mC2==0) mC2=mC1/2.0;
				if (mC1==0) { mC1=Cest/3.0; mC2=2.0*Cest/3.0; }
				double f0=-Fv(a,0)*((tt*tau+tau)-(tt*tau));
				double fmC2=Fv(a,mC2)*F2(a,mC2,n*tau,tt*tau+tau,tt*tau);
				double fM=Fv(a,Fm)*F2(a,mC2,n*tau,tt*tau+tau,tt*tau);
				double fmC1=Fv(a,mC1)*F2(a,mC1,n*tau,tt*tau+tau,tt*tau);
				double fC=Fv(a,Cest)*F2(a,Cest,n*tau,tt*tau+tau,tt*tau);
				e2=eps2+(1.5/maxl)*(mC2*fabs(f0+fmC2)+fabs(mC1-mC2)*fabs(2*fM-fmC1-fmC2)+2.0*fabs(Cest-mC2)*fabs(fmC1-fC));
				opt_n2=(1.5/(fabs(I-I3)-eps2))*(mC2*fabs(f0+fmC2)+fabs(mC1-mC2)*fabs(2*fM-fmC1-fmC2)+2.0*fabs(Cest-mC2)*fabs(fmC1-fC));
			}
			printf("N %d eps %g ts %g tj %g a %g b %g S %g %d I %g Q %g diffS %g %g diffI %g %g est1 %g est2 %g on1 %g on2 %g oC %g timeS %g timeI %g\n",maxl,eps,tt*tau,n*tau,a,b,I2,realmaxl,I3,I,fabs(I2-I),fabs(I2-I)/I,fabs(I3-I),fabs(I3-I)/I,e1,e2,opt_n1,opt_n2,Cest,t1,t2);
		}		
		ret+=(Fvs?Fvs[tt]:FF(tt*tau+tau*0.5))*I;
	}
	no_prints=0;
	if (F_notimes==0) printf("S %d ",GetTickCount()-t1);
	return ret/(1.0-a);
}
// approximation of integral MLF representation
double MF_intF_Fv(int n,double tau,double a,int maxl=1000,double eps=1e-10,double (*FF)(double)=F,std::vector<double> *oLc=NULL,std::vector<double> *ologLc=NULL, int on=0,int mn=0,int only_append=0,int is_Fvalue=0,double Fvalue=0)
{
	int t1=GetTickCount();
	double ret=0.0;
	double wk,vk,C=1.0,maxv1=0.0,maxv2=0.0,mC1,mC2;
	double tj=n*tau;
	int l;
	std::vector<double> Lc,logLc,*lLc=&Lc,*llogLc=&logLc;
	double Fm,Fd;
	double wk1,wk2,wk3;
	static std::vector<double> Kc,logKc;
	static int old_n=-1;
	int fill_kc=0;
	if ((maxl!=Kc.size())||(n!=old_n))
	{
		Kc.resize(0);
		logKc.resize(0);
		old_n=n;
		fill_kc=1;
	}
	// upper integration limit for t_s+1=tj ( maxC for with integrated function become < eps )
	do
	{
		double ks=(tj-tau)*pow(C,1.0/a);
		double ks1=tj*pow(C,1.0/a);
		double k2=log(fabs(Fv(a,C)))-(1.0/a)*log(C)-pow(C,1.0/a)*tj;
		double v=fabs(exp(ks+k2)-exp(ks1+k2));
		if (v>eps)
			C*=1.5;
		else 
			break;
	}
	while (1);
	Fm=-a*cos(PI*a)/(1.0-a); // maximum of Fv
	Fd=5*(sqrt(3.0)/3)*sqrt((a*a/((1.0-a)*(1.0-a)))-Fm*Fm); // n * distance between maximum and inflection points
	mC1=Fm+Fd;
	mC2=Fm-Fd;
	if (mC2<0.0) mC2=0.0;
	if (mC1<0.0) mC1=0.0;
	if (mC2==0) mC2=mC1/2.0;
	if (mC1==0) { mC1=C/3.0; mC2=2.0*C/3.0; }
	// 1/3 of points from 0 to Fm-Fd, 1/3 between Fm-Fd and Fm+Fd, 1/3 from Fm+Fd to max C
	// equal steps in the first two thirds, linearly increasing step in the last third
	wk1=mC2/(maxl/3);
	wk2=(mC1-mC2)/(maxl/3);
	wk3=2.0*(C-mC2)/(maxl/3);
	if (only_append==0)
	for (l=0;l<maxl;l++)
	{
		double kc;
		if (l<(maxl/3))   { wk=wk1; vk=l*wk+0.5*wk; }
	else {  if (l<2*(maxl/3)) { wk=wk2; vk=mC2+(l-(maxl/3))*wk+0.5*wk; }
	else {  		    wk=wk2+(wk3-wk2)*(double)(l-2*(maxl/3))/(maxl/3); vk=mC1+(l-2*(maxl/3))*wk+0.5*wk; }}
		Lc.push_back(0.0);
		logLc.push_back(0.0);
		if (fill_kc==1)
		{
			logKc.push_back(0.0);
			kc=wk*pow(vk,-(1.0/a))*exp(-pow(vk,1.0/a)*tj)*Fv(a,vk);
			if (kc!=0.0)
				Kc.push_back(kc);
			else
			{
				logKc[l]=sign(Fv(a,vk)); // save kc in logarithm form if it is too small
				Kc.push_back(log(wk)-(1.0/a)*log(vk)-pow(vk,1.0/a)*tj+log(fabs(Fv(a,vk))));
			}
		}
	}
	if (oLc!=NULL) { lLc=oLc; llogLc=ologLc; }
	for (int t=on;t<n-mn;t++)
		for (l=0;l<maxl;l++)
		{
			if (l<(maxl/3))   { wk=wk1; vk=l*wk+0.5*wk; }
		else {  if (l<2*(maxl/3)) { wk=wk2; vk=mC2+(l-(maxl/3))*wk+0.5*wk; }
		else {  		    wk=wk2+(wk3-wk2)*(double)(l-2*(maxl/3))/(maxl/3); vk=mC1+(l-2*(maxl/3))*wk+0.5*wk; }}
			double sk=exp(pow(vk,1.0/a)*t*tau)-exp(pow(vk,1.0/a)*((t+1)*tau));
			double olc=lLc[0][l];
			if (llogLc[0][l]==0)
			{
				lLc[0][l]+=sk*((is_Fvalue==0)?FF(t*tau+tau*0.5):Fvalue);
				if (!finite(lLc[0][l])) // approximate log(Lc[l]) on overflow
				{
					lLc[0][l]=log(fabs(olc));
					llogLc[0][l]=sign(olc);
				}
			}
			else
			{	// assume new sk >>old Lc
				sk=pow(vk,1.0/a)*t*tau+log(exp(pow(vk,1.0/a)*tau)-1)+log(fabs(((is_Fvalue==0)?FF(t*tau+tau*0.5):Fvalue))); // ln(sk)
				lLc[0][l]=sk;
				llogLc[0][l]=-sign(((is_Fvalue==0)?FF(t*tau+tau*0.5):Fvalue));
			}
		}
	if (only_append==1) return 0.0;
	for (l=0;l<maxl;l++)
	{
		if ((llogLc[0][l]==0)&&(logKc[l]==0))
			Lc[l]=lLc[0][l]*Kc[l];
		if ((llogLc[0][l]!=0)&&(logKc[l]==0))
			Lc[l]=llogLc[0][l]*sign(Kc[l])*exp(lLc[0][l]+log(fabs(Kc[l])));
		if ((llogLc[0][l]==0)&&(logKc[l]!=0))
			Lc[l]=logKc[l]*sign(lLc[0][l])*exp(Kc[l]+log(fabs(lLc[0][l])));
		if ((llogLc[0][l]!=0)&&(logKc[l]!=0))
			Lc[l]=logKc[l]*llogLc[0][l]*exp(Kc[l]+lLc[0][l]);
		if (!finite(Lc[l])) Lc[l]=0.0;
	}
	if (F_nosorting==0)
		ret=sum_sort_pyramid(Lc)/(1.0-a);
	else
	{
		for (l=0;l<maxl;l++)
			ret+=Lc[l];
		ret/=1.0-a;
	}
	if (F_notimes==0) printf("Fv(%g) %d ",C,GetTickCount()-t1);
	return ret;
}
////////////////////
int last_choice;
double MF_intF_choice(int n,double tau,double a,double b,int maxl=1000,int maxkiter=1000,double eps=1e-10,double (*FF)(double)=F,std::vector<double> *oLc1=NULL,std::vector<double> *oLc2=NULL,std::vector<double> *ologLc2=NULL, int on=0,int mn=0,int append_all=0,int is_Fvalue=0,double Fvalue=0)
{
	double tj=n*tau;
	double maxz=(a/(1-a))*pow(tj,a);
	double limit0=1.0;
	if ((maxz<limit0)||(b!=a))
	{
		if (append_all)
			MF_intF_Fv(n,tau,a,maxl,eps,FF,oLc2,ologLc2,on,mn,1,is_Fvalue,Fvalue);
		last_choice=1;
		return MF_intF(n,tau,a,b,maxl,maxkiter,eps,FF,oLc1,on,mn,0,is_Fvalue,Fvalue);
	}
	last_choice=2;
	return MF_intF_Fv(n,tau,a,maxl,eps,FF,oLc2,ologLc2,on,mn,0,is_Fvalue,Fvalue);
}
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// solver for convective diffusion equation
// D_alpha F = d2/dx2 F -d/dx F + f
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
double f(double x, double t, double a)
{
	if (F_func==0) return (24.0/(1.0-a))*t*t*t*t*mittag_leffler(a,5.0,(-a/(1.0-a))*pow(t,a),1e-20);
	if (F_func==1) return (2.0*x-1.0)*t*t-2*t*t+(2.0/(1.0-a))*x*(x-1.0)*t*t*mittag_leffler(a,3.0,(-a/(1.0-a))*pow(t,a),1e-20);
	if (F_func==2) return (120.0/(1.0-a))*t*t*t*t*t*sin(PI*x)*mittag_leffler(a,6.0,(-a/(1.0-a))*pow(t,a),1e-20)+PI*t*t*t*t*t*(PI*sin(PI*x)+cos(PI*x));
	return 0.0;
}
double CC(double x,double t)
{
	if (F_func==0) return t*t*t*t;
	if (F_func==1) return x*(x-1.0)*t*t;
	if (F_func==2) return t*t*t*t*t*sin(PI*x);
	return 0.0;
}
// returns maximal error
double solve_and_check(double a,double T, int n,int m,int simple=0,int maxl=1000,int maxkiter=1000,double eps=1e-10)
{
	std::vector<double> C,prevC,Rp,al,bt;
	std::vector< std::vector<double> > Lc1,Lc2,logLc2,prev_dtCs;
	double prev_dtC=0;
	double h=1.0/m,tau=T/n;
	double A,B,Cc; // tridiagonal matrix coefs
	double Ii,err=0;
	A=-(1.0/(h*h))-(1.0/(2.0*h));
	Cc=-(1.0/(h*h))+(1.0/(2.0*h));
	int t0=GetTickCount();
	// initial values
	for (int i=0;i<=m;i++)
	{
		C.push_back(0.0);
		prevC.push_back(0.0);
		Rp.push_back(0.0);
		al.push_back(0.0);
		bt.push_back(0.0);
		Lc1.push_back(std::vector<double>());
		Lc2.push_back(std::vector<double>());
		logLc2.push_back(std::vector<double>());
		if (simple)
			prev_dtCs.push_back(std::vector<double>());
		for (int j=0;j<maxl;j++)
		{
			Lc1[i].push_back(0.0);
			Lc2[i].push_back(0.0);
			logLc2[i].push_back(0.0);
		}
	}
	// iterations
	Ii=MF_int_choice(tau,tau,0,a,a,maxl,maxkiter,eps);
	B=(Ii/(1.0-a))+(2.0/(h*h));
	for (int i=1;i<=n;i++)
	{
		int t1=GetTickCount();
		last_choice=0;
		max_kiter=0;
		// form linear system
		for (int j=0;j<=m;j++)
		{
			Rp[j]=(Ii/(1.0-a))*C[j]+f(j*h,i*tau,a);
			if (i>=2)
			{
				prev_dtC=(C[j]-prevC[j])/tau;
				if (simple==0)
					Rp[j]-=MF_intF_choice(i,tau,a,a,maxl,maxkiter,eps,NULL,&Lc1[j],&Lc2[j],&logLc2[j], i-2,1,1,1,prev_dtC);
				else
				{
					prev_dtCs[j].push_back(prev_dtC);
					Rp[j]-=	MF_intF_simple(i,tau,a,a,simple-1,maxl,maxkiter,eps,NULL,1,&prev_dtCs[j][0]);
				}
			}
		}
		// save C
		for (int j=0;j<=m;j++)
			prevC[j]=C[j];
		// solve linear system by Thomas algorithm
		// forward sweep
		al[0]=0;
		bt[0]=CC(0,i*tau);
		for (int j=1;j<=m;j++)
		{
			al[j]=Cc/(B-A*al[j-1]);
			bt[j]=(Rp[j]-A*bt[j-1])/(B-A*al[j-1]);
		}
		// backward sweep
		C[m]=CC(m*h,i*tau);
		for (int j=m-1;j>=0;j--)
			C[j]=bt[j]-al[j]*C[j+1];
		// calculate error
		for (int j=0;j<=m;j++)
		{
			double d=fabs(C[j]-CC(h*j,i*tau));
			err+=d*d;
		}
		if (F_notimes>2)
			printf("%g err %g time %d max_kiter %d last_choice %d\n",i*tau,(err/((m+1.0)*i)),GetTickCount()-t1,max_kiter,last_choice);
	}
	if (F_notimes==2)
		printf("total(%d) T %g a %g err %g time %d\n",simple,n*tau,a,(err/((m+1.0)*n)),GetTickCount()-t0);
	return err/((m+1.0)*n);
}
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////// testing ////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
void derivative_test()
{
	double T=2.0;
	double tau=0.01;
	for (int tt=0;tt<=(T/tau);tt++)
		printf("D[%g] = 1 - %g 0.99 - %g %g %g %g 0.9 - %g %g %g %g 0.8 - %g %g %g %g 0.7 - %g %g %g %g 0.6 - %g %g %g %g 0.5 - %g %g %g %g 0.4 - %g %g %g %g 0.3 - %g %g %g %g 0.2 - %g %g %g %g 0.1 - %g %g %g %g 0.01 - %g %g %g %g 0 - %g\n", 
			tt*tau,F(tt*tau),
			MF_intF_choice(tt,tau,0.99,0.99,100,1000),MF_intF_simple(tt,tau,0.99,0.99,0,100,1000),MF_intF_simple(tt,tau,0.99,0.99,1,100,1000),MF_intF_quadrature(tt*tau,0.99,0.99),
			MF_intF_choice(tt,tau,0.9,0.9,100,1000),MF_intF_simple(tt,tau,0.9,0.9,0,100,1000),MF_intF_simple(tt,tau,0.9,0.9,1,100,1000),MF_intF_quadrature(tt*tau,0.9,0.9),
			MF_intF_choice(tt,tau,0.8,0.8,100,1000),MF_intF_simple(tt,tau,0.8,0.8,0,100,1000),MF_intF_simple(tt,tau,0.8,0.8,1,100,1000),MF_intF_quadrature(tt*tau,0.8,0.8),
			MF_intF_choice(tt,tau,0.7,0.7,100,1000),MF_intF_simple(tt,tau,0.7,0.7,0,100,1000),MF_intF_simple(tt,tau,0.7,0.7,1,100,1000),MF_intF_quadrature(tt*tau,0.7,0.7),
			MF_intF_choice(tt,tau,0.6,0.6,100,1000),MF_intF_simple(tt,tau,0.6,0.6,0,100,1000),MF_intF_simple(tt,tau,0.6,0.6,1,100,1000),MF_intF_quadrature(tt*tau,0.6,0.6),
			MF_intF_choice(tt,tau,0.5,0.5,100,1000),MF_intF_simple(tt,tau,0.5,0.5,0,100,1000),MF_intF_simple(tt,tau,0.5,0.5,1,100,1000),MF_intF_quadrature(tt*tau,0.5,0.5),
			MF_intF_choice(tt,tau,0.4,0.4,100,1000),MF_intF_simple(tt,tau,0.4,0.4,0,100,1000),MF_intF_simple(tt,tau,0.4,0.4,1,100,1000),MF_intF_quadrature(tt*tau,0.4,0.4),
			MF_intF_choice(tt,tau,0.3,0.3,100,1000),MF_intF_simple(tt,tau,0.3,0.3,0,100,1000),MF_intF_simple(tt,tau,0.3,0.3,1,100,1000),MF_intF_quadrature(tt*tau,0.3,0.3),
			MF_intF_choice(tt,tau,0.2,0.2,100,1000),MF_intF_simple(tt,tau,0.2,0.2,0,100,1000),MF_intF_simple(tt,tau,0.2,0.2,1,100,1000),MF_intF_quadrature(tt*tau,0.2,0.2),
			MF_intF_choice(tt,tau,0.1,0.1,100,1000),MF_intF_simple(tt,tau,0.1,0.1,0,100,1000),MF_intF_simple(tt,tau,0.1,0.1,1,100,1000),MF_intF_quadrature(tt*tau,0.1,0.1),
			MF_intF_choice(tt,tau,0.01,0.01,100,1000),MF_intF_simple(tt,tau,0.01,0.01,0,100,1000),MF_intF_simple(tt,tau,0.01,0.01,1,100,1000),MF_intF_quadrature(tt*tau,0.01,0.01),
			IF(tt*tau));
}
void derivative_test_power()
{
	double Ts[]={0.1,0.5,1.0,2.0};
	double Ps[]={0.1,0.3,0.6,1.0};
	int np=4;
	int nt=4;
	double T=2.0;
	double tau=0.01;
	int maxl=300, maxkiter=1000;
	F_func=3;
	for (int j=0;j<np;j++)
	{
	ppow=Ps[j];
	printf("%g\n",ppow);
	for (int i=0;i<nt;i++)
	{
		int tt=Ts[i]/tau;
		printf("D[%g] = 1 - %g 0.99 - %g %g %g %g 0.9 - %g %g %g %g 0.8 - %g %g %g %g 0.7 - %g %g %g %g 0.6 - %g %g %g %g 0.5 - %g %g %g %g 0.4 - %g %g %g %g 0.3 - %g %g %g %g 0.2 - %g %g %g %g 0.1 - %g %g %g %g 0.01 - %g %g %g %g 0 - %g\n", 
			tt*tau,F(tt*tau),
			MF_intF_choice(tt,tau,0.99,0.99,maxl,maxkiter),MF_intF_simple(tt,tau,0.99,0.99,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.99,0.99,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.99,0.99),
			MF_intF_choice(tt,tau,0.9,0.9,maxl,maxkiter),MF_intF_simple(tt,tau,0.9,0.9,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.9,0.9,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.9,0.9),
			MF_intF_choice(tt,tau,0.8,0.8,maxl,maxkiter),MF_intF_simple(tt,tau,0.8,0.8,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.8,0.8,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.8,0.8),
			MF_intF_choice(tt,tau,0.7,0.7,maxl,maxkiter),MF_intF_simple(tt,tau,0.7,0.7,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.7,0.7,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.7,0.7),
			MF_intF_choice(tt,tau,0.6,0.6,maxl,maxkiter),MF_intF_simple(tt,tau,0.6,0.6,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.6,0.6,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.6,0.6),
			MF_intF_choice(tt,tau,0.5,0.5,maxl,maxkiter),MF_intF_simple(tt,tau,0.5,0.5,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.5,0.5,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.5,0.5),
			MF_intF_choice(tt,tau,0.4,0.4,maxl,maxkiter),MF_intF_simple(tt,tau,0.4,0.4,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.4,0.4,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.4,0.4),
			MF_intF_choice(tt,tau,0.3,0.3,maxl,maxkiter),MF_intF_simple(tt,tau,0.3,0.3,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.3,0.3,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.3,0.3),
			MF_intF_choice(tt,tau,0.2,0.2,maxl,maxkiter),MF_intF_simple(tt,tau,0.2,0.2,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.2,0.2,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.2,0.2),
			MF_intF_choice(tt,tau,0.1,0.1,maxl,maxkiter),MF_intF_simple(tt,tau,0.1,0.1,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.1,0.1,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.1,0.1),
			MF_intF_choice(tt,tau,0.01,0.01,maxl,maxkiter),MF_intF_simple(tt,tau,0.01,0.01,0,maxl,maxkiter),MF_intF_simple(tt,tau,0.01,0.01,1,maxl,maxkiter),MF_intF_quadrature(tt*tau,0.01,0.01),
			IF(tt*tau));
	}
	}
}
void derivativeAB_test()
{
	double T=1.0;
	double tau=0.1;
	double a=0.8;
	for (int tt=0;tt<=(T/tau);tt++)
		printf("D[%g] = 1 - %g 0.99 - %g %g %g %g 0.9 - %g %g %g %g 0.8 - %g %g %g %g 0.7 - %g %g %g %g 0.6 - %g %g %g %g 0.5 - %g %g %g %g 0.4 - %g %g %g %g 0.3 - %g %g %g %g 0.2 - %g %g %g %g 0.1 - %g %g %g %g 0.01 - %g %g %g %g 0 - %g\n", 
			tt*tau,F(tt*tau),
			MF_intF_choice(tt,tau,a,1,100,1000),MF_intF_simple(tt,tau,a,1,0,100,1000),MF_intF_simple(tt,tau,a,1.0,1,100,1000),MF_intF_quadrature(tt*tau,a,1.0),
			MF_intF_choice(tt,tau,a,0.9,100,1000),MF_intF_simple(tt,tau,a,0.9,0,100,1000),MF_intF_simple(tt,tau,a,0.9,1,100,1000),MF_intF_quadrature(tt*tau,a,0.9),
			MF_intF_choice(tt,tau,a,0.8,100,1000),MF_intF_simple(tt,tau,a,0.8,0,100,1000),MF_intF_simple(tt,tau,a,0.8,1,100,1000),MF_intF_quadrature(tt*tau,a,0.8),
			MF_intF_choice(tt,tau,a,0.7,100,1000),MF_intF_simple(tt,tau,a,0.7,0,100,1000),MF_intF_simple(tt,tau,a,0.7,1,100,1000),MF_intF_quadrature(tt*tau,a,0.7),
			MF_intF_choice(tt,tau,a,0.6,100,1000),MF_intF_simple(tt,tau,a,0.6,0,100,1000),MF_intF_simple(tt,tau,a,0.6,1,100,1000),MF_intF_quadrature(tt*tau,a,0.6),
			MF_intF_choice(tt,tau,a,0.5,100,1000),MF_intF_simple(tt,tau,a,0.5,0,100,1000),MF_intF_simple(tt,tau,a,0.5,1,100,1000),MF_intF_quadrature(tt*tau,a,0.5),
			MF_intF_choice(tt,tau,a,0.4,100,1000),MF_intF_simple(tt,tau,a,0.4,0,100,1000),MF_intF_simple(tt,tau,a,0.4,1,100,1000),MF_intF_quadrature(tt*tau,a,0.4),
			MF_intF_choice(tt,tau,a,0.3,100,1000),MF_intF_simple(tt,tau,a,0.3,0,100,1000),MF_intF_simple(tt,tau,a,0.3,1,100,1000),MF_intF_quadrature(tt*tau,a,0.3),
			MF_intF_choice(tt,tau,a,0.2,100,1000),MF_intF_simple(tt,tau,a,0.2,0,100,1000),MF_intF_simple(tt,tau,a,0.2,1,100,1000),MF_intF_quadrature(tt*tau,a,0.2),
			MF_intF_choice(tt,tau,a,0.1,100,1000),MF_intF_simple(tt,tau,a,0.1,0,100,1000),MF_intF_simple(tt,tau,a,0.1,1,100,1000),MF_intF_quadrature(tt*tau,a,0.1),
			MF_intF_choice(tt,tau,a,0.01,100,1000),MF_intF_simple(tt,tau,a,0.01,0,100,1000),MF_intF_simple(tt,tau,a,0.01,1,100,1000),MF_intF_quadrature(tt*tau,a,0.01),
			IF(tt*tau));
}
void integrals_test()
{
	double tau=0.1;
	double a=0.8;
	F_nosimple=0;
	F_notimes=1;
	int tt=30,N=100,e=10;
	tt=30;
	printf("\n%g ",MF_intF_simple(tt,tau,0.01,0.01,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.1,0.1,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.2,0.2,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.3,0.3,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.4,0.4,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.5,0.5,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.6,0.6,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.7,0.7,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.8,0.8,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.9,0.9,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,0.99,0.99,2,N,1000,pow(10.0,-e)));
	for (int tt=1;tt<10;tt++)
		printf("\n%g ",MF_intF_simple(tt,tau,a,a,2,N,1000,pow(10.0,-e)));	
	tt=30;
	printf("\n%g ",MF_intF_simple(tt,tau,a,a,2,N,1000,pow(10.0,-e)));
	tt=30;
	for (int N=25;N<250;N+=25)
		printf("\n%g ",MF_intF_simple(tt,tau,a,a,2,N,1000,pow(10.0,-e)));
	N=100;
	for (int e=5;e<=15;e++)
		printf("\n%g ",MF_intF_simple(tt,tau,a,a,2,N,1000,pow(10.0,-e)));
}
void integralsAB_test()
{
	double tau=0.1;
	double a=0.8;
	F_nosimple=0;
	F_notimes=1;
	int tt=10,N=100,e=10;
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.01,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.1,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.2,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.3,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.4,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.5,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.6,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.7,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.8,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,0.9,2,N,1000,pow(10.0,-e)));
	printf("\n%g ",MF_intF_simple(tt,tau,a,1.0,2,N,1000,pow(10.0,-e)));
}
void convection_diffusion_test(double T,double a,int n,int eq=0)
{
	int ms[5]={10,20,40,80,160};
	double errs[5];
	double conv[5];
	int mn=5;
	for (int i=0;i<mn;i++)
	{
		if (eq) n=ms[i];
		int t1=GetTickCount();
		errs[i]=solve_and_check(a,T,ms[i],n);
		int t2=GetTickCount();
		if (i==0) 
			conv[i]=0;
		else
			conv[i]=log(errs[i-1]/errs[i])/log(ms[i]/ms[i-1]);
		printf("a %g n %d m %d err %g conv rate %g time %d ms\n",a,n,ms[i],errs[i],conv[i],t2-t1);
	}
}
void convection_diffusion_test_full()
{
	convection_diffusion_test(1.0,0.1,1000,1);
//	convection_diffusion_test(1.0,0.2,1000,1);
//	convection_diffusion_test(1.0,0.3,1000,1);
//	convection_diffusion_test(1.0,0.4,1000,1);
	convection_diffusion_test(1.0,0.5,1000,1);
//	convection_diffusion_test(1.0,0.6,1000,1);
//	convection_diffusion_test(1.0,0.7,1000,1);
//	convection_diffusion_test(1.0,0.8,1000,1);
	convection_diffusion_test(1.0,0.9,1000,1);
	convection_diffusion_test(1.0,0.1,1000);
//	convection_diffusion_test(1.0,0.2,1000);
//	convection_diffusion_test(1.0,0.3,1000);
//	convection_diffusion_test(1.0,0.4,1000);
	convection_diffusion_test(1.0,0.5,1000);
//	convection_diffusion_test(1.0,0.6,1000);
//	convection_diffusion_test(1.0,0.7,1000);
//	convection_diffusion_test(1.0,0.8,1000);
	convection_diffusion_test(1.0,0.9,1000);
}
void convection_diffusion_test_time()
{
	F_notimes=3;
	for (double a=0.9;a>0.05;a-=0.1)
	{
		solve_and_check(a,1.0,6000,100,0,50);
		printf("\n");
		solve_and_check(a,1.0,6000,100,1,50);
		printf("\n");
		solve_and_check(a,1.0,6000,100,2,50);
		printf("\n");
	}
	solve_and_check(0.01,1.0,6000,100,0,50);
	printf("\n");
	solve_and_check(0.01,1.0,6000,100,1,50);
	printf("\n");
	solve_and_check(0.01,1.0,6000,100,2,50);
	printf("\n");
}
/// main/////////////////
int main(int argc,char **argv)
{
	F_noquadrature=1;
	F_nosimple=1;
	F_notimes=0;
	F_func=0;
	F_nosorting=1;
	int test=0;
	if (argc>=2) test=atoi(argv[1]);
	if (argc>=3) F_func=atoi(argv[2]);
	if (argc>=4) F_notimes=atoi(argv[3]);
	if (argc>=5) F_noquadrature=atoi(argv[4]);
	if (argc>=6) F_nosimple=atoi(argv[5]);
	if (argc>=7) F_nosorting=atoi(argv[6]);
	printf("Func %d Times %d Quad %d Simple %d Sorting %d\n",F_func,1-F_notimes,1-F_noquadrature,1-F_nosimple,1-F_nosorting);
	if (test==0) derivative_test();
	if (test==1) integrals_test();
	if (test==2) derivativeAB_test();
	if (test==3) integralsAB_test();
	if (test==4) convection_diffusion_test_full();
	if (test==5) convection_diffusion_test_time();
	if (test==6) derivative_test_power();
	return 0;
}
