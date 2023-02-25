#define _USE_MATH_DEFINES
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
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
omp_lock_t lock;
#endif

#define pow __builtin_pow
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
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////

int BS=10; // block size
int NB=2,sNB; // number of block
int N,sN;  // number of nodes
double A = 0.1; // 1-A for calc_V
double *F; // function values
double *BVK; // values of F
int flags=0; // bit 1 - no first-second order boundary conditions transition
double irr_delay=0; // delay before the first irrigation
double inter_irr_delay=0; // delay between successive irrigations
double stat_div=100.0;
int stat_maxiter=1000;
double stat_eps=1e-5;
int irr=0;
int fit=0;
double Hmult=10.0;
double HtoPmult=98/Hmult;
double klin=0;
double averianov_power=3.5;
double bounds[18][2]={{0.8,1.0}, // a
		     {0.8,1.0}, // b
		     {1.0,5.0}, // L
		     {0.5,2.0}, // vgm1
		     {0.01,0.4}, // vgm2
		     {0.1,1.0}, // vgm3
		     {0.01,0.4}, // vgm4
		     {0.000005/100.0,0.005/100.0}, // k
		     {0.6,1.0}, // min wetness
		     {0.6,1.0}, // max wetness
		     {1,100}, // irr amount
		     {0.005,10*0.001}, // EVT coefs max, perc_k max
		     {0.25,-0.25}, // upper/lower bc restrictions
		     {1e-7,0},// kk restriction in bc
			 {-5,5}, // func_power limits - space
			 {0,1},  // sigmoid second parameter limits - space
			 {-5,5}, // func_power limits - time
			 {0,1}};  // sigmoid second parameter limits - time
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
double kb2_coef=0.5; // % of integrals calculated by kb_row2
// generalized Caputo derivative
int func_in_kernel = 0; // 0 -x, 1 - x^1/2, 2 - x^2, 3 - x^k
int integr_max_niter=100000;
double func_power = 1.0; // for space variables
double func_power_t =0.0; // for time variable
double sigmoid2=1.0; // for space variables
double sigmoid2_t=1.0; // for time variable
double sigmoid_a,sigmoid_b,sigmoid_c,sigmoid_d;
double sigmoid_inv_a,sigmoid_inv_b,sigmoid_inv_c,sigmoid_inv_d;
double global_eps=1e-5;
double global_eps2=1e-5; // used to control kb_row2
double g(double t)
{
	if (func_in_kernel == 1)
		return sqrt(t);
	if (func_in_kernel == 2)
		return t*t;
	if (func_in_kernel == 3)
		return pow(t,func_power);
	if (func_in_kernel==4) // sigmoid (space)
	{
		if (func_power>0.0) // exponental sigmoid
			return (1.0/sigmoid_a)*((1.0/(1.0+exp(-func_power*(sigmoid_b*t-sigmoid_c))))-sigmoid_d);
		else
			return sigmoid_a*log((sigmoid_b*t+sigmoid_c)/(1-(sigmoid_b*t+sigmoid_c)))+sigmoid_d;
	}
	return t;
}
// should return kf(n)(t) given kf(n-1)(t)=der_nm1
// for n=0 return f(t)
double inv_g_der(double t, int n, double der_nm1)
{
	static std::vector<std::vector<double> > sigmoid_coefs;
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
			return pow(t,1.0/func_power);
		if (n == 1)
			return ((1.0/func_power) * pow(t,(1.0/func_power)-1))*(der_nm1 / pow(t,1.0/func_power));
		return der_nm1*((1.0/func_power) - n + 1) / t;
	}
	if (func_in_kernel==4)
	{
		if (func_power>0.0)
		{
			if (n==0)
				return sigmoid_inv_a*log((sigmoid_inv_b*t+sigmoid_inv_c)/(1-(sigmoid_inv_b*t+sigmoid_inv_c)))+sigmoid_inv_d;
			if (n>=1)
			{
				double n1f=1;
				for (int i=1;i<=n-1;i++) n1f*=i;
				double p1=pow(-1.0,n-1)*sigmoid_inv_a*pow(sigmoid_inv_b,n)*n1f/pow(sigmoid_inv_b*t+sigmoid_inv_c,n);
				double p2=sigmoid_inv_a*pow(sigmoid_inv_b,n)*n1f/pow(1-(sigmoid_inv_b*t+sigmoid_inv_c),n);
				return p1+p2;
			}			
		}
		else
		{
			if (n==0)
				return (1.0/sigmoid_inv_a)*((1.0/(1.0+exp(func_power*(sigmoid_inv_b*t-sigmoid_inv_c))))-sigmoid_inv_d);
			else
			{
				n--;
				double v=1.0/(1.0+exp(func_power*(sigmoid_inv_b*t-sigmoid_inv_c))),vp;
				double k1=-func_power*sigmoid_inv_b;
#ifdef _OPENMP
omp_set_lock(&lock);
#endif
				for (int i=sigmoid_coefs.size();i<=n;i++)
				{
					sigmoid_coefs.push_back(std::vector<double>());
					if (i==0)
					{
						sigmoid_coefs[0].push_back(1.0);
						sigmoid_coefs[0].push_back(-1.0);
					}
					else
					{
						sigmoid_coefs[i].push_back(0);
						for (int j=0;j<i+1;j++)
							sigmoid_coefs[i].push_back(sigmoid_coefs[i-1][j]*(j+1));
						sigmoid_coefs[i].push_back(0);
						for (int j=0;j<i+2;j++)
							sigmoid_coefs[i][j]=-sigmoid_coefs[i][j]+sigmoid_coefs[i][j+1];
					}
				}
#ifdef _OPENMP
omp_unset_lock(&lock);
#endif
				double ret=0.0;
				vp=v;
#ifdef _OPENMP
omp_set_lock(&lock);
#endif
				for (int i=0;i<sigmoid_coefs[n].size();i++)
				{
					ret+=vp*sigmoid_coefs[n][i];
					vp*=v;
				}
#ifdef _OPENMP
omp_unset_lock(&lock);
#endif
				ret*=pow(k1,n+1)/sigmoid_inv_a;
				return ret;
			}
		}
	}
	return ((n == 0) ? t : ((n == 1) ? (der_nm1 / t) : 0));
}
////////////////////////////////////////////
// time-space-fractional solver 1d /////////////
// Da_tH=div(CvDb_zH))-S/C(H)
////////////////////////////////////////////
class H_solver {
public:
	double *b_U; // pressure head
	double *U;
	double *Al; // alpha coefficients
	double *Bt; // beta coefficients
	double *Om; // right part
	std::vector<double*> oldH; // for time-fractional derivatives
	// for 3d
	double **rp_mult;
	double **spr;
	// steps
	double tau; // time step length 
	int tstep; // current time step
	double L,dL,sL,sdL; // domain length and space variable step length
	// H equation and common
	double alpha, gamma; // alpha - time derivative power, gamma - space derivative power
	double H0; // initial condition for H
	double Da,Dg; 
	double v; // root layer depth
	double Tr,Ev; // transpiration and evaporation rates
	double k; // filtration coefficient in saturated soil
	double saved_k; // saved filtration coefficient
	double Hbottom;
	// VGM parameters
	double vgm_n;
	double vgm_s0;
	double vgm_s1;
	double vgm_a;
	double av_p;
	double *vgm_ns,*vgm_s0s,*vgm_s1s,*vgm_as,*vgm_h0,*vgm_h1,*vgm_k,*av_ps; // VGM coefs for layered soil
	int vgm_nlayers;
	// EV per day
	double *EV_T;
	double **EV_F;
	double *EV_C;
	double *LAI;
	double ev_mu;
	int nEV_T;
	int nEV_F;
	int isEV;
	// variable root length 
	int nrl;
	double *rlT,*rlV;	
	// fixed percipetation
	double *perc_T, *perc_A;
	double perc_k;
	int nperc;
	// fixed irr
	double *irr_T, *irr_A;
	int nirr;
	// fixed internal points
	double *fixed_Z,*fixed_U;
	int nfixed;
	// irrigation scheduling 
	double min_wetness; // min wetness to apply irrigation
	double max_wetness; // wetness to be after irrigation
	double irr_volume; // volume of irrigation water
	double irr_time; // time range to apply irrigation water
	double p_soil,p_water; // densities of soil and pore water
	int irr_start_step; 
	int last_irr_step;
	int bottom_cond; 
	int rp_form,cv_form,k_form,no_irr;
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
			if ((f1 / f) < (f2 / f1))
				step /= (f2/f1)/(f1/f);
			while ((t + step) > t1)
				step /= 2.0;
			f = f2;
		}
		while ((t1-t)>global_eps*global_eps*global_eps);
		return sum;
	}
	//int((g(t_j+ja)-g(tau))^(-alpha),tau=t0,...,t1)
	// using recursive subdivision
	double _kb(int n, double t0, double t1, double gtj,int sing,double alpha)
	{
		double v1, v2;
		v1 = __kb(n, t0, t1, gtj,alpha);
		v2 = __kb(n*2, t0, t1, gtj,alpha);
		if ((t1 - t0) < global_eps) 
		{
			if (sing)
				return ___kb(n*16,t0,t1,gtj,alpha);
			return 0.5*(v1+v2);
		}
		if (fabs(v2 - v1)>global_eps)
		{
			v1 = _kb(n, t0, 0.5*(t0 + t1), gtj,sing,alpha);
			v2 = _kb(n, 0.5*(t0 + t1), t1,gtj,sing,alpha);
			return v1 + v2;
		}
		else
			return v2;
	}
	// using taylor series on gti-g(x) in g(ti)
	// for right-part - series on g(x)-gti in g(t1)
	double _kb_row(double t0, double t1, double gtj,double alpha)
	{
		double sum = 0.0, v = 0.0, v2, v3, v4;
		double i = 0;
		int niter = 0;
		if ((alpha == 1.0)&&(g(t1)==gtj))
			return 1.0;
		if ((alpha==1.0)&&(g(t1)!=gtj))
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
				if (!finite(sum))
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
			v2 =( (pow(gh, 1 - alpha) / pow(gl, 1 - alpha)) - 1.0) / (1 - alpha); // I0/gl^(1-a)
			sum += v2*v;
			while (fabs(v*v2) > global_eps)
			{
				i += 1.0;
				// gh/(n+1-a) * f(n+1)(g(t1))/gl^(1-a)
				v = inv_g_der(g1, i + 1.0, v);
				v /= (1 - alpha + i) / gh;
				// v2 - Bi, v4 - Ci
				if (i == 1.0)
					v4 = (gl - gh)/gh;
				else
					v4 *= (gl - gh)*((i - 1.0) / i)*((-alpha + i) / (gh*(i - 1.0)));
				v2 = -(v2 + v4);
				sum += v2*v;
				niter++;
				if (niter > integr_max_niter)
					break;
				if (!finite(sum))
					break;
			}
		}
		return sum;
	}
	// using newton binomial on gtj and taylor series of f'(x) in g(t1)
	std::vector< std::vector< std::vector<double> > > row2_precalc;
	double _kb_row2(double t0, double t1, double gtj, int idx,int idx2,double alpha)
	{
		double sum = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4, v5, v6, v7;
		double gt0 = g(t0), gt1 = g(t1);
		double i = 0, m,eeps=global_eps2;
		int mode=0; 
		int found;
		int niter = 0;
		if ((alpha == 1.0)&&(g(t1)==gtj))
			return 1.0;
		if ((alpha==1.0)&&(g(t1)!=gtj))
			return 0.0;
#ifdef _OPENMP
omp_set_lock(&lock);
#endif
		if (idx!=-1)
		{
			while (row2_precalc.size()<(idx + 1))
				row2_precalc.push_back(std::vector< std::vector<double> >());
			while (row2_precalc[idx].size()<(idx2 + 1))
				row2_precalc[idx].push_back(std::vector<double>());
		}
#ifdef _OPENMP
omp_unset_lock(&lock);
#endif
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
			v2 = pow(gt0, 1-alpha);
			v6 = pow(gt1,1-alpha) / pow(gt0,1-alpha);
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
#ifdef _OPENMP
omp_set_lock(&lock);
#endif
			if (idx!=-1)
			if (row2_precalc[idx][idx2].size()>(int)i)
			{
				v3 = row2_precalc[idx][idx2][(int)i];
				found = 1;
			}
#ifdef _OPENMP
omp_unset_lock(&lock);
#endif
			if (found == 0)
			{
				if ((i == 0.0)&&(mode==0))
					v3 = (t1 - t0) / gt1;
				else
				{
					v3 = 0.0;
					m = 0.0;
					// f(m+1)(gd)/m!
					v4 = inv_g_der(gt1, 1, inv_g_der(gt1, 0, 0));
					// I(n,m)=integrate(x^n*(x-g(t1))^m,x,g(t0),g(t1));
					// I(n,0)=(1/n+1)*(g(t1)^(n+1)-g(t0)^n+1)
					if (mode==0)
						v5 = (1.0 / (i + 1.0))*(1.0 - v6);
					else
						v5 = -(1.0 / (-alpha-i + 1.0))*(1.0 - v6);
					v3 += v4*v5;
					v7 = 1;
					while (fabs(v4*v5) > (fabs(v3)*global_eps))
					{
						m+=1.0;
						// (f(m+1)(gd)/m!)
						v4 = inv_g_der(gt1, m + 1.0, v4);
						v4 /= m;
						if (mode==0)
							v4 /= ((i + m + 1.0) / (gt1*m));
						else
							v4 /= ((-alpha-i + m + 1.0) / (gt1*m));

						if (m == 1.0)
						{
							if (mode == 0)
								v7 = (1.0 / (gt1))*(v6*(gt0 - gt1));
							else
								v7 = (1.0 / (gt1))*(gt0 - gt1);
						}
						else
						{
							if (mode==0)
								v7 *= (gt0 - gt1)*((m - 1.0) / m)*((i + m) / (gt1*(m - 1.0)));
							else
								v7 *= (gt0 - gt1)*((m - 1.0) / m)*((-alpha-i + m) / (gt1*(m - 1.0)));
						}
						// I(n,m)=-(I(n,m-1)+(1/(g(t1)*m))*(g(t0)^(n+1)*(g(t0)-g(t1))^m))/((n+m+1)/(g(t1)*m)
						v5 = -(v5 + v7);
						v3 += v4*v5;
						niter++;
						if (niter>integr_max_niter)
							break;
						if (!finite(v3))
							break;
					}
				}
#ifdef _OPENMP
omp_set_lock(&lock);
#endif
				if (idx!=-1)
				{
					while (row2_precalc[idx][idx2].size()<(i + 1.0))
						row2_precalc[idx][idx2].push_back(0);
					row2_precalc[idx][idx2][(int)i] = v3;
				}
#ifdef _OPENMP
omp_unset_lock(&lock);
#endif
			}
			sum += v1*v2*v3;
			i+=1.0;
			niter++;
			if (niter>integr_max_niter)
				break;
			if (!finite(sum))
				break;
		} while (fabs(v1*v2*v3) > eeps);
		return sum;
	}
	//b(t0,t1,a) = int((g(x_a)-g(tau))^(-alpha),tau=x_t0,x_t1)
	// use newton binomial near t0==0
	// use taylor series near t1==a
	// move usage boundary to ensure minimal number of iteration done by both algorithms
	double *kb_space=NULL;
	double *kb_time=NULL;
	double calc_int(double alpha, int t0,int t1,int a,double dL,int time)
	{
		double v=0;
		if (alpha == 1.0)
		{
			if (t1==a)
				return 1.0;
			else
				return 0.0;
		}
		//printf("%d %d %d %g %d %g %g -> ",t0,t1,a,alpha,func_in_kernel,func_power,sigmoid2);
		if (func_in_kernel==0)
		    return _kb_row(t0*dL,t1*dL,g(a*dL), alpha);
		if ((t1<=kb2_coef*a)&&(!((func_in_kernel==3)&&(func_power==1))))
			v = _kb_row2(t0*dL,t1*dL,g(a*dL),time,t0,alpha);
		else
			v = _kb_row(t0*dL,t1*dL,g(a*dL), alpha);
		if (!finite(v))
			v=_kb(1, t0*dL,t1*dL,g(a*dL),t1==a,alpha);			
		//printf("%g\n",v);
		return v;
	}
	double kb(double alpha, int t0,int t1,int a,double dL,int time=0)
	{
		if (time==0)
    		    return kb_space[a*(sN+2)+t0];
        return kb_time[t0];
	}
	// calculate values of F for block
	void calc_v(int block,double *F,double *V,double dL,int with_i=0,int with_zero=0,int clear_precalc=0)
	{
		for (int i = 0;i<BS;i++)
			V[i] = 0.0;
		if (A == 0.0)
		{
			if (with_i)
				for (int i=block*BS+1;i<(block+1)*BS+1;i++)
					V[i-(block*BS+1)]=F[i];
			return;
		}
#pragma omp parallel for
		for (int i=block*BS+1;i<(block+1)*BS+1;i++)
		{
			double v=0.0;
			int end=i-1;
			if (with_i) end=i;
			for (int j = ((with_zero==0)?1:0);j <= end;j++)
			{
				double kv;
				if (j!=0)
					kv =A*kb(1 - A, j-1, j, i, dL); 
				else
					kv =A*kb(1 - A, j, j+1, i+1, dL); 
				v += kv*F[j];
			}
			V[i-(block*BS+1)]=v;
		}
	}
	
	// minimum (0.5) ET at 0, maximum (1.5) at 12
	double day_night_coef()
	{
	    if (flags&4) return 1.0;
	    return 1+0.5*sin(M_PI*((tstep*tau/3600.0)+6.0)/12.0);
	}
	void vgm_get_coefs(int i)
	{
	    if (vgm_nlayers!=0)
	    {
		for (int j=0;j<vgm_nlayers;j++)
		    if (((i*dL)>=vgm_h0[j])&&((i*dL)<vgm_h1[j]))
		    {
			vgm_s0=vgm_s0s[j];
			vgm_s1=vgm_s1s[j];
			vgm_a=vgm_as[j];
			vgm_n=vgm_ns[j];
			k=vgm_k[j];
			av_p=av_ps[j];
			return;
		    }
		double h1,h0,minh0=1e300,maxh1=0;
		int i1,i0;
		int mi1,mi0;
		int first=1;
		for (int j=0;j<vgm_nlayers;j++)
		{
		    if (vgm_h0[j]<minh0)
		    {
			minh0=vgm_h0[j];
			mi0=j;
		    }
		    if (vgm_h1[j]>maxh1)
		    {
			maxh1=vgm_h1[j];
			mi1=j;
		    }
		    for (int k=0;k<vgm_nlayers;k++)
			if (j!=k)
			if (((i*dL)>=vgm_h1[j])&&((i*dL)<vgm_h0[k]))
			{
			    if (first)
			    {
				h1=vgm_h1[j];
				i1=j;
				h0=vgm_h0[k];
				i0=k;
				first=0;
			    }
			    if (vgm_h1[j]>h1)
			    {
				h1=vgm_h1[j];
				i1=j;
			    }
			    if (vgm_h0[k]<h0)
			    {
				h0=vgm_h0[k];
				i0=k;
			    }
			}
		}
		if (first==0)
		{
		    double _k=((i*dL)-h1)/(h0-h1);
		    vgm_s0=_k*vgm_s0s[i0]+(1.0-_k)*vgm_s0s[i1];
		    vgm_s1=_k*vgm_s1s[i0]+(1.0-_k)*vgm_s1s[i1];
		    vgm_a=_k*vgm_as[i0]+(1.0-_k)*vgm_as[i1];
		    vgm_n=_k*vgm_ns[i0]+(1.0-_k)*vgm_ns[i1];
		    k=_k*vgm_k[i0]+(1.0-_k)*vgm_k[i1];
		    av_p=_k*av_ps[i0]+(1.0-_k)*av_ps[i1];
		}
		else
		{
		    if ((i*dL)<=minh0)
		    {
			vgm_s0=vgm_s0s[mi0];
			vgm_s1=vgm_s1s[mi0];
			vgm_a=vgm_as[mi0];
			vgm_n=vgm_ns[mi0];
			k=vgm_k[mi0];
			av_p=av_ps[mi0];
		    }
		    else
		    {
			if ((i*dL)>=maxh1)
			{
			    vgm_s0=vgm_s0s[mi1];
			    vgm_s1=vgm_s1s[mi1];
			    vgm_a=vgm_as[mi1];
			    vgm_n=vgm_ns[mi1];
			    k=vgm_k[mi1];
			    av_p=av_ps[mi1];
			}
		    }
		}
	    }
	}
	double wetness(int i, double u = -1.0, int from_u = 0)
	{
		double P = -U[i];
		double Ch = 0.5;
		if (from_u)
			P = -u;
		/////////// non-physical tests ////////////////////
		if (cv_form == 1) // logarithmic linearized from literature
		{
			if (P >= 10) Ch = 0.05;
			if ((P<10) && (P >= 1)) Ch = 0.125 - (P - 1.0)*(0.125 - 0.05) / 9.0;
			if ((P<1) && (P >= 0.1)) Ch = 0.225 - (P - 0.1)*(0.225 - 0.125) / 0.9;
			if ((P<0.1) && (P >= 0.01)) Ch = 0.42 - (P - 0.01)*(0.42 - 0.225) / 0.09;
			if ((P<0.01) && (P >= 0.0)) Ch = 0.5 - P*(0.5 - 0.42) / 0.01;
		}
		if (cv_form == 2) // linear
		{
			if (P >= 10) Ch = 0.05;
			if ((P<10) && (P >= 0.0)) Ch = 0.5 - P*(0.5 - 0.05) / 10.0;
		}
		if (cv_form == 3) // logarithmic
		{
			if (P>0.001)
			{
				P = log10(P);
				if (P >= 1)
					Ch = 0.05;
				else
					Ch = 0.5 - (P + 3.0)*(0.5 - 0.05) / 4.0;
			}
		}
		/////////////// van Genuchten - Mualem /////////////////////
		if (cv_form == 5)
		{
			vgm_get_coefs(i);
			P *= HtoPmult;
			if (P <= 0.0)
				Ch = vgm_s1;
			else
				Ch = vgm_s0+((vgm_s1 - vgm_s0) / pow(1 + pow(vgm_a*P, vgm_n), (1 - 1 / vgm_n)));
		}
		return Ch;
	}
	double inv_dw_dh(int i)
	{
		double P = -U[i];
		double Ch = 0.0;
		/////////// non-physical tests ////////////////////
		if (cv_form == 1) // logarithmic linearized from literature - in experiments wetness was equal to dW/dH
		{
			if (P >= 10) Ch = 0.05;
			if ((P<10) && (P >= 1)) Ch = 0.125 - (P - 1.0)*(0.125 - 0.05) / 9.0;
			if ((P<1) && (P >= 0.1)) Ch = 0.225 - (P - 0.1)*(0.225 - 0.125) / 0.9;
			if ((P<0.1) && (P >= 0.01)) Ch = 0.42 - (P - 0.01)*(0.42 - 0.225) / 0.09;
			if ((P<0.01) && (P >= 0.0)) Ch = 0.5 - P*(0.5 - 0.42) / 0.01;
		}
		if (cv_form == 2) // linear - in experiments wetness was equal to dW/dH
		{
			if (P >= 10) Ch = 0.05;
			if ((P<10) && (P >= 0.0)) Ch = 0.5 - P*(0.5 - 0.05) / 10.0;
		}
		if (cv_form == 3) // logarithmic - in experiments wetness was equal to dW/dH
		{
			if (P>0.001)
			{
				P = log10(P);
				if (P >= 1)
					Ch = 0.05;
				else
					Ch = 0.5 - (P + 3.0)*(0.5 - 0.05) / 4.0;
			}
		}
		/////////////// van Genuchten - Mualem /////////////////////
		if (cv_form==5)
		{
			vgm_get_coefs(i);
			if (P <= 0.0)
				Ch = 0.0;
			else			    
			{
				double k_=HtoPmult;
				double hn = pow(k_, vgm_n), h2n = pow(k_, 2.0*vgm_n);
				double aPn=pow(vgm_a*P, vgm_n);
				Ch = ((((1 - vgm_n)*hn*vgm_s0 + (vgm_n - 1)*hn*vgm_s1)*aPn*pow(1 + hn*aPn, (1 / vgm_n))) / (h2n*P*aPn*aPn + 2 *hn* P*aPn + P));
			}
		}
		if (Ch!=0.0)
			return 1.0/Ch;
		return 1e20;
	}
	double avg_root_layer_wetness()
	{
		double sum = 0.0;
		int n = 0;
		for (int i = 0;i<N + 1;i++)
			if (i*dL <= v)
			{
			    sum += wetness(i);
    			    n++;
			}
			else
				break;
		if (n) sum /= n;
		return sum;
	}
	double total_water_content()
	{
		double sum = 0.0;
		for (int i = 0;i < N + 1;i++)
			sum += wetness(i)*dL;
		return sum*p_soil/p_water;
	}
	// non-linear filtration coefficient 
	double KK(int i)
	{
		double Kr=1.0;
		vgm_get_coefs(i);
		// by VGM
		if (k_form==1)
			Kr = pow(((wetness(i) - vgm_s0) / (vgm_s1 - vgm_s0)), 0.5)*pow(1.0 - pow(1.0 - pow(((wetness(i) - vgm_s0) / (vgm_s1 - vgm_s0)), (1.0 / (1.0 - 1.0 / vgm_n))), (1.0 - 1.0 / vgm_n)), 2.0);
		// by Averianov
		if (k_form==2)
			Kr = pow(((wetness(i) - vgm_s0) / (vgm_s1 - vgm_s0)), av_p);
		return Kr*k;
	}
	double _I(double &irr_am)
	{
	    double I=0;
		if (no_irr==0)
		{
			if (nirr != 0) // fixed irrigation
			{
				// calc linear combination
				int i;
				double t = tau*tstep;
				for (i = 0;i<nirr;i++)
					if (irr_T[i]>t)
						break;
				if ((i == 0) || (i == nirr))
				{
					if (i == nirr) i--;
					I = irr_A[i];
				}
				else
				{
					double k = (t - irr_T[i - 1]) / (irr_T[i] - irr_T[i - 1]);
					I = k*irr_A[i] + (1 - k)*irr_A[i - 1];
				}
			}
			else
			{
				double aw = avg_root_layer_wetness();
				if (irr_start_step != -1) // irrigation is being applied
				{
					I = irr_volume / irr_time;
					if ((((tstep - irr_start_step)*tau) > irr_time) ||
						(aw >= max_wetness))
					if (irr_am!=0)
					{
						if (fit!=1)
							printf("irrigation amount - %g mm %g %g\n", irr_am*1000.0, irr_start_step*tau, tstep*tau);
						irr_am = 0.0;
						I = 0.0;
						irr_start_step = -1; // stop irrigation	
						last_irr_step=tstep; // save last irrigation step
					}
				}
				else				    
				{
				    if (tstep*tau>irr_delay) // check for initial delay
					if ((tstep-last_irr_step)*tau>=inter_irr_delay) // check for the delay between irrigations
					    if (aw < min_wetness) // start irrigation
					    	irr_start_step = tstep;
				}
			}
		}
		return I;
	}
	// sink term (right part) - roots water uptake
	double Rp(int i)
	{
		double ret=0.0;
		if ((i*dL)<=v)
		{
			if (rp_form==1)
			// beans
				ret=(1.44-0.14*(i*dL/v)-0.61*(i*dL*i*dL/(v*v))+0.69*(i*dL*i*dL*i*dL/(v*v*v)))*Tr*day_night_coef();
			if (rp_form==2)
				ret=Tr*day_night_coef();
		}
		return ret*inv_dw_dh(i);
	}
	// upper boundary condition (DbU=Uc)
	double Uc()
	{
		static double irr_am = 0.0;
		if (tstep == 1) irr_am = 0.0;
		double E=Ev*day_night_coef(); // evaporation
		double I=0.0; // irrigation
		double P=0.0; // precipitation
		if (rp_form == 3)
			E += Tr*day_night_coef();
		I=_I(irr_am);
		if (nperc != 0) // percipitation
		{
			// calc linear conbination
			int i;
			double t = tau*tstep;
			double p = 0.0;
			for (i = 0;i<nperc;i++)
				if (perc_T[i]>t)
					break;
			if ((i == 0) || (i == nperc))
			{
				if (i == nperc) i--;
				p= perc_A[i];
			}
			else
			{
				double k = (t - perc_T[i - 1]) / (perc_T[i] - perc_T[i - 1]);
				p= k*perc_A[i] + (1 - k)*perc_A[i - 1];
			}
			P+=p*perc_k;
		}
		irr_am += I*tau;
		I*=perc_k;
		I+=P;
		double kk=KK(0);
		int b=0;
		double v=Gamma(2.0-gamma)*dL*(E-I)/kk;
		if (flags&64)
		{
		    if (kk<bounds[13][0]) {v=0.0;b=1;}
		    if (v>bounds[12][0]) {v=bounds[12][0];b=2;}
		    if (v<bounds[12][1]) {v=bounds[12][1];b=3;}
		}
		return v;
	}	
	void set_EV_TR()
	{
		double t=tau*tstep;
		double v=0.0,k;
		double l=0.0;
		int i=0;
		if (klin!=0.0)
		    this->k=this->saved_k+klin*t;
		// ev_t have to be sorted by T
		    // calc linear conbination of EV
		    for (i=0;i<nEV_T;i++)
			if (EV_T[i]>t)
				break;
		    if ((i==0)||(i==nEV_T))
		    {
			if (i == nEV_T) i--;
			for (int j=0;j<nEV_F;j++)
				v+=EV_F[j][i]*EV_C[j];
			l=LAI[i];
		    }
		    else
		    {
			k=(t-EV_T[i-1])/(EV_T[i]-EV_T[i-1]);
			for (int j=0;j<nEV_F;j++)
				v+=k*EV_F[j][i]*EV_C[j]+(1-k)*EV_F[j][i-1]*EV_C[j];
			l=k*LAI[i]+(1-k)*LAI[i-1];
		    }
		// get root length
		if (nrl)
		{
		    for (i=0;i<nrl;i++)
			if (rlT[i]>t)
				break;
		    if ((i==0)||(i==nrl))
		    {
			if (i == nrl) i--;
			this->v=rlV[i];
		    }
		    else
		    {
			k=(t-rlT[i-1])/(rlT[i]-rlT[i-1]);
			this->v=k*rlV[i]+(1-k)*rlV[i-1];
		    }
		}
		// divide EV into Tr and E
		double M=1.0-exp(-ev_mu*l);
		Tr=M*v;
		Ev=(1-M)*v;
	}
	// AHi-1 - RHi + BHi+1 = Psi
	double A_(int i)
	{
		return Dg*KK(i-1)*inv_dw_dh(i)*kb(gamma, i - 1, i, i,dL)*((gamma!=1)?(1-gamma):1);
	}
	double B(int i)
	{
		return Dg*KK(i+1)*inv_dw_dh(i)*kb(gamma, i - 1, i, i,dL)*((gamma!=1)?(1-gamma):1);
	}
	double R(int i)
	{
		return inv_dw_dh(i)*(KK(i+1)+ KK(i-1))*Dg*kb(gamma, i - 1, i, i,dL)*((gamma!=1)?(1-gamma):1)+Da*kb(alpha, tstep, tstep+1, tstep+1,tau,1)*((alpha!=1)?(1-alpha):1);
	}
	void reset_D(double tau_m)
	{
	    tau=tau_m;
	    Da=pow(tau,-1)/Gamma(2.0-alpha); // Caputo derivative for T
	    Dg=pow(dL,-2)/Gamma(2.0-gamma); // psi-Caputo derivative for x
	}
	H_solver(int rpf,int cvf,int irr,int bc,double tau_m,double a,double b,FILE *log,int kpf,double _L=3.0,double _v=0.5,double Hb=1e300)
	{
		perc_k=1.0;
		fixed_Z=fixed_U=NULL;
		vgm_ns=vgm_s0s=vgm_s1s=vgm_as=vgm_h0=vgm_h1=vgm_k=av_ps=NULL;
		vgm_nlayers=0;
		nfixed=0;
		nrl=0;
		rlT=rlV=NULL;
		Hbottom=Hb;
		U=b_U=new double[N+2];

		vgm_n=1.25;
		vgm_s0=0.0736;
		vgm_s1=0.55;
		vgm_a=0.066;
		av_p=averianov_power;

		isEV=0;
		nirr = 0;
		nperc = 0;

		Al=new double[sN+2];
		Bt=new double[sN+ 2];
		Om=new double[sN+ 2];

		L = _L; // total depth
		H0 = -800.0 / 1000.0; // initial H	
		v = _v; // root layer depth
		Tr = 3.0*0.001 / (24 * 3600); // transp. 3.0 mm per day
		Ev = 4.0*0.001 / (24 * 3600); // evapor. 4.0 mm per day
		
		saved_k=k = 0.0002 / 100.0; // saturated filtration coefficient
		alpha=a;
		gamma=b;

		bottom_cond=bc; // 0 -dU/dn=0, 1 - U=H0
		k_form=kpf;
		rp_form=rpf;
		cv_form=cvf;
		no_irr=irr;

		min_wetness=0.385 ; // minimal wetness
		max_wetness=0.54; // needed wetness
		p_soil = 2540;
		p_water=1000;

		irr_volume = (max_wetness - min_wetness)*p_soil*v / p_water; // in m
		irr_time = 3600 * 24; // time range to apply irrigation
		irr_start_step = -1;
		last_irr_step=0;

		dL = L/N;
		sL = L;
		sdL = dL;
		tau = tau_m;
		tstep=1;

		reset_D(tau);
		// initial conditions
		for (int i = 0;i < N + 1;i++)
			if (Hbottom==1e300)
    			    U[i] = H0-i*dL;
    			else
    			    U[i]=H0-(i/(float)N)*(H0-Hbottom);
    		// precalc integrals
		if (log)
		{
			fprintf(log,"H_solver(right part %d, C(H) %d,no_irr %d,boundary cond %d,tau %g,a %g,b %g)\n",rpf,cvf,irr,bc,tau_m,a,b);
			printf("H_solver(right part %d, C(H) %d,no_irr %d,boundary cond %d,tau %g,a %g,b %g)\n",rpf,cvf,irr,bc,tau_m,a,b);
			fflush(stdout);
		}
	}
	// set working row and its saved values vector
	void get_F_oldF(double **U_,std::vector<double*> **old)
	{
		U_[0]=U;
		old[0]=&oldH;
	}
	// alpha coefficients
	void al1()
	{
		if ((inv_dw_dh(0)!=1e20)||(flags&1)) // first order condition in saturated zone
			Al[1] = 1;
		else
			Al[1] = 0;
		if (fixed_Z==NULL)
    		    for (int i = 1;i < N;i++)		
			Al[i + 1] = B(i) / (R(i) - A_(i)*Al[i]);
		else // fixed values inside the domain - set Al=0, Bt=v
		{
        		for (int i = 1;i < N;i++)		
        		{
        		    int found=0;
        		    for (int j=0;j<nfixed-1;j++)
        			if ((fixed_Z[j+1]>=i*sdL)&&(fixed_Z[j]<i*sdL))
        			{
        			    found =1;
        			    break;
        			}
			    if (nfixed==1)
        			if ((fixed_Z[0]>=i*sdL)&&(fixed_Z[0]<(i+1)*sdL))
        			    found =1;
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
		double *U_;
		std::vector<double*> *old;
		A=1-gamma;
		get_F_oldF(&U_,&old);
		// div (k Db_z F)
		for (int i = 1;i < N;i++)
			F[i] = (U_[i + 1] - U_[i]);
		F[0]= (U_[1] - U_[0]);
		double prevB = 0;
		for (int i = 0;i < NB;i++)
		{
			if (i != 0)
				prevB = BVK[BS - 1];
			calc_v(i, F, BVK,dL,0,1,((tstep==1)?1:0));
			for (int j = 0;j < BS;j++)
				Om[1 + i*BS + j] = -Dg*inv_dw_dh(1 + i*BS + j)*(KK(1 + i*BS + j)*BVK[j]- KK(i*BS + j)*((j==0)?prevB:BVK[j-1])) - Da*kb(alpha, tstep, tstep+1, tstep+1,tau,1)*((alpha!=1)?(1-alpha):1)*U_[1 + i*BS + j] + Rp(1 + i*BS + j);
			for (int j = 0;j < BS;j++)
				Om[1 + i*BS + j] += (KK(1 + i*BS + j)- KK(i*BS + j))/dL;
		}		
		// Da_t F part
		if (alpha!=1.0)
		{
#pragma omp parallel for
		for (int i = 0;i < NB;i++)
			for (int j = 0;j < BS;j++)
			{
				double time_sum = 0;
				if (old->size()>=2)
				{
					int os=old->size();
					double **oldp=&old[0][1];
					double prev;
					int id;
					id=1+i*BS+j;
					prev=oldp[0][id];
					for (int t = 0;t < os- 1;t++,oldp++)
					{
						double kv;
						double cur=oldp[0][id];
						kv = kb(alpha, t, t+1, tstep+1,tau,1)*(1-alpha);
						time_sum += kv*(cur - prev);
						prev=cur;
						
					}
				}
				Om[1 + i*BS + j] += Da*time_sum;
			}
		}
	}
	// beta coeffients
	void bt1()
	{
		if ((inv_dw_dh(0)!=1e20)||(flags&1)) // first order condition if saturated in 0
			Bt[1] = -Uc();
		else
			Bt[1] = -Uc()*KK(1)/dL; // E-I
		if (fixed_Z==NULL)
		{
		    for (int i = 1;i < N;i++)
			Bt[i + 1] = (Al[i + 1]/B(i)) * (A_(i)*Bt[i] - Om[i]);
		}
		else // fixed values inside the domain - set Al=0, Bt=v
		{
    		    double v=0;
    		    double *fixed_F=fixed_U;
        		for (int i = 1;i < N;i++)		
        		{
        		    int found=0;
        		    for (int j=0;j<nfixed-1;j++)
        			if ((fixed_Z[j+1]>=sdL*i)&&(fixed_Z[j]<sdL*i))
        			{
        			    found =1;
        			    if (fixed_Z[j+1]!=fixed_Z[j])
        				v=fixed_F[j]+((sdL*i-fixed_Z[j])/(fixed_Z[j+1]-fixed_Z[j]))*(fixed_F[j+1]-fixed_F[j]);
        			    break;
        			}
			    if (nfixed==1)
        			if ((fixed_Z[0]>=sdL*i)&&(fixed_Z[0]<sdL*(i+1)))
        			{
        			    found =1;
				    v=fixed_F[0];
				}
        		    if (found==0)
				Bt[i + 1] = (Al[i + 1]/B(i)) * (A_(i)*Bt[i] - Om[i]);
			    else 
				Bt[i+1]=v;
			}
		}
	}
	// calc F
	void U1()
	{
		double *U_;
		std::vector<double*> *old;
		get_F_oldF(&U_,&old);
		al1();
		Om1();
		bt1();
			double Qb = 0;
			if (bottom_cond == 0)
			{
				for (int i = 0;i < N;i++)
					F[i] = (U_[i + 1] - U_[i])/dL;
				for (int i = 0;i < NB;i++)
					calc_v(NB-1, F, BVK,dL,0,0, ((tstep == 1) ? 1 : 0));
				Qb = -BVK[BS - 1];
			}
				if (bottom_cond == 1)
				{
					if (Hbottom!=1e300)
					U_[N]=Hbottom;
					else
					U_[N] = H0 - N*dL;
				}
				else
					if (fabs(Bt[N]+Qb) > 1e-15)
					{
						if (fabs(1.0 - Al[N]) > 1e-15)
							U_[N] = (Bt[N]+Qb) / (1.0 - Al[N]);
						else
							U_[N] = U_[N - 1]+Qb;
					}
					else
					{
						if (fabs(1.0 - Al[N]) > 1e-15)
							U_[N] = 0.0;
						else
							U_[N] = U_[N - 1]+Qb;
					}				
		for (int i = N - 1;i >= 0;i--)
			U_[i] = Al[i + 1] * U_[i + 1] + Bt[i + 1];
		U_[N + 1] = U[N];
		// save H for time-fractional derivative calculations
		if (alpha!=1.0)
		{
			double *hs = new double[N + 2];
			memcpy(hs, b_U, (N + 2)*sizeof(double));
			old->push_back(hs);
		}
	}
	void calc_step()
	{
		// sigmoid function
		if (func_in_kernel==4)
		{
			// space
			double fp=func_power;
			if (func_power<0) fp=-fp;
			sigmoid_d=1.0/(1.0+exp(fp*sigmoid2));
			sigmoid_a=(1.0/(1.0+exp(-fp*(2-sigmoid2))))-sigmoid_d;
			sigmoid_b=2;
			sigmoid_c=sigmoid2;							  
			sigmoid_inv_a=0.5/fp;
			sigmoid_inv_d=0.5*sigmoid2;
			sigmoid_inv_c=sigmoid_d;
			sigmoid_inv_b=sigmoid_a;
			if (func_power<=0.0)
			{
				double s;
				s=sigmoid_inv_a;sigmoid_inv_a=sigmoid_a;sigmoid_a=s;
				s=sigmoid_inv_b;sigmoid_inv_b=sigmoid_b;sigmoid_b=s;
				s=sigmoid_inv_c;sigmoid_inv_c=sigmoid_c;sigmoid_c=s;
				s=sigmoid_inv_d;sigmoid_inv_d=sigmoid_d;sigmoid_d=s;
			}
		}
		///// precalc integrals
		if (kb_space==NULL)
		{
		    kb_space=new double[(sN+2)*(sN+2)];
#pragma omp parallel for		
		    for (int aa=1;aa<sN+1;aa++)
			for (int tt0=0;tt0<aa;tt0++)
			{
			    kb_space[aa*(sN+2)+tt0]=calc_int(gamma,tt0,tt0+1,aa,dL,0);
				//if ((aa==20)||(aa==40))
				//	printf("%d %d %g\n",aa,tt0,kb_space[aa*(sN+2)+tt0]);
			}
		}
		double sfp;
		int f4=0;
		if (func_in_kernel==4) {func_in_kernel=3;f4=1;}
		if (func_in_kernel==3)
		{
		    sfp=func_power;
		    func_power=func_power_t;
		}
		if (kb_time) delete [] kb_time;
		kb_time=new double[tstep+1];
#pragma omp parallel for
		for (int aa=0;aa<=tstep;aa++)
		{
		    kb_time[aa]=calc_int(alpha,aa,aa+1,tstep+1,tau,1);
			//if ((tstep==100)||(tstep==200))
			//	printf("%d %d %g\n",tstep,aa,kb_time[aa]);
		}
		if (func_in_kernel==3) func_power=sfp;		
		if (f4==1) func_in_kernel=4;
		//////
		if (isEV)
		    set_EV_TR();
		U1();
		tstep++;
	}
	~H_solver()
	{
		for (int i = 0;i < oldH.size();i++)
			delete oldH[i];
		delete [] b_U;
		delete [] Al;
		delete [] Bt;
		delete [] Om;
	}
};
// fit parameters (by particle swarm) and solve the problem
// values_{T,Z,F} - values to match
// check_{T,Z,F} - values for checking
// init_{Z,F} - initial values
// EV_T,EV_F[] - evapotranspiration values (EV_T must be sorted)
// nEV_F - number of different EV estimations
// to_fit:
//		bit 0 - alpha
//		bit 1 - gamma
//		bit 2 - L
// 		bit 3 - VGM parameters
// 		bit 4 - different EV combination weights
//	    bit 5 - filtration k
//	    bit 6 - percipitation k 
//		sigmoid for space
//		bit 7 - func power in g(x)=x^fp or sigmoid
//		bit 8 - second parameter of sigmoid
//		sigmoid for time
//		bit 9 - func power in g(x)=x^fp or sigmoid
//		bit 10 - second parameter of sigmoid
// pso_{n,o,fi_p,fi_g) - classic particle swarm algorithms' parameters
int adaptive_pso=0; // adaptive pso: o - omega change parameter p , fi_p - fi_p,fi_g changing parameter C, fi_g - vmax calculating parameter N
double restart_prox=0.0; // probability of particles "restart" - random initialization
int use_steepest_descent=0; // 1 - do steepest descent at the end of PSO on the best particle, 2 - do on all particles / work only with adaptive PSO
int st_desc_max_iter=100;
int st_desc_eps=1e-16;
int init_print=1;
double *vgm_ns,*vgm_s0s,*vgm_s1s,*vgm_as,*vgm_h0,*vgm_h1,*vgm_k,*av_ps; // VGM coefs for layered soil
int vgm_nlayers=0;
int nrl=0; // variable root length 
double *rlT,*rlV;
int rpf=1;
double rld=0.5;
double Hbottom=1e300;
char *filename=NULL;
int kpf=2;
double metric=0; // 0 - sum sq, 1 - max rel err, 2 - sum rel err
H_solver *init_solver(double *p,int to_fit,
	double *init_Z,double *init_F,int ninit,
	double *perc_T, double *perc_A, int nperc,
	double *irr_T, double *irr_A, int nirr,
	double *EV_T,double **EV_F,double *LAI,double ev_mu,int nEV_T,int nEV_F,
	int irr,int bc,double tau_m,double *param_values,int nparam_values)
{
	double a=1.0;
	double g=1.0;
	double L=3.0;
	int offs=1;
	int pv_offs=0;
	if (to_fit&1) {if (p[offs]>bounds[0][1]) p[offs]=bounds[0][1]; if (p[offs]<bounds[0][0]) p[offs]=bounds[0][0]; a=p[offs++];} else if (pv_offs<nparam_values) a=param_values[pv_offs++];
	if (to_fit&2) {if (p[offs]>bounds[1][1]) p[offs]=bounds[1][1]; if (p[offs]<bounds[1][0]) p[offs]=bounds[1][0]; g=p[offs++];} else if (pv_offs<nparam_values) g=param_values[pv_offs++];
	if (to_fit&4) {if (p[offs]>bounds[2][1]) p[offs]=bounds[2][1]; if (p[offs]<bounds[2][0]) p[offs]=bounds[2][0]; L=p[offs++];} else if (pv_offs<nparam_values) L=param_values[pv_offs++];
	H_solver *ret=new H_solver(rpf,5,irr,bc,tau_m,a,g,NULL,kpf,L,rld,Hbottom);
	if (to_fit&8)
	{
		ret->vgm_n=p[offs++];
		ret->vgm_s0=p[offs++];
		ret->vgm_s1=p[offs++];
		ret->vgm_a=p[offs++];
	}
	else
	{
	    if (pv_offs<nparam_values) ret->vgm_n=param_values[pv_offs++];
	    if (pv_offs<nparam_values) ret->vgm_s0=param_values[pv_offs++];
	    if (pv_offs<nparam_values) ret->vgm_s1=param_values[pv_offs++];
	    if (pv_offs<nparam_values) ret->vgm_a=param_values[pv_offs++];
	}
	if (vgm_nlayers!=0)
	{
	    ret->vgm_s0s=vgm_s0s;
	    ret->vgm_s1s=vgm_s1s;
	    ret->vgm_as=vgm_as;
	    ret->vgm_ns=vgm_ns;
	    ret->vgm_h0=vgm_h0;
	    ret->vgm_h1=vgm_h1;
	    ret->vgm_nlayers=vgm_nlayers;
	    ret->vgm_k=vgm_k;
		ret->av_ps=av_ps;
	    if (init_print==1)
	    {
		printf("%d layered soil\n",vgm_nlayers);
		for (int i=0;i<vgm_nlayers;i++)
		    printf("%g %g %g %g %g %g %g %g\n",vgm_s0s[i],vgm_s1s[i],vgm_as[i],vgm_ns[i],vgm_h0[i],vgm_h1[i],vgm_k[i],av_ps[i]);
	    }
	}
	ret->isEV=1;
	ret->EV_T=EV_T;
	ret->EV_F=EV_F;
	ret->nEV_T=nEV_T;
	ret->nEV_F=nEV_F;
	ret->LAI=LAI;
	ret->ev_mu=ev_mu;
	ret->EV_C=new double[nEV_F];
	ret->nrl=nrl;
	ret->rlT=rlT;
	ret->rlV=rlV;
	if (to_fit&16)
    	    for (int i=0;i<nEV_F;i++)
		ret->EV_C[i]=p[offs++];
	else
	    for (int i=0;i<nEV_F;i++)
		if (pv_offs<nparam_values) ret->EV_C[i]=param_values[pv_offs++];
	if (to_fit & 32)
	{
	    if (vgm_nlayers!=0)
		for (int i=0;i<vgm_nlayers;i++)
		    vgm_k[i]=p[offs++]/Hmult;
	    else
		ret->saved_k=ret->k = p[offs++]/Hmult;
	}
	else
	    if (pv_offs<nparam_values) ret->saved_k=ret->k=param_values[pv_offs++]/Hmult;
	if (irr==1)
	{
	    if (to_fit & 64)
		ret->perc_k = p[offs++];
	    else
		if (pv_offs<nparam_values) ret->perc_k=param_values[pv_offs++];
	}
	if (irr==0)
	{
	    ret->perc_k=p[offs++];
	    ret->irr_time=86400;
	    if (to_fit & 64)
	    {
		ret->min_wetness=p[offs++];
		ret->max_wetness=p[offs++];
		ret->irr_volume=p[offs++];
	    }
	    else
	    {
		if (pv_offs<nparam_values) ret->min_wetness=param_values[pv_offs++];
		if (pv_offs<nparam_values) ret->max_wetness=param_values[pv_offs++];
		if (pv_offs<nparam_values) ret->irr_volume=param_values[pv_offs++];
	    }
	    if (ret->min_wetness>ret->max_wetness)
	    {
	        double s=ret->min_wetness;
	        ret->min_wetness=ret->max_wetness;
	        ret->max_wetness=s;
	    }
	}
	if (to_fit&128)  {if (p[offs]>bounds[14][1]) p[offs]=bounds[14][1]; if (p[offs]<bounds[14][0]) p[offs]=bounds[14][0];func_power=p[offs++];} else if (pv_offs<nparam_values) func_power=param_values[pv_offs++];
	if (to_fit&256)  {if (p[offs]>bounds[15][1]) p[offs]=bounds[15][1]; if (p[offs]<bounds[15][0]) p[offs]=bounds[15][0];sigmoid2=p[offs++];} else if (pv_offs<nparam_values) sigmoid2=param_values[pv_offs++];
	if (to_fit&512)  {if (p[offs]>bounds[16][1]) p[offs]=bounds[16][1]; if (p[offs]<bounds[16][0]) p[offs]=bounds[16][0];func_power_t=p[offs++];} else if (pv_offs<nparam_values) func_power_t=param_values[pv_offs++];
	if (to_fit&1024) {if (p[offs]>bounds[17][1]) p[offs]=bounds[17][1]; if (p[offs]<bounds[17][0]) p[offs]=bounds[17][0]; sigmoid2_t=p[offs++];} else if (pv_offs<nparam_values) sigmoid2_t=param_values[pv_offs++];
	// set initial values
	if (init_print==1)
	{
	    printf("%g %g(%d %g %g %g %g) %g(%g) %g %g %g %g ",a,g,func_in_kernel,func_power,sigmoid2,func_power_t,sigmoid2_t,L,ret->Hbottom,ret->vgm_n,ret->vgm_s0,ret->vgm_s1,ret->vgm_a);
	    for (int i=0;i<nEV_F;i++)
		printf("%g ",ret->EV_C[i]);
	    printf("%g(%g) %g",ret->k,klin,ret->perc_k);
	    if (irr==0)
		printf(" %g %g %g",ret->min_wetness,ret->max_wetness,ret->irr_volume);
	    printf(" %d %g",rpf,ret->v);
	    printf("\n");
	    init_print=0;
	}
	if ((ninit>0)&&(flags&32)) // steady state solution with fixed points for initial values
	{
	  int NN=N;
	  int sni = ret->no_irr;
	  double *su=new double[NN];
	  double diff=0.0;
	  int niter=0;
	  // find steady state for given fixed values of F
	  ret->fixed_Z=init_Z;
	  ret->fixed_U=init_F;
	  ret->nfixed=ninit;
	  ret->Tr=ret->Ev=0;
	  ret->alpha=1.0;
	  ret->reset_D(ret->tau/stat_div);
	  ret->no_irr = 2;
	  do
	  {
	    memcpy(su,ret->b_U,NN*sizeof(double));
	    ret->calc_step();
	    diff=0.0;
	    for (int i=0;i<NN;i++)
		diff+=(ret->b_U[i]-su[i])*(ret->b_U[i]-su[i]);
	    if (diff<stat_eps)
		break;
	  }
	  while ((niter++)<stat_maxiter);  
	  ret->fixed_Z=ret->fixed_U=NULL;
	  ret->nfixed=0;
	  ret->alpha=a;
	  ret->no_irr = sni;
	  ret->tstep=1;
	  ret->reset_D(ret->tau*stat_div);
	}
	if ((ninit>0)&&(!(flags&32))) // linearly interpolated initial values
	for (int i = 0;i < N + 1;i++)
	{
		double v=init_F[0];
		int j;
		if ((i*ret->dL)>=init_Z[0])
		{
			for (j = 1;j < ninit;j++)
				if (((i*ret->dL) >= init_Z[j - 1]) && ((i*ret->dL) < init_Z[j]))
				{
					double k = (i*ret->dL - init_Z[j - 1]) / (init_Z[j] - init_Z[j - 1]);
					v = (1.0 - k)*init_F[j - 1] + k*init_F[j];
					break;
				}
			if (j == ninit)
			{
				v = init_F[ninit - 1];
				if (ninit > 2)
				{
					double k = (init_F[ninit-1] - init_F[ninit - 2]) / (init_Z[ninit - 1] - init_Z[ninit - 2]);
					v += ((i*ret->dL)-init_Z[ninit-1])*k;
				}

			}
		}
		else
		{
			if (ninit > 2)
			{
				double k = (init_F[1] - init_F[0]) / (init_Z[1] - init_Z[0]);
				v -= (init_Z[0] - (i*ret->dL))*k;
			}
		}
		ret->U[i] = v;
	}
	// set percipitation and irrigation
	ret->nirr = nirr;
	ret->nperc = nperc;
	ret->irr_A = irr_A;
	ret->irr_T = irr_T;
	ret->perc_A = perc_A;
	ret->perc_T = perc_T;
	return ret;
}
void clear_solver(H_solver *ss,int to_fit)
{
	if (to_fit&16)
		delete [] ss->EV_C;
	delete ss;
}
double solve_and_test(double *p,int to_fit,
	double *init_Z,double *init_F,int ninit,
	double *perc_T, double *perc_A, int nperc,
	double *irr_T, double *irr_A, int nirr,
	double *EV_T,double **EV_F,double *LAI,double ev_mu,int nEV_T,int nEV_F,
	int irr,int bc,double tau_m,FILE *fi,
	double *values_T,double *values_Z,double *values_F,int nvalues,double *param_values,int nparam_values,double t)
{
	double err=0.0;	
	double *vs=new double [nvalues];
	H_solver *ss=init_solver(p,to_fit,init_Z,init_F,ninit, perc_T, perc_A, nperc, irr_T, irr_A, nirr, EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,param_values,nparam_values);
	for (double tt = 0;tt <= t;tt += ss->tau)
	{
		// save old values
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
			{
				for (int j=0;j<N-1;j++)
					if (((ss->dL*j)<values_Z[i])&&((ss->dL*(j+1))>=values_Z[i]))
					{
						double k2=(values_Z[i]-(ss->dL*j))/ss->dL;
						vs[i]=(1-k2)*ss->U[j]+k2*ss->U[j+1];
						break;
					}
			}
		// solve
		ss->calc_step();
		// add to err
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
			{
				double k1=(values_T[i]-tt)/ss->tau;
				    for (int j=0;j<N;j++)
					if (((ss->dL*j)<values_Z[i])&&((ss->dL*(j+1))>=values_Z[i]))
					{
						double k2=(values_Z[i]-(ss->dL*j))/ss->dL;
						double v=(1-k2)*ss->U[j]+k2*ss->U[j+1];
						v=(1-k1)*vs[i]+k1*v;
						if (v>0.0)
						    v=0.0;
						if (metric==0)
							err+=(values_F[i]-v)*(values_F[i]-v);
						if ((metric==1)||(metric==2))
							if (values_F[i]!=0)
							{
								double rel=fabs((values_F[i]-v)/values_F[i]);
								if (rel>err)
								{
									if (metric==1)
										err=rel;
									else
										err+=rel;
								}
							}
							
					}
			}
		if (!finite(ss->b_U[0]))
		{
		     err=1e300;
		     break;
		}
	}
	clear_solver(ss,to_fit);
	delete [] vs;
	if (!finite(err)) err = 1e300;
	return err;
}
double rnd(int i,int idx,int size)
{
    //if (idx>(2<<size))
	return ((rand() % 10000) / (10000.0 - 1.0));
    //return idx&(1<<i);
}
void init_particle(double *particle, int to_fit,int nEV_F,int idx,int size)
{
	int s = 0;	
	// a,b from 0.9 to 1.0
	if (to_fit & 1) particle[1+s++] = bounds[0][0] + (bounds[0][1]-bounds[0][0])*rnd(s,idx,size);
	if (to_fit & 2)	particle[1+s++] = bounds[1][0] + (bounds[1][1]-bounds[1][0])*rnd(s,idx,size);
	if (to_fit & 4)	particle[1+s++] = bounds[2][0] + (bounds[2][1]-bounds[2][0])*rnd(s,idx,size);
	// vgm paramters
	if (to_fit & 8)
	{
	    do
	    {
		particle[1+s] = bounds[3][0] + (bounds[3][1]-bounds[3][0])*rnd(s,idx,size);
		particle[1 + s+1] = bounds[4][0] + (bounds[4][1]-bounds[4][0])*rnd(s,idx,size);
		particle[1 + s+2] = bounds[5][0] + (bounds[5][1]-bounds[5][0])*rnd(s,idx,size);
		particle[1 + s+3] = bounds[6][0] + (bounds[6][1]-bounds[6][0])*rnd(s,idx,size);
	    }
	    while (particle[1 + s+1]>particle[1 + s+2]);
	    s+=4;
	}
	// EVT coeffs from 0 to 1
	if (to_fit & 16)
	{
	    double sum=0;
	    int ss=s;
	    do
	    {
		s=ss;
		sum=0;
		for (int i = 0;i < nEV_F;i++)
		{
			particle[1 + s++] = bounds[11][0]*rnd(s,idx,size);
			sum+=particle[s];			
		}
		if (sum<=bounds[11][0])
		    break;
	    }
	    while (1);
	}
	// filtration k
	if (to_fit & 32)
	{
	    if (vgm_nlayers!=0)
		for (int i=0;i<vgm_nlayers;i++)
		    particle[1 + s++] = bounds[7][0] + (bounds[7][1]-bounds[7][0])*rnd(s,idx,size);
	    else
		particle[1 + s++] = bounds[7][0] + (bounds[7][1]-bounds[7][0])*rnd(s,idx,size);
	}
	// percipitation k
	if (to_fit & 64)
	{
	    particle[1 + s++] = bounds[11][1]*rnd(s,idx,size);
	    if (irr==0)
	    {
		particle[1 + s++] = bounds[8][0] + (bounds[8][1]-bounds[8][0])*rnd(s,idx,size);
		particle[1 + s++] = bounds[9][0] + (bounds[9][1]-bounds[9][0])*rnd(s,idx,size);
		particle[1 + s++] = bounds[10][0] + (bounds[10][1]-bounds[10][0])*rnd(s,idx,size);
	    }
	}
	if (to_fit & 128) particle[1+s++] = bounds[14][0] + (bounds[14][1]-bounds[14][0])*rnd(s,idx,size);
	if (to_fit & 256) particle[1+s++] = bounds[15][0] + (bounds[15][1]-bounds[15][0])*rnd(s,idx,size);
	if (to_fit & 512) particle[1+s++] = bounds[16][0] + (bounds[16][1]-bounds[16][0])*rnd(s,idx,size);
	if (to_fit & 1024) particle[1+s++] = bounds[17][0] + (bounds[17][1]-bounds[17][0])*rnd(s,idx,size);
	for (int i = 0;i < s;i++)
		particle[i + 1 + 2 * s] = particle[i + 1];
	for (int i = 0;i<s;i++)
		particle[i + 1 + s] = 0.0;
}
double steepest_descent(double *p,int to_fit,int size,double *vmax,double *init_Z,double *init_F,int ninit,
	double *perc_T, double *perc_A, int nperc,
	double *irr_T, double *irr_A, int nirr,
	double *EV_T,double **EV_F,double *LAI,double ev_mu,int nEV_T,int nEV_F,
	int irr,int bc,double tau_m,FILE *fi,
	double *values_T,double *values_Z,double *values_F,int nvalues,double *param_values,int nparam_values,double t,double pso_eps,int pso_max_iter)
{
	double *vs=new double[1+size],vnew,*sp=new double[1+size];
	double step=1.0; // starting step
	double ret=1e300;
	int iter=0;
	vs[0]=solve_and_test(p,to_fit,init_Z,init_F,ninit,perc_T,perc_A,nperc,irr_T,irr_A,nirr,EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t);
	if (vs[0]==1e300) goto r;
	do
	{
	    for (int i=0;i<size;i++)
		{
		    double st=step*((vmax[i]==0)?1:vmax[i]);
			double ssp=p[1+i];
		    p[1+i]+=st;			
		    vs[i+1]=solve_and_test(p,to_fit,init_Z,init_F,ninit,perc_T,perc_A,nperc,irr_T,irr_A,nirr,EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t);
		    vs[i+1]=(vs[i+1]-vs[0])/st; // dF/dpi
		    p[1+i]=ssp;
		}
	    // x+=step*dF/dpi
	    for (int i=0;i<size;i++)
		{
		    sp[1+i]=p[1+i];
		    p[1+i]-=step*vs[i+1];
		}
	    vnew=solve_and_test(p,to_fit,init_Z,init_F,ninit,perc_T,perc_A,nperc,irr_T,irr_A,nirr,EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t);
	    for (int i=0;i<size;i++)
		    printf ("%g %g ",p[1+i],vs[i+1]);
	    printf("%d vnew %g v0 %g step %g\n",iter,vnew,vs[0],step);
	    if (vnew>=vs[0])
	    {
			// move back and decrease step
			for (int i=0;i<size;i++)
				p[1+i]=sp[1+i];
			step/=2.0;
			if (step<st_desc_eps)
				break;
	    }
	    else
	    {
			if (fabs(vnew-vs[0])<st_desc_eps)	
			{
				ret=vnew;
				goto r;
			}
			vs[0]=vnew;
	    }
	    if ((iter++)>st_desc_max_iter)
			break;
	}
	while (1);		
	ret=vs[0];
r:	
	delete [] vs;
	delete [] sp;
	return ret;
}
void fit_and_solve(double t,double save_tau,double out_tau,int irr,int bc,double tau_m,
	double *init_Z, double *init_F, int ninit,
	double *perc_T, double *perc_A, int nperc,
	double *irr_T, double *irr_A, int nirr,
	double *values_T,double *values_Z,double *values_F,int nvalues,
	double *check_T,double *check_Z,double *check_F,int ncheck,
	double *EV_T,double **EV_F,double *LAI,double ev_mu,int nEV_T,int nEV_F,
	int to_fit,
	int pso_n,double pso_o,double pso_fi_p,double pso_fi_g,double pso_eps,int pso_max_iter,
	int fit,double *param_values,int nparam_values,char *et_file,char *pv_file)
{
	double ad_pso_p,ad_pso_C; // adaptive PSO parameters
	double ad_v0,*ad_vmax;
	double *ad_pso_fi_p,*ad_pso_fi_g; // per-variable fi_p, fi_g for adaptive PSO
	int size=0;
	double **particles;
	int best;
	double *best_p;
	int iter=0;
	char fn[2048],rfn[2048],bfn[2048];
	if (et_file)
	{
	    char *e=et_file;
	    while (e[0])
	    {
		if (e[0]=='.')
		    e[0]=0;
		e++;
	    }
	}
	if (pv_file)
	{
	    char *e=pv_file;
	    while (e[0])
	    {
		if (e[0]=='.')
		    e[0]=0;
		e++;
	    }
	}	
	sprintf(fn,"log_%s_%s.txt",(et_file?et_file:""),(pv_file?pv_file:""));
	sprintf(rfn,"res_%s_%s.txt",(et_file?et_file:""),(pv_file?pv_file:""));
	sprintf(bfn,"pv_best_%s_%s.txt",(et_file?et_file:""),(pv_file?pv_file:""));
	if (filename)
	{
	    sprintf(fn,"log_%s.txt",filename);
	    sprintf(rfn,"res_%s.txt",filename);
	    sprintf(bfn,"pv_best_%s.txt",filename);
	}
	FILE *fi1 = fopen(fn, "wt");
	// number of variables to optimize
	if (to_fit&1) size++;
	if (to_fit&2) size++;
	if (to_fit&4) size++;
	if (to_fit&8) size+=4;
	if (to_fit&16) size+=nEV_F;
	if (to_fit & 32)
	{
	    if (vgm_nlayers!=0)
		size+=vgm_nlayers;
	    else	
		size++;
	}
	if (to_fit & 64) { if (irr==1) size++; else size+=4; }
	if (to_fit&128) size++;
	if (to_fit&256) size++;
	if (to_fit&512) size++;
	if (to_fit&1024) size++;
	if (size==0) return;
	best_p=new double[2*size+1];
	if (et_file)
	{
    	    printf("et_file - %s\n",et_file);
    	    fprintf(fi1,"#et_file - %s\n",et_file);
    	}
	if (pv_file)
	{
    	    printf("pv_file - %s\n",pv_file);
    	    fprintf(fi1,"#pv_file - %s\n",pv_file);
    	}
	if (fit==1)
	{
	    for (int i=0;i<18;i++)
	    {
		fprintf(fi1,"#bounds[%d]={%g,%g}\n",i,bounds[i][0],bounds[i][1]);
		printf("bounds[%d]={%g,%g}\n",i,bounds[i][0],bounds[i][1]);
	    }
	    particles=new double *[pso_n];
    	    for (int i=0;i<pso_n;i++)
		particles[i]=new double[3*size+2]; // particles[0] contains f value, particles[1+size].. contain velocities, particles[1+2*size]... contain particle's best known position, particles[1+3*size] contains best known f value
    	    // initialize
    	    int nit=0;
    	    do
    	    {
		init_particle(particles[0], to_fit,nEV_F,0,size);
		particles[0][0]=particles[0][1+3*size]=solve_and_test(particles[0],to_fit,init_Z,init_F,ninit,perc_T,perc_A,nperc,irr_T,irr_A,nirr,EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi1,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t);
		printf(".");fflush(stdout);
		if ((nit++)==100)
		    break;
	    }
	    while ((particles[0][0]>=1e10)&&(!(flags&8)));
	    best=0;
	    fprintf(fi1, "#initial 0 - %g\n", particles[0][0]);
	    printf("initial 0 - %g\n", particles[0][0]);
	    for (int i=1;i<pso_n;i++)
	    {
	        int nit=0;
    	        do
    	        {
		    init_particle(particles[i], to_fit,nEV_F,i,size);
		    particles[i][0]=particles[i][1+3*size]=solve_and_test(particles[i],to_fit,init_Z,init_F,ninit, perc_T, perc_A, nperc, irr_T, irr_A, nirr, EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi1,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t);
		    printf(".");fflush(stdout);
		    if ((nit++)==100)
			break;
	        }
	        while ((particles[i][0]>=1e10)&&(!(flags&8)));
		if (particles[i][0]<particles[best][0])
			best=i;
		fprintf(fi1, "#initial %d - %g\n",i, particles[i][0]);
		printf("initial %d - %g\n",i, particles[i][0]);
	    }
	    // save best known position
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
		// adaptive PSO - calc initial and max velocity
		if (adaptive_pso)
		{
			double *xm=new double [2*size]; // min-max x in initial population
			double *ad_v0v=new double[size];
			ad_v0=0;
			for (int i=0;i<size;i++) ad_v0v[i]=0;
			ad_vmax=new double[size];
			ad_pso_fi_p=new double[size];
			ad_pso_fi_g=new double[size];
			for (int i=0;i<pso_n;i++)
				for (int j=0;j<size;j++)
				{
					ad_v0+=particles[i][j+1]*particles[i][j+1];
					ad_v0v[j]+=particles[i][j+1]*particles[i][j+1];
					if (i==0)
						xm[2*j+0]=xm[2*j+1]=particles[i][j+1];
					else
					{
						if (particles[i][j+1]<xm[2*j+0]) xm[2*j+0]=particles[i][j+1];
						if (particles[i][j+1]>xm[2*j+1]) xm[2*j+1]=particles[i][j+1];
					}
				}
			ad_v0=sqrt(ad_v0);
			ad_v0/=pso_n;
			for (int i=0;i<size;i++)
			{
			    ad_vmax[i]=(xm[2*i+1]-xm[2*i+0])/pso_fi_g;
				if (ad_vmax[i]==0.0) ad_vmax[i]=1.0/pso_fi_g;
			    ad_v0v[i]=sqrt(ad_v0v[i]);
			    ad_v0v[i]/=pso_n;
			}
			delete [] xm;
			ad_pso_p=pso_o; ad_pso_C=pso_fi_p;
			pso_o=1.0;
			for (int i=0;i<size;i++)
			{
			    ad_pso_fi_p[i]=ad_pso_C*ad_v0v[i]/ad_vmax[i];
			    ad_pso_fi_g[i]=ad_pso_C*(1-ad_v0v[i]/ad_vmax[i]);
			    if (ad_pso_fi_p[i]>2.0) ad_pso_fi_p[i]=2.0;
			    if (ad_pso_fi_g[i]>2.0) ad_pso_fi_g[i]=2.0;
    			    if (ad_pso_fi_p[i]<0.0) ad_pso_fi_p[i]=0.0;
			    if (ad_pso_fi_g[i]<0.0) ad_pso_fi_g[i]=0.0;
			}
			delete [] ad_v0v;
			printf("adaptive PSO: ||v0||=%g, initial PSO params values %g",ad_v0,pso_o);
			for (int i=0;i<size;i++)
			    printf(",(%g->%g,%g)",ad_vmax[i],ad_pso_fi_p[i],ad_pso_fi_g[i]);
			printf("\n");
		}
	    // process
	    if (pso_max_iter>=1)
	    do
	    {
		// adaptive PSO - change PSO parameters
		if (adaptive_pso)
		{
			double ve=ad_v0*exp(-(2*(iter+1)/(float)(pso_max_iter-iter))*(2*(iter+1)/(float)(pso_max_iter-iter)));
			double vavg=0.0;
			double *vavgc=new double[size];
			for (int j=0;j<size;j++)
			    vavgc[j]=0;
			for (int i=0;i<pso_n;i++)
				for (int j=0;j<size;j++)
				{
					vavg+=particles[i][j+1+size]*particles[i][j+1+size];
					vavgc[j]+=particles[i][j+1+size]*particles[i][j+1+size];
				}
			vavg=sqrt(vavg);
			vavg/=pso_n;
			for (int j=0;j<size;j++)
			{
			    vavgc[j]=sqrt(vavgc[j]);
			    vavgc[j]/=pso_n;
			}
			// change omega
			if (vavg>ve)
				pso_o/=ad_pso_p;
			if (vavg<ve)
				pso_o*=ad_pso_p;
			if (pso_o>2) pso_o=2;
			if (pso_o<0) pso_o=0;
			// change fi_p,fi_g
			for (int i=0;i<size;i++)
			{
			    ad_pso_fi_p[i]=ad_pso_C*vavgc[i]/ad_vmax[i];
			    ad_pso_fi_g[i]=ad_pso_C*(1-vavgc[i]/ad_vmax[i]);
			    if (ad_pso_fi_p[i]>2.0) ad_pso_fi_p[i]=2.0;
			    if (ad_pso_fi_g[i]>2.0) ad_pso_fi_g[i]=2.0;
    			    if (ad_pso_fi_p[i]<0.0) ad_pso_fi_p[i]=0.0;
			    if (ad_pso_fi_g[i]<0.0) ad_pso_fi_g[i]=0.0;
			}
			printf("adaptive PSO - vavg %g ve %g params values %g",vavg,ve,pso_o);
			for (int i=0;i<size;i++)
			    printf(",(%g->%g,%g)",vavgc[i],ad_pso_fi_p[i],ad_pso_fi_g[i]);
			printf("\n");
			delete [] vavgc;
		}
		for (int i=0;i<pso_n;i++)
		{
			// update velocity
			for (int j=0;j<size;j++)
			{
				double rp=(rand()%10000)/10000.0;
				double rg=(rand()%10000)/10000.0;
				if (adaptive_pso) // percoordinate fi_p,fi_g
				{
				    pso_fi_p=ad_pso_fi_p[j];
				    pso_fi_g=ad_pso_fi_g[j];
				}
				particles[i][j+1+size]=pso_o*particles[i][j+1+size]+pso_fi_p*rp*(particles[i][j+1+2*size]-particles[i][j+1])+pso_fi_g*rg*(best_p[j+1]-particles[i][j+1]);				    
			}
			// update position
			for (int j=0;j<size;j++)
				particles[i][1+j]+=particles[i][j+1+size];
			// restart particle
			double rp=(rand()%10000)/10000.0;			
			if (rp<((pso_o<1)?(1-pso_o):0.0)*restart_prox)
			{
			    init_particle(particles[i], to_fit,nEV_F,i,size);
			    printf("r");
			}
			// calc f value
			particles[i][0]=solve_and_test(particles[i],to_fit,init_Z,init_F,ninit, perc_T, perc_A, nperc, irr_T, irr_A, nirr, EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi1,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t);
			// update bests
			if (particles[i][0]<particles[i][1+3*size])
			{
				for (int j=0;j<size;j++)
					particles[i][j+1+2*size]=particles[i][j+1];
				particles[i][1+3*size]=particles[i][0];
			}
			if (particles[i][0]<best_p[0])
			{
				for (int j=0;j<size;j++)
					best_p[j+1]=particles[i][j+1];
				best_p[0]=particles[i][0];
			}
			fflush(stdout);
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
	// make additional steepest descent iterations
	if (adaptive_pso)
	{
		if (use_steepest_descent==1)
best_p[0]=steepest_descent(best_p,to_fit,size,ad_vmax,init_Z,init_F,ninit,perc_T,perc_A,nperc,irr_T,irr_A,nirr,EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi1,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t,pso_eps,pso_max_iter);
		if (use_steepest_descent==2)
		{
			int best=-1;
			for (int i=0;i<pso_n;i++)
			{
particles[i][0]=steepest_descent(particles[i],to_fit,size,ad_vmax,init_Z,init_F,ninit,perc_T,perc_A,nperc,irr_T,irr_A,nirr,EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,fi1,values_T,values_Z,values_F,nvalues,param_values,nparam_values,t,pso_eps,pso_max_iter);
				if (best==-1)
					best=0;
				else
					if (particles[i][0]<particles[best][0])
						best=i;
				for (int j=0;j<=size;j++)
					best_p[j]=particles[best][j];
			}
				
		}
	}
	if (fit==2)
	    for (int i=0;i<size;i++)
		best_p[i+1]=param_values[i];
	// solve with best parameters values
	H_solver *ss;
	init_print=1;
	if (fit==1)
	{
	    ss=init_solver(best_p,to_fit,init_Z,init_F,ninit, perc_T, perc_A, nperc, irr_T, irr_A, nirr, EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,param_values,nparam_values);
	    FILE *fi3 = fopen(bfn, "wt");
	    fprintf(fi3,"%g %g %d %g %g %g %g %g %g %g %g %g ",ss->alpha,ss->gamma,func_in_kernel,func_power,sigmoid2, func_power_t,sigmoid2_t,ss->L,ss->vgm_n,ss->vgm_s0,ss->vgm_s1,ss->vgm_a);
	    for (int i=0;i<nEV_F;i++)
		fprintf(fi3,"%g ",ss->EV_C[i]);
	    if (vgm_nlayers!=0)
	    {
		for (int i=0;i<vgm_nlayers;i++)
		    fprintf(fi3,"%g ",vgm_k[i]*Hmult);
		fprintf(fi3,"%g",ss->perc_k);
	    }
	    else
		fprintf(fi3,"%g %g",ss->k*Hmult,ss->perc_k);
	    if (irr==0)
		fprintf(fi3," %g %g %g",ss->min_wetness,ss->max_wetness,ss->irr_volume);
	    fprintf(fi3,"\n");
	    fclose(fi3);
	}
	if (fit==2)
	    ss=init_solver(best_p,to_fit,init_Z,init_F,ninit, perc_T, perc_A, nperc, irr_T, irr_A, nirr, EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,irr,bc,tau_m,NULL,0);	    
	int nsave = 0;
	int nout=0;
	int d,oldd=-1;
	int logn=0,maxlogn=0;
	double err=0.0,srel_err=0.0;
	int nerr=0;	
	double *vs=new double [ncheck];
	FILE *fi2;
	fi2 = fopen(rfn, "wt");
	if (fit==1)
	    if (t<check_T[ncheck-1])
		t=check_T[ncheck-1];
	fit=2;
	for	(double tt = 0;tt <= t;tt += ss->tau)
	{
		// save result
		if (tt ==0)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g %g %g \n", tt,(double)i*ss->dL,ss->U[i],ss->wetness(i));
			fflush(fi2);
			nsave++;
		}
		// save old 
		for (int i=0;i<ncheck;i++)
			if ((check_T[i]>=tt)&&(check_T[i]<(tt+ss->tau)))
			{
				for (int j=0;j<N-1;j++)
					if (((ss->dL*j)<check_Z[i])&&((ss->dL*(j+1))>=check_Z[i]))
					{
						double k2=(check_Z[i]-(ss->dL*j))/ss->dL;
						vs[i]=(1-k2)*ss->U[j]+k2*ss->U[j+1];
						break;
					}
			}
		// solve	
		ss->calc_step();
		// add to err
		int is_out = 0;
		logn=0;
		for (int i=0;i<ncheck;i++)
			if ((check_T[i]>=tt)&&(check_T[i]<(tt+ss->tau)))
			{
				double k1=(check_T[i]-tt)/ss->tau;
				for (int j=0;j<N;j++)
				    if (((ss->dL*j)<check_Z[i])&&((ss->dL*(j+1))>=check_Z[i]))
				    {
					double k2=(check_Z[i]-(ss->dL*j))/ss->dL;
					double v=(1-k2)*ss->U[j]+k2*ss->U[j+1];
					v=(1-k1)*vs[i]+k1*v;
					if (v>0.0)
					    v=0.0;
					err+=(check_F[i]-v)*(check_F[i]-v);
					if ((check_F[i]==0)&&(v!=0.0))
					    srel_err+=1.0;
					if (check_F[i]!=0)
					{
					    if (fabs((check_F[i]-v)/check_F[i])<1)
						srel_err+=fabs((check_F[i]-v)/check_F[i]);
					    else 
						srel_err+=1.0;
					}
					nerr++;
					if (is_out == 0)
						fprintf(fi1, "%g ", check_T[i]);
					fprintf(fi1, "%g %g %g ", check_Z[i], check_F[i], v);
					logn++;
					is_out = 1;
				    }
			}
		if (logn>maxlogn) maxlogn=logn;
		if ((is_out)||(tt>nout*out_tau))
		{
		    if (is_out==0)
			{
			fprintf(fi1,"%g ",tt);	
			for (int i=0;i<ncheck;i++)
				for (int j=0;j<N;j++)
				    if (((ss->dL*j)<check_Z[i])&&((ss->dL*(j+1))>=check_Z[i]))
				    {
					double k2=(check_Z[i]-(ss->dL*j))/ss->dL;
					double v=(1-k2)*ss->U[j]+k2*ss->U[j+1];
					if (v>0.0)
					    v=0.0;
					fprintf(fi1, "%g %g %g ", check_Z[i], check_F[i], v);
				    }
		    }
		    fprintf(fi1,"rlw %g twc %g sum_err %g avg_err %g avg_rel_err %g\n",ss->avg_root_layer_wetness(),ss->total_water_content(),err,(nerr?err/nerr:0),(nerr?srel_err/nerr:0));
		    nout++;
		}
		// save result
		if (tt > nsave*save_tau)
		{
			for (int i = 0;i < N + 1;i++)
				fprintf(fi2, "%g %g %g %g \n", tt,(double)i*ss->dL,ss->U[i],ss->wetness(i));
			fflush(fi2);
			nsave++;
		}
		if (!finite(ss->b_U[0]))
		{
		    err=1e300;
		    break;
		}
	}
	fprintf(fi1,"#checking err: %g\n",err);
	clear_solver(ss,to_fit);
	fclose(fi1);
}
/// main/////////////////
int main(int argc,char **argv)
{
	double Tm = 24 * 90 * 3600.0 + 1000.0;
	double Sm = 2 * 3600.0;
	double Om = 2.0*3600.0;
	int bc=0;
	double tau_m=1000.0/20.0;
	char *et_file=NULL;
	char *pv_file=NULL;
	double _ev_mu=1e100;
	int _pso_n=-1;
	double _pso_o=1e100;
	double _pso_fi_p=1e100;
	double _pso_fi_g=1e100;
	double _pso_eps=1e100;
	int _pso_max_iter=-1;
	char *fi_file=NULL;
#ifdef _OPENMP
	omp_init_lock(&lock);
#endif
	if (argc==1)
	{
		printf("Tm : end time\n");
		printf("Sm : save time\n");
		printf("Om : log output time\n");
		printf("BS : grid block size\n");
		printf("NB : number of grid blocks\n");
		printf("I : 1 - irrigation disabled\n");
		printf("Idelay: delay before the first irrigation application\n");
		printf("interIdelay: delay between successive irrigation applications\n");
		printf("B : bottom boundary condition - 0 -dU/dn=0, 1 - U=H0\n");
		printf("Tau : tau multiplier\n");
		printf("AVG: 0 - 0-order averaging, 1 - first order averaging\n");
		printf("Fit: 1 - fit and solve, 2 - solve fit given values\n");
		printf("fl: additional flags: \n\t\tbit 0 - 1 to 2 order upper bc transition, \n\t\tbit 2 - no day/night coef\n\t\tbit 3 - inf value particles\n\t\tbit 4 - steady state for initial distribution\n\t\tbit 5 - restrictions in upper boundary condition\n");
		printf("stDIV: time step divisor for steady state initial value computations\n");
		printf("stMI: maximal number of iterations for steady state initial value computations\n");
		printf("stEPS: eps for steady state initial value computations\n");
		printf("bounds_i_j: bounds for parameter changes in PSO search:\n\t\t i_0 - min, i_1 - max\n\t\t 0 - alpha, 1 - beta\n\t\t\
			2 - L\n\t\t(3,4,5,6) - van Genuchten-Mualem n,s0,s1,a\n\t\t 7 - filtration coefficient\n\t\t\
			8,9,10 - min/max wetness and irrigation amount for irrigation application\n\t\t\
			11 - EV coefs max and percipitation coef max");
		printf("vgm: name of file with layers soil VGM coefficients in form [s0,s1,n,a,h0,h1]\n");
		printf("et_file: name of file with ET values\n");
		printf("pv_file: name of file with fixed parameters values\n");
		printf("Hmult: multiplier for H values (1 - H in m, 10 - H in 10*m, etc)\n");
		printf("rpf: 1,2 - transpiration in right part, 3 - in upped boundary condition\n");
		printf("rld: root layer depth\n");
		printf("rl_file: file which contains the pairs <time,root length>\n");
		printf("ev_mu,pso_n,pso_o,pso_fi_p,pso_fi_g,pso_eps,pso_max_iter: used to override fitting params\n");
		printf("filename: output filenames prefix\n");
		printf("Hbottom: value of H at the bottom of the domain for boundary condition=1\n");
		printf("fi_file: fit input filename (default: fit_input.txt for 1D, fit_input_3d.txt for 3D)\n");
		printf("klin: linear coefficient for Kfilt change in time ((m/s)/s)\n");
		printf("kpf: kf(H) form - 1 - van Genuchten-Mualem, 2 - Averianov\n");
	double _ev_mu=1e100;
	int _pso_n=-1;
	double _pso_o=1e100;
	double _pso_fi_p=1e100;
	double _pso_eps=1e100;
	int _pso_max_iter=-1;
	}
	for (int i=1;i<argc;i+=2)
	{
		if (strcmp(argv[i],"Tm")==0)
			Tm=atof(argv[i+1]);
		if (strcmp(argv[i],"Sm")==0)
			Sm=atof(argv[i+1]);
		if (strcmp(argv[i],"Om")==0)
			Om=atof(argv[i+1]);
		if (strcmp(argv[i],"BS")==0)
			BS=atoi(argv[i+1]);
		if (strcmp(argv[i],"NB")==0)
			NB=atoi(argv[i+1]);
		if (strcmp(argv[i],"I")==0)
			irr=atoi(argv[i+1]);
		if (strcmp(argv[i],"B")==0)
			bc=atoi(argv[i+1]);
		if (strcmp(argv[i],"Tau")==0)
			tau_m=atof(argv[i+1]);
		if (strcmp(argv[i],"Idelay")==0)
			irr_delay=atof(argv[i+1]);
		if (strcmp(argv[i],"interIdelay")==0)
			inter_irr_delay=atof(argv[i+1]);
		if (strcmp(argv[i],"Fit")==0)
			fit=atoi(argv[i+1]);
		if (strcmp(argv[i], "fl") == 0)
			flags = atoi(argv[i + 1]);
		if (strcmp(argv[i], "stDIV") == 0)
			stat_div = atof(argv[i + 1]);
		if (strcmp(argv[i], "stMI") == 0)
			stat_maxiter = atoi(argv[i + 1]);
		if (strcmp(argv[i], "stEPS")== 0)
			stat_eps = atof(argv[i + 1]);
		if (strcmp(argv[i], "rpf")== 0)
			rpf = atoi(argv[i + 1]);
		if (strcmp(argv[i], "rld")== 0)
			rld = atof(argv[i + 1]);
		if (strcmp(argv[i], "kpf")== 0)
			kpf = atoi(argv[i + 1]);
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
				if ((ii>=0)&&(ii<18))
				    bounds[ii][j]=v;
			}
		    }
		}
		if (strstr(argv[i],"vgm")!=NULL)
		{
		    FILE *fi=fopen(argv[i+1],"rt");
		    if (fi)
		    {
			char str[1024];
			while (fgets(str,1024,fi)) vgm_nlayers++;
			vgm_ns=new double[vgm_nlayers];
			vgm_as=new double[vgm_nlayers];
			vgm_s0s=new double[vgm_nlayers];
			vgm_s1s=new double[vgm_nlayers];
			vgm_h0=new double[vgm_nlayers];
			vgm_h1=new double[vgm_nlayers];
			vgm_k=new double[vgm_nlayers];
			av_ps=new double[vgm_nlayers];
			vgm_nlayers=0;
			fseek(fi,0,SEEK_SET);			
			while (fscanf(fi,"%lg %lg %lg %lg %lg %lg %lg %lg\n",vgm_s0s+vgm_nlayers,vgm_s1s+vgm_nlayers,vgm_ns+vgm_nlayers,vgm_as+vgm_nlayers,vgm_h0+vgm_nlayers,vgm_h1+vgm_nlayers,vgm_k+vgm_nlayers,av_ps+vgm_nlayers)==8)
			{
			    vgm_k[vgm_nlayers]/=Hmult;
			    vgm_nlayers++;
			}
		    }		    
		}
		if (strstr(argv[i],"rl_file")!=NULL)
		{
		    FILE *fi=fopen(argv[i+1],"rt");
		    if (fi)
		    {
			char str[1024];
			nrl=0;
			while (fgets(str,1024,fi)) nrl++;
			rlT=new double[nrl];
			rlV=new double[nrl];
			nrl=0;
			fseek(fi,0,SEEK_SET);			
			while (fscanf(fi,"%lg %lg\n",rlT+nrl,rlV+nrl)==2)
			    nrl++;
		    }		    
		}
		if (strstr(argv[i],"et_file")!=NULL)
		    et_file=argv[i+1];
		if (strstr(argv[i],"pv_file")!=NULL)
		    pv_file=argv[i+1];
		if (strstr(argv[i],"H_mult")!=NULL)
		{
		    Hmult=atof(argv[i+1]);
		    HtoPmult=98/Hmult;
		}
		if (strstr(argv[i],"ev_mu")!=NULL)
		    _ev_mu=atof(argv[i+1]);
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
		if (strstr(argv[i],"filename")!=NULL)
		    filename=argv[i+1];
		if (strcmp(argv[i], "Hbottom")== 0)
			Hbottom = atof(argv[i + 1]);
		if (strstr(argv[i],"fi_file")!=NULL)
		    fi_file=argv[i+1];
		if (strstr(argv[i],"klin")!=NULL)
		    klin=atof(argv[i+1]);
		if (strstr(argv[i],"averianov_power")!=NULL)
		    averianov_power=atof(argv[i+1]);
		if (strcmp(argv[i], "FK") == 0)
		{
			func_in_kernel = atoi(argv[i + 1]);
			if (func_in_kernel==0) func_power=1;
			if (func_in_kernel==1) func_power=0.5;
			if (func_in_kernel==2) func_power=2;
		}
		if (strcmp(argv[i], "FP") == 0)
			func_power = atof(argv[i + 1]);
		if (strcmp(argv[i], "S2") == 0)
			sigmoid2 = atof(argv[i + 1]);
		if (strcmp(argv[i], "FPt") == 0)
			func_power_t = atof(argv[i + 1]);
		if (strcmp(argv[i], "S2t") == 0)
			sigmoid2_t = atof(argv[i + 1]);
		if (strcmp(argv[i], "int_niter") == 0)
			integr_max_niter = atoi(argv[i + 1]);
		if (strcmp(argv[i], "EPS") == 0)
			global_eps = atof(argv[i + 1]);
		if (strcmp(argv[i], "EPS2") == 0)
			global_eps2 = atof(argv[i + 1]);
		if (strcmp(argv[i], "metric") == 0)
			metric = atoi(argv[i + 1]);
		if (strcmp(argv[i], "kb2_coef") == 0)
			kb2_coef = atof(argv[i + 1]);
		if (strcmp(argv[i], "APSO") == 0)
			adaptive_pso = atoi(argv[i + 1]);
		if (strcmp(argv[i], "restart_prox") == 0)
			restart_prox = atof(argv[i + 1]);
		if (strcmp(argv[i], "use_steepest_descent") == 0)
			use_steepest_descent = atoi(argv[i + 1]);
		if (strcmp(argv[i], "st_desc_max_iter") == 0)
			st_desc_max_iter = atoi(argv[i + 1]);
		if (strcmp(argv[i], "st_desc_eps") == 0)
			st_desc_eps = atof(argv[i + 1]);
		if (strcmp(argv[i], "newrand") == 0)
			srand(time(NULL));
	}
	sN=N=(BS*NB);
	sNB = NB;
	printf("N %d (%d*%d) tend %g tsave %g \n",N,BS,NB,Tm,Sm);
	fflush(stdout);		
	F=new double[N+2];
	BVK=new double[BS];
	if (fit!=0)
	{
		FILE *fi1;
		if (fi_file==NULL)
		    fi1=fopen("fit_input.txt", "rt");
		else
		    fi1 = fopen(fi_file, "rt");
		double *values_T,*values_Z,*values_F;
		double *check_T,*check_Z,*check_F;
		double *init_Z,*init_F;		
		double *perc_T, *perc_A;
		double *irr_T, *irr_A;
		double param_values[20];
		int nparam_values=0;
		int nvalues;
		int ninit;
		int ncheck;
		int nperc, nirr;
		double *EV_T,**EV_F,*LAI;
		double ev_mu;
		int nEV_T,nEV_F;
		int to_fit;
		int pso_n;
		double pso_o, pso_fi_p, pso_fi_g, pso_eps;
		int pso_max_iter;
		char str[2048];
		// read data
		if (fscanf(fi1,"%d %d %d %d %d %d %d %d %lg\n",&to_fit,&nvalues,&ninit,&ncheck,&nperc,&nirr,&nEV_T,&nEV_F,&ev_mu)!=9) return 0;
		if (fscanf(fi1,"%d %lg %lg %lg %lg %d\n",&pso_n,&pso_o,&pso_fi_p,&pso_fi_g,&pso_eps,&pso_max_iter)!=6) return 0;
		if (_ev_mu!=1e100) ev_mu=_ev_mu;
		if (_pso_n!=-1) pso_n=_pso_n;
		if (_pso_o!=1e100) pso_o=_pso_o;
		if (_pso_fi_p!=1e100) pso_fi_p=_pso_fi_p;
		if (_pso_fi_g!=1e100) pso_fi_g=_pso_fi_g;
		if (_pso_eps!=1e100) pso_eps=_pso_eps;
		if (_pso_max_iter!=-1) pso_max_iter=_pso_max_iter;
		printf("%d %d %d %d %d %d %d %d %lg\n",to_fit,nvalues,ninit,ncheck,nperc,nirr,nEV_T,nEV_F,ev_mu);
		printf("%d %lg %lg %lg %lg %d\n",pso_n,pso_o,pso_fi_p,pso_fi_g,pso_eps,pso_max_iter);
		// values to fit -  in 10m
		values_T=new double[nvalues];
		values_Z=new double[nvalues]; //  Z for 1d, (x,y,z) for 3d
		values_F=new double[nvalues];
		for (int i=0;i<nvalues;i++)
		    if (fscanf(fi1,"%lg %lg %lg\n",values_T+i,values_Z+i,values_F+i)!=3)
		    	return 0;
		    else
		        values_F[i]*=Hmult;
		// initial H
		init_Z = new double[ninit]; //  Z for 1d, (x,y,z) for 3d
		init_F = new double[ninit];
		for (int i = 0;i<ninit;i++)
		    if (fscanf(fi1, "%lg %lg\n", init_Z + i, init_F + i) != 2)
		    	return 0;
		    else
		        init_F[i]*=Hmult;
		// percipitation - fixed on upper plane for 3d in m/s
		perc_T = new double[nperc];
		perc_A = new double[nperc];
		for (int i = 0;i<nperc;i++)
			if (fscanf(fi1, "%lg %lg\n", perc_T + i, perc_A + i) != 2)
				return 0;
		// irrigation - fixed on upper plane for 3d in m/s
		irr_T = new double[nirr];
		irr_A = new double[nirr];
		for (int i = 0;i < nirr;i++)
			if (fscanf(fi1, "%lg %lg\n", irr_T + i, irr_A + i) != 2)
				return 0;
		// values to check in 10m
		check_T=new double[ncheck];
		check_Z=new double[ncheck]; //  Z for 1d, (x,y,z) for 3d
		check_F=new double[ncheck];
		for (int i=0;i<ncheck;i++)
		    if (fscanf(fi1,"%lg %lg %lg\n",check_T+i,check_Z+i,check_F+i)!=3)
		    	return 0;
		    else
		        check_F[i]*=Hmult;
		// evapotranspiration in m/s
		EV_T=new double[nEV_T]; // T for 1d, (T,x,y) for 3d
		EV_F=new double*[nEV_F];
		LAI=new double[nEV_T];
		for (int j = 0;j < nEV_F;j++)
			EV_F[j] = new double[nEV_T];
		int kk = 0;
		FILE *sfi1=fi1,*ff;
		if (et_file)
		    if (ff=fopen(et_file,"rt"))
			fi1=ff;
		for (int i=0;i<nEV_T;i++)
		{
			if (fscanf(fi1, "%lg ", EV_T + i) != 1)
				return 0;
			if (fscanf(fi1, "%lg ", LAI + i) != 1)
				return 0;
			for (int j = 0;j < nEV_F;j++)
				if (fscanf(fi1, "%lg ", EV_F[j] + i) != 1)
					return 0;
			if (i==0)
			    printf("ev00 - %g\n",EV_F[0][0]);
			int j=fscanf(fi1, "\n");
		}
		if (et_file)
		{
		    fclose(ff);
		    fi1=sfi1;
		}
		// read fixed parameter values
		if (pv_file)
		    if (ff=fopen(pv_file,"rt"))
			fi1=ff;
		while (fscanf(fi1,"%lg ",&param_values[nparam_values++])==1)
		    if (nparam_values==20)
		        break;
		if (pv_file)
		    fclose(ff);
		if (nparam_values!=20)
		    nparam_values--;
		// run
		fit_and_solve(Tm,Sm,Om,irr,bc,tau_m,init_Z,init_F,ninit,perc_T,perc_A,nperc,irr_T,irr_A,nirr,values_T,values_Z,values_F,nvalues,check_T,check_Z,check_F,ncheck,EV_T,EV_F,LAI,ev_mu,nEV_T,nEV_F,to_fit,pso_n,pso_o,pso_fi_p,pso_fi_g,pso_eps,pso_max_iter,fit,&param_values[0],nparam_values,et_file,pv_file);
	}
#ifdef _OPENMP
	omp_destroy_lock(&lock);
#endif
	return 0;
}