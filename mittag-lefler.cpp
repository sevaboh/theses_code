// mittag-lefler.cpp : Defines the entry point for the console application.
//
// Visit http://www.johndcook.com/stand_alone_code.html for the source of this code and more like it.

#include <cmath>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <math.h>
#include <stdlib.h>
#ifdef linux
#include <float.h>
#endif
#include <vector>
#include <algorithm>
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
bool AbsGreat(double x, double y) {
    return fabs(x) > fabs(y);
}
bool GreaterPair(std::pair<double,double> x, std::pair<double,double> y) {
    return x.first > y.first;
}
double sum_sort_pyramid(std::vector<double> values)
{
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
	v+=v1;
	v/=(double)(sd+1);
	if (sd<50000) // recursive subdivition
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
		if (((fabs(z)>3.14*a)&&(b<(1+a)))&&((a>0)&&(a<1)))
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

