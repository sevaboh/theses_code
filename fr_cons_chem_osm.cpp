#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <vector>
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
// variables
double beta;
double alpha;
double tau;
// constants
#define M 100
double cv=0.34;
double l=25.0;
double mu=0.00095;
double sigma=0.38;
double d=0.02;
double k=0.01;
double nu=2.8e-5;
double C0=200;
double H0=10;
double sa(double x) {return (1/(alpha*alpha))*pow(x,2.0*(1.0-alpha)); }
double ra(double x) {return ((1.0-alpha)/(alpha*alpha))*pow(x,1.0-2.0*alpha); }
double h=l/(M+1.0);
// arrays
double H[M+2],C[M+2];
double A[M+2],B[M+2],S[M+2],F[M+2];
std::vector<double *> Hs,Cs;
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
double v(int i) { return k*H[i]-nu*C[i]; }
double w(int j,int v) { return (pow(tau,1.0-beta)/Gamma(2.0-beta))*(pow((double)(j-v+1.0),(double)(1.0-beta))-pow((double)(j-v),(double)(1.0-beta))); }
double AC(int i) { return (0.5/h)*(d*((sa(i*h)/h)-(ra(i*h)/2))-(sa(i*h)/(4.0*h))*(v(i+1)-v(i-1))); }
double BC(int i) { return (0.5/h)*(d*((sa(i*h)/h)+(ra(i*h)/2))+(sa(i*h)/(4.0*h))*(v(i+1)-v(i-1))); }
double SC(int i) { return AC(i)+BC(i)+sigma/(pow(tau,beta)*Gamma(2.0-beta));}
double FC(int i) 
{
    double sum=0;
    if (Cs.size()>=2)
    for (int j=0;j<Cs.size()-1;j++)
	sum+=w(Cs.size(),j)*(Cs[j+1][i]-Cs[j][i])/tau;
    return sigma*(sum-(C[i]/(pow(tau,beta)*Gamma(2.0-beta))))-
	   0.5*(d/h)*((sa(i*h)/h)*(C[i-1]-2.0*C[i]+C[i+1])+(ra(i*h)/2)*(C[i+1]-C[i-1]))-
	   0.5*(sa(i*h)/(4.0*h*h))*(v(i+1)-v(i-1)*(C[i+1]-C[i-1]));
}
double AH(int i) { return (0.5*cv/h)*((sa(i*h)/h)-(ra(i*h)/2)); }
double BH(int i) { return (0.5*cv/h)*((sa(i*h)/h)+(ra(i*h)/2)); }
double SH(int i) { return AH(i)+BH(i)+1.0/(pow(tau,beta)*Gamma(2.0-beta));}
double FH(int i) 
{
    double sum=0;
    if (Hs.size()>=2)
    for (int j=0;j<Hs.size()-1;j++)
	sum+=w(Hs.size(),j)*(Hs[j+1][i]-Hs[j][i])/tau;
    return (sum-(H[i]/(pow(tau,beta)*Gamma(2.0-beta))))-
	   0.5*(cv/h)*((sa(i*h)/h)*(H[i-1]-2.0*H[i]+H[i+1])+(ra(i*h)/2)*(H[i+1]-H[i-1]))+
	   0.5*(mu/h)*((sa(i*h)/h)*(C[i-1]-2.0*C[i]+C[i+1]+Cs[Cs.size()-1][i-1]-2.0*Cs[Cs.size()-1][i]+Cs[Cs.size()-1][i+1])+
		       (ra(i*h)/2)*(C[i+1]-C[i-1]+Cs[Cs.size()-1][i+1]-Cs[Cs.size()-1][i-1]));
}
void solveC()
{
	A[1]=0;
	B[1]=C0;
	for (int i=1;i<=M;i++)
	    A[i+1]=BC(i)/(SC(i)-AC(i)*A[i]);
	for (int i=1;i<=M;i++)
	    B[i+1]=(A[i+1]/BC(i))*(AC(i)*B[i]-FC(i));
	C[M+1]=0;
	for (int i=M;i>=0;i--)
	    C[i]=A[i+1]*C[i+1]+B[i+1];
}
void solveH()
{
	A[1]=0;
	B[1]=0;
	for (int i=1;i<=M;i++)
	    A[i+1]=BH(i)/(SH(i)-AH(i)*A[i]);
	for (int i=1;i<=M;i++)
	    B[i+1]=(A[i+1]/BH(i))*(AH(i)*B[i]-FH(i));
	H[M+1]=0;
	for (int i=M;i>=0;i--)
	    H[i]=A[i+1]*H[i+1]+B[i+1];
}
void calc_step()
{
	solveC();
	solveH();
	double *sC=new double[M+2];
	double *sH=new double[M+2];
	memcpy(sC,C,(M+2)*sizeof(double));
	memcpy(sH,H,(M+2)*sizeof(double));
	Cs.push_back(sC);
	Hs.push_back(sH);
}
void initialize()
{
	for (int i=0;i<M+2;i++)
	{
	    C[i]=0.0;
	    H[i]=H0;
	}
	for (int i=1;i<Cs.size();i++)
	    delete [] Cs[i];
	for (int i=1;i<Hs.size();i++)
	    delete [] Hs[i];
	Cs.clear();
	Hs.clear();
	Cs.push_back(C);
	Hs.push_back(H);
}
void solve(double a,double b,double t,int n,int ns)
{
	alpha=a;
	beta=b;
	tau=t;
	initialize();
	for (int i=0;i<n;i++)
	{
		calc_step();
		if ((i!=0)&&((i%ns)==0))
		for (int j=0;j<=M+1;j++)
			printf("a %g b %g tau %g t %g x %g C %g H %g\n",a,b,t,(i+1)*t,j*h,C[j],H[j]);
	}
}
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
int main(int argc, char**argv)
{
    if (argc==6)
        solve(atof(argv[1]),atof(argv[2]),atof(argv[3]),atoi(argv[4]),atoi(argv[5]));
    return 0;
}