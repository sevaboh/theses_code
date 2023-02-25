/* Author:Vsevolod Bohaienko */
/* САРПОК 3D river pollution modelling */
/* right part - convection-diffusion with transformation to complex potential space
/* left part: */
/* notesting - dC/dt+tq*D[a+1]C */
/* testing - with only d*deltaC in right part. Only for sequencial algorithm. Without complex space transformation */
/* testing2 - 1d diffuse from initial*/
/* testing3 - same as basic with different complex potential space transformation.without cache optimization. */
/* testing4 - D[a]C+tqD[a+1]C without cache optimization. */
#define OCL_BLOCKED_SUM
//#define CACHE_OPT 16
//#define NOTESTING
//#define TESTING
//#define TESTING2
//#define TESTING3
#define TESTING4
#if defined(TESTING3) || defined(TESTING4)
	#define NOTESTING
	#undef CACHE_OPT
#endif
#define EPS 1E-10
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_DEPRECATE
#include <string.h>
#include "river_fr1.h"
extern int ruslo_flags; 
extern double ocl_v;
extern double Gamma(double); // Gamma function
// steps and grid transformations
#if defined(TESTING3) || defined(TESTING4)
#define fi0 5.0
#define h1 (2.0*fi0/(2.0*m+1.0))
#define h2 (1.0/n)
#define fi(i) ((i)*h1)
#define psi(j) (((j)-0.5)*h2)
#else
#define h1 (2.0/(2.0*m+1.0))
#define h2 (params.q1/n)
#define fi(i) (((i)-1.0)*h1)
#define psi(j) (((j)-1.5)*h2)
#endif
double ruslo_solver_fr1::fi_(int i) { return fi(i);}
double ruslo_solver_fr1::psi_(int j) { return psi(j);}
// коефициенты дискретизированного уравнения
#ifdef NOTESTING
	#define A(i,j) ((sqv(fi(i),psi(j))/h1)*((Dm/h1)-0.5))
	#define S(i,j) ((sqv(fi(i),psi(j))/h1)*((Dm/h1)+0.5))
#endif
#ifdef TESTING
	#define A(i,j) ((1.0/h1)*((Dm/h1)))
	#define S(i,j) ((1.0/h1)*((Dm/h1)))
#endif
#ifndef TESTING2
#ifndef TESTING4
	#define B(i,j) ((sigm/tau)+(2.0*sigm*params.tq1/(ptplus*Ga))+A(i,j)+S(i,j))
	#define F0(i,j) ((-(sigm/tau))*C0[i-2-offset_y][j-2-offset_x]+(2.0*sigm*params.tq1/(ptplus*Ga))*(prevC[prevC.size()-1][i-2-offset_y][j-2-offset_x]-2.0*C0[i-2-offset_y][j-2-offset_x]))
#ifndef CACHE_OPT
double ruslo_solver_fr1::F(int i,int j)
{
	double r=F0(i,j);
	double s=0.0;
	int jj=(((signed)prevC.size()-3)/2)+1;
	for (int ii=0;ii<jj;ii++)
		s+=ptminus*(piminus[jj-ii+1]-piminus[jj-ii])*(prevC[(2*(ii+1)-1)-1][i-2-offset_y][j-2-offset_x]-2.0*prevC[(2*(ii+1)-1)][i-2-offset_y][j-2-offset_x]+prevC[(2*(ii+1)-1)+1][i-2-offset_y][j-2-offset_x]);
    s*=2.0*sigm*params.tq1/(tau*tau*Ga);
	return r+s;
}
#else
#define F(i,j) (F0(i,j)+(2.0*sigm*params.tq1/(tau*tau*Ga))*ss[i-1-offset_y])
#endif
#else
#define B(i,j) ((sigm/(pt*Ga))*((1.0/twoaminus)+(2.0*params.tq1/tau))+A(i,j)+S(i,j))
#define F0(i,j) (-(ptminus/twoaminus)*C0[i-2-offset_y][j-2-offset_x]+(2.0*params.tq1/pt)*(prevC[prevC.size()-1][i-2-offset_y][j-2-offset_x]-2.0*C0[i-2-offset_y][j-2-offset_x]))
double ruslo_solver_fr1::F(int i,int j)
{
	double r=F0(i,j);
	double s=0.0,s2=0.0;
	int jj=(((signed)prevC.size()-3)/2)+1;
	for (int ii=0;ii<jj;ii++)
		s+=ptminus*(piminus[jj-ii+1]-piminus[jj-ii])*(prevC[(2*(ii+1)-1)-1][i-2-offset_y][j-2-offset_x]-2.0*prevC[(2*(ii+1)-1)][i-2-offset_y][j-2-offset_x]+prevC[(2*(ii+1)-1)+1][i-2-offset_y][j-2-offset_x]);
	for (int ii=0;ii<jj-1;ii++)
		s2+=ptminus*(piminus_half[jj-ii+1]-piminus_half[jj-ii])*(-prevC[(2*(ii+1)-1)][i-2-offset_y][j-2-offset_x]+prevC[(2*(ii+1)-1)+2][i-2-offset_y][j-2-offset_x]);
	if (prevC.size()>=2)
		s2+=ptminus*(piminus_half[2]-piminus_half[1])*(-prevC[2*jj-1][i-2-offset_y][j-2-offset_x]+C0[i-2-offset_y][j-2-offset_x]);
    s*=(2.0*params.tq1/tau);
	return (sigm/(tau*Ga))*(r+s+s2);
}
#endif
#endif
#ifdef NOTESTING
	#define P(i,j) (sqv(fi(i),psi(j))*Dm/(h2*h2))
	#define R(i,j) (sqv(fi(i),psi(j))*Dm/(h2*h2))
#endif
#ifdef TESTING
	#define P(i,j) (1.0*Dm/(h2*h2))
	#define R(i,j) (1.0*Dm/(h2*h2))
#endif
#ifdef TESTING2
	#define P(i,j) (sigm/(2*hb*Gb))
	#define R(i,j) (sigm/(2*hb*Gb))
#endif
#ifndef TESTING4
#ifndef TESTING2
	#define Q(i,j) ((sigm/tau)+(2.0*sigm*params.tq1/(ptplus*Ga))+P(i,j)+R(i,j))
	#define O0(i,j) (-(sigm/tau)*C1[i-2-offset_y][j-2-offset_x]+(2.0*sigm*params.tq1/(ptplus*Ga))*(prevC[prevC.size()-1][i-2-offset_y][j-2-offset_x]-2.0*C1[i-2-offset_y][j-2-offset_x]))
#else
	#define Q(i,j) ((1.0/tau)+(a0*params.tq1/(tau*tau*Ga))+P(i,j)+R(i,j))
	#define O0(i,j) (-((1.0/tau)+(a0*params.tq1/(tau*tau*Ga)))*C0[i-2-offset_y][j-2-offset_x]-P(i,j)*(C0[i-2-offset_y][j-1-offset_x]-2.0*C0[i-2-offset_y][j-2-offset_x]+C0[i-2-offset_y][j-3-offset_x]))
#endif
#ifndef CACHE_OPT
double ruslo_solver_fr1::O(int i,int j)
{
	double r=O0(i,j);
	double s=0.0;
#ifdef TESTING2
	int jj=prevC.size();
	for (int ii=1;ii<jj;ii++)
		s+=(piminus[jj-ii-1]-piminus[jj-ii])*(prevC[ii][i-2-offset_y][j-2-offset_x]-prevC[ii-1][i-2-offset_y][j-2-offset_x]);
    s*=params.tq1/(tau*tau*Ga);
	return r-s;
#else
	int jj=(((signed)prevC.size()-4)/2)+1;
	for (int ii=0;ii<jj;ii++)
		s+=ptminus*(piminus[jj-ii+1]-piminus[jj-ii])*(prevC[2*(ii+1)-1][i-2-offset_y][j-2-offset_x]-2.0*prevC[2*(ii+1)][i-2-offset_y][j-2-offset_x]+prevC[2*(ii+1)+1][i-2-offset_y][j-2-offset_x]);
    s*=2.0*sigm*params.tq1/(tau*tau*Ga);
	return r+s;
#endif
}
#else
#define O(i,j) (O0(i,j)+(2.0*sigm*params.tq1/(tau*tau*Ga))*ss[j-1-offset_x])
#endif
#else
#define Q(i,j) ((sigm/(pt*Ga))*((1.0/twoaminus)+(2.0*params.tq1/tau))+P(i,j)+R(i,j))
#define O0(i,j) ((ptminus/twoaminus)*((twoaminus-1.0)*(C1[i-2-offset_y][j-2-offset_x]-prevC[prevC.size()-1][i-2-offset_y][j-2-offset_x])-C1[i-2-offset_y][j-2-offset_x])+(2.0*params.tq1/pt)*(prevC[prevC.size()-1][i-2-offset_y][j-2-offset_x]-2.0*C1[i-2-offset_y][j-2-offset_x]))
double ruslo_solver_fr1::O(int i,int j)
{
	double r=O0(i,j);
	double s=0.0,s2=0.0;
	int jj=(((signed)prevC.size()-4)/2)+1;
	for (int ii=0;ii<jj;ii++)
		s+=ptminus*(piminus[jj-ii+1]-piminus[jj-ii])*(prevC[2*(ii+1)-1][i-2-offset_y][j-2-offset_x]-2.0*prevC[2*(ii+1)][i-2-offset_y][j-2-offset_x]+prevC[2*(ii+1)+1][i-2-offset_y][j-2-offset_x]);
	for (int ii=0;ii<jj;ii++)
		s2+=ptminus*(piminus[jj-ii+1]-piminus[jj-ii])*(-prevC[2*(ii+1)-1][i-2-offset_y][j-2-offset_x]+prevC[2*(ii+1)+1][i-2-offset_y][j-2-offset_x]);
    s*=(2.0*params.tq1/tau);
	return (sigm/(tau*Ga))*(r+s+s2);
}
#endif
ruslo_solver_fr1::ruslo_solver_fr1()
{
	C_m1_value=1;
	L=20;
	Dm=0.1;
	sigm=0.2;
	tau=1;
	params.q1=0.85;
	params.tq1=0.005;
	params.alpha=0.8;
	local_n=n=1000;
	local_m=m=100;
	offset_x=offset_y=0;
	step=0;
	C0=NULL;
	C1=NULL;
	alpha=NULL;
	beta=NULL;
	wait_var1=wait_var2=1;
#if defined(TESTING3) || defined(TESTING4)
	params.a=3650;
	params.a1=2.55;
	params.H=0.1;
#endif
}
ruslo_solver_fr1::~ruslo_solver_fr1()
{
    for (int i=0;i<local_m;i++)
	{
		delete [] C0[i];
		delete [] C1[i];
		delete [] alpha[i];
		delete [] beta[i];
	}
	for (int i=0;i<prevC.size();i++)
	{
		for (int j=0;j<local_m;j++)
			delete [] prevC[i][j];
		delete prevC[i];
	}
	delete [] C0;
	delete [] C1;
	delete [] alpha;
	delete [] beta;
#ifdef CACHE_OPT
	delete [] ss;
#endif
}
// in storage: i=[0,local_m-1],j=[0,local_n-1] in coef functions i=[2,m+1],j=[2,n+1]
// (local_storage_i(lsi),local_storage_j(lsj))->(offset_x+2+lsi,offset_y+2+lsj
// returns ||(vx,vy)||^2
double inline ruslo_solver_fr1::sqv(double f,double p)
{
#if defined(TESTING3) || defined(TESTING4)
   double s1=sin(0.5*M_PI*p);
   double s2=exp(0.5*M_PI*f)*cos(0.5*M_PI*p)+params.a1;
   return (4*params.a/(M_PI*M_PI))*((1/(exp(M_PI*f)*s1*s1))+(1/(s2*s2)));
#else
   double s1=sin(M_PI*p/(2*params.q1));
   double s2=sinh(M_PI*(f-1)/(2*params.q1));
   return s1*s1+s2*s2;
#endif
}
// прогоночные коефициенты обратной прогонки
void ruslo_solver_fr1::calc_alpha1_CPU(int f,int l)
{
   int i,j;
   int n_minus=2;
   if ((offset_x+l)!=n) n_minus=1;
   wait_var2=1;
   if (f==0)
   {
       // save C1
		double **CC=new double*[local_m];
		for (i=0;i<local_m;i++)
			CC[i]=new double[local_n];
           if ((ruslo_flags&1)==0)
		for (j=0;j<local_m;j++)
			for (i=0;i<local_n;i++)
				CC[j][i]=C1[j][i];
	   else
	   {
    	   for (j=0;j<ocl_offset_y;j++)
	       for (i=0;i<local_n;i++)
				CC[j][i]=C1[j][i];
    	   for (j=ocl_offset_y;j<local_m;j++)
	       for (i=0;i<ocl_offset_x;i++)
				CC[j][i]=C1[j][i];
	   }
	   prevC.push_back(CC);
	   // precalculate ptminus
	   if (piminus.size()<prevC.size())
#ifndef TESTING2
		   for (i=piminus.size();i<2*prevC.size();i++)
		   {
			   piminus.push_back(pow((double)i,1.0-params.alpha));
#ifdef TESTING4
			   piminus_half.push_back(pow((double)i-0.5,1.0-params.alpha));
#endif
		   }
#else
		   for (i=piminus.size();i<prevC.size();i++)
			   piminus.push_back(a0*(pow((double)(i+1),1.0-params.alpha)-pow((double)i,1.0-params.alpha)));
#endif
	   wait_var1=0;
   }
   while (wait_var1);
   // boudnary conditions
   if (offset_y==0)
   for (i=f;i<=l-n_minus;i++)
     alpha[0][i]=0;
   // propagation
   for (i=f;i<=l-n_minus;i++)
     for (j=1;j<=local_m-1;j++)
#ifndef TESTING2
		 alpha[j][i]=A(offset_y+j+1,offset_x+i+2)/(B(offset_y+j+1,offset_x+i+2)-alpha[j-1][i]*S(offset_y+j+1,offset_x+i+2));
#else
		 alpha[j][i]=0.0; // 1D for testing2
#endif
}
void ruslo_solver_fr1::calc_beta1_CPU(int f,int l)
{
   int i,j;
   int n_minus=2;
   if ((offset_x+l)!=n) n_minus=1;
   // boudnary conditions
   if (offset_y==0)
   for (i=f;i<=l-n_minus;i++)
#ifndef TESTING2
     beta[0][i]=1;
#else
	 beta[0][i]=C0[0][i];
#endif
   // propagation
   for (i=f;i<=l-n_minus;i++)
   {
#ifdef CACHE_OPT
	int jj=(((signed)prevC.size()-3)/2)+1;
    for (j=1;j<=local_m-1;j++)
		ss[j]=0.0;
	for (int kk=0;kk<=(jj/CACHE_OPT);kk++)
		for (int ll=0;ll<CACHE_OPT;ll++)
		{
			int ii=kk*CACHE_OPT+ll;
			if (ii<jj)
		    for (j=1;j<=local_m-1;j++)
				ss[j]+=ptminus*(piminus[jj-ii+1]-piminus[jj-ii])*(prevC[(2*(ii+1)-1)-1][j-1][i]-2.0*prevC[(2*(ii+1)-1)][j-1][i]+prevC[(2*(ii+1)-1)+1][j-1][i]);
		}
#endif
     for (j=1;j<=local_m-1;j++)
#ifndef TESTING2
       beta[j][i]=(alpha[j][i]/A(offset_y+j+1,offset_x+i+2))*(S(offset_y+j+1,offset_x+i+2)*beta[j-1][i]-F(offset_y+j+1,offset_x+i+2));
#else
	   beta[j][i]=C0[j-1][i]; // 1D for testing2
#endif
   }
}
// коефициенты прямой прогонки
void ruslo_solver_fr1::calc_alpha2_CPU(int f,int l)
{
   int i,j;
   int m_minus=2;
   if ((offset_y+l)!=m) m_minus=1;
   // save C0
#ifndef TESTING2
   wait_var1=1;
   if (f==0)
   {
	   double **CC=new double*[local_m];
	   for (i=0;i<local_m;i++)
		   CC[i]=new double[local_n];
           if ((ruslo_flags&1)==0)
	   for (j=0;j<local_m;j++)
	       for (i=0;i<local_n;i++)
				CC[j][i]=C0[j][i];
	   else
	   {
    	   for (j=0;j<ocl_offset_y;j++)
	       for (i=0;i<local_n;i++)
				CC[j][i]=C0[j][i];
    	   for (j=ocl_offset_y;j<local_m;j++)
	       for (i=0;i<ocl_offset_x;i++)
				CC[j][i]=C0[j][i];
	   }
	   prevC.push_back(CC);
	   wait_var2=0;
   }
   while(wait_var2);
#endif
   // boudnary conditions
   if (offset_x==0)
   for (j=f;j<=l-m_minus;j++)
#ifndef TESTING2
     alpha[j][0]=1;
#else
     alpha[j][0]=0;
#endif
   // propagation
   for (j=f;j<=l-m_minus;j++)
     for (i=1;i<=local_n-1;i++)
       alpha[j][i]=P(offset_y+j+2,offset_x+i+1)/(Q(offset_y+j+2,offset_x+i+1)-alpha[j][i-1]*R(offset_y+j+2,offset_x+i+1));
   // boudnary conditions
#ifdef TESTING2
   for (j=f;j<=l-m_minus;j++)
     alpha[j][local_n-1]=0;
#endif
}
void ruslo_solver_fr1::calc_beta2_CPU(int f,int l)
{
   int i,j;
   int m_minus=2;
   if ((offset_y+l)!=m) m_minus=1;
   // boudnary conditions
   if (offset_x==0)
   for (j=f;j<=l-m_minus;j++)
     beta[j][0]=0;
   // propagation
   for (j=f;j<=l-m_minus;j++)
   {
#ifdef CACHE_OPT
	double s=0.0;
	int jj=(((signed)prevC.size()-4)/2)+1;
    for (i=1;i<=local_n-1;i++)
		ss[i]=0.0;
	for (int kk=0;kk<=(jj/CACHE_OPT);kk++)
		for (int ll=0;ll<CACHE_OPT;ll++)
		{
			int ii=kk*CACHE_OPT+ll;
			if (ii<jj)
		     for (i=1;i<=local_n-1;i++)
				ss[i]+=ptminus*(piminus[jj-ii+1]-piminus[jj-ii])*(prevC[2*(ii+1)-1][j][i-1]-2.0*prevC[2*(ii+1)][j][i-1]+prevC[2*(ii+1)+1][j][i-1]);
		}
#endif
     for (i=1;i<=local_n-1;i++)
       beta[j][i]=(alpha[j][i]/P(offset_y+j+2,offset_x+i+1))*(R(offset_y+j+2,offset_x+i+1)*beta[j][i-1]-O(offset_y+j+2,offset_x+i+1));
   }
   // boudnary conditions
#ifdef TESTING2
   for (j=f;j<=l-m_minus;j++)
     beta[j][local_n-1]=0;
#endif
}
// functions for GPU+CPU computations
void ruslo_solver_fr1::calc_alpha1(int f,int l)
{
#ifdef USE_OPENCL
	if (ruslo_flags&1)
	{
		// do timing estimation and sets GPU offsets
		if (!(ruslo_flags&2))
		{
		if (!(ruslo_flags&4))
		{
    		    if (step==0)
			ocl_offset_x=(l-f)*ocl_v;
		}
		else
		{
		    if ((step%2)==0)
			tl0=_GetTickCount();
		    if ((step%2)==1)
			tl0=_GetTickCount()-tl0;
		    if (est_first)
		    {
			if (est_first==2)
			    t10=tl0;
			if (est_first==6)
			{
			    t20=tl0;
			    est_first=0;
			    char str[1024];
			    sprintf(str,"RUSLO FR1 OCL est done first step %d t1 %g t5 %g v %g",dec_step,t10,t20,ocl_v*dec_ocl_v);
			    SWARN(str);
			}
			else
    			    est_first++;
		    }
		    else
		    {
			if ((step%2)==1)
			{
			double est=t10+(t20-t10)*(step-dec_step)/5;
			if (fabs(tl0-est)>0.1*est)
			{
			    dec_ocl_v/=2.0;
			    est_first=1;
			    dec_step=step;
			    // send prevC from CPU to GPU 
			    int nox=(l-f)*ocl_v*dec_ocl_v;
			    char str[1024];
			    sprintf(str,"RUSLO FR1 OCL decrease step %d est %g real %g new v %g ox %d newox %d",step,est,tl0,ocl_v*dec_ocl_v,ocl_offset_x,nox);
			    SWARN(str);
	    		    if (ocl_offset_x-nox)
				for (int i=0;i<ocl_offset_y;i++)
				{
				    for (int j=0;j<2*step;j++)
					queue->EnqueueWriteBuffer(b_additional,prevC[j][i]+nox,sizeof(double)*(local_n*local_m*(1+j)+i*local_n+nox),sizeof(double)*(ocl_offset_x-nox));
				    queue->EnqueueWriteBuffer(b_C1,C1[i]+nox,sizeof(double)*(i*local_n+nox),sizeof(double)*(ocl_offset_x-nox));
				    queue->EnqueueWriteBuffer(b_C0,C0[i]+nox,sizeof(double)*(i*local_n+nox),sizeof(double)*(ocl_offset_x-nox));
				}
        		    gpu_sync_h(b_C1,C1);		
			    dec_flag=1;
			}
			}
		    }
		    ocl_offset_x=(l-f)*ocl_v*dec_ocl_v;
		}
		}
		// realloc gpu buffer for prevC
		if (gpu_prevC_size<(2*step+3))
		{
			int os=gpu_prevC_size;
			double *b=new double[os*local_n*local_m];
			while (gpu_prevC_size<(2*step+3)) gpu_prevC_size*=2;
			queue->EnqueueBuffer(b_additional,b);
			delete b_additional;
			b_additional=prg->create_buffer(CL_MEM_READ_WRITE,sizeof(double)*local_n*local_m*gpu_prevC_size,NULL);
			queue->EnqueueWriteBuffer(b_additional,b,0,sizeof(double)*local_n*local_m*os);	
		}
		// CPU1->GPU prevC synch
		if (ocl_offset_x)
		for (int i=ocl_offset_y;i<local_m;i++)
			queue->EnqueueWriteBuffer(b_additional,C1[i],sizeof(double)*(local_n*local_m*(1+2*step)+i*local_n),sizeof(double)*ocl_offset_x);
		// do process
		if ((f+ocl_offset_x)<l)
			gpu_run_kernel(k_av,f+ocl_offset_x,l,step,1,this,sizeof(ruslo_solver_fr1));
		calc_alpha1_CPU(f,f+ocl_offset_x);
		return;
	}
#endif
	calc_alpha1_CPU(f,l);
}
void ruslo_solver_fr1::calc_beta1(int f,int l)
{
#ifdef USE_OPENCL
	if (ruslo_flags&1)
	{
		if ((f+ocl_offset_x)<l)
#ifdef OCL_BLOCKED_SUM
			gpu_run_kernel(k_bv,f+ocl_offset_x,l,step,16,this,sizeof(ruslo_solver_fr1));
#else
			gpu_run_kernel(k_bv,f+ocl_offset_x,l,step,1,this,sizeof(ruslo_solver_fr1));
#endif
		calc_beta1_CPU(f,f+ocl_offset_x);
		return;
	}
#endif
	calc_beta1_CPU(f,l);
}
void ruslo_solver_fr1::calc_alpha2(int f,int l)
{
#ifdef USE_OPENCL
	if (ruslo_flags&1)
	{
		// do timing estimation and sets GPU offsets
		if (!(ruslo_flags&2))
		{
		    if (!(ruslo_flags&4))
		    {
			if (step==0)
			    ocl_offset_y=(l-f)*ocl_v;
		    }
		    else
		    {
			// do prevC CPU->GPU copy
			if (dec_flag==1)
			{
			    int noy=(l-f)*ocl_v*dec_ocl_v;
			    char str[1024];
			    sprintf(str,"RUSLO FR1 OCL decrease step oy %d newoy %d",ocl_offset_y,noy);
			    SWARN(str);
			    if (ocl_offset_x)
				for (int i=noy;i<ocl_offset_y;i++)
				{
				    for (int j=0;j<2*step+1;j++)
					queue->EnqueueWriteBuffer(b_additional,prevC[j][i],sizeof(double)*(local_n*local_m*(1+j)+i*local_n),sizeof(double)*ocl_offset_x);
				    queue->EnqueueWriteBuffer(b_C1,C1[i],sizeof(double)*(i*local_n),sizeof(double)*ocl_offset_x);
				    queue->EnqueueWriteBuffer(b_C0,C0[i],sizeof(double)*(i*local_n),sizeof(double)*ocl_offset_x);
				}
        		    gpu_sync_v(b_C0,C0);		
			    dec_flag=0;
    			}
			ocl_offset_y=(l-f)*ocl_v*dec_ocl_v;
	    	    }
    		    if (step==0)
        		    gpu_sync_h(b_C1,C1);		
		}
		// CPU1->GPU prevC synch
		if (local_n-ocl_offset_x)
		for (int i=0;i<ocl_offset_y;i++)
			queue->EnqueueWriteBuffer(b_additional,C0[i]+ocl_offset_x,sizeof(double)*(local_n*local_m*(1+2*step+1)+i*local_n+ocl_offset_x),sizeof(double)*(local_n-ocl_offset_x));
		if ((f+ocl_offset_y)<l)
			gpu_run_kernel(k_ah,f+ocl_offset_y,l,step,1,this,sizeof(ruslo_solver_fr1));
		calc_alpha2_CPU(f,f+ocl_offset_y);
		return;
	}
#endif
	calc_alpha2_CPU(f,l);
}
void ruslo_solver_fr1::calc_beta2(int f,int l)
{
#ifdef USE_OPENCL
	if (ruslo_flags&1)
	{
		if ((f+ocl_offset_y)<l)
#ifdef OCL_BLOCKED_SUM
			gpu_run_kernel(k_bh,f+ocl_offset_y,l,step,16,this,sizeof(ruslo_solver_fr1));
#else
			gpu_run_kernel(k_bh,f+ocl_offset_y,l,step,1,this,sizeof(ruslo_solver_fr1));
#endif
		calc_beta2_CPU(f,f+ocl_offset_y);
		return;
	}
#endif
	calc_beta2_CPU(f,l);
}
#ifdef USE_OPENCL
void ruslo_solver_fr1::calc_C2(int f,int l)
{
	if (ruslo_flags&1)
	{
		if ((f+ocl_offset_y)<l)
			gpu_run_kernel(k_C2,f+ocl_offset_y,l,step,1,this,sizeof(ruslo_solver_fr1));
		local1D_solver::calc_C2(f,f+ocl_offset_y);
		gpu_sync_v(b_C0,C0);
	}
	else
		local1D_solver::calc_C2(f,l);
}
void ruslo_solver_fr1::calc_C1(int f,int l)
{
	if (ruslo_flags&1)
	{
		if ((f+ocl_offset_x)<l)
			gpu_run_kernel(k_C1,f+ocl_offset_x,l,step,1,this,sizeof(ruslo_solver_fr1));
		local1D_solver::calc_C1(f,f+ocl_offset_x);
		gpu_sync_h(b_C1,C1);
	}
	else
		local1D_solver::calc_C1(f,l);
}
#endif
// initial conditions on C
void ruslo_solver_fr1::initial_C()
{
   int i,j;
   for (i=0;i<local_n;i++)
     for (j=0;j<local_m;j++)
        C0[j][i]=C1[j][i]=0.0;
   for (i=0;i<local_n;i++)
	    C0[0][i]=C1[0][i]=1.0;
#ifdef TESTING2 
   for (i=0;i<local_n;i++)
     for (j=0;j<local_m;j++)
		C0[j][i]=C1[j][i]=i*((double)local_n/(local_n-1.0))*h2*(params.q1-i*((double)local_n/(local_n-1.0))*h2);
#endif
}
double ruslo_solver_fr1::X(double f,double p)
{
#if !defined(TESTING3) && !defined(TESTING4)
	if (f>0.99) f=0.99; // to prevent too big values
	double l1=0.5*sqrt(L*L-1);
	double s1=sinh(M_PI*(f-1)/params.q1);
	double c1=cosh(M_PI*(f-1)/(2*params.q1));
	double c2=cos(M_PI*p/(2*params.q1));
    return l1*s1/(c1*c1-c2*c2);
#else
	  return params.H*L*exp(0.5*M_PI*f)*sin(0.5*M_PI*p)+p*L*(0.5-params.H);
#endif
}
double ruslo_solver_fr1::Y(double f,double p)
{
#if !defined(TESTING3) && !defined(TESTING4)
	if (f>0.99) f=0.99; // to prevent too big values
	double l1=0.5*sqrt(L*L-1);
	double s1=sin(M_PI*p/params.q1);
	double c1=cosh(M_PI*(f-1)/(2*params.q1));
	double c2=cos(M_PI*p/(2*params.q1));
    return -l1*s1/(c1*c1-c2*c2);
#else
    return params.H*L*exp(0.5*M_PI*f)*cos(0.5*M_PI*p)+f*L*(0.5-params.H);
#endif
}
void ruslo_solver_fr1::init_process(int no_alloc)
{
        delete [] C0;
        delete [] C1;
        delete [] alpha;
        delete [] beta;
		// allocating data
        C0=new double *[local_m];
        C1=new double *[local_m];
        alpha=new double *[local_m];
        beta=new double *[local_m];
        for (int i=0;i<local_m;i++)
        {
          C0[i]=new double [local_n];
          C1[i]=new double [local_n];
          alpha[i]=new double [local_n];
          beta[i]=new double [local_n];
		  memset(C0[i],0,local_n*sizeof(double));
		  memset(C1[i],0,local_n*sizeof(double));
		  memset(alpha[i],0,local_n*sizeof(double));
		  memset(beta[i],0,local_n*sizeof(double));
        }
#ifdef CACHE_OPT
		ss=new double[local_n+local_m];
#endif
		initial_C();
		// precalculations
		if (params.alpha<1E-6) params.alpha=1E-6;
		if (params.alpha>1.0) params.alpha=1.0;
		Ga=Gamma(2.0-params.alpha);
		ptplus=pow(tau,1.0+params.alpha);
		pt=pow(tau,params.alpha);
		ptminus=pow(tau,1.0-params.alpha);
#ifdef TESTING2
		if (params.alpha<0.0) params.alpha=0.0;
		if (params.alpha>=0.99) params.alpha=0.99;
		if (Dm<1.0) Dm=1.0;
		if (Dm>2.0) Dm=2.0;
		Ga=Gamma(1.0-params.alpha);
		Gb=Gamma(3.0-Dm);
		hb=pow(h2,Dm);
		a0=ptminus/(1.0-params.alpha);
#endif
#ifdef TESTING4
		twoaminus=pow(2.0,1.0-params.alpha);
#endif
		step=0;
#ifdef USE_OPENCL
		if (l1d_threads!=1) ruslo_flags&=~1;
		if (ruslo_flags&1)
		{
			ocl_offset_x=ocl_offset_y=0;
			dec_ocl_v=1.0;
			est_first=1;
			dec_step=0;
			dec_flag=0;
			init_opencl(this,sizeof(ruslo_solver_fr1));
			gpu_prevC_size=10;
			b_additional=prg->create_buffer(CL_MEM_READ_WRITE,sizeof(double)*local_n*local_m*gpu_prevC_size,NULL);
		}
#endif
}
void ruslo_solver_fr1::send_parameters(interprocess_io *io,int msock,int update)
{
   io->send(msock,&Dm,sizeof(double));
   io->send(msock,&sigm,sizeof(double));
   io->send(msock,&tau,sizeof(double));
   io->send(msock,&params,sizeof(params));
}
void ruslo_solver_fr1::recv_parameters(interprocess_io *io,int msock,int update)
{
    io->recv(msock,&Dm,sizeof(double));
    io->recv(msock,&sigm,sizeof(double));
    io->recv(msock,&tau,sizeof(double));
    io->recv(msock,&params,sizeof(params));
}
void ruslo_solver_fr1::copy_parameters(local1D_solver *slv,int update)
{
	ruslo_solver_fr1 *rs=dynamic_cast<ruslo_solver_fr1 *>(slv);
	Dm=rs->Dm;
	sigm=rs->sigm;
	tau=rs->tau;
	params=rs->params;
}
// serializing
void ruslo_solver_fr1::serialize(FILE *fi)
{
	fwrite(&n,sizeof(n),1,fi);
	fwrite(&m,sizeof(m),1,fi);
	fwrite(&Dm,sizeof(Dm),1,fi);
	fwrite(&sigm,sizeof(sigm),1,fi);
	fwrite(&tau,sizeof(tau),1,fi);
	fwrite(&params,sizeof(params),1,fi);
}
void ruslo_solver_fr1::deserialize(FILE *fi)
{
	fread(&n,sizeof(n),1,fi);
	fread(&m,sizeof(m),1,fi);
	fread(&Dm,sizeof(Dm),1,fi);
	fread(&sigm,sizeof(sigm),1,fi);
	fread(&tau,sizeof(tau),1,fi);
	fread(&params,sizeof(params),1,fi);
	local_n=n;
	local_m=m;
}
#ifdef USE_OPENCL
// additional - first n*m - piminus, other - prevC
#define fr1_ocl "\n\
#define alpha_ (*((__global double *)(class+%d)))\n\
#define q1 (*((__global double *)(class+%d)))\n\
#define sigm (*((__global double *)(class+%d)))\n\
#define tau (*((__global double *)(class+%d)))\n\
#define tq1 (*((__global double *)(class+%d)))\n\
#define ptplus (*((__global double *)(class+%d)))\n\
#define Ga (*((__global double *)(class+%d)))\n\
#define Dm (*((__global double *)(class+%d)))\n\
#define ptminus (*((__global double *)(class+%d)))\n\
#define h1 (2.0/(2.0*m+1.0)) \n\
#define h2 (q1/n)\n\
#define fi(i) (((i)-1.0)*h1)\n\
#define psi(j) (((j)-1.5)*h2)\n\
#define A(i,j) ((sqv(fi(i),psi(j),class)/h1)*((Dm/h1)-0.5))\n\
#define S(i,j) ((sqv(fi(i),psi(j),class)/h1)*((Dm/h1)+0.5))\n\
double sqv(double f,double p,__global char *class)\n\
{\n\
   double s1=sin(M_PI*p/(2.0*q1));\n\
   double s2=sinh(M_PI*(f-1)/(2.0*q1));\n\
   return s1*s1+s2*s2;\n\
}\n\
#define B(i,j) ((sigm/tau)+(2.0*sigm*tq1/(ptplus*Ga))+A(i,j)+S(i,j)) \n\
#define F0(i,j,step) ((-(sigm/tau))*C0[(i-2-offset_y)*local_n+(j-2-offset_x)]+(2.0*sigm*tq1/(ptplus*Ga))*(prevC[local_n*local_m*(1+step)+(i-2-offset_y)*local_n+(j-2-offset_x)]-2.0*C0[(i-2-offset_y)*local_n+(j-2-offset_x)]))\n\
double F(int i,int j,int step,__global double *C0,__global double *prevC,double s,__global char *class) \n\
{\n\
	double r=F0(i,j,step);\n\
    s*=ptminus*2.0*sigm*tq1/(tau*tau*Ga);\n\
	return r+s;\n\
}\n\
#define P_(i,j) (sqv(fi(i),psi(j),class)*Dm/(h2*h2)) \n\
#define R(i,j) (sqv(fi(i),psi(j),class)*Dm/(h2*h2))\n\
#define Q(i,j) ((sigm/tau)+(2.0*sigm*tq1/(ptplus*Ga))+P_(i,j)+R(i,j))\n\
#define O0(i,j,step) (-(sigm/tau)*C1[(i-2-offset_y)*local_n+(j-2-offset_x)]+(2.0*sigm*tq1/(ptplus*Ga))*(prevC[local_n*local_m*(1+step)+(i-2-offset_y)*local_n+(j-2-offset_x)]-2.0*C1[(i-2-offset_y)*local_n+(j-2-offset_x)]))\n\
double O(int i,int j,int step,__global double *C1,__global double *prevC,double s,__global char *class) \n\
{\n\
	double r=O0(i,j,step);\n\
    s*=ptminus*2.0*sigm*tq1/(tau*tau*Ga);\n\
	return r+s;\n\
}\n\
__kernel void Av(__global char *class,__global double *C0,__global double *C1,__global double *alpha,__global double *beta,int f,int l,__global double *prevC,int step)\n\
{\n\
   int row=get_global_id(0)+f;\n\
   int j; \n\
   int n_minus=2; \n\
   if ((offset_x+l)!=n) n_minus=1; \n\
   for (j=0;j<local_m;j++)\n\
	  prevC[local_n*local_m*(1+2*step)+j*local_n+row]=C1[j*local_n+row];\n\
   if (row<=l-n_minus) {\n\
		if (row==f)\n\
			prevC[2*step]=pow((double)(2*step),1.0-alpha_);\n\
		if (offset_y==0) \n\
			alpha[0*local_n+row]=0; \n\
		for (j=1;j<=local_m-1;j++) \n\
		  alpha[j*local_n+row]=A(offset_y+j+1,offset_x+row+2)/(B(offset_y+j+1,offset_x+row+2)-alpha[(j-1)*local_n+row]*S(offset_y+j+1,offset_x+row+2)); \n\
	}\n\
}\n\
__kernel void Ah(__global char *class,__global double *C0,__global double *C1,__global double *alpha,__global double *beta,int f,int l,__global double *prevC,int step)\n\
{\n\
   int row=get_global_id(0)+f;\n\
   int i;\n\
   int m_minus=2;\n\
   if ((offset_y+l)!=m) m_minus=1;\n\
   for (i=0;i<local_n;i++)\n\
      prevC[local_n*local_m*(1+2*step+1)+row*local_n+i]=C0[row*local_n+i];\n\
   if (row<=l-m_minus) {\n\
	   if (row==f)\n\
			prevC[2*step+1]=pow((double)(2*step+1),1.0-alpha_);\n\
	   if (offset_x==0) \n\
			alpha[row*local_n+0]=1; \n\
	   for (i=1;i<=local_n-1;i++)\n\
			alpha[row*local_n+i]=P_(offset_y+row+2,offset_x+i+1)/(Q(offset_y+row+2,offset_x+i+1)-alpha[row*local_n+(i-1)]*R(offset_y+row+2,offset_x+i+1));\n\
   }\n\
}\n"
#ifdef OCL_BLOCKED_SUM
#define fr1_ocl_B "\
__kernel void Bv(__global char *class,__global double *C0,__global double *C1,__global double *alpha,__global double *beta,int f,int l,__global double *prevC,int step)\n\
{\n\
   int row=get_group_id(0)+f;\n\
   int id=get_local_id(0);\n\
   __local double ps[16]; \n\
   double s;\n\
   int j; \n\
   int n_minus=2;\n\
   if ((offset_x+l)!=n) n_minus=1;\n\
   if (row<=l-n_minus) {\n\
   if (id==0) \n\
   {\n\
		if (offset_y==0)\n\
			beta[0*local_n+row]=1;\n\
   }\n\
   for (j=1;j<=local_m-1;j++)\n\
   {\n\
      s=0.0;\n\
	  int jj=(((signed)(step*2+1)-3)/2)+1;\n\
	  for (int ss=0;ss<1+(jj/16);ss++) \n\
	  {\n\
		ps[id]=0.0;\n\
		if ((id+ss*16)<jj) \n\
		ps[id]=(prevC[jj-(id+ss*16)+1]-prevC[jj-(id+ss*16)])*(prevC[local_n*local_m*(1+(2*((id+ss*16)+1)-1)-1)+(j-1)*local_n+row]-\n\
							  2.0*prevC[local_n*local_m*(1+(2*((id+ss*16)+1)-1))+(j-1)*local_n+row]+\n\
								prevC[local_n*local_m*(1+(2*((id+ss*16)+1)-1)+1)+(j-1)*local_n+row]);\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		if (id==0) \n\
			for (int kk=0;kk<16;kk++) \n\
				s+=ps[kk];\n\
	  }\n\
	  if (id==0) \n\
	    beta[j*local_n+row]=(alpha[j*local_n+row]/A(offset_y+j+1,offset_x+row+2))*(S(offset_y+j+1,offset_x+row+2)*beta[(j-1)*local_n+row]-F(offset_y+j+1,offset_x+row+2,(2*step),C0,prevC,s,class));\n\
   }\n\
   }\n\
}\n\
__kernel void Bh(__global char *class,__global double *C0,__global double *C1,__global double *alpha,__global double *beta,int f,int l,__global double *prevC,int step)\n\
{\n\
   int row=get_group_id(0)+f;\n\
   int id=get_local_id(0);\n\
   __local double ps[16]; \n\
   double s;\n\
   int i; \n\
   int m_minus=2;\n\
   if ((offset_y+l)!=m) m_minus=1;\n\
   if (row<=l-m_minus) {\n\
   if (id==0) \n\
   {\n\
		if (offset_x==0)\n\
			beta[row*local_n+0]=0;\n\
   }\n\
   for (i=1;i<=local_n-1;i++)\n\
   {\n\
      s=0.0;\n\
      int jj=(((signed)(step*2+2)-4)/2)+1;\n\
	  for (int ss=0;ss<1+(jj/16);ss++) \n\
	  {\n\
		ps[id]=0.0;\n\
		if ((id+ss*16)<jj) \n\
		ps[id]=(prevC[jj-(id+ss*16)+1]-prevC[jj-(id+ss*16)])*(prevC[local_n*local_m*(1+2*((id+ss*16)+1)-1)+row*local_n+(i-1)]-\n\
				2.0*prevC[local_n*local_m*(1+2*((id+ss*16)+1))+row*local_n+(i-1)]+\n\
					prevC[local_n*local_m*(1+2*((id+ss*16)+1)+1)+row*local_n+(i-1)]);\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		if (id==0) \n\
			for (int kk=0;kk<16;kk++) \n\
				s+=ps[kk];\n\
	  }\n\
	  if (id==0) \n\
	      beta[row*local_n+i]=(alpha[row*local_n+i]/P_(offset_y+row+2,offset_x+i+1))*(R(offset_y+row+2,offset_x+i+1)*beta[row*local_n+i-1]-O(offset_y+row+2,offset_x+i+1,(2*step+1),C1,prevC,s,class));\n\
   }\n\
   }\n\
}\n"
#else
#define fr1_ocl_B "\
__kernel void Bv(__global char *class,__global double *C0,__global double *C1,__global double *alpha,__global double *beta,int f,int l,__global double *prevC,int step)\n\
{\n\
   int row=get_group_id(0)+f;\n\
   double s;\n\
   int j; \n\
   int n_minus=2;\n\
   if ((offset_x+l)!=n) n_minus=1;\n\
   if (row<=l-n_minus) {\n\
   if (offset_y==0)\n\
	 beta[0*local_n+row]=1;\n\
   for (j=1;j<=local_m-1;j++)\n\
   {\n\
      s=0.0;\n\
	  int jj=(((signed)(step*2+1)-3)/2)+1;\n\
	  for (int ss=0;ss<jj;ss++) \n\
		s+=(prevC[jj-ss+1]-prevC[jj-ss])*(prevC[local_n*local_m*(1+(2*(ss+1)-1)-1)+(j-1)*local_n+row]-\n\
							  2.0*prevC[local_n*local_m*(1+(2*(ss+1)-1))+(j-1)*local_n+row]+\n\
								prevC[local_n*local_m*(1+(2*(ss+1)-1)+1)+(j-1)*local_n+row]);\n\
	  beta[j*local_n+row]=(alpha[j*local_n+row]/A(offset_y+j+1,offset_x+row+2))*(S(offset_y+j+1,offset_x+row+2)*beta[(j-1)*local_n+row]-F(offset_y+j+1,offset_x+row+2,(2*step),C0,prevC,s,class));\n\
   }\n\
   }\n\
}\n\
__kernel void Bh(__global char *class,__global double *C0,__global double *C1,__global double *alpha,__global double *beta,int f,int l,__global double *prevC,int step)\n\
{\n\
   int row=get_group_id(0)+f;\n\
   double s;\n\
   int i; \n\
   int m_minus=2;\n\
   if ((offset_y+l)!=m) m_minus=1;\n\
   if (row<=l-m_minus) {\n\
   if (offset_x==0)\n\
	 beta[row*local_n+0]=0;\n\
   for (i=1;i<=local_n-1;i++)\n\
   {\n\
      s=0.0;\n\
      int jj=(((signed)(step*2+2)-4)/2)+1;\n\
	  for (int ss=0;ss<jj;ss++) \n\
		s+=(prevC[jj-ss+1]-prevC[jj-ss])*(prevC[local_n*local_m*(1+2*(ss+1)-1)+row*local_n+(i-1)]-\n\
				2.0*prevC[local_n*local_m*(1+2*(ss+1))+row*local_n+(i-1)]+\n\
					prevC[local_n*local_m*(1+2*(ss+1)+1)+row*local_n+(i-1)]);\n\
      beta[row*local_n+i]=(alpha[row*local_n+i]/P_(offset_y+row+2,offset_x+i+1))*(R(offset_y+row+2,offset_x+i+1)*beta[row*local_n+i-1]-O(offset_y+row+2,offset_x+i+1,(2*step+1),C1,prevC,s,class));\n\
   }\n\
   }\n\
}\n"
#endif
char *ruslo_solver_fr1::get_ocl_prog_text()
{	
	char *base=new char[2*strlen(l1d_ocl_base)];
	char *frc=new char[2*strlen(fr1_ocl)];
	sprintf(frc,fr1_ocl,((char *)&this->params.alpha)-(char *)this,
							((char *)&this->params.q1)-(char *)this,
							((char *)&this->sigm)-(char *)this,
							((char *)&this->tau)-(char *)this,
							((char *)&this->params.tq1)-(char *)this,
							((char *)&this->ptplus)-(char *)this,
							((char *)&this->Ga)-(char *)this,
							((char *)&this->Dm)-(char *)this,
							((char *)&this->ptminus)-(char *)this);
	sprintf(base,l1d_ocl_base,((char *)&this->offset_x)-(char *)this,
								  ((char *)&this->offset_y)-(char *)this,
								  ((char *)&this->local_n)-(char *)this,
								  ((char *)&this->local_m)-(char *)this,
								  ((char *)&this->n)-(char *)this,
								  ((char *)&this->m)-(char *)this);
	char *fr1_ocl_code=new char[strlen(frc)+strlen(base)+strlen(fr1_ocl_B)];
	strcpy(fr1_ocl_code,base);
	strcat(fr1_ocl_code,frc);
	strcat(fr1_ocl_code,fr1_ocl_B);
	delete [] base;
	delete [] frc;
	return fr1_ocl_code;
}
#endif
