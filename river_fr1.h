/* Author:Vsevolod Bohaienko */
/* —¿–œŒ  3D river pollution modelling */
#include <math.h>
#include <stdio.h>
#include <vector>
#include "local1D_iface.h"
using namespace std;
typedef struct
{
	double q1;
	double tq1;
	double alpha;
	double a;
	double a1;
	double H;
}  ruslo_solver_fr1_params;
class ruslo_solver_fr1: public local1D_solver
{
public:
	int task_id() {return 3;}
	// for partial summing
	double *ss;
	// additional distributed data
	vector<double **> prevC;
	// in storage: i=[0,m-1],j=[0,n-1] in coef functions i=[2,m+1],j=[2,n+1]
	// returns ||(vx,vy)||^2
	double inline sqv(double f,double p);
	double fi_(int i);
	double psi_(int j);
	// transformations from complex into physical coordinates
	double X(double f,double p);
	double Y(double f,double p);
	// parameters
	ruslo_solver_fr1_params params;	
	double L;
	double sigm;
	double Dm;
	double tau;
	// mulithreading 
	int wait_var1,wait_var2;
	// precalculations
	double Ga,ptplus,ptminus,pt,twoaminus;
	vector<double> piminus,piminus_half;
	// precalculations for testing problem2
	double Gb,a0,hb;
	// methods
#ifndef CACHE_OPT
	double F(int,int);
	double O(int,int);
#endif
	void calc_alpha1_CPU(int f,int l);
	void calc_beta1_CPU(int f,int l);
	void calc_alpha2_CPU(int f,int l);
	void calc_beta2_CPU(int f,int l);
	void calc_alpha1(int f,int l);
	void calc_beta1(int f,int l);
	void calc_alpha2(int f,int l);
	void calc_beta2(int f,int l);
	// initial conditions on C
	void initial_C();
	void init_process(int no_alloc=0);
    void send_parameters(interprocess_io *io,int msock,int update=0);
	void recv_parameters(interprocess_io *io,int msock,int update=0);
	void copy_parameters(local1D_solver *slv,int update=0);
	// serializing
	void serialize(FILE *fi);
	void deserialize(FILE *fi);
	ruslo_solver_fr1();
	~ruslo_solver_fr1();
#ifdef USE_OPENCL
	char *get_ocl_prog_text();
	void calc_C1(int f,int l);
	void calc_C2(int f,int l);
	int gpu_prevC_size;
#endif
};
