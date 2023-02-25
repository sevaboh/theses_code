#include <math.h>
#include <string.h>
#include <stdio.h>
#include <CL/opencl.h>
double log2(double x) {return log(x)/log(2.0);}
extern int double_ext;

char *pcr_small_kernels="\n\
/*\n\
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.\n\
 *\n\
 * Please refer to the NVIDIA end user license agreement (EULA) associated\n\
 * with this source code for terms and conditions that govern your use of\n\
 * this software. Any use, reproduction, disclosure, or distribution of\n\
 * this software and related documentation outside the terms of the EULA\n\
 * is strictly prohibited.\n\
 *\n\
 */\n\
 \n\
 /*\n\
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.\n\
 * \n\
 * Tridiagonal solvers.\n\
 * Device code for parallel cyclic reduction (PCR).\n\
 *\n\
 * Original CUDA kernels: UC Davis, Yao Zhang & John Owens, 2009\n\
 * \n\
 * NVIDIA, Nikolai Sakharnykh, 2009\n\
 */\n\
\n\
#pragma OPENCL EXTENSION %s : enable \n\n\
//#define NATIVE_DIVIDE\n\
\n\
__kernel void pcr_small_systems_kernel(__global double *a_d, __global double *b_d, __global double *c_d, __global double *d_d, __global double *x_d, \n\
									   __local double *shared, int system_size, int num_systems, int iterations)\n\
{\n\
    int thid = get_local_id(0);\n\
    int blid = get_group_id(0);\n\
\n\
	int delta = 1;\n\
\n\
	__local double* a = shared;\n\
	__local double* b = &a[system_size+1];\n\
	__local double* c = &b[system_size+1];\n\
	__local double* d = &c[system_size+1];\n\
	__local double* x = &d[system_size+1];\n\
\n\
	a[thid] = a_d[thid + blid * system_size];\n\
	b[thid] = b_d[thid + blid * system_size];\n\
	c[thid] = c_d[thid + blid * system_size];\n\
	d[thid] = d_d[thid + blid * system_size];\n\
  \n\
	double aNew, bNew, cNew, dNew;\n\
  \n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
	// parallel cyclic reduction\n\
	for (int j = 0; j < iterations; j++)\n\
	{\n\
		int i = thid;\n\
\n\
		if(i < delta)\n\
		{\n\
#ifndef NATIVE_DIVIDE\n\
			double tmp2 = c[i] / b[i+delta];\n\
#else\n\
			double tmp2 = native_divide(c[i], b[i+delta]);\n\
#endif\n\
			bNew = b[i] - a[i+delta] * tmp2;\n\
 			dNew = d[i] - d[i+delta] * tmp2;\n\
			aNew = 0;\n\
			cNew = -c[i+delta] * tmp2;	\n\
		}\n\
		else if((system_size-i-1) < delta)\n\
		{\n\
#ifndef NATIVE_DIVIDE\n\
			double tmp = a[i] / b[i-delta];\n\
#else\n\
			double tmp = native_divide(a[i], b[i-delta]);\n\
#endif\n\
			bNew = b[i] - c[i-delta] * tmp;\n\
			dNew = d[i] - d[i-delta] * tmp;\n\
			aNew = -a[i-delta] * tmp;\n\
			cNew = 0;			\n\
		}\n\
		else		    \n\
		{\n\
#ifndef NATIVE_DIVIDE\n\
			double tmp1 = a[i] / b[i-delta];\n\
			double tmp2 = c[i] / b[i+delta];\n\
#else\n\
			double tmp1 = native_divide(a[i], b[i-delta]);\n\
			double tmp2 = native_divide(c[i], b[i+delta]);\n\
#endif\n\
   			bNew = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;\n\
 			dNew = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;\n\
			aNew = -a[i-delta] * tmp1;\n\
			cNew = -c[i+delta] * tmp2;\n\
		}\n\
\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
        \n\
		b[i] = bNew;\n\
 		d[i] = dNew;\n\
		a[i] = aNew;\n\
		c[i] = cNew;	\n\
    \n\
		delta *= 2;\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
\n\
	if (thid < delta)\n\
	{\n\
		int addr1 = thid;\n\
		int addr2 = thid + delta;\n\
		double tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];\n\
#ifndef NATIVE_DIVIDE\n\
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;\n\
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;\n\
#else\n\
		x[addr1] = native_divide((b[addr2] * d[addr1] - c[addr1] * d[addr2]), tmp3);\n\
		x[addr2] = native_divide((d[addr2] * b[addr1] - d[addr1] * a[addr2]), tmp3);\n\
#endif\n\
	}\n\
    \n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
    \n\
    x_d[thid + blid * system_size] = x[thid];\n\
}\n\
__kernel void pcr_kernel1(__global double *a, __global double *b, __global double *c, __global double *d, __global double *a2, __global double *b2, __global double *c2, __global double *d2,\n\
									   int system_size, int delta)\n\
{\n\
    int thid = get_global_id(0);\n\
	double aNew, bNew, cNew, dNew;\n\
	int i = thid;\n\
\n\
	if(i < delta)\n\
	{\n\
#ifndef NATIVE_DIVIDE\n\
		double tmp2 = c[i] / b[i+delta];\n\
#else\n\
		double tmp2 = native_divide(c[i], b[i+delta]);\n\
#endif\n\
		bNew = b[i] - a[i+delta] * tmp2;\n\
 		dNew = d[i] - d[i+delta] * tmp2;\n\
		aNew = 0;\n\
		cNew = -c[i+delta] * tmp2;	\n\
	}\n\
	else if((system_size-i-1) < delta)\n\
	{\n\
#ifndef NATIVE_DIVIDE\n\
		double tmp = a[i] / b[i-delta];\n\
#else\n\
		double tmp = native_divide(a[i], b[i-delta]);\n\
#endif\n\
		bNew = b[i] - c[i-delta] * tmp;\n\
		dNew = d[i] - d[i-delta] * tmp;\n\
		aNew = -a[i-delta] * tmp;\n\
		cNew = 0;			\n\
	}\n\
	else		    \n\
	{\n\
#ifndef NATIVE_DIVIDE\n\
		double tmp1 = a[i] / b[i-delta];\n\
		double tmp2 = c[i] / b[i+delta];\n\
#else\n\
		double tmp1 = native_divide(a[i], b[i-delta]);\n\
		double tmp2 = native_divide(c[i], b[i+delta]);\n\
#endif\n\
   		bNew = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;\n\
 		dNew = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;\n\
		aNew = -a[i-delta] * tmp1;\n\
		cNew = -c[i+delta] * tmp2;\n\
	}\n\
\n\
	b2[i] = bNew;\n\
 	d2[i] = dNew;\n\
	a2[i] = aNew;\n\
	c2[i] = cNew;	\n\
}\n\
__kernel void pcr_kernel2(__global double *a, __global double *b, __global double *c, __global double *d, __global double *x, \n\
									  int system_size, int delta)\n\
{\n\
    int thid = get_global_id(0);\n\
	if (thid < delta)\n\
	{\n\
		int addr1 = thid;\n\
		int addr2 = thid + delta;\n\
		double tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];\n\
#ifndef NATIVE_DIVIDE\n\
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;\n\
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;\n\
#else\n\
		x[addr1] = native_divide((b[addr2] * d[addr1] - c[addr1] * d[addr2]), tmp3);\n\
		x[addr2] = native_divide((d[addr2] * b[addr1] - d[addr1] * a[addr2]), tmp3);\n\
#endif\n\
	}\n\
}\n\
\n\
__kernel void pcr_branch_free_kernel(__global double *a_d, __global double *b_d, __global double *c_d, __global double *d_d, __global double *x_d, \n\
									 __local double *shared, int system_size, int num_systems, int iterations)\n\
{\n\
	int thid = get_local_id(0);\n\
    int blid = get_group_id(0);\n\
\n\
	int delta = 1;\n\
\n\
	__local double* a = shared;\n\
	__local double* b = &a[system_size+1];\n\
	__local double* c = &b[system_size+1];\n\
	__local double* d = &c[system_size+1];\n\
	__local double* x = &d[system_size+1];\n\
\n\
	a[thid] = a_d[thid + blid * system_size];\n\
	b[thid] = b_d[thid + blid * system_size];\n\
	c[thid] = c_d[thid + blid * system_size];\n\
	d[thid] = d_d[thid + blid * system_size];\n\
  \n\
	double aNew, bNew, cNew, dNew;\n\
  \n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
	// parallel cyclic reduction\n\
	for (int j = 0; j < iterations; j++)\n\
	{\n\
		int i = thid;\n\
\n\
		int iRight = i+delta;\n\
		iRight = iRight & (system_size-1);\n\
\n\
		int iLeft = i-delta;\n\
		iLeft = iLeft & (system_size-1);\n\
\n\
#ifndef NATIVE_DIVIDE\n\
		double tmp1 = a[i] / b[iLeft];\n\
		double tmp2 = c[i] / b[iRight];\n\
#else\n\
		double tmp1 = native_divide(a[i], b[iLeft]);\n\
		double tmp2 = native_divide(c[i], b[iRight]);\n\
#endif\n\
\n\
		bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;\n\
		dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;\n\
		aNew = -a[iLeft] * tmp1;\n\
		cNew = -c[iRight] * tmp2;\n\
\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
        \n\
		b[i] = bNew;\n\
 		d[i] = dNew;\n\
		a[i] = aNew;\n\
		c[i] = cNew;	\n\
    \n\
	    delta *= 2;\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
\n\
	if (thid < delta)\n\
	{\n\
		int addr1 = thid;\n\
		int addr2 = thid + delta;\n\
		double tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];\n\
#ifndef NATIVE_DIVIDE\n\
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;\n\
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;\n\
#else\n\
		x[addr1] = native_divide((b[addr2] * d[addr1] - c[addr1] * d[addr2]), tmp3);\n\
		x[addr2] = native_divide((d[addr2] * b[addr1] - d[addr1] * a[addr2]), tmp3);\n\
#endif\n\
	}\n\
    \n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
    \n\
    x_d[thid + blid * system_size] = x[thid];\n\
}";
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 /*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 * 
 * Tridiagonal solvers.
 * Host code for parallel cyclic reduction (PCR).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */


#define MAX_GPU_COUNT 1
#define BENCH_ITERATIONS 0

const int pcrNumKernels = 2;

const char *pcrKernelNames[pcrNumKernels] = { 
	"pcr_small_systems_kernel",			// original version
	"pcr_branch_free_kernel",			// optimized branch-free version
};   

cl_kernel pcrKernel[MAX_GPU_COUNT];
cl_kernel pcrKernel_gl[2];

// small systems version
int runPcrKernels(cl_command_queue *cqCommandQue,int devCount, cl_mem *dev_a, cl_mem *dev_b, cl_mem *dev_c, cl_mem *dev_d, cl_mem *dev_x, int system_size, int *workSize)
{
	size_t szGlobalWorkSize[MAX_GPU_COUNT];
    size_t szLocalWorkSize[MAX_GPU_COUNT];
	cl_event GPUExecution[MAX_GPU_COUNT];
	cl_int errcode;

	int iterations = log2(system_size/2);

	for (int i = 0; i < devCount; i++)
	{
		int num_systems = workSize[i];

		// set kernel arguments
		errcode  = clSetKernelArg(pcrKernel[i], 0, sizeof(cl_mem), (void *) &dev_a[i]);
		errcode |= clSetKernelArg(pcrKernel[i], 1, sizeof(cl_mem), (void *) &dev_b[i]);
		errcode |= clSetKernelArg(pcrKernel[i], 2, sizeof(cl_mem), (void *) &dev_c[i]);
		errcode |= clSetKernelArg(pcrKernel[i], 3, sizeof(cl_mem), (void *) &dev_d[i]);
		errcode |= clSetKernelArg(pcrKernel[i], 4, sizeof(cl_mem), (void *) &dev_x[i]);
		errcode |= clSetKernelArg(pcrKernel[i], 5, (system_size+1)*5*sizeof(double), NULL);
		errcode |= clSetKernelArg(pcrKernel[i], 6, sizeof(int), &system_size);
		errcode |= clSetKernelArg(pcrKernel[i], 7, sizeof(int), &num_systems);
		errcode |= clSetKernelArg(pcrKernel[i], 8, sizeof(int), &iterations);
		if (errcode != CL_SUCCESS) return 0;

		// set execution parameters
		szLocalWorkSize[i] = system_size;
		szGlobalWorkSize[i] = num_systems * szLocalWorkSize[i];
	    
		// warm up
		errcode = clEnqueueNDRangeKernel(cqCommandQue[i], pcrKernel[i], 1, NULL, &szGlobalWorkSize[i], &szLocalWorkSize[i], 0, NULL, &GPUExecution[i]);
		clFlush(cqCommandQue[i]);
		if (errcode != CL_SUCCESS) return 0;
	}
	clWaitForEvents(devCount, GPUExecution);


	// run computations on GPUs in parallel
	double sum_time = 0.0;
	for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
	{
		for (int i = 0; i < devCount; i++)
		{
			errcode = clEnqueueNDRangeKernel(cqCommandQue[i], pcrKernel[i], 1, NULL, &szGlobalWorkSize[i], &szLocalWorkSize[i], 0, NULL, &GPUExecution[i]);
			clFlush(cqCommandQue[i]);
			if (errcode != CL_SUCCESS) return 0;
		}
		clWaitForEvents(devCount, GPUExecution);
	}
	return 1;
}
// version that uses global memory
int runPcrKernels_gl(cl_command_queue *cqCommandQue,int devCount, cl_mem *dev_a, cl_mem *dev_b, cl_mem *dev_c, cl_mem *dev_d, cl_mem *dev_x, int system_size, int *workSize, cl_mem *dev_a2, cl_mem *dev_b2, cl_mem *dev_c2, cl_mem *dev_d2)
{
	size_t szGlobalWorkSize[MAX_GPU_COUNT];
    size_t szLocalWorkSize[MAX_GPU_COUNT];
	cl_event GPUExecution[MAX_GPU_COUNT];
	cl_int errcode;
	int delta=1;
	int iterations = log2(system_size/2);

	int num_systems = workSize[0];

	// set kernel arguments
	errcode = clSetKernelArg(pcrKernel_gl[0], 8, sizeof(int), &system_size);
	if (errcode != CL_SUCCESS) return 0;

	// set execution parameters
	szLocalWorkSize[0] = 1;
	szGlobalWorkSize[0] = system_size;
	
	for (int j=0;j<iterations;j++)
	{
		errcode = clSetKernelArg(pcrKernel_gl[0], ((j&1)*4)+0, sizeof(cl_mem), (void *) &dev_a[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], ((j&1)*4)+1, sizeof(cl_mem), (void *) &dev_b[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], ((j&1)*4)+2, sizeof(cl_mem), (void *) &dev_c[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], ((j&1)*4)+3, sizeof(cl_mem), (void *) &dev_d[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], ((1-(j&1))*4)+0, sizeof(cl_mem), (void *) &dev_a2[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], ((1-(j&1))*4)+1, sizeof(cl_mem), (void *) &dev_b2[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], ((1-(j&1))*4)+2, sizeof(cl_mem), (void *) &dev_c2[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], ((1-(j&1))*4)+3, sizeof(cl_mem), (void *) &dev_d2[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[0], 9, sizeof(int), &delta);
		errcode |= clEnqueueNDRangeKernel(cqCommandQue[0], pcrKernel_gl[0], 1, NULL, &szGlobalWorkSize[0], &szLocalWorkSize[0], 0, NULL, &GPUExecution[0]);
		clFlush(cqCommandQue[0]);
		clWaitForEvents(devCount, GPUExecution);
		if (errcode != CL_SUCCESS) return 0;
		delta*=2;
	}
	if ((iterations&1)==0)
	{
		errcode = clSetKernelArg(pcrKernel_gl[1], 0, sizeof(cl_mem), (void *) &dev_a[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[1], 1, sizeof(cl_mem), (void *) &dev_b[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[1], 2, sizeof(cl_mem), (void *) &dev_c[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[1], 3, sizeof(cl_mem), (void *) &dev_d[0]);
	}
	else
	{
		errcode = clSetKernelArg(pcrKernel_gl[1], 0, sizeof(cl_mem), (void *) &dev_a2[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[1], 1, sizeof(cl_mem), (void *) &dev_b2[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[1], 2, sizeof(cl_mem), (void *) &dev_c2[0]);
		errcode |= clSetKernelArg(pcrKernel_gl[1], 3, sizeof(cl_mem), (void *) &dev_d2[0]);
	}
	errcode |= clSetKernelArg(pcrKernel_gl[1], 4, sizeof(cl_mem), (void *) &dev_x[0]);
	errcode |= clSetKernelArg(pcrKernel_gl[1], 5, sizeof(int), &system_size);
	errcode |= clSetKernelArg(pcrKernel_gl[1], 6, sizeof(int), &delta);
	errcode |= clEnqueueNDRangeKernel(cqCommandQue[0], pcrKernel_gl[1], 1, NULL, &szGlobalWorkSize[0], &szLocalWorkSize[0], 0, NULL, &GPUExecution[0]);
	clFlush(cqCommandQue[0]);
	clWaitForEvents(devCount, GPUExecution);
	if (errcode != CL_SUCCESS) return 0;
	return 1;
}
////////////////////////////////////////////////////////////////////////////
//  id - version of kernel
int pcr_small_systems_init(cl_context cxGPUContext,int id)
{
	cl_int errcode;
	size_t program_length=strlen(pcr_small_kernels);
	char *s = new char[2 * program_length];
	sprintf(s, pcr_small_kernels, ((double_ext == 0) ? "cl_amd_fp64" : "cl_khr_fp64"));
	program_length = strlen(s);
    // create program
    cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&s, &program_length, &errcode);
	if (errcode != CL_SUCCESS)
		return 0;
	// build program
    errcode = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	if (errcode != CL_SUCCESS)
		return 0;
	// create kernels
	pcrKernel[0] = clCreateKernel(cpProgram, pcrKernelNames[id], &errcode);
	if (errcode != CL_SUCCESS)
		return 0;
	pcrKernel_gl[0] = clCreateKernel(cpProgram, "pcr_kernel1", &errcode);
	if (errcode != CL_SUCCESS)
		return 0;
	pcrKernel_gl[1] = clCreateKernel(cpProgram, "pcr_kernel2", &errcode);
	if (errcode != CL_SUCCESS)
		return 0;
	return 1;
}
void pcr_solver(cl_mem A, cl_mem B, cl_mem C, cl_mem R, cl_mem X,int N,cl_command_queue *q,int ver,cl_mem A2, cl_mem B2, cl_mem C2, cl_mem R2)
{
	int ws = 1;
	if (ver==0)
		runPcrKernels(q, 1, &A, &B, &C, &R, &X, N, &ws);
	if (ver==1)
		runPcrKernels_gl(q, 1, &A, &B, &C, &R, &X, N, &ws,&A2,&B2,&C2,&R2);
}
