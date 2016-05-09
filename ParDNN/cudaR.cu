/*   Copyright: GTC2016, Unblock Deep Neural Network Performance Limiter by CUDA.
 *   Author: Peng Zhao, Nvidia, ParallelR.com
 */

// CUDA headfile
#include "cuda_runtime.h"
#include "cublas_v2.h"
// Basic C
#include <stdlib.h>
#include <stdio.h>
// R library
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>


#ifndef DEBUG
#define DEBUG 0
#endif


// treat it as C code
extern "C" {
    SEXP gemm_cuda(SEXP A, SEXP B, SEXP transA, SEXP transB, SEXP devID);
    SEXP pmax_cuda(SEXP A, SEXP threshold, SEXP devID);
}


// CUDA: simple implementation of pmax 
__global__ void pmax_kernel(double *A, const int M, const int N, const double threshold){
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if(tid<M*N){
      A[tid] = (A[tid] > threshold)?A[tid]:0;
   }
   return;
}


// GEMM cuda call by .CAll format and simplified for DNN
SEXP gemm_cuda(SEXP A, SEXP B, SEXP transA, SEXP transB, SEXP devID)
{


   double alpha = 1.0;
   double beta  = 0.0;

   // GPU memory allocation
   double *A_host = NULL, *B_host = NULL;
   double *A_d    = NULL, *B_d    = NULL, *Rval_d = NULL;
   // matrix dimension
   int     m, n, k;
   int     mt, nt, kt, lda, ldb;
   // transform
   int     tA, tB, gpuID;
  
   // R to C 
   A_host = REAL(A);
   B_host = REAL(B);
   SEXP RdimA = getAttrib(A, R_DimSymbol);
   SEXP RdimB = getAttrib(B, R_DimSymbol);
   // original shape
   m   = INTEGER(RdimA)[0];
   k   = INTEGER(RdimA)[1];
   n   = INTEGER(RdimB)[1];
   // transposed shape
   mt  = m;
   nt  = n;
   kt  = k;
   lda = INTEGER(RdimA)[0];
   ldb = INTEGER(RdimB)[0];

   tA  = INTEGER(transA)[0];
   tB  = INTEGER(transB)[0];

   gpuID = INTEGER(devID)[0];

   // Note that cublas follows fortran order.
   cublasOperation_t cuTransA = CUBLAS_OP_N;
   cublasOperation_t cuTransB = CUBLAS_OP_N;

   if(tA == 1) {
       cuTransA = CUBLAS_OP_T;
       mt = k;
       kt = m;
   }

   if(tB == 1) {
       cuTransB = CUBLAS_OP_T;
       nt = INTEGER(RdimB)[0];
       kt = INTEGER(RdimB)[1];
   }

   // set GPU ID
   cudaSetDevice(gpuID);
   cublasHandle_t handle;
   cublasCreate(&handle);

   SEXP Rval;
   PROTECT(Rval = allocVector(REALSXP, mt*nt));

   // Memory allocation in GPU
   cudaMalloc(&A_d,  mt*kt*sizeof(double));
   if(NULL == A_d) {
      printf("\nNo RAM space in GPU!\n");
      goto FREE_RESOURCE;
   }
   
   cudaMalloc(&B_d,  kt*nt*sizeof(double));
   if(NULL == B_d) {
      printf("\nNo RAM space in GPU!\n");
      goto FREE_RESOURCE;
   }

   cudaMalloc(&Rval_d,  mt*nt*sizeof(double));
   if(NULL == Rval_d) {
      printf("\nNo RAM space in GPU!\n");
      goto FREE_RESOURCE;
   }

   // memory copy
   cudaMemcpy(A_d, A_host, mt*kt*sizeof(double), cudaMemcpyHostToDevice); 
   cudaMemcpy(B_d, B_host, kt*nt*sizeof(double), cudaMemcpyHostToDevice); 

   // cuBLAS: double precision matrix multiplication, DGEMM
   cublasDgemm(handle, cuTransA, cuTransB, mt, nt, kt, &alpha, A_d, lda, B_d, ldb, &beta, Rval_d, mt);
   cudaMemcpy(REAL(Rval), Rval_d, mt*nt*sizeof(double), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();

FREE_RESOURCE:  
   cublasDestroy(handle);
   if(A_d) {cudaFree(A_d); A_d=NULL;}
   if(B_d) {cudaFree(B_d); B_d=NULL;}
   if(Rval_d) {cudaFree(Rval_d); Rval_d=NULL;}

   UNPROTECT(1);
   return Rval;

}


// Specified for DNN by .CAll format
SEXP pmax_cuda(SEXP A, SEXP threshold, SEXP devID)
{
   // data structure for GPU
   double *A_host = NULL;
   double *A_d = NULL;
   double gw = 0;
   int    mm = 0, nn = 0;
   int    gpuID = 0;
  
   // data transfer from R to C by pointers
   A_host = REAL(A);
   SEXP Rdim = getAttrib(A, R_DimSymbol);
   mm   = INTEGER(Rdim)[0];
   nn   = INTEGER(Rdim)[1];
   gw   = REAL(threshold)[0];
   gpuID = INTEGER(devID)[0];

   // for multiple GPU case 
   cudaSetDevice(gpuID);
   
   // return value, allocated in C and can be used in R directly
   SEXP Rval;
   PROTECT(Rval = allocVector(REALSXP, mm*nn));

   // GPU memory allocation
   cudaMalloc(&A_d,  mm*nn*sizeof(double));
   if(NULL == A_d) {
      printf("\nNo RAM space in GPU!\n");
      UNPROTECT(1);
      return R_NilValue;
   }
   
   // memory copy from CPU to GPU
   cudaMemcpy(A_d, A_host, mm*nn*sizeof(double), cudaMemcpyHostToDevice); 
   
   // CUDA: pmax, really computation parts
   pmax_kernel<<<(mm*nn-1)/512+1, 512>>>(A_d, mm, nn, gw);
   cudaMemcpy(REAL(Rval), A_d, mm*nn*sizeof(double), cudaMemcpyDeviceToHost); 
   cudaDeviceSynchronize();

   // Free unused memory of GPU
   if(A_d) {cudaFree(A_d); A_d=NULL;}

   UNPROTECT(1);
   return Rval;
}
