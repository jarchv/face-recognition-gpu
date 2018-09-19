#ifndef pca_cuh
#define pca_cuh


#include "tools.cuh"

void    copyV(float *v1, float *v2, int n);
void    copyM(float **m1, float **m2, int n);
void    matmul(float **X, float *B, float *C, int n);
void    updateAA(float **AA, float lambda, float *xnew, int n);
void    copy2Mat(float **V, float *Vi,int col, int n);
void    vectorOp(float *A, float *B,int n, float op);
void    powerMethod(float **A, float **eigenVec, float *eigenVal, int n, float tol);
float   dotV(float *V1, float *V2, float n);
float   normdiff(float *A, float *B, int n);
float*  getProyection(float *Img, float **Wk, int m, int k);

#endif