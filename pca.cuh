#ifndef pca_cuh
#define pca_cuh


#include "tools.cuh"

void    copyV(float*, float*, int);
void    copyM(float**, float**, int);
void    matmul(float**, float*, float*, int);
void    updateAA(float**, float, float*, int);
void    copy2Mat(float **V, float *Vi,int col, int n);
float   dotV(float *V1, float *V2, float n);
void    vectorOp(float *A, float *B,int n, float op);
float   normdiff(float *A, float *B, int n);
void    powerMethod(float **A, float **eigenVec, float *eigenVal, int n, float tol);
float*  getProyection(float *Img, float **Wk, int m, int k);

#endif