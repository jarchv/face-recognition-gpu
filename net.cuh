#ifndef net_cuh
#define net_cuh

#include "tools.cuh"

float** getW(int n, int m);
float*  getBias(int n, float init);
float*  getPred(float *input, int isize, int osize, float **nn_W, float *nn_b);

__device__ void softMax(int classes);
__device__ void crossEntropy(float* labels, int classes, int inputId, int tid);
__device__ void maxIndex(int size);

__global__ void gpu_train( 
    float* dev_Y, 
    int idxInput,
    int nCPs, 
    int classes,   
    float* dev_flatW,
    float* dev_b,
    float* dev_labels,
    float lr,
    int epoch);

__global__ void gpu_test(
    float* dev_Y, 
    int idxInput,
    int nCPs, 
    int classes,   
    float* dev_flatW,
    float* dev_b,
    float* dev_labels,
    float lr    
);

#endif