#ifndef tools_cuh
#define tools_cuh

#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define PATH "faces/"

int maxIndx(float *pred_in, int size);
float norm(float *, int);
void VecNormalizer(float *, int);
void readImages(float**, int , int, int, int);
void getS(float**, float**, float**, float**, float*, int, int);
void getW(float **A, float **B, float **W, int m, int n);
void showImage(float *I, int w, int h, int scale);
float *Reconstructor(float *Img, float **Wk, int m, int k);
void getProyection(float *Img, float **Wk, int m, int k, float* Y);

#endif