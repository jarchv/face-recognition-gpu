
#include "net.cuh"

__device__ float    pred[40];
__device__ float  	output[40];
__device__ float 	loss;
__device__ int		label_pred;
__device__ float    acc = 0.0;

float **getW(int n, int m){
	float **wtemp = new float*[n];
	for(int in=0; in < n; in++){
		wtemp[in] = new float[m];
	}
	for(int in=0; in < n; in++){
		for(int im=0; im < m; im++){
			wtemp[in][im] = (float)(rand()%100);
		}
		VecNormalizer(wtemp[in], m);
	}
	return wtemp;
}

float *getBias(int n, float init){
	float *btemp = new float[n];
	for (int in = 0; in < n; ++in)
		btemp[in] = init;
	return btemp;
}

float *getPred(float *input, int isize, int osize, float **nn_W, float *nn_b){
	float temp;
	float *prediction = new float[osize];

	for(int i=0; i < osize; i++){
		temp = 0.0;
		for(int j=0; j < isize; j++)
			temp += nn_W[i][j]*input[j];

		temp += nn_b[i];
		prediction[i] = temp;
	}

	return prediction;
}

__device__ void softMax(int classes){
    float sum_temp = 0.0;
    for (int is = 0; is < 40; is++){
		output[is] = exp(pred[is]);
		sum_temp  += output[is];
	}

	for (int is = 0; is < classes; is++)
		output[is] = output[is]/sum_temp;
}

__device__ void crossEntropy(float* labels, int classes, int inputId, int tid, int epoch){
	float temp = 0.0;
	for (int is = 0; is < classes; is++)
		temp -= labels[inputId*classes + is]*log(output[is]); 

	if (tid == 0)
		loss += temp/((float)classes * 320);
		if (inputId == 397)
			printf("epoch %3d -> loss = %f\n", epoch, loss);  
}

__device__ void backProp(float* dev_flatW, float* dev_b,float* dev_Y, int nCPs, float lr,int inputId, int tid, float delta)
{
	for (int ivec = 0; ivec < nCPs; ivec++)
		dev_flatW[tid * nCPs + ivec] -= delta * dev_Y[inputId*nCPs + ivec] * lr;
	dev_b[tid]   -= delta * lr; 
}

__global__ void gpu_train( 
                float* 	dev_Y, 
                int 	idxInput,
                int 	nCPs, 
                int 	classes,   
                float* 	dev_flatW,
                float* 	dev_b,
                float* 	dev_labels,
				float 	lr,
				int 	epoch)

{
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    float temp;
    if (tid < classes){
		if (tid == 0 && idxInput == 0)
			loss = 0.0;
        temp = 0.0;
        for (int idx = 0; idx < nCPs; idx++){
            temp += dev_flatW[tid * nCPs + idx] * dev_Y[idxInput*nCPs + idx];
        }
        temp += dev_b[tid];
        pred[tid] = temp;
		__syncthreads();
		if (tid == 0)
        	softMax(classes);
		__syncthreads();
		if (tid == 0)
        	crossEntropy(dev_labels, classes, idxInput, tid, epoch);
		
		float derv = -dev_labels[idxInput*classes + tid]*(1.0-output[tid]);
		backProp(dev_flatW, dev_b, dev_Y, nCPs, lr, idxInput, tid, derv);
		/*
		for (int ivec = 0; ivec < nCPs; ivec++)
        {
            dev_flatW[tid * nCPs + ivec] -= derv * dev_Y[idxInput*nCPs + ivec] * lr;
        }

		dev_b[tid]   -= derv * lr; 
		*/
		__syncthreads();
    }
}

__device__ void maxIndex(int size){
	float temp = pred[0];
	int indx = 0;
	for (int i = 1; i < size; i++)
	{
		if (pred[i] > temp){
			indx = i;
			temp = pred[i];
		}
	}
	label_pred = indx;
}
__global__ void gpu_test( 
				float* 	dev_Y, 
				int 	idxInput,
				int 	nCPs, 
				int 	classes,   
				float* 	dev_flatW,
				float* 	dev_b,
				float* 	dev_labels,
				float 	lr)

{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    float temp;
    if (tid < classes){
		if (tid == 0 && idxInput == 0)
			loss = 0.0;
        temp = 0.0;
        for (int idx = 0; idx < nCPs; idx++){
            temp += dev_flatW[tid * nCPs + idx] * dev_Y[idxInput*nCPs + idx];
        }
        temp += dev_b[tid];
        pred[tid] = temp;
		__syncthreads();

		if (tid == 0)
        	softMax(classes);
		__syncthreads();
			
		if (tid == 0){
			maxIndex(classes);
			if (((idxInput/10) == label_pred)){
				acc += 1.0;
				if (idxInput == 399)
					printf("\nAccuracy : %f%%\n", 100.0*((float)acc)/80.0);
			}
		}
	}	
}