#include "tools.cuh"
#include "pca.cuh"
#include "net.cuh"

int main(int argc, char *argv[]){
    int         nfiles      = 40;      //  # folders
    int         nimages     = 10;      //  # imgs x folder

    float**     X;
    float**     Xm;
    float**     Xm_t;
    float**     S;
    float**     W;
    float*      muV;
    float*      temp;

    int         X_rows=nfiles*nimages;
    int         imgw  = 92;
    int         imgh  = 112;
    int         X_cols= imgh*imgw;   //  h = 112, w = 92

    X           = (float **)malloc(X_rows*sizeof(float *));
    Xm          = (float **)malloc(X_rows*sizeof(float *));
    Xm_t        = (float **)malloc(X_cols*sizeof(float *));
    S           = (float **)malloc(X_rows*sizeof(float *));
    muV         = (float  *)malloc(X_cols*sizeof(float  ));
    temp        = (float  *)malloc(X_cols*sizeof(float  ));
    W           = (float **)malloc(X_cols*sizeof(float *));

    for (int i = 0; i < X_rows; i++)
    {
        X[i]    = (float *)malloc(X_cols*sizeof(float));
        Xm[i]   = (float *)malloc(X_cols*sizeof(float));
        S[i]    = (float *)malloc(X_rows*sizeof(float));
    }

    for (int i = 0; i < X_cols; i++)
    {
        Xm_t[i] = (float *)malloc(X_rows*sizeof(float));
        W[i]    = (float *)malloc(X_rows*sizeof(float));
    }   
 
    readImages(X,nfiles,nimages,imgh,imgw);
    getS(S,X,Xm,Xm_t, muV, X_rows, X_cols);
    
    float tol = 1e-20;

    float** autoVec;
    float*  autoVal;

    autoVec     = (float **)malloc(X_rows*sizeof(float *));
    autoVal     = (float  *)malloc(X_rows*sizeof(float  ));

    for (int i = 0; i < X_rows; i++)
    {
        autoVec[i]    = (float *)malloc(X_rows*sizeof(float));
    }

    powerMethod(S, autoVec, autoVal, X_rows, tol);
    getW(Xm_t, autoVec, W, X_cols, X_rows);
    /*
    int nEigenVectors = 5;
    printf("Eigenvectors (1-%d):\n", nEigenVectors);
    printf("=============\n\n");
    for(int j=0; j < nEigenVectors; j++){
        for(int i=0; i< X_cols; i++){
            temp[i] = W[i][j];
        }
        printf("Vec[%2d] = -> Press ESC\n", j);
        showImage(temp, imgw, imgh, 4);
    }

    printf("\nReconstructor:\n");
    printf("==============\n");
    for(int ir=0; ir < X_rows; ir+=5){
        temp = Reconstructor(Xm[0], W, X_cols, ir);
        printf("\tk -> %d\n", ir);
        showImage(temp, imgw, imgh,4);   
    }
    */

    free(temp);
    int key = 300;
    printf("\nfeatures = %d\n", key);
    printf("==============\n");

    float **Y;
    Y = (float **)malloc(X_rows*sizeof(float*));
    for (int iy = 0; iy < X_rows; iy++)
    {
        Y[iy] = (float *)malloc(key*sizeof(float));
    }

    for (int i = 0; i < X_rows; i++){
        getProyection(Xm[i], W, X_cols, key, Y[i]);
    }

    float **nn_W      = getW(nfiles, X_cols);
    float  *nn_b      = getBias(nfiles, 0.0);
    

    float **labels     = (float **)malloc(X_rows*sizeof(float*));
    float  *output     = (float  *)malloc(nfiles*sizeof(float ));
    float* flat_labels = (float *)malloc(X_rows*nfiles*sizeof(float));
    for (int in=0; in < X_rows; in++){
        labels[in] = (float *)malloc(nfiles*sizeof(float));
    }

    printf("\nSetting labels...\n" );
    for (int in=0; in < X_rows; in++){
        for (int il = 0; il < nfiles; il++)
        {
            labels[in][il] = (float)((in / 10) == il); 
            flat_labels[in*nfiles + il] = (float)((in / 10) == il); 
        }
    } 

    float* dev_labels;
    cudaMalloc((void **)&dev_labels, X_rows*nfiles*sizeof(float));
    cudaMemcpy(dev_labels, flat_labels, X_rows*nfiles*sizeof(float),cudaMemcpyHostToDevice);

    float lr = 5e-1;

    float *flatY    = (float *)malloc(X_rows*key*sizeof(float   ));
    float *flatW    = (float *)malloc(nfiles*X_cols*sizeof(float));

    for (int irow  = 0; irow  < X_rows; irow++)
    {
        for (int jkey = 0; jkey < key; jkey++)
        {
            flatY[irow * key + jkey] = Y[irow][jkey];
        }
    }

    for (int ifile = 0; ifile < nfiles; ifile++)
    {
        for (int j = 0; j < X_cols; j++)
        {
            flatW[ifile*X_cols + j] = nn_W[ifile][j];
        }
    }

    float *dev_flatW;
    float *dev_b;
    float *dev_Y;

    cudaMalloc((void **)&dev_flatW, nfiles*X_cols*sizeof(float));
    cudaMalloc((void **)&dev_b    , nfiles       *sizeof(float));
    cudaMalloc((void **)&dev_Y    , X_rows*key   *sizeof(float));

    cudaMemcpy(dev_flatW, flatW, nfiles*X_cols*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b    , nn_b ,        nfiles*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Y    , flatY,    X_rows*key*sizeof(float),cudaMemcpyHostToDevice);

    clock_t         start;
    clock_t         stop;

    printf("\nTraining:\n");
    printf("========\n");
    start   = clock();
    for (int epoch = 1; epoch <= 100; epoch++)
    {
        for (int ifile = 0; ifile < X_rows; ifile++)
        {
            gpu_train<<<1, 64>>>(   dev_Y,
                                    ifile,
                                    key,  
                                    nfiles,  
                                    dev_flatW, 
                                    dev_b, 
                                    dev_labels, 
                                    lr,
                                    epoch);
        }
    }
    stop   = clock();
    double elapsed = ((double)(stop-start))/CLOCKS_PER_SEC;
    printf("\nTraining : %5lf \n", elapsed);

    for (int ifile = 0; ifile < X_rows; ifile++)
    {
        gpu_test<<<1, 64>>>(dev_Y,
            ifile,
            key,  
            nfiles,  
            dev_flatW, 
            dev_b, 
            dev_labels, 
            lr);
    }

    /*
    cudaMemcpy(flatW, dev_flatW, nfiles*X_cols*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(nn_b , dev_b    ,        nfiles*sizeof(float),cudaMemcpyDeviceToHost);

    
    for (int ifile = 0; ifile < nfiles; ifile++)
    {
        for (int j = 0; j < X_cols; j++)
        {
            nn_W[ifile][j] = flatW[ifile*X_cols + j];
        }
    }

    float *pred_test       = new float[nfiles];
    int rand_indx;
    float valid=0;
    for (int i = 0; i < X_rows; i++){
        rand_indx = rand()%X_rows;
        pred_test = getPred(Y[rand_indx], key, nfiles, nn_W, nn_b);

        printf("label[%3d]\t= %2d\n", rand_indx/10, maxIndx(pred_test, nfiles));
        if ((rand_indx/10) == maxIndx(pred_test, nfiles)){
            valid++;
        }
        //showImage(X[rand_indx], imgw, imgh,4);
    }

    float n = (float)X_rows;
    printf("\nAccuracy : %f%%\n", 100*valid/n);
    */
/*
_________________________________________________________________________________
*/
    cudaFree(dev_flatW);
    cudaFree(dev_b    );
    for (int i = 0; i < X_rows; i++)
    {
        free(X [i]      );
        free(Xm[i]      );
        free(S [i]      );
        free(autoVec[i] );
        free(Y[i]       );
    }

    for (int i = 0; i < X_cols; i++)
    {
        free(Xm_t[i]);
    }   
    free(X   );
    free(Xm  );
    free(Xm_t);
    free(S   );
    free(muV );
    free(Y   );
    free(autoVal);
    free(autoVec);

    return 0;
}
