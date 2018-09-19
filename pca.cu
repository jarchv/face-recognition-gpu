#include "pca.cuh"

void copyV(float *v1, float *v2, int n){
    for(int i=0; i<n; i++){
        v2[i] = v1[i];
    }
}

void copyM(float **m1, float **m2, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            m2[i][j]=m1[i][j];
        }
    }
}

void matmul(float **X, float *B, float *C, int n){
    float temp;
    for(int iA=0; iA<n;iA++){        
        temp = 0.0;
        for(int k=0;k<n; k++){
            temp+=X[iA][k]*B[k];
        }
        C[iA]=temp;
    }
}

void updateAA(float **AA, float lambda, float *xnew, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            AA[i][j] -= lambda*xnew[i]*xnew[j];
        }
    }
}

void copy2Mat(float **V, float *Vi,int col, int n){
    for(int i=0; i<n; i++){
        V[i][col] = Vi[i];
    }
}


float dotV(float *V1, float *V2, float n){
    float temp = 0.0;
    for(int i=0; i<n; i++){
        temp+=V1[i]*V2[i];
    }
    return temp;
}

void vectorOp(float *A, float *B,int n, float op){
    for(int i=0; i<n; i++){
        A[i] = A[i] + op*B[i];
    }
}

float normdiff(float *A, float *B, int n){
    float temp=0.0;
    for(int i=0; i<n; i++){
        temp+= pow(A[i]-B[i],2); 
    }
    return sqrt(temp);
}

void powerMethod(float **A, float **eigenVec, float *eigenVal, int n, float tol){
    float **AA = new float*[n];
    printf("\n");

    for(int i=0; i<n; i++){
        AA[i] = new float[n];
    }

    copyM(A,AA,n);
    float *xini = new float[n];

    for(int j=0;j<n;j++){
        if(j==0){xini[j] = 1.0;}
        else{xini[j] = 0.0;}
    }
    float *x0 = new float[n];
    float *xnew = new float[n];
    float lambda; 
    float ol_lambda;
    
    for(int iv=0; iv<n;iv++){
        printf("\rgetting eigenvectors %3d%%",(int)(100*(iv+1)/n));
        fflush(stdout);        
        copyV(xini,x0,n);
        ol_lambda = 0.0;
        int it = 0;
        while(1){
            matmul(AA,x0,xnew,n);
            lambda = norm(xnew,n);
            VecNormalizer(xnew,n);
            
            if((abs((ol_lambda-lambda)/ol_lambda)<tol) and (normdiff(x0,xnew,n)<tol)){
                break;
            }

            ol_lambda = lambda;
            copyV(xnew,x0,n); 

            if(it > 1000){break;}
            it++;           
        }
        float fac = dotV(xnew,xini,n);
        vectorOp(xini,xnew,n,-fac);
        eigenVal[iv] = lambda;
        copy2Mat(eigenVec,xnew,iv,n);
        updateAA(AA,lambda,xnew,n);
    }
    printf(" done!\n");
}