#include "tools.cuh"

int thread_count;

int maxIndx(float *pred_in, int size){
	float temp = pred_in[0];
	int indx = 0;
	for (int i = 1; i < size; i++)
	{
		if (pred_in[i] > temp){
			indx = i;
			temp = pred_in[i];
		}
	}
	return indx;
}

float norm(float *A, int n){
    float temp=0.0;

    for(int i=0; i<n; i++){
        temp+= A[i]*A[i];
    }
    return sqrt(temp);
}

void VecNormalizer(float *A, int n){
    float mod = norm(A,n);
    for(int i=0; i<n; i++){
        A[i]/=mod;
    }
}

void readImages(float **X, int folders, int nimgs, int h, int w){
    std::string     ifolder;
    std::string     jimage;
    std::string     filename;
    
    int             n=0;
    
    for(int i = 1; i <= folders; i++){
        for(int j = 1; j <= nimgs; j++){
            ifolder     = "s" + std::to_string(i)+"/";
            jimage      = std::to_string(j) + ".pgm";
            filename    = PATH + ifolder + jimage;       
            cv::Mat I   = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  
            n = (i-1)*nimgs + (j-1);
            
            for(int ri = 0; ri < h; ri++){
                for(int ci = 0; ci < w; ci++){ 
                    X[n][ri*w + ci] = (float)I.at<uint8_t>(ri,ci);
                }
            }
            printf("folder %2d, file %2d\n", i, j);
/*
*                Show Images
*                ===========
*                cv::imshow("image",I);
*                cv::waitKey(0);                 
*/
            }
        }

    printf("\nreadImages: %d images\n\n", n+1);
}

void getS(float **S, float **X, float **Xm, float **Xm_t, float *V, int X_rows, int X_cols){
    //   Get Xm (all column with mu_j = 0)
      int mu;
      for(int pi = 0; pi < X_cols; pi++){
          mu = 0;
          for(int ni = 0; ni < X_rows; ni++){
              mu += X[ni][pi];
          }
          mu = mu/((float)X_rows);
          V[pi] = mu;
          for(int ni = 0; ni < X_rows; ni++){
              Xm[ni][pi] = X[ni][pi] - mu;
          }
      }
      std::cout<<"Xm \t\t<- Done!"<<std::endl;
  
  //  Get transpose
      for(int ni = 0; ni < X_rows; ni++){
          for(int pi = 0; pi < X_cols; pi++){
              Xm_t[pi][ni] = Xm[ni][pi];
          }
      }
      std::cout<<"Xm_t \t\t<- Done!"<<std::endl;    
  
  //  Get S
      float temp;
      for(int sr = 0; sr < X_rows; sr++){
          for(int sc=0; sc < X_rows; sc++){
              temp = 0.0;
              for(int it = 0; it < X_cols; it++){
                  temp += Xm[sr][it]*Xm_t[it][sc]; 
              }
              S[sr][sc] = temp;
          }
      }    
      std::cout<<"S(Xm*Xm_t) \t<- Done!"<<std::endl;   
  }
  
  void getW(float **A, float **B, float **W, int m, int n){
    printf("\nGetting W ...\n");
    float temp;

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            temp = 0.0;
            for(int k=0; k<n; k++){
                temp += A[i][k]*B[k][j];                
            }
            W[i][j] = temp;
        }
    }
    
    float *vecW_j = new float[m];
    for(int jW=0; jW<n; jW++){
        for(int iW=0; iW<m; iW++){
            vecW_j[iW] = W[iW][jW];
        }
        VecNormalizer(vecW_j, m);
        for(int iW=0; iW<m; iW++){
            W[iW][jW] = vecW_j[iW];
        }       
    }
}

void showImage(float *I, int w, int h, int scale){
    float_t *temp;    
    temp = (float *)malloc(w*h*sizeof(float));
    for(int i=0; i < w*h; i++){
        temp[i] = (float_t)I[i];
    }

    cv::Mat src(h,w,CV_32FC1,temp);
    cv::Mat dst;
    cv::normalize(src,dst,0.0,1.0,cv::NORM_MINMAX,CV_32FC1);
    if (scale > 1)
        cv::resize(dst, dst, cv::Size(dst.cols*scale, dst.rows*scale));
    cv::imshow("img", dst);
    cv::waitKey(0);

    delete [] temp;
}

float *Reconstructor(float *Img, float **Wk, int m, int k){
    float *Y       = new float[k];
    float *newI    = new float[m];
    float temp;

    for(int jk=0; jk < k; jk++){
        temp = 0.0;
        for(int im=0; im<m; im++){
            temp += Img[im]*Wk[im][jk];
        }
        Y[jk] = temp;
    }
    
    for(int im=0; im < m; im++){
        temp =0.0;
        for(int ik=0; ik<k; ik++){
            temp+= Y[ik]*Wk[im][ik];
        }
        newI[im] = temp;
    }
    return newI;
}

void getProyection(float *Img, float **Wk, int m, int k, float* Y){
    float temp;

    for(int jk=0; jk < k; jk++){
        temp = 0.0;
//#       pragma omp parallel for num_threads(thread_count) reduction(+:temp)
        for(int im=0; im<m; im++){
            temp += Img[im]*Wk[im][jk];
        }
        Y[jk] = temp;
    }

    VecNormalizer(Y, k);
}