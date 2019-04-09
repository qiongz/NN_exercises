#include<vector>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<cstdio>
#include<ctime>
#include"mkl.h"
#include"mkl_vml_functions.h"
#define METHOD_INT VSL_RNG_METHOD_UNIFORMBITS_STD

using namespace std;

class nn_lr
{
public:
    nn_lr(int n_f,int n_c):n_features(n_f),n_classes(n_c=2) {
        initialize_weights();
	seed=time(0);
    }
    ~nn_lr() {};
    void fit(const vector<float>&,const vector<int>&,const int &,const int,const
             float,float,int,bool);
    void predict(const vector<float>&,vector<int> &,const int&);
    float predict_accuracy(const vector<float>&,const vector<int> &,vector<int> &,const int&);

private:
    int n_features,n_classes;
    float Lambda;
    float *W,*dW,*b,*db;
    unsigned seed;

    void shuffle(const vector<float> &_X,const vector<int> &_Y,float *X,float *Y,int n_sample,int skip);
    void batch(const float *X,const float *Y,float *X_batch,float *Y_batch,int batch_size, int batch_id);
    float cost_function(float *A,float *J,const float *Y,const int &n_sample);
    void forward_propagate(float *A,const float *X,const int &n_sample);
    void backward_propagate(float *A,float*I, const float *X, const float *Y,const int &n_sample);
    float max(float *x,int range,int &index_max);
    float sigmoid(float);
    void initialize_weights();
};


float nn_lr::sigmoid(float z) {
    return 1.0/(1+exp(-z));
}


void nn_lr::initialize_weights() {
    srand(time(0));
    W=new float[n_features*n_classes];
    dW=new float[n_features*n_classes];
    b=new float[n_classes];
    db=new float[n_classes];
    for(int i=0; i<n_features*n_classes; i++)
        W[i]=rand()*1.0/RAND_MAX*0.001;
    memset(b,0,sizeof(float)*n_classes);
}


void nn_lr::shuffle(const vector<float> &_X,const vector<int> &_Y,float *X,float *Y,int n_sample, int skip) {
    VSLStreamStatePtr rndStream;
    vslNewStream(&rndStream, VSL_BRNG_MT19937,seed);
    vslSkipAheadStream(rndStream,n_sample*(skip+1));
    unsigned int *i_buffer = new unsigned int[n_sample];
    viRngUniformBits(METHOD_INT, rndStream, n_sample, i_buffer);
    for(int i=0; i<n_sample; i++) {
        int ri=i_buffer[i]%n_sample;
        for(int j=0; j<n_features; j++)
            X[i*n_features+j]=_X[ri*n_features+j];
        for(int j=0; j<n_classes; j++)
            Y[i*n_classes+j]=_Y[ri*n_classes+j];
    }
    vslDeleteStream(&rndStream);
    delete i_buffer;
}

void nn_lr::batch(const float* X,const float *Y,float *X_batch,float *Y_batch,int batch_size,int batch_id) {
    cblas_scopy(n_features*batch_size,X+batch_size*batch_id*n_features,1,X_batch,1);
    cblas_scopy(n_classes*batch_size,Y+batch_size*batch_id*n_classes,1,Y_batch,1);
}

float nn_lr::cost_function(float *A,float *J,const float *Y,const int &n_sample){
    vsLn(n_sample*n_classes,A,J);
    float cost=-cblas_sdot(n_sample*n_classes,Y,1,J,1)/(n_sample);
    cost+=0.5*Lambda*cblas_sdot(n_classes*n_features,W,1,W,1)/(n_sample);
    return cost;
}
float nn_lr::max(float *x,int range,int &index_max){
   float max_val=x[0];
   index_max=0;
   for(int i=1;i<range;i++)
     if(x[i]>max_val){
        max_val=x[i];
	index_max=i;
     }
   return max_val;
}

// Vectorized version of forward_propagate
void nn_lr::forward_propagate(float *A,const float *X,const int &n_sample){
    float alpha_=1;
    float beta_=0;
    // Forward propagation (from X to A1)
    // A(n_sample,n_classes) := X(n_sample,n_features) * W(n_classes, n_features).T,  
    // A[j+i*n_classes]= sum_k (X[k+i*n_features]*W[k+j*n_features])
    // A(i,j): a[i+j*lda] for ColMajor, a[j+i*lda] for RowMajor, lda is the leading dimension
    // op(X)(n_sample,n_features), op(W)( n_features,n_classes)
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n_sample,n_classes,n_features,
	        alpha_, X, n_features, W, n_features, beta_, A, n_classes);

    // A(n_sample,n_classes) := A(n_sample,n_classes)+b(:,n_classes)
      for(int i=0;i<n_sample;i++)
	vsAdd(n_classes,A+i*n_classes,b,A+i*n_classes);

    // calculate the softmax of A
    int id_max;
    for(int i=0;i<n_sample;i++){
	// find the largest A and substract it to prevent overflow
	float A_max=max(A+i*n_classes,n_classes,id_max);
	for(int k=0;k<n_classes;k++) A[k+i*n_classes]-=A_max;
        vsExp(n_classes,A+i*n_classes,A+i*n_classes);   // A:=exp(A-A_max)
	float A_norm=1.0/cblas_sasum(n_classes,A+i*n_classes,1);
        for(int k=0;k<n_classes;k++)
	   A[i*n_classes+k]*=A_norm;   // A:=exp(A-A_max)/\sum_i(exp(A_i-A_max))
    }
}

// Vectorized version of backward_propagate
void nn_lr::backward_propagate(float *A,float *I, const float *X, const float *Y,const int &n_sample) {
    float alpha_=1.0/n_sample;
    float beta_=0;
    // Backward propagation (get gradients)
    vsMul(n_sample*n_classes,Y,A,A);     // A:=Y.*A
    vsSub(n_sample*n_classes,A,Y,A);     // dA:=A-Y
    // dW(n_classes,n_features):= dA(n_sample,n_classes).T * X(n_sample,n_features)
    // op(dA)(n_class,n_sample), op(X)(n_sample,n_features)
    cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,n_classes,n_features,n_sample, 
		    alpha_, A, n_classes, X, n_features, beta_, dW, n_features);
    // dW:=dW+Lambda*W
    cblas_saxpy(n_classes*n_features,Lambda/n_sample,W,1,dW,1);

    // db(n_classes):= alpha_*dA(n_sample,n_classes).T * I(n_classes)
    cblas_sgemv(CblasRowMajor,CblasTrans,n_sample,n_classes,alpha_,A,n_sample,I,1,beta_,db,1);
}

void nn_lr::fit(const vector<float> &_X,const vector<int> &_Y,const int &n_sample, const int num_epoches=2000,const float learning_rate=0.01,const float _lambda=0,int batch_size=128,bool print_cost=false) {
    Lambda=_lambda;
    // tempory space for storing the shuffled training data sets
    float *X=new float[_X.size()];
    float *Y=new float[_Y.size()];
    // batch training data sets
    float *X_batch=new float[batch_size*n_features];
    float *Y_batch=new float[batch_size*n_classes];
    // tempory space used for vectorized calculation of b,cost
    float *I=new float[batch_size];
    float *J=new float[batch_size*n_classes];
    memset(I,1,sizeof(float)*batch_size);
    // activation layer
    float *A=new float[batch_size*n_classes];
    
    float cost,cost_batch;
    for(int i=0; i<num_epoches; i++) {
	// shuffle training data sets for one epoch
        shuffle(_X,_Y,X,Y,n_sample,i);
        cost=0;
	// batch training until all the data sets are used
        for(int s=0; s<n_sample/batch_size; s++) {
            batch(X,Y,X_batch,Y_batch,batch_size,s);
	    // forward_propagate and calculate the activation layer
            forward_propagate(A,X_batch,batch_size);
	    cost_batch=cost_function(A,J,Y_batch,batch_size);
            backward_propagate(A,I,X_batch,Y_batch,batch_size);
            // update the weights and bias
            cblas_saxpy(n_classes*n_features,-learning_rate,dW,1,W,1);   // W:=W-learning_rate*dW
	    cblas_saxpy(n_classes,-learning_rate,db,1,b,1);              // b:=b-learning_rate*db
            cost+=cost_batch;
        }

	cost/=n_sample/batch_size;
        // print the cost
        if(print_cost && i%50==0){
            printf("Cost at epoch %d:  %.6f\n",i,cost);
	}
    }
    delete A,J,I,Y_batch,X_batch,Y,X;
}

void  nn_lr::predict(const vector<float>& _X,vector<int> &Y_prediction,const int &n_sample) {
    float *X=new float[_X.size()];
    float *A=new float[n_sample*n_classes];
    for(int i=0;i<_X.size();i++) X[i]=_X[i];

    // Forward propagation (from X to cost)
    forward_propagate(A,X,n_sample);

    // get the predicted values
    int k_max;
    Y_prediction.assign(n_sample,0);
    for(int i=0; i<n_sample; i++) {
	float max_prob=max(A+i*n_classes,n_classes,k_max);
        Y_prediction[i]=k_max;
    }
    delete A,X;
}


float nn_lr::predict_accuracy(const vector<float>& _X,const vector<int> & Y,vector<int> &Y_prediction,const int &n_sample) {
    float *X=new float[_X.size()];
    float *A=new float[n_sample*n_classes];
    for(int i=0;i<_X.size();i++) X[i]=_X[i];

    // Forward propagation (from X to cost)
    forward_propagate(A,X,n_sample);

    Y_prediction.assign(n_sample,0);
    float accuracy=0;
    int k_max;
    for(int i=0; i<n_sample; i++) {
	float max_prob=max(A+i*n_classes,n_classes,k_max);
        Y_prediction[i]=k_max;
        accuracy+=(Y[i]==k_max?1:0);
    }
    accuracy/=n_sample;
    delete A,X;
    return accuracy;
}
