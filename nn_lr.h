#include<vector>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<cstdio>
#include<ctime>
#include"mkl.h"
#define METHOD_INT VSL_RNG_METHOD_UNIFORMBITS_STD

using namespace std;
class nn_lr
{
public:
    nn_lr(int n_f,int n_c):n_features(n_f),n_classes(n_c) {
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

    void shuffle(float *X,float *Y,int n_sample, int skip);
    void batch(const float *X,const float *Y,float *X_batch,float *Y_batch,int batch_size, int batch_id);
    float cost_function(float *A,float *J,const float *Y,const int &n_sample);
    void forward_propagate(float *A,const float *X,const int &n_sample);
    void backward_propagate(float *A, const float *X, const float *Y,const int &n_sample);
    void gradient_approx(float *A,float *J,const float *X,const float *Y,const int &n_sample);
    void get_softmax(float *A,const int &n_sample);
    float max(float *x,int range,int &index_max);
    void initialize_weights();
};
