#include<vector>
#include<cstdlib>
#include<cstring>
#include<string>
#include<cmath>
#include<cstdio>
#include<ctime>
#include<random>
#include"mkl.h"
#define METHOD_INT VSL_RNG_METHOD_UNIFORMBITS_STD

using namespace std;
class dnn
{
public:
    // default initializer without hidden-layers
    dnn(int n_f,int n_c):n_features(n_f),n_classes(n_c) {
	n_layers=2;
        initialize_weights();
	layer_dims.push_back(n_features);
	layer_dims.push_back(n_classes);
	seed=time(0);
    }
    dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string> act_types):n_features(n_f),n_classes(n_c),n_layers(n_h+2){
	layer_dims.push_back(n_features);
	for(int i=0;i<n_h;i++) layer_dims.push_back(dims[i]);
	layer_dims.push_back(n_classes);
        initialize_weights();
	activations_types.assign(act_types);
	seed=time(0);
    }
    ~dnn() {};
    void fit(const vector<float>&,const vector<int>&,const int &,const int,const
             float,float,int,bool);
    void predict(const vector<float>&,vector<int> &,const int&);
    float predict_accuracy(const vector<float>&,const vector<int> &,vector<int> &,const int&);

private:
    int n_features,n_classes,n_layers;
    float Lambda;
    vector<float*> dW,db;
    vector<float*> A,dA;
    vector<int> layer_dims;
    vector<string> activation_types;
    unsigned weights_seed;

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
