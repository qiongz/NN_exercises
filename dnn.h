#ifndef DNN_H
#define DNN_H
#include<cstdlib>
#include<cmath>
#include<cstdio>
#include<vector>
#include<iostream>
#include<cstring>
#include<string>
#include<ctime>
#include<random>
#include"mkl.h"
#define METHOD_INT VSL_RNG_METHOD_UNIFORMBITS_STD

using namespace std;
class dnn{
private:
    int n_features,n_classes,n_layers;
    float Lambda;
    vector<float*> W,b;
    vector<float*> dW,db;
    vector<float*> A,dZ;
    vector<int> layer_dims;
    vector<string> activation_types;
    unsigned weights_seed,shuffle_seed;

    void shuffle(float *X,float *Y,int n_sample, int skip);
    void batch(const float *X,const float *Y,float *X_batch,float *Y_batch,int batch_size, int batch_id);
    void batch(const float *X,float *X_batch,int batch_size, int batch_id);
    float cost_function(const float *Y,const int &n_sample);
    void forward_activated_propagate(const int &l ,const int &n_sample);
    void backward_propagate(const int &l,const int &n_sample);
    void multi_layers_forward(const int &n_sample);
    void multi_layers_backward(const float *Y,const int &n_sample);
    void weights_update(const float &learning_rate);
    void get_softmax(const int &n_sample);
    void sigmoid_activate(const int &l,const int &n_sample);
    void ReLU_activate(const int &l,const int &n_sample);
    void LeakyReLU_activate(const int &l,const int &n_sample);
    float max(float *x,int range,int &index_max);
    void initialize_weights();
public:
    // default initializer without hidden-layers
    dnn(int n_f,int n_c){
	shuffle_seed=time(0);
	n_features=n_f;
	n_classes=n_c;
	n_layers=2;
	layer_dims.push_back(n_features);
	layer_dims.push_back(n_classes);
	activation_types.push_back("NULL");
	activation_types.push_back("softmax");
        initialize_weights();
    }
    dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string>& act_types){
	shuffle_seed=time(0);
	n_features=n_f;
	n_classes=n_c;
	n_layers=n_h+2;
	layer_dims.push_back(n_features);
	for(int i=0;i<n_h;i++) layer_dims.push_back(dims[i]);
	layer_dims.push_back(n_classes);
	activation_types.push_back("NULL");
	for(int i=0;i<n_h;i++)
	activation_types.push_back(act_types[i]);
	activation_types.push_back("softmax");
        initialize_weights();
    }
    ~dnn(){
      for(int i=0;i<n_layers;i++)
       delete db[i],b[i],dW[i],W[i];
      
      db.clear();
      b.clear();
      dW.clear();
      W.clear();
    }
    void fit(const vector<float>&,const vector<int>&,const int &,const int,const float,float,int,bool);
    void train_and_dev(const vector<float>&,const vector<int>&,const vector<float>&,const vector<int>&,const int &,const int &,const int,const float,float,int,bool);
    void predict(const vector<float>& _X,vector<int> &Y_prediction,const int &n_sample,const int &batch_size);
    float predict_accuracy(const vector<float>& _X,const vector<int> &Y,vector<int> &Y_prediction,const int &n_sample,const int &batch_size);
};
#endif
