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
#define METHOD_FLOAT VSL_RNG_METHOD_UNIFORM_STD

using namespace std;
class dnn{
private:
    int n_features,n_classes,n_layers;
    float Lambda;
    bool dropout=false;
    vector<float*> W,b;
    vector<float*> dW,db;
    vector<float*> A,dZ;
    vector<int> layer_dims;
    vector<float> keep_probs;
    vector<string> activation_types;
    unsigned weights_seed,mkl_seed;
    long mkl_rnd_skipped=0;  

    void shuffle(float *X,float *Y,int n_sample);
    void batch(const float *X,const float *Y,float *X_batch,float *Y_batch,int batch_size, int batch_id);
    void batch(const float *X,float *X_batch,int batch_size, int batch_id);
    float cost_function(const float *Y,const int &n_sample);
    void dropout_regularization();
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
    dnn(int n_f,int n_c);
    dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string>& act_types);
    dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string>& act_types,const vector<float>& k_ps);
    ~dnn();
    void fit(const vector<float>&,const vector<int>&,const int &,const int,const float,float,int,bool);
    void train_and_dev(const vector<float>&,const vector<int>&,const vector<float>&,const vector<int>&,const int &,const int &,const int,const float,float,int,bool);
    void predict(const vector<float>& _X,vector<int> &Y_prediction,const int &n_sample,const int &batch_size);
    float predict_accuracy(const vector<float>& _X,const vector<int> &Y,vector<int> &Y_prediction,const int &n_sample,const int &batch_size);
};
#endif
