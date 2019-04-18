/**
 * layer class for constructing 
 * deep neural networks
 *
 * @author qiong zhu
 * @version 0.1 18/04/2019
 */
#ifndef LAYERS_H
#define LAYERS_H
#include<cstdlib>
#include<random>
#include<cmath>
#include<cstdio>
#include<vector>
#include<iostream>
#include<cstring>
#include<string>
#include<ctime>
#include"mkl.h"
#define EPSILON 1e-12
#define METHOD_INT VSL_RNG_METHOD_UNIFORMBITS_STD
#define METHOD_FLOAT VSL_RNG_METHOD_UNIFORM_STD
using namespace std;


class layers {
public:
    /**
     * constructor
     * @param d layer dimension
     * @param _keep_prob  keep_prob
     * @param dropout if dropout is used
     * @param _layer_type layer type
     * @param _act activation type
     */
    layers(int d,float _keep_prob=1,bool _dropout=false,string _layer_type="None",string _act="None"):dim(d),keep_prob(_keep_prob),dropout(_dropout),layer_type(_layer_type),activation(_act) {
	    optimizer="sgd";
	    batch_norm=false;
	    Lambda=0;
	    is_init=false;
	    prev=NULL;
	    next=NULL;
    }
    ~layers();

    /***
     * Initialize all the layer variables
     * @param _n  n_sample
     * @param _lambda Lambda
     * @param _optimizer optimizer 
     * @param _batch_norm batch normalization parameter
     */
    void initialize(const int &_n,const float &_lambda,const string &_optimizer,const bool &_batch_norm);
    /// allocate dropout mask memory
    void init_dropout_mask();
    /// allocate weights memory and initialize
    void init_weights();
    /// initialize momentum and rms weights for Adam optimization
    void init_momentum_rms();
    /// initialize batch normalization weights 
    void init_batch_norm_weights();
    /// initialize layer caches
    void init_caches(const int &_n_sample,const bool &is_bp);
    /// set up dropout mask
    void set_dropout_mask();
    /// clear layer caches
    void clear_caches(const bool &is_bp);
    /// print layer parameters
    void print_layer_parameters();

    /// get the max value given a float vector
    float getmax(float *x,const int &range);
   
    // activation functions 
    void sigmoid_activate();
    void ReLU_activate();
    void sigmoid_backward();
    void ReLU_backward();
    void get_softmax();

    /**
     * Forward propagate and activation for each layer    <br/>
     * input: prev->A,W,b,prev-> dropout_mask             <br/>
     *
     * update:                                            <br/>
     * if prev->dropout==true:
     *    prev->A=prev->A.* prev->dropout_mask            <br/>
     * A=activation_function(W.T* prev->A+b)              <br/>
     *
     * output: A                                          <br/>
     * @param eval if eval==true, dropout is not used
     * @return A updated
     */
    void forward_activated_propagate(const bool &eval);


    /**
     * backward propagate for each layer <br/>
     * if J is the total mean cost, denote all  <br/>
     * \f$\partial{J}/\partial{A}\to dA\f$, \f$\partial{J}/\partial{Z}\to dZ\f$,  <br/>
     * \f$\partial{J}/\partial{W}\to dW\f$, \f$\partial{J}/\partial{b}\to db\f$, <br/>
     * \f$\partial{A}/\partial{Z}\to dF\f$  <br/>
     * and denote layer_dims[l]->n_[l], * for dot product, .* for element-wise product <br/>
     * input: dZ,prev->A,W                                                      <br/>
     *
     * update:                                                                  <br/>
     * db(dim)=sum(dZ(n_sample,dim),axis=0)                             <br/>
     * dW(dim,prev->dim)=dZ(n_sample,dim).T*prev->A(n_sample,prev->dim)        <br/>
     * dA=activation_backward(prev->A(n_sample,prev->n)                           <br/>
     * prev->dZ(n_sample,prev->dim)=(dZ(n_sample,dim)*W(dim,prev->dim)).*dA(n_sample,prev->dim)   <br/>
     *
     * output: prev->dZ,dW,db                                              <br/>
     * @return prev->dZ,dW,db updated
     */
    void backward_propagate();

    /**
     * update all the weights W and bias b with gradient descent <br/>
     * decrease the learning to 1% of the initial learning rate after all the training sets consumed <br/>
     * learning_rate=initial_learning_rate*(1-step/num_epochs)+0.01*initial_learning_rate <br/>
     *
     * for each layer <br/>
     * W:=W-learning_rate*dW  <br/>
     * b:=b-learning_rate*db  <br/>
     *
     * @param learning_rate  learning rate
     * @param num_epochs No. of epochs in the update
     * @param step
     * @return W,b updated
     */
    void gradient_descent_optimize(const float &initial_learning_rate,const int & num_epochs, const int &step);

    /**
     * update all the weights W and bias b with Adam optimizer <br/>
     * decrease the learning to 1% of the initial learning rate after all the training sets consumed <br/>
     * learning_rate=initial_learning_rate*(1-step/num_epochs)+0.01*initial_learning_rate <br/>
     *
     * for each layer <br/>
     * W:=W-learning_rate*dW  <br/>
     * b:=b-learning_rate*db  <br/>
     *
     * @param learning_rate  learning rate
     * @param num_epochs No. of epochs in the update
     * @param epoch_step epoch step
     * @param train_step total train step
     * @return W,b updated
     */
    void Adam_optimize(const float &initial_learning_rate,const float &beta_1,const float &beta_2,const int &num_epochs,const int &epoch_step,const int &train_step);

    layers *prev,*next;   /// pointer to the previous and next layer
    int dim,n_sample;      /// layer dimension and sample size
    string layer_type,activation,optimizer;   /// layer,activation and optimization types
    VSLStreamStatePtr rndStream;      /// pointer to the mkl random number generator
    unsigned weights_seed,mkl_seed;  /// seed for generate weights and dropout masks
    bool dropout,batch_norm,is_init;  /// if dropout is used, if batch_norm is used, if variables are initialized
    float keep_prob;                 /// keep probability
    float Lambda;                /// L2-regularization parameter
    float *A,*dropout_mask;   /// layer activation values and dropout mask
    float *W,*b;    /// weights and bias
    float *dW,*db;  /// gradients
    float *B,*dB,*G,*dG; // batch normalization weights
    float *dZ,*dA;   /// gradients dZ and gradients cache dA
    float *VdW,*Vdb,*SdW,*Sdb; /// momentum and rms for dW,db used for Adam optimization
    float *VdW_corrected,*Vdb_corrected,*SdW_corrected,*Sdb_corrected; // bias corrected momentum and rms for Adam optimization
};
#endif
