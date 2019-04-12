/// deep neural networks exercises
/** 
 * logistic regression,
 * feed forward neural network,
 * convolutional neural networks (to be updated),... 
 * @author qiong zhu
 * @version 0.1 12/04/2019
 */

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
public:
    /** 
     * constructor without hidden layers, perform logistic regression
     * @param n_f  No. of features in X
     * @param n_c  No. of classes in Y
     */
    dnn(int n_f,int n_c);
    /**
     * constructor with hidden layer dimensions and activation types specified
     * @param n_f No. of features in X
     * @param n_c No. of classes in Y
     * @param n_h No. of hidden layers 
     * @param dims integer vector containing hidden layer dimension
     * @param act_types string vector containing activation types for the hidden layers
     */
    dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string>& act_types);
    /**
     * constructor with hidden layer dimensions, activation types and dropout keep_probs specified
     * @param n_f No. of features in X
     * @param n_c No. of classes in Y
     * @param n_h No. of hidden layers 
     * @param dims integer vector containing hidden layer dimension
     * @param act_types string vector containing activation types for the hidden layers
     * @param k_ps keep probabilities for dropout in the hidden layers
     */
    dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string>& act_types,const vector<float>& k_ps);
    /// destructor, clean the memory space 
    ~dnn();
    /**
     * Perform stochastic batch gradient training and evaluation using the validation(developing) data sets
     * @param X_train training datasets X
     * @param Y_train training datasets Y
     * @param X_dev validation datasets X
     * @param Y_dev validation datasets Y
     * @param n_train No. of training samples
     * @param n_dev No. of validataion samples
     * @param num_epochs No. of epochs to train
     * @param learning_rate learning rate of of gradients updating
     * @param lambda L2-regularization factor
     * @param batch_size  batch size in the stochastic batch gradient training
     * @param print_cost print the training/validation cost every 50 epochs if print_cost==true
     * @return weights and bias W,b updated in the object
     */
    void train_and_dev(const vector<float>&X_train,const vector<int>&Y_train,const vector<float>&X_dev,const vector<int>&Y_dev,const int &n_train,const int &n_dev,const int num_epochs,const float learning_rate,float lambda,int batch_size,bool print_cost);
    /**
     * Perform prediction for the given unlabeled datasets
     * @param X datasets X
     * @param Y_prediction output integer vector containing the predicted labels
     * @param n_sample No. of samples in the datasets
     * @return the predicted labels stored in Y_prediction
     */
    void predict(const vector<float>& X,vector<int> &Y_prediction,const int &n_sample);
    /**
     * Predict and calculate the prediction accuracy for the given labeled datasets
     * @param X datasets X
     * @param Y datasets Y (labels)
     * @param Y_prediction output integer vector containing the predicted labels
     * @param n_sample No. of samples in the datasets
     * @return accuracy , and the predicted labels stored in Y_prediction
     */
    float predict_accuracy(const vector<float>& _X,const vector<int> &Y,vector<int> &Y_prediction,const int &n_sample);
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

    /**
     * Allocate memory space for W,dW,b,db,
     * initialize weights W with random
     * normal distributions and b with zeros
     */
    void initialize_weights();
    /**
     * Shuffle the datasets
     * @param X data X
     * @param Y data Y
     * @param n_sample range (or No. of samples) in X,Y to be shuffled
     * @return X,Y being shuffled
    */
    void shuffle(float *X,float *Y,int n_sample);

    /** 
     * Obtain a batch datasets from the full datasets (used for training/developing)
     * @param X pointer to data X
     * @param Y pointer to data Y
     * @param X_batch  pointer to datasets batched from X
     * @param Y_batch  pointer to datasets batched from Y
     * @param batch_size  batch size
     * @param batch_id  No. of batches extracted, used as an offset
     * @return batched dataset stored in X_batch,Y_batch
     */  
    void batch(const float *X,const float *Y,float *X_batch,float *Y_batch,int batch_size, int batch_id);
    /**
     * Obtain a batch datasets from the full datasets  (used for predicting)
     * @param X pointer to data X
     * @param X_batch  pointer to datasets batched from X
     * @param batch_size  batch size
     * @param batch_id  No. of batches extracted, used as an offset
     * @return batched dataset stored in X_batch
     */
    void batch(const float *X,float *X_batch,int batch_size, int batch_id);
    /**
     * Get the maximum value and index from the array
     * @param x pointer to the array
     * @param range range of array
     * @param index_max reference pointer to argmax of the array
     * @return the maximum value,argmax of the array store in index_max
     */
    float max(const float *x,int range,int &index_max);
    /**
     * Perform sigmoid activation for the given layer
     * input: Z[l] (stored in A[l])
     *
     * update:
     * A[l]=1/(1+exp(-Z[l])
     *
     * output: A[l]
     * @param l layer index
     * @param n_sample No. of samples in the datasets
     * @return A[l] stored with activated neurons
     */
    void sigmoid_activate(const int &l,const int &n_sample);
    /**
     * Perform ReLU activation for the given layer
     * input: Z[l] (stored in A[l])
     *
     * update:
     * A[l]=  Z[l], if Z[l]>0
     *        0   ,  otherwise
     *
     * output: A[l]
     * @param l layer index
     * @param n_sample No. of samples in the datasets
     * @return A[l] stored with activated neurons
     */
    void ReLU_activate(const int &l,const int &n_sample);

    /**
     * Calculate the softmax of the given(by default the final) layer
     * l=n_layers-1
     * input: Z[l] (stored in A[l])
     *
     * update:
     * A[l][i]= exp(Z[l][i])/\sum_i(exp(Z[l][i]))
     *
     * output: A[l]
     * @param n_sample No. of samples in the datasets
     * @return A[l] stored with the softmax neurons
     */
    void get_softmax(const int &n_sample);
    /** 
     * Forward propagate and activation for each layer
     * input: A[l],W[l+1],b[l+1]
     *
     * update:
     * A[l+1]=activation_function(W[l+1].T*A[l]+b[l])
     *
     * output: A[l+1]
     * @param l layer index
     * @param n_sample No. of samples in the datasets
     * @return A[l+1] updated
     */
    void forward_activated_propagate(const int &l ,const int &n_sample);
    /**
     * backward propagate for each layer
     * if J is the total mean cost, denote all 
     * \partial{J}/\partial{A}-> dA, \partial{J}/\partial{Z}-> dZ, 
     * \partial{J}/\partial{W}-> dW, \partial{J}/\partial{b}-> db,
     * \partial{A}/\partial{Z}-> dF
     * and denote layer_dims[l]->n_[l], * for dot product, .* for element-wise product
     * input: dZ[l],A[l-1],W[l]
     *
     * update:
     * db[l](n[l])=sum(dZ[l](n_sample,n[l]),axis=0)
     * dW[l](n[l],n[l-1])=dZ[l](n_sample,n[l]).T*A[l-1](n_sample,n[l-1])
     * dF=activation_backward(A[l-1](n_sample,n[l-1])
     * dZ[l-1](n_sample,n[l])=(dZ[l](n_sample,n[l])*W[l](n[l],n[l-1])).*dF(n_sample,n[l-1])
     *
     * output: dZ[l-1],dW[l],db[l]
     * @param l layer index
     * @param n_sample No. of samples in the datasets
     * @return dZ[l-1],dW[l],db[l] updated
     */
    void backward_propagate(const int &l,const int &n_sample);
    /// multi layers forward propagate and activation
    void multi_layers_forward(const int &n_sample);
    /// multi layers backward propagate to get gradients
    void multi_layers_backward(const float *Y,const int &n_sample);
    /**
     * update all the weights W and bias b
     *
     * for each l from 1 to n_layers-1
     * W[l]:=W[l]-learning_rate*dW[l]
     * b[l]:=b[l]-learning_rate*db[l]
     *
     * @param learning_rate  learning rate
     * @return W,b updated
     */
    void weights_update(const float &learning_rate);

    /**
     * Perform dropout regularization
     * if No. of hidden layers >0 and dropout==true
     * randomly shut down neurons in each layer according 
     * to keep probabilities keep_probs
     */
    void dropout_regularization();
    /**
     * Calculate the mean cost using the cross-entropy loss
     * input: A[n_layers-1], Y
     *
     * update:
     * J=-Y.*log(A[n_layers-1])
     * cost=sum(J)/n_sample
     *
     * output:
     * @param Y pointer to the datasets Y
     * @param n_sample No. of samples in the datasets
     * @retrun the mean cost
     */
    float cost_function(const float *Y,const int &n_sample);
};
#endif
