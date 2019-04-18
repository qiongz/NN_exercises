/**
 * deep neural networks exercises
 * logistic regression,
 * feed forward neural network,
 * convolutional neural networks (to be updated),...
 * @author qiong zhu
 * @version 0.2 18/04/2019
 */

#ifndef DNN_H
#define DNN_H
#include"layers.h"

class dnn {
public:
    /**
     * constructor with input and output layers
     * @param n_f  No. of features in X
     * @param n_c  No. of classes in Y
     * @param head input layer
     * @param tail output layer
     */
    dnn(int n_f,int n_c,layers *head,layers *tail);

    /// destructor, clean the memory space
    ~dnn();

    /** 
     * insert hidden layer after the input layer
     * @param new_layer  the new layer to be inserted
     */
    void insert_after_input(layers * new_layer);
    /// insert hidden layer before the output layer
    void insert_before_output(layers * new_layer);

    /// initialize all layer variables
    void initialize_layers(const int &n_sample,const float &_lambda,const string &optimizer,const bool &batch_norm);
    /// initialize layer caches
    void initialize_layers_caches(const int &n_sample,const bool &is_bp);
    /// clar layer caches
    void clear_layers_caches(const bool &is_bp);
    /// return argmax of given vector in a range
    int get_argmax(const float *x,const int &range);

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
     * @param optimizer if "gradient_descent", uses mini-batch gradient descent, if "Adam", use Adam optimizer
     * @param batch_norm  if true, batch normalization is used
     * @param print_cost print the training/validation cost every 50 epochs if print_cost==true
     * @return weights and bias W,b updated in the object
     */
    void train_and_dev(const vector<float>&X_train,const vector<int>&Y_train,const vector<float>&X_dev,const vector<int>&Y_dev,const int &n_train,const int &n_dev,const int num_epochs,float learning_rate,float lambda,int batch_size,string optimizer,bool batch_norm,bool print_cost);

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


    /// multi layers forward propagate and activation
    void multi_layers_forward(const bool &eval);
    /// multi layers backward propagate to get gradients
    void multi_layers_backward(const float *Y,const int &n_sample);

    void gradient_descent_optimize(const float &initial_learning_rate,const int & num_epochs, const int &step);
    void Adam_optimize(const float &initial_learning_rate,const float &beta_1,const float &beta_2,const int &num_epochs,const int &epoch_step,const int &train_step);
    /**
     * Calculate the mean cost using the cross-entropy loss  <br/>
     * input: output->A, Y   <br/>
     *
     * update:                   <br/>
     * J=-Y.*log(output->A)  <br/>
     * cost=sum(J)/n_sample      <br/>
     *
     * output:
     * @param Y pointer to the datasets Y
     * @param n_sample No. of samples in the datasets
     * @retrun the mean cost
     */
    float cost_function(const float *Y,const int &n_sample);

private: 
    layers *input,*output;  /// layers pointer to the input and output layers
    int n_features,n_classes,n_layers;  /// No. of features, classes and layers
    VSLStreamStatePtr rndStream;  /// pointer to the mkl rng
    unsigned mkl_seed;       /// mkl rng seed
    float Lambda;          /// L2-regularization factor
};
#endif
