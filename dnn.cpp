#include"dnn.h"

dnn::dnn(int n_f,int n_c,layers *head,layers *tail) {
    Lambda=0;
    mkl_seed=time(0);
    vslNewStream(&rndStream, VSL_BRNG_MT19937,mkl_seed);
    n_features=n_f;
    n_classes=n_c;
    n_layers=2;
    input=head;
    output=tail;
    input->next=output;
    input->prev=NULL;
    output->prev=input;
    output->next=NULL;
}

dnn::~dnn() {
    vslDeleteStream(&rndStream);
}

void dnn::insert_after_input(layers * new_layer) {
    new_layer->prev=input;
    new_layer->next=input->next;
    input->next->prev=new_layer;
    input->next=new_layer;
    n_layers+=1;
}

void dnn::insert_before_output(layers *new_layer) {
    new_layer->next=output;
    new_layer->prev=output->prev;
    output->prev->next=new_layer;
    output->prev=new_layer;
    n_layers+=1;
}


//Fisher and Yates' shuffle algorithm
void dnn::shuffle(float *X,float *Y,int n_sample) {
    unsigned int *i_buffer = new unsigned int[n_sample];
    // generater n_sample unsigned integer random numbers and store in the buffer i_buffer
    viRngUniformBits(METHOD_INT, rndStream, n_sample, i_buffer);
    for(int i=n_sample-1; i>=1; i--) {
        // find the random index which is smaller than i
        int j=i_buffer[i]%i;
        // swap sample vectors X[i] and X[j]
        cblas_sswap(n_features,X+i*n_features,1,X+j*n_features,1);
        // swap sample vectors Y[i] and Y[j] accordingly
        cblas_sswap(n_classes,Y+i*n_classes,1,Y+j*n_classes,1);
    }
    delete i_buffer;
}

void dnn::batch(const float* X,const float *Y,float *X_batch,float *Y_batch,int batch_size,int batch_id) {
    // copy the number of batch_size samples start from (batch-id*batch_size)-th position
    // at X and Y vector to X_batch and Y_batch
    cblas_scopy(n_features*batch_size,X+batch_size*batch_id*n_features,1,X_batch,1);
    cblas_scopy(n_classes*batch_size,Y+batch_size*batch_id*n_classes,1,Y_batch,1);
}

void dnn::batch(const float* X,float *X_batch,int batch_size,int batch_id) {
    // copy the number of batch_size samples start from (batch-id*batch_size)-th position
    // at X vector to X_batch
    cblas_scopy(n_features*batch_size,X+batch_size*batch_id*n_features,1,X_batch,1);
}

// cost function should only be called after or without backward-propagation
float dnn::cost_function(const float *Y,const int &n_sample) {
    // create tempory space to store the cross-entropy loss vector
    float *J=new float[n_classes*n_sample];
    // copy softmax-layer to J, add EPSILON=1e-12 to prevent underflow
    // J=A+EPSILON
    for(int i=0; i<n_sample*n_classes; i++)
        J[i]=output->A[i]+EPSILON;
    // J=log(J)
    vsLn(n_sample*n_classes,J,J);
    // cost=-Y*log(J)/n_sample
    float cost=-cblas_sdot(n_sample*n_classes,Y,1,J,1)/n_sample;
    // if L2-regularization parameter Lambda!=0, add cost from regularization
    if(Lambda>EPSILON)
        // cost += Lambda/(2*n_sample)*sum(W.^2)
	for(layers *n=input->next;n!=NULL;n=n->next)
            cost+=0.5*Lambda*cblas_sdot(n->dim*n->prev->dim,n->W,1,n->W,1)/n_sample;
    delete J;
    return cost;
}

// perform multi-layers forward activated propagate
void dnn::multi_layers_forward(const bool &eval) {
    // from 2nd to output layer
    for(layers *n=input->next; n!=NULL; n=n->next) 
        n->forward_activated_propagate(eval);
}

// perform multi-layers backward propagate
void dnn::multi_layers_backward(const float *Y,const int &n_sample) {
    // softmax layer: dZ[l-1]=(A-Y)/n_sample
    // dZ=A-Y
    vsSub(n_sample*n_classes,output->A,Y,output->dZ);
    // dZ=dZ/n_sample
    cblas_sscal(n_sample*n_classes,1.0/n_sample,output->dZ,1);
    // get gradients for all the layers down to l=1
    for(layers *n=output; n!=input; n=n->prev)
        n->backward_propagate();
}


void dnn::Adam_optimize(const float &initial_learning_rate,const float &beta_1,const float &beta_2,const int &num_epochs,const int &epoch_step,const int &train_step) {
    for(layers *n=input->next; n!=NULL; n=n->next)
	if(n->layer_type!="Pool")
        n->Adam_optimize(initial_learning_rate,beta_1,beta_2,num_epochs,epoch_step,train_step);
}

void dnn::gradient_descent_optimize(const float &initial_learning_rate,const int &num_epochs,const int &epoch_step) {
    for(layers *n=input->next; n!=NULL; n=n->next)
	if(n->layer_type!="Pool")
        n->gradient_descent_optimize(initial_learning_rate,num_epochs,epoch_step);
}

void dnn::initialize_layers(const int &n_sample,const float &lambda,const string &optimizer,const bool &batch_norm) {
    for(layers *n=input; n!=NULL; n=n->next)
        n->initialize(n_sample,lambda,optimizer,batch_norm);
}

void dnn::initialize_layers_caches(const int &n_sample,const bool &is_bp) {
    for(layers *n=input; n!=NULL; n=n->next)
        n->init_caches(n_sample,is_bp);
}

void dnn::clear_layers_caches(const bool &is_bp){
    for(layers *n=input;n!=NULL;n=n->next)
	    n->clear_caches(is_bp);
}

int dnn::get_argmax(const float *x,const int &range){
	int max_idx=0;
	for(int i=0;i<range;i++)
		if(x[i]>x[max_idx])
			max_idx=i;
	return max_idx;
}

void dnn::print_layers(){
    cout<<"----------------------  layer parameters  --------------------"<<endl;
    for(layers *n=input;n!=NULL;n=n->next)
	    n->print_parameters();
}

void dnn::train_and_dev(const vector<float> &_X_train,const vector<int> &_Y_train,const vector<float>& _X_dev, const vector<int>& _Y_dev,const int &n_train, const int &n_dev, const int num_epochs=100,float learning_rate=0.001,const float _lambda=0,int batch_size=128,string optimizer="gradient_descent",bool batch_norm=false,bool print_cost=false,int print_period=1) {
    Lambda=_lambda;
    // tempory space for storing the shuffled training/developing data sets
    float *X_train=new float[_X_train.size()];
    float *Y_train=new float[_Y_train.size()];
    float *X_dev=new float[_X_dev.size()];
    float *Y_dev=new float[_Y_dev.size()];

    // deep copy _X_train,_Y_train,_X_dev,_Y_dev
    for(int i=0; i<_X_train.size(); i++) X_train[i]=_X_train[i];
    for(int i=0; i<_Y_train.size(); i++) Y_train[i]=_Y_train[i];
    for(int i=0; i<_X_dev.size(); i++) X_dev[i]=_X_dev[i];
    for(int i=0; i<_Y_dev.size(); i++) Y_dev[i]=_Y_dev[i];

    initialize_layers(batch_size,Lambda,optimizer,batch_norm);

    float *Y_batch=new float[batch_size*n_classes];
    float cost_train,cost_dev,cost_batch;
    int num_train_batches=n_train/batch_size;
    // use num_dev_batches*batch_size for simplicity
    int num_dev_batches=n_dev/batch_size;
    long train_step=0;

    for(int i=0; i<num_epochs+1; i++) {
        // shuffle training data sets for one epoch
        shuffle(X_train,Y_train,n_train);
        cost_train=0;
        // batch training until all the data sets are used
        for(int s=0; s<num_train_batches; s++) {
	    train_step=i*num_train_batches+s;
            // batch feed input->A from X_train
            batch(X_train,Y_train,input->A,Y_batch,batch_size,s);
            multi_layers_forward(false);
            multi_layers_backward(Y_batch,batch_size);
            cost_batch=cost_function(Y_batch,batch_size);
            if(optimizer=="Adam")
                Adam_optimize(learning_rate,0.9,0.999,num_epochs,i,train_step);
            else
              gradient_descent_optimize(learning_rate,num_epochs,i);
            cost_train+=cost_batch;
            //printf("Cost of train at train_step %6d :  %.8f \n",train_step,cost_batch);
        }
        cost_train/=num_train_batches;
        // evaluate the developing data sets and print the training/dev cost
        if(print_cost && i%print_period==0) {
            cost_dev=0;
            for(int s=0; s<num_dev_batches; s++) {
                // batch feed A[0] from X_dev
                batch(X_dev,Y_dev,input->A,Y_batch,batch_size,s);
                multi_layers_forward(true);
                cost_batch=cost_function(Y_batch,batch_size);
                cost_dev+=cost_batch;
            }
            cost_dev/=num_dev_batches;

            printf("Cost of train/validation at epoch %4d :  %.8f %.8f \n",i,cost_train,cost_dev);
        }
    }
    // clean the dropout masks, layer caches and layer gradient caches
    delete Y_batch;
    clear_layers_caches(true);
    delete Y_dev,X_dev,Y_train,X_train;
}

void dnn::predict(const vector<float>& _X,vector<int> &Y_prediction,const int &n_sample) {
    // predict with small batch_size each time to reduce memory usage
    int batch_size=128;
    // tempory space for the datasets for batching
    float *X=new float[_X.size()];
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];

    initialize_layers_caches(batch_size,false);

    int n_batch=n_sample/batch_size;
    int n_residual=n_sample%batch_size;
    int k_max;
    Y_prediction.assign(n_sample,-1);
    for(int s=0; s<n_batch; s++) {
        // batch feed A[0] from X and feed forward propagate
        batch(X,input->A,batch_size,s);
        multi_layers_forward(true);
        // get the predictions from the softmax layer
        for(int k=0; k<batch_size; k++) {
            int j=s*batch_size+k;
	    k_max=get_argmax(output->A+k*n_classes,n_classes);
            Y_prediction[j]=k_max;
        }
    }
    // get the predictions for the residual samples
    if(n_residual>0) {
        memcpy(input->A,X+n_batch*batch_size*n_features,sizeof(float)*n_residual*n_features);
        multi_layers_forward(true);
        for(int k=0; k<n_residual; k++) {
            int j=n_batch*batch_size+k;
	    k_max=get_argmax(output->A+k*n_classes,n_classes);
            Y_prediction[j]=k_max;
        }
    }

    clear_layers_caches(false);
    delete X;
}

float dnn::predict_accuracy(const vector<float>& _X,const vector<int> &Y,vector<int> &Y_prediction,const int &n_sample) {
    // predict with small batch_size each time to reduce memory usage
    int batch_size=128;
    // tempory space for the datasets for batching
    float *X=new float[_X.size()];
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];

    initialize_layers_caches(batch_size,false);

    int n_batch=n_sample/batch_size;
    int n_residual=n_sample%batch_size;
    int k_max;
    Y_prediction.assign(n_sample,-1);
    for(int s=0; s<n_batch; s++) {
        // batch feed A[0] from X and feed forward propagate
        batch(X,input->A,batch_size,s);
        multi_layers_forward(true);
        // get the predictions from the softmax layer
        for(int k=0; k<batch_size; k++) {
            int j=s*batch_size+k;
	    k_max=get_argmax(output->A+k*n_classes,n_classes);
            Y_prediction[j]=k_max;
        }
    }
    // get the predictions for the residual samples
    if(n_residual>0) {
        memcpy(input->A,X+n_batch*batch_size*n_features,sizeof(float)*n_residual*n_features);
        multi_layers_forward(true);
        for(int k=0; k<n_residual; k++) {
            int j=n_batch*batch_size+k;
	    k_max=get_argmax(output->A+k*n_classes,n_classes);
            Y_prediction[j]=k_max;
        }
    }

    // calculate the accuracy of the prediction
    float accuracy=0;
    for(int i=0; i<n_sample; i++)
        accuracy+=(Y[i]==Y_prediction[i]?1:0);
    accuracy/=n_sample;

    clear_layers_caches(false);
    delete X;
    return accuracy;
}
