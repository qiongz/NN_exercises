#include"dnn.h"
#include<iostream>
#include"utils.h"
#define EPSILON 1e-10

void dnn::initialize_weights() {
    default_random_engein rng(weights_seed);
    normal_distribution<float> dist_norm(0,1);
    // n_layers-1 W,dW,b,db
    for(int i=1;i<n_layers;i++){
	W.push_back(new float[layer_dims[i-1]*layer_dims[i]]);
	dW.push_back(new float[layer_dims[i-1]*layer_dims[i]]);
	b.push_back(new float[layer_dims[i]]);
	db.push_back(new float[layer_dims[i]]);
	for(int j=0;j<layer_dims[i-1]*layer_dims[i];j++)
          if(i==layers-1)
          W[i][j]=dist_norm(rng)*sqrt(1.0/layer_dims[i-1]);
	  else
          W[i][j]=dis_norm(rng)*sqrt(2.0/layer_dims[i-1]);  // factor "2" for ReLU
	memset(b[i],0,sizeof(float)*layer_dims[i]);
    }
}

//Fisher and Yates' shuffle algorithm
void dnn::shuffle(float *X,float *Y,int n_sample, int skip) {
    VSLStreamStatePtr rndStream;
    vslNewStream(&rndStream, VSL_BRNG_MT19937,seed);
    vslSkipAheadStream(rndStream,n_sample*(skip+1));
    unsigned int *i_buffer = new unsigned int[n_sample];
    float *X_temp=new float[n_features];
    float *Y_temp=new float[n_classes];
    viRngUniformBits(METHOD_INT, rndStream, n_sample, i_buffer);
    for(int i=n_sample-1; i>=0; i--) {
        int j=i_buffer[i]%(i+1);
        if(j!=i) {
            cblas_sswap(n_features,X+i*n_features,1,X+j*n_features,1);
            cblas_sswap(n_classes,Y+i*n_classes,1,Y+j*n_classes,1);
        }
    }
    vslDeleteStream(&rndStream);
    delete Y_temp,X_temp,i_buffer;
}

void dnn::batch(const float* X,const float *Y,float *X_batch,float *Y_batch,int batch_size,int batch_id) {
    cblas_scopy(n_features*batch_size,X+batch_size*batch_id*n_features,1,X_batch,1);
    cblas_scopy(n_classes*batch_size,Y+batch_size*batch_id*n_classes,1,Y_batch,1);
}

float dnn::cost_function(float *A,float *J,const float *Y,const int &n_sample) {
    vsLn(n_sample*n_classes,A,J);
    float cost=-cblas_sdot(n_sample*n_classes,Y,1,J,1)/n_sample;
    cost+=0.5*Lambda*cblas_sdot(n_classes*n_features,W,1,W,1)/n_sample;
    return cost;
}

float dnn::max(float *x,int range,int &index_max) {
    float max_val=x[0];
    index_max=0;
    for(int i=1; i<range; i++)
        if(x[i]>max_val) {
            max_val=x[i];
            index_max=i;
        }
    return max_val;
}

// softmax-layer  A[L-1][:,k]=exp(A[L-1][:,k])/\sum_j(exp(A[L-1][:,j]))
void dnn::get_softmax(const int &n_sample) {
    // calculate the softmax of A[L-1]
    // layer_dims[n_layers-1] = n_classes
    int id_max;
    for(int i=0; i<n_sample; i++) {
        // find the largest A and substract it to prevent overflow
        float A_max=max(A[n_layers-1]+i*n_classes,n_classes,id_max);
        for(int k=0; k<n_classes; k++) A[n_layers-1][k+i*n_classes]-=A_max;
        vsExp(n_classes,A[n_layers-1]+i*n_classes,A[n_layers-1]+i*n_classes);   // A:=exp(A-A_max)
        float A_norm=1.0/cblas_sasum(n_classes,A[n_layers-1]+i*n_classes,1);
        for(int k=0; k<n_classes; k++)
            A[n_layers-1][i*n_classes+k]*=A_norm;   // A:=exp(A-A_max)/\sum_i(exp(A_i-A_max))
    }
}

/*
void dnn::gradient_approx(float *A,float *J,const float *X,const float *Y,const int &n_sample) {
    float epsilon=1e-4;
    float delta_cost;
    float W_orig,b_orig;
    for(int i=0; i<n_classes; i++)
        for(int j=0; j<n_features; j++) {
	    W_orig=W[j+i*n_features];

            W[j+i*n_features]+=epsilon;
            forward_propagate(A,X,n_sample);
            get_softmax(A,n_sample);
            delta_cost=cost_function(A,J,Y,n_sample);

            W[j+i*n_features]-=2*epsilon;
            forward_propagate(A,X,n_sample);
            get_softmax(A,n_sample);
            delta_cost-=cost_function(A,J,Y,n_sample);

            dW[j+i*n_features]=delta_cost*0.5/epsilon;
            W[j+i*n_features]=W_orig;
        }
    for(int j=0; j<n_classes; j++) {
	b_orig=b[j];

        b[j]+=epsilon;
        forward_propagate(A,X,n_sample);
        get_softmax(A,n_sample);
        delta_cost=cost_function(A,J,Y,n_sample);

        b[j]-=2*epsilon;
        forward_propagate(A,X,n_sample);
        get_softmax(A,n_sample);
        delta_cost-=cost_function(A,J,Y,n_sample);

        db[j]=delta_cost*0.5/epsilon;
        b[j]=b_orig;
    }
}
*/

void dnn::sigmoid_activate(const int &l,const int &n_sample){
    for(int i=0;i<layer_dims[l]*n_sample;i++)
       A[l][i]=-A[l][i];
    vsExp(layer_dims[l]*n_sample,A[l],A[l]);
    for(int i=0;i<layer_dims[l]*n_sample;i++)
       A[l][i]+=1;
    vsDiv(layer_dims[l]*n_sample,A[l],A[l]);
}

// Vectorized version of forward_activated_propagate for each layer
// A[l]=activate(Z[l-1]*W.T+b)
void dnn::forward_activated_propagate(const int &l, const int &n_sample) {
    float alpha_=1;
    float beta_=0;
    // linear forward propagation (from A[l-1] to Z[l])
    // Z[l](n_sample,n_l) := A[l-1](n_sample,n_{l-1}) * W(n_l, n_{l-1}).T,
    // Z(i,j): a[i+j*lda] for ColMajor, a[j+i*lda] for RowMajor, lda is the leading dimension
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n_sample,layer_dims[l],layer_dims[l-1],
                alpha_, A[l-1], layer_dims[l-1], W[l], layer_dims[l-1], beta_, A[l], layer_dims[l]);

    // Z[l]:= Z[l]+b[l]
    for(int i=0; i<n_sample; i++)
        vsAdd(layer_dims[l],A[l]+i*layer_dims[l],b[l],A[l]+i*layer_dims[l]);

    // final output: A[l]= f(Z[l])
    if(activation_types[l-1]=="sigmoid")
	sigmoid_activate(l,n_sample);
}

// Vectorized version of backward_propagate, 
// (all dA[l-1],dW[l] is shifted by 1: dA[l-1]-> dA[l-2], dW[l]-> dW[l-1] 
// dJ/dA[l-1]=dJ/dA[l]*dA[l]/dA[l-1]= dJ/dA[l]*f'(Z[l-1])*W[l].T
// dJ/dW[l]=dJ/dA[l]*dA[l]/dW[l-1]= dJ/dA[l]*f'(Z[l-1])*A[l-1].T
// dJ/db[l-1]=dJ/dA[l]*dA[l]/db[l-1]= dJ/dA[l]*f'(Z[l-1])
void dnn::backward_propagate(const int & l,const int &n_sample) {
    // create tempory space for f'() factor
    float* dF=new float[layer_dims[l]*n_sample];
    float alpha_=1;
    float beta_=0;
    if(activation_types[l-1]=="sigmoid"){
    	// for sigmoid activation function, f'(z)=f(z)(1-f(z))    
	vsMul(layer_dims[l]*n_sample,A[l],A[l],dF);  // dF= A.*A
	vsSub(layer_dims[l]*n_sample,dF,A[l],dF);     // dF=dF-A
    }
    // calculate dJ/dA[l]*f'(Z[l-1]), the previous dA[l-1] (or 'dA_{l}') is overwritten and pass down
    vsMul(layer_dims[l]*n_sample,dA[l-1],dF,dA[l-1]);   // dA[l-1]=dA[l].*dF
    // calculate db[l]:=sum(dA[l](n_sample,n_l),axis=0)
    for(int i=0;i<layer_dims[l];i++){
        db[l][i]=0;
        for(int j=0; j<n_sample; j++)
            db[l][i]+=A[j*layer_dims[l]+i];
        db[l][i]/=n_sample;
    } 

    // dJ/dW[l] = dJ/dA[l]*f'(Z[l-1])*A[l-1].T
    // dW[l](n_l,n_{l-1}):= dF(n_sample,n_l).T * A[l-1](n_sample,n_{l-1})
    // op(dA)(n_class,n_sample), op(X)(n_sample,n_features)
    cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,layer_dims[l],layer_dim[l-1],n_sample,
                 alpha_, dA[l], layer_dims[l], A[l-1], layer_dims[l-1], beta_, dW[l], layer_dims[l-1]);

    // dW:=dW+Lambda*W
    if(Lambda>1e-8)
        cblas_saxpy(layer_dims[l]*layer_dims[l-1],Lambda/n_sample,W[l],1,dW[l],1);

    // dJ/dA[l-1] = dJ/dA[l]*f'(Z[l-1])*W[l].T
    cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,n_sample,layer_dim[l-1],layer_dim[l],
                 alpha_, dA[l], layer_dims[l], W[l], layer_dims[l-1], beta_, dA[l-1], layer_dims[l-1]);

}

void dnn::multi_layers_forward(const int &n_sample){
    for(int l=1;l<n_layers;l++)
      forward_propagate(l,n_sample);
    get_softmax(n_sample);
}

void dnn::multi_layers_backward(const float *Y,const int &n_sample){
     // softmax layer: dJ/dA[L-2]=(A-Y)/n_sample (dA has dimsion L-1
     float alpha_=1.0/n_sample;
     vsSub(n_sample*layer_dims[n_layers-1],A[n_layers-1],Y,dA[n_layers-2]);
     for(int i=0;i<n_sample*layer_dims[n_layers-1];i++)  dA[n_layers-2][i]*=alpha;

     // get gradients
     for(int l=n_layers-1;l>=1;l--)
	backward_propagate(l,n_sample);
}


void dnn::fit(const vector<float> &_X,const vector<int> &_Y,const int &n_sample, const int num_epochs=2000,const float learning_rate=0.01,const float _lambda=0,int batch_size=128,bool print_cost=false) {
    Lambda=_lambda;
    // tempory space for storing the shuffled training data sets
    float *X=new float[_X.size()];
    float *Y=new float[_Y.size()];

    // deep copy X,Y
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];
    for(int i=0; i<_Y.size(); i++) Y[i]=_Y[i];

    // initialize layer caches and layer gradients
    for(int l=0;l<n_layers;l++)
       A.push_back(new float[layer_dims[l]*batch_size]);
    // the first-layer of dA is not calculated
    for(int l=1;l<n_layers;l++)   
       dA.push_back(new float[layer_dims[l]*batch_size]);

    // batch training data sets
    float *Y_batch=new float[batch_size*n_classes];
    float cost,cost_batch;
    for(int i=0; i<num_epochs; i++) {
        // shuffle training data sets for one epoch
        shuffle(X,Y,n_sample,i);
        cost=0;
        // batch training until all the data sets are used
        for(int s=0; s<n_sample/batch_size; s++) {
            // feed A[0] with X_batch
            batch(X,Y,A[0],Y_batch,batch_size,s);
            // forward_propagate and calculate the activation layer
	    multi_layers_forward(batch_size);
            cost_batch=cost_function(Y_batch,batch_size);
	    multi_layers_backward(Y_batch,batch_size);
	    weights_update(learning_rate);
            cost+=cost_batch;
        }

        cost/=n_sample/batch_size;
        // print the cost
        if(print_cost && i%100==0) {
            printf("Cost at epoch %d:  %.8f\n",i,cost);
        }
    }
    delete A,J,Y_batch,X_batch,Y,X;
}

void  dnn::predict(const vector<float>& _X,vector<int> &Y_prediction,const int &n_sample) {
    float *X=new float[_X.size()];
    float *A=new float[n_sample*n_classes];
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];

    // Forward propagation (from X to cost)
    forward_propagate(A,X,n_sample);
    get_softmax(A,n_sample);

    // get the predicted values
    int k_max;
    Y_prediction.assign(n_sample,0);
    for(int i=0; i<n_sample; i++) {
        float max_prob=max(A+i*n_classes,n_classes,k_max);
        Y_prediction[i]=k_max;
    }
    delete A,X;
}


float dnn::predict_accuracy(const vector<float>& _X,const vector<int> & Y,vector<int> &Y_prediction,const int &n_sample) {
    float *X=new float[_X.size()];
    float *A=new float[n_sample*n_classes];
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];

    // Forward propagation (from X to cost)
    forward_propagate(A,X,n_sample);
    get_softmax(A,n_sample);

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
