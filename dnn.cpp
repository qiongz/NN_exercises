#include"dnn.h"
#define EPSILON 1e-10
void dnn::initialize_weights() {
    weights_seed=123457;
    default_random_engine rng(weights_seed);
    normal_distribution<float> dist_norm(0,1);
    // initialalize W[0],dW[0],b[0],db[0] with dummy NULL pointer
    W.push_back(NULL);
    dW.push_back(NULL);
    b.push_back(NULL);
    db.push_back(NULL);
    for(int i=1; i<n_layers; i++) {
        W.push_back(new float[layer_dims[i-1]*layer_dims[i]]);
        dW.push_back(new float[layer_dims[i-1]*layer_dims[i]]);
        b.push_back(new float[layer_dims[i]]);
        db.push_back(new float[layer_dims[i]]);
        for(int j=0; j<layer_dims[i-1]*layer_dims[i]; j++)
            if(i==n_layers-1)
                W[i][j]=dist_norm(rng)*sqrt(1.0/layer_dims[i-1]);
            else
                W[i][j]=dist_norm(rng)*sqrt(2.0/layer_dims[i-1]);  // factor "2" for ReLU
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

void dnn::batch(const float* X,float *X_batch,int batch_size,int batch_id) {
    cblas_scopy(n_features*batch_size,X+batch_size*batch_id*n_features,1,X_batch,1);
}

float dnn::cost_function(const float *Y,const int &n_sample) {
    // cost function should only be called after or without backward-propagation
    vsLn(n_sample*n_classes,A[n_layers-1],A[n_layers-1]);
    float cost=-cblas_sdot(n_sample*n_classes,Y,1,A[n_layers-1],1)/n_sample;
    if(Lambda>1e-10)
    for(int l=1;l<n_layers;l++)
       cost+=0.5*Lambda*cblas_sdot(layer_dims[l]*layer_dims[l-1],W[l],1,W[l],1)/n_sample;
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

// softmax-layer  A[L-1][:,k]=exp(Z[L-1][:,k])/\sum_j(exp(Z[L-1][:,j]))
void dnn::get_softmax(const int &n_sample) {
    // calculate the softmax of Z[L-1]
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

void dnn::sigmoid_activate(const int &l,const int &n_sample) {
    for(int i=0; i<layer_dims[l]*n_sample; i++)
        A[l][i]=-A[l][i];
    vsExp(layer_dims[l]*n_sample,A[l],A[l]);
    for(int i=0; i<layer_dims[l]*n_sample; i++)
        A[l][i]+=1;
    vsInv(layer_dims[l]*n_sample,A[l],A[l]);
}

// Vectorized version of forward_activated_propagate for each layer
// input: A[l-1],W[l],b[l]   output: A[l]
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
    if(activation_types[l]=="sigmoid")
        sigmoid_activate(l,n_sample);
    else if(activation_types[l-1]=="softmax")
	get_softmax(n_sample);
}

// Vectorized version of backward_propagate,
// input:  dZ[l],W[l],A[l-1]   output: db[l],dW[l],dZ[l-1]
void dnn::backward_propagate(const int & l,const int &n_sample) {
    float alpha_=1;
    float beta_=0;
    // calculate db_{l}:=sum(dZ_{l},axis=0)
    // db[l]:=sum(dZ[l](n_sample,layer_dims[l]),axis=0)
    for(int i=0; i<layer_dims[l]; i++) {
        db[l][i]=0;
        for(int j=0; j<n_sample; j++)
            db[l][i]+=dZ[l][j*layer_dims[l]+i];
    }

    // dW[l](n_l,n_{l-1}):= dZ[l](n_sample,n_{l}) * A[l-1](n_sample,n_{l-1})
    // op(dZ)(n_{l},n_sample), op(A[l-1])(n_sample,n_{l-1})
    cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,layer_dims[l],layer_dims[l-1],n_sample,
                 alpha_, dZ[l], layer_dims[l], A[l-1], layer_dims[l-1], beta_, dW[l], layer_dims[l]);

    // dW:=dW+Lambda*W
    if(Lambda>1e-8)
        cblas_saxpy(layer_dims[l]*layer_dims[l-1],Lambda/n_sample,W[l],1,dW[l],1);

    // only calculate dZ[l] for l=1,...,n_layers-1,
    // dZ[0] is stored as a dummy NULL pointer
    if(l-1>0) {
        // create tempory space for dA[l-1]/dZ[l-1]
        float *dF=new float[layer_dims[l-1]*n_sample];
        // for sigmoid activation function, f'(z)=f(z)(1-f(z))
        if(activation_types[l-1]=="sigmoid") {
            // dF=f'(Z_{l-1})=A_{l-1}*(1-A_{l-1}) =A_{l-1}-A_{l-1}*A_{l-1}
            vsMul(layer_dims[l-1]*n_sample,A[l-1],A[l-1],dF);
            vsSub(layer_dims[l-1]*n_sample,A[l-1],dF,dF);
        }
        // dZ[l-1]=dZ[l]*W[l].*f'(z)
        // calculate dZ[l](n_sample,n_{l})* W[l](n_{l},n_{l-1})
        cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,n_sample,layer_dims[l-1],layer_dims[l],
                     alpha_, dZ[l], layer_dims[l], W[l], layer_dims[l-1], beta_, dZ[l-1], layer_dims[l-1]);

        // dZ[l-1] <- dZ[l-1].*f'(z)
        vsMul(layer_dims[l]*n_sample,dZ[l-1],dF,dZ[l-1]);
        delete dF;
    }
}

void dnn::multi_layers_forward(const int &n_sample) {
    for(int l=1; l<n_layers; l++)
        forward_activated_propagate(l,n_sample);
    get_softmax(n_sample);
}

void dnn::multi_layers_backward(const float *Y,const int &n_sample) {
    // softmax layer: dZ[l-1]=(A-Y)/n_sample
    float alpha_=1.0/n_sample;
    vsSub(n_sample*layer_dims[n_layers-1],A[n_layers-1],Y,dZ[n_layers-1]);
    for(int i=0; i<n_sample*layer_dims[n_layers-1]; i++)  dZ[n_layers-1][i]*=alpha_;

    // get gradients
    for(int l=n_layers-1; l>=1; l--)
        backward_propagate(l,n_sample);
}

void dnn::weights_update(const float &learning_rate){
    for(int l=1;l<n_layers;l++){
       // W[l]:=W[l]-learning_rate*dW[l]
       cblas_saxpy(layer_dims[l]*layer_dims[l-1],-learning_rate,dW[l],1,W[l],1); 
       // b[l]:=b[l]-learning_rate*db[l]
       cblas_saxpy(layer_dims[l],-learning_rate,db[l],1,b[l],1); 
    }
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
    for(int l=0; l<n_layers; l++)
        A.push_back(new float[layer_dims[l]*batch_size]);

    // initialize the first-layer of dZ as dummy NULL pointer
    dZ.push_back(NULL);
    // l=1 to n_layers-1 of dZ are updated
    for(int l=1; l<n_layers; l++)
        dZ.push_back(new float[layer_dims[l]*batch_size]);

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
            multi_layers_backward(Y_batch,batch_size);
            cost_batch=cost_function(Y_batch,batch_size);
            weights_update(learning_rate);
            cost+=cost_batch;
        }

        cost/=n_sample/batch_size;
        // print the cost
        if(print_cost && i%100==0) {
            printf("Cost at epoch %d:  %.8f\n",i,cost);
        }
    }
    delete Y_batch;
    for(int l=0;l<n_layers;l++)
	    delete dZ[l],A[l];
    dZ.clear();
    A.clear();
    delete Y,X;
}

void  dnn::predict(const vector<float>& _X,vector<int> &Y_prediction,const int &n_sample,const int &batch_size) {
    float *X=new float[_X.size()];
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];
    // initialize layer caches and layer gradients
    for(int l=0; l<n_layers; l++)
        A.push_back(new float[layer_dims[l]*batch_size]);

    int n_batch=n_sample/batch_size;
    int n_residual=n_sample%batch_size;
    // get the predicted values
    int k_max;
    Y_prediction.assign(n_sample,0);
    // batch training until all the data sets are used
    for(int s=0; s<n_batch; s++) {
      // Forward propagation (from X to cost)
      batch(X,A[0],batch_size,s);
      multi_layers_forward(batch_size);
      for(int k=0;k<batch_size;k++){
	  float max_prob=max(A[n_layers-1]+(s*batch_size+k)*n_classes,n_classes,k_max);
	  Y_prediction[s*batch_size+k]=k_max;
      }
    }
    if(n_residual>0){
      cblas_scopy(n_residual*n_features,X+batch_size*(n_sample/batch_size),1,A[0],1);
      multi_layers_forward(batch_size);
      for(int k=0;k<n_residual;k++){
	  float max_prob=max(A[n_layers-1]+(n_batch*batch_size+k)*n_classes,n_classes,k_max);
	  Y_prediction[batch_size*batch_size+k]=k_max;
      }
    }

    for(int l=0; l<n_layers; l++)
        delete A[l];
    delete X;
}


float dnn::predict_accuracy(const vector<float>& _X,const vector<int> &Y,vector<int> &Y_prediction,const int &n_sample,const int &batch_size) {
    float *X=new float[_X.size()];
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];
    // initialize layer caches and layer gradients
    for(int l=0; l<n_layers; l++)
        A.push_back(new float[layer_dims[l]*batch_size]);

    int n_batch=n_sample/batch_size;
    int n_residual=n_sample%batch_size;
    // get the predicted values
    int k_max;
    Y_prediction.assign(n_sample,0);
    // batch training until all the data sets are used
    for(int s=0; s<n_batch; s++) {
      // Forward propagation (from X to cost)
      batch(X,A[0],batch_size,s);
      multi_layers_forward(batch_size);
      for(int k=0;k<batch_size;k++){
	  float max_prob=max(A[n_layers-1]+(s*batch_size+k)*n_classes,n_classes,k_max);
	  Y_prediction[s*batch_size+k]=k_max;
      }
    }
    if(n_residual>0){
      cblas_scopy(n_residual*n_features,X+batch_size*(n_sample/batch_size),1,A[0],1);
      multi_layers_forward(batch_size);
      for(int k=0;k<n_residual;k++){
	  float max_prob=max(A[n_layers-1]+(n_batch*batch_size+k)*n_classes,n_classes,k_max);
	  Y_prediction[batch_size*batch_size+k]=k_max;
      }
    }

    float accuracy=0;
    for(int i=0; i<n_sample; i++) 
        accuracy+=(Y[i]==k_max?1:0);
    accuracy/=n_sample;

    for(int l=0; l<n_layers; l++)
        delete A[l];
    delete X;
    return accuracy;
}
