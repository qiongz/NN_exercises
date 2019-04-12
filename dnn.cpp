#include"dnn.h"
#define EPSILON 1e-12

dnn::dnn(int n_f,int n_c){
	mkl_seed=time(0);
	n_features=n_f;
	n_classes=n_c;
	n_layers=2;
	layer_dims.push_back(n_features);
	layer_dims.push_back(n_classes);
	activation_types.push_back("NULL");
	activation_types.push_back("softmax");
        initialize_weights();
    }

dnn::dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string>& act_types){
	mkl_seed=time(0);
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
	keep_probs.assign(n_h,1);
        initialize_weights();
}

dnn::dnn(int n_f,int n_c,int n_h,const vector<int>& dims,const vector<string>& act_types,const vector<float>& k_ps){
	mkl_seed=time(0);
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
	dropout=true;
	keep_probs.assign(k_ps.begin(),k_ps.end());
        initialize_weights();
}

dnn::~dnn(){
      for(int i=0;i<n_layers;i++)
       delete db[i],b[i],dW[i],W[i];
      
      db.clear();
      b.clear();
      dW.clear();
      W.clear();
    }

void dnn::initialize_weights() {
    weights_seed=123457;
    default_random_engine rng(weights_seed);
    normal_distribution<float> dist_norm(0,1);
    // initialalize W[0],dW[0],b[0],db[0] with dummy NULL pointer
    float *null_ptr=NULL;
    W.push_back(null_ptr);
    dW.push_back(null_ptr);
    b.push_back(null_ptr);
    db.push_back(null_ptr);
    for(int l=1; l<n_layers; l++) {
        W.push_back(new float[layer_dims[l-1]*layer_dims[l]]);
        dW.push_back(new float[layer_dims[l-1]*layer_dims[l]]);
        b.push_back(new float[layer_dims[l]]);
        db.push_back(new float[layer_dims[l]]);
        for(int j=0; j<layer_dims[l-1]*layer_dims[l]; j++)
            if(l==n_layers-1 || activation_types[l]=="sigmoid")
                W[l][j]=dist_norm(rng)*sqrt(1.0/layer_dims[l-1]);
            else if(activation_types[l]=="ReLU" || activation_types[l]=="LeakyReLU")
                W[l][j]=dist_norm(rng)*sqrt(2.0/layer_dims[l-1]);  // factor "2" for ReLU
        memset(b[l],0,sizeof(float)*layer_dims[l]);
    }
}

//Fisher and Yates' shuffle algorithm
void dnn::shuffle(float *X,float *Y,int n_sample) {
    VSLStreamStatePtr rndStream;
    vslNewStream(&rndStream, VSL_BRNG_MT19937,mkl_seed);
    vslSkipAheadStream(rndStream,mkl_rnd_skipped);
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
    mkl_rnd_skipped+=n_sample;
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
    float *J=new float[layer_dims[n_layers-1]*n_sample];
    for(int i=0;i<n_sample*layer_dims[n_layers-1];i++) 
       J[i]=A[n_layers-1][i]+EPSILON;
    vsLn(n_sample*layer_dims[n_layers-1],J,J);
    float cost=-cblas_sdot(n_sample*layer_dims[n_layers-1],Y,1,J,1)/n_sample;
    if(Lambda>EPSILON)
    for(int l=1;l<n_layers;l++)
       cost+=0.5*Lambda*cblas_sdot(layer_dims[l]*layer_dims[l-1],W[l],1,W[l],1)/n_sample;
    delete J;
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
    int id_max;
    for(int i=0; i<n_sample; i++) {
        // find the largest A and substract it to prevent overflow
        float A_max=max(A[n_layers-1]+i*layer_dims[n_layers-1],layer_dims[n_layers-1],id_max);
        for(int k=0; k<layer_dims[n_layers-1]; k++) 
	  A[n_layers-1][k+i*layer_dims[n_layers-1]]-=A_max;
        vsExp(layer_dims[n_layers-1],A[n_layers-1]+i*layer_dims[n_layers-1],A[n_layers-1]+i*layer_dims[n_layers-1]);   // A:=exp(A-A_max)
        float A_norm=1.0/cblas_sasum(layer_dims[n_layers-1],A[n_layers-1]+i*layer_dims[n_layers-1],1);
        for(int k=0; k<layer_dims[n_layers-1]; k++)
	 // A:=exp(A-A_max)/\sum_i(exp(A_i-A_max))
         A[n_layers-1][i*layer_dims[n_layers-1]+k]*=A_norm;   
    }
}

void dnn::sigmoid_activate(const int &l,const int &n_sample) {
    for(int i=0; i<layer_dims[l]*n_sample; i++)
        A[l][i]=-A[l][i];
    vsExp(layer_dims[l]*n_sample,A[l],A[l]);
    for(int i=0; i<layer_dims[l]*n_sample; i++)
        A[l][i]+=1;
    vsInv(layer_dims[l]*n_sample,A[l],A[l]);
}

void dnn::ReLU_activate(const int &l,const int &n_sample){
    for(int i=0;i<layer_dims[l]*n_sample;i++)
      A[l][i]=(A[l][i]>0?A[l][i]:0);
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
    else if(activation_types[l]=="ReLU")
        ReLU_activate(l,n_sample);
    else if(activation_types[l]=="softmax")
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
                 alpha_, dZ[l], layer_dims[l], A[l-1], layer_dims[l-1], beta_, dW[l], layer_dims[l-1]);

    // dW:=dW+Lambda*W
    if(Lambda>EPSILON)
        cblas_saxpy(layer_dims[l]*layer_dims[l-1],Lambda/n_sample,W[l],1,dW[l],1);

    // only calculate dZ[l] for l=1,...,n_layers-1,
    // dZ[0] is stored as a dummy NULL pointer
    if(l-1>0) {
        // create tempory space for dA[l-1]/dZ[l-1]
        float *dF=new float[layer_dims[l-1]*n_sample];
	memset(dF,1,sizeof(float)*layer_dims[l-1]*n_sample);
        // for sigmoid activation function, f'(z)=f(z)(1-f(z))
        if(activation_types[l-1]=="sigmoid") 
	{
            // dF=f'(Z_{l-1})=A_{l-1}*(1-A_{l-1}) =A_{l-1}-A_{l-1}*A_{l-1}
            vsMul(layer_dims[l-1]*n_sample,A[l-1],A[l-1],dF);
            vsSub(layer_dims[l-1]*n_sample,A[l-1],dF,dF);
        }
	else if(activation_types[l-1]=="ReLU"){
	    for(int i=0;i<layer_dims[l-1]*n_sample;i++)
		dF[i]=(A[l-1][i]>0?1:0); 
	}
        // dZ[l-1]=dZ[l]*W[l].*f'(z)
        // calculate dZ[l](n_sample,n_{l-1}=dZ[l](n_sample,n_{l})* W[l](n_{l},n_{l-1})
        cblas_sgemm (CblasRowMajor,CblasNoTrans,CblasNoTrans,n_sample,layer_dims[l-1],layer_dims[l],
                     alpha_, dZ[l], layer_dims[l], W[l], layer_dims[l-1], beta_, dZ[l-1], layer_dims[l-1]);

        // dZ[l-1] <- dZ[l-1].*f'(z)
        vsMul(layer_dims[l-1]*n_sample,dZ[l-1],dF,dZ[l-1]);
        delete dF;
    }
}

void dnn::multi_layers_forward(const int &n_sample) {
    for(int l=1; l<n_layers; l++)
        forward_activated_propagate(l,n_sample);
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


void dnn::dropout_regularization(){
    VSLStreamStatePtr rndStream;
    vslNewStream(&rndStream, VSL_BRNG_MT19937,mkl_seed);
    vslSkipAheadStream(rndStream,mkl_rnd_skipped);
    float *f_buffer=new float[32];
    for(int l=1;l<n_layers-1;l++){
      int count=0;
      do{
        vsRngUniform(METHOD_FLOAT,rndStream,32,f_buffer,0,1.0);
	for(int j=0;j<32;j++)
	    if(count+j<layer_dims[l])
	      A[l][count+j]=(A[l][count+j]<keep_probs[l]?1:0)/keep_probs[l];
	    else
              break;
	count+=32;
	mkl_rnd_skipped+=32;
      }while(count<layer_dims[l]);
    }
}

void dnn::weights_update(const float &learning_rate){
    for(int l=1;l<n_layers;l++){
       // W[l]:=W[l]-learning_rate*dW[l]
       cblas_saxpy(layer_dims[l]*layer_dims[l-1],-learning_rate,dW[l],1,W[l],1); 
       // b[l]:=b[l]-learning_rate*db[l]
       cblas_saxpy(layer_dims[l],-learning_rate,db[l],1,b[l],1); 
    }
}

void dnn::fit(const vector<float> &_X,const vector<int> &_Y,const int &n_sample, const int num_epochs=500,const float learning_rate=0.01,const float _lambda=0,int batch_size=128,bool print_cost=false) {
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
    int num_batches=n_sample/batch_size;
    for(int i=0; i<num_epochs; i++) {
        // shuffle training data sets for one epoch
        shuffle(X,Y,n_sample);
        cost=0;
        // batch training until all the data sets are used
        for(int s=0; s<num_batches; s++) {
            // feed A[0] with X_batch
            batch(X,Y,A[0],Y_batch,batch_size,s);
            // forward_propagate and calculate the activation layer
            multi_layers_forward(batch_size);
            multi_layers_backward(Y_batch,batch_size);
            cost_batch=cost_function(Y_batch,batch_size);
            weights_update(learning_rate);
            cost+=cost_batch;
        }
	if(dropout==true)
          dropout_regularization();
        cost/=num_batches;
        // print the cost
        if(print_cost && i%10==0) {
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

void dnn::train_and_dev(const vector<float> &_X_train,const vector<int> &_Y_train,const vector<float>& _X_dev, const vector<int>& _Y_dev,const int &n_train, const int &n_dev, const int num_epochs=500,const float learning_rate=0.01,const float _lambda=0,int batch_size=128,bool print_cost=false) {
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
    float cost_train,cost_dev,cost_batch;
    int num_train_batches=n_train/batch_size;
    int num_dev_batches=n_dev/batch_size; // only use num_dev_batches*batch_size for simplity
    long num_rnd_skipped;
    for(int i=0; i<num_epochs; i++) {
        // shuffle training data sets for one epoch
        shuffle(X_train,Y_train,n_train);
        cost_train=0;
        // batch training until all the data sets are used
        for(int s=0; s<num_train_batches; s++) {
            // feed A[0] with X_batch
            batch(X_train,Y_train,A[0],Y_batch,batch_size,s);
            // forward_propagate and calculate the activation layer
            multi_layers_forward(batch_size);
            multi_layers_backward(Y_batch,batch_size);
            cost_batch=cost_function(Y_batch,batch_size);
            weights_update(learning_rate);

            cost_train+=cost_batch;
        }
	if(dropout==true)
          dropout_regularization();

        cost_train/=num_train_batches;
        // print the cost
        if(print_cost && i%10==0) {
            // calculate the cost for dev set
	    cost_dev=0;	
            for(int s=0; s<num_dev_batches; s++) {
            // feed A[0] with X_batch
            batch(X_dev,Y_dev,A[0],Y_batch,batch_size,s);
            // forward_propagate and calculate the activation layer
            multi_layers_forward(batch_size);
            cost_batch=cost_function(Y_batch,batch_size);
            cost_dev+=cost_batch;
            }
	    cost_dev/=num_dev_batches;

            printf("Cost of train/dev at epoch %d :  %.8f %.8f \n",i,cost_train,cost_dev);
        }
    }
    delete Y_batch;
    for(int l=0;l<n_layers;l++)
	    delete dZ[l],A[l];
    dZ.clear();
    A.clear();
    delete Y_dev,X_dev,Y_train,X_train;
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
	  float max_prob=max(A[n_layers-1]+k*n_classes,n_classes,k_max);
	  Y_prediction[s*batch_size+k]=k_max;
      }
    }
    if(n_residual>0){
      cblas_scopy(n_residual*n_features,X+batch_size*n_batch*n_features,1,A[0],1);
      multi_layers_forward(batch_size);
      for(int k=0;k<n_residual;k++){
	  float max_prob=max(A[n_layers-1]+k*n_classes,n_classes,k_max);
	  Y_prediction[n_batch*batch_size+k]=k_max;
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
    Y_prediction.assign(n_sample,-1);
    for(int s=0; s<n_batch; s++) {
      memcpy(A[0],X+s*batch_size*n_features,sizeof(float)*batch_size*n_features);
      multi_layers_forward(batch_size);
      for(int k=0;k<batch_size;k++){
	  int j=s*batch_size+k;
	  float max_prob=max(A[n_layers-1]+k*n_classes,n_classes,k_max);
	  Y_prediction[j]=k_max;
      }
    }
    if(n_residual>0){
      memcpy(A[0],X+n_batch*batch_size*n_features,sizeof(float)*n_residual*n_features);
      multi_layers_forward(batch_size);
      for(int k=0;k<n_residual;k++){
	  int j=n_batch*batch_size+k;
	  float max_prob=max(A[n_layers-1]+k*n_classes,n_classes,k_max);
	  Y_prediction[j]=k_max;
      }
    }

    float accuracy=0;
    for(int i=0; i<n_sample; i++) 
        accuracy+=(Y[i]==Y_prediction[i]?1:0);
    accuracy/=n_sample;

    for(int l=0; l<n_layers; l++)
        delete A[l];
    delete X;
    return accuracy;
}
