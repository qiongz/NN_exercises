#include"nn_lr.h"
#include<iostream>
#include"utils.h"
#define EPSILON 1e-10

void nn_lr::initialize_weights() {
    //srand(time(0));
    W=new float[n_features*n_classes];
    dW=new float[n_features*n_classes];
    b=new float[n_classes];
    db=new float[n_classes];
    for(int i=0; i<n_features*n_classes; i++)
        W[i]=rand()*1.0/RAND_MAX*0.01;
    //memset(W,0,sizeof(float)*n_classes*n_features);
    memset(b,0,sizeof(float)*n_classes);
}


//Fisher and Yates' shuffle algorithm
void nn_lr::shuffle(float *X,float *Y,int n_sample, int skip) {
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
	    memcpy(X_temp,X+i*n_features,sizeof(float)*n_features);
	    memcpy(X+i*n_features,X+j*n_features,sizeof(float)*n_features);
	    memcpy(X+j*n_features,X_temp,sizeof(float)*n_features);
	    memcpy(Y_temp,Y+i*n_classes,sizeof(float)*n_classes);
	    memcpy(Y+i*n_classes,Y+j*n_classes,sizeof(float)*n_classes);
	    memcpy(Y+j*n_classes,Y_temp,sizeof(float)*n_classes);
        }
    }
    vslDeleteStream(&rndStream);
    delete Y_temp,X_temp,i_buffer;
}

void nn_lr::batch(const float* X,const float *Y,float *X_batch,float *Y_batch,int batch_size,int batch_id) {
    cblas_scopy(n_features*batch_size,X+batch_size*batch_id*n_features,1,X_batch,1);
    cblas_scopy(n_classes*batch_size,Y+batch_size*batch_id*n_classes,1,Y_batch,1);
}

float nn_lr::cost_function(float *A,float *J,const float *Y,const int &n_sample) {
    vsLn(n_sample*n_classes,A,J);
    float cost=-cblas_sdot(n_sample*n_classes,Y,1,J,1)/n_sample;
    cost+=0.5*Lambda*cblas_sdot(n_classes*n_features,W,1,W,1)/n_sample;
    // for-loop version
    
    /*
    float cost=0;
    for(int i=0;i<n_classes*n_sample;i++)
      cost-=Y[i]*log(A[i]+EPSILON);
    for(int i=0;i<n_classes*n_features;i++)
      cost+=0.5*Lambda*pow(W[i],2);
    cost/=n_sample;
    */
   
    return cost;
}

float nn_lr::max(float *x,int range,int &index_max) {
    float max_val=x[0];
    index_max=0;
    for(int i=1; i<range; i++)
        if(x[i]>max_val) {
            max_val=x[i];
            index_max=i;
        }
    return max_val;
}

// softmax of A:  A[:,k]=exp(A[:,k])/\sum_j(exp(A[:,j]))
void nn_lr::get_softmax(float *A,const int &n_sample) {
    // calculate the softmax of A

    int id_max;
    
    for(int i=0; i<n_sample; i++) {
        // find the largest A and substract it to prevent overflow
        float A_max=max(A+i*n_classes,n_classes,id_max);
        for(int k=0; k<n_classes; k++) A[k+i*n_classes]-=A_max;
        vsExp(n_classes,A+i*n_classes,A+i*n_classes);   // A:=exp(A-A_max)
        float A_norm=1.0/cblas_sasum(n_classes,A+i*n_classes,1);
        for(int k=0; k<n_classes; k++)
            A[i*n_classes+k]*=A_norm;   // A:=exp(A-A_max)/\sum_i(exp(A_i-A_max))
    }
    

    // for-loop version for double-check
    /*
    for(int i=0;i<n_sample;i++){
    float A_max=max(A+i*n_classes,n_classes,id_max);
    float A_norm=0;
    for(int k=0;k<n_classes;k++) {
    	A[k+i*n_classes]-=A_max;
    	A[k+i*n_classes]=exp(A[k+i*n_classes]);
    	A_norm+=A[k+i*n_classes];
    }
    A_norm=1.0/A_norm;
        for(int k=0;k<n_classes;k++)
       A[i*n_classes+k]*=A_norm;   // A:=exp(A-A_max)/\sum_i(exp(A_i-A_max))
    }
    */
   
}

void nn_lr::gradient_approx(float *A,float *J,const float *X,const float *Y,const int &n_sample) {
    float epsilon=1e-5;
    float delta_cost;
    for(int i=0; i<n_classes; i++)
        for(int j=0; j<n_features; j++) {
            W[j+i*n_features]+=epsilon;
            forward_propagate(A,X,n_sample);
            get_softmax(A,n_sample);
            delta_cost=cost_function(A,J,Y,n_sample);
            W[j+i*n_features]-=2*epsilon;
            forward_propagate(A,X,n_sample);
            get_softmax(A,n_sample);
            delta_cost-=cost_function(A,J,Y,n_sample);
            dW[j+i*n_features]=delta_cost*0.5/epsilon;
            W[j+i*n_features]+=epsilon;
        }
    for(int j=0; j<n_classes; j++) {
        b[j]+=epsilon;
        forward_propagate(A,X,n_sample);
        get_softmax(A,n_sample);
        delta_cost=cost_function(A,J,Y,n_sample);
        b[j]-=2*epsilon;
        forward_propagate(A,X,n_sample);
        get_softmax(A,n_sample);
        delta_cost-=cost_function(A,J,Y,n_sample);

        db[j]=delta_cost*0.5/epsilon;
        b[j]+=epsilon;
    }
}

// Vectorized version of forward_propagate  for one-layer
// A=X*W.T+b
void nn_lr::forward_propagate(float *A,const float *X,const int &n_sample) {
    float alpha_=1;
    float beta_=0;
    
    // Forward propagation (from X to A1)
    // A(n_sample,n_classes) := X(n_sample,n_features) * W(n_classes, n_features).T,
    // A[j+i*n_classes]= sum_k (X[k+i*n_features]*W[k+j*n_features])
    // A(i,j): a[i+j*lda] for ColMajor, a[j+i*lda] for RowMajor, lda is the leading dimension
    // op(X)(n_sample,n_features), op(W)( n_features,n_classes)
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n_sample,n_classes,n_features,
                alpha_, X, n_features, W, n_features, beta_, A, n_classes);

    // A(n_sample,n_classes) := A(n_sample,n_classes)+b(:,n_classes)
    for(int i=0; i<n_sample; i++)
        vsAdd(n_classes,A+i*n_classes,b,A+i*n_classes);
   

    // for-loop version for double-check
    /*   
    for(int i=0;i<n_sample;i++)
      for(int j=0;j<n_classes;j++){
        A[i*n_classes+j]=0;
        for(int k=0;k<n_features;k++)
          A[i*n_classes+j]+=X[k+i*n_features]*W[k+j*n_features];
        A[i*n_classes+j]+=b[j];
      }
    */ 
}

// Vectorized version of backward_propagate
void nn_lr::backward_propagate(float *A, const float *X, const float *Y,const int &n_sample) {
    float alpha_=1.0/n_sample;
    float beta_=0;
   
    // Backward propagation (get gradients)
    vsSub(n_sample*n_classes,A,Y,A);     // dA(i,k):=A(i,k)-1{Y(i,k)==1}
    // dW(n_classes,n_features):= dA(n_sample,n_classes).T * X(n_sample,n_features)
    // op(dA)(n_class,n_sample), op(X)(n_sample,n_features)
    cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,n_classes,n_features,n_sample,
                 alpha_, A, n_classes, X, n_features, beta_, dW, n_features);
    // dW:=dW+Lambda*W
    if(Lambda>1e-8)
        cblas_saxpy(n_classes*n_features,Lambda/n_sample,W,1,dW,1);

    // db(n_classes):= sum(dA(n_sample,n_classes),axis=0)
    for(int i=0; i<n_classes; i++) {
        db[i]=0;
        for(int j=0; j<n_sample; j++)
            db[i]+=A[j*n_classes+i];
        db[i]/=n_sample;
    }
   


    // for-loop version for double-check
    /*
     for(int i=0;i<n_sample*n_classes;i++)
    A[i]=A[i]-Y[i];
     for(int i=0;i<n_classes;i++)
    for(int j=0;j<n_features;j++){
      dW[i*n_features+j]=Lambda*W[i*n_features+j];
      for(int k=0;k<n_sample;k++)
      dW[i*n_features+j]+=A[k*n_classes+i]*X[k*n_features+j];
      dW[i*n_features+j]/=n_sample;
    }
  */ 
}

void nn_lr::fit(const vector<float> &_X,const vector<int> &_Y,const int &n_sample, const int num_epochs=2000,const float learning_rate=0.01,const float _lambda=0,int batch_size=128,bool print_cost=false) {
    Lambda=_lambda;
    // tempory space for storing the shuffled training data sets
    float *X=new float[_X.size()];
    float *Y=new float[_Y.size()];
    // deep copy X,Y
    for(int i=0; i<_X.size(); i++) X[i]=_X[i];
    for(int i=0; i<_Y.size(); i++) Y[i]=_Y[i];

    // batch training data sets
    float *X_batch=new float[batch_size*n_features];
    float *Y_batch=new float[batch_size*n_classes];
    // tempory space used for vectorized calculation of b,cost
    float *J=new float[batch_size*n_classes];
    // activation layer
    float *A=new float[batch_size*n_classes];

    float *dW_approx=new float [n_classes*n_features];
    float *db_approx=new float [n_classes];
    float cost,cost_batch;
    for(int i=0; i<num_epochs; i++) {
        // shuffle training data sets for one epoch
        shuffle(X,Y,n_sample,i);
        cost=0;
        // batch training until all the data sets are used
        for(int s=0; s<n_sample/batch_size; s++) {
            batch(X,Y,X_batch,Y_batch,batch_size,s);
            // forward_propagate and calculate the activation layer
            forward_propagate(A,X_batch,batch_size);
            get_softmax(A,batch_size);
            cost_batch=cost_function(A,J,Y_batch,batch_size);
            //gradient_approx(A,J,X_batch,Y_batch,batch_size);
            //memcpy(dW_approx,dW,sizeof(float)*n_classes*n_features);
            //memcpy(db_approx,db,sizeof(float)*n_classes);
            backward_propagate(A,X_batch,Y_batch,batch_size);
	   /* 
            cout<<"dW difference"<<endl;
            for(int i=0; i<n_classes; i++) {
                for(int j=0; j<n_features; j++) {
                    float diff=dW_approx[i*n_features+j]-dW[j+i*n_features];
                    if(abs(diff/(dW_approx[i*n_features+j]+1e-8))>1e-5)
                        cout<<diff/(dW_approx[j+i*n_features]+1e-8)<<"  ";
                }
                cout<<endl;
            }
	    */
	    
	   
            // update the weights and bias
            cblas_saxpy(n_classes*n_features,-learning_rate,dW,1,W,1);   // W:=W-learning_rate*dW
            cblas_saxpy(n_classes,-learning_rate,db,1,b,1);              // b:=b-learning_rate*db
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

void  nn_lr::predict(const vector<float>& _X,vector<int> &Y_prediction,const int &n_sample) {
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


float nn_lr::predict_accuracy(const vector<float>& _X,const vector<int> & Y,vector<int> &Y_prediction,const int &n_sample) {
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
