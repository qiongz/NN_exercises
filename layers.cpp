#include"layers.h"

layers::~layers() {
    if(layer_type!="Input" && is_init==true) {
        delete W,b,dW,db;
        if(dropout==true){
            delete dropout_mask;
            vslDeleteStream(&rndStream);
	}
        if(optimizer=="Adam")
            delete VdW,Vdb,SdW,Sdb;
        if(batch_norm==true)
            delete B,dB,G,dG;
    }
}

float layers::getmax(float *x,const int &range) {
    float max_val=x[0];
    for(int i=0; i<range; i++)
        if(x[i]>max_val)
            max_val=x[i];
    return max_val;
}


void layers::sigmoid_activate() {
    // A=-A
    for(int i=0; i<n_sample*dim; i++)
        A[i]=-A[i];
    // A=exp(A)
    vsExp(n_sample*dim,A,A);
    // A=A+1
    for(int i=0; i<n_sample*dim; i++)
        A[i]+=1;
    // A=1/A
    vsInv(n_sample*dim,A,A);
}

void layers::ReLU_activate() {
    // Z (stored in A)
    // A= Z, if Z>0
    //       0,    otherwise
    for(int i=0; i<n_sample*dim; i++)
        A[i]=(A[i]>0?A[i]:0);
}

void layers::sigmoid_backward() {
    // dA= A*(1-A) =A-A*A
    vsMul(n_sample*dim,A,A,dA);
    vsSub(n_sample*dim,A,dA,dA);
    // dZ=dA.*dZ
    vsMul(n_sample*dim,dZ,dA,dZ);
}

void layers::ReLU_backward() {
    // dA=\partial{A}/\partial{Z}
    // dA= 1, if A>0
    //     0, otherwise
    for(int i=0; i<n_sample*dim; i++)
        dA[i]=(A[i]>0?1:0);
    vsMul(n_sample*dim,dZ,dA,dZ);
}

// softmax  A[:,k]=exp(Z[:,k])/\sum_j(exp(Z[:,j]))
void layers::get_softmax() {
    // calculate the softmax for each sample of A
    for(int i=0; i<n_sample; i++) {
        // find the largest value A and substract it to prevent overflow
        float A_max=getmax(A+i*dim,dim);
        // substract the largest value, A[i]=A[i]-A_max
        for(int k=0; k<dim; k++)
            A[k+i*dim]-=A_max;
        // A=exp(A)
        vsExp(dim,A+i*dim,A+i*dim);
        float A_norm=1.0/cblas_sasum(dim,A+i*dim,1);
        // A[i]=exp(A[i])/\sum_i(exp(A[j]))
        cblas_sscal(dim,A_norm,A+i*dim,1);
    }
}

void layers::print_layer_parameters() {
    cout<<"layer type: "<<layer_type<<endl;
    cout<<"dim: "<<dim<<", ";
    if(prev!=NULL)
        cout<<"  prev->dim: "<<prev->dim<<endl;
    else
        cout<<endl;
    cout<<"activation: "<<activation<<endl;
    cout<<"dropout: "<<dropout<<endl;
    if(dropout==true)
        cout<<"keep prob: "<<keep_prob<<endl;
    cout<<"batch normalization: "<<batch_norm<<endl;
}


void layers::initialize(const int &_n,const float &_lambda, const string &_optimizer,const bool &_batch_norm) {
    n_sample=_n;
    Lambda=_lambda;
    optimizer=_optimizer;
    batch_norm=_batch_norm;
    is_init=true;
    init_caches(_n,true);
    if(dropout==true)
        init_dropout_mask();
    if(layer_type!="Input") {
        init_weights();
        if(optimizer=="Adam")
            init_momentum_rms();
        if(batch_norm==true)
            init_batch_norm_weights();
    }
}

void layers::init_dropout_mask() {
    mkl_seed=time(0);
    vslNewStream(&rndStream, VSL_BRNG_MT19937,mkl_seed);
    dropout_mask=new float[dim];
}

void layers::set_dropout_mask() {
    // number generation
    vsRngUniform(METHOD_FLOAT,rndStream,dim,dropout_mask,0,1.0);
    // randomly assign 1/0 according to the
    // keep probabilities in each layer
    for(int j=0; j<dim; j++)
        dropout_mask[j]=(dropout_mask[j]<keep_prob?1:0);
    // scale the masks by 1/keep_probs[l]
    cblas_sscal(dim,1.0/keep_prob,dropout_mask,1);
}

void layers::init_weights() {
    weights_seed=time(0);
    default_random_engine rng(weights_seed);
    // random number drawn from normal distribution with  mean=0 and std=1
    normal_distribution<float> dist_norm(0,1);
    W=new float[dim*prev->dim];
    dW=new float[dim*prev->dim];
    b=new float[dim];
    db=new float[dim];
    // initialize weights from random No.s from normal distribution
    for(int j=0; j<dim*prev->dim; j++)
        // scale W with sqrt(1/dim) to prevent Z becoming too large/small
        // so that the gradients won't vanishing/exploding
        // sqrt(1/dim) for "sigmoid"
        if(activation=="sigmoid" || activation=="softmax")
            W[j]=dist_norm(rng)*sqrt(1.0/(prev->dim));
    // sqrt(2/dim) for "ReLU"
        else if(activation=="ReLU")
            W[j]=dist_norm(rng)*sqrt(2.0/(prev->dim));
    // initialize bias b with zeros
    memset(b,0,sizeof(float)*dim);
}

void layers::init_momentum_rms() {
    /// create memory space for Adam optimizer variables
    VdW=new float[dim*prev->dim];
    SdW=new float[dim*prev->dim];
    Vdb=new float[dim];
    Sdb=new float[dim];
    VdW_corrected=new float[dim*prev->dim];
    SdW_corrected=new float[dim*prev->dim];
    Vdb_corrected=new float[dim];
    Sdb_corrected=new float[dim];
    memset(VdW,0,sizeof(float)*dim*prev->dim);
    memset(SdW,0,sizeof(float)*dim*prev->dim);
    memset(Vdb,0,sizeof(float)*dim);
    memset(Sdb,0,sizeof(float)*dim);
}

void layers::init_batch_norm_weights() {
    default_random_engine rng(weights_seed);
    // random number drawn from normal distribution with  mean=0 and std=1
    normal_distribution<float> dist_norm(0,1);
    B=new float[dim];
    dB=new float[dim];
    G=new float[dim*prev->dim];
    dG=new float[dim*prev->dim];
    for(int j=0; j<dim*prev->dim; j++)
        // scale the G to the same range to W
        // sqrt(1/prev->dim) for "sigmoid"
        if(activation=="sigmoid")
            G[j]=dist_norm(rng)*sqrt(1.0/prev->dim);
    // sqrt(2/prev->dim) for "ReLU"
        else if(activation=="ReLU")
            G[j]=dist_norm(rng)*sqrt(2.0/prev->dim);

    memset(B,0,sizeof(float)*dim);
}

void layers::init_caches(const int &_n_sample,const bool &is_bp) {
    n_sample=_n_sample;
    A=new float[dim*n_sample];
    if(prev!=NULL && is_bp==true) {
        dZ=new float[dim*n_sample];
        dA=new float[n_sample*dim];
    }
}

void layers::clear_caches(const bool &is_bp) {
    delete A;
    if(prev!=NULL && is_bp==true)
        delete dZ,dA;
}


// Vectorized version of forward_activated_propagate for the layers layer
// input: prev->A,W,b,prev->dropout_mask   output: A
void layers::forward_activated_propagate(const bool &eval) {
    float alpha_=1;
    float beta_=0;
    // if dropout is used, A[l-1]=A[l-1].*DropM[l-1], (softmax layer does not dropout)
    if(prev->dropout==true && eval==false) {
        prev->set_dropout_mask();
        for(int i=0; i<n_sample; i++)
            vsMul(prev->dim,prev->A+i*prev->dim,prev->dropout_mask,prev->A+i*prev->dim);
    }

    // linear forward propagation (from A[l-1] to Z[l])
    // Z[l](n_sample,n_l) := A[l-1](n_sample,n_{l-1}) * W(n_l, n_{l-1}).T,
    // Z(i,j): a[i+j*lda] for ColMajor, a[j+i*lda] for RowMajor, lda is the leading dimension
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n_sample,dim,prev->dim,
                alpha_, prev->A, prev->dim, W, prev->dim, beta_, A, dim);

    // Z[l]:= Z[l]+b[l]
    for(int i=0; i<n_sample; i++)
        vsAdd(dim,A+i*dim,b,A+i*dim);

    // activation output: A= activation_function(Z)
    if(activation=="sigmoid")
        sigmoid_activate();
    else if(activation=="ReLU")
        ReLU_activate();
    else if(activation=="softmax")
        get_softmax();
}


// Vectorized version of backward_propagate for the layers layer
// input:  dZ[l],W[l],A[l-1]   output: db[l],dW[l],dZ[l-1]
void layers::backward_propagate() {
    float alpha_=1;
    float beta_=0;

    // calculate db_{l}:=sum(dZ_{l},axis=0)
    // db[l]:=sum(dZ[l](n_sample,dim[l]),axis=0)
    for(int i=0; i<dim; i++) {
        db[i]=0;
        for(int j=0; j<n_sample; j++)
            db[i]+=dZ[j*dim+i];
    }

    // dW[l](n_l,n_{l-1}):= dZ[l](n_sample,n_{l}) * A[l-1](n_sample,n_{l-1})
    // op(dZ)(n_{l},n_sample), op(A[l-1])(n_sample,n_{l-1})
    cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,dim,prev->dim,n_sample,
                 alpha_, dZ, dim, prev->A, prev->dim, beta_, dW, prev->dim);

    // dW:=dW+Lambda*W
    if(Lambda>EPSILON)
        cblas_saxpy(dim*prev->dim,Lambda/n_sample,W,1,dW,1);

    // only calculate dZ[l] for l=1,...,n_layers-1,
    if(prev->layer_type!="Input") {
        // dZ[l-1]=dZ[l]*W[l]
        // calculate dZ[l](n_sample,n_{l-1}=dZ[l](n_sample,n_{l})* W[l](n_{l},n_{l-1})
        cblas_sgemm (CblasRowMajor,CblasNoTrans,CblasNoTrans,n_sample,prev->dim,dim,
                     alpha_, dZ, dim, W, prev->dim, beta_,prev->dZ, prev->dim);

        // dZ[l-1] <- dZ[l-1].*dF
        if(prev->activation=="sigmoid")
            prev->sigmoid_backward();
        else if(prev->activation=="ReLU")
            prev->ReLU_backward();
    }

}

void layers::gradient_descent_optimize(const float &initial_learning_rate,const int &num_epochs,const int &step) {
    // learning rate decay
    float learning_rate=initial_learning_rate*(1-step*1.0/num_epochs)+0.01*initial_learning_rate;
    // W[l]:=W[l]-learning_rate*dW[l]
    cblas_saxpy(dim*prev->dim,-learning_rate,dW,1,W,1);
    // b[l]:=b[l]-learning_rate*db[l]
    cblas_saxpy(dim,-learning_rate,db,1,b,1);
}

void layers::Adam_optimize(const float &initial_learning_rate,const float &beta_1,const float &beta_2,const int &num_epochs,const int &epoch_step,const int &train_step) {
    // learning rate decay
    float learning_rate=initial_learning_rate*(1-epoch_step*1.0/num_epochs)+0.01*initial_learning_rate;
    float bias_beta_1,bias_beta_2;
    bias_beta_1=1.0/(1-pow(beta_1,train_step+1));
    bias_beta_2=1.0/(1-pow(beta_2,train_step+1));

    // VdW[l]=beta_1*VdW[l]
    cblas_sscal(dim*prev->dim,beta_1,VdW,1);
    // VdW[l]+=(1-beta_1)*dW[l]
    cblas_saxpy(dim*prev->dim,(1-beta_1),dW,1,VdW,1);
    // Vdb[l]=beta_1*Vdb[l]
    cblas_sscal(dim,beta_1,Vdb,1);
    // Vdb[l]+=(1-beta_1)*db[l]
    cblas_saxpy(dim,(1-beta_1),db,1,Vdb,1);

    // SdW[l]=beta_2*SdW[l]
    cblas_sscal(dim*prev->dim,beta_2,SdW,1);
    //dW[l] replaced by dW[l]*dW[l], (will not be used anymore)
    vsMul(dim*prev->dim,dW,dW,dW);
    // SdW[l]+=(1-beta_2)*dW[l]
    cblas_saxpy(dim*prev->dim,(1-beta_2),dW,1,SdW,1);
    // Sdb[l]=beta_2*Sdb[l]
    cblas_sscal(dim,beta_2,Sdb,1);
    //db[l] replaced by db[l]*db[l], (will not be used anymore)
    vsMul(dim,db,db,db);
    // Sdb[l]+=(1-beta_2)*db[l]
    cblas_saxpy(dim,(1-beta_2),db,1,Sdb,1);

    memcpy(VdW_corrected,VdW,sizeof(float)*dim*prev->dim);
    memcpy(SdW_corrected,SdW,sizeof(float)*dim*prev->dim);
    memcpy(Vdb_corrected,Vdb,sizeof(float)*dim);
    memcpy(Sdb_corrected,Sdb,sizeof(float)*dim);

    // get bias corrected momentum and rms
    cblas_sscal(dim*prev->dim, bias_beta_1,VdW_corrected,1);
    cblas_sscal(dim, bias_beta_1,Vdb_corrected,1);
    cblas_sscal(dim*prev->dim, bias_beta_2,SdW_corrected,1);
    cblas_sscal(dim, bias_beta_2,Sdb_corrected,1);
    // SdW,Sdb=sqrt(SdW),sqrt(Sdb)
    vsSqrt(dim*prev->dim,SdW_corrected,SdW_corrected);
    vsSqrt(dim,Sdb_corrected,Sdb_corrected);
    // add epsilon to prevent overflow
    for(int i=0; i<dim*prev->dim; i++)
        SdW_corrected[i]+=1e-8;
    for(int i=0; i<dim; i++)
        Sdb_corrected[i]+=1e-8;
    // VdW,Vdb= VdW/sqrt(SdW+epsilon),Vdb/sqrt(Sdb+epsilon)
    vsDiv(dim*prev->dim,VdW_corrected,SdW_corrected,VdW_corrected);
    vsDiv(dim,Vdb_corrected,Sdb_corrected,Vdb_corrected);

    // update the weights and bias
    cblas_saxpy(dim*prev->dim,-learning_rate,VdW_corrected,1,W,1);
    cblas_saxpy(dim,-learning_rate,Vdb_corrected,1,b,1);
}
