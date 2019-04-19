#include"layers.h"

layers::~layers() {
  if(is_init) {
    if(layer_type=="Hidden" || layer_type=="Conv2d" || layer_type=="Output") {
      delete W,b,dW,db;
      if(dropout==true) {
        delete dropout_mask;
        vslDeleteStream(&rndStream);
      }
      if(optimizer=="Adam")
        delete VdW,Vdb,SdW,Sdb;
      if(batch_norm==true)
        delete B,dB,G,dG;
    }
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
  // dZ= A*(1-A) =A-A*A
  vsMul(n_sample*dim,A,A,dZ);
  vsSub(n_sample*dim,A,dZ,dZ);
  // dZ=dA.*dZ
  vsMul(n_sample*dim,dZ,dA,dZ);
}

void layers::ReLU_backward() {
  // dA=\partial{A}/\partial{Z}
  // dA= 1, if A>0
  //     0, otherwise
  for(int i=0; i<n_sample*dim; i++)
    dZ[i]=(A[i]>0?1:0);
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

void layers::print_parameters() {
  if(layer_type=="Input") {
    cout<<"layer type: "<<layer_type<<endl;
    cout<<"         L: "<<L<<endl;
    cout<<"       dim: "<<dim<<endl;
    cout<<" n_channel: "<<n_channel<<endl;
    if(dropout==true)
      cout<<" keep_prob: "<<keep_prob<<endl;
    cout<<"activation: "<<activation<<endl;
    cout<<endl;
  }
  else if (layer_type=="Hidden" || layer_type=="Output") {
    cout<<"layer type: "<<layer_type<<endl;
    cout<<"       dim: "<<dim<<endl;
    cout<<"     dim_W: ( "<<dim<<" x "<<prev->dim<<" )"<<endl;
    if(dropout==true)
      cout<<" keep_prob: "<<keep_prob<<endl;
    cout<<"activation: "<<activation<<endl;
    cout<<endl;
  }
  else if (layer_type=="Conv2d") {
    cout<<"layer type: "<<layer_type<<endl;
    cout<<"         L: "<<L<<endl;
    cout<<"       dim: ( "<<n_channel<<" x "<<L<<" x "<<L<<" )"<<endl;
    cout<<"    filter: ( "<<filter_size<<" x "<<filter_size<<" )"<<endl;
    cout<<"     dim_W: ( "<<n_channel<<" x "<<prev->n_channel<<" x "<<filter_size<<" x "<<filter_size<<" )"<<endl;
    cout<<" n_channel: "<<n_channel<<endl;
    cout<<"  paddling: "<<paddling<<endl;
    cout<<"    stride: "<<stride<<endl;
    cout<<"activation: "<<activation<<endl;
    if(dropout==true)
      cout<<"keep prob: "<<keep_prob<<endl;
    cout<<endl;
  }
  else if (layer_type=="Pool") {
    cout<<"layer type: "<<layer_type<<endl;
    cout<<"         L: "<<L<<endl;
    cout<<"       dim: ( "<<n_channel<<" x "<<L<<" x "<<L<<" )"<<endl;
    cout<<"    filter: ( "<<filter_size<<" x "<<filter_size<<" )"<<endl;
    cout<<" n_channel: "<<n_channel<<endl;
    cout<<"    stride: "<<stride<<endl;
    cout<<endl;
  }
}


void layers::initialize(const int &_n,const float &_lambda, const string &_optimizer,const bool &_batch_norm) {
  n_sample=_n;
  Lambda=_lambda;
  optimizer=_optimizer;
  batch_norm=_batch_norm;
  if(!is_init) {
    if(layer_type=="Conv2d") {
      L=(prev->L-filter_size+2*paddling)/stride+1;
      area=L*L;
      dim=area*n_channel;
      dim_W=n_channel*filter_size*filter_size*prev->n_channel;
      dim_b=dim;

      init_weights();
      if(optimizer=="Adam")
        init_momentum_rms();
      if(dropout==true)
        init_dropout_mask();
    }
    if(layer_type=="Pool") {
      paddling=0;
      L=(prev->L-filter_size)/stride+1;
      area=L*L;
      filter_area=filter_size*filter_size;
      dim=area*n_channel;

      if(dropout==true)
        init_dropout_mask();
    }
    else if(layer_type=="Hidden" || layer_type=="Output") {
      n_channel=1;
      dim_W=dim*prev->dim;
      dim_b=dim;
      init_weights();
      if(optimizer=="Adam")
        init_momentum_rms();
      if(dropout==true)
        init_dropout_mask();
    }
    else if(layer_type=="Input")
      if(dropout==true)
        init_dropout_mask();

    init_caches(n_sample,true);
  }
  is_init=true;
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
  W=new float[dim_W];
  dW=new float[dim_W];
  b=new float[dim_b];
  db=new float[dim_b];
  // initialize weights from random No.s from normal distribution
  for(int j=0; j<dim_W; j++)
    // scale W with sqrt(1/dim) to prevent Z becoming too large/small
    // so that the gradients won't vanishing/exploding
    // sqrt(1/dim) for "sigmoid"
    if(activation=="sigmoid" || activation=="softmax")
      W[j]=dist_norm(rng)*sqrt(1.0/(prev->dim));
  // sqrt(2/dim) for "ReLU"
    else if(activation=="ReLU")
      W[j]=dist_norm(rng)*sqrt(2.0/(prev->dim));
  // initialize bias b with zeros
  memset(b,0,sizeof(float)*dim_b);
}

void layers::init_momentum_rms() {
  /// create memory space for Adam optimizer variables
  VdW=new float[dim_W];
  SdW=new float[dim_W];
  Vdb=new float[dim_b];
  Sdb=new float[dim_b];
  VdW_corrected=new float[dim_W];
  SdW_corrected=new float[dim_W];
  Vdb_corrected=new float[dim_b];
  Sdb_corrected=new float[dim_b];
  memset(VdW,0,sizeof(float)*dim_W);
  memset(SdW,0,sizeof(float)*dim_W);
  memset(Vdb,0,sizeof(float)*dim_b);
  memset(Sdb,0,sizeof(float)*dim_b);
}

void layers::init_batch_norm_weights() {
  default_random_engine rng(weights_seed);
  // random number drawn from normal distribution with  mean=0 and std=1
  normal_distribution<float> dist_norm(0,1);
  B=new float[dim_b];
  dB=new float[dim_b];
  G=new float[dim_W];
  dG=new float[dim_W];
  for(int j=0; j<dim_W; j++)
    // scale the G to the same range to W
    // sqrt(1/prev->dim) for "sigmoid"
    if(activation=="sigmoid")
      G[j]=dist_norm(rng)*sqrt(1.0/prev->dim);
  // sqrt(2/prev->dim) for "ReLU"
    else if(activation=="ReLU")
      G[j]=dist_norm(rng)*sqrt(2.0/prev->dim);

  memset(B,0,sizeof(float)*dim_b);
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

  /// Z(n_sample,n_channel,L,L)=W(n_channel,filter_size,filter_size,prev->n_channel)*prev->A(n_sample,prev->n_channel,prev->L,prev->L)
  /// L=(prev->L+2*paddling-filter_size)/stride+1, sum over (filter_size/stride, filter_size/stride,prev->n_channel)
  if(layer_type=="Conv2d") {
    long index_A,index_W,index_prev_A;
    for(int m=0; m<n_sample; m++)
      for(int n=0; n<n_channel; n++)
        for(int i=0; i<L; i++)
          for(int j=0; j<L; j++) {
            index_A=m*dim+n*area+i*L+j;
            A[index_A]=b[n*area+i*L+j];
            for(int l=0; l<prev->n_channel; l++)
              for(int fl=0; fl<filter_size; fl++) {
                // calcuate index outside of the innermost loop
                int il=fl+i*stride-paddling;
                if(il>=0 &&il<prev->L) {
                  index_W=n*prev->n_channel*filter_area+l*filter_area+fl*filter_size;
                  index_prev_A=m*prev->dim+l*prev->area+il*prev->L+j*stride-paddling;
                  // if convolution prod with paddling
                  if(j*stride-paddling<0 || j*stride-paddling+filter_size>=prev->L)
                    for(int fw=0; fw<filter_size; fw++) {
                      int iw=fw+j*stride-paddling;
                      if(iw>=0 && iw<prev->L)
                        A[index_A]+=W[index_W+fw]*prev->A[index_prev_A+fw];
                    }
                  // convolution within in A, just use dot-product
                  else
                    A[index_A]+=cblas_sdot(filter_size,W+index_W,1,prev->A+index_prev_A,1);
                }
              }
          }
  }
  /// Z(n_sample,n_channel,L,L)=max{ prev->A(n_sample,prev->n_channel,prev->L,prev->L) ,(filter_size,stride)}
  /// L=(prev->L-filter_size)/stride+1, sum over (filter_size,filter_size)/stride, leave n_channel unchanged
  else if (layer_type=="Pool") {
    long index_A,argmax_Z,pool_index,pool_outside_index;
    for(int m=0; m<n_sample; m++)
      // n_channel==prev->n_channel
      for(int n=0; n<n_channel; n++)
        for(int i=0; i<L; i++)
          for(int j=0; j<L; j++) {
            /// max pooling, does not change No. of channels
            index_A=m*dim+n*area+i*L+j;
            argmax_Z=m*prev->dim+n*prev->area+i*stride*prev->L+j*stride;
            for(int fl=0; fl<filter_size; fl++) {
              int il=fl+i*stride;
              pool_outside_index=m*prev->dim+n*prev->area+il*prev->L+j*stride;
              for(int fw=0; fw<filter_size; fw++) {
                pool_index=pool_outside_index+fw;
                if(prev->A[pool_index]>prev->A[argmax_Z])
                  argmax_Z=pool_index;
              }
            }
            A[index_A]=prev->A[argmax_Z];
          }
  }
  else {
    // linear forward propagation (from A[l-1] to Z[l])
    // Z[l](n_sample,n_l) := A[l-1](n_sample,n_{l-1}) * W(n_l, n_{l-1}).T,
    // Z(i,j): a[i+j*lda] for ColMajor, a[j+i*lda] for RowMajor, lda is the leading dimension
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n_sample,dim,prev->dim,
                alpha_, prev->A, prev->dim, W, prev->dim, beta_, A, dim);
    // Z[l]:= Z[l]+b[l]
    for(int i=0; i<n_sample; i++)
      vsAdd(dim,A+i*dim,b,A+i*dim);
  }
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
  if(layer_type!="Pool")
    for(int i=0; i<dim; i++) {
      db[i]=0;
      for(int j=0; j<n_sample; j++)
        db[i]+=dZ[j*dim+i];
    }

  /// dW(n_channel,filter_size,filter_size,prev->n_channel)=dZ(n_sample,n_channel,L,L)*prev->A(n_sample,prev->n_channel,prev->L,prev->L)
  // sum over (n_sample,(prev->L-L)/stride,(prev->L-L)/stride)
  if(layer_type=="Conv2d") {
    long index_dW,index_dZ,index_prev_A;
    for(int n=0; n<n_channel; n++)
      for(int l=0; l<prev->n_channel; l++)
        for(int fl=0; fl<filter_size; fl++)
          for(int fw=0; fw<filter_size; fw++) {
            index_dW=n*prev->n_channel*filter_area+l*filter_area+fl*filter_size+fw;
            dW[index_dW]=0;
            for(int m=0; m<n_sample; m++)
              for(int i=0; i<L; i++) {
                int il=i*stride+fl-paddling;
                if(il>=0 && il<prev->L) {
                  index_dZ=m*dim+n*area+i*L;
                  index_prev_A=m*prev->dim+l*prev->area+il*prev->L+fw-paddling;
                  // if convolution on the paddling of prev->A
                  if(fw-paddling<0 || fw-paddling+(L-1)*stride+1>=prev->L)
                    for(int j=0; j<L; j++) {
                      int iw=j*stride+fw-paddling;
                      if(iw>=0 && iw<prev->L)
                        dW[index_dW]+=dZ[index_dZ+j]*prev->A[index_prev_A+j*stride];
                    }
                  // convolution within the valid range of prev->A
                  else
                    dW[index_dW]+=cblas_sdot(L,dZ+index_dZ,1,prev->A+index_prev_A,stride);
                }
              }
          }
  }
  else if(layer_type=="Hidden" || layer_type=="Output") {
    // dW[l](n_l,n_{l-1}):= dZ[l](n_sample,n_{l}) * A[l-1](n_sample,n_{l-1})
    // op(dZ)(n_{l},n_sample), op(A[l-1])(n_sample,n_{l-1})
    cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,dim,prev->dim,n_sample,
                 alpha_, dZ, dim, prev->A, prev->dim, beta_, dW, prev->dim);
  }

  // dW:=dW+Lambda*W
  if(layer_type!="Pool" && Lambda>EPSILON)
    cblas_saxpy(dim*prev->dim,Lambda/n_sample,W,1,dW,1);

  /// prev->dA(n_sample,prev->n_channel,prev->L,prev->L)=dZ(n_sample,n_channel,L,L)*W(n_channel,prev->n_channel,filter_size,filter_size)
  // sum over (n_channel,filter_size/stride,filter_size/stride)
  if(prev->layer_type!="Input") {
    if(layer_type=="Conv2d") {
      long index_dA,index_W,index_dZ;
      for(int m=0; m<n_sample; m++)
        for(int l=0; l<prev->n_channel; l++)
          for(int il=0; il<prev->L; il++)
            for(int iw=0; iw<prev->L; iw++) {
              index_dA=m*prev->dim+l*prev->area+il*prev->L+iw;
              prev->dA[index_dA]=0;
              for(int n=0; n<n_channel; n++)
                for(int fl=0; fl<filter_size; fl++) {
                  int i=(il+paddling-fl)/stride;
                  if(i>=0 && i<L) {
                    index_W=n*prev->n_channel*filter_area+l*filter_area+fl*filter_size;
                    index_dZ=m*dim+n*area+i*L;
                    for(int fw=0; fw<filter_size; fw++) {
                      int j=(iw+paddling-fw)/stride;
                      if(j>=0 &&j<L)
                        prev->dA[index_dA]+=W[index_W+fw]*dZ[index_dZ+j];
                    }
                  }
                }
            }
    }
    else if(layer_type=="Pool") {
      long index_dZ,argmax_A,pool_index,pool_outside_index;
      // no activation for Pool, so dA=dZ
      memset(prev->dA,0,sizeof(float)*n_sample*prev->dim);
      for(int m=0; m<n_sample; m++)
        // n_channel == prev->n_channel
        for(int n=0; n<n_channel; n++)
          for(int i=0; i<L; i++)
            for(int j=0; j<L; j++) {
              index_dZ=m*dim+n*area+i*L+j;
              // find the argmax of the prev->A within filter_area
              argmax_A=m*prev->dim+n*prev->area+i*stride*prev->L+j*stride;
              for(int fl=0; fl<filter_size; fl++) {
                int il=fl+i*stride;
                pool_outside_index=m*prev->dim+n*prev->area+il*prev->L+j*stride;
                for(int fw=0; fw<filter_size; fw++) {
                  pool_index=pool_outside_index+fw;
                  if(prev->A[pool_index]>prev->A[argmax_A])
                    argmax_A=pool_index;
                }
              }
              // backpropagate the graidents only to the argmax in dA within filter_area
              prev->dA[argmax_A]=dZ[index_dZ];
            }
    }
    else {
      // dA[l-1]=dZ[l]*W[l]
      // calculate dZ[l](n_sample,n_{l-1}=dZ[l](n_sample,n_{l})* W[l](n_{l},n_{l-1})
      cblas_sgemm (CblasRowMajor,CblasNoTrans,CblasNoTrans,n_sample,prev->dim,dim,
                   alpha_, dZ, dim, W, prev->dim, beta_,prev->dA, prev->dim);
    }
    // dZ[l-1] <- dA[l-1].*dF
    if(prev->activation=="sigmoid")
      prev->sigmoid_backward();
    else if(prev->activation=="ReLU")
      prev->ReLU_backward();
    else if(prev->activation=="None")
      memcpy(prev->dZ,prev->dA,sizeof(float)*n_sample*prev->dim);
  }

}

void layers::gradient_descent_optimize(const float &initial_learning_rate,const int &num_epochs,const int &step) {
  // learning rate decay
  float learning_rate=initial_learning_rate*(1-step*1.0/num_epochs)+0.01*initial_learning_rate;
  // W[l]:=W[l]-learning_rate*dW[l]
  cblas_saxpy(dim_W,-learning_rate,dW,1,W,1);
  // b[l]:=b[l]-learning_rate*db[l]
  cblas_saxpy(dim_b,-learning_rate,db,1,b,1);
}

void layers::Adam_optimize(const float &initial_learning_rate,const float &beta_1,const float &beta_2,const int &num_epochs,const int &epoch_step,const int &train_step) {
  // learning rate decay
  float learning_rate=initial_learning_rate*(1-epoch_step*1.0/num_epochs)+0.01*initial_learning_rate;
  float bias_beta_1,bias_beta_2;
  bias_beta_1=1.0/(1-pow(beta_1,train_step+1));
  bias_beta_2=1.0/(1-pow(beta_2,train_step+1));

  // VdW[l]=beta_1*VdW[l]
  cblas_sscal(dim_W,beta_1,VdW,1);
  // VdW[l]+=(1-beta_1)*dW[l]
  cblas_saxpy(dim_W,(1-beta_1),dW,1,VdW,1);
  // Vdb[l]=beta_1*Vdb[l]
  cblas_sscal(dim_b,beta_1,Vdb,1);
  // Vdb[l]+=(1-beta_1)*db[l]
  cblas_saxpy(dim_b,(1-beta_1),db,1,Vdb,1);

  // SdW[l]=beta_2*SdW[l]
  cblas_sscal(dim_W,beta_2,SdW,1);
  //dW[l] replaced by dW[l]*dW[l], (will not be used anymore)
  vsMul(dim_W,dW,dW,dW);
  // SdW[l]+=(1-beta_2)*dW[l]
  cblas_saxpy(dim_W,(1-beta_2),dW,1,SdW,1);
  // Sdb[l]=beta_2*Sdb[l]
  cblas_sscal(dim_b,beta_2,Sdb,1);
  //db[l] replaced by db[l]*db[l], (will not be used anymore)
  vsMul(dim_b,db,db,db);
  // Sdb[l]+=(1-beta_2)*db[l]
  cblas_saxpy(dim_b,(1-beta_2),db,1,Sdb,1);

  memcpy(VdW_corrected,VdW,sizeof(float)*dim_W);
  memcpy(SdW_corrected,SdW,sizeof(float)*dim_W);
  memcpy(Vdb_corrected,Vdb,sizeof(float)*dim_b);
  memcpy(Sdb_corrected,Sdb,sizeof(float)*dim_b);

  // get bias corrected momentum and rms
  cblas_sscal(dim_W, bias_beta_1,VdW_corrected,1);
  cblas_sscal(dim_b, bias_beta_1,Vdb_corrected,1);
  cblas_sscal(dim_W, bias_beta_2,SdW_corrected,1);
  cblas_sscal(dim_b, bias_beta_2,Sdb_corrected,1);
  // SdW,Sdb=sqrt(SdW),sqrt(Sdb)
  vsSqrt(dim_W,SdW_corrected,SdW_corrected);
  vsSqrt(dim_b,Sdb_corrected,Sdb_corrected);
  // add epsilon to prevent overflow
  for(int i=0; i<dim_W; i++)
    SdW_corrected[i]+=1e-8;
  for(int i=0; i<dim_b; i++)
    Sdb_corrected[i]+=1e-8;
  // VdW,Vdb= VdW/sqrt(SdW+epsilon),Vdb/sqrt(Sdb+epsilon)
  vsDiv(dim_W,VdW_corrected,SdW_corrected,VdW_corrected);
  vsDiv(dim_b,Vdb_corrected,Sdb_corrected,Vdb_corrected);

  // update the weights and bias
  cblas_saxpy(dim_W,-learning_rate,VdW_corrected,1,W,1);
  cblas_saxpy(dim_b,-learning_rate,Vdb_corrected,1,b,1);
}
