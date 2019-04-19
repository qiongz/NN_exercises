#include"init.h"
#include"utils.h"
#include"layers.h"
#include"dnn.h"

int main(int argc,char *argv[]) {
    vector<float> X_train,X_dev;
    vector<int> Y_train_orig,Y_dev_orig;
    vector<int> Y_train,Y_dev;
    vector<int> Y_prediction;
    float accuracy,Lambda,learning_rate;
    int batch_size,num_epochs;
    init_argv(num_epochs,batch_size,learning_rate,Lambda,argc,argv);

    int n_train,n_dev,n_features,n_classes=10;
    load_data("datasets/train.csv",X_train,Y_train_orig,n_train,n_features);
    load_data("datasets/dev.csv",X_dev,Y_dev_orig,n_dev,n_features);
    onehot(Y_train_orig,Y_train,n_classes);
    onehot(Y_dev_orig,Y_dev,n_classes);

    layers input_layer(n_features,28,0.9,true,"Input","None"); /// L=28,                 dim:= 28 x 28 x 1 x _n
    layers conv_layer_1(16,1,false,"Conv2d","ReLU",1,3,1);      /// L=(28+2*1-3)/1+1=28  dim:= 28 x 28 x 16 x _n
    layers pool_layer_1(16,1,false,"Pool","None",0,2,2);      /// L=(28-2)/2+1=14,       dim:= 14 x 14 x 16 x _n
    layers conv_layer_2(32,1,false,"Conv2d","ReLU",1,3,1);      /// L=(14+2*1-3)/1+1=14, dim:= 14 x 14 x 32 x _n
    layers pool_layer_2(32,1,false,"Pool","None",0,2,2);      /// L=(14-2)/2=6,          dim:= 6 x 6 x 32 x _n
    layers conv_layer_3(64,1,false,"Conv2d","ReLU",1,3,1);      /// L=(6+2*1-3)/1+1=6,   dim:= 6 x 6 x 64 x _n
    layers pool_layer_3(64,1,false,"Pool","None",0,2,2);      /// L=(6-2)/2=2,           dim:= 2 x 2 x 64 x _n
    layers fully_connected_layer_1(512,1,0.5,true,"Hidden","ReLU"); ///                  dim:= 512 x _n
    layers fully_connected_layer_2(256,1,0.6,true,"Hidden","ReLU"); ///                  dim:= 256 x _n
    layers output_layer(n_classes,1,1,false,"Output","softmax");   ///                   dim:= 10 x _n

    dnn clr(n_features,n_classes,&input_layer,&output_layer);
    clr.insert_before_output(&conv_layer_1);
    clr.insert_before_output(&pool_layer_1);
    clr.insert_before_output(&conv_layer_2);
    clr.insert_before_output(&pool_layer_2);
    clr.insert_before_output(&conv_layer_3);
    clr.insert_before_output(&pool_layer_3);
    clr.insert_before_output(&fully_connected_layer_1);
    clr.insert_before_output(&fully_connected_layer_2);
    
    clr.train_and_dev(X_train,Y_train,X_dev,Y_dev,n_train,n_dev,num_epochs,learning_rate,Lambda,batch_size,"Adam",false,true,1);
    accuracy=clr.predict_accuracy(X_dev,Y_dev_orig,Y_prediction,n_dev);
    cout<<"validation set accuracy:"<<accuracy<<endl;


    // print the validation set and check the results by eye
    /*  
    int id=0;
    string id_str;
    while(id>=0 && id<n_dev) {
        cout << "Enter image id (0<=id<"<<n_dev<<"):" << endl;
        getline(cin,id_str);
        id=stoi(id_str,nullptr);
        if(id>=0 && id<n_dev) {
            print(X_dev,id,28,28);
            cout<<"Predicted: "<<Y_prediction[id]<<endl;
            cout<<"Actual: "<<Y_dev_orig[id]<<endl;
        }
    }
    */
    return 0;
}
