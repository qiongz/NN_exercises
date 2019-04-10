#include"init.h"
#include"utils.h"
#include"nn_lr.h"

using namespace std;
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


    nn_lr  clr(n_features,n_classes);
    clr.fit(X_train,Y_train,n_train,num_epochs,learning_rate,Lambda,batch_size,true);
    accuracy=clr.predict_accuracy(X_dev,Y_dev_orig,Y_prediction,n_dev);
    cout<<"validation set accuracy:"<<accuracy<<endl;
   

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

    return 0;
}
