#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include"nn_lr.cpp"

using namespace std;

void load_data(string filename,vector<float> &X,vector<int> &Y,int &n_sample,int &n_features) {
    int i,count=0;
    ifstream datafile(filename);
    if(datafile.is_open()) {
        string line,element;
        vector<string> values_str;
        getline(datafile,line,'\n');
        while(std::getline(datafile,line,'\n')) {
            istringstream iss_line(line);
            while(std::getline(iss_line,element,','))
                values_str.push_back(element);
            Y.push_back(stoi(values_str.front(),nullptr));
            n_features=values_str.size()-1;
            for(i=1; i<n_features; i++)
                X.push_back(stod(values_str[i],nullptr)/255.0);
            count++;
            values_str.clear();
        }
        datafile.close();
    }
    n_sample=count;
}

void scale_features(vector<float> &X_train,vector<float> &X_dev,const int &n_train,const int &n_dev,const int &n_features) {
    vector<float> var,mean;
    var.assign(n_features,0);
    mean.assign(n_features,0);
    // calculate the mean and std (var)
    for(int i=0; i<n_features; i++) {
        for(int j=0; j<n_train; j++)
            mean[i]+=X_train[j*n_features+i];
        mean[i]/=n_train;
        for(int j=0; j<n_train; j++)
            var[i]+=pow(X_train[j*n_features+i]-mean[i],2);
        var[i]/=n_train-1;
        var[i]=sqrt(var[i]);
    }
    for(int i=0; i<n_features; i++) {
        for(int j=0; j<n_train; j++)
            X_train[i+j*n_features]=(X_train[i+j*n_features]-mean[i])/var[i];
        for(int j=0; j<n_dev; j++)
            X_dev[i+j*n_features]=(X_dev[i+j*n_features]-mean[i])/var[i];
    }
    var.clear();
    mean.clear();
}

void onehot(const vector<int> &Y,vector<int> &Y_onehot,int n_classes) {
    int  n_sample=Y.size();
    Y_onehot.assign(n_sample*n_classes,0);
    for(int k=0; k<n_classes; k++)
        for(int i=0; i<n_sample; i++)
            Y_onehot[k*n_sample+i]=(Y[i]==k?1:0);
}

void print(const vector<float> &X,int id,int height,int width) {
    int n_pixes=height*width;
    for(int i=0; i<width; i++)
        cout<<"-";
    cout<<endl;
    float mean=0;
    for(int i=0; i<n_pixes; i++)
        mean+=X[id*n_pixes+i];
    mean/=n_pixes;
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++)
            if(X[i*width+j+id*n_pixes]>mean)
                cout<<"*";
            else
                cout<<" ";
        cout<<endl;
    }
    for(int i=0; i<width; i++)
        cout<<"-";
    cout<<endl;
}

int main(int argc,char *argv[]) {
    vector<float> X_train,X_dev;
    vector<int> Y_train_orig,Y_dev_orig;
    vector<int> Y_train,Y_dev;
    vector<int> Y_prediction;
    float accuracy;
    int n_train,n_dev,n_features,n_classes=10;
    load_data("datasets/train.csv",X_train,Y_train_orig,n_train,n_features);
    load_data("datasets/dev.csv",X_dev,Y_dev_orig,n_dev,n_features);
    onehot(Y_train_orig,Y_train,n_classes);
    onehot(Y_dev_orig,Y_dev,n_classes);

    cout<<"n_train:="<<n_train<<endl;
    cout<<"n_dev:="<<n_dev<<endl;
    cout<<"n_features:="<<n_features<<endl;

    nn_lr  clr(n_features,n_classes);
    clr.fit(X_train,Y_train,n_train,1000,0.001,0,128,true);
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
