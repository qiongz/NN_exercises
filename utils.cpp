#include"utils.h"

void load_data(string filename,vector<float> &X,vector<int> &Y,int &n_sample,int &n_features) {
    int i,count=0;
    ifstream datafile(filename);
    std::list<string>:: iterator it;
    if(datafile.is_open()) {
        string line,element;
        list<string> values_str;
        getline(datafile,line);
        while(std::getline(datafile,line)) {
            istringstream iss_line(line);
            while(std::getline(iss_line,element,','))
                values_str.push_back(element);
            Y.push_back(stoi(values_str.front(),nullptr));
	    values_str.pop_front();
            n_features=values_str.size();
	    for(it=values_str.begin();it!=values_str.end();it++)
                X.push_back(stod(*it,nullptr)/255.0);
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
    for(int i=0; i<n_sample; i++)
	Y_onehot[Y[i]%n_classes+i*n_classes]=1;
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

void print(const float *X,int id,int height,int width) {
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
