#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<list>
#include<cmath>

using namespace std;

void load_data(string filename,vector<float> &X,vector<int> &Y,int &n_sample,int &n_features);
void scale_features(vector<float> &X_train,vector<float> &X_dev,const int &n_train,const int &n_dev,const int &n_features);
void onehot(const vector<int> &Y,vector<int> &Y_onehot,int n_classes);
void print(const vector<float> &X,int id,int height,int width);
void print(const float* X,int id,int height,int width);
