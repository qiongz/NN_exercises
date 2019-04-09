#ifndef INIT_H
#define INIT_H
#include<unistd.h>
#include<cstdlib>
#include<cmath>
#include<iostream>
#include<stdexcept>
#include<chrono>
void usage(char *);
void init_argv(int &num_epochs, int &batch_size, float &learning_rate, float &lambda,int argc,char *argv[]);
#endif
