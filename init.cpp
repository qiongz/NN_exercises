#include"init.h"
void usage(char *target) {
    std::cout<<"Usage: "<<target<<" [Options]\n";
    std::cout<<"Options:\n";
    std::cout<<"  -n                       num_epochs\n";
    std::cout<<"  -b                       batch_size\n";
    std::cout<<"  -a                       learning_rate\n";
    std::cout<<"  -l                       lambda\n";
}

void init_argv(int &num_epochs, int &batch_size, float &learning_rate, float &lambda,int argc,char *argv[])
{
    num_epochs=500;
    batch_size=128;
    learning_rate=0.01;
    lambda=0;

    extern char *optarg;
    int ch,errFlag;
    errFlag=0;
    while((ch=getopt(argc,argv,"n:b:a:l:h:"))!=-1) {
        switch(ch) {
        case 'n':
            num_epochs=atoi(optarg);
            break;
        case 'b':
            batch_size=atoi(optarg);
            break;
        case 'a':
            learning_rate=atof(optarg);
            break;
        case 'l':
            lambda=atof(optarg);
            break;
        case 'h':
            errFlag++;
            break;
        default:
            errFlag++;
            break;
        }
    }
    if(errFlag) {
        usage(argv[0]);
        exit(0);
    }
}
