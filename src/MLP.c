#include"../include/MLP.h"
#define SIGMOID 0
#define RELU 1
#define TANH 2



double sigmoid(double x){
return 1/(1+exp(-x));
}

double tanh(double x){
return tanh(x);
}

double relu(double x){
if(x<0)
	return 0;
return x;
}