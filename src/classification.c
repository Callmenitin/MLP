#include<stdio.h>
#include"../include/MLP.h"


int main(){

struct Classifier obj={30,1,SIGMOID,STOCHASTIC,0.0000000001,20,2,"traindata.csv"};
setDataSetParameters(398,0,30,0);
train(obj);
classify(obj,"traindata.csv",50,1,30);
return 0;	
}