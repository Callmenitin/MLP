#include<stdio.h>
#include"../include/MLP.h"


int main(){

struct Classifier obj={13,1,RELU,BATCH,0.0000000001,120000,2,"housepricetrain.csv"};
setDataSetParameters(300,0,13,13);
train(obj);
predict(obj,"housepricetrain.csv",100,0,12);
return 0;	
}