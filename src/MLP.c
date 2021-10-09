#include"../include/MLP.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

double **inputs;
double **hidden,**weights;
int rows,start,end,output;
double iop[100][100];

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

void setDataSetParameters(int r,int s,int e,int o){
rows=r;
start=s;
end=e;
output=o;    
}

void classify(struct Classifier obj){

struct Neuron inputNeurons[obj.inputNodes];
struct Neuron hiddenNeurons[obj.hiddenLayers][obj.inputNodes];


for(int i=0;i<obj.inputNodes;i++)
{
for(int j=0;j<obj.inputNodes;j++)
{
inputNeurons[i].weights[j]=(rand()%10);

}
}

inputs=(double **)malloc(rows*sizeof(double*));
for(int i=0;i<rows;i++){
inputs[i] =(double *)malloc(end*sizeof(double));}


readCSV(obj.fileName);

for(int i=0;i<rows;i++){
    printf("Row:%d\n",i+1);
    for(int j=0;j<end;j++){
printf("%f, ",inputs[i][j]);

    }
    printf("\n");
}

}



void readCSV(char *fileName){

FILE* filePointer = fopen(fileName, "r");
if (NULL == filePointer)
{
printf("Error opening %s file. Make sure you mentioned the file path correctly\n", fileName);
exit(0);
}

char* line=(char*)malloc(1024* sizeof(char));


for(int i=0;i<rows;i++) 
{
fgets(line, 1024, filePointer); 
char* tok = strtok(line, ",");

for(int j=0;j<end;j++)
{
printf("%s,",tok);
inputs[i][j]=atof(tok);
tok = strtok(NULL,",");
}

printf("\n");

}
 

}


