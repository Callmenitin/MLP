#include"../include/MLP.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

double **inputs,*outputs,*errors;
double learningRate;
double **hidden,**weights;
int rows,_start,_end,output;
int isNumerical=-1;
struct OutputMap *mappings;
int _mapInt=0;
int _input;
int _output_neurons;

double sigmoid(double x){
return 1.0/(1.0+exp(-x));
}

double mytanh(double x){
return tanh(x);
}

double relu(double x){
if(x<0)
	return 0;
return x;
}

void setDataSetParameters(int r,int s,int e,int o){
rows=r;
_start=s;
_end=e;
output=o;    
}

void feedForward(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],struct OutputNeuron outputNeurons[_mapInt-1],int a,int b);
void backPropagation(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],int a,int b);

void train(struct Classifier obj){

struct Neuron inputNeurons[obj.inputNodes];
struct Neuron hiddenNeurons[obj.hiddenLayers][obj.inputNodes];


_input=obj.inputNodes;

//Weight initialization from Input layer to 1st Hidden layer
for(int i=0;i<obj.inputNodes;i++)
{
for(int j=0;j<obj.inputNodes;j++)
{
inputNeurons[i].weights[j]=(rand()%10);
}
}

//Weight initialization from 1st hidden layer to last Hidden layer

for(int i=0;i<obj.hiddenLayers-1;i++)
{
for(int j=0;j<obj.inputNodes;j++)
{
for(int k=0;k<obj.inputNodes;k++) 
hiddenNeurons[i][j].weights[k]=(rand()%10);
}
}

inputs=(double **)malloc(rows*sizeof(double*));

for(int i=0;i<rows;i++)
{
inputs[i] =(double *)malloc(_end*sizeof(double));
}

mappings=(struct OutputMap*)malloc(rows*sizeof(struct OutputMap));
outputs=(double *)malloc(rows*sizeof(double));
errors=(double *)malloc(rows*sizeof(double));

readCSV(obj.fileName);

_output_neurons=_mapInt-1;

struct OutputNeuron outputNeurons[_output_neurons];

//Weight initialization from last hidden layer to output layer
for(int i=0;i<obj.inputNodes;i++)
{
for(int j=0;j<_output_neurons;j++)
{
hiddenNeurons[obj.hiddenLayers-1][1].weights[j]=(rand()%10);
}
}

for(int i=0;i<obj.iterations;i++)
{
int start,end;

    if(obj.backPropagation==STOCHASTIC){
        start=(rand()%rows)+1;
        end=start+1;
    }
    if(obj.backPropagation==MINI_BATCH){
        start=(rand()%(rows-obj.batchSize))+1;
        end=start+obj.batchSize;
    }
    else{
        start=0;
        end=rows;
    }
    feedForward(obj,inputNeurons,hiddenNeurons,outputNeurons,start,end);
   }
}

void readCSV(char *fileName){

FILE* filePointer = fopen(fileName, "r");
if (NULL == filePointer)
{
printf("Please check file if its name and format is valid");
exit(0);
}

char* line=(char*)malloc(2048*sizeof(char));
for(int i=0;i<rows;i++) 
{
fgets(line,2048,filePointer); 
char* tok = strtok(line, ",");
for(int j=0;j<_end;j++)
{
if(j>=_start)
{
if(j==output)
{
if(isNumber(tok)==1)
{
isNumerical=1;
inputs[i][j]=atof(tok);
}
else
{
isNumerical=0;
inputs[i][j]=getMappingInteger(tok); 
}

}
else
{inputs[i][j]=atof(tok);}
}
tok = strtok(NULL,",");
}
}
free(line);
}

int isNumber(char *str){

int start=0,end=strlen(str);
if(str[0]=='-')
{
start=1;
}
while(start<end){
    if(!(str[start]>='0'&&str[start]<='9'))
        return 0;
    ++start;
}
    return 1;
}

int getMappingInteger(char *map){

int find=-1;
for(int i=0;i<_mapInt;i++)
{
int valid=strcmp(map,mappings[i]._class);
if(valid==0)
{
find=mappings[i].value;
return find;
}
}

if(find==-1){
find=_mapInt;
mappings[_mapInt]._class=(char *)malloc(sizeof(char *));
strcpy(mappings[_mapInt]._class,map);
mappings[_mapInt].value=_mapInt;
++_mapInt;
return find;
}
}

double crossEntropy(struct OutputNeuron output[])
{
    double sum=0.0;
    for(int i=0;i<_output_neurons;i++)
    {
        sum+=output[i].actual*log(output[i].soft);
    }
    return -1*sum;
}

void feedForward(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],struct OutputNeuron outputNeurons[_mapInt-1],int start,int end)
{

for(int i=start;i<end;i++)
{
int _itr_start,_itr_end;

if(output==_start)
{_itr_start=output+1; _itr_end=_end;}
else
{_itr_start=start; _itr_end=_end-1; }  

//From input Layer to first hidden layer
for(int k=0;k<obj.inputNodes;k++)
{
hiddenNeurons[0][k].value=0.0;
int index=0;

for(int ptr=_itr_start;ptr<_itr_end;ptr++)
{
hiddenNeurons[0][k].value+=inputs[i][ptr]*inputNeurons[index].weights[k];
++index; 
}
}
//from 2nd Hidden layer to last hidden Layer

for(int k=1;k<obj.hiddenLayers;k++)
{
int previous=k-1;
for(int _neuron=0;_neuron<obj.inputNodes;_neuron++)
{
hiddenNeurons[k][_neuron].value=0.0;
for(int ptr=0;ptr<obj.inputNodes;ptr++)
{
double _value;
if(obj.activationFunction==SIGMOID){
_value=sigmoid(hiddenNeurons[previous][ptr].value);
}
else if(obj.activationFunction==TANH){
_value=mytanh(hiddenNeurons[previous][ptr].value);
}
else{
_value=relu(hiddenNeurons[previous][ptr].value);
}
hiddenNeurons[k][_neuron].value+=_value*hiddenNeurons[previous][ptr].weights[_neuron];
}
}
}

//From last hidden layer to output layer

for(int ii=0;ii<_output_neurons;ii++)
{
int hidden_layer=obj.hiddenLayers-1;
outputNeurons[ii].net=0.0;
for(int jj=0;jj<obj.inputNodes;jj++)
{
double _value;
if(obj.activationFunction==SIGMOID){
_value=sigmoid(hiddenNeurons[hidden_layer][jj].value);
}
else if(obj.activationFunction==TANH){
_value=mytanh(hiddenNeurons[hidden_layer][jj].value);
}
else{
_value=relu(hiddenNeurons[hidden_layer][jj].value);
}
outputNeurons[ii].net+=_value*hiddenNeurons[hidden_layer][jj].weights[ii];
}
}

double total=0.0;

for(int ptr=0;ptr<_output_neurons;ptr++){
    total+=exp(outputNeurons[ptr].net);
}
for(int ptr=0;ptr<_output_neurons;ptr++){
    outputNeurons[ptr].soft=exp(outputNeurons[ptr].net)/total;
}
double _actual=inputs[i][output];

for(int ptr=0;ptr<_output_neurons;ptr++)
{
    if(ptr==_actual)    
    { 
outputNeurons[ptr].actual=1.0;
    }
    else
    {
outputNeurons[ptr].actual=0.0;
    }  
}

errors[i]=crossEntropy(outputNeurons);

} 

}


void backPropagation(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],int a,int b){










}




/*
for(int i=0;i<obj.inputNodes;i++)
{
    printf("Input Neuron:%d\n",i);
for(int j=0;j<obj.inputNodes;j++)
{
printf(" %.2f",inputNeurons[i].weights[j]);

}
printf("\n");
}


for(int i=0;i<obj.hiddenLayers-1;i++)
{
    printf("Weight matrix for hidden layer:%d\n",i);
for(int j=0;j<obj.inputNodes;j++)
{
for(int k=0;k<obj.inputNodes;k++) 
printf("%.2f ",hiddenNeurons[i][j].weights[k]);

printf("\n");
}
printf("\n");
}


printf("Status of hidden layers and its neurons\n");

for(int ii=0;ii<obj.hiddenLayers;ii++){
printf("Hidden layer:%d\n",ii);
for(int jj=0;jj<obj.inputNodes;jj++){
printf("Neuron:%d, Value=%.2f\n",jj,hiddenNeurons[ii][jj].value);

}}
printf("\n");
*/