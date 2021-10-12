#include"../include/MLP.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

double **inputs,*outputs,*errors;
struct Neuron *_input_weights;
struct Neuron **_hidden_weights;
double learningRate;
double **hidden,**weights;
int rows,_start,_end,output;
int isNumerical=-1;
struct OutputMap *mappings;
int _mapInt=0;
int _input;
int _output_neurons;
int _hiddenNodes=2;

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

void feedForward(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],struct OutputNeuron outputNeurons[_mapInt],int a,int b);
void backPropagation(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],struct OutputNeuron outputNeurons[_mapInt],int a,int b);

void train(struct Classifier obj){

struct Neuron inputNeurons[obj.inputNodes];
struct Neuron hiddenNeurons[obj.hiddenLayers][_hiddenNodes];

_input=obj.inputNodes;

//Weight initialization from Input layer to 1st Hidden layer
for(int i=0;i<obj.inputNodes;i++)
{
for(int j=0;j<_hiddenNodes;j++)
{
inputNeurons[i].weights[j]=0.0005*(rand()%10)+0.0001;
}
}

//Weight initialization from 1st hidden layer to last Hidden layer

for(int i=0;i<obj.hiddenLayers-1;i++)
{
for(int j=0;j<_hiddenNodes;j++)
{
for(int k=0;k<_hiddenNodes;k++) 
hiddenNeurons[i][j].weights[k]=0.0005*(rand()%10)+0.0001;
}
}

inputs=(double **)malloc(rows*sizeof(double*));
for(int i=0;i<rows;i++)
{
inputs[i] =(double *)malloc((_end+1)*sizeof(double));
}

mappings=(struct OutputMap*)malloc(rows*sizeof(struct OutputMap));
outputs=(double *)malloc(rows*sizeof(double));
errors=(double *)malloc(rows*sizeof(double));
_input_weights=(struct Neuron*)malloc(obj.inputNodes*sizeof(struct Neuron));
_hidden_weights=(struct Neuron **)malloc(obj.hiddenLayers*sizeof(struct Neuron*));

for(int i=0;i<obj.hiddenLayers;i++)
{
_hidden_weights[i] =(struct Neuron*)malloc(_hiddenNodes*sizeof(struct Neuron));
}

readCSV(obj.fileName);

_output_neurons=_mapInt;

struct OutputNeuron outputNeurons[_output_neurons];

//Weight initialization from last hidden layer to output layer
for(int i=0;i<_hiddenNodes;i++)
{
for(int j=0;j<_output_neurons;j++)
{
hiddenNeurons[obj.hiddenLayers-1][i].weights[j]=0.0005*(rand()%10)+0.0001;
}
}


for(int i=0;i<obj.iterations;i++)
{
int start,end;
printf("Iterations:%d    ",i+1);

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
    backPropagation(obj,inputNeurons,hiddenNeurons,outputNeurons,start,end);
}


for(int i=0;i<obj.inputNodes;i++)
{
for(int j=0;j<_hiddenNodes;j++)
{
_input_weights[i].weights[j]=inputNeurons[i].weights[j];
}
}

for(int i=0;i<obj.hiddenLayers-1;i++)
{
for(int j=0;j<_hiddenNodes;j++)
{
for(int k=0;k<_hiddenNodes;k++) 
{_hidden_weights[i][j].weights[k]=hiddenNeurons[i][j].weights[k];
}
}
}

for(int i=0;i<_hiddenNodes;i++)
{
for(int j=0;j<_output_neurons;j++)
{
_hidden_weights[obj.hiddenLayers-1][i].weights[j]=hiddenNeurons[obj.hiddenLayers-1][i].weights[j];
}
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
for(int j=0;j<=_end;j++)
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

if(find==-1)
{
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

void feedForward(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],struct OutputNeuron outputNeurons[_mapInt],int start,int end)
{

for(int i=start;i<end;i++)
{
int _itr_start,_itr_end;

if(output==_start)
{_itr_start=output+1; _itr_end=_end;}
else
{_itr_start=start; _itr_end=_end-1; }  

//From input Layer to first hidden layer
for(int k=0;k<_hiddenNodes;k++)
{
hiddenNeurons[0][k].value=0.0;
int index=0;

for(int ptr=_itr_start;ptr<=_itr_end;ptr++)
{
hiddenNeurons[0][k].value+=inputs[i][ptr]*inputNeurons[index].weights[k];
++index; 
}

}
//from 2nd Hidden layer to last hidden Layer

for(int k=1;k<obj.hiddenLayers;k++)
{
int previous=k-1;
for(int _neuron=0;_neuron<_hiddenNodes;_neuron++)
{
hiddenNeurons[k][_neuron].value=0.0;
for(int ptr=0;ptr<_hiddenNodes;ptr++)
{
double _value;
if(obj.activationFunction==SIGMOID)
{
_value=sigmoid(hiddenNeurons[previous][ptr].value);
}
else if(obj.activationFunction==TANH)
{
_value=mytanh(hiddenNeurons[previous][ptr].value);
}
else
{
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
for(int jj=0;jj<_hiddenNodes;jj++)
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

for(int ptr=0;ptr<_output_neurons;ptr++)
{
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

void backPropagation(struct Classifier obj,struct Neuron inputNeurons[_input],struct Neuron hiddenNeurons[][_input],struct OutputNeuron outputNeurons[_mapInt],int start,int end)
{

double average=0.0;
for(int i=start;i<end;i++){
    average+=errors[i];    
}

average=average/(double)(end-start);
printf("Error:%.10f\n",average);

//From last hidden layer to output layer
int last_layer=obj.hiddenLayers-1;
for(int i=0;i<_hiddenNodes;i++)
{
for(int j=0;j<_output_neurons;j++)
{
double _term1;
if(obj.activationFunction==SIGMOID){
_term1=sigmoid(hiddenNeurons[last_layer][i].value);
}
else if(obj.activationFunction==TANH){
_term1=mytanh(hiddenNeurons[last_layer][i].value);
}
else{
_term1=relu(hiddenNeurons[last_layer][i].value);
}
double _term2=outputNeurons[j].soft*(1-outputNeurons[j].soft);
double _term3=(outputNeurons[j].soft-outputNeurons[j].actual);
double _partial_derivative=_term1*_term2*_term3;
hiddenNeurons[last_layer][i].weights[j]=hiddenNeurons[last_layer][i].weights[j]-obj.learningRate*_partial_derivative;
}

}

for(int i=0;i<_mapInt;i++)
{
    double _first=outputNeurons[i].soft-outputNeurons[i].actual;
    double _second=outputNeurons[i].soft*(1-outputNeurons[i].soft);
    outputNeurons[i].error_wrt_net=_first*_second;
}

int _hidden_itr=obj.hiddenLayers-2;

while(_hidden_itr>=0)
{
for(int i=0;i<_hiddenNodes;i++)
{
for(int j=0;j<_hiddenNodes;j++)
{
double _term1,_term2;
if(obj.activationFunction==SIGMOID){
_term1=sigmoid(hiddenNeurons[_hidden_itr][i].value);
_term2=sigmoid(hiddenNeurons[_hidden_itr+1][j].value);
}
else if(obj.activationFunction==TANH){
_term1=mytanh(hiddenNeurons[_hidden_itr][i].value);
_term2=mytanh(hiddenNeurons[_hidden_itr+1][j].value);
}
else{
_term1=relu(hiddenNeurons[_hidden_itr][i].value);
_term2=relu(hiddenNeurons[_hidden_itr+1][j].value);
}
double _term3=0.0;
for(int ii=0;ii<_mapInt;ii++)
{

double weightFactor=1;
for(int ptr=_hidden_itr;ptr<obj.hiddenLayers;ptr++)
{
weightFactor=weightFactor*hiddenNeurons[ptr][ii].weights[ii];
}
_term3+=outputNeurons[ii].error_wrt_net*weightFactor;

}
double _total=_term1*_term2*_term3;
hiddenNeurons[_hidden_itr][i].weights[j]=hiddenNeurons[_hidden_itr][i].weights[j]-obj.learningRate*_total;
}
}
--_hidden_itr;
}

//From Input layer to first hidden layer
int _itr_start;
if(output==_start)
{_itr_start=output+1;}
else
{_itr_start=start; } 

for(int i=0;i<obj.inputNodes;i++)
{
for(int edge=0;edge<_hiddenNodes;edge++)
{
double _data_value=inputs[rows-5][_itr_start];
double _term1,_term2;

_term1=_data_value;
if(obj.activationFunction==SIGMOID){
_term2=sigmoid(hiddenNeurons[0][edge].value);
}
else if(obj.activationFunction==TANH){
_term2=sigmoid(hiddenNeurons[0][edge].value);
}
else{
_term2=sigmoid(hiddenNeurons[0][edge].value);
}

double _term3=0.0;
for(int ii=0;ii<_mapInt;ii++)
{
_term3+=outputNeurons[ii].error_wrt_net*inputNeurons[i].weights[edge];
}

double _total=_term1*_term2*_term3;
inputNeurons[i].weights[edge]=inputNeurons[i].weights[edge]-obj.learningRate*_total;
}
++_itr_start;
}

}

void test(struct Classifier obj,char *fileName,int rows,int start,int end,int output)
{

int total=rows;
int count=0;

printf("Testing the data....");
char* _output_Filename="output.csv";
FILE *inputFile = fopen(fileName, "r");

double *_inputs;
_inputs=(double *)malloc(end*sizeof(double));

char* line=(char*)malloc(2048*sizeof(char));
for(int i=0;i<rows;i++) 
{
fgets(line,2048,inputFile); 
char* tok = strtok(line, ",");
for(int j=0;j<=end;j++)
{
if(j>=_start)
{
if(j==output)
{
if(isNumber(tok)==1)
{
_inputs[j]=atof(tok);
}
else
{
_inputs[j]=getIntOfClass(tok); 
}
}
else
{_inputs[j]=atof(tok);}
}
tok = strtok(NULL,",");
}
int _itr_start,_itr_end;

if(output==start)
{_itr_start=output+1; _itr_end=end;}
else
{_itr_start=start; _itr_end=end-1; }  

//From input Layer to first hidden layer
for(int k=0;k<_hiddenNodes;k++)
{
_hidden_weights[0][k].value=0.0;
int index=0;
for(int ptr=_itr_start;ptr<=_itr_end;ptr++)
{
_hidden_weights[0][k].value+=_inputs[ptr]*_input_weights[index].weights[k];
++index; 
}
}

//from 2nd Hidden layer to last hidden Layer

for(int k=1;k<obj.hiddenLayers;k++)
{
int previous=k-1;
for(int _neuron=0;_neuron<_hiddenNodes;_neuron++)
{
_hidden_weights[k][_neuron].value=0.0;
for(int ptr=0;ptr<_hiddenNodes;ptr++)
{
double _value;
if(obj.activationFunction==SIGMOID){
_value=sigmoid(_hidden_weights[previous][ptr].value);
}
else if(obj.activationFunction==TANH){
_value=mytanh(_hidden_weights[previous][ptr].value);
}
else{
_value=relu(_hidden_weights[previous][ptr].value);
}
_hidden_weights[k][_neuron].value+=_value*_hidden_weights[previous][ptr].weights[_neuron];
}
}
}

struct OutputNeuron _O_Neurons[_mapInt];
//From last hidden layer to output layer
int hidden_layer=obj.hiddenLayers-1;
for(int ii=0;ii<_mapInt;ii++)
{
_O_Neurons[ii].net=0.0;
for(int jj=0;jj<_hiddenNodes;jj++)
{
double _value;
if(obj.activationFunction==SIGMOID){
_value=sigmoid(_hidden_weights[hidden_layer][jj].value);
}
else if(obj.activationFunction==TANH){
_value=mytanh(_hidden_weights[hidden_layer][jj].value);
}
else{
_value=relu(_hidden_weights[hidden_layer][jj].value);

}
_O_Neurons[ii].net+=_value*_hidden_weights[hidden_layer][jj].weights[ii];
}
}

double _total=0.0;
for(int ptr=0;ptr<_mapInt;ptr++)
{
    _total+=exp(_O_Neurons[ptr].net);
}
for(int ptr=0;ptr<_mapInt;ptr++)
{
    _O_Neurons[ptr].soft=exp(_O_Neurons[ptr].net)/_total;
}

int max=0;
for(int ptr=1;ptr<_mapInt;ptr++)
{
if(_O_Neurons[ptr].soft>_O_Neurons[max].soft)
{
    max=ptr;
}
}
if(_inputs[output]==max){++count;}
}

free(line);
printf("\nAccuracy acheived:%.2f",(count/(double)total)*100);

}

int getIntOfClass(char *class)
{

for(int i=0;i<_mapInt;i++)
{
int valid=strcmp(class,mappings[i]._class);
if(valid==0)
return i;
}

}
