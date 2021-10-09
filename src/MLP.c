#include"../include/MLP.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

double **inputs;
double **hidden,**weights;
int rows,start,end,output;
int isNumerical=-1;
struct OutputMap *mappings;
int _mapInt=0;

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
for(int i=0;i<rows;i++)
{
inputs[i] =(double *)malloc(end*sizeof(double));
}
mappings=(struct OutputMap*)malloc(rows*sizeof(struct OutputMap));
readCSV(obj.fileName);

for(int i=0;i<rows;i++){
    printf("Row:%d\n",i+1);
    for(int j=0;j<end;j++){
printf("%f, ",inputs[i][j]);

    }
    printf("\n");
}

 for(int j=0;j<_mapInt;j++){
printf("%s:%d\n, ",mappings[j]._class,mappings[j].value);

    }



}



void readCSV(char *fileName){

FILE* filePointer = fopen(fileName, "r");
if (NULL == filePointer)
{
printf("Please check file if its name and format is valid");
exit(0);
}

char* line=(char*)malloc(1024* sizeof(char));
for(int i=0;i<rows;i++) 
{
fgets(line, 1024, filePointer); 
char* tok = strtok(line, ",");
for(int j=0;j<end;j++)
{
if(j>=start)
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


