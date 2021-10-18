#define SIGMOID 0
#define RELU 1
#define TANH 2
#define BATCH 0
#define MINI_BATCH 1
#define STOCHASTIC 2

struct Classifier
{
int inputNodes;
int hiddenLayers;
int activationFunction;
int backPropagation;
double learningRate;
int iterations;
int batchSize;
char *fileName;
};

struct OutputMap
{
char *_class;
int value;
};

struct Neuron
{
double weights[50];
double value;
};

struct OutputNeuron
{
double net;
double soft;
double actual;
double error_wrt_net;
};

double sigmoid(double x);
double mytanh(double x);
double relu(double x);
void train(struct Classifier obj);
void setDataSetParameters(int rows,int start,int end,int output);
void readCSV(char *fileName);
int isNumber(char*);
int getMappingInteger(char *);
int getIntOfClass(char *);
double crossEntropy(struct OutputNeuron[]);
void test(struct Classifier,char *fileName,int rows,int start,int end,int output);
void predict(struct Classifier,char *fileName,int rows,int start,int end);
void classify(struct Classifier,char *fileName,int rows,int start,int end);
double MSE(double,double);

