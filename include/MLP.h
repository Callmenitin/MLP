struct Classifier
{
int inputNodes;
int hiddenNodes;
int hiddenLayers;
int activationFunction;
int costFunction;
int backPropgationFunction;
};


double sigmoid(double x);
double tanh(double x);
double relu(double x);