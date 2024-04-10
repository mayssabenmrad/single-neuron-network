#include "neuron.h"

// Train the neuron using backpropagation
int train(Neuron *neuron, float **inputs, float *outputs, int N, int epochs,
	  float learning_rate);