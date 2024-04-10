#include "neuron.h"
#include "activation_functions.h"
#include <stdio.h>

int main()
{
	// Create a new neuron with the sigmoid activation function
	Neuron *neuron = new_neuron(sigmoid);

	// Print the weights of the neuron
	for (int i = 0; i < 2; i++) {
		printf("Weight %d: %f\n", i, neuron->weights[i]);
	}

	// Print the bias of the neuron
	printf("Bias: %f\n", neuron->bias);

	// Test the activation function of the neuron
	printf("sigmoid(0) = %f\n", neuron->activation_function(0));

	// Test the run_neuron function
	float input[2] = { 1, 2 };
	printf("The output of the neuron is: %f\n", run_neuron(neuron, input));

	// Free the memory allocated for the neuron
	free_neuron(neuron);

	return 0;
}
