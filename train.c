/*
 * This file defines the training process of our neuron using backpropagation.

 * The training process consists of the following steps:
 	1. Forward pass: Calculate the output of the neuron given an input.
 	2. Calculate the loss value of the neuron on the dataset.
 	3. Backward pass: Calculate the derivatives of the loss function with
  	   respect to the weights of the neuron using the chain rule.
 	4. Update the weights of the neuron using the derivatives and a learning
  	   rate.

 * The training process is repeated for a number of epochs to improve the
   performance of the neuron on the dataset.
*/

#include "neuron.h"
#include "loss_function.h"
#include "derivatives.h"
#include <stdlib.h>
#include <stdio.h>

// Train the neuron using backpropagation
int train(Neuron *neuron, float **inputs, float *outputs, int N, int epochs,
	  float learning_rate)
{
	// Define the predictions array to store the output of the neuron
	float *predictions = malloc(N * sizeof(float));

	// Loop over the number of epochs
	for (int i = 0; i < epochs; i++) {
		// Forward pass: Calculate the output of the neuron for each
		// inputs sample
		for (int j = 0; j < N; j++) {
			predictions[j] = run_neuron(neuron, inputs[j]);
		}

		// Calculate the loss value of the neuron on the dataset
		// using the Cross Entropy Loss function
		float loss = cross_entropy_loss(outputs, predictions, N);

		// Print the loss value
		printf("Epoch %d: Loss = %f\n", i, loss);

		// Backward pass: Calculate the derivatives of the loss function
		// with respect to the weights of the neuron

		// Define an array to store the derivatives of the loss function
		float *dL_dw = (float *)malloc(2 * sizeof(float));

		// Check if memory allocation was successful
		if (dL_dw == NULL) {
			printf("Error: Unable to allocate memory\n");
			return -1;
		}

		// Calculate the average of the derivatives of the loss function
		// with respect to the weights of the neuron over the dataset
		// to update the weights.
		for (int j = 0; j < N; j++) {
			// Calculate the derivative of the loss function with
			// respect to the weights of the neuron.
			dL_dw[0] = dL_dw[0] + dL_dWk(outputs[j], predictions[j],
						     inputs[j][0]);
			dL_dw[1] = dL_dw[1] + dL_dWk(outputs[j], predictions[j],
						     inputs[j][1]);
		}
		dL_dw[0] = dL_dw[0] / N;
		dL_dw[1] = dL_dw[1] / N;

		// Update the weights of the neuron using the derivatives and
		// the learning rate
		// Wk = Wk + learning_rate * dL/dWk
		neuron->weights[0] =
			neuron->weights[0] - learning_rate * dL_dw[0];
		neuron->weights[1] =
			neuron->weights[1] - learning_rate * dL_dw[1];
	}

	return 0;
}