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
#include <stdio.h>

// Train the neuron using backpropagation
// The function takes the following arguments:
// 	* neuron: A pointer to the neuron to be trained.
// 	* inputs: An array of input samples.
// 	* outputs: An array of real outputs to the sample inputs.
// 	* samples_num: The number of samples in the dataset.
// 	* learning_rate: The learning rate used to update the weights of the
// 	  neuron.
// 	* epsilon: The minimum loss value at which the training process stops.
int train(Neuron *neuron, float **inputs, float *outputs, int samples_num,
	  float learning_rate, float epsilon)
{
	// Define the predictions array to store the output of the neuron
	float predictions[samples_num];

	// Define the loss variable to hold the loss value of the neuron
	float loss;

	// Define the epoch variable to keep track of the number of epochs
	int epoch = 0;

	// Keep adjusting the weights of the neuron using the gradients descent
	// algorithm until the loss value converges to a minimum value.
	do {
		// Forward pass: Calculate the output of the neuron for each
		// inputs sample
		for (int i = 0; i < samples_num; i++) {
			predictions[i] = run_neuron(neuron, inputs[i]);
		}

		// Calculate the loss value of the neuron on the dataset
		// using the Cross Entropy Loss function
		loss = cross_entropy_loss(outputs, predictions, samples_num);

		// Print the loss value of the neuron at each epoch then update
		// the epoch variable
		printf("Epoch %d: Loss = %f\n", epoch, loss);
		epoch++;

		// -------------------------------------------------------------

		// Backward pass: Calculate the derivatives of the loss function
		// with respect to the weights of the neuron

		// Define an array to store the derivatives of the loss function
		float dL_dw[2];

		// Calculate the average of the derivatives of the loss function
		// with respect to the weights of the neuron over the dataset
		// to update the weights.
		for (int i = 0; i < samples_num; i++) {
			// Calculate the derivative of the loss function with
			// respect to the weights of the neuron.
			dL_dw[0] += dL_dWk(outputs[i], predictions[i],
					   inputs[i][0]);
			dL_dw[1] += dL_dWk(outputs[i], predictions[i],
					   inputs[i][1]);
		}
		dL_dw[0] = dL_dw[0] / samples_num;
		dL_dw[1] = dL_dw[1] / samples_num;

		// Update the weights of the neuron using the derivatives and
		// the learning rate
		// Wk = Wk + learning_rate * dL/dWk
		neuron->weights[0] += learning_rate * dL_dw[0];
		neuron->weights[1] += learning_rate * dL_dw[1];
	} while (loss > epsilon);

	return 0;
}