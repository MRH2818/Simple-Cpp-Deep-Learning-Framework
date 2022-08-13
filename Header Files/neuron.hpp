/*
 ____      ____     ___________    _____      _____     _______      _______
|    |    |    |   |           |   \    \    /    /    |       |    |       |
|    |____|    |   |    _______|    \    \  /    /     |       |    |       |
|              |   |   |_______      \    \/    /      |_______|    |_______|
|     ____     |   |   ________|      \        /        _______      _______
|    |    |    |   |   |_______        |      |        |       |    |       |
|____|    |____|   |___________|       |______|        |_______|    |_______|


This file is obselete. I was an lazy idiot when I made this file that couldn't be bothered doing basic
linear algebra and thinking about matrices.

*/

#pragma once
#include <vector>

class Neuron {
private:
	double(*activationFunctionInputDouble)(double) = NULL;
	double(*activationFunctionInputVector)(std::vector<double>, std::vector<double>, double) = NULL;

public:
	double bias;
	std::vector<double> weights;

	Neuron(std::vector<double> connectionWeights, double neuronBias = 0) {
		this->bias = neuronBias;
		this->weights = connectionWeights;
	}
	// Activation function takes the dot product of this neuron weights and other neuron outputs
	Neuron(std::vector<double> connectionWeights, double neuronBias, double(*activationFunc)(double)) {
		this->bias = neuronBias;
		this->activationFunctionInputDouble = activationFunc;
		this->weights = connectionWeights;
	}
	// Activation function takes neuron outputs, the neuron's weights, and the neuron's bias as inputs
	Neuron (std::vector<double> connectionWeights, double neuronBias, double(*activationFunc)(std::vector<double> weights, std::vector<double> neuronOutputs, double bias)) {
		this->bias = neuronBias;
		this->activationFunctionInputVector = activationFunc;
		this->weights = connectionWeights;
	}
	// Activation function takes neuron outputs, the neuron's weights, and the neuron's bias as inputs
	void setCustomActivationFunction(double(*activationFunc)(std::vector<double> weights, std::vector<double> neuronOutputs, double bias)) {
		this->activationFunctionInputVector = activationFunc;
	}

	double propogateCalculations(std::vector<double> neuronOutputsVector) {
		if (activationFunctionInputDouble == NULL) {
			if (activationFunctionInputVector == NULL) {
				throw "Activation function not defined";
			}
			// If activationFunctionInputDouble is not defined, return custom activation function
			return activationFunctionInputVector(weights, neuronOutputsVector, bias);
		}
		if (neuronOutputsVector.size() != weights.size()) {
			throw "Weights vector and Neuron outputs vector don't match";
		}

		// Calculate dot product of two vectors + bias
		double dotProduct = bias;
		for (int i = 0; i < neuronOutputsVector.size(); i++) {
			dotProduct += weights[i] * neuronOutputsVector[i];
		}
		return activationFunctionInputDouble(dotProduct);
	}
};
