#pragma once
#include <vector>
#include <stdexcept>
#include "activationfunctions.hpp"

namespace deeplframework {
	class NeuronLayer {
	private:
		int numOfNeurons;
		int numOfInputs;
		// Each element is specific to a single neuron
		std::vector<std::vector<double>> weights;
		std::vector<double> biases;
		mutable std::vector<double> lastLayerOutput;
		mutable std::vector<double> lastLayerOutputBA;

	public:
		// On default, a rectified linear activation function is used (ReLU)
		double(*activationFunction)(double) = activationFunctions::ReLU;
		double(*activationFunctionDerivative)(double) = activationFunctionDerivatives::ReLU;

		NeuronLayer(unsigned int numberOfNeurons, unsigned int numberOfWeightsPerNeuron, double defaultBiasValue = 0, double defaultWeightsValue = 0) {
			if (numberOfNeurons == 0 || numberOfWeightsPerNeuron == 0) {
				throw std::runtime_error("More weights and/or neurons are required for a layer");
			}
			this->numOfNeurons = numberOfNeurons;
			for (unsigned int i = 0; i < numberOfNeurons; i++) {
				biases.push_back(defaultBiasValue);
				weights.push_back({});

				for (unsigned int j = 0; j < numberOfWeightsPerNeuron; j++) {
					weights[i].push_back(defaultWeightsValue);
				}
			}

			this->numOfInputs = numberOfWeightsPerNeuron;
		}
		NeuronLayer(unsigned int numberOfNeurons, std::vector<std::vector<double>> connectionWeights, std::vector<double> neuronBiases) {
			if (numberOfNeurons == 0) {
				throw std::runtime_error("More neurons are required for a layer");
			} if (connectionWeights.size() != numberOfNeurons) {
				throw "Weights matrix is invalid";
			}
			else if (connectionWeights[0].size() == 0) {
				throw std::runtime_error("Weights matrix is invalid");
			} if (neuronBiases.size() != numberOfNeurons) {
				throw std::runtime_error("Biases list is invalid");
			}

			this->numOfNeurons = numberOfNeurons;
			this->numOfInputs = connectionWeights[0].size();
			this->biases = neuronBiases;
			this->weights = connectionWeights;
		}
		void setWeight(unsigned int i, unsigned int j, double value) {
			weights[i][j] = value;
		}
		void setBias(unsigned int i, double value) {
			biases[i] = value;
		}
		int getNumOfNeurons() {
			return numOfNeurons;
		}
		int getNumOfInputs() {
			return numOfInputs;
		}
		std::vector<double> getBiases() {
			return biases;
		}
		std::vector<std::vector<double>> getWeights() {
			return weights;
		}
		std::vector<double> getRecordedOutput(bool beforeActivationFunction = false) {
			return (beforeActivationFunction) ? lastLayerOutputBA : lastLayerOutput;
		}
		// Calculate dot product of weights matrix and neuronInputs vector + biases vector. NeuronInputs vector length needs to
		// be equal to the number of weights per neuron
		std::vector<double> propogateCalculations(std::vector<double> neuronInputs, bool recordActivations = false) {
			if (neuronInputs.size() != weights[0].size()) {
				throw std::runtime_error("Neuron outputs vector is invalid");
			}

			if (recordActivations) lastLayerOutputBA = {};

			std::vector<double> output;
			for (unsigned int i = 0; i < weights.size(); i++) {
				double sum = biases[i];
				for (unsigned int j = 0; j < neuronInputs.size(); j++) {
					sum += weights[i][j] * neuronInputs[j];
				}
				if (recordActivations) lastLayerOutputBA.push_back(sum);
				output.push_back(activationFunction(sum));
			}
			if (recordActivations) lastLayerOutput = output;
			return output;
		}
	};
}
