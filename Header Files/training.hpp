#pragma once
#include "deeplframework.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <ctime>

namespace deeplframework {
	namespace backpropogationTraining {
		struct NeuronDerivative {
			double bias;
			double output;
			std::vector<double> weights;

			NeuronDerivative(double b, double o, std::vector<double> w) {
				this->bias = b;
				this->output = o;
				this->weights = w;
			}
		};

		class DerivativeSet {
		private:
			// All variables are derivatives
			std::vector<std::vector<NeuronDerivative>> neuronDerivatives;

		public:
			// Model to store network gradient
			DerivativeSet(std::vector<int> layerShape, int numOfInputs) {
				int numOfWeights = numOfInputs;

				for (unsigned int i = 0; i < layerShape.size(); i++) {
					// Instantiate neurons layer by layer
					std::vector<NeuronDerivative> neurons;
					
					for (int j = 0; j < layerShape[i]; j++) {
						std::vector<double> weights;
						for (int k = 0; k < numOfWeights; k++) weights.push_back(0);

						neurons.push_back(NeuronDerivative(0, 0, weights));
					}
					neuronDerivatives.push_back(neurons);

					numOfWeights = layerShape[i];
				}
			}
			// For one neuron
			void setOutputDerivative(int layer, int neuronIndex, double s) {
				neuronDerivatives[layer][neuronIndex].output = s;
			}
			// For one neuron
			void setBiasDerivative(int layer, int neuronIndex, double s) {
				neuronDerivatives[layer][neuronIndex].bias = s;
			}
			// For one neuron connection
			void setWeightDerivative(int layer, int neuronIndex, int conIndex, double s) {
				neuronDerivatives[layer][neuronIndex].weights[conIndex] = s;
			}
			// For one layer
			void setLayerOfDerivatives(std::vector<NeuronDerivative> mat, int layer) {
				if (mat.size() != neuronDerivatives[layer].size())
					throw std::runtime_error("Input is invalid: Size does not match original layer");

				neuronDerivatives[layer] = mat;
			}
			std::vector<std::vector<NeuronDerivative>> getNeuronDerivatives() {
				return neuronDerivatives;
			}
		};

		// Provided inputTrainingDataGen function should return a vector with the expected input training data. Assumes MSE cost function.
		// CAUTION - THE ENTIRE inputTrainingDataGen VECTOR OUTPUT IS LOADED IN MEMORY
		NeuralNetwork mse_fit(NeuralNetwork& model, const int numOfMiniBatches, const int numOfTrainingSamples, std::vector<double>(*inputTrainingDataGen)(int dataIndex),
			std::vector<double>(*expectedOutputDataGen)(int dataIndex), const int epochs = 11, const double learningRate = 0.1, const bool showUpdates = true, const int numOfSamplesBetweenUpdates = 100) {

			const int samplesPerBatch = numOfTrainingSamples / numOfMiniBatches;
			const double learningRateTimesRofNumSamples = learningRate * (1.0 / (double) samplesPerBatch);

			NeuralNetwork newModel = model;

			for (int e = 1; e <= epochs; e++) {

				std::vector<double> outputs;
				std::vector<double> expectedOutputs;

				// Learn from batches
				for (int batch = 0; batch < numOfMiniBatches; batch++) {
					// For progress updates
					double cost = 0;
					double timeStarted = std::time(nullptr);

					// Instantiate variables
					std::vector<NeuronLayer> modelLayers = {};
					// On default, all variables in the following object are set to 0
					DerivativeSet gradient = DerivativeSet(newModel.getLayerShape(), newModel.getNumOfInputs());

					// Calculate gradient
					for (int sample = batch * samplesPerBatch; sample < samplesPerBatch * (batch + 1); sample++) {
						DerivativeSet newGradient = DerivativeSet(newModel.getLayerShape(), newModel.getNumOfInputs());

						// Run neural network
						std::vector<double> inputs = inputTrainingDataGen(sample);
						outputs = newModel.run(inputs, true);
						modelLayers = newModel.getLayers();
						expectedOutputs = expectedOutputDataGen(sample);

						// Calculate cost, if updates will be shown
						if (showUpdates) {
							for (int o = 0; o < outputs.size(); o++) {
								double difference = outputs[o] - expectedOutputs[o];
								cost += (difference * difference);
							}
						}

						for (int l = modelLayers.size() - 1; l > -1; l--) {
							std::vector<NeuronDerivative> neuronDerivatives;

							double(*activationDerivative)(double) = model.getLayers()[l].activationFunctionDerivative;

							for (int n = 0; n < modelLayers[l].getNumOfNeurons(); n++) {
								double biasderivative;
								double aderivative;

								// Represents a jacobian matrix
								std::vector<double> weightderivatives;

								if (l + 1 == modelLayers.size()) {
									// Get derivative of bias to MSE cost function. This is equal to the derivative of the neuron activation to the cost function
									aderivative = activationDerivative(modelLayers[l].getRecordedOutput(true)[n]) * 2 * (outputs[n] - expectedOutputs[n]);
								}
								else {
									// Calculate aderivative
									aderivative = 0;
									for (int p = 0; p < modelLayers[l + 1].getNumOfNeurons(); p++) {
										aderivative += modelLayers[l + 1].getWeights()[p][n] * newGradient.getNeuronDerivatives()[l + 1][p].output;
									}
									aderivative *= activationDerivative(modelLayers[l].getRecordedOutput(true)[n]);
								}

								// This is because the bias has no coefficient, so its derivative in the weighted sum is 1.
								biasderivative = aderivative;

								// Get neuronDerivatives from previous gradient
								NeuronDerivative ndPGradient = gradient.getNeuronDerivatives()[l][n];

								// Calculate weight derivatives + weight derivatives from previous gradient for this neuron
								for (int wi = 0; wi < modelLayers[l].getNumOfInputs(); wi++) {
									double activation = (l == 0) ? inputs[wi] : modelLayers[l - 1].getRecordedOutput(false)[wi];

									weightderivatives.push_back((activation * aderivative) + ndPGradient.weights[wi]);
								}

								// Add derivatives with derivatives from previous gradient
								neuronDerivatives.push_back(NeuronDerivative(biasderivative + ndPGradient.bias, aderivative + ndPGradient.output, weightderivatives));
							}

							newGradient.setLayerOfDerivatives(neuronDerivatives, l);
						}

						gradient = newGradient;

						// Show updates
						if (showUpdates == true && (sample + 1) % numOfSamplesBetweenUpdates == 0) {
							std::cout << "Epoch: " << e << "\tBatch: " << batch + 1 << "\tSample: " << sample + 1 << "\tCost: " << cost / ((double) sample + 1.0) << "\t";
							std::cout << "Time elapsed: " << std::time(nullptr) - timeStarted << "s\n";
						}
					}

					std::vector<std::vector<NeuronDerivative>> gradientDerivatives = gradient.getNeuronDerivatives();

					// Print progress updates, if they are enabled
					if (showUpdates) {
						// Average out cost
						cost /= (double)samplesPerBatch;
						std::cout << "Epoch: " << e << "\tBatch: " << batch + 1 << "\tCost: " << cost << "\t";
						std::cout << "Time elapsed: " << std::time(nullptr) - timeStarted << "s\t";
						std::cout << "Samples: " << samplesPerBatch << "\n";
					}

					// Slightly modify newModel with average gradient
					for (int layer = 0; layer < modelLayers.size(); layer++) {
						for (int neuron = 0; neuron < modelLayers[layer].getNumOfNeurons(); neuron++) {
							// Modify bias
							double change = learningRateTimesRofNumSamples * gradientDerivatives[layer][neuron].bias;
							newModel.setLayerBias(layer, neuron, (modelLayers[layer].getBiases()[neuron] - change));

							// Modify weights
							for (int weightIndex = 0; weightIndex < modelLayers[layer].getNumOfInputs(); weightIndex++) {
								double change = learningRateTimesRofNumSamples * gradientDerivatives[layer][neuron].weights[weightIndex];
								newModel.setLayerWeight(layer, neuron, weightIndex, (modelLayers[layer].getWeights()[neuron][weightIndex] - change));
							}
						}
					}
				}

			}

			// Return newModel
			return newModel;
		}
	}
}
