#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <random>

#include "neuronlayer.hpp"

namespace deeplframework {
	class NeuralNetwork {
	private:
		mutable std::vector<NeuronLayer> layers;
		std::vector<int> layerShape;
		unsigned int numInputs;

	public:
		// Empty network
		NeuralNetwork() {
			this->layers = {};
			this->numInputs = 0;
			this->layerShape = {};
		}
		// Each element in layerShape list shows the amount of Neurons in that layer. The number of layers will
		// be equal of the length of the list. DefaultBiasValue and defaultWeightsValue apply for all neurons in the network
		NeuralNetwork(std::vector<int> layerShape, unsigned int numberOfInputs, int defaultBiasValue = 0, int defaultWeightsValue = 0) {
			this->layers = {};
			this->layerShape = layerShape;
			for (unsigned int i = 0; i < layerShape.size(); i++) {
				unsigned int numWeights = (i == 0) ? numberOfInputs : layerShape[i - 1];
				this->layers.push_back(NeuronLayer(layerShape[i], numWeights, defaultBiasValue, defaultWeightsValue));
			}
			this->numInputs = numberOfInputs;
		}
		// Weights and biases are assumed to be initialized in the layer list elements
		NeuralNetwork(std::vector<NeuronLayer> networkLayers, unsigned int numberOfInputs) {
			this->layers = networkLayers;
			this->numInputs = numberOfInputs;

			for (unsigned int i = 0; i < networkLayers.size(); i++) {
				layerShape.push_back(networkLayers[i].getNumOfNeurons());
			}
		}

		// Set activation function for a single layer in the network. On default, a rectified linear activation function
		// is used (ReLU)
		void setActivationFunction(unsigned int layerIndex, double(*activationFunc)(double), double(*activationFuncDerivative)(double)) {
			layers[layerIndex].activationFunction = activationFunc;
			layers[layerIndex].activationFunctionDerivative = activationFuncDerivative;
		}
		// Set activation function for a single layer in the network. On default, a rectified linear activation function
		// is used (ReLU)
		void setActivationForAllLayers(double(*activationFunc)(double), double(*activationFuncDerivative)(double)) {
			for (int l = 0; l < layers.size(); l++) {
				setActivationFunction(l, activationFunc, activationFuncDerivative);
			}
		}
		void setLayerWeight(unsigned int layerIndex, unsigned int neuronIndex, unsigned int connIndex, double weightValue) {
			this->layers[layerIndex].setWeight(neuronIndex, connIndex, weightValue);
		}
		void setLayerBias(unsigned int layerIndex, unsigned int neuronIndex, double biasValue) {
			this->layers[layerIndex].setBias(neuronIndex, biasValue);
		}
		int getNumOfInputs() {
			return this->numInputs;
		}
		std::vector<int> getLayerShape() {
			return this->layerShape;
		}
		std::vector<NeuronLayer> getLayers() {
			return layers;
		}
		std::vector<double> run(std::vector<double> inputs, bool recordActivations = false) {
			std::vector<double> layerInputs = inputs;
			for (unsigned int i = 0; i < layers.size(); i++) {
				layerInputs = layers[i].propogateCalculations(layerInputs, recordActivations);
			}
			return layerInputs;
		}

		// See format on GitHub page
		static bool WriteToBinaryFile(NeuralNetwork network, const char* path) {
			std::ofstream os;
			os.open(path, std::ios::trunc | std::ios::binary);

			bool success = false;
			
			if (os.is_open()) {
				std::vector<NeuronLayer> layers = network.getLayers();
				int numOfLayers = layers.size();

				os.write((char*) &numOfLayers, 4);

				// Write layers
				for (int l = 0; l < layers.size(); l++) {
					int numOfNeurons = layers[l].getNumOfNeurons();
					int numOfInputs = layers[l].getNumOfInputs();
					os.write((char*)&numOfNeurons, 4);
					os.write((char*)&numOfInputs, 4);

					std::vector<double> biases = layers[l].getBiases();
					std::vector<std::vector<double>> weights = layers[l].getWeights();

					// Write biases -> 1 bias for each neuron
					for (int b = 0; b < biases.size(); b++) {
						os.write((char*)&biases[b], sizeof(biases[b]));
					}

					// Write weights
					for (int n = 0; n < weights.size(); n++) {
						for (int wi = 0; wi < weights[n].size(); wi++) {
							os.write((char*)&weights[n][wi], sizeof(weights[n][wi]));
						}
					}
				}
				success = true;
			}
			os.close();
			return success;
		}

		static NeuralNetwork ReadBinaryFile(const char* path) {
			std::ifstream is;
			is.open(path, std::ios::binary);

			if (is.is_open()) {
				std::vector<NeuronLayer> networkLayers;

				int numOfLayers = 0;
				int numOfNetworkInputs = 0;

				// Read number of layers
				is.read((char*)&numOfLayers, 4);

				for (int l = 0; l < numOfLayers; l++) {
					int numOfNeurons = 0;
					int numOfInputs = 0;

					is.read((char*)&numOfNeurons, 4);
					is.read((char*)&numOfInputs, 4);

					if (l == 0) numOfNetworkInputs = numOfInputs;

					std::vector<double> biases;
					std::vector<std::vector<double>> weights;

					for (int b = 0; b < numOfNeurons; b++) {
						double bias = 0;
						is.read((char*)&bias, sizeof(bias));
						biases.push_back(bias);
					}

					for (int n = 0; n < numOfNeurons; n++) {
						weights.push_back({});
						for (int wi = 0; wi < numOfInputs; wi++) {
							double weight = 0;
							is.read((char*)&weight, sizeof(weight));
							weights[n].push_back(weight);
						}
					}

					networkLayers.push_back(NeuronLayer(numOfNeurons, weights, biases));
				}

				return NeuralNetwork(networkLayers, numOfNetworkInputs);
			}
			return NeuralNetwork();
		}

		// Solely so people can visualize the network. THERE IS NO READTEXTFILE FUNCTION. Function returns success status
		static bool WriteToTextFile(NeuralNetwork network, const char *path) {
			int numOfInputs = network.getNumOfInputs();
			std::vector<int> layerShape = network.getLayerShape();
			std::vector<NeuronLayer> layers = network.getLayers();

			std::string content = "Number of inputs: " + std::to_string(numOfInputs) + "\n";

			for (unsigned int l = 0; l < layers.size(); l++) {
				content += "Layer: " + std::to_string(l + 1) + "\n"
					+ "Biases:\n";

				// Print biases
				std::vector<double> biases = layers[l].getBiases();
				
				for (int b = 0; b < biases.size(); b++) {
					content += std::to_string(biases[b]) + " ";
				}
				content += "\n";

				// Print weights

				content += "Weights:\n";

				std::vector<std::vector<double>> weights = layers[l].getWeights();

				for (int n = 0; n < weights.size(); n++) {
					for (int wi = 0; wi < weights[n].size(); wi++) {
						content += std::to_string(weights[n][wi]) + " ";
					}
					content += "\n";
				}
				content += "\n";
			}

			std::ofstream os;

			os.open(path, std::ios::trunc);

			if (os.is_open()) {
				os << content;
				
				os.close();
				return true;
			}
			else {
				return false;
			}
		}

		static double GetRandomDouble(double randMin, double randMax) {
			std::uniform_real_distribution<double> range(randMin, randMax);
			std::mt19937 gen;
			gen.seed(std::random_device{}());
			
			return range(gen);
		}

		static NeuralNetwork CreateRandomNetwork(std::vector<int> layerShape, int numOfInputs, double weightDifference, double biasDifference) {
			std::vector<NeuronLayer> layers;

			int numOfLayerInputs = numOfInputs;
			for (int l = 0; l < layerShape.size(); l++) {
				std::vector<double> biases;
				std::vector<std::vector<double>> weights;

				for (int n = 0; n < layerShape[l]; n++) {
					biases.push_back(GetRandomDouble(-biasDifference, biasDifference));
					weights.push_back({});

					for (int wi = 0; wi < numOfLayerInputs; wi++) {
						weights[n].push_back(GetRandomDouble(-weightDifference, weightDifference));
					}
				}

				layers.push_back(NeuronLayer(layerShape[l], weights, biases));

				numOfLayerInputs = layerShape[l];
			}

			return NeuralNetwork(layers, numOfInputs);
		}
	};
}
