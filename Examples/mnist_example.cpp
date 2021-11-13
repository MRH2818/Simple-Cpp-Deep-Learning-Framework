#include <iostream>
#include "deeplframework.hpp"
#include <vector>
#include <string>

using namespace std;
using namespace deeplframework;
using namespace deeplframework::data;

MnistDataReader mdr("MNIST_DATA/train-labels.idx1-ubyte", "MNIST_DATA/train-images.idx3-ubyte");

std::vector<double> getInput(int id) {
    return mdr.getImageInput(id);
}
std::vector<double> getOutput(int id) {
    return mdr.getLabelOutput(id);
}

int main () {
    // Training MNIST Network
    cout << "Training Network - MNIST\n";

    system("title Training Network - MNIST");

    mdr.open();
    NeuralNetwork mnistNetwork = NeuralNetwork::CreateRandomNetwork({ 30, 10 }, 784, 1, 0);
    mnistNetwork.setActivationForAllLayers(activationFunctions::sigmoid, activationFunctionDerivatives::sigmoid);

    mnistNetwork = backpropogationTraining::mse_fit(mnistNetwork, 3000, 60000, getInput, getOutput, 3, 1.5, true, 60001);
    mdr.close();

    NeuralNetwork::WriteToBinaryFile(mnistNetwork, "mnist_network.bin");
    NeuralNetwork::WriteToTextFile(mnistNetwork, "mnist_network.txt");

    cout << "\nDone!!\n";

    return 0;
}
