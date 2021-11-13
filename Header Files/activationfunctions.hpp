#pragma once
#include <cmath>

namespace deeplframework {
	// Is heavily subject to change
	namespace activationFunctions {
		double linear(double x) {
			return x;
		}
		double ReLU(double x) {
			return (x > 0) ? x : 0;
		}
		double sigmoid(double x) {
			return 1 / (1 / std::exp(x) + 1);
		}
		double tanh(double x) {
			return std::tanh(x);
		}
	}
	namespace activationFunctionDerivatives {
		double linear(double x) {
			return 1;
		}
		// Technically, when x is zero the derivative is undefined, but this function sets the derivative to 1 when x is 0
		double ReLU(double x) {
			return (x > 0) ? 1 : 0;
		}
		double sigmoid(double x) {
			double sig = activationFunctions::sigmoid(x);
			return sig * (1 - sig);
		}
		double tanh(double x) {
			double cosh = std::cosh(x);
			return 1 / (cosh * cosh);
		}
	}
}
