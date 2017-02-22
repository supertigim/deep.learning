#include <math.h>
#include "ActivationFunctions.h"

namespace ActFunc
{
	F Sigmoid(const F& x) {
		return 1.0 / (1.0 + exp(-x));
	}

	// not from x. y = getSigmoid(x).
	F SigmoidGradFromY(const F& y) {
		return (1.0 - y) * y;
	}

	F RELU(const F& x) {
		return MAX2(0.0, x);
	}

	// RELU Grad from X == RELU Grad from Y
	F RELUGradFromY(const F& x) {
		if (x > 0.0) return 1.0;
		else return 0.0;
	}

	F LRELU(const F& x) {
		return x > 0.0 ? x : 0.01*x;
	}

	// RELU Grad from X == RELU Grad from Y
	F LRELUGradFromY(const F& x) {
		if (x > 0.0) 	return 1.0;
		else 			return 0.01;
	}

	void applySigmoidToVector(const int& num_bias, VectorND<F>& vector) {
		const int num = vector.num_dimension_ - num_bias;
		// don't apply activation function to bias
		for (int d = 0; d < num; ++d) {
			vector[d] = Sigmoid(vector[d]);
		}
	}

	void applyRELUToVector(const int& num_bias, VectorND<F>& vector) {
		const int num = vector.num_dimension_ - num_bias;
		// don't apply activation function to bias
		for (int d = 0; d < num; ++d) {
			vector[d] = RELU(vector[d]);
		}
	}

	void applyLRELUToVector(const int& num_bias, VectorND<F>& vector) {
		const int num = vector.num_dimension_ - num_bias;
		// don't apply activation function to bias
		for (int d = 0; d < num; ++d) 
			vector[d] = LRELU(vector[d]);
	}
}

// end of file
