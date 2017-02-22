#pragma once

#include "VectorND.h"

namespace ActFunc
{
	F LRELU(const F& x);
	F RELU(const F& x);
	F Sigmoid(const F& x);

	F LRELUGradFromY(const F& x);
	F RELUGradFromY(const F& x);
	F SigmoidGradFromY(const F& y);

	void applyLRELUToVector(const int& num_bias, VectorND<F>& vector);
	void applyRELUToVector(const int& num_bias, VectorND<F>& vector);
	void applySigmoidToVector(const int& num_bias, VectorND<F>& vector);
}

// end of file
