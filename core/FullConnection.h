#pragma once


#include "ConnectionBase.h"
#include "Matrix.h"

class FullConnection : public ConnectionBase
{
public:
	// fully connected layer
	Matrix<F> weights_;
	Matrix<F> delta_weights_; // Temporary array to store weight matrices from previous step for momentum term.

	void forward(const VectorND<F>& pre_layer_acts_, VectorND<F>& next_layer_acts) const;
	void backward(const VectorND<F>& next_layer_grads_, VectorND<F>& pre_layer_grads_) const;

	//Note: next_layer_grad and prev_layer_act
	// eta is learning rate, alpha is momentum coefficient.
	void updateWeights(const F& eta, 
					const F& alpha, 
					const VectorND<F>& next_layer_grad, 
					const VectorND<F>& prev_layer_act);

	void writeTXT(std::ofstream& of) const {
		weights_.writeTXT(of);
	}

	void check() const {
		weights_.check();
		delta_weights_.check();
	}

};

// end of file
