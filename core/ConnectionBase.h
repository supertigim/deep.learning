#pragma once

#include <iostream>
#include "VectorND.h"

class ConnectionBase
{
public:
	virtual void forward(const VectorND<F>& pre_layer_acts_, VectorND<F>& next_layer_acts) const {
		std::cout << "ConnectionBase::forward()" << endl;
		exit(1);
	}

	virtual void backward(const VectorND<F>& next_layer_grads_, VectorND<F>& pre_layer_grads_) const {
		std::cout << "ConnectionBase::backward()" << endl;
		exit(1);
	}

	virtual void updateWeights(	const F& eta, 
								const F& alpha, 
								const VectorND<F>& next_layer_grad, 
								const VectorND<F>& prev_layer_act) {
		std::cout << "ConnectionBase::updateWeights()" << endl;
		exit(1);
	}

	virtual void writeTXT(std::ofstream& of) const {
		std::cout << "ConnectionBase::writeTXT()" << endl;
		exit(1);
	}

	virtual void check() const {
		std::cout << "ConnectionBase::check()" << endl;
		exit(1);
	}
};

// end of file
