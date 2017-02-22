#pragma once

#include "Matrix.h"

class LayerBase
{
public:
	typedef enum { Sigmoid = 0, ReLU, LReLU } ActType;

	ActType act_type_;

	VectorND<F> act_;	// activation values of neurons
	VectorND<F> grad_;	// gradients for back-propagation

public:
	LayerBase(){}
	~LayerBase(){}
	
	void initialize(const int& dim, const ActType& type);
	void resize(const int& dim);
	void assignErrorToGrad(const VectorND<F>& target);
	void activate();
	void multiplyActGradToGrad();
	void setActType(const ActType& type);
	void check() const {
		act_.check();
		grad_.check();
	}
};

// end of file 
