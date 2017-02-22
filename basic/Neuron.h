#pragma once

#include <iostream>

const char endl[] = "\n";
const float DEFAULT_WEIGHT = 2.0f;
const float  DEFAULT_BIAS =	1.0f;

class Neuron {

public:
	Neuron()
		: w_(DEFAULT_WEIGHT), b_(DEFAULT_BIAS)
	{}

	Neuron(const double w, const double b)
		: w_(w), b_(b)
	{}

	double feedForward(const double& input) {
		input_ = input;

		// get Sigma
		const double sigma = w_*input + b_;
		output_ = getActivation(sigma);
		return output_;
	}

	void propBackward(const double& target){

		// learning rate should not be big enough to overfit
		// ex> 0.1 makes overfitting with the given training set 
		const double eta = 0.01; // learning rate

		const double grad = (output_ - target) * getActGrad(output_);

		// last input_ came from d(wx+b)/dw = x
		w_ -= eta * grad * input_;

		// last 1.0 came from d(wx+b)/db
		b_ -= eta * grad * 1.0;
	}

	double getActivation(const double& x){
		// linear 
		return x;
	}

	// derivate of y = x is dy/dx = 1
	double getActGrad(const double& x){
		return 1.0;
	}

	void display(){
		std::cout << "input: " << input_ << " | output: " << output_ << endl;
	}

	double operator () () { return output_; }

private:
	double w_, b_;				// weight and bias
	double input_, output_; 	// saved for back propagation

};

// end of file 
