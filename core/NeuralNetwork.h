#pragma once 

#include <iostream>
#include "Array1D.h"
#include "Matrix.h"
#include "ConnectionBase.h"
#include "FullConnection.h"
#include "LayerBase.h"
#include "ConvConnection2D.h"

class NeuralNetwork
{
public:
    int num_input_;
    int num_output_;
    int num_all_layers_;    // num_all_layers_ = num_hidden_layers_ + 2

    F   bias_;              // constant bias
    F   eta_;               // learning rate
    F   alpha_;             // momentum term coefficient

	Array1D<LayerBase> layers_; // layers_[0] = input_layer, layers_[num_all_layers_-1] = output_layer

    Array1D<ConnectionBase*>   connections_; // connections_[0] connects layer 0 and layer 1

    NeuralNetwork();
    NeuralNetwork(const int& _num_input, const int& _num_output, const int& _num_hidden_layers);
	~NeuralNetwork();

    void initialize(const int& _num_input, const int& _num_output, const int& _num_hidden_layers);
 
	void setFullConnection(const int& con_ix, const F& rand_scale, const F& rand_min);
	ConvConnection2D* setConvConnection2D(const int& con_ix);

    // forward propagation
    void feedForward();
    void setInputVector(const VectorND<F>& input);

    // back propagation
    void propBackward(const VectorND<F>& target);    // backward propagation
	void propBackwardError(const VectorND<F>& target);
	void updateConnectionWeights();

    // output access
    int getOutputIXEpsilonGreedy(const F& epsilon);
    int getOutputIXMaxComponent();
    int getOutputIXProbability();
    F   getOutputValueMaxComponent();
	F   getL2NormError(const VectorND<F>& desired);
	F   getLinfNormError(const VectorND<F>& desired); // http://lmp.ime.eb.br/intro_analise_de_erro.pdf
	F   getOutput(const int& ix);

    void copyOutputVectorTo(const bool& copy_bias, VectorND<F>& copy);

	void readTXT(const std::string& filename);
	void writeTXT(const std::string& filename) const;

	void check() const
	{
		for (int i = 0; i < layers_.num_elements_; ++i)
			layers_[i].check();

		for (int i = 0; i < connections_.num_elements_; ++i)
			connections_[i]->check();
	}
};

// end of file 
