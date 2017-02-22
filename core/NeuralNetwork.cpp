#include "NeuralNetwork.h"
#include "ConnectionFactory.h"
#include <ctime>

NeuralNetwork::NeuralNetwork() {
	// do nothing
}

NeuralNetwork::~NeuralNetwork() {
	for (int c = 0; c < connections_.num_elements_; ++c) {
		delete connections_[c];
	}
}

NeuralNetwork::NeuralNetwork(const int& _num_input, const int& _num_output, const int& _num_hidden_layers) {
    initialize(_num_input, _num_output, _num_hidden_layers);
}

void NeuralNetwork::initialize(const int& _num_input, const int& _num_output, const int& _num_hidden_layers) {
	num_input_ = _num_input;                       
	num_output_ = _num_output;
	num_all_layers_ = _num_hidden_layers + 2; // hidden layers + 1 input layer + 1 output layer

	bias_ = (F)1;
	eta_ = (F)0.15;
	alpha_ = (F)0.5;

	// initialize all layers
	layers_.initialize(num_all_layers_);

	// from input layer to the last of hidden layer
	for (int l = 0; l < num_all_layers_-1; ++l) {
		layers_[l].initialize(num_input_ + 1, LayerBase::LReLU);	// +1 is bias
	}
	// output layer 
	layers_[num_all_layers_ - 1].initialize(num_output_ + 1, LayerBase::LReLU); // +1 is bias

	// initialize connections between layers
	connections_.initialize(num_all_layers_ - 1);
	connections_.assignAllValues(nullptr);
	for (int c = 0; c < connections_.num_elements_; ++c) {
		setFullConnection(c, 0.1f, 0.01f);
	}
}

void NeuralNetwork::setFullConnection(const int& con_ix, const F& rand_scale, const F& rand_min)
{
	assert(con_ix >= 0);
	assert(con_ix < connections_.num_elements_);

	SAFE_DELETE(connections_[con_ix]);

	const int num_next = layers_[con_ix + 1].act_.num_dimension_ - 1;
	const int num_prev = layers_[con_ix].act_.num_dimension_;
	connections_[con_ix] = (ConnectionBase*)ConnectionFactory::getFullConnection(num_next, num_prev, rand_scale, rand_min);
}

ConvConnection2D* NeuralNetwork::setConvConnection2D(const int& con_ix)
{
	assert(con_ix >= 0);
	assert(con_ix < connections_.num_elements_);

	SAFE_DELETE(connections_[con_ix]);

	ConvConnection2D* new_conn = ConnectionFactory::getConvConnection2D();

	connections_[con_ix] = (ConnectionBase*)new_conn;

	return new_conn;
}

void NeuralNetwork::feedForward()
{
    for (int l = 0; l < connections_.num_elements_; ++l) {
        connections_[l]->forward(layers_[l].act_, layers_[l + 1].act_);
		layers_[l + 1].activate();
    }
}

// backward propagation
void NeuralNetwork::propBackward(const VectorND<F>& target) {
	//Note: two steps are implemented separately in case for combining multiple neural networks
	propBackwardError(target);
	updateConnectionWeights();
}

void NeuralNetwork::propBackwardError(const VectorND<F>& target) {
	layers_[layers_.num_elements_ - 1].assignErrorToGrad(target);

	for (int l = connections_.num_elements_ - 1; l >= 0; --l) {
		layers_[l + 1].multiplyActGradToGrad();

		connections_[l]->backward(layers_[l + 1].grad_, layers_[l].grad_);
	}
}

void NeuralNetwork::updateConnectionWeights() {
	for (int l = connections_.num_elements_ - 1; l >= 0; l--)
		connections_[l]->updateWeights(eta_, alpha_, layers_[l + 1].grad_, layers_[l].act_);
}

void NeuralNetwork::setInputVector(const VectorND<F>& input) {
    if (input.num_dimension_ < num_input_)  // num_input_ = number of input values + 1 bias
        std::cout << "Input dimension is wrong" << endl;

	layers_[0].act_.copyPartial(input, 0, 0, num_input_);
}

int NeuralNetwork::getOutputIXMaxComponent() {
    const VectorND<F>& output_layer_act(layers_[layers_.num_elements_ - 1].act_);

    F max = output_layer_act[0];
    int ix = 0;

    for (int d = 1; d < num_output_; ++d) {
        if (max < output_layer_act[d]) {
            max = output_layer_act[d];
            ix = d;
        }
    }

    return ix;
}

// return within number of output
// ex. if output is 3, then return will be among 0~2
int NeuralNetwork::getOutputIXEpsilonGreedy(const F& epsilon) {
    if (epsilon > 0.0) {
    	//srand((unsigned int)time(nullptr));
    	//std::cout << rand() % num_output_ << endl;
        if ((F)rand() / RAND_MAX < epsilon) {
        	//std::cout << "Epsilon Greedy" << endl;
            return rand() % num_output_;
        }
    }
    return getOutputIXMaxComponent();
}

int NeuralNetwork::getOutputIXProbability() {
    const VectorND<F>& output_layer_act(layers_[layers_.num_elements_ - 1].act_);

    VectorND<F> possibility;
    possibility.initialize(num_output_, true);

    F sum = 0;

    for (int d = 0; d < num_output_; ++d) {
        sum += output_layer_act[d];
    }

    if (sum == 0.0) return 0;

    F accum = 0.0;

    for (int d = 0; d < num_output_; ++d) {
        accum += output_layer_act[d] / sum;
        possibility[d] = accum;
    }

    const F r = (F)rand() / RAND_MAX;

    for (int d = 0; d < num_output_; ++d) {
        if (r < possibility[d]) return d;
    }

    return num_output_ - 1;
}

F NeuralNetwork::getOutput(const int& ix)
{
	return layers_[layers_.num_elements_ - 1].act_[ix];
}

F NeuralNetwork::getOutputValueMaxComponent()
{
	const VectorND<F>& output_layer_act(layers_[layers_.num_elements_ - 1].act_);

    F max = output_layer_act[0];
    int ix = 0;

    for (int d = 1; d < num_output_; ++d) {
        if (max < output_layer_act[d]) {
            max = output_layer_act[d];
            ix = d;
        }
    }

    return max;
}

void NeuralNetwork::copyOutputVectorTo(const bool& copy_bias, VectorND<F>& copy)
{
    const VectorND<F>& output_layer_act(layers_[layers_.num_elements_ - 1].act_);

    if (copy_bias == false) {
        
        //TODO: do not reinitialize if copy.num_dimension >= num_output_.num_dimension
        copy.initialize(num_output_, false);

        for (int d = 0; d < num_output_; ++d) {
            copy[d] = output_layer_act[d];
        }
    }
    else {
        copy.initialize(num_output_ + 1, false);

        for (int d = 0; d < num_output_ + 1; ++d) {
            copy[d] = output_layer_act[d];
        }
    }
}

void NeuralNetwork::readTXT(const std::string& filename) {
	std::ifstream ifs;
	ifs.open(filename);
	//TODO: check file error
	int num_cons = 0;
	ifs >> num_cons;
	std::cout << "Number of connections = " << num_cons << endl;

	ifs.close();
}

void NeuralNetwork::writeTXT(const std::string& filename) const {
	std::ofstream of;
	of.open(filename);

	of << connections_.num_elements_ << endl;

	for (int l = 0; l < connections_.num_elements_; ++l) {
		connections_[l]->writeTXT(of);
	}

	of << layers_.num_elements_ << endl;

	for (int l = 0; l < layers_.num_elements_; ++l) {
		layers_[l].act_.writeTXT(of);
	}

	of.close();
}

F NeuralNetwork::getL2NormError(const VectorND<F>& desired) {
	const VectorND<F>& output_layer_act(layers_[layers_.num_elements_ - 1].act_);

	F sum = (F)0;

	for (int d = 0; d < num_output_; ++d) {
		sum += POW2(desired[d] - output_layer_act[d]);
	}

	return sqrt(sum);
}

F NeuralNetwork::getLinfNormError(const VectorND<F>& desired) {
	const VectorND<F>& output_layer_act(layers_[layers_.num_elements_ - 1].act_);

	F temp = (F)0;

	for (int d = 0; d < num_output_; ++d) {
		temp = MAX2(temp, ABS(desired[d] - output_layer_act[d]));
	}

	return temp;
}


// end of file 
