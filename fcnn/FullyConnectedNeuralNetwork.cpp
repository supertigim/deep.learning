#include <iostream>
#include "NeuralNetwork.h"

class dataSet {
public:
	VectorND<F> x_,y_;
};

void training_test() {

	// number of inputs : 2 
	//
	// input 값을 어떤 값으로 바꾸어도 결과는 0.3f로 되는 Neural Network로 훈련 
	// input 1 - 0.1f
	// input 2 - 0.2f
	//
	// target result: 0.3f 
	//
	VectorND<F> x(2);
	x[0] = 0.1f;
	x[1] = 0.2f;

	VectorND<F> y_target(1);
	y_target[0] = 0.3f;

	VectorND<F> y_predicted(1);

	NeuralNetwork nn;
	// input:2  / output:1 / hidden layer:1
	nn.initialize(2,1,1);
	// learning rate 
	nn.alpha_ = 0.1;

	// number of training 
	const int train = 100;
	
	for (int i = 0; i < train; ++i)
	{
		nn.setInputVector(x);
		nn.feedForward();
		nn.propBackward(y_target);
	}

	nn.setInputVector(x);
	std::cout << "Input: " << x[0] << ", " << x[1] << endl;
	nn.feedForward();
	nn.copyOutputVectorTo(false, y_predicted);
	std::cout << "Predicted Value: " << y_predicted << endl;
}


void XOR_problem() {

	dataSet xor_data[4];

	for(int i = 0 ; i < 4 ; ++i) {
		xor_data[i].x_.initialize(2);
		xor_data[i].y_.initialize(1);
	}

	xor_data[0].x_[0] = 0.0f; 
	xor_data[0].x_[1] = 0.0f;
	xor_data[0].y_[0] = 0.0f;

	xor_data[1].x_[0] = 0.0f; 
	xor_data[1].x_[1] = 1.0f;
	xor_data[1].y_[0] = 1.0f;

	xor_data[2].x_[0] = 1.0f; 
	xor_data[2].x_[1] = 0.0f;
	xor_data[2].y_[0] = 1.0f;

	xor_data[3].x_[0] = 1.0f; 
	xor_data[3].x_[1] = 1.0f;
	xor_data[3].y_[0] = 0.0f;

	NeuralNetwork nn;
	
	// input:2  / output:1 / hidden layer:1
	nn.initialize(2,1,1);

	/*
	If number of hidden layers is 2 or more, you should use LRELU as activation function with 2500 times training 
	However, it is not easy to train multiple hidden layer  
	
	hidden layer를 2개 쓰면, LRELU로 해야 예측할 수 있다. 
	training은 약 2500일때... 
	*/
	
	// learning rate 
	//nn.eta_ = (F) 1e-4;
	nn.eta_ = (F) 0.1f;
	// momentum coefficient 
	nn.alpha_ = (F)0.9;

	VectorND<F> y_temp(1);

	// number of training 
	const int train = 10000;
	for (int t = 0; t < train; t++)
	{
		std::cout << "[Training: " << t << "]" << endl;

		for(int i = 0 ; i < 4 ; i++) {
			nn.setInputVector(xor_data[i].x_);	
			nn.feedForward();

			nn.copyOutputVectorTo(false, y_temp);
			std::cout << xor_data[i].x_[0] <<":" << xor_data[i].x_[1]
					<< " -> " << xor_data[i].y_ << " | prediction: " << y_temp << endl;

			nn.propBackward(xor_data[i].y_);
		}
		std::cout << endl;	
	}
}

// display how to use this application like help option
void usage_display(){
	std::cout << "Usage:" << endl;
	std::cout << "  ./fcnn <choice> " << endl << endl;
	std::cout << "Choice:" << endl;
	std::cout << "   1                   Fully Connected NN Test" << endl;
	std::cout << "   2                   XOR Problem      " << endl;
}

int main (int argc, char** argv) {

	if(argc != 2) {
		usage_display();
		return 1;
	}

	int arg = std::atoi(argv[1]);

	switch (arg){
	case 1:
		training_test();
		break;
	case 2: 
		XOR_problem();
		break;
	default:
		usage_display();
		break;
	}
	return 0;
}


// end of file 
