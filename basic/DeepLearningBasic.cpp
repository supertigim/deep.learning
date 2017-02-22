#include "Neuron.h"
#include <random>
#include <ctime>

enum{
	E_NUM_DIFF = 1,
	E_PROP_BACKWARD
};

// randome number generator 
template<typename T> inline
typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
	std::mt19937 rng((unsigned int) time(NULL));
    std::uniform_real_distribution<T> dst(min, max);
    return dst(rng);
}

// y = (a * x) + b
float getF(const float& x){

	const float a = 0.5;
	const float b = 0.1;

	return a * x + b;
}

// numeric differentiation
void numeric_diff(){

	const float x = 1.0; 							// input
	const float dx = uniform_rand(0.001f,0.1f); 	// randomly set

	// derivative 
	const float df = getF(x + dx) - getF(x);
	const float dfdx = df / dx;

	//const float f_target = 0.7;	// output that is randomly set 
	const float f_target = uniform_rand(0.1f,10.1f);

	// df/dx = (f_target - getF(x)) / dx_target
	// it means that you can get dx_target when you know df/dx 
	const float dx_target = (f_target - getF(x)) / dfdx;

	std::cout << "f_target = " << f_target << endl;
	std::cout << "f(x + dx_target) = " << getF(x + dx_target) << endl;

	// belows is 0.2 so, to get 0.7 as ouput, 1.2 (1.0 + 0.2) is expected as input
	std::cout << "dx_target = " << dx_target << endl 
				<< "-------------------------"<< endl;
	std::cout 	<< "[RESULT] " << endl
				<< "x = x + dx_target "<< endl
				<< "F(x=" << x+dx_target << ") = " << f_target << endl;
}

// test backward propagation using 1 neural 
void show_basic_theory_for_training(){

	// random value for weight and bias
	const double weight = uniform_rand(-1.0f,1.0f);
	const double bias = uniform_rand(1.0f,10.0f);

	std::cout<< "W: " << weight << " | B: " << bias << "\n";

	Neuron neuron(weight, bias);

	// expected function : y = W * x + B 
	//			   ,where  W: 5.0f / B:1.0f
	const double training_data_set[]=	{1.0f,	// input
										 6.0f,	// ouput desired 

										 2.0f,
										 11.0f,

										 7.0f,
										 36.0f};

	// traing here //////////////////////////////////////////////////////

	// accuracy can reach 100 percent after 5000 times training
	for (int i = 0 ;  i < 500 ; ++i){

		neuron.feedForward(training_data_set[0]);
		neuron.propBackward(training_data_set[1]);	

		neuron.feedForward(training_data_set[2]);
		neuron.propBackward(training_data_set[3]);	

		neuron.feedForward(training_data_set[4]);
		neuron.propBackward(training_data_set[5]);	
	}

	// result verification  ///////////////////////////////////////////////

	neuron.feedForward(training_data_set[0]);
	neuron.display();
	std::cout << "accuracy : " << neuron()/training_data_set[1] << endl;
	neuron.feedForward(training_data_set[2]);
	neuron.display();		
	std::cout << "accuracy : " << neuron()/training_data_set[3] << endl;
	neuron.feedForward(training_data_set[4	]);
	neuron.display();		
	std::cout << "accuracy : " << neuron()/training_data_set[5] << endl;
}

// display how to use this application like help option
void usage_display(){
	std::cout << "Usage:" << endl;
	std::cout << "  ./deeplearning <choice> " << endl << endl;
	std::cout << "Choice:" << endl;
	std::cout << "   1                   Numeric Differentiation" << endl;
	std::cout << "   2                   Back Propagartion      " << endl;
}

int main(int argc, char** argv) {

	if(argc != 2) {
		usage_display();
		return 1;
	}

	int arg = std::atoi(argv[1]);

	switch(arg) {
		case E_NUM_DIFF:
			numeric_diff();
			break;
		case E_PROP_BACKWARD:
			show_basic_theory_for_training();
			break;
		default:
			usage_display();
	}
	return 0;
}

//end of file
