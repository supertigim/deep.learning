#include "NeuralNetwork.h"
#include <ctime>

NeuralNetwork nn_;

int main(int argc, char** argv)
{
	const int width = 10, height = 10;

	VectorND<F> input_image(width*height), 
				output(2), 
				desired(2);

	input_image.assignAllValues((F)0);

	nn_.initialize(width * height, 		// 입력 수
				output.num_dimension_,	// output is (x, y) 
				2); 					// hidden layer 수

	nn_.eta_ = (F)1e-4;
	nn_.alpha_ = (F)0.9;

	// hidden layer 2 --> need initialization here, because of poor implementation :)
	nn_.layers_[1].initialize(width*height * 2 + 1, LayerBase::ReLU);
	
	// first hidden layer into convolutional 
	{
		ConvFilter2D filter;
		filter.initialize(5, 5, 		// filter 크기 
						 1, 1,  		// stride 크기 
						 2, 2,			// padding 크기 
						 0.1, 0.01);	// random varialbe scope 
		ConvImage2D im;
		im.width_ = width;
		im.height_ = height;
		ConvImage2D om;
		om.width_ = width;
		om.height_ = height;

		// 0 --> first hidden layer 
		ConvConnection2D *new_conn = nn_.setConvConnection2D(0);
		new_conn->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, 0, om, 0));
		new_conn->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, 0, om, width*height));
	}

	// second hidden layer intto convolutional 
	{
		ConvFilter2D filter;
		filter.initialize(width, height,
						 1, 1, 
						 0, 0, 
						 0.1, 0.01);
		ConvImage2D im;
		im.width_ = width;
		im.height_ = height;

		// data 출력 형태 
		ConvImage2D om;
		om.width_ = 1;
		om.height_ = 1;

		// 1 --> second hidden layer 
		// why two channel -> x / y 값 
		ConvConnection2D *new_conn = nn_.setConvConnection2D(1);
		new_conn->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, 0, om, 0));
		new_conn->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, width * height, om, 1));
	}

	srand((unsigned int)time(0));

	while (1)
	{
		F max_error = (F)0;

		for (int r = 0; r < width*height*100; r++)
		{
			// Create training dataset 
			const int rand_i = rand() % width;
			const int rand_j = rand() % height;

			// 점하나만 찍을 때 
			input_image.values_[rand_i + width* rand_j] = (F)1.0;

			// 십자가 그림 이미지
			//input_image.values_[rand_i - 1 + width* rand_j] = (T)1.0;
			//input_image.values_[rand_i + width* rand_j] 	= (F)1.0;
			//input_image.values_[rand_i + 1 + width* rand_j] = (T)1.0;
			//input_image.values_[rand_i + width* (rand_j - 1)] = (T)1.0;
			//input_image.values_[rand_i + width* (rand_j + 1)] = (T)1.0;

			// x/y 좌표 
			desired[0] = (F)rand_i / (F)width;
			desired[1] = (F)rand_j / (F)height;

			nn_.setInputVector(input_image);
			nn_.feedForward();
			nn_.copyOutputVectorTo(false, output);

			const F linferror = nn_.getLinfNormError(desired);
			//std::cout << "F: " << dynamic_cast<ConvConnection2D*>(nn_.connections_[0])->channel_list_[0]->filter_.weights_ << endl;

			max_error = MAX2(linferror, max_error);

			nn_.propBackward(desired);

			// Image buffer reset 
			input_image.values_[rand_i + width* rand_j] = (F)0.0;
		}

		std::cout << "Max error = " << max_error << endl;

		if (max_error < 0.00015) {
			nn_.writeTXT("NN.txt");
			return 0;
		}
	}

	return 0;
}

// end of file
