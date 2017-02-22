#include "LayerBase.h"
#include "ActivationFunctions.h"

void LayerBase::resize(const int& dim) {
	act_.initialize(dim, true);
	grad_.initialize(dim, true);

	act_[act_.num_dimension_ - 1] = 1.0;	// last element is bias
}

void LayerBase::initialize(const int& dim, const ActType& type) {
	act_type_ = type;
	resize(dim);
}

void LayerBase::assignErrorToGrad(const VectorND<F>& target) {
	// target may include bias or not
	assert(act_.num_dimension_ >= target.num_dimension_); 
	assert(grad_.num_dimension_ >= target.num_dimension_);
	assert(act_.num_dimension_ == grad_.num_dimension_);

	for (int d = 0; d < act_.num_dimension_ - 1; ++d) {
		grad_[d] = (target[d] - act_[d]);
	}
}

// first input -> number of bias 
void LayerBase::activate() {
	switch (act_type_) {
	case Sigmoid:
		ActFunc::applySigmoidToVector(1, act_);		
		break;

	case LReLU:
		ActFunc::applyLRELUToVector(1, act_);		
		break;

	case ReLU:
	default:
		ActFunc::applyRELUToVector(1, act_);
		break;
	}
}

void LayerBase::multiplyActGradToGrad()
{
	switch (act_type_)
	{
	case Sigmoid:
		// skip last component (bias)
		for (int d = 0; d < act_.num_dimension_ - 1; ++d) {
			grad_[d] *= ActFunc::SigmoidGradFromY(act_[d]);
		}
		break;
	case LReLU:
		// skip last component (bias)
		for (int d = 0; d < act_.num_dimension_ - 1; ++d) {
			grad_[d] *= ActFunc::LRELUGradFromY(act_[d]);
		}
		break;
	case ReLU:
	default:
		// skip last component (bias)
		for (int d = 0; d < act_.num_dimension_ - 1; ++d) {
			grad_[d] *= ActFunc::RELUGradFromY(act_[d]);
		}
		break;
	}
}

void LayerBase::setActType(const ActType& type)
{
	act_type_ = type;
}


// end of file 
