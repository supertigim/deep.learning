#pragma once

#include "ConnectionBase.h"
#include "ConvChannel2D.h"
#include <vector>

class ConvConnection2D : public ConnectionBase
{
public:
	std::vector<ConvChannel2D*> channel_list_;

	~ConvConnection2D() {
		for (int i = 0; i < channel_list_.size(); ++i) {
			SAFE_DELETE(channel_list_[i]);
		}
	}

	void forward(const VectorND<F>& pre_layer_acts_, VectorND<F>& next_layer_acts) const;
	void backward(const VectorND<F>& next_layer_grads_, VectorND<F>& pre_layer_grads_) const;
	void updateWeights(const F& eta, 
					const F& alpha, 
					const VectorND<F>& next_layer_grad, 
					const VectorND<F>& prev_layer_act);

	void check() const {
		//weights_.check();
		//delta_weights_.check();
	}
};

// end of file
