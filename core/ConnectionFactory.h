#pragma once

#include "FullConnection.h"
#include "ConvConnection2D.h"

namespace ConnectionFactory {

	FullConnection *getFullConnection(const int& num_next, 
									const int& num_prev, 
									const F& rand_scale, 
									const F& rand_min) {
		
		FullConnection *new_conn = new FullConnection;

		// row x column = (dimension of next layer  - 1 for bias) x  (dimension of prev layer - this includes bias)
		// -1 is for bias. y = W [x b]^T. Don't subtract 1 if you want [y b]^T = W [x b]^T.
		//const int num_next = layers_[c + 1].act_.num_dimension_ - 1;
		//const int num_prev = layers_[c].act_.num_dimension_;
		new_conn->weights_.initialize(num_next, num_prev);
		new_conn->delta_weights_.initialize(num_next, num_prev);

		// random initialization
		new_conn->weights_.assignRandom(rand_scale, rand_min);
		new_conn->delta_weights_.assignAll(0.0);

		return new_conn;
	}

	ConvConnection2D *getConvConnection2D() {
		ConvConnection2D *new_conn = new ConvConnection2D;

		return new_conn;
	}
};

// end of file
