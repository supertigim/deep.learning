#include "FullConnection.h"

void FullConnection::forward(const VectorND<F>& pre_layer_acts_, VectorND<F>& next_layer_acts) const {
	// The last component of acts_ is the bias and thus shouldn't be updated. 
	weights_.multiply(pre_layer_acts_, next_layer_acts);
}

void FullConnection::backward(const VectorND<F>& next_layer_grads_, VectorND<F>& pre_layer_grads_) const {
	weights_.multiplyTransposed(next_layer_grads_, pre_layer_grads_);
}

//Note: next_layer_grad and prev_layer_act
void FullConnection::updateWeights(	const F& eta, 
									const F& alpha, 
									const VectorND<F>& next_layer_grad, 
									const VectorND<F>& prev_layer_act) {
	// eta : learning rate
	// alpha : momentum coefficient

	const int num_rows = weights_.num_rows_;
	const int num_cols = weights_.num_cols_;

	for (int row = 0; row < num_rows; ++row) {
		for (int col = 0; col < num_cols; ++col) {
			F &old_delta_w = delta_weights_.getValue(row, col);

			const F delta_w = eta * next_layer_grad[row] * prev_layer_act[col] + alpha * old_delta_w;
			weights_.getValue(row, col) += delta_w;
			// update for the momentum term in next time step
			old_delta_w = delta_w;           
		}
	}
	
	//weights_.normalizeAllRows(1e-3);
}

// end of file
