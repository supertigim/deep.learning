#include "Convolution2D.h"
#include "ConvConnection2D.h"

void ConvConnection2D::forward(const VectorND<F>& pre_layer_acts_, VectorND<F>& next_layer_acts) const  {
	
	for (int c = 0; c < channel_list_.size(); ++c) {
		ConvImage2D in_temp(channel_list_[c]->input_image_.width_, channel_list_[c]->input_image_.height_, pre_layer_acts_.values_ + channel_list_[c]->in_offset_);
		ConvImage2D out_temp(channel_list_[c]->output_image_.width_, channel_list_[c]->output_image_.height_, next_layer_acts.values_ + channel_list_[c]->out_offset_);

		Conv2D::forward(channel_list_[c]->filter_, in_temp, out_temp);
	}	
}

void ConvConnection2D::backward(const VectorND<F>& next_layer_grads_, VectorND<F>& pre_layer_grads_) const {
	for (int c = 0; c < channel_list_.size(); ++c) {
		ConvImage2D in_grad_temp(channel_list_[c]->input_image_.width_, channel_list_[c]->input_image_.height_, pre_layer_grads_.values_ + channel_list_[c]->in_offset_);
		ConvImage2D out_grad_temp(channel_list_[c]->output_image_.width_, channel_list_[c]->output_image_.height_, next_layer_grads_.values_ + channel_list_[c]->out_offset_);

		Conv2D::backward(channel_list_[c]->filter_, out_grad_temp, in_grad_temp);
	}
}

void ConvConnection2D::updateWeights(const F& eta, const F& alpha, const VectorND<F>& next_layer_grad, const VectorND<F>& prev_layer_act) {
	for (int c = 0; c < channel_list_.size(); ++c) {
		ConvImage2D in_act_temp(channel_list_[c]->input_image_.width_, channel_list_[c]->input_image_.height_, prev_layer_act.values_ + channel_list_[c]->in_offset_);
		ConvImage2D out_grad_temp(channel_list_[c]->output_image_.width_, channel_list_[c]->output_image_.height_, next_layer_grad.values_ + channel_list_[c]->out_offset_);

		Conv2D::updateWeights(eta, alpha, out_grad_temp, in_act_temp, channel_list_[c]->filter_);
	}
}

// end of file
