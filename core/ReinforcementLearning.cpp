#include "ReinforcementLearning.h"


void ReinforcementLearning::initialize() {
	// initialize neural network
	const int num_hidden_layers = 1;

	std::cout << "[ReinforcementLearning::initialize]input num:" << num_state_variables_ * num_input_histories_ << endl;
	nn_.initialize(num_state_variables_ * num_input_histories_, num_game_actions_, num_hidden_layers);

	for (int i = 0; i <= num_hidden_layers + 1; ++i)
		nn_.layers_[i].act_type_ = LayerBase::ReLU;

	nn_.eta_ = 1e-5;
	nn_.alpha_ = 0.9;

	gamma_ = 0.9f;

	// initialize replay memory
	memory_.reserve(1e5);

	old_input_vector_.initialize(nn_.num_input_, true);
	next_input_vector_.initialize(nn_.num_input_, true);
}

void ReinforcementLearning::initializeConv2D(int height, int width) {
	
	const int num_channels = 1;

	//assert(num_exp_replay_ >= num_input_histories_);

	nn_.initialize(num_state_variables_ * num_input_histories_
				, num_game_actions_
				, 2);

	nn_.layers_[0].act_type_ = LayerBase::LReLU;
	nn_.layers_[1].act_type_ = LayerBase::LReLU;
	nn_.layers_[2].act_type_ = LayerBase::LReLU;

	nn_.eta_ = 1e-4;
	nn_.alpha_ = 0.9;

	nn_.layers_[0].act_type_ = LayerBase::LReLU;
	//conn 0 : filter
	nn_.layers_[1].act_type_ = LayerBase::LReLU;
	//conn 1 : averaging
	nn_.layers_[2].act_type_ = LayerBase::LReLU;
	//conn 2 : full
	nn_.layers_[3].act_type_ = LayerBase::LReLU;

	nn_.layers_[1].initialize(num_input_histories_ * num_state_variables_ * num_channels + 1, LayerBase::LReLU);
	nn_.layers_[2].initialize(num_input_histories_ * num_channels + 1, LayerBase::LReLU);
	nn_.setFullConnection(1, 0.1f, 0.01f);
	nn_.setFullConnection(2, 0.1f, 0.01f);

	const int first_filter_size = 5; // 5 by 5 
	{
		ConvFilter2D filter;
		filter.initialize(first_filter_size, first_filter_size
						, 1, 1
						, 2, 2
						, 0.1, 0.01);
		ConvImage2D im;
		im.width_ = width;
		im.height_ = height;;
		ConvImage2D om;
		om.width_ = width;
		om.height_ = height;

		ConvConnection2D *new_conv = nn_.setConvConnection2D(0);

		int in_count = 0;
		int out_count = 0;
		for (int h = 0; h < num_input_histories_; ++h) {
			for (int ch = 0; ch < num_channels; ++ch) {
				new_conv->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, in_count, om, out_count));
				out_count += num_state_variables_;
			}
			in_count += num_state_variables_;
		}
	}

	{
		ConvFilter2D filter;
		filter.initialize(width, height, 1, 1, 0, 0, 0.1, 0.01);
		ConvImage2D im;
		im.width_ = width;
		im.height_ = height;
		ConvImage2D om;
		om.width_ = 1;
		om.height_ = 1;

		int in_count = 0;
		int out_count = 0;

		ConvConnection2D *new_conv = nn_.setConvConnection2D(1);

		for (int h = 0; h < num_input_histories_; ++h) {
			for (int ch = 0; ch < num_channels; ++ch) {
				new_conv->channel_list_.push_back(new ConvChannel2D(filter, (F)0.1, (F)0.01, im, in_count, om, out_count));

				in_count += num_state_variables_;
				++out_count;
			}
		}
	}

	memory_.reserve(1e5);

	gamma_ = 0.95f;

	old_input_vector_.initialize(nn_.num_input_, true);
	next_input_vector_.initialize(nn_.num_input_, true);
}

void ReinforcementLearning::trainReward(bool boost) {
	// train with last memory
	trainReward(0, boost);	
}

void ReinforcementLearning::trainBatch(int tr_num, bool boost) {
	
	for (int m_tr = memory_.num_elements_ - 2; m_tr >= num_input_histories_; --m_tr) {
		
		const int m = rand() % (memory_.num_elements_ - 1 - num_input_histories_) + num_input_histories_;
		
		if (m != memory_.num_elements_ - 2) 
		{	
			const int inv_m = m - (memory_.num_elements_ - 1);
			trainReward(inv_m, boost);					
		}

		if(tr_num > 0)	--tr_num;
		else  	   		break;
	}

}

void ReinforcementLearning::trainRewardMemory(bool boost) {
	// std::cout << "State History Num:" << memory_.count() << "\n";
	for (int ix_from_end = 0; ix_from_end > -(memory_.count() - num_input_histories_); --ix_from_end)
		trainReward(ix_from_end, boost);
}

void ReinforcementLearning::trainReward(const int ix_from_end, bool boost) {
	// guess next Q value
	makeInputVectorFromHistory(ix_from_end, next_input_vector_);

	//const float reward_ix = history_.getValue(history_.array_.num_elements_ - 1 - 0 + ix_from_end).reward_;
	const float reward_ix = memory_.getRewardFromLast(ix_from_end);

	nn_.setInputVector(next_input_vector_);
	nn_.feedForward();

	const float next_Q = reward_ix < 0.0f ? 0.0f : nn_.getOutputValueMaxComponent(); // final status test
	//const float next_Q = nn_.getOutputValueMaxComponent();

	//const int selected_dir = history_.getValue(history_.getLastIndex() -1 + ix_from_end).choice_; // last history is in one step future 
	const int selected_dir = memory_.getSelectedIxFromLast(ix_from_end);
	//TODO: default training direction!

	makeInputVectorFromHistory(ix_from_end-1, old_input_vector_);

	if(!boost){
		nn_.setInputVector(old_input_vector_); // old input
		nn_.feedForward();
		nn_.copyOutputVectorTo(false, reward_vector_);
		reward_vector_[selected_dir] = reward_ix + gamma_ * next_Q;
		nn_.propBackward(reward_vector_);
		nn_.check();	
	}
	else {
		//const float high_reward_th = 0.8;
		const float high_reward_tr_ep = 0.1f;

		//if (high_reward_th <= reward_ix || reward_ix < 0.1)
		{
			int count = 0;
			while (true) {
			//{
				nn_.setInputVector(old_input_vector_); // old input
				nn_.feedForward();
				nn_.copyOutputVectorTo(false, reward_vector_);
				
				const float target = reward_ix + gamma_ * next_Q;
				const float error = ABS(reward_vector_[selected_dir] - target);
				
				reward_vector_[selected_dir] = reward_ix + gamma_ * next_Q;

				nn_.propBackward(reward_vector_);

				nn_.check();

				if (error < high_reward_tr_ep || count > 10000)
				{
					//std::cout << "High reward training error " << error << " " << reward_vector_[selected_dir] << " " << (reward_ix + gamma_ * next_Q) << endl;
					break;
				}
				++count;
			}
			//std::cout << "High reward training end" << endl;
		}
	}
}

void ReinforcementLearning::forward() {
	makeInputVectorFromHistory(0, old_input_vector_);
	nn_.setInputVector(old_input_vector_);
	nn_.feedForward();
}

// push back this to history
void ReinforcementLearning::recordHistory(const VectorND<F>& state_vector, const F& reward, const int& choice, const VectorND<F>& q_values) {
	memory_.append(state_vector, choice, reward, q_values);
}

void ReinforcementLearning::makeInputVectorFromHistory(const int& ix_from_end, VectorND<F>& input) {
	for (int r = 0, count = 0; r < num_input_histories_; ++r, count += num_state_variables_) {
		const VectorND<F> &state_vector 
				= memory_.getStateVectorFromLast(ix_from_end - r);

		input.copyPartial(state_vector, count, 0, num_state_variables_);
	}
}

// end of file
