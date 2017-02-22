#pragma once

#include "NeuralNetwork.h"
#include "RLMemory.h"

class ReinforcementLearning {
public:
	//int num_exp_replay_;			// histories for experience replay training
	int num_input_histories_;		// input to nn
	int num_state_variables_;   	// dimension of the state variables of the game
	int num_game_actions_;			// num outputs of the game

	float gamma_;
	
	NeuralNetwork nn_;
	RLMemory memory_;

	VectorND<F> old_input_vector_, reward_vector_, next_input_vector_;
public:
	void initialize();
	void initializeConv2D(int height, int width);

	void trainReward();
	void trainBatch(int tr_num);
	void trainRewardMemory();
	void trainReward(const int ix_from_end);
	
	void forward();

	void recordHistory(const VectorND<F>& state_vector, const F& reward, const int& choice, const VectorND<F>& q_values);
	void makeInputVectorFromHistory(const int& ix_from_end, VectorND<F>& input);
};

// end of file
