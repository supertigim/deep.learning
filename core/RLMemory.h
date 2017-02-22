#pragma once

#include "VectorND.h"
#include <vector>

const int MAX_RESERVE = 1000;

class RLMemory {
public:
	int num_elements_;
	int num_reserve_;

	std::vector<VectorND<F> > state_vector_array_;
	std::vector<int> selected_array_;
	std::vector<F> reward_array_;
	std::vector<VectorND<F> > q_values_array_;

public:
	RLMemory()
		: num_reserve_(MAX_RESERVE)
		, num_elements_(0)
	{}

	int count() { return num_elements_;}
	
	void reserve(const int& num_reserve)
	{
		state_vector_array_.reserve(num_reserve);
		selected_array_.reserve(num_reserve);
		reward_array_.reserve(num_reserve);
		q_values_array_.reserve(num_reserve);

		num_reserve_ = num_reserve;
	}

	void reset()
	{
		num_elements_ = 0;

		state_vector_array_.clear();
		selected_array_.clear();
		reward_array_.clear();
		q_values_array_.clear();

		reserve(num_reserve_);
	}

	void append(const VectorND<F>& state_vector, const int& choice, const F& reward, const VectorND<F>& q_values)
	{
		assert(num_elements_ < num_reserve_);

		state_vector_array_.push_back(state_vector);
		selected_array_.push_back(choice);
		reward_array_.push_back(reward);
		q_values_array_.push_back(q_values);

		++num_elements_;
	}

	// ix_from_last = 0 returns last element, use -1, -2 ,...
	const VectorND<F>& getStateVectorFromLast(const int& ix_from_last) {
		return state_vector_array_[num_elements_ - 1 + ix_from_last];
	}

	const int& getSelectedIxFromLast(const int& ix_from_last) {
		return selected_array_[num_elements_ - 1 + ix_from_last];
	}

	const F& getRewardFromLast(const int& ix_from_last) {
		return reward_array_[num_elements_ - 1 + ix_from_last];
	}

	// ix_from_last = 0 returns last element, use -1, -2 ,...
	const VectorND<F>& getQValuesFromLast(const int& ix_from_last) {
		return q_values_array_[num_elements_ - 1 + ix_from_last];
	}

};

// end of file
