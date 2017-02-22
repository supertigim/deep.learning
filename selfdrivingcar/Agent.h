#pragma once 

#include "ReinforcementLearning.h"

class TestDrivingScene;

class Agent {
protected:
	TestDrivingScene* simul_;
	bool is_training_;
	ReinforcementLearning rl_;

	float reward_sum_;
	float reward_max_;
public:
	Agent(TestDrivingScene* simul);

	void init();

	const int getSelectedDir();
	void driveCar();
protected:
	bool handleKey();
};

// end of file
