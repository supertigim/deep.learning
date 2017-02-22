#include "Agent.h"
#include "TestDrivingScene.h"

Agent::Agent(TestDrivingScene* simul)
	: simul_(simul)
	, reward_sum_(0.0f)
	, reward_max_(0.0f)
	, is_training_(false)
{}

void Agent::init(){

	simul_->compat_state_ = true;
	simul_->state_buffer_.initialize(simul_->getNumStateVariables(), true);

	rl_.num_input_histories_ = 1;
	//rl_.num_exp_replay_ = 0;
	std::cout << "Agent::init() " << simul_->getNumStateVariables() << endl;
	rl_.num_state_variables_ = simul_->getNumStateVariables();
	rl_.num_game_actions_ = simul_->getNumActions();//TODO: obtain from game, left, right, stay

	rl_.initialize();

	for (int h = 0; h < rl_.num_input_histories_; h++) {
		rl_.recordHistory(simul_->getStateBuffer(), 0.0f, 2, VectorND<float>(simul_->getNumActions())); // choice 2 is stay
	}
}

const int Agent::getSelectedDir(){
	int selected_dir;

	// user supervised mode
	if (simul_->getKeyPressed(GLFW_KEY_LEFT) == true) selected_dir = 0;
	else if (simul_->getKeyPressed(GLFW_KEY_RIGHT) == true) selected_dir = 1;
	// AI mode
	else  {
		selected_dir = is_training_ == true ? rl_.nn_.getOutputIXEpsilonGreedy(0.2f) : rl_.nn_.getOutputIXEpsilonGreedy(0.0f);	
		//std::cout << "selected_dir: " << selected_dir << endl;
	}

	if (simul_->getKeyPressed(GLFW_KEY_Q) == true) {

		rl_.nn_.writeTXT("nn.txt");
		std::cout << "writing complete" << endl;
	}

	return selected_dir;
}
bool Agent::handleKey(){
	
	simul_->pollEvents();

	// Check if the ESC key was pressed or the window was closed
	if (simul_->getKeyPressed(GLFW_KEY_ESCAPE) || simul_->getWindowShouldClose())
		return false;

	// training mode change key input
	static bool key_reset_flag = true;
	if (simul_->getKeyPressed(GLFW_KEY_SPACE) == true) {

		if(key_reset_flag == true) {
			is_training_ = !is_training_;

			key_reset_flag = false;

			if (is_training_) {
				std::cout << "Back ground training mode" << endl;
			}
			else {
				std::cout << "Interactive rendering mode" << endl;
			}
		}
	}
	else {
		key_reset_flag = true;
	}
	return true;
}

void Agent::driveCar() {

	while(true) {
		
		if(!handleKey()) break;

		rl_.forward();
		VectorND<float> output_vector_temp;
		rl_.nn_.copyOutputVectorTo(false, output_vector_temp);
		//VectorND<float> output_target_temp;

		const int selected_dir = getSelectedDir();	// epsilon-greedy
		simul_->processInput(selected_dir);			//TODO: multiple input
		
		float reward;

		// 0 : continue
		// 1 : terminate
		int isTerminated; 
		simul_->update(!is_training_, reward, isTerminated);

		// record state and reward
		rl_.recordHistory(simul_->getStateBuffer(), reward, selected_dir, output_vector_temp);

		reward_sum_ += reward;

		// start state replay training at terminal state
		// this is terminal state
		if(isTerminated) {
			std::cout << "(max:" << reward_max_ << ") " 
						<< "Reward sum " << reward_sum_ << endl;
			
			int tr_num = 10;
			if (reward_max_ < reward_sum_) {

				reward_max_ = reward_sum_;
				tr_num = 100;

				std::cout << "**************************" << endl;
				std::cout << "** New record : " << reward_max_ << " **" << endl;
				std::cout << "**************************" << endl;
			}

			reward_sum_ = 0.0f;

			//rl_.trainStochastic(tr_num);
			//rl_.trainReward();
			
			if(is_training_)
				rl_.trainRewardMemory();
/*
			// train backward (state replay)
			if( true) {
				for(int tr = 0; tr < tr_num; tr ++) {
					for (int m_tr = rl_.state_.num_elements_ - 2; m_tr >= rl_.num_input_histories_; m_tr--) {
						// stochastic training
						int m = rand() % (rl_.state_.num_elements_ - 1 - rl_.num_input_histories_) + rl_.num_input_histories_;

						// state index from end
						const int inv_m = m - (rl_.state_.num_elements_ - 1);

						rl_.makeInputVectorFromHistory(inv_m, rl_.next_input_vector_);
						rl_.nn_.setInputVector(rl_.next_input_vector_);
						rl_.nn_.feedForward();

						// if next is not the terminal state
						//if (m == rl_.state_.num_elements_ - 2) 
						//	continue;
						
						const float Q_next = reward < 0.0f ? 0.0f : rl_.nn_.getOutputValueMaxComponent();
						const float reward_ix = rl_.state_.getRewardFromLast(inv_m);
						const int selected_dir = rl_.state_.getSelectedIxFromLast(inv_m);
						const float Q_target = reward_ix + rl_.gamma_ * Q_next;
						// Q_target = ...;
						const float high_reward_tr_ep = 0.09f;

						// forward propagation from previous inputs
						rl_.makeInputVectorFromHistory(inv_m-1, rl_.old_input_vector_);
						
						for (int i = 0; i < 10000; i ++) {
							rl_.nn_.setInputVector(rl_.old_input_vector_);
							rl_.nn_.feedForward();
							rl_.nn_.copyOutputVectorTo(false, output_target_temp);
							const float error = ABS(output_target_temp[selected_dir] - Q_target);

							output_target_temp[selected_dir] = Q_target;

							rl_.nn_.propBackward(output_target_temp);

							//std::cout << "Q_target" << Q_target 
							//		<< " output_target_temp[selected_dir]" << output_target_temp[selected_dir]
							//		<<" error" << error << endl;
							if (error < high_reward_tr_ep)
								break;
						}

						rl_.nn_.check();
					}
				}
			}
//*/
			// reset memory
			rl_.memory_.reset();		
			for (int h = 0; h < rl_.num_input_histories_; h++) {
				rl_.recordHistory(simul_->getStateBuffer(), 0.0f, 2, VectorND<float>(simul_->getNumActions())); // choice 2 is stay
			}
		}

		if (!is_training_){
			simul_->render();
		}
	}

	glfwTerminate();
}

// end of file
