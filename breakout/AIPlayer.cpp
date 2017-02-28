#include "AIPlayer.h"
#include <thread>
#include <chrono>

AIPlayer::AIPlayer(Breakout* game)
	: game_(game)
	, is_training_(true)
{
	if(is_training_ != game_->isTraining()) {
		game_->toggleTrainigMode();
	}
}

void AIPlayer::keyProcess(char ch){
	
	switch(ch){
    case Player::SPACEBAR_KEY:
    case Player::ENTER_KEY:
    	is_training_ = !is_training_;
    	game_->toggleTrainigMode();
    	break;
	}
}

void AIPlayer::initialize(){
	rl_.num_input_histories_ = 4;
	
	//std::cout << "Agent::init() " << simul_->getNumStateVariables() << endl;
	rl_.num_state_variables_ = game_->screenSize();
	rl_.num_game_actions_ = game_->getNumActions();

	rl_.initializeConv2D(game_->height(), game_->width());

	game_->makeScene();
	game_->flipBuffer();


	for (int h = 0; h < rl_.num_input_histories_; h++) {
		rl_.recordHistory(game_->getStateBuffer(), 0.0f, 0, VectorND<float>(game_->getNumActions())); // choice 0 is stay
	}
}

const int AIPlayer::getSelectedDir(){
	 const int selected_dir = is_training_ == true ? 
	 			rl_.nn_.getOutputIXEpsilonGreedy(0.2f) : rl_.nn_.getOutputIXEpsilonGreedy(0.0f);	

	 const F epsilonMinimumValue = 0.001f;
	 epsilon_ = epsilon_ < epsilonMinimumValue? epsilonMinimumValue: epsilon_*0.0999;

	 return selected_dir;
}

void AIPlayer::run(){

	VectorND<F> output_vector_temp;
	int reward_sum = 0;
	int reward_max = 0;
	std::cout << "Game Start!! \n";

	while(true) {

		if (is_training_ == true) {
            
#if defined(__APPLE__)           
            system("clear");
#else 
            system("cls");
#endif 
            game_->resetBuffers();

			std::cout << "Training... " << "\n";
        }else{
        	std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        
		rl_.forward();
		rl_.nn_.copyOutputVectorTo(false, output_vector_temp);

	
		const int selected_dir = getSelectedDir();	// epsilon-greedy
		
		std::cout << "selected_dir - " << selected_dir << endl;

		switch (selected_dir) {
        case 2:
        	game_->movePaddle(Breakout::LEFT);
            break;
        case 1:
            game_->movePaddle(Breakout::RIGHT);
            break;
        case 0:
            // do nothing
            break;
        default:
            std::cout << "Wrong direction " << endl;
        }
		
		game_->makeScene();

        if(is_training_ == false){
        	game_->render(); // need to render for conv	
        }
        game_->flipBuffer();

        float reward = game_->updateSatus();

        bool isTerminated = false;
        if( reward < 0){
        	isTerminated = true;
        	reward = 0.0f;
        }

        // record state and reward
		rl_.recordHistory(game_->getStateBuffer(), reward, selected_dir, output_vector_temp);
		reward_sum += reward;

		if(is_training_){
			if (reward_max < reward_sum) {
				reward_max = reward_sum;
				rl_.trainRewardMemory(false);
			}
			else
				rl_.trainReward(false);
				//rl_.trainBatch(tr_num);
				//rl_.trainReward();
		}			

		if (isTerminated) {		
			reward_sum = 0.0f;
			rl_.memory_.reset();

			for (int h = 0; h < rl_.num_input_histories_; h++) {
				rl_.recordHistory(game_->getStateBuffer(), 0.0f, 0, VectorND<float>(game_->getNumActions())); // choice 0 is stay
			}	
		}
    }
}


// end of file
