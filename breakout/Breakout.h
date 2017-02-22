#pragma once

#include "Common.h"
#include "VectorND.h"
#include "ConsoleGL.h"

const F DEFAULT_WALL_WIDTH = 0.05f;
const F DEFAULT_PADDLE_WIDTH_RATIO = 0.3f;

const F PADDLE_MOVE_RATE = 0.33f;
const F START_POSITION = 0.5f;
const F DEFAULT_BALL_SPEED = 0.5f;
const F UPDATE_BALL_SPEED = 0.04f;

class Breakout : public ConsoleGL {
public:
	typedef enum {
		STAY = 0,
		RIGHT = 1,
		LEFT = 2
	} DirType;

public:
    Breakout();
    ~Breakout();

    void toggleTrainigMode() { trainig_mode_ = !trainig_mode_;}
    bool isTraining() {return trainig_mode_;}
    int getNumActions(){return 3;} //left, right, stay

    void restart();
    const VectorND<float>& getStateBuffer();
    void printStateBuffer();

    void movePaddle(DirType dir, F dx = PADDLE_MOVE_RATE);
    F updateSatus(float dt = UPDATE_BALL_SPEED);
    
    void makeScene();
protected:
    void normalize(F& x, F& y);

private:
    float paddle_x_;    
    float ball_x_, ball_y_;
    float ball_vel_x_, ball_vel_y_;

    const float wall_thickness_;
    const float paddle_width_;

    bool trainig_mode_;

    VectorND<float> state_buffer_;
};


// end of file
