
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

ADD_LIBRARY(DLCore Matrix.cpp ActivationFunctions.cpp LayerBase.cpp NeuralNetwork.cpp Convolution2D.cpp ConvConnection2D.cpp FullConnection.cpp LayerBase.cpp ReinforcementLearning.cpp)

