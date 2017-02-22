cmake_minimum_required( VERSION 3.0 )
project( CONVOLUTIONAL_NEURAL_NETWORK )

ADD_SUBDIRECTORY( "../core" "${CMAKE_CURRENT_BINARY_DIR}/lib_build" )
INCLUDE_DIRECTORIES( ../core )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

MESSAGE( STATUS "Compiling Convolutional Neural Network" )
set ( SOURCE_FILES ConvolutionalNeuralNetwork.cpp )
add_executable( cnn ${SOURCE_FILES} )
target_link_libraries( cnn DLCore )
