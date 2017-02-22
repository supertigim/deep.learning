cmake_minimum_required( VERSION 3.0 )
project( FULLY_CONNECTED_NEURAL_NETWORK )

ADD_SUBDIRECTORY( "../core" "${CMAKE_CURRENT_BINARY_DIR}/lib_build" )
INCLUDE_DIRECTORIES( ../core )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

MESSAGE( STATUS "Compiling Fully Connected Neural Network" )
set ( SOURCE_FILES FullyConnectedNeuralNetwork.cpp )
add_executable( fcnn ${SOURCE_FILES} )
target_link_libraries( fcnn DLCore)
