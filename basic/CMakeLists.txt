cmake_minimum_required( VERSION 3.0 )
project( DEEP_LEARNING_BASIC )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

MESSAGE( STATUS "Compiling Deep Learning Basic" )
set ( SOURCE_FILES DeepLearningBasic.cpp )
add_executable( deaplearning ${SOURCE_FILES} )
target_link_libraries( deaplearning )
