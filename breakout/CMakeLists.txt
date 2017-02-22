cmake_minimum_required( VERSION 3.0 )
project( CNN_DQN_BASED_GAME )

ADD_SUBDIRECTORY( "../core" "${CMAKE_CURRENT_BINARY_DIR}/lib_build" )

FIND_PACKAGE( Boost COMPONENTS system thread REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} ../core )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )
set( Boost_USE_STATIC_LIBS ON)
set( Boost_USE_MULTITHREADED ON)

IF ( Boost_FOUND )
	MESSAGE( STATUS "Compiling CNN based Game - Breakout" )
	set ( SOURCE_FILES ConsoleGL.cpp Breakout.cpp ConvDQNGameMain.cpp Player.cpp AIPlayer.cpp)
	add_executable( breakout ${SOURCE_FILES} )
	#target_link_libraries( breakout ${Boost_LIBS} pthread NNLib)
	target_link_libraries( breakout ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} pthread DLCore)
	
ELSE ()
	MESSAGE( FATAL_ERROR "Boost is NOT found in your system" )
ENDIF ()
