cmake_minimum_required(VERSION 3.3)
project(HelloWorld)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package( OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(HelloWorld ${SOURCE_FILES})
target_link_libraries( HelloWorld ${OpenCV_LIBS})
