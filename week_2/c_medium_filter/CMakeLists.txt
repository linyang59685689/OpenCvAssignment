cmake_minimum_required(VERSION 3.15)
project(clion01)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(clion01 medium_filter.cpp)
target_link_libraries( clion01 ${OpenCV_LIBS} )