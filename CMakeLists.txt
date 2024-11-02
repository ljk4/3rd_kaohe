cmake_minimum_required(VERSION 3.17.0)
project(3rd VERSION 0.1.0 LANGUAGES C CXX)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(3rd 3rd.cpp)

target_link_libraries( 3rd ${OpenCV_LIBS} )