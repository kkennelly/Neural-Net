cmake_minimum_required(VERSION 3.17)
project(main)

set(CMAKE_CXX_STANDARD 14)

include_directories(../inc)
file(GLOB SOURCES ./*.cpp)

add_executable(main ../main.cpp ${SOURCES})