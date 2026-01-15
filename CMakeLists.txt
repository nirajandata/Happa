cmake_minimum_required(VERSION 4.2)
project(Happa)

set(CMAKE_CXX_STANDARD 26)
target_include_directories(happa
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(Happa main.cpp)
