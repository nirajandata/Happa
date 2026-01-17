cmake_minimum_required(VERSION 4.2)

project(Happa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Happa
        src/main.cpp
)

target_include_directories(Happa
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
