cmake_minimum_required(VERSION 3.20)

project(Happa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

add_library(happa INTERFACE)

target_include_directories(happa INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(Happa_Test
        test/test.cpp
        test/test_vector.hpp
)

target_include_directories(Happa_Test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(Happa_Benchmark
        benchmark/benchmark_matrix.cpp
)

target_include_directories(Happa_Benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(Happa_Benchmark PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
)

add_executable(happa_vector_bench
        benchmark/benchmark_vector.cpp
)

target_link_libraries(happa_vector_bench PRIVATE
        happa
        benchmark::benchmark
        benchmark::benchmark_main
)