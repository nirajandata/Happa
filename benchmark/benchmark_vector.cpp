//
// Created by nirakira on 4/24/26.
//

#include <benchmark/benchmark.h>
#include "Happa/Vector.hpp"
#include <random>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<double> dist(-100, 100);

static happa::Vector<double> random_vector(size_t n) {
    happa::Vector<double> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = dist(gen);
    return v;
}

static void BM_VectorDot(benchmark::State& state) {
    size_t n = state.range(0);
    auto a = random_vector(n);
    auto b = random_vector(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a.dot(b));
    }
}
BENCHMARK(BM_VectorDot)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_VectorNormalize(benchmark::State& state) {
    size_t n = state.range(0);
    auto v = random_vector(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(v.normalized());
    }
}
BENCHMARK(BM_VectorNormalize)->Arg(100)->Arg(1000);

BENCHMARK_MAIN();