#include <benchmark/benchmark.h>
#include <random>
#include "Happa/Matrix.hpp"

template <class Func>
void RunMatrixMultiplication(benchmark::State& state, Func func) {
    size_t N = state.range(0);
    happa::Matrix<double> A(N, N), B(N, N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& x : A.data) x = dist(gen);
    for (auto& x : B.data) x = dist(gen);

    for (auto _ : state) {
        auto C = func(A, B);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}

static void BM_MultiplyStandard(benchmark::State& state) {
    RunMatrixMultiplication(state, [](const auto& a, const auto& b) {
        return a.multiply_standard(b);
    });
}

static void BM_MultiplyTiled(benchmark::State& state) {
    RunMatrixMultiplication(state, [](const auto& a, const auto& b) {
        return a.tiled_multiply(b);
    });
}

static void BM_MultiplyStrassen(benchmark::State& state) {
    RunMatrixMultiplication(state, [](const auto& a, const auto& b) {
        return a.strassen(b);
    });
}

BENCHMARK(BM_MultiplyStandard)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_MultiplyTiled)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_MultiplyStrassen)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();