#pragma once

#include <cassert>
#include <chrono>
#include <print>
#include <random>
#include "Matrix.h"

namespace happa::test {

    consteval bool run_constexpr_tests() {
        Matrix<int> A = {{1, 2}, {3, 4}};
        auto I = Matrix<int>::identity(2);
        if (I(0, 0) != 1 || I(0, 1) != 0 || I(1, 1) != 1) return false;

        auto B = A + A;
        if (B(0, 0) != 2 || B(1, 1) != 8) return false;

        auto C = A * 2;
        if (C(1, 0) != 6) return false;

        if (A.trace() != 5) return false;
        auto AT = A.transpose();
        if (AT(0, 1) != 3 || AT(1, 0) != 2) return false;

        Matrix<int> M1 = {{1, 2}, {3, 4}};
        Matrix<int> M2 = {{2, 0}, {0, 1}};
        auto M3 = M1.multiply_standard(M2);
        if (M3(0, 0) != 2 || M3(1, 0) != 6 || M3(1, 1) != 4) return false;

        Matrix<int> fib = {{1, 1}, {1, 0}};
        auto fib2 = fib ^ 2;
        if (fib2(0, 0) != 2 || fib2(1, 1) != 1) return false;

        return true;
    }

    static_assert(run_constexpr_tests());

    inline void run_benchmark(size_t N) {
        std::println("\n--- Benchmarking Matrix Multiplication (Size: {}x{}) ---", N, N);

        Matrix<double> A(N, N), B(N, N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(1.0, 10.0);

        for (auto& x : A.data) x = dist(gen);
        for (auto& x : B.data) x = dist(gen);

        auto start_str = std::chrono::steady_clock::now();
        auto res_str = A.strassen(B);
        auto end_str = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff_str = end_str - start_str;
        std::println("Strassen (O(N^2.81)) Time: {:.4f}s", diff_str.count());

        auto start_std = std::chrono::steady_clock::now();
        auto res_std = A.multiply_standard(B);
        auto end_std = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff_std = end_std - start_std;
        std::println("Standard (O(N^3)) Time:  {:.4f}s", diff_std.count());



        double max_diff = 0;
        for (size_t i = 0; i < res_std.data.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(res_std.data[i] - res_str.data[i]));
        }
        std::println("Numerical Consistency (Max Diff): {:.2e}", max_diff);
    }

    inline void run_all_tests() {
        std::println("--- Running Happa Matrix Test Suite ---");

        Matrix<int> S1 = {{1, 5, 9, 2}, {3, 7, 4, 8}, {6, 2, 0, 5}, {4, 1, 9, 3}};
        Matrix<int> S2 = {{5, 2, 1, 8}, {7, 3, 0, 4}, {1, 6, 2, 9}, {0, 4, 5, 3}};

        auto standard_res = S1.multiply_standard(S2);
        auto strassen_res = S1.strassen(S2);
        assert(standard_res == strassen_res);
        std::println("[PASS] Strassen Matches Standard.");

        Matrix<long long> f = {{1, 1}, {1, 0}};
        auto f10 = f ^ 10;
        assert(f10(0, 1) == 55);
        std::println("[PASS] Fibonacci Expo (10) = 55");

        Matrix<int> sort_me = {{5, 9, 1}, {8, 2, 4}};
        sort_me.sort_rows();
        sort_me.sort_cols();
        assert(sort_me(0, 1) == 4 && sort_me(1, 1) == 5);
        std::println("[PASS] Range-based Sorting.");

       // run_benchmark(512);

        std::println("---------------------------------------");
        std::println("All tests passed.");
    }
}