#pragma once

#include <cassert>
#include <chrono>
#include <iostream>
#include <print>
#include <stdexcept>
#include <random>
#include <vector>
#include <numeric>
#include "Matrix.h"

namespace happa::test {
    inline void flush_cache() {
        static constexpr size_t cache_size = 64 * 1024 * 1024; // 64 MB
        static std::vector<char> dummy(cache_size, 1);
        volatile char sink = std::accumulate(dummy.begin(), dummy.end(), char(0));
        (void) sink;
    }

    consteval bool run_constexpr_tests() {
        Matrix<int> A = {{1, 2}, {3, 4}};
        auto I = Matrix<int>::identity(2);
        if (I(0, 0) != 1 || I(0, 1) != 0 || I(1, 1) != 1) return false;
        Matrix<int> fib = {{1, 1}, {1, 0}};
        auto fib2 = fib ^ 2;
        if (fib2(0, 0) != 2 || fib2(1, 1) != 1) return false;
        return true;
    }

    static_assert(run_constexpr_tests());

    inline void test_division() {
        std::println("\n--- Testing Matrix Division (LU Solve/Inverse) ---");

        auto is_near = [](double a, double b, double epsilon = 1e-9) {
            return std::abs(a - b) < epsilon;
        };

        Matrix<double> A = {{2.0, 1.0}, {1.0, 3.0}};
        Matrix<double> B = {{5.0}, {10.0}};

        auto X = A.solve(B);
        assert(is_near(X(0, 0), 1.0));
        assert(is_near(X(1, 0), 3.0));
        std::println("[PASS] solve(AX=B) correct.");

        Matrix<double> M1 = {{1, 2}, {3, 4}};
        Matrix<double> M2 = {{5, 6}, {7, 8}};
        Matrix<double> C = M1 * M2;
        Matrix<double> Result = C / M2;

        for (size_t i = 0; i < M1.data.size(); ++i) {
            assert(is_near(Result.data[i], M1.data[i]));
        }
        std::println("[PASS] operator/ (Matrix Right-Division) correct.");

        auto invA = A.inverse();
        auto I_check = A * invA;
        auto I = Matrix<double>::identity(2);
        for (size_t i = 0; i < 4; ++i) assert(is_near(I_check.data[i], I.data[i]));
        std::println("[PASS] inverse() correct.");

        Matrix<double> D = {{3, 8}, {4, 6}};
        assert(is_near(D.determinant(), -14.0));
        std::println("[PASS] determinant() correct.");

    }

    inline void run_comprehensive_benchmark() {
        std::println("\n{:-^95}", " HAPPA MATRIX BENCHMARK ");
        std::println("{:>8} | {:>15} | {:>18} | {:>22}", "Size", "New Tiled (s)", "Auto", "Strassen (s)");
        std::println("{:-^95}", "");

        std::vector<size_t> sizes = {64, 128, 192, 256, 384, 512, 640, 768, 896, 1024};
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (size_t N: sizes) {
            Matrix<double> A(N, N), B(N, N);
            for (auto &x: A.data) x = dist(gen);
            for (auto &x: B.data) x = dist(gen);

            // Standard Multiplication
            flush_cache();
            auto s1 = std::chrono::steady_clock::now();
            auto r1 = A.multiply_standard(B);
            auto e1 = std::chrono::steady_clock::now();
            double t1 = std::chrono::duration<double>(e1 - s1).count();

            //(Tiled)
            flush_cache();
            auto s2 = std::chrono::steady_clock::now();
            auto r2 = A * B;
            auto e2 = std::chrono::steady_clock::now();
            double t2 = std::chrono::duration<double>(e2 - s2).count();

            //Strassen
            flush_cache();
            auto s3 = std::chrono::steady_clock::now();
            auto r3 = A.strassen(B);
            auto e3 = std::chrono::steady_clock::now();
            double t3 = std::chrono::duration<double>(e3 - s3).count();

            std::println("{:>8} | {:>15.6f} | {:>18.6f} | {:>22.6f}", N, t1, t2, t3);
        }
        std::println("{:-^95}\n", "");
    }

    inline void run_all_tests() {
        std::println("--- Running Happa Matrix Test Suite ---");

        Matrix<int> S1 = {{1, 2}, {3, 4}};
        Matrix<int> S2 = {{5, 6}, {7, 8}};
        assert(S1.multiply_standard(S2) == S1.strassen(S2));
        assert(S1.multiply_standard(S2) == S1.tiled_multiply(S2));

        std::println("[PASS] All multiplication variants yield identical results.");

        Matrix<long long> f = {{1, 1}, {1, 0}};
        assert((f ^ 10)(0, 1) == 55);
        std::println("[PASS] Binary Exponentiation logic correct.");

        test_division();
        // run_comprehensive_benchmark();

        std::println("All tests and benchmarks completed.");
    }
}
