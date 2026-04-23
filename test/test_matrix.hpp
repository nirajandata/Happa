#pragma once

#include <cassert>
#include <chrono>
#include <iostream>
#include <print>
#include <stdexcept>
#include <random>
#include <vector>
#include <numeric>
#include "Happa/Matrix.hpp"

namespace happa::test {

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

        for (size_t i = 0; i < 4; ++i)
            assert(is_near(I_check.data[i], I.data[i]));

        std::println("[PASS] inverse() correct.");

        Matrix<double> D = {{3, 8}, {4, 6}};
        assert(is_near(D.determinant(), -14.0));
        std::println("[PASS] determinant() correct.");
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

        std::println("All tests completed.");
    }
}