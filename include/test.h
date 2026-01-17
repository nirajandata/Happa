// test.h
#pragma once

#include <cassert>
#include <print>
#include <stdexcept>
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
        auto M3 = M1 * M2;
        if (M3(0, 0) != 2 || M3(1, 0) != 6 || M3(1, 1) != 4) return false;

        Matrix<int> fib = {{1, 1}, {1, 0}};
        auto fib2 = fib ^ 2;
        if (fib2(0, 0) != 2 || fib2(1, 1) != 1) return false;

        auto fib0 = fib ^ 0;
        if (fib0(0, 0) != 1 || fib0(0, 1) != 0) return false;

        return true;
    }

    static_assert(run_constexpr_tests(), "Matrix logic failed static_assert!");


    inline void run_all_tests() {
        std::println("--- Running Happa Matrix Test Suite ---");

        Matrix<int> S1 = {{1, 5, 9, 2}, {3, 7, 4, 8}, {6, 2, 0, 5}, {4, 1, 9, 3}};
        Matrix<int> S2 = {{5, 2, 1, 8}, {7, 3, 0, 4}, {1, 6, 2, 9}, {0, 4, 5, 3}};

        auto standard_res = S1.multiply_standard(S2);
        auto strassen_res = S1.strassen(S2);

        assert(standard_res == strassen_res);
        std::println("[PASS] Strassen results match Standard results.");

        //Fibonacci with matrix exponentiation
        Matrix<long long> f = {{1, 1}, {1, 0}};
        auto f10 = f ^ 10;
        assert(f10(0, 1) == 55);
        std::println("[PASS] Binary Exponentiation (Fibonacci 10) = {}", f10(0, 1));

        // Sorting Tests
        // 5 9 1
        // 8 2 4
        Matrix<int> sort_me = {{5, 9, 1}, {8, 2, 4}};

        sort_me.sort_rows();
        // Row 0: 1 5 9
        // Row 1: 2 4 8
        assert(sort_me(0, 0) == 1 && sort_me(0, 2) == 9);
        assert(sort_me(1, 0) == 2 && sort_me(1, 2) == 8);

        sort_me.sort_cols();
        // Col 0: [1, 2] -> 1, 2
        // Col 1: [5, 4] -> 4, 5
        // Col 2: [9, 8] -> 8, 9
        // Matrix is now:
        // 1 4 8
        // 2 5 9
        assert(sort_me(0, 1) == 4);
        assert(sort_me(1, 1) == 5);
        std::println("[PASS] Row and Column Range-based sorting.");

        //Exception Handling
        try {
            Matrix<int> err1(2, 3);
            Matrix<int> err2(4, 5);
            auto fail = err1 * err2;
            (void)fail;
            assert(false);
        } catch (const std::invalid_argument& e) {
            std::println("[PASS] Caught expected dimension mismatch: {}", e.what());
        }

        std::println("---------------------------------------");
        std::println("All tests passed successfully.");
    }
}