//
// Created by nirakira on 1/16/26.
//
#pragma once

#include <iostream>
#include <print>
#include "Matrix.h"

inline void test_matrix() {
    using namespace happa;
    Matrix<int> mat(3, 3);


    std::cout << "Test Case 1: Matrix Initialization\n";
    std::cout << "Rows: " << mat.rows << ", Cols: " << mat.cols << "\n";
    std::cout << "Element (0, 0): " << mat(0, 0) << "\n";
    std::cout << "Element (2, 2): " << mat(2, 2) << "\n\n";

    mat(0, 0) = 5;
    mat(0, 2) = 10;
    mat(1, 1) = 7;

    std::cout << "Test Case 2: Set/Get Elements\n";
    std::cout << "Element (0, 0): " << mat(0, 0) << "\n";
    std::cout << "Element (0, 2): " << mat(0, 2) << "\n";
    std::cout << "Element (1, 1): " << mat(1, 1) << "\n\n";

    mat(1, 0) = 3;
    std::cout << "Test Case 3: Modify Elements\n";
    std::cout << "Element (1, 0): " << mat(1, 0) << "\n\n";

    Matrix<double> mat2(2, 2);

    mat2.data = {1.2, 3.4, 5.5, 6.6};

    std::cout << "Test Case 4: Non-Zero Initialization\n";
    std::cout << "Element (0, 0): " << mat2(0, 0) << "\n";
    std::cout << "Element (0, 1): " << mat2(0, 1) << "\n";
    std::cout << "Element (1, 0): " << mat2(1, 0) << "\n";
    std::cout << "Element (1, 1): " << mat2(1, 1) << "\n";


    std::print("{}", mat2);

    happa::Matrix<float> m = {
        {1.1, 2.2},
        {3.3, 4.4},
        {5.5, 6.6}
    };

    std::println("{}", m);
    std::println("Transposed:\n{}", m.transpose());

    happa::Matrix ma = {{5, 9}, {4, 2}};
    std::ranges::sort(ma);

    std::cout << "After sort:\n";
    std::println("{}", ma);

    std::cout << "\nFlat data: [";
    for (size_t i = 0; i < ma.data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ma.data[i];
    }
    std::cout << "]\n";
}

