//
// Created by nirakira on 1/16/26.
//
#pragma once

#include <iostream>
#include "Matrix.h"

inline void test_matrix() {

    Matrix<int> mat(3,3);


    std::cout << "Test Case 1: Matrix Initialization\n";
    std::cout << "Rows: " << mat.rows << ", Cols: " << mat.cols << "\n";
    std::cout << "Element (0, 0): " << mat(0, 0) << "\n";
    std::cout << "Element (2, 2): " << mat(2, 2) << "\n\n";

    // Test Case 2: Set/Get Elements
    mat(0, 0) = 5;
    mat(0, 2) = 10;
    mat(1, 1) = 7;

    std::cout << "Test Case 2: Set/Get Elements\n";
    std::cout << "Element (0, 0): " << mat(0, 0) << "\n";
    std::cout << "Element (0, 2): " << mat(0, 2) << "\n";
    std::cout << "Element (1, 1): " << mat(1, 1) << "\n\n";

    // Test Case 3: Modify Elements
    mat(1, 0) = 3;
    std::cout << "Test Case 3: Modify Elements\n";
    std::cout << "Element (1, 0): " << mat(1, 0) << "\n\n";

    // Test Case 4: Non-Zero Initialization
    Matrix<int> mat2(2,2);
    mat2.data = {1, 2, 3, 4};

    std::cout << "Test Case 4: Non-Zero Initialization\n";
    std::cout << "Element (0, 0): " << mat2(0, 0) << "\n";
    std::cout << "Element (0, 1): " << mat2(0, 1) << "\n";
    std::cout << "Element (1, 0): " << mat2(1, 0) << "\n";
    std::cout << "Element (1, 1): " << mat2(1, 1) << "\n";


}

