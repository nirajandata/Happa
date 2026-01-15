//
// Created by nirakira on 1/16/26.
//

#pragma once

#include <vector>

template<typename T>
struct Matrix {
    size_t rows{};
    size_t cols{};
    std::vector<T> data;


    constexpr Matrix(size_t row_, size_t col_) : rows(row_), cols(col_) {
        data.resize(rows * cols, 0);
    }

    template<class Self>
    constexpr auto& operator()(this Self&& self, size_t row_, size_t col_) {
        return self.data[row_ * self.cols + col_];
    }
};

