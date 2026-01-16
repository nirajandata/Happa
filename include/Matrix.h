//
// Created by nirakira on 1/16/26.
//
#pragma once
#include <vector>
#include <ranges>
#include <format>
#include <algorithm>

namespace happa {
    template<typename T>
    concept MatrixElement = std::formattable<T, char> && (std::is_arithmetic_v<T> || requires(T a, T b)
    {
        { a + b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { T{0} };
    });

    template<MatrixElement T>
    struct Matrix {
        using value_type = T;
        size_t rows{};
        size_t cols{};
        std::vector<T> data;

        constexpr Matrix(size_t row_, size_t col_) : rows(row_), cols(col_) {
            data.resize(rows * cols, 0);
        }

        constexpr Matrix(std::initializer_list<std::initializer_list<T> > list) {
            rows = list.size();
            cols = list.begin()->size();
            data.reserve(rows * cols);
            for (const auto &row_list: list) {
                data.insert(data.end(), row_list.begin(), row_list.end());
            }
        }

        template<class Self>
        [[nodiscard]] constexpr auto &operator()(this Self &&self, size_t row_, size_t col_) {
            return self.data[row_ * self.cols + col_];
        }

        [[nodiscard]] constexpr Matrix<T> transpose() const noexcept {
            Matrix<T> result(cols, rows);
            auto indices = std::views::cartesian_product(
                std::views::iota(0u, rows),
                std::views::iota(0u, cols)
            );
            for (auto [i, j]: indices) {
                result(j, i) = (*this)(i, j);
            }
            return result;
        }

        [[nodiscard]] constexpr Matrix<T> operator^(size_t exp);

        template<typename Self>
        constexpr auto begin(this Self &&self) { return self.data.begin(); }

        template<typename Self>
        constexpr auto end(this Self &&self) { return self.data.end(); }

        constexpr void sort_rows() {
            for (size_t i = 0; i < rows; ++i) {
                auto row_start = data.begin() + i * cols;
                auto row_end = row_start + cols;
                std::ranges::sort(row_start, row_end);
            }
        }

        constexpr void sort_cols() {
            for (size_t j = 0; j < cols; ++j) {
                std::vector<T> col_data;
                col_data.reserve(rows);
                for (size_t i = 0; i < rows; ++i) {
                    col_data.push_back((*this)(i, j));
                }
                std::ranges::sort(col_data);
                for (size_t i = 0; i < rows; ++i) {
                    (*this)(i, j) = col_data[i];
                }
            }
        }
    };

    template<typename T>
    Matrix(std::initializer_list<std::initializer_list<T> >) -> Matrix<T>;
}

template<happa::MatrixElement T>
struct std::formatter<happa::Matrix<T> > {
    constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

    auto format(const happa::Matrix<T> &mat, std::format_context &ctx) const {
        auto out = std::format_to(ctx.out(), "Matrix({}x{}):\n", mat.rows, mat.cols);
        for (auto row: mat.data | std::views::chunk(mat.cols)) {
            out = std::format_to(out, "[");
            bool first = true;
            for (auto elem: row) {
                if (!first) out = std::format_to(out, ", ");
                out = std::format_to(out, "{}", elem);
                first = false;
            }
            out = std::format_to(out, "]\n");
        }
        return out;
    }
};
