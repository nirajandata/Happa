//
// Created by nirakira on 1/16/26.
//
#pragma once
#include <vector>
#include <ranges>
#include <format>
#include <string_view>
#include <print>


template<typename T>
concept MatrixElement = std::is_arithmetic_v<T> && std::formattable<T, char>;

namespace happa {
    template<MatrixElement T>
    struct Matrix {
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

        [[nodiscard]] constexpr Matrix<T> transpose() const {
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
    };
}

template<MatrixElement T>
struct std::formatter<happa::Matrix<T> > {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }

    auto format(const happa::Matrix<T> &mat, std::format_context &ctx) const {
        auto out = ctx.out();

        out = std::format_to(out, "Matrix({}x{}):\n", mat.rows, mat.cols);

        auto rows = mat.data | std::views::chunk(mat.cols);

        for (auto row: rows) {
            out = std::format_to(out, "{}\n", row);
        }
        return out;
    }
};
