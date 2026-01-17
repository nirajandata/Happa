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
    concept MatrixElement =
            std::semiregular<T> &&      std::formattable<T, char> &&           requires(T a, T b)
            {
                { a + b } -> std::convertible_to<T>;
                { a - b } -> std::convertible_to<T>;
                { a * b } -> std::convertible_to<T>;
                { a += b } -> std::same_as<T &>;
                { a -= b } -> std::same_as<T &>;
                { a *= b } -> std::same_as<T &>;
                { T{0} };
                { T{1} };
            };


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

        static constexpr Matrix identity(size_t n) {
            Matrix res(n, n);
            std::ranges::fill(res.data | std::views::stride(n + 1), T{1});
            return res;
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

        [[nodiscard]] friend constexpr Matrix operator+(Matrix lhs, const Matrix &rhs) { return lhs += rhs; }
        [[nodiscard]] friend constexpr Matrix operator-(Matrix lhs, const Matrix &rhs) { return lhs -= rhs; }

        constexpr Matrix &operator*=(const T &scalar) {
            for (auto &x: data) x *= scalar;
            return *this;
        }

        [[nodiscard]] friend constexpr Matrix operator*(Matrix mat, const T &scalar) { return mat *= scalar; }
        [[nodiscard]] friend constexpr Matrix operator*(const T &scalar, Matrix mat) { return mat *= scalar; }

        [[nodiscard]] constexpr Matrix multiply_standard(const Matrix &other) const {
            if (cols != other.rows) throw std::invalid_argument("Dimension mismatch");
            Matrix result(rows, other.cols);
            for (size_t i: std::views::iota(0u, rows)) {
                for (size_t k: std::views::iota(0u, cols)) {
                    T temp = (*this)(i, k);
                    if (temp == T{0}) continue;
                    for (size_t j: std::views::iota(0u, other.cols)) {
                        result(i, j) += temp * other(k, j);
                    }
                }
            }
            return result;
        }

        [[nodiscard]] constexpr Matrix operator*(const Matrix &other) const {
            if (rows > 128 && cols > 128) {
                return strassen(other);
            }
            return multiply_standard(other);
        }

        [[nodiscard]] constexpr Matrix operator^(size_t exp) const {
            if (rows != cols) throw std::invalid_argument("Matrix must be square for exponentiation");
            Matrix res = Matrix::identity(rows);
            Matrix base = *this;
            while (exp > 0) {
                if (exp % 2 == 1) res = res * base;
                base = base * base;
                exp /= 2;
            }
            return res;
        }

        constexpr Matrix &operator+=(const Matrix &other) {
            std::ranges::transform(data, other.data, data.begin(), std::plus<T>());
            return *this;
        }

        constexpr Matrix &operator-=(const Matrix &other) {
            std::ranges::transform(data, other.data, data.begin(), std::minus<T>());
            return *this;
        }

        [[nodiscard]] constexpr T trace() const {
            auto diagonal = data | std::views::stride(cols + 1);
            return std::ranges::fold_left(diagonal, T{0}, std::plus<T>());
        }

    private:
        struct SubView {
            const Matrix &ref;
            size_t r_start, c_start, size;

            const T &operator()(size_t i, size_t j) const {
                return ref.data[(r_start + i) * ref.cols + (c_start + j)];
            }
        };

        static Matrix strassen_rec(SubView A, SubView B) {
            const size_t n = A.size;

            if (n <= 64) {
                Matrix res(n, n);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t k = 0; k < n; ++k) {
                        T temp = A(i, k);
                        if (temp == T{0}) continue;
                        for (size_t j = 0; j < n; ++j) {
                            res(i, j) += temp * B(k, j);
                        }
                    }
                }
                return res;
            }

            const size_t mid = n / 2;

            auto a11 = SubView{A.ref, A.r_start, A.c_start, mid};
            auto a12 = SubView{A.ref, A.r_start, A.c_start + mid, mid};
            auto a21 = SubView{A.ref, A.r_start + mid, A.c_start, mid};
            auto a22 = SubView{A.ref, A.r_start + mid, A.c_start + mid, mid};

            auto b11 = SubView{B.ref, B.r_start, B.c_start, mid};
            auto b12 = SubView{B.ref, B.r_start, B.c_start + mid, mid};
            auto b21 = SubView{B.ref, B.r_start + mid, B.c_start, mid};
            auto b22 = SubView{B.ref, B.r_start + mid, B.c_start + mid, mid};

            auto add_v = [&](SubView v1, SubView v2) {
                Matrix res(mid, mid);
                for (size_t i = 0; i < mid; ++i)
                    for (size_t j = 0; j < mid; ++j) res(i, j) = v1(i, j) + v2(i, j);
                return res;
            };
            auto sub_v = [&](SubView v1, SubView v2) {
                Matrix res(mid, mid);
                for (size_t i = 0; i < mid; ++i)
                    for (size_t j = 0; j < mid; ++j) res(i, j) = v1(i, j) - v2(i, j);
                return res;
            };

            auto m1 = strassen_rec(SubView{add_v(a11, a22), 0, 0, mid}, SubView{add_v(b11, b22), 0, 0, mid});
            auto m2 = strassen_rec(SubView{add_v(a21, a22), 0, 0, mid}, SubView{b11, 0, 0, mid});
            auto m3 = strassen_rec(SubView{a11, 0, 0, mid}, SubView{sub_v(b12, b22), 0, 0, mid});
            auto m4 = strassen_rec(SubView{a22, 0, 0, mid}, SubView{sub_v(b21, b11), 0, 0, mid});
            auto m5 = strassen_rec(SubView{add_v(a11, a12), 0, 0, mid}, SubView{b22, 0, 0, mid});
            auto m6 = strassen_rec(SubView{sub_v(a21, a11), 0, 0, mid}, SubView{add_v(b11, b12), 0, 0, mid});
            auto m7 = strassen_rec(SubView{sub_v(a12, a22), 0, 0, mid}, SubView{add_v(b21, b22), 0, 0, mid});

            Matrix C(n, n);
            for (size_t i = 0; i < mid; ++i) {
                for (size_t j = 0; j < mid; ++j) {
                    C(i, j) = m1(i, j) + m4(i, j) - m5(i, j) + m7(i, j);
                    C(i, j + mid) = m3(i, j) + m5(i, j);
                    C(i + mid, j) = m2(i, j) + m4(i, j);
                    C(i + mid, j + mid) = m1(i, j) - m2(i, j) + m3(i, j) + m6(i, j);
                }
            }
            return C;
        }

    public:
        [[nodiscard]] constexpr Matrix strassen(const Matrix &other) const {
            if (cols != other.rows) throw std::invalid_argument("Dimension mismatch");

            size_t p = std::bit_ceil(std::max({rows, cols, other.cols}));

            Matrix A_pad(p, p), B_pad(p, p);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j) A_pad(i, j) = (*this)(i, j);
            for (size_t i = 0; i < other.rows; ++i)
                for (size_t j = 0; j < other.cols; ++j) B_pad(i, j) = other(i, j);

            Matrix C_pad = strassen_rec(SubView{A_pad, 0, 0, p}, SubView{B_pad, 0, 0, p});

            Matrix result(rows, other.cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < other.cols; ++j) result(i, j) = C_pad(i, j);

            return result;
        }

        [[nodiscard]] bool operator==(const Matrix<T> &other) const {
            return rows == other.rows && cols == other.cols;
        }

        template<typename Self>
        constexpr auto begin(this Self &&self) { return self.data.begin(); }

        template<typename Self>
        constexpr auto end(this Self &&self) { return self.data.end(); }


        constexpr void sort_rows() {
            for (auto row: data | std::views::chunk(cols)) {
                std::ranges::sort(row);
            }
        }

        constexpr void sort_cols() {
            for (size_t j : std::views::iota(0u, cols)) {
                auto col_view = std::views::iota(0u, rows)
                              | std::views::transform([&](size_t i) { return (*this)(i, j); });

                auto col_data = std::ranges::to<std::vector<T>>(col_view);
                std::ranges::sort(col_data);

                for (size_t i : std::views::iota(0u, rows)) (*this)(i, j) = col_data[i];
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
