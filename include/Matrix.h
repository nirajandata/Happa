//
// Created by nirakira on 1/16/26.
//
#pragma once
#include <vector>
#include <ranges>
#include <format>
#include <algorithm>
#include <stdexcept>
#include <bit>

namespace happa {
    template<typename T>
    concept MatrixElement =
            std::semiregular<T> && std::formattable<T, char> && requires(T a, T b)
            {
                { a + b } -> std::convertible_to<T>;
                { a - b } -> std::convertible_to<T>;
                { a * b } -> std::convertible_to<T>;
                { a / b } -> std::convertible_to<T>;
                { a += b } -> std::same_as<T &>;
                { a -= b } -> std::same_as<T &>;
                { a *= b } -> std::same_as<T &>;
                { a *= b } -> std::same_as<T &>;
                { T{0} };
                { T{1} };
                { std::abs(a) };
            };

    template<MatrixElement T>
    struct Matrix {
        using value_type = T;
        size_t rows{};
        size_t cols{};
        std::vector<T> data;

        constexpr Matrix(size_t row_, size_t col_) : rows(row_), cols(col_) {
            data.resize(rows * cols, T{0});
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

        constexpr Matrix &operator*=(const T &scalar) {
            for (auto &x: data) x *= scalar;
            return *this;
        }

        constexpr Matrix &operator/=(const T &scalar) {
            for (auto &x: data) x /= scalar;
            return *this;
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

        [[nodiscard]] constexpr Matrix &operator+=(const Matrix &other) noexcept {
            std::ranges::transform(data, other.data, data.begin(), std::plus<T>());
            return *this;
        }

        [[nodiscard]] constexpr Matrix &operator-=(const Matrix &other) noexcept {
            std::ranges::transform(data, other.data, data.begin(), std::minus<T>());
            return *this;
        }

        [[nodiscard]] constexpr Matrix &operator*=(const Matrix &other) noexcept {
            std::ranges::transform(data, other.data, data.begin(), std::multiplies<T>());
            return *this;
        }

        [[nodiscard]] constexpr Matrix &operator/=(const Matrix &other) noexcept {
            std::ranges::transform(data, other.data, data.begin(), std::divides<T>());
            return *this;
        }

        [[nodiscard]] constexpr T trace() const {
            auto diagonal = data | std::views::stride(cols + 1);
            return std::ranges::fold_left(diagonal, T{0}, std::plus<T>());
        }

        [[nodiscard]] constexpr Matrix inverse() const {
            return this->solve(Matrix::identity(rows));
        }


        //do not use this
        // it's only for verification and speed benchmarking against other mat-mul algos
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

        [[nodiscard]] constexpr Matrix tiled_multiply(const Matrix &other) const {
            if (cols != other.rows) throw std::invalid_argument("Dimension mismatch");

            Matrix<T> result(rows, other.cols);
            const size_t n = rows;
            const size_t m = cols;
            const size_t p = other.cols;

            constexpr size_t L1_BLOCK = 32;
            constexpr size_t L2_BLOCK = 128;

            for (size_t ii = 0; ii < n; ii += L2_BLOCK) {
                for (size_t kk = 0; kk < m; kk += L2_BLOCK) {
                    for (size_t jj = 0; jj < p; jj += L2_BLOCK) {
                        size_t i_max = std::min(ii + L2_BLOCK, n);
                        size_t k_max = std::min(kk + L2_BLOCK, m);
                        size_t j_max = std::min(jj + L2_BLOCK, p);

                        for (size_t i1 = ii; i1 < i_max; i1 += L1_BLOCK) {
                            for (size_t k1 = kk; k1 < k_max; k1 += L1_BLOCK) {
                                for (size_t j1 = jj; j1 < j_max; j1 += L1_BLOCK) {
                                    size_t i1_max = std::min(i1 + L1_BLOCK, i_max);
                                    size_t k1_max = std::min(k1 + L1_BLOCK, k_max);
                                    size_t j1_max = std::min(j1 + L1_BLOCK, j_max);

                                    for (size_t i = i1; i < i1_max; ++i) {
                                        for (size_t k = k1; k < k1_max; ++k) {
                                            T a_ik = (*this)(i, k);
                                            if (a_ik == T{0}) continue;

                                            for (size_t j = j1; j < j1_max; ++j) {
                                                result(i, j) += a_ik * other(k, j);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }

        [[nodiscard]] constexpr Matrix operator*(const Matrix &other) const {
            if (cols != other.rows) throw std::invalid_argument("Dimension mismatch");

            size_t max_dim = std::max({rows, cols, other.cols});

            if (max_dim >= 512 && rows == cols && cols == other.cols) {
                size_t padded_size = std::bit_ceil(max_dim);
                double waste_ratio = static_cast<double>((padded_size * padded_size) / (max_dim * max_dim));

                if (waste_ratio < 1.25) [[unlikely]] {
                    return strassen(other);
                }
            }

            return tiled_multiply(other);
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

            if (n <= 128) {
                Matrix tempA(n, n), tempB(n, n);
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = 0; j < n; ++j) {
                        tempA(i, j) = A(i, j);
                        tempB(i, j) = B(i, j);
                    }
                return tempA.tiled_multiply(tempB);
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

            Matrix t1 = add_v(a11, a22);
            Matrix t2 = add_v(b11, b22);
            auto m1 = strassen_rec(SubView{t1, 0, 0, mid}, SubView{t2, 0, 0, mid});

            Matrix t3 = add_v(a21, a22);
            auto m2 = strassen_rec(SubView{t3, 0, 0, mid}, b11);

            Matrix t4 = sub_v(b12, b22);
            auto m3 = strassen_rec(a11, SubView{t4, 0, 0, mid});

            Matrix t5 = sub_v(b21, b11);
            auto m4 = strassen_rec(a22, SubView{t5, 0, 0, mid});

            Matrix t6 = add_v(a11, a12);
            auto m5 = strassen_rec(SubView{t6, 0, 0, mid}, b22);

            Matrix t7 = sub_v(a21, a11);
            Matrix t8 = add_v(b11, b12);
            auto m6 = strassen_rec(SubView{t7, 0, 0, mid}, SubView{t8, 0, 0, mid});

            Matrix t9 = sub_v(a12, a22);
            Matrix t10 = add_v(b21, b22);
            auto m7 = strassen_rec(SubView{t9, 0, 0, mid}, SubView{t10, 0, 0, mid});

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

        [[nodiscard]] constexpr Matrix solve(const Matrix &B) const {
            if (rows != cols) throw std::invalid_argument("Matrix must be square to solve");
            if (rows != B.rows) throw std::invalid_argument("Dimension mismatch with B");

            size_t n = rows;
            Matrix LU = *this;
            std::vector<size_t> P(n);
            std::iota(P.begin(), P.end(), 0);

            for (size_t k = 0; k < n; ++k) {
                size_t pivot = k;
                T max_val = std::abs(LU(k, k));
                for (size_t i = k + 1; i < n; ++i) {
                    T val = std::abs(LU(i, k));
                    if (val > max_val) {
                        max_val = val;
                        pivot = i;
                    }
                }

                if (max_val < 1e-18) throw std::runtime_error("Matrix is singular");

                if (pivot != k) {
                    for (size_t j = 0; j < n; ++j) std::swap(LU(k, j), LU(pivot, j));
                    std::swap(P[k], P[pivot]);
                }

                T inv_pivot = T{1} / LU(k, k);
                for (size_t i = k + 1; i < n; ++i) {
                    LU(i, k) *= inv_pivot;
                    T multiplier = LU(i, k);

                    for (size_t j = k + 1; j < n; ++j) {
                        LU(i, j) -= multiplier * LU(k, j);
                    }
                }
            }

            Matrix X(n, B.cols);
            for (size_t j = 0; j < B.cols; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    X(i, j) = B(P[i], j);
                    for (size_t k = 0; k < i; ++k) {
                        X(i, j) -= LU(i, k) * X(k, j);
                    }
                }
            }

            for (size_t j = 0; j < B.cols; ++j) {
                for (int i = (int) n - 1; i >= 0; --i) {
                    for (size_t k = i + 1; k < n; ++k) {
                        X(i, j) -= LU(i, k) * X(k, j);
                    }
                    X(i, j) /= LU(i, i);
                }
            }

            return X;
        }

        [[nodiscard]] constexpr T determinant() const {
            size_t n = rows;
            Matrix LU = *this;
            int sign = 1;
            for (size_t k = 0; k < n; ++k) {
                size_t pivot = k;
                T max_v = std::abs(LU(k, k));
                for (size_t i = k + 1; i < n; ++i) if (std::abs(LU(i, k)) > max_v) {
                    max_v = std::abs(LU(i, k));
                    pivot = i;
                }
                if (max_v < 1e-18) return T{0};
                if (pivot != k) {
                    for (size_t j = 0; j < n; ++j) std::swap(LU(k, j), LU(pivot, j));
                    sign *= -1;
                }
                for (size_t i = k + 1; i < n; ++i) {
                    LU(i, k) /= LU(k, k);
                    T mult = LU(i, k);
                    for (size_t j = k + 1; j < n; ++j) LU(i, j) -= mult * LU(k, j);
                }
            }
            T det = static_cast<T>(sign);
            for (size_t i = 0; i < n; ++i) det *= LU(i, i);
            return det;
        }

        [[nodiscard]] constexpr Matrix operator/(const Matrix &other) const {
            if (other.rows != other.cols) throw std::invalid_argument("Division requires square denominator");
            auto AT = this->transpose();
            auto BT = other.transpose();
            return BT.solve(AT).transpose();
        }

        [[nodiscard]] bool operator==(const Matrix<T> &other) const {
            return rows == other.rows && cols == other.cols && data == other.data;
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
            for (size_t j: std::views::iota(0u, cols)) {
                auto col_view = std::views::iota(0u, rows)
                                | std::views::transform([&](size_t i) { return (*this)(i, j); });

                auto col_data = std::ranges::to<std::vector<T> >(col_view);
                std::ranges::sort(col_data);

                for (size_t i: std::views::iota(0u, rows)) (*this)(i, j) = col_data[i];
            }
        }

        [[nodiscard]] friend constexpr Matrix operator*(Matrix mat, const T &scalar) { return mat *= scalar; }
        [[nodiscard]] friend constexpr Matrix operator/(Matrix mat, const T &scalar) { return mat /= scalar; }
        [[nodiscard]] friend constexpr Matrix operator+(Matrix lhs, const Matrix &rhs) { return lhs += rhs; }
        [[nodiscard]] friend constexpr Matrix operator-(Matrix lhs, const Matrix &rhs) { return lhs -= rhs; }
    };

    template<typename T>
    Matrix(std::initializer_list<std::initializer_list<T> >) -> Matrix<T>;
}

template<happa::MatrixElement T>
struct std::formatter<happa::Matrix<T> > {
    constexpr auto static parse(std::format_parse_context &ctx) { return ctx.begin(); }

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
