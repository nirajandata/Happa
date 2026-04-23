//
// Created by nirakira on 4/24/26.
//

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <format>
#include <stdexcept>
#include <concepts>

namespace happa {

template<typename T>
concept VectorElement = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { a += b } -> std::same_as<T&>;
    { a -= b } -> std::same_as<T&>;
    { a *= b } -> std::same_as<T&>;
    { a /= b } -> std::same_as<T&>;
    { T{0} };
    { T{1} };
    { std::abs(a) };
};

template<VectorElement T>
class Vector {
public:
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    constexpr Vector() = default;
    explicit constexpr Vector(size_t n) : data_(n, T{0}) {}
    constexpr Vector(std::initializer_list<T> list) : data_(list) {}

    [[nodiscard]] constexpr size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] constexpr T& operator[](size_t i) { return data_[i]; }
    [[nodiscard]] constexpr const T& operator[](size_t i) const { return data_[i]; }

    constexpr iterator begin() noexcept { return data_.begin(); }
    constexpr const_iterator begin() const noexcept { return data_.begin(); }
    constexpr iterator end() noexcept { return data_.end(); }
    constexpr const_iterator end() const noexcept { return data_.end(); }

    constexpr Vector& operator+=(const T& scalar) {
        for (auto& x : data_) x += scalar;
        return *this;
    }
    constexpr Vector& operator-=(const T& scalar) {
        for (auto& x : data_) x -= scalar;
        return *this;
    }
    constexpr Vector& operator*=(const T& scalar) {
        for (auto& x : data_) x *= scalar;
        return *this;
    }
    constexpr Vector& operator/=(const T& scalar) {
        for (auto& x : data_) x /= scalar;
        return *this;
    }

    constexpr Vector& operator+=(const Vector& other) {
        if (size() != other.size()) throw std::invalid_argument("Vector sizes differ");
        std::ranges::transform(data_, other.data_, data_.begin(), std::plus<T>());
        return *this;
    }
    constexpr Vector& operator-=(const Vector& other) {
        if (size() != other.size()) throw std::invalid_argument("Vector sizes differ");
        std::ranges::transform(data_, other.data_, data_.begin(), std::minus<T>());
        return *this;
    }
    constexpr Vector& operator*=(const Vector& other) { // Hadamard product
        if (size() != other.size()) throw std::invalid_argument("Vector sizes differ");
        std::ranges::transform(data_, other.data_, data_.begin(), std::multiplies<T>());
        return *this;
    }
    constexpr Vector& operator/=(const Vector& other) {
        if (size() != other.size()) throw std::invalid_argument("Vector sizes differ");
        std::ranges::transform(data_, other.data_, data_.begin(), std::divides<T>());
        return *this;
    }

    [[nodiscard]] constexpr T norm_l2() const {
        return std::sqrt(dot(*this));
    }
    [[nodiscard]] constexpr T norm_l1() const {
        return std::ranges::fold_left(data_ | std::views::transform([](T x) { return std::abs(x); }), T{0}, std::plus<T>());
    }
    [[nodiscard]] constexpr T norm_linf() const {
        auto abs_view = data_ | std::views::transform([](T x) { return std::abs(x); });
        return *std::ranges::max_element(abs_view);
    }

    [[nodiscard]] constexpr T dot(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Vector sizes differ");
        return std::inner_product(data_.begin(), data_.end(), other.data_.begin(), T{0});
    }

    [[nodiscard]] constexpr Vector cross(const Vector& other) const {
        if (size() != 3 || other.size() != 3)
            throw std::invalid_argument("Cross product only defined for 3D vectors");
        return Vector{
            (*this)[1] * other[2] - (*this)[2] * other[1],
            (*this)[2] * other[0] - (*this)[0] * other[2],
            (*this)[0] * other[1] - (*this)[1] * other[0]
        };
    }

    [[nodiscard]] constexpr Vector normalized() const {
        T n = norm_l2();
        if (n == T{0}) throw std::runtime_error("Cannot normalize zero vector");
        return *this / n;
    }

    [[nodiscard]] constexpr T angle(const Vector& other) const {
        T d = dot(other);
        T denom = norm_l2() * other.norm_l2();
        if (denom == T{0}) throw std::runtime_error("Zero vector in angle calculation");
        return std::acos(d / denom);
    }

    [[nodiscard]] constexpr Vector project_onto(const Vector& other) const {
        T scalar = dot(other) / other.dot(other);
        return other * scalar;
    }

    [[nodiscard]] constexpr Vector componentwise_min(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Vector sizes differ");
        Vector result(size());
        std::ranges::transform(data_, other.data_, result.data_.begin(), [](T a, T b) { return std::min(a, b); });
        return result;
    }
    [[nodiscard]] constexpr Vector componentwise_max(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Vector sizes differ");
        Vector result(size());
        std::ranges::transform(data_, other.data_, result.data_.begin(), [](T a, T b) { return std::max(a, b); });
        return result;
    }

    constexpr void resize(size_t n) {
        data_.resize(n, T{0});
    }

    [[nodiscard]] constexpr std::vector<T> to_std_vector() const { return data_; }
    constexpr Vector(const std::vector<T>& vec) : data_(vec) {}

    [[nodiscard]] constexpr bool operator==(const Vector& other) const {
        return data_ == other.data_;
    }

    friend constexpr Vector operator+(Vector lhs, const Vector& rhs) { return lhs += rhs; }
    friend constexpr Vector operator-(Vector lhs, const Vector& rhs) { return lhs -= rhs; }
    friend constexpr Vector operator*(Vector lhs, const Vector& rhs) { return lhs *= rhs; }
    friend constexpr Vector operator/(Vector lhs, const Vector& rhs) { return lhs /= rhs; }

    friend constexpr Vector operator+(Vector vec, const T& scalar) { return vec += scalar; }
    friend constexpr Vector operator-(Vector vec, const T& scalar) { return vec -= scalar; }
    friend constexpr Vector operator*(Vector vec, const T& scalar) { return vec *= scalar; }
    friend constexpr Vector operator/(Vector vec, const T& scalar) { return vec /= scalar; }

    friend constexpr Vector operator+(const T& scalar, Vector vec) { return vec += scalar; }
    friend constexpr Vector operator-(const T& scalar, Vector vec) {
        for (auto& x : vec.data_) x = scalar - x;
        return vec;
    }
    friend constexpr Vector operator*(const T& scalar, Vector vec) { return vec *= scalar; }

private:
    std::vector<T> data_;
};

template<typename T>
Vector(std::initializer_list<T>) -> Vector<T>;

}

template<happa::VectorElement T>
struct std::formatter<happa::Vector<T>> : std::formatter<std::string> {
    auto format(const happa::Vector<T>& vec, format_context& ctx) const {
        auto out = std::format_to(ctx.out(), "[");
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) out = std::format_to(out, ", ");
            out = std::format_to(out, "{}", vec[i]);
        }
        out = std::format_to(out, "]");
        return out;
    }
};