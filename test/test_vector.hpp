#include "Happa/Vector.hpp"
#include <cassert>
#include <cmath>
#include <print>

void test_vector() {
    using namespace happa;

    Vector<int> v1 = {1, 2, 3};
    Vector<int> v2 = {4, 5, 6};
    assert(v1.size() == 3);
    assert(v1[1] == 2);
    assert(v1.dot(v2) == 1*4 + 2*5 + 3*6);

    Vector<double> a = {1.0, 0.0, 0.0};
    Vector<double> b = {0.0, 1.0, 0.0};
    auto c = a.cross(b);
    assert(c[0] == 0.0 && c[1] == 0.0 && c[2] == 1.0);

    assert(std::abs(a.norm_l2() - 1.0) < 1e-12);
    Vector<double> d = {3.0, 4.0};
    assert(std::abs(d.norm_l2() - 5.0) < 1e-12);
    assert(std::abs(d.norm_l1() - 7.0) < 1e-12);
    assert(std::abs(d.norm_linf() - 4.0) < 1e-12);

    Vector<double> u = {2.0, 3.0};
    Vector<double> v = {4.0, 5.0};
    auto proj = u.project_onto(v);
    assert(std::abs(proj[0] - 92.0/41.0) < 1e-12);
    assert(std::abs(proj[1] - 115.0/41.0) < 1e-12);

    Vector<double> x = {1.0, 0.0};
    Vector<double> y = {0.0, 1.0};
    assert(std::abs(x.angle(y) - std::acos(0.0)) < 1e-12);

    auto min_vec = u.componentwise_min(v);
    assert(min_vec[0] == 2.0 && min_vec[1] == 3.0);
    auto max_vec = u.componentwise_max(v);
    assert(max_vec[0] == 4.0 && max_vec[1] == 5.0);

    auto scaled = u * 2.0;
    assert(scaled[0] == 4.0 && scaled[1] == 6.0);
    auto added = u + 1.0;
    assert(added[0] == 3.0 && added[1] == 4.0);

    std::println("All vector tests passed.");
}