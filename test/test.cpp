#include <iostream>
#include "test_matrix.hpp"
#include "test_vector.hpp"

int main() {
    happa::test::run_all_tests(); //for matrix
    test_vector(); //vector test
}
