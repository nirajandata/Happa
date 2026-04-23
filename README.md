# Happa

### Scientific Computational Library

**Happa** is a header‑only linear algebra library that puts **cache efficiency first**.  
 
- All algorithms are written in **C++26** with `constexpr` support where possible.

##  Quick Start

```cpp
#include <happa/Matrix.hpp>
#include <iostream>

int main() {
    using namespace happa;
    Matrix<double> A = {{1, 2}, {3, 4}};
    Matrix<double> B = {{5, 6}, {7, 8}};

    auto C = A * B;                     // adaptive multiply
    auto D = A.transpose();
    auto x = A.solve({{5}, {11}});      // solve Ax = b

    std::cout << std::format("{}\n", C);
}
```

## Performance

Benchmarked using Google Benchmark on matrix multiplication.

### Results (512×512)

| Method      | Time (ms) | Speedup |
|------------|----------|--------|
| Standard   | 2850 ms  | 1.0×   |
| Tiled      | 2266 ms  | 1.25×  |
| Strassen   | 1866 ms  | 1.52×  |

### Summary

- Tiled multiplication improves cache locality (~25% faster)
- Strassen outperforms both for larger matrices
- For small sizes (≤128), overhead reduces Strassen’s benefit