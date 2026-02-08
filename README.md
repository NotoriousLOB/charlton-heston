# CHARLTON

**C**onformal **H**yperbolic **A**ccelerated **R**ough **L**évy **T**ransform for **O**ption **N**umerics

> *"You can have my Bermudan swaptions when you pry them from my COLD DEAD HAND!!!"*
> — Charlton (Rough) Heston

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)

A high-performance C++ library for pricing and calibration in the **Rough Heston model**, implementing state-of-the-art numerical methods including SINH-acceleration, the BL-modified fractional Adams method, and the Conformal Bootstrap principle for error control.

## Features

- 🚀 **SINH-Acceleration**: Conformal deformation for ultra-fast Fourier inversion (20-60 quadrature points vs 200-400 for standard methods)
- 🎯 **BL-Modified Adams**: Corrected fractional Riccati solver for short-maturity accuracy
- ✅ **Conformal Bootstrap**: Rigorous a posteriori error control eliminates "ghost calibration"
- 📊 **Comprehensive Greeks**: Machine-precision sensitivities via Complex Step Differentiation
- ⚡ **SIMD Optimized**: AVX2/AVX512/NEON vectorization
- 🧵 **Parallel**: OpenMP batch processing for multiple strikes/frequencies
- 🔧 **Modern C++17**: Header-only template library with CMake build system

## Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.16+
- FFTW3 (auto-fetched if not found)
- Optional: OpenMP, Intel MKL

### Build

```bash
git clone https://github.com/[username]/charlton.git
cd charlton
mkdir build && cd build
cmake .. -DCHARLTON_BUILD_TESTS=ON -DCHARLTON_BUILD_BENCHMARKS=ON
make -j$(nproc)
```

### Basic Usage

```cpp
#include "charlton.hpp"

// Set up model parameters
charlton::RoughHestonPricer<double>::ModelParams params;
params.S0 = 1.0;        // Spot price
params.r = 0.0;         // Risk-free rate
params.T = 1.0/52.0;    // 1 week maturity
params.H = 0.12;        // Hurst parameter
params.lambda = 0.1;    // Mean reversion speed
params.theta = 0.3156;  // Long-term variance
params.nu = 0.331;      // Vol-of-vol
params.rho = -0.681;    // Correlation
params.V0 = 0.0392;     // Initial variance

// Create pricer and price an option
charlton::RoughHestonPricer<double> pricer(params);
double price = pricer.price_put(1.0);  // ATM put

// Compute implied volatility
double iv = charlton::RoughHestonPricer<double>::implied_volatility(
    price, params.S0, 1.0, params.T, params.r, false);
```

### Computing Greeks

```cpp
charlton::RoughHestonGreeks<double> greeks(params);
auto result = greeks.compute(1.0, charlton::GreekSet::CORNUCOPIA);

std::cout << "Delta: " << result.delta << "\n";
std::cout << "Gamma: " << result.gamma << "\n";
std::cout << "Vanna: " << result.vanna << "\n";
std::cout << "Roughness (dV/dH): " << result.roughness << "\n";
```

### Model Calibration

```cpp
// Set up calibrator
charlton::RoughHestonCalibrator<double>::CalibrationParams cal_params;
cal_params.S0 = 1.0;
cal_params.r = 0.0;

charlton::RoughHestonCalibrator<double> calibrator(cal_params);

// Add market quotes
for (const auto& quote : market_data) {
    calibrator.add_quote({quote.T, quote.K, quote.iv, false});
}

// Calibrate
auto initial_guess = calibrator.generate_initial_guess();
auto result = calibrator.calibrate_adam(initial_guess);

result.print();  // Display calibrated parameters
```

## Performance

| Operation | Time | Accuracy |
|-----------|------|----------|
| Single price (1 week) | ~1-5 ms | < 10⁻⁸ relative error |
| Essential Greeks | ~10-20 ms | Machine precision |
| Full Cornucopia | ~50-100 ms | Machine precision |
| Calibration (100 quotes) | ~10-30 s | < 0.1% parameter recovery |

Compared to Markovian approximations (BL2): **100-1000× faster** with **superior accuracy**.

## Documentation

- [Analysis & Design](docs/ANALYSIS.md) - Comprehensive technical analysis
- [Paper Abstract](docs/PAPER_ABSTRACT.md) - Academic paper proposal
- API Reference - Coming soon

## Testing

```bash
# Run tests
./build/tests/charlton_tests

# Run benchmarks
./build/benchmarks/charlton_benchmarks

# Run examples
./build/examples/example_basic_pricing
./build/examples/example_greeks
./build/examples/example_calibration
```

## Key Algorithms

### SINH-Acceleration

The Fourier inversion contour is deformed using:
```
χ_{ω₁,b,ω}(y) = iω₁ + b·sinh(iω + y)
```

This transforms oscillatory integrands into exponentially decaying ones, reducing required quadrature points by 5-10×.

### BL-Modified Adams Method

The fractional Riccati equation is solved with asymptotic correction:
```
h̃(ξ,t) = h(ξ,t)/(1+|ξ|)
h̃_as(ξ,t) = -0.5(ξ²+iξ)t^α/((1+|ξ|)Γ(α+1))
```

This corrects O(Δ^α|ξ|²) errors at short maturities.

### Conformal Bootstrap

Error is controlled by comparing prices from multiple contour deformations. If they agree within ε, the true price is within 100ε with high probability.

## References

1. Boyarchenko, S., De Innocentis, M., & Levendorski˘i, S. (2025). "Fast Reliable Pricing and Calibration of the Rough Heston Model"
2. El Euch, O. & Rosenbaum, M. (2019). "The characteristic function of rough Heston models", Mathematical Finance 29:3-38
3. Bayer, C. & Breneis, S. (2023). "Weak markovian approximations of rough Heston"
4. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). "Volatility is rough", Quantitative Finance 18(6):933-949

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This implementation is based on the pioneering work of Svetlana Boyarchenko, Marco De Innocentis, and Sergei Levendorski˘i on SINH-acceleration and the BL-modified fractional Adams method.

---

*"You can have my rough volatility model when you pry it from my COLD DEAD HAND!"*
