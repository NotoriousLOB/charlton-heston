/**
 * CHARLTON Test Data
 * 
 * Reference values and test parameters from the paper
 */

#ifndef CHARLTON_TEST_DATA_HPP
#define CHARLTON_TEST_DATA_HPP

#include <vector>
#include <cmath>

namespace charlton_test {

// Parameters from equation (1.3) in the paper
// El Euch-Rosenbaum calibrated parameters for S&P 500 (Jan 7, 2010)
struct ElEuchRosenbaumParams {
    static constexpr double alpha = 0.62;      // H + 0.5, so H = 0.12
    static constexpr double H = 0.12;
    static constexpr double gamma = 0.1;       // Mean reversion speed
    static constexpr double rho = -0.681;      // Correlation
    static constexpr double theta = 0.3156;    // Long-term variance
    static constexpr double nu = 0.331;        // Vol-of-vol
    static constexpr double v0 = 0.0392;       // Initial variance
};

// TSLA calibrated parameters from Section 7.1
struct TSLAParams {
    static constexpr double H = 0.011913;
    static constexpr double lambda = 2.36609;
    static constexpr double theta = 0.424949;
    static constexpr double nu = 0.5780;  // sigma = gamma*nu = 1.36839
    static constexpr double rho = -0.178493;
    static constexpr double v0 = 0.527527;
};

// Benchmark prices from Table 1 (T = 1/252, S0 = 1)
struct BenchmarkPrice {
    double K;
    double V;
};

inline std::vector<BenchmarkPrice> get_table1_benchmarks() {
    return {
        {0.95, 2.4557955e-07},
        {0.975, 1.29117047e-04},
        {1.0, 5.0111443104e-03},
        {1.025, 9.16277402e-05},
        {1.05, 3.3118e-08}
    };
}

// Benchmark prices from Table 2 (T = 1/52, S0 = 1)
inline std::vector<BenchmarkPrice> get_table2_benchmarks() {
    return {
        {0.8, 1.78e-05},
        {0.85, 1.89042e-04},
        {0.9, 1.390943e-03},
        {0.95, 6.975898e-03},
        {1.0, 0.023896768},
        {1.05, 6.556374e-03},
        {1.1, 9.78149e-04},
        {1.15, 6.73e-05}
    };
}

// Test tolerance for price comparisons
constexpr double PRICE_TOLERANCE = 1e-6;
constexpr double RELATIVE_TOLERANCE = 1e-4;

} // namespace charlton_test

#endif // CHARLTON_TEST_DATA_HPP
