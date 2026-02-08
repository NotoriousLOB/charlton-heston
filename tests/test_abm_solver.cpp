/**
 * Tests for BL-Modified Fractional Adams Solver
 */

#include <gtest/gtest.h>
#include "charlton.hpp"
#include "test_data.hpp"

using namespace charlton;
using namespace charlton_test;

class ABMSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default test parameters
        H = ElEuchRosenbaumParams::H;
        T = 1.0 / 252.0;  // 1 day
        N = 1000;
        gamma = ElEuchRosenbaumParams::gamma;
        theta = ElEuchRosenbaumParams::theta;
        nu = ElEuchRosenbaumParams::nu;
        rho = ElEuchRosenbaumParams::rho;
    }
    
    double H, T, gamma, theta, nu, rho;
    size_t N;
};

TEST_F(ABMSolverTest, ConstructorValidatesParameters) {
    // Valid parameters should not throw
    EXPECT_NO_THROW({
        FractionalABMSolver<double> solver(H, T, N, gamma, theta, nu, rho);
    });
    
    // Invalid H should throw
    EXPECT_THROW({
        FractionalABMSolver<double> solver(-0.1, T, N, gamma, theta, nu, rho);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        FractionalABMSolver<double> solver(0.6, T, N, gamma, theta, nu, rho);
    }, std::invalid_argument);
    
    // Invalid correlation should throw
    EXPECT_THROW({
        FractionalABMSolver<double> solver(H, T, N, gamma, theta, nu, 1.5);
    }, std::invalid_argument);
}

TEST_F(ABMSolverTest, SolveSingleReturnsFiniteValue) {
    FractionalABMSolver<double> solver(H, T, N, gamma, theta, nu, rho);
    
    // Test at various frequencies
    std::vector<std::complex<double>> test_freqs = {
        {0.0, -0.5},
        {1.0, -0.5},
        {10.0, -0.5},
        {100.0, -0.5}
    };
    
    for (const auto& u : test_freqs) {
        auto result = solver.solve_single(u);
        EXPECT_TRUE(std::isfinite(result.real())) 
            << "Real part not finite for u = " << u;
        EXPECT_TRUE(std::isfinite(result.imag())) 
            << "Imaginary part not finite for u = " << u;
    }
}

TEST_F(ABMSolverTest, SolveBatchMatchesSingle) {
    FractionalABMSolver<double> solver(H, T, N, gamma, theta, nu, rho);
    
    std::vector<std::complex<double>> batch = {
        {0.0, -0.5},
        {1.0, -0.5},
        {5.0, -0.5}
    };
    
    std::vector<std::complex<double>> batch_result;
    solver.solve_batch(batch, batch_result);
    
    for (size_t i = 0; i < batch.size(); ++i) {
        auto single_result = solver.solve_single(batch[i]);
        EXPECT_NEAR(batch_result[i].real(), single_result.real(), 1e-10);
        EXPECT_NEAR(batch_result[i].imag(), single_result.imag(), 1e-10);
    }
}

TEST_F(ABMSolverTest, DecayRateEstimateIsPositive) {
    FractionalABMSolver<double> solver(H, T, N, gamma, theta, nu, rho);
    
    double v0 = ElEuchRosenbaumParams::v0;
    double decay = solver.get_decay_rate_estimate(T, v0);
    
    EXPECT_GT(decay, 0.0) << "Decay rate should be positive";
}

TEST_F(ABMSolverTest, CharacteristicFunctionAtZero) {
    // At u = 0, the characteristic function should be 1
    // So the exponent should be 0
    FractionalABMSolver<double> solver(H, T, N, gamma, theta, nu, rho);
    
    std::complex<double> u(0.0, 0.0);
    auto exponent = solver.solve_single(u);
    
    // The exponent at u=0 should be close to 0
    EXPECT_NEAR(exponent.real(), 0.0, 1e-6);
    EXPECT_NEAR(exponent.imag(), 0.0, 1e-6);
}

TEST_F(ABMSolverTest, ConvergenceWithRefinement) {
    // Test that solution converges as N increases
    std::complex<double> u(10.0, -0.5);
    
    std::vector<size_t> N_values = {100, 200, 400, 800};
    std::vector<std::complex<double>> results;
    
    for (size_t n : N_values) {
        FractionalABMSolver<double> solver(H, T, n, gamma, theta, nu, rho);
        results.push_back(solver.solve_single(u));
    }
    
    // Check convergence - differences should decrease
    for (size_t i = 1; i < results.size(); ++i) {
        double diff = std::abs(results[i] - results[i-1]);
        double prev_diff = (i > 1) ? std::abs(results[i-1] - results[i-2]) : diff * 2;
        
        // Convergence should be roughly O(N^(-α))
        EXPECT_LT(diff, prev_diff * 1.5) 
            << "Solution not converging at refinement " << i;
    }
}
