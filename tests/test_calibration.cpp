/**
 * Tests for Rough Heston Calibration Engine
 */

#include <gtest/gtest.h>
#include "charlton.hpp"
#include "test_data.hpp"

using namespace charlton;
using namespace charlton_test;

class CalibrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cal_params.S0 = 1.0;
        cal_params.r = 0.0;
        cal_params.q = 0.0;
        cal_params.max_iterations = 100;
        cal_params.tolerance = 1e-4;
        cal_params.step_size = 0.01;
        
        // True parameters for synthetic data
        true_params.S0 = 1.0;
        true_params.r = 0.0;
        true_params.q = 0.0;
        true_params.T = 1.0/52.0;  // Will be overridden per quote
        true_params.H = 0.1;
        true_params.lambda = 2.0;
        true_params.theta = 0.04;
        true_params.nu = 0.5;
        true_params.rho = -0.6;
        true_params.V0 = 0.04;
    }

    RoughHestonCalibrator<double>::CalibrationParams cal_params;
    RoughHestonPricer<double>::ModelParams true_params;
};

TEST_F(CalibrationTest, InitialGuessIsGenerated) {
    RoughHestonCalibrator<double> calibrator(cal_params);
    
    auto guess = calibrator.generate_initial_guess();
    
    EXPECT_GT(guess.H, 0.0);
    EXPECT_GT(guess.lambda, 0.0);
    EXPECT_GT(guess.theta, 0.0);
    EXPECT_GT(guess.nu, 0.0);
    EXPECT_GT(guess.V0, 0.0);
    EXPECT_GE(guess.rho, -1.0);
    EXPECT_LE(guess.rho, 1.0);
}

TEST_F(CalibrationTest, CalibrationConvergesForSyntheticData) {
    // Generate synthetic market data
    std::vector<double> maturities = {1.0/52.0, 2.0/52.0};
    std::vector<double> moneyness = {0.9, 0.95, 1.0, 1.05, 1.1};
    
    auto quotes = generate_test_market_data(
        cal_params.S0, cal_params.r,
        true_params,
        maturities, moneyness, 0.0
    );
    
    RoughHestonCalibrator<double> calibrator(cal_params);
    calibrator.add_quotes(quotes);
    
    auto initial = calibrator.generate_initial_guess();
    auto result = calibrator.calibrate_adam(initial);
    
    // Should converge for noiseless data
    EXPECT_LT(result.rmse, 0.01);
    
    // Parameters should be close to true values
    EXPECT_NEAR(result.H, true_params.H, 0.05);
    EXPECT_NEAR(result.lambda, true_params.lambda, 0.5);
    EXPECT_NEAR(result.theta, true_params.theta, 0.01);
    EXPECT_NEAR(result.nu, true_params.nu, 0.1);
    EXPECT_NEAR(result.rho, true_params.rho, 0.1);
    EXPECT_NEAR(result.V0, true_params.V0, 0.01);
}

TEST_F(CalibrationTest, CalibrationHandlesNoisyData) {
    // Generate synthetic market data with noise
    std::vector<double> maturities = {1.0/52.0, 2.0/52.0};
    std::vector<double> moneyness = {0.9, 0.95, 1.0, 1.05, 1.1};
    
    double noise_level = 0.005;  // 0.5% IV noise
    auto quotes = generate_test_market_data(
        cal_params.S0, cal_params.r,
        true_params,
        maturities, moneyness, noise_level
    );
    
    RoughHestonCalibrator<double> calibrator(cal_params);
    calibrator.add_quotes(quotes);
    
    auto initial = calibrator.generate_initial_guess();
    auto result = calibrator.calibrate_adam(initial);
    
    // RMSE should be close to noise level
    EXPECT_LT(result.rmse, noise_level * 3.0);
    
    // Parameters should be reasonably close
    EXPECT_NEAR(result.H, true_params.H, 0.1);
    EXPECT_NEAR(result.rho, true_params.rho, 0.2);
}

TEST_F(CalibrationTest, CalibrationResultIsPrintable) {
    RoughHestonCalibrator<double> calibrator(cal_params);
    auto result = calibrator.generate_initial_guess();
    
    // Should not throw when printing
    EXPECT_NO_THROW(result.print());
}

TEST_F(CalibrationTest, EmptyQuotesReturnsZeroError) {
    RoughHestonCalibrator<double> calibrator(cal_params);
    
    auto result = calibrator.generate_initial_guess();
    
    // With no quotes, RMSE should be zero
    EXPECT_EQ(result.rmse, 0.0);
}

TEST_F(CalibrationTest, CalibrationWithSingleMaturity) {
    std::vector<double> maturities = {1.0/52.0};
    std::vector<double> moneyness = {0.9, 0.95, 1.0, 1.05, 1.1};
    
    auto quotes = generate_test_market_data(
        cal_params.S0, cal_params.r,
        true_params,
        maturities, moneyness, 0.0
    );
    
    RoughHestonCalibrator<double> calibrator(cal_params);
    calibrator.add_quotes(quotes);
    
    auto initial = calibrator.generate_initial_guess();
    auto result = calibrator.calibrate_adam(initial);
    
    // Should still converge with single maturity
    EXPECT_LT(result.rmse, 0.02);
}

TEST_F(CalibrationTest, CalibrationWithWideStrikeRange) {
    std::vector<double> maturities = {1.0/52.0, 2.0/52.0};
    std::vector<double> moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    
    auto quotes = generate_test_market_data(
        cal_params.S0, cal_params.r,
        true_params,
        maturities, moneyness, 0.0
    );
    
    RoughHestonCalibrator<double> calibrator(cal_params);
    calibrator.add_quotes(quotes);
    
    auto initial = calibrator.generate_initial_guess();
    auto result = calibrator.calibrate_adam(initial);
    
    // Should handle wide strike range
    EXPECT_LT(result.rmse, 0.01);
}

TEST_F(CalibrationTest, ParameterBoundsAreRespected) {
    std::vector<double> maturities = {1.0/52.0};
    std::vector<double> moneyness = {0.9, 1.0, 1.1};
    
    auto quotes = generate_test_market_data(
        cal_params.S0, cal_params.r,
        true_params,
        maturities, moneyness, 0.0
    );
    
    RoughHestonCalibrator<double> calibrator(cal_params);
    calibrator.add_quotes(quotes);
    
    auto initial = calibrator.generate_initial_guess();
    auto result = calibrator.calibrate_adam(initial);
    
    // Check parameter bounds
    EXPECT_GE(result.H, 0.01);
    EXPECT_LE(result.H, 0.49);
    EXPECT_GE(result.lambda, 0.01);
    EXPECT_GE(result.theta, 0.001);
    EXPECT_GE(result.nu, 0.01);
    EXPECT_GE(result.rho, -0.99);
    EXPECT_LE(result.rho, 0.99);
    EXPECT_GE(result.V0, 0.001);
}
