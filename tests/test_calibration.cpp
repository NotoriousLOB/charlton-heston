/**
 * Tests for Rough Heston Calibration Engine (C API)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

#include "charlton.h"

extern "C" {
#include "test_data.h"
}

class CalibrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cal_params.S0 = 1.0;
        cal_params.r = 0.0;
        cal_params.q = 0.0;
        cal_params.max_iterations = 100;
        cal_params.tolerance = 1e-4;
        cal_params.step_size = 0.01;

        true_params.S0 = 1.0;
        true_params.r = 0.0;
        true_params.q = 0.0;
        true_params.T = 1.0 / 52.0;
        true_params.H = 0.1;
        true_params.lambda = 2.0;
        true_params.theta = 0.04;
        true_params.nu = 0.5;
        true_params.rho = -0.6;
        true_params.V0 = 0.04;
    }

    charlton_calibration_params cal_params;
    charlton_model_params true_params;
};

TEST_F(CalibrationTest, InitialGuessIsGenerated) {
    charlton_market_quote quotes[1] = {{ 0.5, 1.0, 0.2, 0 }};
    charlton_calibration_result guess = charlton_generate_initial_guess(&cal_params, quotes, 1);

    EXPECT_GT(guess.H, 0.0);
    EXPECT_GT(guess.lambda, 0.0);
    EXPECT_GT(guess.theta, 0.0);
    EXPECT_GT(guess.nu, 0.0);
    EXPECT_GT(guess.V0, 0.0);
    EXPECT_GE(guess.rho, -1.0);
    EXPECT_LE(guess.rho, 1.0);
}

TEST_F(CalibrationTest, CalibrationConvergesForSyntheticData) {
    double maturities[] = {1.0 / 52.0, 2.0 / 52.0};
    double moneyness[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    charlton_market_quote quotes[10];
    charlton_generate_test_market_data(&true_params, cal_params.S0, cal_params.r,
                                       maturities, 2, moneyness, 5, quotes);

    charlton_calibration_result guess = charlton_generate_initial_guess(&cal_params, quotes, 10);
    charlton_calibration_result result;
    charlton_calibrate_adam(&cal_params, quotes, 10, &guess, &result);

    EXPECT_LT(result.rmse, 0.01);
    EXPECT_NEAR(result.H, true_params.H, 0.05);
    EXPECT_NEAR(result.lambda, true_params.lambda, 0.5);
    EXPECT_NEAR(result.theta, true_params.theta, 0.01);
    EXPECT_NEAR(result.nu, true_params.nu, 0.1);
    EXPECT_NEAR(result.rho, true_params.rho, 0.1);
    EXPECT_NEAR(result.V0, true_params.V0, 0.01);
}

TEST_F(CalibrationTest, CalibrationWithSingleMaturity) {
    double maturities[] = {1.0 / 52.0};
    double moneyness[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    charlton_market_quote quotes[5];
    charlton_generate_test_market_data(&true_params, cal_params.S0, cal_params.r,
                                       maturities, 1, moneyness, 5, quotes);

    charlton_calibration_result guess = charlton_generate_initial_guess(&cal_params, quotes, 5);
    charlton_calibration_result result;
    charlton_calibrate_adam(&cal_params, quotes, 5, &guess, &result);

    EXPECT_LT(result.rmse, 0.02);
}

TEST_F(CalibrationTest, CalibrationWithWideStrikeRange) {
    double maturities[] = {1.0 / 52.0, 2.0 / 52.0};
    double moneyness[] = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    charlton_market_quote quotes[14];
    charlton_generate_test_market_data(&true_params, cal_params.S0, cal_params.r,
                                       maturities, 2, moneyness, 7, quotes);

    charlton_calibration_result guess = charlton_generate_initial_guess(&cal_params, quotes, 14);
    charlton_calibration_result result;
    charlton_calibrate_adam(&cal_params, quotes, 14, &guess, &result);

    EXPECT_LT(result.rmse, 0.01);
}

TEST_F(CalibrationTest, ParameterBoundsAreRespected) {
    double maturities[] = {1.0 / 52.0};
    double moneyness[] = {0.9, 1.0, 1.1};
    charlton_market_quote quotes[3];
    charlton_generate_test_market_data(&true_params, cal_params.S0, cal_params.r,
                                       maturities, 1, moneyness, 3, quotes);

    charlton_calibration_result guess = charlton_generate_initial_guess(&cal_params, quotes, 3);
    charlton_calibration_result result;
    charlton_calibrate_adam(&cal_params, quotes, 3, &guess, &result);

    EXPECT_GE(result.H, 0.01);
    EXPECT_LE(result.H, 0.49);
    EXPECT_GE(result.lambda, 0.01);
    EXPECT_GE(result.theta, 0.001);
    EXPECT_GE(result.nu, 0.01);
    EXPECT_GE(result.rho, -0.99);
    EXPECT_LE(result.rho, 0.99);
    EXPECT_GE(result.V0, 0.001);
}

TEST_F(CalibrationTest, LBFGSCalibrationRuns) {
    double maturities[] = {1.0 / 52.0};
    double moneyness[] = {0.9, 1.0, 1.1};
    charlton_market_quote quotes[3];
    charlton_generate_test_market_data(&true_params, cal_params.S0, cal_params.r,
                                       maturities, 1, moneyness, 3, quotes);

    charlton_calibration_result guess = charlton_generate_initial_guess(&cal_params, quotes, 3);
    charlton_calibration_result result;
    int rc = charlton_calibrate_lbfgs(&cal_params, quotes, 3, &guess, &result);

    EXPECT_EQ(rc, CHARLTON_OK);
    EXPECT_TRUE(std::isfinite(result.rmse));
    EXPECT_GT(result.iterations, 0);
}

TEST_F(CalibrationTest, LOBSyntheticQuotesGenerated) {
    charlton_model_params p = true_params;
    p.T = 0.5;
    double T_grid[] = {0.25, 0.5};
    double K_grid[] = {0.9, 1.0, 1.1};
    charlton_market_quote quotes[6];

    int rc = charlton_lob_synth_quotes(&p, 2, 3, T_grid, K_grid,
                                        10.0, 1.5, 0.5, quotes);
    EXPECT_EQ(rc, CHARLTON_OK);

    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::isfinite(quotes[i].iv));
        EXPECT_GT(quotes[i].iv, 0.0);
        EXPECT_GT(quotes[i].T, 0.0);
        EXPECT_GT(quotes[i].K, 0.0);
    }
}
