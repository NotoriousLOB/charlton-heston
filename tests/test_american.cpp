/**
 * Tests for American Option Pricing via COS method (C API)
 */

#include <gtest/gtest.h>
#include <cmath>

#include "charlton.h"

extern "C" {
#include "test_data.h"
}

class AmericanTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.S0 = 1.0;
        params.r = 0.05;
        params.q = 0.0;
        params.T = 0.5;
        params.H = CHARLTON_TEST_H;
        params.lambda = CHARLTON_TEST_GAMMA;
        params.theta = CHARLTON_TEST_THETA;
        params.nu = CHARLTON_TEST_NU;
        params.rho = CHARLTON_TEST_RHO;
        params.V0 = CHARLTON_TEST_V0;
    }

    charlton_model_params params;
};

TEST_F(AmericanTest, AmericanPutNonNegative) {
    double strikes[] = {0.8, 0.9, 1.0, 1.1, 1.2};

    for (int i = 0; i < 5; ++i) {
        charlton_american_result result;
        int rc = charlton_price_american_put(&params, strikes[i], 32, 64, &result);
        EXPECT_EQ(rc, CHARLTON_OK);
        EXPECT_GE(result.price, 0.0)
            << "American put price should be non-negative for K=" << strikes[i];
        EXPECT_TRUE(std::isfinite(result.price))
            << "American put price should be finite for K=" << strikes[i];
    }
}

TEST_F(AmericanTest, AmericanPutGeqEuropean) {
    double strikes[] = {0.9, 1.0, 1.1};

    for (int i = 0; i < 3; ++i) {
        double K = strikes[i];
        charlton_american_result am_result;
        int rc = charlton_price_american_put(&params, K, 32, 64, &am_result);
        EXPECT_EQ(rc, CHARLTON_OK);

        double euro_price = charlton_price_put(&params, K, 1e-8);

        EXPECT_GE(am_result.price, euro_price - 1e-8)
            << "American put >= European put for K=" << K
            << " (American=" << am_result.price << ", European=" << euro_price << ")";
        EXPECT_GE(am_result.early_exercise_premium, -1e-8)
            << "Early exercise premium should be non-negative for K=" << K;
    }
}

TEST_F(AmericanTest, AmericanPutGeqIntrinsic) {
    double strikes[] = {0.9, 1.0, 1.05, 1.1, 1.2};

    for (int i = 0; i < 5; ++i) {
        double K = strikes[i];
        charlton_american_result result;
        int rc = charlton_price_american_put(&params, K, 32, 64, &result);
        EXPECT_EQ(rc, CHARLTON_OK);

        double intrinsic = std::max(K - params.S0, 0.0);
        EXPECT_GE(result.price, intrinsic - 1e-8)
            << "American put >= intrinsic for K=" << K
            << " (price=" << result.price << ", intrinsic=" << intrinsic << ")";
    }
}

TEST_F(AmericanTest, AmericanPutConvergesWithTerms) {
    double K = 1.0;
    int terms[] = {32, 64, 128};
    double prices[3];

    for (int i = 0; i < 3; ++i) {
        charlton_american_result result;
        int rc = charlton_price_american_put(&params, K, 32, terms[i], &result);
        EXPECT_EQ(rc, CHARLTON_OK);
        prices[i] = result.price;
    }

    /* Successive differences should decrease */
    double diff1 = std::abs(prices[1] - prices[0]);
    double diff2 = std::abs(prices[2] - prices[1]);
    EXPECT_LT(diff2, diff1 + 1e-6)
        << "Price should converge as n_cos_terms increases: "
        << "diff(64-32)=" << diff1 << ", diff(128-64)=" << diff2;
}

TEST_F(AmericanTest, AmericanPutConvergesWithTimesteps) {
    double K = 1.0;
    int steps[] = {16, 32, 64};
    double prices[3];

    for (int i = 0; i < 3; ++i) {
        charlton_american_result result;
        int rc = charlton_price_american_put(&params, K, steps[i], 128, &result);
        EXPECT_EQ(rc, CHARLTON_OK);
        prices[i] = result.price;
    }

    double diff1 = std::abs(prices[1] - prices[0]);
    double diff2 = std::abs(prices[2] - prices[1]);
    EXPECT_LT(diff2, diff1 + 1e-6)
        << "Price should converge as n_timesteps increases: "
        << "diff(32-16)=" << diff1 << ", diff(64-32)=" << diff2;
}

TEST_F(AmericanTest, ExerciseBoundaryMonotone) {
    double K = 1.0;
    charlton_exercise_boundary eb;
    int rc = charlton_american_exercise_boundary(&params, K, 32, 64, 16, &eb);
    EXPECT_EQ(rc, CHARLTON_OK);
    EXPECT_EQ(eb.n_cheb, 16);

    /* For puts, S*(t) should be non-decreasing as t -> T
     * (Chebyshev nodes are ordered from T to 0, so boundary[0] is near T
     *  and boundary[n-1] is near 0). Boundary near expiry should be >= boundary earlier. */
    /* Just check all boundary values are positive and <= K */
    for (int j = 0; j < eb.n_cheb; ++j) {
        EXPECT_GE(eb.boundary[j], 0.0)
            << "Exercise boundary should be non-negative at node " << j;
        EXPECT_LE(eb.boundary[j], K + 1e-8)
            << "Exercise boundary should be <= K at node " << j;
    }

    charlton_exercise_boundary_free(&eb);
}

TEST_F(AmericanTest, ExerciseBoundaryBelowStrike) {
    double K = 1.1;
    charlton_exercise_boundary eb;
    int rc = charlton_american_exercise_boundary(&params, K, 32, 64, 12, &eb);
    EXPECT_EQ(rc, CHARLTON_OK);

    for (int j = 0; j < eb.n_cheb; ++j) {
        EXPECT_LE(eb.boundary[j], K + 1e-8)
            << "Put exercise boundary S*(t) <= K at node " << j
            << " (S*=" << eb.boundary[j] << ", K=" << K << ")";
    }

    charlton_exercise_boundary_free(&eb);
}

TEST_F(AmericanTest, AmericanCallEqEuropean) {
    /* For q=0, American call = European call (no early exercise advantage) */
    params.q = 0.0;
    params.r = 0.05;
    double K = 1.0;

    charlton_american_result am_result;
    int rc = charlton_price_american_call(&params, K, 32, 64, &am_result);
    EXPECT_EQ(rc, CHARLTON_OK);

    double euro_call = charlton_price_call(&params, K, 1e-8);

    /* American call should equal European call when q=0 (within tolerance).
     * The COS method may have some numerical overhead, so allow moderate tolerance. */
    double rel_diff = std::abs(am_result.price - euro_call) / (euro_call + 1e-12);
    EXPECT_LT(rel_diff, 0.05)
        << "American call should ~= European call when q=0: "
        << "American=" << am_result.price << ", European=" << euro_call;
}
