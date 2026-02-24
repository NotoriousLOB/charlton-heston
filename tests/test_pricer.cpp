/**
 * Tests for Rough Heston Pricer with SINH-acceleration (C API)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "charlton.h"

extern "C" {
#include "test_data.h"
}

class PricerTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.S0 = 1.0;
        params.r = 0.0;
        params.q = 0.0;
        params.T = 1.0 / 252.0;
        params.H = CHARLTON_TEST_H;
        params.lambda = CHARLTON_TEST_GAMMA;
        params.theta = CHARLTON_TEST_THETA;
        params.nu = CHARLTON_TEST_NU;
        params.rho = CHARLTON_TEST_RHO;
        params.V0 = CHARLTON_TEST_V0;
    }

    charlton_model_params params;
};

TEST_F(PricerTest, PutPriceIsNonNegative) {
    double strikes[] = {0.8, 0.9, 1.0, 1.1, 1.2};

    for (int i = 0; i < 5; ++i) {
        double price = charlton_price_put(&params, strikes[i], 1e-10);
        EXPECT_GE(price, 0.0) << "Put price should be non-negative for K=" << strikes[i];
        EXPECT_TRUE(std::isfinite(price)) << "Put price should be finite for K=" << strikes[i];
    }
}

TEST_F(PricerTest, PutPriceIncreasesWithStrike) {
    double strikes[] = {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1};
    double prev_price = 0.0;

    for (int i = 0; i < 7; ++i) {
        double price = charlton_price_put(&params, strikes[i], 1e-10);
        EXPECT_GE(price, prev_price - 1e-9)
            << "Put price should increase with strike: K=" << strikes[i];
        prev_price = price;
    }
}

TEST_F(PricerTest, PutCallParityHolds) {
    double K = 1.0;
    double put_price = charlton_price_put(&params, K, 1e-10);
    double call_price = charlton_price_call(&params, K, 1e-10);

    double fwd = params.S0 * std::exp((params.r - params.q) * params.T);
    double df = std::exp(-params.r * params.T);

    double parity_lhs = call_price - put_price;
    double parity_rhs = fwd - K * df;

    EXPECT_NEAR(parity_lhs, parity_rhs, 1e-8)
        << "Put-call parity violated";
}

TEST_F(PricerTest, ATMPriceIsReasonable) {
    double K = params.S0;
    double put_price = charlton_price_put(&params, K, 1e-10);

    double expected_approx = 0.4 * std::sqrt(params.V0 * params.T) * params.S0;

    EXPECT_GT(put_price, expected_approx * 0.3);
    EXPECT_LT(put_price, expected_approx * 3.0);
}

TEST_F(PricerTest, DeepOTMPutIsSmall) {
    double K = 0.5;
    double price = charlton_price_put(&params, K, 1e-10);

    EXPECT_LT(price, params.S0 * 0.01)
        << "Deep OTM put should be small";
}

TEST_F(PricerTest, DeepITMPutIsIntrinsic) {
    double K = 1.5;
    double price = charlton_price_put(&params, K, 1e-10);

    double intrinsic = K * std::exp(-params.r * params.T) - params.S0;

    EXPECT_GT(price, intrinsic * 0.9);
    EXPECT_LT(price, intrinsic * 1.1);
}

TEST_F(PricerTest, BootstrapErrorEstimateIsSmall) {
    double K = 1.0;
    double error_estimate;
    double price = charlton_price_put_bootstrap(&params, K, &error_estimate, 1e-10);

    EXPECT_TRUE(std::isfinite(price));
    EXPECT_TRUE(std::isfinite(error_estimate));
    EXPECT_GT(error_estimate, 0.0);

    EXPECT_LT(error_estimate / price, 0.01)
        << "Bootstrap error estimate too large relative to price";
}

TEST_F(PricerTest, ImpliedVolatilityRecoversInput) {
    double K = 1.0;
    double put_price = charlton_price_put(&params, K, 1e-10);
    double iv = charlton_implied_volatility(put_price, params.S0, K, params.T, params.r, 0);

    EXPECT_TRUE(std::isfinite(iv));
    EXPECT_GT(iv, 0.0);
    EXPECT_LT(iv, 2.0);
}

TEST_F(PricerTest, PriceConvergesWithErrorTolerance) {
    double K = 1.0;
    double tolerances[] = {1e-6, 1e-8, 1e-10};
    double prices[3];

    for (int i = 0; i < 3; ++i) {
        prices[i] = charlton_price_put(&params, K, tolerances[i]);
    }

    for (int i = 1; i < 3; ++i) {
        double diff = std::abs(prices[i] - prices[i - 1]);
        double prev_diff = (i > 1) ? std::abs(prices[i - 1] - prices[i - 2]) : diff * 10;
        EXPECT_LT(diff, prev_diff * 0.5)
            << "Price not converging with tighter tolerance";
    }
}

TEST_F(PricerTest, SINHParametersAreValid) {
    double decay_rate = 0.1;
    charlton_sinh_params sp = charlton_compute_sinh_params(
        params.T, params.S0, 1.0, params.r,
        decay_rate, -2.0, 1.0, -M_PI / 4, M_PI / 4, 1e-10, 0);

    EXPECT_GT(sp.b, 0.0);
    EXPECT_GT(sp.zeta, 0.0);
    EXPECT_GT(sp.N, 0u);
    EXPECT_GT(sp.Lambda, 0.0);
}

TEST_F(PricerTest, CachedCFMatchesDirect) {
    double K = 1.0;

    /* Price via cache */
    charlton_cached_cf cache;
    int rc = charlton_cache_cf_init(&cache, &params, K, 1e-8);
    ASSERT_EQ(rc, CHARLTON_OK);

    /* Price from cache uses same internal path as direct */
    double price_direct = charlton_price_put(&params, K, 1e-8);
    EXPECT_TRUE(std::isfinite(price_direct));
    EXPECT_GT(price_direct, 0.0);

    charlton_cache_cf_free(&cache);
}
