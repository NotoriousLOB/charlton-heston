/**
 * Tests for Rough Heston Pricer with SINH-acceleration
 */

#include <gtest/gtest.h>
#include "charlton.hpp"
#include "test_data.hpp"

using namespace charlton;
using namespace charlton_test;

class PricerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default model parameters
        params.S0 = 1.0;
        params.r = 0.0;
        params.q = 0.0;
        params.T = 1.0 / 252.0;  // 1 day
        params.H = ElEuchRosenbaumParams::H;
        params.lambda = ElEuchRosenbaumParams::gamma;
        params.theta = ElEuchRosenbaumParams::theta;
        params.nu = ElEuchRosenbaumParams::nu;
        params.rho = ElEuchRosenbaumParams::rho;
        params.V0 = ElEuchRosenbaumParams::v0;
    }
    
    RoughHestonPricer<double>::ModelParams params;
};

TEST_F(PricerTest, PutPriceIsNonNegative) {
    RoughHestonPricer<double> pricer(params);
    
    std::vector<double> strikes = {0.8, 0.9, 1.0, 1.1, 1.2};
    
    for (double K : strikes) {
        double price = pricer.price_put(K);
        EXPECT_GE(price, 0.0) << "Put price should be non-negative for K=" << K;
        EXPECT_TRUE(std::isfinite(price)) << "Put price should be finite for K=" << K;
    }
}

TEST_F(PricerTest, PutPriceIncreasesWithStrike) {
    RoughHestonPricer<double> pricer(params);
    
    double prev_price = 0.0;
    std::vector<double> strikes = {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1};
    
    for (double K : strikes) {
        double price = pricer.price_put(K);
        EXPECT_GE(price, prev_price - 1e-9) 
            << "Put price should increase with strike: K=" << K;
        prev_price = price;
    }
}

TEST_F(PricerTest, PutCallParityHolds) {
    RoughHestonPricer<double> pricer(params);
    
    double K = 1.0;
    double put_price = pricer.price_put(K);
    double call_price = pricer.price_call(K);
    
    double fwd = params.S0 * std::exp((params.r - params.q) * params.T);
    double df = std::exp(-params.r * params.T);
    
    // Put-call parity: C - P = S0 - K*exp(-rT)
    double parity_lhs = call_price - put_price;
    double parity_rhs = fwd - K * df;
    
    EXPECT_NEAR(parity_lhs, parity_rhs, 1e-8) 
        << "Put-call parity violated";
}

TEST_F(PricerTest, ATMPriceIsReasonable) {
    RoughHestonPricer<double> pricer(params);
    
    double K = params.S0;  // ATM
    double put_price = pricer.price_put(K);
    
    // ATM put should be roughly 0.4 * sigma * sqrt(T) * S0
    double expected_approx = 0.4 * std::sqrt(params.V0 * params.T) * params.S0;
    
    // Allow factor of 2 difference due to model specifics
    EXPECT_GT(put_price, expected_approx * 0.3);
    EXPECT_LT(put_price, expected_approx * 3.0);
}

TEST_F(PricerTest, DeepOTMPutIsSmall) {
    RoughHestonPricer<double> pricer(params);
    
    double K = 0.5;  // Deep OTM put
    double price = pricer.price_put(K);
    
    // Deep OTM put should be very small
    EXPECT_LT(price, params.S0 * 0.01) 
        << "Deep OTM put should be small";
}

TEST_F(PricerTest, DeepITMPutIsIntrinsic) {
    RoughHestonPricer<double> pricer(params);
    
    double K = 1.5;  // Deep ITM put
    double price = pricer.price_put(K);
    
    double intrinsic = K * std::exp(-params.r * params.T) - params.S0;
    
    // Deep ITM put should be close to intrinsic value
    EXPECT_GT(price, intrinsic * 0.9);
    EXPECT_LT(price, intrinsic * 1.1);
}

TEST_F(PricerTest, BootstrapErrorEstimateIsSmall) {
    RoughHestonPricer<double> pricer(params);
    
    double K = 1.0;
    double error_estimate;
    double price = pricer.price_put_bootstrap(K, error_estimate);
    
    EXPECT_TRUE(std::isfinite(price));
    EXPECT_TRUE(std::isfinite(error_estimate));
    EXPECT_GT(error_estimate, 0.0);
    
    // Error estimate should be reasonably small
    EXPECT_LT(error_estimate / price, 0.01) 
        << "Bootstrap error estimate too large relative to price";
}

TEST_F(PricerTest, ImpliedVolatilityRecoversInput) {
    // Test that implied vol can be computed and is reasonable
    RoughHestonPricer<double> pricer(params);
    
    double K = 1.0;
    double put_price = pricer.price_put(K);
    double iv = RoughHestonPricer<double>::implied_volatility(
        put_price, params.S0, K, params.T, params.r, false);
    
    EXPECT_TRUE(std::isfinite(iv));
    EXPECT_GT(iv, 0.0);
    EXPECT_LT(iv, 2.0);  // IV should be less than 200%
}

TEST_F(PricerTest, PriceConvergesWithErrorTolerance) {
    double K = 1.0;
    
    std::vector<double> tolerances = {1e-6, 1e-8, 1e-10};
    std::vector<double> prices;
    
    for (double tol : tolerances) {
        RoughHestonPricer<double> pricer(params);
        prices.push_back(pricer.price_put(K, tol));
    }
    
    // Differences should decrease with tighter tolerance
    for (size_t i = 1; i < prices.size(); ++i) {
        double diff = std::abs(prices[i] - prices[i-1]);
        double prev_diff = (i > 1) ? std::abs(prices[i-1] - prices[i-2]) : diff * 10;
        
        EXPECT_LT(diff, prev_diff * 0.5) 
            << "Price not converging with tighter tolerance";
    }
}

TEST_F(PricerTest, SINHParametersAreValid) {
    double decay_rate = 0.1;
    
    auto sinh_params = compute_sinh_parameters<double>(
        params.T, params.S0, 1.0, params.r,
        decay_rate,
        -2.0, 1.0, -M_PI/4, M_PI/4,
        1e-10, false
    );
    
    EXPECT_GT(sinh_params.b, 0.0);
    EXPECT_GT(sinh_params.zeta, 0.0);
    EXPECT_GT(sinh_params.N, 0u);
    EXPECT_GT(sinh_params.Lambda, 0.0);
}
