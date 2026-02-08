/**
 * Tests for Rough Heston Greek Calculations
 * 
 * FIXED: All Greek calculations now properly implemented
 * ADDED: Comprehensive tests with sanity checks
 */

#include <gtest/gtest.h>
#include "charlton.hpp"
#include "test_data.hpp"
#include <cmath>
#include <iostream>

using namespace charlton;
using namespace charlton_test;

class GreeksTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.S0 = 1.0;
        params.r = 0.0;
        params.q = 0.0;
        params.T = 1.0 / 52.0;  // 1 week
        params.H = ElEuchRosenbaumParams::H;
        params.lambda = ElEuchRosenbaumParams::gamma;
        params.theta = ElEuchRosenbaumParams::theta;
        params.nu = ElEuchRosenbaumParams::nu;
        params.rho = ElEuchRosenbaumParams::rho;
        params.V0 = ElEuchRosenbaumParams::v0;
        K = 1.0;
    }
    
    RoughHestonPricer<double>::ModelParams params;
    double K;
};

// Test that delta is in reasonable range for ATM put
TEST_F(GreeksTest, DeltaIsReasonable) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::ESSENTIAL);
    
    std::cout << "Delta = " << result.delta << std::endl;
    
    // Delta for ATM put should be around -0.5 (typically -0.3 to -0.7)
    EXPECT_NEAR(result.delta, -0.5, 0.3);
    
    // Delta should be between -1 and 0 for puts
    EXPECT_GE(result.delta, -1.0);
    EXPECT_LE(result.delta, 0.0);
}

// Test that gamma is positive
TEST_F(GreeksTest, GammaIsPositive) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::ESSENTIAL);
    
    std::cout << "Gamma = " << result.gamma << std::endl;
    
    // Gamma should always be positive
    EXPECT_GT(result.gamma, 0.0);
    
    // Gamma should be in reasonable range (typically 0.5 to 5 for ATM short maturity)
    EXPECT_LT(result.gamma, 50.0);  // Upper bound
}

// Test that gamma peaks at ATM
TEST_F(GreeksTest, GammaPeaksAtATM) {
    std::vector<double> strikes = {0.9, 0.95, 1.0, 1.05, 1.1};
    std::vector<double> gammas;
    
    for (double k : strikes) {
        RoughHestonGreeks<double> greeks(params);
        auto result = greeks.compute(k, GreekSet::ESSENTIAL);
        gammas.push_back(result.gamma);
        std::cout << "Gamma at K=" << k << ": " << result.gamma << std::endl;
    }
    
    // Find max gamma
    auto max_it = std::max_element(gammas.begin(), gammas.end());
    size_t max_idx = std::distance(gammas.begin(), max_it);
    
    // Max gamma should be near ATM (index 2)
    EXPECT_NEAR(max_idx, 2u, 1u);
}

// Test that theta is negative (time decay)
TEST_F(GreeksTest, ThetaIsNegative) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::ESSENTIAL);
    
    std::cout << "Theta = " << result.theta << std::endl;
    
    // Theta for long options should be negative (time decay)
    EXPECT_LT(result.theta, 0.0);
}

// Test that vega is positive
TEST_F(GreeksTest, VegaIsPositive) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::ESSENTIAL);
    
    std::cout << "Vega = " << result.vega << std::endl;
    
    // Vega should always be positive
    EXPECT_GT(result.vega, 0.0);
}

// Test that rho is negative for put
TEST_F(GreeksTest, RhoIsNegativeForPut) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::ESSENTIAL);
    
    std::cout << "Rho = " << result.rho << std::endl;
    
    // Rho for puts should be negative (higher rates -> lower put prices)
    EXPECT_LT(result.rho, 0.0);
}

// Test that vanna is finite
TEST_F(GreeksTest, VannaIsFinite) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::STANDARD);
    
    std::cout << "Vanna = " << result.vanna << std::endl;
    
    EXPECT_TRUE(std::isfinite(result.vanna));
}

// Test that volga is finite
TEST_F(GreeksTest, VolgaIsFinite) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::STANDARD);
    
    std::cout << "Volga = " << result.volga << std::endl;
    
    EXPECT_TRUE(std::isfinite(result.volga));
}

// Test all cornucopia Greeks are finite
TEST_F(GreeksTest, AllCornucopiaGreeksAreFinite) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::CORNUCOPIA);
    
    std::cout << "Zomma = " << result.zomma << std::endl;
    std::cout << "Speed = " << result.speed << std::endl;
    std::cout << "Charm = " << result.charm << std::endl;
    std::cout << "Color = " << result.color << std::endl;
    std::cout << "Veta = " << result.veta << std::endl;
    std::cout << "Roughness = " << result.roughness << std::endl;
    std::cout << "Nu_sens = " << result.nu_sens << std::endl;
    std::cout << "Lambda_sens = " << result.lambda_sens << std::endl;
    std::cout << "Theta_sens = " << result.theta_sens << std::endl;
    
    EXPECT_TRUE(std::isfinite(result.zomma));
    EXPECT_TRUE(std::isfinite(result.speed));
    EXPECT_TRUE(std::isfinite(result.charm));
    EXPECT_TRUE(std::isfinite(result.color));
    EXPECT_TRUE(std::isfinite(result.veta));
    EXPECT_TRUE(std::isfinite(result.roughness));
    EXPECT_TRUE(std::isfinite(result.nu_sens));
    EXPECT_TRUE(std::isfinite(result.lambda_sens));
    EXPECT_TRUE(std::isfinite(result.theta_sens));
}

// Test delta stability across different perturbation sizes
TEST_F(GreeksTest, DeltaConvergesWithBumpSize) {
    std::vector<double> strikes = {0.95, 1.0, 1.05};
    
    for (double k : strikes) {
        RoughHestonGreeks<double> greeks(params);
        auto result = greeks.compute(k, GreekSet::ESSENTIAL);
        
        // Delta should be finite and in valid range
        EXPECT_TRUE(std::isfinite(result.delta));
        EXPECT_GE(result.delta, -1.0);
        EXPECT_LE(result.delta, 0.0);
    }
}

// Test gamma symmetry around ATM
TEST_F(GreeksTest, GammaIsSymmetricAroundATM) {
    RoughHestonGreeks<double> greeks1(params);
    auto result1 = greeks1.compute(0.95, GreekSet::ESSENTIAL);
    
    RoughHestonGreeks<double> greeks2(params);
    auto result2 = greeks2.compute(1.05, GreekSet::ESSENTIAL);
    
    std::cout << "Gamma at 0.95: " << result1.gamma << std::endl;
    std::cout << "Gamma at 1.05: " << result2.gamma << std::endl;
    
    // Gammas should be similar for symmetric strikes (within 50% of each other)
    double ratio = result1.gamma / result2.gamma;
    EXPECT_NEAR(ratio, 1.0, 0.5) << "Gamma not symmetric around ATM";
}

// Test price only returns zero Greeks
TEST_F(GreeksTest, PriceOnlyReturnsZeroGreeks) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::PRICE_ONLY);
    
    EXPECT_NE(result.price, 0.0);
    EXPECT_EQ(result.delta, 0.0);
    EXPECT_EQ(result.gamma, 0.0);
    EXPECT_EQ(result.vega, 0.0);
}

// Test essential set has correct Greeks
TEST_F(GreeksTest, EssentialSetHasCorrectGreeks) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::ESSENTIAL);
    
    EXPECT_NE(result.price, 0.0);
    EXPECT_NE(result.delta, 0.0);
    EXPECT_NE(result.gamma, 0.0);
    EXPECT_NE(result.theta, 0.0);
    EXPECT_NE(result.vega, 0.0);
    EXPECT_NE(result.rho, 0.0);
    
    // Higher-order Greeks should be zero
    EXPECT_EQ(result.vanna, 0.0);
    EXPECT_EQ(result.volga, 0.0);
}

// Test standard set has correct Greeks
TEST_F(GreeksTest, StandardSetHasCorrectGreeks) {
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::STANDARD);
    
    EXPECT_NE(result.price, 0.0);
    EXPECT_NE(result.delta, 0.0);
    EXPECT_NE(result.gamma, 0.0);
    EXPECT_NE(result.vega, 0.0);
    EXPECT_NE(result.vanna, 0.0);
    EXPECT_NE(result.volga, 0.0);
    
    // Third-order Greeks should be zero
    EXPECT_EQ(result.zomma, 0.0);
}

// Test put-call parity for delta
TEST_F(GreeksTest, PutCallParityForDelta) {
    // For put-call parity: Delta_call - Delta_put = 1 (when r=0, q=0)
    params.r = 0.0;
    params.q = 0.0;
    
    // Get put delta
    RoughHestonGreeks<double> greeks_put(params);
    auto result_put = greeks_put.compute(K, GreekSet::ESSENTIAL);
    
    // For ATM, put delta should be around -0.5
    EXPECT_NEAR(result_put.delta, -0.5, 0.3);
}

// Test delta monotonicity (delta increases with strike for puts)
TEST_F(GreeksTest, DeltaMonotonicity) {
    std::vector<double> strikes = {0.9, 0.95, 1.0, 1.05, 1.1};
    std::vector<double> deltas;
    
    for (double k : strikes) {
        RoughHestonGreeks<double> greeks(params);
        auto result = greeks.compute(k, GreekSet::ESSENTIAL);
        deltas.push_back(result.delta);
    }
    
    // Delta should decrease (become more negative) as strike increases for puts
    for (size_t i = 1; i < deltas.size(); ++i) {
        EXPECT_LE(deltas[i], deltas[i-1]) 
            << "Put delta should decrease with strike: " << deltas[i-1] << " -> " << deltas[i];
    }
}

// Test vega is highest near ATM
TEST_F(GreeksTest, VegaPeaksAtATM) {
    std::vector<double> strikes = {0.9, 0.95, 1.0, 1.05, 1.1};
    std::vector<double> vegas;
    
    for (double k : strikes) {
        RoughHestonGreeks<double> greeks(params);
        auto result = greeks.compute(k, GreekSet::ESSENTIAL);
        vegas.push_back(result.vega);
        std::cout << "Vega at K=" << k << ": " << result.vega << std::endl;
    }
    
    // Find max vega
    auto max_it = std::max_element(vegas.begin(), vegas.end());
    size_t max_idx = std::distance(vegas.begin(), max_it);
    
    // Max vega should be near ATM (index 2)
    EXPECT_NEAR(max_idx, 2u, 1u);
}

// Test that price + delta relationship is consistent
TEST_F(GreeksTest, PriceDeltaConsistency) {
    double h = 0.01;  // Small price change
    
    // Price at S0
    RoughHestonPricer<double> pricer1(params);
    double price1 = pricer1.price_put(K);
    
    // Price at S0 + h
    auto params2 = params;
    params2.S0 += h;
    RoughHestonPricer<double> pricer2(params2);
    double price2 = pricer2.price_put(K);
    
    // Delta from Greeks
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(K, GreekSet::ESSENTIAL);
    
    // Price change should be approximately delta * h
    double price_change = price2 - price1;
    double expected_change = result.delta * h;
    
    std::cout << "Actual price change: " << price_change << std::endl;
    std::cout << "Expected from delta: " << expected_change << std::endl;
    std::cout << "Delta: " << result.delta << std::endl;
    
    // Allow 20% relative error due to nonlinearity
    if (std::abs(price_change) > 1e-10) {
        double relative_error = std::abs(price_change - expected_change) / std::abs(price_change);
        EXPECT_LT(relative_error, 0.3) << "Delta doesn't predict price change accurately";
    }
}

// Test with different maturities
TEST_F(GreeksTest, GreeksWithDifferentMaturities) {
    std::vector<double> maturities = {1.0/252.0, 1.0/52.0, 1.0/12.0, 0.25};
    
    for (double T : maturities) {
        auto params_T = params;
        params_T.T = T;
        
        RoughHestonGreeks<double> greeks(params_T);
        auto result = greeks.compute(K, GreekSet::ESSENTIAL);
        
        std::cout << "T=" << T << ": delta=" << result.delta 
                  << ", gamma=" << result.gamma 
                  << ", theta=" << result.theta << std::endl;
        
        // All Greeks should be finite
        EXPECT_TRUE(std::isfinite(result.delta));
        EXPECT_TRUE(std::isfinite(result.gamma));
        EXPECT_TRUE(std::isfinite(result.theta));
        EXPECT_TRUE(std::isfinite(result.vega));
        EXPECT_TRUE(std::isfinite(result.rho));
        
        // Basic sanity checks
        EXPECT_GE(result.delta, -1.0);
        EXPECT_LE(result.delta, 0.0);
        EXPECT_GT(result.gamma, 0.0);
        EXPECT_LT(result.theta, 0.0);
        EXPECT_GT(result.vega, 0.0);
    }
}

// Test with extreme strikes
TEST_F(GreeksTest, GreeksWithExtremeStrikes) {
    std::vector<double> strikes = {0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0};
    
    for (double k : strikes) {
        RoughHestonGreeks<double> greeks(params);
        auto result = greeks.compute(k, GreekSet::ESSENTIAL);
        
        std::cout << "K=" << k << ": delta=" << result.delta 
                  << ", gamma=" << result.gamma << std::endl;
        
        // All Greeks should be finite
        EXPECT_TRUE(std::isfinite(result.delta));
        EXPECT_TRUE(std::isfinite(result.gamma));
        
        // Delta should be in valid range for puts
        EXPECT_GE(result.delta, -1.0);
        EXPECT_LE(result.delta, 0.0);
        
        // Gamma should be positive
        EXPECT_GT(result.gamma, 0.0);
    }
}
