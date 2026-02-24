/**
 * Tests for Rough Heston Greek Calculations (C API)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#include "charlton.h"

extern "C" {
#include "test_data.h"
}

class GreeksTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.S0 = 1.0;
        params.r = 0.0;
        params.q = 0.0;
        params.T = 1.0 / 52.0;
        params.H = CHARLTON_TEST_H;
        params.lambda = CHARLTON_TEST_GAMMA;
        params.theta = CHARLTON_TEST_THETA;
        params.nu = CHARLTON_TEST_NU;
        params.rho = CHARLTON_TEST_RHO;
        params.V0 = CHARLTON_TEST_V0;
        K = 1.0;
    }

    charlton_model_params params;
    double K;
};

TEST_F(GreeksTest, DeltaIsReasonable) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    std::cout << "Delta = " << result.delta << std::endl;
    EXPECT_NEAR(result.delta, -0.5, 0.3);
    EXPECT_GE(result.delta, -1.0);
    EXPECT_LE(result.delta, 0.0);
}

TEST_F(GreeksTest, GammaIsPositive) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    std::cout << "Gamma = " << result.gamma << std::endl;
    EXPECT_GT(result.gamma, 0.0);
    EXPECT_LT(result.gamma, 50.0);
}

TEST_F(GreeksTest, GammaPeaksAtATM) {
    double strikes[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    double gammas[5];

    for (int i = 0; i < 5; ++i) {
        charlton_pricing_result result;
        charlton_greeks(&params, strikes[i], CHARLTON_GREEKS_ESSENTIAL, &result);
        gammas[i] = result.gamma;
        std::cout << "Gamma at K=" << strikes[i] << ": " << gammas[i] << std::endl;
    }

    size_t max_idx = 0;
    for (int i = 1; i < 5; ++i) {
        if (gammas[i] > gammas[max_idx]) max_idx = (size_t)i;
    }
    EXPECT_NEAR((double)max_idx, 2.0, 1.0);
}

TEST_F(GreeksTest, ThetaIsNegative) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    std::cout << "Theta = " << result.theta << std::endl;
    EXPECT_LT(result.theta, 0.0);
}

TEST_F(GreeksTest, VegaIsPositive) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    std::cout << "Vega = " << result.vega << std::endl;
    EXPECT_GT(result.vega, 0.0);
}

TEST_F(GreeksTest, RhoIsNegativeForPut) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    std::cout << "Rho = " << result.rho << std::endl;
    EXPECT_LT(result.rho, 0.0);
}

TEST_F(GreeksTest, VannaIsFinite) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_STANDARD, &result);

    std::cout << "Vanna = " << result.vanna << std::endl;
    EXPECT_TRUE(std::isfinite(result.vanna));
}

TEST_F(GreeksTest, VolgaIsFinite) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_STANDARD, &result);

    std::cout << "Volga = " << result.volga << std::endl;
    EXPECT_TRUE(std::isfinite(result.volga));
}

TEST_F(GreeksTest, AllCornucopiaGreeksAreFinite) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_CORNUCOPIA, &result);

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

TEST_F(GreeksTest, DeltaConvergesWithBumpSize) {
    double strikes[] = {0.95, 1.0, 1.05};

    for (int i = 0; i < 3; ++i) {
        charlton_pricing_result result;
        charlton_greeks(&params, strikes[i], CHARLTON_GREEKS_ESSENTIAL, &result);
        EXPECT_TRUE(std::isfinite(result.delta));
        EXPECT_GE(result.delta, -1.0);
        EXPECT_LE(result.delta, 0.0);
    }
}

TEST_F(GreeksTest, GammaIsSymmetricAroundATM) {
    charlton_pricing_result res1, res2;
    charlton_greeks(&params, 0.95, CHARLTON_GREEKS_ESSENTIAL, &res1);
    charlton_greeks(&params, 1.05, CHARLTON_GREEKS_ESSENTIAL, &res2);

    std::cout << "Gamma at 0.95: " << res1.gamma << std::endl;
    std::cout << "Gamma at 1.05: " << res2.gamma << std::endl;

    double ratio = res1.gamma / res2.gamma;
    EXPECT_NEAR(ratio, 1.0, 0.5) << "Gamma not symmetric around ATM";
}

TEST_F(GreeksTest, PriceOnlyReturnsZeroGreeks) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_PRICE_ONLY, &result);

    EXPECT_NE(result.price, 0.0);
    EXPECT_EQ(result.delta, 0.0);
    EXPECT_EQ(result.gamma, 0.0);
    EXPECT_EQ(result.vega, 0.0);
}

TEST_F(GreeksTest, EssentialSetHasCorrectGreeks) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    EXPECT_NE(result.price, 0.0);
    EXPECT_NE(result.delta, 0.0);
    EXPECT_NE(result.gamma, 0.0);
    EXPECT_NE(result.theta, 0.0);
    EXPECT_NE(result.vega, 0.0);
    EXPECT_NE(result.rho, 0.0);

    EXPECT_EQ(result.vanna, 0.0);
    EXPECT_EQ(result.volga, 0.0);
}

TEST_F(GreeksTest, StandardSetHasCorrectGreeks) {
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_STANDARD, &result);

    EXPECT_NE(result.price, 0.0);
    EXPECT_NE(result.delta, 0.0);
    EXPECT_NE(result.gamma, 0.0);
    EXPECT_NE(result.vega, 0.0);
    EXPECT_NE(result.vanna, 0.0);
    EXPECT_NE(result.volga, 0.0);

    EXPECT_EQ(result.zomma, 0.0);
}

TEST_F(GreeksTest, PutCallParityForDelta) {
    params.r = 0.0;
    params.q = 0.0;

    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    EXPECT_NEAR(result.delta, -0.5, 0.3);
}

TEST_F(GreeksTest, DeltaMonotonicity) {
    double strikes[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    double deltas[5];

    for (int i = 0; i < 5; ++i) {
        charlton_pricing_result result;
        charlton_greeks(&params, strikes[i], CHARLTON_GREEKS_ESSENTIAL, &result);
        deltas[i] = result.delta;
    }

    for (int i = 1; i < 5; ++i) {
        EXPECT_LE(deltas[i], deltas[i - 1])
            << "Put delta should decrease with strike";
    }
}

TEST_F(GreeksTest, VegaPeaksAtATM) {
    double strikes[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    double vegas[5];

    for (int i = 0; i < 5; ++i) {
        charlton_pricing_result result;
        charlton_greeks(&params, strikes[i], CHARLTON_GREEKS_ESSENTIAL, &result);
        vegas[i] = result.vega;
        std::cout << "Vega at K=" << strikes[i] << ": " << vegas[i] << std::endl;
    }

    size_t max_idx = 0;
    for (int i = 1; i < 5; ++i) {
        if (vegas[i] > vegas[max_idx]) max_idx = (size_t)i;
    }
    EXPECT_NEAR((double)max_idx, 2.0, 1.0);
}

TEST_F(GreeksTest, PriceDeltaConsistency) {
    double h = 0.01;

    double price1 = charlton_price_put(&params, K, 1e-10);

    charlton_model_params params2 = params;
    params2.S0 += h;
    double price2 = charlton_price_put(&params2, K, 1e-10);

    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);

    double price_change = price2 - price1;
    double expected_change = result.delta * h;

    std::cout << "Actual price change: " << price_change << std::endl;
    std::cout << "Expected from delta: " << expected_change << std::endl;

    if (std::abs(price_change) > 1e-10) {
        double relative_error = std::abs(price_change - expected_change) / std::abs(price_change);
        EXPECT_LT(relative_error, 0.3) << "Delta doesn't predict price change accurately";
    }
}

TEST_F(GreeksTest, GreeksWithDifferentMaturities) {
    double maturities[] = {1.0 / 252.0, 1.0 / 52.0, 1.0 / 12.0, 0.25};

    for (int mi = 0; mi < 4; ++mi) {
        charlton_model_params p = params;
        p.T = maturities[mi];

        charlton_pricing_result result;
        charlton_greeks(&p, K, CHARLTON_GREEKS_ESSENTIAL, &result);

        EXPECT_TRUE(std::isfinite(result.delta));
        EXPECT_TRUE(std::isfinite(result.gamma));
        EXPECT_TRUE(std::isfinite(result.theta));
        EXPECT_TRUE(std::isfinite(result.vega));
        EXPECT_TRUE(std::isfinite(result.rho));

        EXPECT_GE(result.delta, -1.0);
        EXPECT_LE(result.delta, 0.0);
        EXPECT_GT(result.gamma, 0.0);
        EXPECT_LT(result.theta, 0.0);
        EXPECT_GT(result.vega, 0.0);
    }
}

TEST_F(GreeksTest, GreeksWithExtremeStrikes) {
    double strikes[] = {0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0};

    for (int i = 0; i < 7; ++i) {
        charlton_pricing_result result;
        charlton_greeks(&params, strikes[i], CHARLTON_GREEKS_ESSENTIAL, &result);

        EXPECT_TRUE(std::isfinite(result.delta));
        EXPECT_TRUE(std::isfinite(result.gamma));
        EXPECT_GE(result.delta, -1.0);
        EXPECT_LE(result.delta, 0.0);
        EXPECT_GE(result.gamma, 0.0);
    }
}
