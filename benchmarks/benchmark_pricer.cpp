/**
 * Benchmarks for Rough Heston Pricer
 */

#include <benchmark/benchmark.h>
#include "charlton.hpp"

using namespace charlton;

// Benchmark configuration
static constexpr double S0 = 1.0;
static constexpr double r = 0.0;
static constexpr double q = 0.0;
static constexpr double H = 0.12;
static constexpr double lambda = 0.1;
static constexpr double theta = 0.3156;
static constexpr double nu = 0.331;
static constexpr double rho = -0.681;
static constexpr double V0 = 0.0392;

static RoughHestonPricer<double>::ModelParams create_params(double T) {
    RoughHestonPricer<double>::ModelParams params;
    params.S0 = S0;
    params.r = r;
    params.q = q;
    params.T = T;
    params.H = H;
    params.lambda = lambda;
    params.theta = theta;
    params.nu = nu;
    params.rho = rho;
    params.V0 = V0;
    return params;
}

// Benchmark: Price single put option
static void BM_PricePut_1D(benchmark::State& state) {
    auto params = create_params(1.0 / 252.0);  // 1 day
    RoughHestonPricer<double> pricer(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(pricer.price_put(1.0));
    }
}
BENCHMARK(BM_PricePut_1D);

// Benchmark: Price put option for 1 week maturity
static void BM_PricePut_1W(benchmark::State& state) {
    auto params = create_params(1.0 / 52.0);  // 1 week
    RoughHestonPricer<double> pricer(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(pricer.price_put(1.0));
    }
}
BENCHMARK(BM_PricePut_1W);

// Benchmark: Price put option for 1 month maturity
static void BM_PricePut_1M(benchmark::State& state) {
    auto params = create_params(1.0 / 12.0);  // 1 month
    RoughHestonPricer<double> pricer(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(pricer.price_put(1.0));
    }
}
BENCHMARK(BM_PricePut_1M);

// Benchmark: Price multiple strikes (vectorized)
static void BM_PriceMultipleStrikes(benchmark::State& state) {
    auto params = create_params(1.0 / 52.0);
    
    std::vector<double> strikes = {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2};
    
    for (auto _ : state) {
        for (double K : strikes) {
            RoughHestonPricer<double> pricer(params);
            benchmark::DoNotOptimize(pricer.price_put(K));
        }
    }
    state.SetItemsProcessed(state.iterations() * strikes.size());
}
BENCHMARK(BM_PriceMultipleStrikes);

// Benchmark: Price with different error tolerances
static void BM_PricePut_Tolerance_1e6(benchmark::State& state) {
    auto params = create_params(1.0 / 52.0);
    RoughHestonPricer<double> pricer(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(pricer.price_put(1.0, 1e-6));
    }
}
BENCHMARK(BM_PricePut_Tolerance_1e6);

static void BM_PricePut_Tolerance_1e8(benchmark::State& state) {
    auto params = create_params(1.0 / 52.0);
    RoughHestonPricer<double> pricer(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(pricer.price_put(1.0, 1e-8));
    }
}
BENCHMARK(BM_PricePut_Tolerance_1e8);

static void BM_PricePut_Tolerance_1e10(benchmark::State& state) {
    auto params = create_params(1.0 / 52.0);
    RoughHestonPricer<double> pricer(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(pricer.price_put(1.0, 1e-10));
    }
}
BENCHMARK(BM_PricePut_Tolerance_1e10);

// Benchmark: Conformal Bootstrap pricing
static void BM_PricePut_Bootstrap(benchmark::State& state) {
    auto params = create_params(1.0 / 52.0);
    RoughHestonPricer<double> pricer(params);
    
    for (auto _ : state) {
        double error_estimate;
        benchmark::DoNotOptimize(pricer.price_put_bootstrap(1.0, error_estimate));
    }
}
BENCHMARK(BM_PricePut_Bootstrap);

// Benchmark: Implied volatility calculation
static void BM_ImpliedVolatility(benchmark::State& state) {
    auto params = create_params(1.0 / 52.0);
    RoughHestonPricer<double> pricer(params);
    double price = pricer.price_put(1.0);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            RoughHestonPricer<double>::implied_volatility(price, S0, 1.0, 1.0/52.0, r, false)
        );
    }
}
BENCHMARK(BM_ImpliedVolatility);

// Benchmark: SINH parameter computation
static void BM_SINHParameters(benchmark::State& state) {
    double T = 1.0 / 52.0;
    double decay_rate = 0.1;
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            compute_sinh_parameters<double>(
                T, S0, 1.0, r, decay_rate,
                -2.0, 1.0, -M_PI/4, M_PI/4,
                1e-10, false
            )
        );
    }
}
BENCHMARK(BM_SINHParameters);

// Benchmark: Multiple maturities (typical calibration scenario)
static void BM_CalibrationScenario(benchmark::State& state) {
    std::vector<double> maturities = {1.0/252.0, 1.0/52.0, 1.0/12.0};
    std::vector<double> strikes = {0.9, 0.95, 1.0, 1.05, 1.1};
    
    for (auto _ : state) {
        for (double T : maturities) {
            auto params = create_params(T);
            RoughHestonPricer<double> pricer(params);
            for (double K : strikes) {
                benchmark::DoNotOptimize(pricer.price_put(K));
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * maturities.size() * strikes.size());
}
BENCHMARK(BM_CalibrationScenario);

// BENCHMARK_MAIN(); // Removed - using benchmark_main library instead
