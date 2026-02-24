/**
 * Benchmarks for Rough Heston Pricer (C API)
 */

#include <benchmark/benchmark.h>
#include <cmath>

#include "charlton.h"

static charlton_model_params default_params(double T = 1.0 / 52.0) {
    charlton_model_params p;
    p.S0 = 1.0; p.r = 0.0; p.q = 0.0; p.T = T;
    p.H = 0.12; p.lambda = 0.1; p.theta = 0.3156;
    p.nu = 0.331; p.rho = -0.681; p.V0 = 0.0392;
    return p;
}

static void BM_PricePut_1D(benchmark::State& state) {
    auto p = default_params(1.0 / 252.0);
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_price_put(&p, 1.0, 1e-10));
}
BENCHMARK(BM_PricePut_1D);

static void BM_PricePut_1W(benchmark::State& state) {
    auto p = default_params(1.0 / 52.0);
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_price_put(&p, 1.0, 1e-10));
}
BENCHMARK(BM_PricePut_1W);

static void BM_PricePut_1M(benchmark::State& state) {
    auto p = default_params(1.0 / 12.0);
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_price_put(&p, 1.0, 1e-10));
}
BENCHMARK(BM_PricePut_1M);

static void BM_PriceMultipleStrikes(benchmark::State& state) {
    auto p = default_params();
    double strikes[] = {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2};
    for (auto _ : state) {
        for (int i = 0; i < 9; ++i)
            benchmark::DoNotOptimize(charlton_price_put(&p, strikes[i], 1e-10));
    }
    state.SetItemsProcessed(state.iterations() * 9);
}
BENCHMARK(BM_PriceMultipleStrikes);

static void BM_PricePut_Tolerance_1e6(benchmark::State& state) {
    auto p = default_params();
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_price_put(&p, 1.0, 1e-6));
}
BENCHMARK(BM_PricePut_Tolerance_1e6);

static void BM_PricePut_Tolerance_1e8(benchmark::State& state) {
    auto p = default_params();
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_price_put(&p, 1.0, 1e-8));
}
BENCHMARK(BM_PricePut_Tolerance_1e8);

static void BM_PricePut_Tolerance_1e10(benchmark::State& state) {
    auto p = default_params();
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_price_put(&p, 1.0, 1e-10));
}
BENCHMARK(BM_PricePut_Tolerance_1e10);

static void BM_PricePut_Bootstrap(benchmark::State& state) {
    auto p = default_params();
    double err;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_price_put_bootstrap(&p, 1.0, &err, 1e-10));
}
BENCHMARK(BM_PricePut_Bootstrap);

static void BM_ImpliedVolatility(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_implied_volatility(0.02, 1.0, 1.0, 1.0/52.0, 0.0, 0));
}
BENCHMARK(BM_ImpliedVolatility);

static void BM_SINHParameters(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_compute_sinh_params(
            1.0/52.0, 1.0, 1.0, 0.0, 0.1, -2.0, 1.0, -M_PI/4, M_PI/4, 1e-10, 0));
}
BENCHMARK(BM_SINHParameters);

static void BM_CachedCF_Init(benchmark::State& state) {
    auto p = default_params();
    for (auto _ : state) {
        charlton_cached_cf cache;
        charlton_cache_cf_init(&cache, &p, 1.0, 1e-8);
        charlton_cache_cf_free(&cache);
    }
}
BENCHMARK(BM_CachedCF_Init);

static void BM_CalibrationScenario(benchmark::State& state) {
    double maturities[] = {1.0/252.0, 1.0/52.0, 1.0/12.0};
    double strikes[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    for (auto _ : state) {
        for (int ti = 0; ti < 3; ++ti) {
            auto p = default_params(maturities[ti]);
            for (int ki = 0; ki < 5; ++ki)
                benchmark::DoNotOptimize(charlton_price_put(&p, strikes[ki], 1e-10));
        }
    }
    state.SetItemsProcessed(state.iterations() * 15);
}
BENCHMARK(BM_CalibrationScenario);

BENCHMARK_MAIN();
