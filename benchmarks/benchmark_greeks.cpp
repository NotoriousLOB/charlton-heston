/**
 * Benchmarks for Greek Calculations (C API)
 */

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdio>

#include "charlton.h"

static charlton_model_params default_params() {
    charlton_model_params p;
    p.S0 = 1.0; p.r = 0.0; p.q = 0.0; p.T = 1.0 / 52.0;
    p.H = 0.12; p.lambda = 0.1; p.theta = 0.3156;
    p.nu = 0.331; p.rho = -0.681; p.V0 = 0.0392;
    return p;
}

static void BM_Greeks_Essential(benchmark::State& state) {
    auto p = default_params();
    charlton_pricing_result r;
    for (auto _ : state) {
        charlton_greeks(&p, 1.0, CHARLTON_GREEKS_ESSENTIAL, &r);
        benchmark::DoNotOptimize(r.delta);
    }
}
BENCHMARK(BM_Greeks_Essential);

static void BM_Greeks_Standard(benchmark::State& state) {
    auto p = default_params();
    charlton_pricing_result r;
    for (auto _ : state) {
        charlton_greeks(&p, 1.0, CHARLTON_GREEKS_STANDARD, &r);
        benchmark::DoNotOptimize(r.vanna);
    }
}
BENCHMARK(BM_Greeks_Standard);

static void BM_Greeks_Cornucopia(benchmark::State& state) {
    auto p = default_params();
    charlton_pricing_result r;
    for (auto _ : state) {
        charlton_greeks(&p, 1.0, CHARLTON_GREEKS_CORNUCOPIA, &r);
        benchmark::DoNotOptimize(r.roughness);
    }
}
BENCHMARK(BM_Greeks_Cornucopia);

static void BM_Greeks_MultipleStrikes(benchmark::State& state) {
    auto p = default_params();
    double strikes[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    charlton_pricing_result r;
    for (auto _ : state) {
        for (int i = 0; i < 5; ++i)
            charlton_greeks(&p, strikes[i], CHARLTON_GREEKS_ESSENTIAL, &r);
    }
    state.SetItemsProcessed(state.iterations() * 5);
}
BENCHMARK(BM_Greeks_MultipleStrikes);

static void BM_Delta_FD(benchmark::State& state) {
    auto p = default_params();
    double h = 0.0001;
    for (auto _ : state) {
        double p1 = charlton_price_put(&p, 1.0, 1e-10);
        charlton_model_params p2 = p;
        p2.S0 += h;
        double pr2 = charlton_price_put(&p2, 1.0, 1e-10);
        benchmark::DoNotOptimize((pr2 - p1) / h);
    }
}
BENCHMARK(BM_Delta_FD);

int main(int argc, char** argv) {
    printf("\n======================================================================\n");
    printf("CHARLTON Greek Calculation Accuracy Verification (C API)\n");
    printf("======================================================================\n");

    charlton_model_params p = default_params();
    charlton_pricing_result r;
    charlton_greeks(&p, 1.0, CHARLTON_GREEKS_CORNUCOPIA, &r);

    printf("\nGreek Values (ATM Put, T=1/52):\n");
    printf("  Price:     %g\n", r.price);
    printf("  Delta:     %g\n", r.delta);
    printf("  Gamma:     %g\n", r.gamma);
    printf("  Theta:     %g\n", r.theta);
    printf("  Vega:      %g\n", r.vega);
    printf("  Rho:       %g\n", r.rho);
    printf("  Vanna:     %g\n", r.vanna);
    printf("  Volga:     %g\n", r.volga);
    printf("  Roughness: %g\n", r.roughness);

    printf("\nRunning Google Benchmark...\n");
    printf("======================================================================\n");

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
