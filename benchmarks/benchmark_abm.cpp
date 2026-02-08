/**
 * Benchmarks for BL-Modified Fractional Adams Solver
 */

#include <benchmark/benchmark.h>
#include "charlton.hpp"

using namespace charlton;

// Benchmark configuration
static constexpr double H = 0.12;
static constexpr double T = 1.0 / 52.0;
static constexpr double lambda_param = 0.1;
static constexpr double theta = 0.3156;
static constexpr double nu = 0.331;
static constexpr double rho = -0.681;

// Benchmark: Single frequency solve with different N
static void BM_SolveSingle_N100(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 100, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_N100);

static void BM_SolveSingle_N200(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_N200);

static void BM_SolveSingle_N500(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 500, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_N500);

static void BM_SolveSingle_N1000(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 1000, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_N1000);

// Benchmark: Batch solve (parallel)
static void BM_SolveBatch_10(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    
    std::vector<std::complex<double>> batch(10);
    for (size_t i = 0; i < 10; ++i) {
        batch[i] = std::complex<double>(i, -0.5);
    }
    
    std::vector<std::complex<double>> result;
    
    for (auto _ : state) {
        solver.solve_batch(batch, result);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * batch.size());
}
BENCHMARK(BM_SolveBatch_10);

static void BM_SolveBatch_50(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    
    std::vector<std::complex<double>> batch(50);
    for (size_t i = 0; i < 50; ++i) {
        batch[i] = std::complex<double>(i * 0.2, -0.5);
    }
    
    std::vector<std::complex<double>> result;
    
    for (auto _ : state) {
        solver.solve_batch(batch, result);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * batch.size());
}
BENCHMARK(BM_SolveBatch_50);

static void BM_SolveBatch_100(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    
    std::vector<std::complex<double>> batch(100);
    for (size_t i = 0; i < 100; ++i) {
        batch[i] = std::complex<double>(i * 0.1, -0.5);
    }
    
    std::vector<std::complex<double>> result;
    
    for (auto _ : state) {
        solver.solve_batch(batch, result);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * batch.size());
}
BENCHMARK(BM_SolveBatch_100);

// Benchmark: Decay rate estimate
static void BM_DecayRateEstimate(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    double v0 = 0.0392;
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.get_decay_rate_estimate(T, v0));
    }
}
BENCHMARK(BM_DecayRateEstimate);

// Benchmark: Different frequencies
static void BM_SolveSingle_Freq1(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    std::complex<double> u(1.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_Freq1);

static void BM_SolveSingle_Freq10(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_Freq10);

static void BM_SolveSingle_Freq100(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, T, 200, lambda_param, theta, nu, rho);
    std::complex<double> u(100.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_Freq100);

// Benchmark: Different maturities
static void BM_SolveSingle_T1D(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, 1.0/252.0, 200, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_T1D);

static void BM_SolveSingle_T1W(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, 1.0/52.0, 200, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_T1W);

static void BM_SolveSingle_T1M(benchmark::State& state) {
    FractionalABMSolver<double> solver(H, 1.0/12.0, 200, lambda_param, theta, nu, rho);
    std::complex<double> u(10.0, -0.5);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(solver.solve_single(u));
    }
}
BENCHMARK(BM_SolveSingle_T1M);

// BENCHMARK_MAIN(); // Removed - using benchmark_main library instead
