/**
 * Benchmarks for Fractional ABM Solver (C API)
 */

#include <benchmark/benchmark.h>
#include <cmath>

#include "charlton.h"

static void BM_SolveSingle_N100(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 100, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx u = 10.0 + -0.5 * CHARLTON_I;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_solve_single(&solver, u, 3));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveSingle_N100);

static void BM_SolveSingle_N200(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx u = 10.0 + -0.5 * CHARLTON_I;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_solve_single(&solver, u, 3));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveSingle_N200);

static void BM_SolveSingle_N500(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 500, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx u = 10.0 + -0.5 * CHARLTON_I;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_solve_single(&solver, u, 3));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveSingle_N500);

static void BM_SolveSingle_N1000(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 1000, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx u = 10.0 + -0.5 * CHARLTON_I;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_solve_single(&solver, u, 3));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveSingle_N1000);

static void BM_SolveBatch_10(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx batch[10], result[10];
    for (int i = 0; i < 10; ++i) batch[i] = (double)i + -0.5 * CHARLTON_I;
    for (auto _ : state) {
        charlton_abm_solve_batch(&solver, batch, 10, result);
        benchmark::DoNotOptimize(result[0]);
    }
    state.SetItemsProcessed(state.iterations() * 10);
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveBatch_10);

static void BM_SolveBatch_50(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx batch[50], result[50];
    for (int i = 0; i < 50; ++i) batch[i] = (double)i * 0.2 + -0.5 * CHARLTON_I;
    for (auto _ : state) {
        charlton_abm_solve_batch(&solver, batch, 50, result);
        benchmark::DoNotOptimize(result[0]);
    }
    state.SetItemsProcessed(state.iterations() * 50);
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveBatch_50);

static void BM_SolveBatch_100(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx batch[100], result[100];
    for (int i = 0; i < 100; ++i) batch[i] = (double)i * 0.1 + -0.5 * CHARLTON_I;
    for (auto _ : state) {
        charlton_abm_solve_batch(&solver, batch, 100, result);
        benchmark::DoNotOptimize(result[0]);
    }
    state.SetItemsProcessed(state.iterations() * 100);
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveBatch_100);

static void BM_DecayRateEstimate(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_decay_rate(&solver, 1.0/52.0, 0.0392));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_DecayRateEstimate);

static void BM_ABM_Init_N256(benchmark::State& state) {
    for (auto _ : state) {
        charlton_abm_solver solver;
        charlton_abm_init(&solver, 0.12, 1.0/52.0, 256, 0.1, 0.3156, 0.331, -0.681);
        charlton_abm_free(&solver);
    }
}
BENCHMARK(BM_ABM_Init_N256);

static void BM_SolveSingle_Freq1(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx u = 1.0 + -0.5 * CHARLTON_I;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_solve_single(&solver, u, 3));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveSingle_Freq1);

static void BM_SolveSingle_Freq10(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx u = 10.0 + -0.5 * CHARLTON_I;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_solve_single(&solver, u, 3));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveSingle_Freq10);

static void BM_SolveSingle_Freq100(benchmark::State& state) {
    charlton_abm_solver solver;
    charlton_abm_init(&solver, 0.12, 1.0/52.0, 200, 0.1, 0.3156, 0.331, -0.681);
    charlton_cmplx u = 100.0 + -0.5 * CHARLTON_I;
    for (auto _ : state)
        benchmark::DoNotOptimize(charlton_abm_solve_single(&solver, u, 3));
    charlton_abm_free(&solver);
}
BENCHMARK(BM_SolveSingle_Freq100);

BENCHMARK_MAIN();
