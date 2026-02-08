/**
 * Benchmarks for Rough Heston Greek Calculations
 * 
 * ADDED: Comparison with reference values and performance metrics
 */

#include <benchmark/benchmark.h>
#include "charlton.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

using namespace charlton;

// Benchmark configuration
static RoughHestonPricer<double>::ModelParams create_params() {
    RoughHestonPricer<double>::ModelParams params;
    params.S0 = 1.0;
    params.r = 0.0;
    params.q = 0.0;
    params.T = 1.0 / 52.0;
    params.H = 0.12;
    params.lambda = 0.1;
    params.theta = 0.3156;
    params.nu = 0.331;
    params.rho = -0.681;
    params.V0 = 0.0392;
    return params;
}

// Benchmark: Essential Greeks only
static void BM_Greeks_Essential(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL));
    }
}
BENCHMARK(BM_Greeks_Essential);

// Benchmark: Standard Greeks
static void BM_Greeks_Standard(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::STANDARD));
    }
}
BENCHMARK(BM_Greeks_Standard);

// Benchmark: Full Cornucopia Greeks
static void BM_Greeks_Cornucopia(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::CORNUCOPIA));
    }
}
BENCHMARK(BM_Greeks_Cornucopia);

// Benchmark: Individual Greek computations
static void BM_Greeks_Delta(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL).delta);
    }
}
BENCHMARK(BM_Greeks_Delta);

static void BM_Greeks_Gamma(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL).gamma);
    }
}
BENCHMARK(BM_Greeks_Gamma);

static void BM_Greeks_Vega(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL).vega);
    }
}
BENCHMARK(BM_Greeks_Vega);

static void BM_Greeks_Theta(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL).theta);
    }
}
BENCHMARK(BM_Greeks_Theta);

// Benchmark: Second-order Greeks
static void BM_Greeks_Vanna(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::STANDARD).vanna);
    }
}
BENCHMARK(BM_Greeks_Vanna);

static void BM_Greeks_Volga(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::STANDARD).volga);
    }
}
BENCHMARK(BM_Greeks_Volga);

// Benchmark: Third-order Greeks
static void BM_Greeks_Zomma(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::CORNUCOPIA).zomma);
    }
}
BENCHMARK(BM_Greeks_Zomma);

static void BM_Greeks_Speed(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::CORNUCOPIA).speed);
    }
}
BENCHMARK(BM_Greeks_Speed);

// Benchmark: Model parameter sensitivities
static void BM_Greeks_Roughness(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::CORNUCOPIA).roughness);
    }
}
BENCHMARK(BM_Greeks_Roughness);

static void BM_Greeks_NuSens(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::CORNUCOPIA).nu_sens);
    }
}
BENCHMARK(BM_Greeks_NuSens);

// Benchmark: Multiple strikes
static void BM_Greeks_MultipleStrikes(benchmark::State& state) {
    auto params = create_params();
    std::vector<double> strikes = {0.9, 0.95, 1.0, 1.05, 1.1};
    
    for (auto _ : state) {
        for (double K : strikes) {
            RoughHestonGreeks<double> greeks(params);
            benchmark::DoNotOptimize(greeks.compute(K, GreekSet::ESSENTIAL));
        }
    }
    state.SetItemsProcessed(state.iterations() * strikes.size());
}
BENCHMARK(BM_Greeks_MultipleStrikes);

// Benchmark: Different error tolerances
static void BM_Greeks_Tolerance_1e6(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL, 1e-6));
    }
}
BENCHMARK(BM_Greeks_Tolerance_1e6);

static void BM_Greeks_Tolerance_1e8(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL, 1e-8));
    }
}
BENCHMARK(BM_Greeks_Tolerance_1e8);

// ============================================================================
// Accuracy Verification Benchmark
// ============================================================================

static void BM_Greeks_Accuracy_Verification(benchmark::State& state) {
    auto params = create_params();
    
    // Test delta accuracy using finite difference verification
    double h = 0.001;
    
    for (auto _ : state) {
        // Price at S0
        RoughHestonPricer<double> pricer1(params);
        double price1 = pricer1.price_put(1.0);
        
        // Price at S0 + h
        auto params2 = params;
        params2.S0 += h;
        RoughHestonPricer<double> pricer2(params2);
        double price2 = pricer2.price_put(1.0);
        
        // Finite difference delta
        double fd_delta = (price2 - price1) / h;
        
        // CSD delta
        RoughHestonGreeks<double> greeks(params);
        auto result = greeks.compute(1.0, GreekSet::ESSENTIAL);
        
        // Compute error
        double error = std::abs(result.delta - fd_delta) / std::abs(fd_delta);
        benchmark::DoNotOptimize(error);
    }
}
BENCHMARK(BM_Greeks_Accuracy_Verification);

// ============================================================================
// Performance Comparison: CSD vs Finite Differences
// ============================================================================

static void BM_Delta_CSD(benchmark::State& state) {
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(greeks.compute(1.0, GreekSet::ESSENTIAL).delta);
    }
}
BENCHMARK(BM_Delta_CSD);

static void BM_Delta_FiniteDifference(benchmark::State& state) {
    auto params = create_params();
    double h = 0.0001;
    
    for (auto _ : state) {
        RoughHestonPricer<double> pricer1(params);
        double p1 = pricer1.price_put(1.0);
        
        auto params2 = params;
        params2.S0 += h;
        RoughHestonPricer<double> pricer2(params2);
        double p2 = pricer2.price_put(1.0);
        
        double delta = (p2 - p1) / h;
        benchmark::DoNotOptimize(delta);
    }
}
BENCHMARK(BM_Delta_FiniteDifference);

// ============================================================================
// Throughput Benchmark
// ============================================================================

static void BM_Greeks_Throughput(benchmark::State& state) {
    auto params = create_params();
    std::vector<double> strikes(100);
    for (int i = 0; i < 100; ++i) {
        strikes[i] = 0.8 + 0.4 * i / 99.0;  // 0.8 to 1.2
    }
    
    for (auto _ : state) {
        for (double K : strikes) {
            RoughHestonGreeks<double> greeks(params);
            auto result = greeks.compute(K, GreekSet::ESSENTIAL);
            benchmark::DoNotOptimize(result);
        }
    }
    state.SetItemsProcessed(state.iterations() * strikes.size());
}
BENCHMARK(BM_Greeks_Throughput);

// Main function for standalone execution
int main(int argc, char** argv) {
    // First, run accuracy verification
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CHARLTON Greek Calculation Accuracy Verification" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    auto params = create_params();
    RoughHestonGreeks<double> greeks(params);
    auto result = greeks.compute(1.0, GreekSet::CORNUCOPIA);
    
    std::cout << "\nGreek Values (ATM Put, T=1/52):" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << "Price:     " << result.price << std::endl;
    std::cout << "Delta:     " << result.delta << std::endl;
    std::cout << "Gamma:     " << result.gamma << std::endl;
    std::cout << "Theta:     " << result.theta << std::endl;
    std::cout << "Vega:      " << result.vega << std::endl;
    std::cout << "Rho:       " << result.rho << std::endl;
    std::cout << "Vanna:     " << result.vanna << std::endl;
    std::cout << "Volga:     " << result.volga << std::endl;
    std::cout << "Zomma:     " << result.zomma << std::endl;
    std::cout << "Speed:     " << result.speed << std::endl;
    std::cout << "Charm:     " << result.charm << std::endl;
    std::cout << "Color:     " << result.color << std::endl;
    std::cout << "Veta:      " << result.veta << std::endl;
    std::cout << "Roughness: " << result.roughness << std::endl;
    
    // Verify delta with finite differences
    double h = 0.001;
    RoughHestonPricer<double> pricer1(params);
    double p1 = pricer1.price_put(1.0);
    
    auto params2 = params;
    params2.S0 += h;
    RoughHestonPricer<double> pricer2(params2);
    double p2 = pricer2.price_put(1.0);
    
    double fd_delta = (p2 - p1) / h;
    double delta_error = std::abs(result.delta - fd_delta) / std::abs(fd_delta);
    
    std::cout << "\nDelta Verification (Finite Difference h=" << h << "):" << std::endl;
    std::cout << "  CSD Delta:      " << result.delta << std::endl;
    std::cout << "  FD Delta:       " << fd_delta << std::endl;
    std::cout << "  Relative Error: " << delta_error * 100 << "%" << std::endl;
    
    // Verify gamma with finite differences
    auto params3 = params;
    params3.S0 -= h;
    RoughHestonPricer<double> pricer3(params3);
    double p3 = pricer3.price_put(1.0);
    
    double fd_gamma = (p2 - 2*p1 + p3) / (h * h);
    double gamma_error = std::abs(result.gamma - fd_gamma) / std::abs(fd_gamma);
    
    std::cout << "\nGamma Verification (Finite Difference h=" << h << "):" << std::endl;
    std::cout << "  CSD Gamma:      " << result.gamma << std::endl;
    std::cout << "  FD Gamma:       " << fd_gamma << std::endl;
    std::cout << "  Relative Error: " << gamma_error * 100 << "%" << std::endl;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Running Google Benchmark..." << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    
    return 0;
}
