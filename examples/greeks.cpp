/**
 * CHARLTON Greeks Example
 * 
 * Demonstrates comprehensive Greek calculations
 */

#include <iostream>
#include <iomanip>
#include "charlton.hpp"

int main() {
    std::cout << "========================================\n";
    std::cout << "CHARLTON Greeks Example\n";
    std::cout << "========================================\n\n";
    
    // Model parameters
    charlton::RoughHestonPricer<double>::ModelParams params;
    params.S0 = 1.0;
    params.r = 0.0;
    params.q = 0.0;
    params.T = 1.0 / 52.0;  // 1 week
    params.H = 0.12;
    params.lambda = 0.1;
    params.theta = 0.3156;
    params.nu = 0.331;
    params.rho = -0.681;
    params.V0 = 0.0392;
    
    double K = 1.0;  // ATM strike
    
    // Create Greek calculator
    charlton::RoughHestonGreeks<double> greeks(params);
    
    std::cout << std::fixed << std::setprecision(8);
    
    // Compute Essential Greeks
    std::cout << "Essential Greeks (ATM Put, T = 1 week):\n";
    std::cout << "----------------------------------------\n";
    
    auto essential = greeks.compute(K, charlton::GreekSet::ESSENTIAL);
    std::cout << "Price:  " << essential.price << "\n";
    std::cout << "Delta:  " << essential.delta << "\n";
    std::cout << "Gamma:  " << essential.gamma << "\n";
    std::cout << "Theta:  " << essential.theta << " (per year)\n";
    std::cout << "Vega:   " << essential.vega << " (per 1% vol)\n";
    std::cout << "Rho:    " << essential.rho << " (per 1% rate)\n";
    
    std::cout << "\n";
    
    // Compute Standard Greeks (includes Vanna, Volga)
    std::cout << "Standard Greeks (+ Vanna, Volga):\n";
    std::cout << "----------------------------------------\n";
    
    auto standard = greeks.compute(K, charlton::GreekSet::STANDARD);
    std::cout << "Vanna:  " << standard.vanna << "\n";
    std::cout << "Volga:  " << standard.volga << "\n";
    
    std::cout << "\n";
    
    // Compute Full Cornucopia Greeks
    std::cout << "Cornucopia Greeks (Full Set):\n";
    std::cout << "----------------------------------------\n";
    
    auto cornucopia = greeks.compute(K, charlton::GreekSet::CORNUCOPIA);
    std::cout << "Zomma:         " << cornucopia.zomma << "\n";
    std::cout << "Speed:         " << cornucopia.speed << "\n";
    std::cout << "Charm:         " << cornucopia.charm << "\n";
    std::cout << "Color:         " << cornucopia.color << "\n";
    std::cout << "Veta:          " << cornucopia.veta << "\n";
    std::cout << "Roughness:     " << cornucopia.roughness << " (dV/dH)\n";
    std::cout << "Nu Sens:       " << cornucopia.nu_sens << " (dV/dν)\n";
    std::cout << "Lambda Sens:   " << cornucopia.lambda_sens << " (dV/dλ)\n";
    std::cout << "Theta Sens:    " << cornucopia.theta_sens << " (dV/dθ)\n";
    
    std::cout << "\n";
    
    // Delta vs Strike profile
    std::cout << "Delta Profile:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Strike    Delta\n";
    std::cout << "----------------------------------------\n";
    
    std::vector<double> strikes = {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2};
    for (double strike : strikes) {
        auto result = greeks.compute(strike, charlton::GreekSet::ESSENTIAL);
        std::cout << std::setw(6) << strike << "    " 
                  << std::setw(10) << result.delta << "\n";
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Example completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}
