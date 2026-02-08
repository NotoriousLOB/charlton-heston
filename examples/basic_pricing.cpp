/**
 * CHARLTON Basic Pricing Example
 * 
 * Demonstrates basic usage of the Rough Heston pricer with SINH-acceleration
 */

#include <iostream>
#include <iomanip>
#include "charlton.hpp"

int main() {
    std::cout << "========================================\n";
    std::cout << "CHARLTON Basic Pricing Example\n";
    std::cout << "========================================\n\n";
    
    // Model parameters (El Euch-Rosenbaum calibrated to S&P 500)
    charlton::RoughHestonPricer<double>::ModelParams params;
    params.S0 = 1.0;           // Spot price
    params.r = 0.0;            // Risk-free rate
    params.q = 0.0;            // Dividend yield
    params.T = 1.0 / 52.0;     // 1 week maturity
    params.H = 0.12;           // Hurst parameter
    params.lambda = 0.1;       // Mean reversion speed
    params.theta = 0.3156;     // Long-term variance
    params.nu = 0.331;         // Vol-of-vol
    params.rho = -0.681;       // Correlation
    params.V0 = 0.0392;        // Initial variance
    
    // Create pricer
    charlton::RoughHestonPricer<double> pricer(params);
    
    // Price put options at various strikes
    std::vector<double> strikes = {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2};
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Put Option Prices (T = 1 week):\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Strike    Price        IV (%)", "\n";
    std::cout << "----------------------------------------\n";
    
    for (double K : strikes) {
        double price = pricer.price_put(K);
        double iv = charlton::RoughHestonPricer<double>::implied_volatility(
            price, params.S0, K, params.T, params.r, false);
        
        std::cout << std::setw(6) << K << "    " 
                  << std::setw(10) << price << "    "
                  << std::setw(6) << iv * 100.0 << "%\n";
    }
    
    std::cout << "\n";
    
    // Demonstrate put-call parity
    double K = 1.0;
    double put_price = pricer.price_put(K);
    double call_price = pricer.price_call(K);
    
    std::cout << "Put-Call Parity Check (K = " << K << "):\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Put Price:  " << put_price << "\n";
    std::cout << "Call Price: " << call_price << "\n";
    
    double fwd = params.S0 * std::exp((params.r - params.q) * params.T);
    double df = std::exp(-params.r * params.T);
    double parity_lhs = call_price - put_price;
    double parity_rhs = fwd - K * df;
    
    std::cout << "C - P = " << parity_lhs << "\n";
    std::cout << "S - K·e^(-rT) = " << parity_rhs << "\n";
    std::cout << "Difference: " << std::abs(parity_lhs - parity_rhs) << "\n";
    
    std::cout << "\n========================================\n";
    std::cout << "Example completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}
