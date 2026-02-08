/**
 * CHARLTON Calibration Example
 * 
 * Demonstrates model calibration to synthetic market data
 */

#include <iostream>
#include <iomanip>
#include "charlton.hpp"

int main() {
    std::cout << "========================================\n";
    std::cout << "CHARLTON Calibration Example\n";
    std::cout << "========================================\n\n";
    
    // True model parameters (what we'll try to recover)
    charlton::RoughHestonPricer<double>::ModelParams true_params;
    true_params.H = 0.1;
    true_params.lambda = 2.0;
    true_params.theta = 0.04;
    true_params.nu = 0.5;
    true_params.rho = -0.6;
    true_params.V0 = 0.04;
    
    // Market setup
    double S0 = 1.0;
    double r = 0.0;
    
    std::cout << "True Model Parameters:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "H      = " << true_params.H << "\n";
    std::cout << "λ      = " << true_params.lambda << "\n";
    std::cout << "θ      = " << true_params.theta << "\n";
    std::cout << "ν      = " << true_params.nu << "\n";
    std::cout << "ρ      = " << true_params.rho << "\n";
    std::cout << "V₀     = " << true_params.V0 << "\n";
    std::cout << "\n";
    
    // Generate synthetic market data
    std::vector<double> maturities = {1.0/52.0, 2.0/52.0};
    std::vector<double> moneyness = {0.9, 0.95, 1.0, 1.05, 1.1};
    
    auto quotes = charlton::generate_test_market_data(
        S0, r, true_params, maturities, moneyness, 0.0
    );
    
    std::cout << "Synthetic Market Data:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Maturity    Strike    IV (%)\n";
    std::cout << "----------------------------------------\n";
    
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& q : quotes) {
        std::cout << std::setw(6) << q.T * 365.0 << "d       "
                  << std::setw(6) << q.K << "    "
                  << std::setw(6) << q.iv * 100.0 << "%\n";
    }
    
    std::cout << "\n";
    
    // Set up calibrator
    charlton::RoughHestonCalibrator<double>::CalibrationParams cal_params;
    cal_params.S0 = S0;
    cal_params.r = r;
    cal_params.q = 0.0;
    cal_params.max_iterations = 200;
    cal_params.tolerance = 1e-5;
    cal_params.step_size = 0.01;
    
    charlton::RoughHestonCalibrator<double> calibrator(cal_params);
    calibrator.add_quotes(quotes);
    
    // Generate initial guess
    auto initial_guess = calibrator.generate_initial_guess();
    
    std::cout << "Initial Guess:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "H      = " << initial_guess.H << "\n";
    std::cout << "λ      = " << initial_guess.lambda << "\n";
    std::cout << "θ      = " << initial_guess.theta << "\n";
    std::cout << "ν      = " << initial_guess.nu << "\n";
    std::cout << "ρ      = " << initial_guess.rho << "\n";
    std::cout << "V₀     = " << initial_guess.V0 << "\n";
    std::cout << "RMSE   = " << initial_guess.rmse << "\n";
    std::cout << "\n";
    
    // Run calibration
    std::cout << "Running calibration...\n";
    std::cout << "----------------------------------------\n";
    
    auto result = calibrator.calibrate_adam(initial_guess);
    
    std::cout << "\nCalibration Results:\n";
    std::cout << "----------------------------------------\n";
    result.print();
    
    std::cout << "\n";
    std::cout << "Parameter Recovery:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Param    True       Calibrated   Error\n";
    std::cout << "----------------------------------------\n";
    
    auto print_comparison = [](const char* name, double true_val, double cal_val) {
        double error = std::abs(true_val - cal_val);
        std::cout << std::setw(6) << name << "    "
                  << std::setw(10) << true_val << "  "
                  << std::setw(10) << cal_val << "  "
                  << std::setw(10) << error << "\n";
    };
    
    print_comparison("H", true_params.H, result.H);
    print_comparison("λ", true_params.lambda, result.lambda);
    print_comparison("θ", true_params.theta, result.theta);
    print_comparison("ν", true_params.nu, result.nu);
    print_comparison("ρ", true_params.rho, result.rho);
    print_comparison("V₀", true_params.V0, result.V0);
    
    std::cout << "\n========================================\n";
    std::cout << "Example completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}
