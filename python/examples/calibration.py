"""
CHARLTON Python Example - Model Calibration

Demonstrates calibration to market implied volatility data.
"""

import charlton as ch
import random

print("=" * 60)
print("CHARLTON Python - Calibration Example")
print("=" * 60)

# =============================================================================
# Example 1: Generate Synthetic Market Data
# =============================================================================
print("\n1. Generating Synthetic Market Data")
print("-" * 60)

# True model parameters (what we'll try to recover)
true_params = {
    'H': 0.1,
    'lambda': 2.0,
    'theta': 0.04,
    'nu': 0.5,
    'rho': -0.6,
    'V0': 0.04
}

print("True Parameters:")
for k, v in true_params.items():
    print(f"  {k:10} = {v}")

# Create a model with true parameters
model = (ch.model()
    .spot(100.0)
    .rate(0.05)
    .hurst(true_params['H'])
    .mean_reversion(true_params['lambda'])
    .long_term_variance(true_params['theta'])
    .vol_of_vol(true_params['nu'])
    .correlation(true_params['rho'])
    .initial_variance(true_params['V0'])
)

# Generate synthetic quotes
pricer = ch.pricer(model)

maturities = [0.25, 0.5, 1.0]  # 3M, 6M, 1Y
strikes = [90, 95, 100, 105, 110]

quotes = []
print("\nSynthetic Market Quotes:")
print(f"{'Maturity':<10} {'Strike':<10} {'IV':<10}")
print("-" * 30)

for T in maturities:
    model_T = (ch.model()
        .spot(100.0)
        .rate(0.05)
        .maturity(T)
        .hurst(true_params['H'])
        .mean_reversion(true_params['lambda'])
        .long_term_variance(true_params['theta'])
        .vol_of_vol(true_params['nu'])
        .correlation(true_params['rho'])
        .initial_variance(true_params['V0'])
    )
    pricer_T = ch.pricer(model_T)
    
    for K in strikes:
        price = pricer_T.put(K)
        iv = pricer_T.iv_put(K)
        quotes.append((T, K, iv))
        print(f"{T:<10.2f} {K:<10.1f} {iv:<10.4%}")

# =============================================================================
# Example 2: Calibrate to Market Data
# =============================================================================
print("\n2. Calibrating to Market Data")
print("-" * 60)

# Create calibrator
calibrator = (ch.calibrator(spot=100.0, rate=0.05)
    .max_iterations(500)
    .tolerance(1e-5)
    .step_size(0.01)
    .add_quotes(quotes)
)

# Generate initial guess
initial_guess = calibrator.generate_initial_guess()
print("\nInitial Guess:")
for k, v in initial_guess.items():
    print(f"  {k:10} = {v:.6f}")

# Run calibration
print("\nRunning calibration...")
result = calibrator.calibrate()

print("\nCalibration Results:")
print(f"{'Param':<10} {'True':<12} {'Calibrated':<12} {'Error':<12}")
print("-" * 50)

for param in ['H', 'lambda', 'theta', 'nu', 'rho', 'V0']:
    true_val = true_params[param]
    cal_val = result[param]
    error = abs(true_val - cal_val)
    print(f"{param:<10} {true_val:<12.6f} {cal_val:<12.6f} {error:<12.6f}")

print(f"\nRMSE: {result['rmse']:.8f}")
print(f"MAE: {result['mae']:.8f}")
print(f"Iterations: {result['iterations']}")
print(f"Converged: {result['converged']}")

# =============================================================================
# Example 3: Calibrate with Custom Initial Guess
# =============================================================================
print("\n3. Calibration with Custom Initial Guess")
print("-" * 60)

custom_guess = {
    'H': 0.15,
    'lambda': 1.5,
    'theta': 0.05,
    'nu': 0.4,
    'rho': -0.5,
    'V0': 0.05
}

result2 = calibrator.calibrate(custom_guess)

print("Results with Custom Initial Guess:")
print(f"RMSE: {result2['rmse']:.8f}")
print(f"Iterations: {result2['iterations']}")

# =============================================================================
# Example 4: Build Pricer from Calibrated Parameters
# =============================================================================
print("\n4. Building Pricer from Calibrated Parameters")
print("-" * 60)

calibrated_model = (ch.model()
    .spot(100.0)
    .rate(0.05)
    .hurst(result['H'])
    .mean_reversion(result['lambda'])
    .long_term_variance(result['theta'])
    .vol_of_vol(result['nu'])
    .correlation(result['rho'])
    .initial_variance(result['V0'])
)

calibrated_pricer = ch.pricer(calibrated_model)

print("Pricing with calibrated model:")
for K in [90, 100, 110]:
    price = calibrated_pricer.put(K)
    iv = calibrated_pricer.iv_put(K)
    print(f"  Strike {K}: Price={price:.6f}, IV={iv:.4%}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
