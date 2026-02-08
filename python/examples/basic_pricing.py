"""
CHARLTON Python Example - Basic Pricing

Demonstrates the fluent interface for option pricing.
"""

import charlton as ch

print("=" * 60)
print("CHARLTON Python - Basic Pricing Example")
print("=" * 60)

# =============================================================================
# Example 1: Simple European Put Pricing
# =============================================================================
print("\n1. Simple European Put Pricing")
print("-" * 40)

# Build a model using the fluent interface
model = (ch.model()
    .spot(100.0)
    .rate(0.05)
    .maturity(1.0)
    .hurst(0.1)
    .mean_reversion(2.0)
    .long_term_variance(0.04)
    .vol_of_vol(0.5)
    .correlation(-0.6)
    .initial_variance(0.04)
)

# Create a European pricer
pricer = ch.pricer(model, ch.ExerciseType.EUROPEAN)

# Price options at various strikes
strikes = [90, 95, 100, 105, 110]
print(f"{'Strike':<10} {'Put Price':<15} {'Call Price':<15}")
print("-" * 40)

for K in strikes:
    put_price = pricer.put(K)
    call_price = pricer.call(K)
    print(f"{K:<10} {put_price:<15.6f} {call_price:<15.6f}")

# =============================================================================
# Example 2: Using Predefined Model Configurations
# =============================================================================
print("\n2. Using Predefined Model Configurations")
print("-" * 40)

# El Euch-Rosenbaum calibrated parameters
model_er = ch.model().el_euch_rosenbaum()
pricer_er = ch.pricer(model_er)

print(f"ATM Put (El Euch-Rosenbaum): {pricer_er.put(1.0):.8f}")

# Standard rough Heston parameters
model_std = ch.model().rough_heston_standard()
pricer_std = ch.pricer(model_std)

print(f"ATM Put (Standard): {pricer_std.put(1.0):.8f}")

# =============================================================================
# Example 3: From ATM Implied Volatility
# =============================================================================
print("\n3. Building Model from ATM Implied Volatility")
print("-" * 40)

atm_iv = 0.20  # 20% ATM implied vol
model_from_iv = (ch.model()
    .spot(100.0)
    .rate(0.05)
    .maturity(0.25)  # 3 months
    .from_atm_iv(atm_iv)
)

pricer_iv = ch.pricer(model_from_iv)
print(f"ATM IV: {atm_iv:.2%}")
print(f"ATM Put Price: {pricer_iv.put(100.0):.6f}")
print(f"Implied Vol from Price: {pricer_iv.iv_put(100.0):.2%}")

# =============================================================================
# Example 4: American Options
# =============================================================================
print("\n4. American Option Pricing")
print("-" * 40)

american_pricer = ch.pricer(model, ch.ExerciseType.AMERICAN)

print(f"{'Strike':<10} {'American Put':<15} {'European Put':<15} {'Premium':<15}")
print("-" * 60)

for K in [90, 100, 110]:
    am_price = american_pricer.put(K)
    eu_price = pricer.put(K)
    premium = am_price - eu_price
    print(f"{K:<10} {am_price:<15.6f} {eu_price:<15.6f} {premium:<15.6f}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
