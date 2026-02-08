"""
CHARLTON Python Example - Greeks Calculation

Demonstrates comprehensive Greek calculations using the fluent interface.
"""

import charlton as ch

print("=" * 60)
print("CHARLTON Python - Greeks Example")
print("=" * 60)

# Build a model
model = (ch.model()
    .spot(100.0)
    .rate(0.05)
    .maturity(0.25)  # 3 months
    .hurst(0.1)
    .mean_reversion(2.0)
    .long_term_variance(0.04)
    .vol_of_vol(0.5)
    .correlation(-0.6)
    .initial_variance(0.04)
)

pricer = ch.pricer(model)

# =============================================================================
# Example 1: Essential Greeks
# =============================================================================
print("\n1. Essential Greeks (Delta, Gamma, Theta, Vega, Rho)")
print("-" * 60)

greeks = pricer.greeks_put(100.0, "essential")

print(f"Price:  {greeks['price']:.6f}")
print(f"Delta:  {greeks['delta']:.6f}")
print(f"Gamma:  {greeks['gamma']:.6f}")
print(f"Theta:  {greeks['theta']:.6f} (per year)")
print(f"Vega:   {greeks['vega']:.6f}")
print(f"Rho:    {greeks['rho']:.6f}")

# =============================================================================
# Example 2: Standard Greeks (includes Vanna, Volga)
# =============================================================================
print("\n2. Standard Greeks (+ Vanna, Volga)")
print("-" * 60)

greeks_std = pricer.greeks_put(100.0, "standard")

print(f"Vanna:  {greeks_std['vanna']:.6f}")
print(f"Volga:  {greeks_std['volga']:.6f}")

# =============================================================================
# Example 3: Full Cornucopia Greeks
# =============================================================================
print("\n3. Cornucopia Greeks (Full Set)")
print("-" * 60)

greeks_full = pricer.greeks_put(100.0, "cornucopia")

print(f"Zomma:         {greeks_full['zomma']:.6f}")
print(f"Speed:         {greeks_full['speed']:.6f}")
print(f"Charm:         {greeks_full['charm']:.6f}")
print(f"Color:         {greeks_full['color']:.6f}")
print(f"Veta:          {greeks_full['veta']:.6f}")
print(f"Roughness:     {greeks_full['roughness']:.6f} (dV/dH)")
print(f"Nu Sens:       {greeks_full['nu_sens']:.6f} (dV/dν)")
print(f"Lambda Sens:   {greeks_full['lambda_sens']:.6f} (dV/dλ)")
print(f"Theta Sens:    {greeks_full['theta_sens']:.6f} (dV/dθ)")

# =============================================================================
# Example 4: Delta Profile Across Strikes
# =============================================================================
print("\n4. Delta Profile Across Strikes")
print("-" * 60)

strikes = [80, 90, 95, 100, 105, 110, 120]

print(f"{'Strike':<10} {'Put Delta':<15} {'Call Delta':<15}")
print("-" * 40)

for K in strikes:
    put_greeks = pricer.greeks_put(K, "essential")
    call_greeks = pricer.greeks_call(K, "essential")
    print(f"{K:<10} {put_greeks['delta']:<15.6f} {call_greeks['delta']:<15.6f}")

# =============================================================================
# Example 5: Gamma Profile (peaks at ATM)
# =============================================================================
print("\n5. Gamma Profile (peaks at ATM)")
print("-" * 60)

print(f"{'Strike':<10} {'Gamma':<15}")
print("-" * 25)

for K in strikes:
    g = pricer.greeks_put(K, "essential")
    print(f"{K:<10} {g['gamma']:<15.6f}")

# =============================================================================
# Example 6: American Greeks with Early Exercise Premium
# =============================================================================
print("\n6. American Greeks (with Early Exercise Premium)")
print("-" * 60)

american_pricer = ch.pricer(model, ch.ExerciseType.AMERICAN)
am_greeks = american_pricer.greeks_put(100.0, "essential")

print(f"American Put Price: {am_greeks['price']:.6f}")
print(f"Delta:              {am_greeks['delta']:.6f}")
print(f"Gamma:              {am_greeks['gamma']:.6f}")
print(f"Early Ex. Premium:  {am_greeks.get('early_exercise_premium', 0.0):.6f}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
