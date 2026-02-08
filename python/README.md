# CHARLTON Python Bindings

High-performance Python interface for the Rough Heston model with a fluent API.

## Installation

### From Source

```bash
cd charlton
mkdir build && cd build
cmake .. -DCHARLTON_BUILD_PYTHON=ON
make -j$(nproc)

# Install Python package
cd python
pip install -e .
```

### Quick Test

```python
import charlton as ch

# Build a model
model = ch.model().spot(100).rate(0.05).hurst(0.1).from_atm_iv(0.2)

# Create a pricer
pricer = ch.pricer(model)

# Price options
print(f"Put price: {pricer.put(100):.6f}")
print(f"Call price: {pricer.call(100):.6f}")
```

## Fluent Interface

CHARLTON provides a chainable, readable API for building models and pricing options.

### Model Building

```python
import charlton as ch

# Method 1: Explicit parameters
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

# Method 2: From ATM implied volatility
model = (ch.model()
    .spot(100.0)
    .rate(0.05)
    .maturity(0.25)
    .from_atm_iv(0.20)  # 20% ATM IV
)

# Method 3: Predefined configurations
model = ch.model().el_euch_rosenbaum()  # S&P 500 calibrated
model = ch.model().rough_heston_standard()  # Standard parameters
```

### Pricing

```python
# Create pricer
pricer = ch.pricer(model)  # European (default)
american_pricer = ch.pricer(model, ch.ExerciseType.AMERICAN)

# Price options
put_price = pricer.put(100.0)
call_price = pricer.call(100.0)

# Implied volatility
iv_put = pricer.iv_put(100.0)
iv_call = pricer.iv_call(100.0)
```

### Greeks

```python
# Essential Greeks (Delta, Gamma, Theta, Vega, Rho)
greeks = pricer.greeks_put(100.0, "essential")
print(f"Delta: {greeks['delta']}")

# Standard Greeks (+ Vanna, Volga)
greeks = pricer.greeks_put(100.0, "standard")

# Full Cornucopia (+ Zomma, Speed, Charm, Color, Veta, Roughness, etc.)
greeks = pricer.greeks_put(100.0, "cornucopia")
print(f"Roughness (dV/dH): {greeks['roughness']}")
```

### Calibration

```python
# Create calibrator
calibrator = (ch.calibrator(spot=100.0, rate=0.05)
    .max_iterations(500)
    .tolerance(1e-5)
    .add_quote(T=0.25, K=95, iv=0.22)
    .add_quote(T=0.25, K=100, iv=0.20)
    .add_quote(T=0.25, K=105, iv=0.19)
)

# Calibrate
result = calibrator.calibrate()

print(f"H: {result['H']}")
print(f"lambda: {result['lambda']}")
print(f"RMSE: {result['rmse']}")
```

## API Reference

### `charlton.model()`

Creates a new `ModelBuilder` with fluent interface.

**Methods:**
- `spot(S0)` - Spot price
- `rate(r)` - Risk-free rate
- `dividend(q)` - Dividend yield
- `maturity(T)` - Time to maturity
- `hurst(H)` - Hurst parameter (0 < H < 0.5)
- `mean_reversion(lambda)` - Mean reversion speed
- `long_term_variance(theta)` - Long-term variance
- `vol_of_vol(nu)` - Volatility of volatility
- `correlation(rho)` - Spot-vol correlation
- `initial_variance(V0)` - Initial variance
- `from_atm_iv(iv)` - Set theta and V0 from ATM IV
- `rough_heston_standard()` - Standard parameter preset
- `el_euch_rosenbaum()` - S&P 500 calibrated preset

### `charlton.pricer(model, exercise_type)`

Creates a pricer from a model.

**Parameters:**
- `model` - ModelBuilder instance
- `exercise_type` - `ExerciseType.EUROPEAN` (default) or `ExerciseType.AMERICAN`

**Methods:**
- `put(strike, error_tol=1e-10)` - Price put option
- `call(strike, error_tol=1e-10)` - Price call option
- `greeks_put(strike, greek_set="standard")` - Compute put Greeks
- `greeks_call(strike, greek_set="standard")` - Compute call Greeks
- `iv_put(strike)` - Compute put implied volatility
- `iv_call(strike)` - Compute call implied volatility
- `price_surface(strikes, maturities, is_call=False)` - Price grid

### `charlton.calibrator(spot, rate, dividend=0.0)`

Creates a model calibrator.

**Methods:**
- `max_iterations(n)` - Set max iterations
- `tolerance(tol)` - Set convergence tolerance
- `step_size(s)` - Set Adam step size
- `add_quote(T, K, iv, is_call=False)` - Add market quote
- `add_quotes(list)` - Add multiple quotes
- `calibrate(initial_guess={})` - Run calibration
- `generate_initial_guess()` - Get heuristic initial guess

### Greek Sets

- `"price_only"` - Price only
- `"essential"` - Delta, Gamma, Theta, Vega, Rho
- `"standard"` - Essential + Vanna, Volga
- `"cornucopia"` - Standard + Zomma, Speed, Charm, Color, Veta, Roughness, Nu, Lambda, Theta sensitivities

## Examples

See the `examples/` directory for complete examples:

- `basic_pricing.py` - Simple pricing examples
- `greeks.py` - Comprehensive Greek calculations
- `calibration.py` - Model calibration workflow

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single price | ~1-5 ms | European, SINH-accelerated |
| Essential Greeks | ~10-20 ms | Complex Step Differentiation |
| Cornucopia Greeks | ~50-100 ms | All sensitivities |
| American price | ~50-200 ms | FFT-accelerated PDE solver |
| Calibration (100 quotes) | ~10-30 s | Adam optimizer |

## License

MIT License - see LICENSE file for details.

## Backronym

**CHARLTON** = **C**onformal **H**yperbolic **A**ccelerated **R**ough **L**évy **T**ransform for **O**ption **N**umerics

> *"You can have my Bermudan swaptions when you pry them from my COLD DEAD HAND!!!"*
> — Charlton (Rough) Heston
