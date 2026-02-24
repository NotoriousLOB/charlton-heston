"""
CHARLTON - Conformal Hyperbolic Accelerated Rough Levy Transform for Option Numerics

Python bindings via ctypes to the C99 shared library.
"""

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

# --- Load shared library ---

def _find_library():
    """Find libcharlton.so in common locations."""
    # Check relative to this file (build directory)
    here = Path(__file__).parent
    candidates = [
        here / ".." / ".." / "build" / "libcharlton_shared.so",
        here / ".." / ".." / "build" / "libcharlton_shared.dylib",
        here / ".." / ".." / "build" / "charlton_shared.dll",
        here / ".." / "build" / "libcharlton_shared.so",
        Path("build") / "libcharlton_shared.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    # Try system paths
    found = ctypes.util.find_library("charlton_shared")
    if found:
        return found

    raise RuntimeError(
        "Cannot find libcharlton_shared. Build with: "
        "cmake -B build && cmake --build build"
    )

_lib = ctypes.CDLL(_find_library())

# --- C struct definitions ---

class ModelParams(ctypes.Structure):
    _fields_ = [
        ("S0", ctypes.c_double),
        ("r", ctypes.c_double),
        ("q", ctypes.c_double),
        ("T", ctypes.c_double),
        ("H", ctypes.c_double),
        ("lambda_", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("nu", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("V0", ctypes.c_double),
    ]

    def __repr__(self):
        return (f"ModelParams(S0={self.S0}, r={self.r}, q={self.q}, T={self.T}, "
                f"H={self.H}, lambda={self.lambda_}, theta={self.theta}, "
                f"nu={self.nu}, rho={self.rho}, V0={self.V0})")


class PricingResult(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("vega", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("vanna", ctypes.c_double),
        ("volga", ctypes.c_double),
        ("zomma", ctypes.c_double),
        ("speed", ctypes.c_double),
        ("charm", ctypes.c_double),
        ("color", ctypes.c_double),
        ("veta", ctypes.c_double),
        ("roughness", ctypes.c_double),
        ("nu_sens", ctypes.c_double),
        ("lambda_sens", ctypes.c_double),
        ("theta_sens", ctypes.c_double),
    ]

    def to_dict(self):
        return {f: getattr(self, f) for f, _ in self._fields_}

    def __repr__(self):
        return f"PricingResult(price={self.price}, delta={self.delta}, gamma={self.gamma})"


class MarketQuote(ctypes.Structure):
    _fields_ = [
        ("T", ctypes.c_double),
        ("K", ctypes.c_double),
        ("iv", ctypes.c_double),
        ("is_call", ctypes.c_int),
    ]


class CalibrationResult(ctypes.Structure):
    _fields_ = [
        ("H", ctypes.c_double),
        ("lambda_", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("nu", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("V0", ctypes.c_double),
        ("rmse", ctypes.c_double),
        ("mae", ctypes.c_double),
        ("iterations", ctypes.c_int),
        ("converged", ctypes.c_int),
    ]

    def __repr__(self):
        return (f"CalibrationResult(H={self.H}, lambda={self.lambda_}, "
                f"theta={self.theta}, nu={self.nu}, rho={self.rho}, V0={self.V0}, "
                f"rmse={self.rmse}, converged={bool(self.converged)})")


class CalibrationParams(ctypes.Structure):
    _fields_ = [
        ("S0", ctypes.c_double),
        ("r", ctypes.c_double),
        ("q", ctypes.c_double),
        ("max_iterations", ctypes.c_int),
        ("tolerance", ctypes.c_double),
        ("step_size", ctypes.c_double),
    ]


# --- Function signatures ---

# Pricing
_lib.charlton_shlib_price_put.restype = ctypes.c_double
_lib.charlton_shlib_price_put.argtypes = [
    ctypes.POINTER(ModelParams), ctypes.c_double, ctypes.c_double
]

_lib.charlton_shlib_price_call.restype = ctypes.c_double
_lib.charlton_shlib_price_call.argtypes = [
    ctypes.POINTER(ModelParams), ctypes.c_double, ctypes.c_double
]

_lib.charlton_shlib_price_put_bootstrap.restype = ctypes.c_double
_lib.charlton_shlib_price_put_bootstrap.argtypes = [
    ctypes.POINTER(ModelParams), ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.c_double
]

# Implied Vol
_lib.charlton_shlib_implied_volatility.restype = ctypes.c_double
_lib.charlton_shlib_implied_volatility.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_int
]

# Greeks
_lib.charlton_shlib_greeks.restype = ctypes.c_int
_lib.charlton_shlib_greeks.argtypes = [
    ctypes.POINTER(ModelParams), ctypes.c_double,
    ctypes.c_int, ctypes.POINTER(PricingResult)
]

# Calibration
_lib.charlton_shlib_calibrate_adam.restype = ctypes.c_int
_lib.charlton_shlib_calibrate_adam.argtypes = [
    ctypes.POINTER(CalibrationParams),
    ctypes.POINTER(MarketQuote), ctypes.c_size_t,
    ctypes.POINTER(CalibrationResult),
    ctypes.POINTER(CalibrationResult)
]

_lib.charlton_shlib_calibrate_lbfgs.restype = ctypes.c_int
_lib.charlton_shlib_calibrate_lbfgs.argtypes = [
    ctypes.POINTER(CalibrationParams),
    ctypes.POINTER(MarketQuote), ctypes.c_size_t,
    ctypes.POINTER(CalibrationResult),
    ctypes.POINTER(CalibrationResult)
]

_lib.charlton_shlib_generate_initial_guess.restype = CalibrationResult
_lib.charlton_shlib_generate_initial_guess.argtypes = [
    ctypes.POINTER(CalibrationParams),
    ctypes.POINTER(MarketQuote), ctypes.c_size_t
]

# --- Greek set constants ---
GREEKS_PRICE_ONLY = 0
GREEKS_ESSENTIAL = 1
GREEKS_STANDARD = 2
GREEKS_CORNUCOPIA = 3


# --- Python API ---

def make_params(S0=1.0, r=0.0, q=0.0, T=1.0, H=0.12, lambda_=0.1,
                theta=0.3156, nu=0.331, rho=-0.681, V0=0.0392):
    """Create model parameters."""
    p = ModelParams()
    p.S0 = S0; p.r = r; p.q = q; p.T = T; p.H = H
    p.lambda_ = lambda_; p.theta = theta; p.nu = nu; p.rho = rho; p.V0 = V0
    return p


def price_put(params, K, error_tol=1e-10):
    """Price a European put option."""
    return _lib.charlton_shlib_price_put(ctypes.byref(params), K, error_tol)


def price_call(params, K, error_tol=1e-10):
    """Price a European call option (via put-call parity)."""
    return _lib.charlton_shlib_price_call(ctypes.byref(params), K, error_tol)


def price_put_bootstrap(params, K, error_tol=1e-10):
    """Price put with Conformal Bootstrap error estimate. Returns (price, error_estimate)."""
    err = ctypes.c_double(0.0)
    price = _lib.charlton_shlib_price_put_bootstrap(
        ctypes.byref(params), K, ctypes.byref(err), error_tol)
    return price, err.value


def implied_volatility(price, S0, K, T, r, is_call=False):
    """Compute implied volatility from option price."""
    return _lib.charlton_shlib_implied_volatility(
        price, S0, K, T, r, int(is_call))


def greeks(params, K, greek_set=GREEKS_ESSENTIAL):
    """Compute Greeks. Returns PricingResult."""
    result = PricingResult()
    _lib.charlton_shlib_greeks(ctypes.byref(params), K, greek_set, ctypes.byref(result))
    return result


def calibrate_adam(cal_params, quotes, initial_guess):
    """Calibrate using Adam optimizer. Returns CalibrationResult."""
    n = len(quotes)
    QuoteArray = MarketQuote * n
    q_arr = QuoteArray()
    for i, q in enumerate(quotes):
        q_arr[i].T = q[0]
        q_arr[i].K = q[1]
        q_arr[i].iv = q[2]
        q_arr[i].is_call = int(q[3]) if len(q) > 3 else 0

    result = CalibrationResult()
    _lib.charlton_shlib_calibrate_adam(
        ctypes.byref(cal_params), q_arr, n,
        ctypes.byref(initial_guess), ctypes.byref(result))
    return result


def calibrate_lbfgs(cal_params, quotes, initial_guess):
    """Calibrate using L-BFGS-B optimizer. Returns CalibrationResult."""
    n = len(quotes)
    QuoteArray = MarketQuote * n
    q_arr = QuoteArray()
    for i, q in enumerate(quotes):
        q_arr[i].T = q[0]
        q_arr[i].K = q[1]
        q_arr[i].iv = q[2]
        q_arr[i].is_call = int(q[3]) if len(q) > 3 else 0

    result = CalibrationResult()
    _lib.charlton_shlib_calibrate_lbfgs(
        ctypes.byref(cal_params), q_arr, n,
        ctypes.byref(initial_guess), ctypes.byref(result))
    return result


def make_cal_params(S0=1.0, r=0.0, q=0.0, max_iterations=100,
                    tolerance=1e-4, step_size=0.01):
    """Create calibration parameters."""
    p = CalibrationParams()
    p.S0 = S0; p.r = r; p.q = q
    p.max_iterations = max_iterations
    p.tolerance = tolerance; p.step_size = step_size
    return p


def make_initial_guess(H=0.1, lambda_=2.0, theta=0.04, nu=0.5, rho=-0.7, V0=0.04):
    """Create initial guess for calibration."""
    g = CalibrationResult()
    g.H = H; g.lambda_ = lambda_; g.theta = theta
    g.nu = nu; g.rho = rho; g.V0 = V0
    g.rmse = 0.0; g.mae = 0.0; g.iterations = 0; g.converged = 0
    return g
