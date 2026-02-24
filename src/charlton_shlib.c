/**
 * CHARLTON Shared Library Wrappers
 *
 * Non-inline wrappers for Python ctypes / shared library usage.
 * Compile: gcc -std=c99 -shared -fPIC -o libcharlton.so charlton_shlib.c
 */
#define CHARLTON_IMPLEMENTATION
#include "charlton.h"

#if defined(__GNUC__) || defined(__clang__)
  #define CHARLTON_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER)
  #define CHARLTON_EXPORT __declspec(dllexport)
#else
  #define CHARLTON_EXPORT
#endif

/* ---- ABM Solver ---- */

CHARLTON_EXPORT int charlton_shlib_abm_init(charlton_abm_solver *s, double H, double T,
                                             size_t N, double gamma_val, double theta_val,
                                             double nu_val, double rho_val) {
    return charlton_abm_init(s, H, T, N, gamma_val, theta_val, nu_val, rho_val);
}

CHARLTON_EXPORT void charlton_shlib_abm_free(charlton_abm_solver *s) {
    charlton_abm_free(s);
}

CHARLTON_EXPORT double charlton_shlib_abm_decay_rate(const charlton_abm_solver *s,
                                                      double T, double v0) {
    return charlton_abm_decay_rate(s, T, v0);
}

/* ---- Pricing ---- */

CHARLTON_EXPORT double charlton_shlib_price_put(const charlton_model_params *params,
                                                 double K, double error_tol) {
    return charlton_price_put(params, K, error_tol);
}

CHARLTON_EXPORT double charlton_shlib_price_call(const charlton_model_params *params,
                                                  double K, double error_tol) {
    return charlton_price_call(params, K, error_tol);
}

CHARLTON_EXPORT double charlton_shlib_price_put_bootstrap(
    const charlton_model_params *params, double K,
    double *error_estimate, double error_tol) {
    return charlton_price_put_bootstrap(params, K, error_estimate, error_tol);
}

/* ---- Implied Volatility ---- */

CHARLTON_EXPORT double charlton_shlib_implied_volatility(
    double price, double S0, double K, double T, double r, int is_call) {
    return charlton_implied_volatility(price, S0, K, T, r, is_call);
}

/* ---- Greeks ---- */

CHARLTON_EXPORT int charlton_shlib_greeks(const charlton_model_params *params, double K,
                                           int greek_set, charlton_pricing_result *result) {
    return charlton_greeks(params, K, greek_set, result);
}

/* ---- Calibration ---- */

CHARLTON_EXPORT int charlton_shlib_calibrate_adam(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes,
    const charlton_calibration_result *initial,
    charlton_calibration_result *result) {
    return charlton_calibrate_adam(cal, quotes, n_quotes, initial, result);
}

CHARLTON_EXPORT int charlton_shlib_calibrate_lbfgs(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes,
    const charlton_calibration_result *initial,
    charlton_calibration_result *result) {
    return charlton_calibrate_lbfgs(cal, quotes, n_quotes, initial, result);
}

CHARLTON_EXPORT charlton_calibration_result charlton_shlib_generate_initial_guess(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes) {
    return charlton_generate_initial_guess(cal, quotes, n_quotes);
}

/* ---- LOB Synthetic ---- */

CHARLTON_EXPORT int charlton_shlib_lob_synth_quotes(
    const charlton_model_params *params,
    size_t n_T, size_t n_K,
    const double *T_grid, const double *K_grid,
    double k_lambda, double alpha_lob, double theta_cancel,
    charlton_market_quote *quotes) {
    return charlton_lob_synth_quotes(params, n_T, n_K, T_grid, K_grid,
                                     k_lambda, alpha_lob, theta_cancel, quotes);
}
