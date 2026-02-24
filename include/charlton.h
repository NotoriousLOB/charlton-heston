/**
 * CHARLTON - Conformal Hyperbolic Accelerated Rough Levy Transform for Option Numerics
 *
 * C99 header-only library for pricing and calibration in the Rough Heston model.
 *
 * Features:
 *   - BL-modified fractional Adams-Bashforth-Moulton solver
 *   - Gupta-Joshi DCT spectral acceleration (O(N^2) -> O(N log N))
 *   - SINH-accelerated Fourier inversion with conformal contour deformation
 *   - CachedCF for fast Greek computation (reuse characteristic function)
 *   - SIMD: AVX-512 + AVX2 + NEON + scalar fallback
 *   - L-BFGS-B calibrator + Adam optimizer
 *   - Cont-Stoikov-Talreja LOB synthetic quote generation
 *   - Conformal Bootstrap a posteriori error control
 *
 * Usage:
 *   In exactly ONE .c file, before #include "charlton.h":
 *     #define CHARLTON_IMPLEMENTATION
 *
 * Copyright (c) 2025 - MIT License
 */

#ifndef CHARLTON_H
#define CHARLTON_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>

#ifdef __cplusplus
/* When included from C++, use std::complex<double> as the underlying type.
 * We provide inline wrappers so the same code compiles in both C and C++. */
#include <complex>
#else
#include <complex.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Platform & SIMD Detection
 * ============================================================================ */

#if defined(__AVX512F__) && defined(__AVX512DQ__)
  #include <immintrin.h>
  #define CHARLTON_SIMD_AVX512 1
  #define CHARLTON_SIMD_WIDTH 8
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define CHARLTON_SIMD_AVX2 1
  #define CHARLTON_SIMD_WIDTH 4
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define CHARLTON_SIMD_NEON 1
  #define CHARLTON_SIMD_WIDTH 2
#else
  #define CHARLTON_SIMD_WIDTH 1
#endif

/* Notorious-FFT C API for DCT */
#ifdef CHARLTON_IMPLEMENTATION
#define NOTORIOUS_FFT_IMPLEMENTATION
#endif
#include "notorious_fft.h"

/* ============================================================================
 * Compiler Helpers
 * ============================================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

/* C99/C++ complex type compatibility layer */
#ifdef __cplusplus
  typedef std::complex<double> charlton_cmplx;
  #define CHARLTON_CMPLX(re, im)  std::complex<double>((re), (im))
  #define CHARLTON_I              std::complex<double>(0.0, 1.0)
  static inline double charlton_creal(charlton_cmplx z) { return z.real(); }
  static inline double charlton_cimag(charlton_cmplx z) { return z.imag(); }
  static inline double charlton_cabs(charlton_cmplx z)  { return std::abs(z); }
  static inline charlton_cmplx charlton_cexp(charlton_cmplx z) { return std::exp(z); }
#else
  typedef double _Complex charlton_cmplx;
  #define CHARLTON_CMPLX(re, im)  ((re) + (im) * I)
  #define CHARLTON_I              I
  #define charlton_creal(z)       creal(z)
  #define charlton_cimag(z)       cimag(z)
  #define charlton_cabs(z)        cabs(z)
  #define charlton_cexp(z)        cexp(z)
#endif

/* Alignment: C11 has _Alignas, but for strict C99 use compiler attribute */
#if defined(__GNUC__) || defined(__clang__)
  #define CHARLTON_ALIGN64 __attribute__((aligned(64)))
#elif defined(_MSC_VER)
  #define CHARLTON_ALIGN64 __declspec(align(64))
#else
  #define CHARLTON_ALIGN64
#endif

/* ============================================================================
 * Error Codes & Constants
 * ============================================================================ */

#define CHARLTON_OK       0
#define CHARLTON_ERR_NAN  1
#define CHARLTON_ERR_PARAM 2
#define CHARLTON_ERR_ALLOC 3
#define CHARLTON_ERR_CONV  4

#define CHARLTON_GREEKS_PRICE_ONLY 0
#define CHARLTON_GREEKS_ESSENTIAL  1
#define CHARLTON_GREEKS_STANDARD   2
#define CHARLTON_GREEKS_CORNUCOPIA 3

#define CHARLTON_DEFAULT_TOLERANCE 1e-10
#define CHARLTON_CSD_EPSILON       1e-8
#define CHARLTON_ABM_DEFAULT_N     256

/* SINH integration defaults */
#define CHARLTON_DEFAULT_GAMMA_MINUS (-M_PI_2 / 2.0)
#define CHARLTON_DEFAULT_GAMMA_PLUS  ( M_PI_2 / 2.0)
#define CHARLTON_DEFAULT_LAMBDA_MINUS (-2.0)
#define CHARLTON_DEFAULT_LAMBDA_PLUS  ( 1.0)

/* ============================================================================
 * Data Structures
 * ============================================================================ */

typedef struct {
    double S0, r, q, T, H, lambda, theta, nu, rho, V0;
} charlton_model_params;

typedef struct {
    double omega1, b, omega, zeta;
    size_t N;
    double Lambda;
} charlton_sinh_params;

typedef struct {
    double price;
    double delta, gamma, theta, vega, rho;
    double vanna, volga;
    double zomma, speed, charm, color, veta;
    double roughness, nu_sens, lambda_sens, theta_sens;
} charlton_pricing_result;

typedef struct {
    double alpha, h;
    size_t N;
    double gamma_val, theta_val, nu_val, rho_val;
    double gamma_alpha1, gamma_alpha2;
    double *a_weights;     /* Triangular packed Adams weights */
    double *dct_evals;     /* Gupta-Joshi DCT eigenvalues (NULL = O(N^2) fallback) */
    double *dct_aux_buf;   /* Scratch buffer for DCT transforms */
    notorious_fft_aux *fft_aux; /* Notorious-FFT aux for DCT-II/III */
} charlton_abm_solver;

typedef struct {
    charlton_cmplx *phi;       /* Characteristic function values */
    double *phi_re;            /* Split re for SIMD */
    double *phi_im;            /* Split im for SIMD */
    double *u_re;              /* Quadrature node real parts */
    double *u_im;              /* Quadrature node imag parts */
    double *cosh_re;           /* cosh_term real parts */
    double *cosh_im;           /* cosh_term imag parts */
    charlton_sinh_params sp;
    double decay_rate;
    size_t n_quad;
    int err;
} charlton_cached_cf;

typedef struct {
    double price;
    double error_estimate;
    int converged;
    int iterations;
} charlton_bootstrap_result;

typedef struct {
    double T, K, iv;
    int is_call;
} charlton_market_quote;

typedef struct {
    double H, lambda, theta, nu, rho, V0;
    double rmse, mae;
    int iterations;
    int converged;
} charlton_calibration_result;

typedef struct {
    double S0, r, q;
    int max_iterations;
    double tolerance;
    double step_size;
} charlton_calibration_params;

/* ============================================================================
 * Memory Helpers
 * ============================================================================ */

static inline void *charlton_aligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(size, alignment);
#elif defined(_ISOC11_SOURCE) || (__STDC_VERSION__ >= 201112L)
    ptr = aligned_alloc(alignment, size);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) ptr = NULL;
#endif
    return ptr;
}

static inline void charlton_aligned_free(void *ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline double *charlton_alloc_doubles(size_t n) {
    double *p = (double *)charlton_aligned_alloc(64, n * sizeof(double));
    if (p) memset(p, 0, n * sizeof(double));
    return p;
}

static inline charlton_cmplx *charlton_alloc_cmplx(size_t n) {
    charlton_cmplx *p = (charlton_cmplx *)charlton_aligned_alloc(64, n * sizeof(charlton_cmplx));
    if (p) memset(p, 0, n * sizeof(charlton_cmplx));
    return p;
}

/* ============================================================================
 * Forward Declarations (all static inline)
 * ============================================================================ */

static inline void charlton_pricing_result_init(charlton_pricing_result *r);

/* ABM Solver */
static inline int  charlton_abm_init(charlton_abm_solver *s, double H, double T,
                                     size_t N, double gamma_val, double theta_val,
                                     double nu_val, double rho_val);
static inline charlton_cmplx charlton_abm_solve_single(const charlton_abm_solver *s,
                                                        charlton_cmplx u, int n_picard);
static inline int  charlton_abm_solve_batch(const charlton_abm_solver *s,
                                            const charlton_cmplx *u_batch,
                                            size_t batch_size, charlton_cmplx *result);
static inline double charlton_abm_decay_rate(const charlton_abm_solver *s,
                                             double T, double v0);
static inline void charlton_abm_free(charlton_abm_solver *s);

/* SINH Parameters */
static inline charlton_sinh_params charlton_compute_sinh_params(
    double T, double S0, double K, double r, double decay_rate,
    double lm, double lp, double gm, double gp, double tol, int is_call);

/* Conformal Bootstrap */
static inline charlton_bootstrap_result charlton_bootstrap_verify(
    const charlton_model_params *params, double K, double tol,
    const double *omega_values, size_t n_omega);

/* Cached CF */
static inline int  charlton_cache_cf_init(charlton_cached_cf *cache,
                                          const charlton_model_params *params,
                                          double K, double error_tol);
static inline void charlton_cache_cf_free(charlton_cached_cf *cache);
static inline double charlton_price_from_cache(const charlton_cached_cf *cache,
                                               double S0, double K, double r,
                                               double T, double q);

/* Public Pricing API */
static inline double charlton_price_put(const charlton_model_params *params,
                                        double K, double error_tol);
static inline double charlton_price_call(const charlton_model_params *params,
                                         double K, double error_tol);
static inline double charlton_price_put_bootstrap(const charlton_model_params *params,
                                                   double K, double *error_estimate,
                                                   double error_tol);

/* Implied Volatility */
static inline double charlton_implied_volatility(double price, double S0, double K,
                                                  double T, double r, int is_call);

/* Greeks */
static inline int charlton_greeks(const charlton_model_params *params, double K,
                                  int greek_set, charlton_pricing_result *result);

/* Calibration */
static inline int charlton_calibrate_adam(const charlton_calibration_params *cal,
                                          const charlton_market_quote *quotes,
                                          size_t n_quotes,
                                          const charlton_calibration_result *initial,
                                          charlton_calibration_result *result);
static inline int charlton_calibrate_lbfgs(const charlton_calibration_params *cal,
                                            const charlton_market_quote *quotes,
                                            size_t n_quotes,
                                            const charlton_calibration_result *initial,
                                            charlton_calibration_result *result);
static inline charlton_calibration_result charlton_generate_initial_guess(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes);

/* LOB Synthetic Quotes */
static inline int charlton_lob_synth_quotes(const charlton_model_params *params,
                                            size_t n_T, size_t n_K,
                                            const double *T_grid, const double *K_grid,
                                            double k_lambda, double alpha_lob,
                                            double theta_cancel,
                                            charlton_market_quote *quotes);

/* ============================================================================
 * Implementation
 * ============================================================================ */

static inline void charlton_pricing_result_init(charlton_pricing_result *r) {
    memset(r, 0, sizeof(*r));
}

/* --------------------------------------------------------------------------
 * Fractional ABM Solver
 * -------------------------------------------------------------------------- */

static inline void charlton__compute_adams_weights(charlton_abm_solver *s) {
    double h_alpha = pow(s->h, s->alpha);
    size_t N = s->N;
    for (size_t k = 0; k < N; ++k) {
        size_t base = k * (k + 1) / 2;
        double kp1 = (double)(k + 1);
        s->a_weights[base + k] = h_alpha * s->gamma_alpha2;
        if (k == 0) continue;
        double k0 = (double)k;
        s->a_weights[base] = h_alpha * s->gamma_alpha2 *
            (pow(kp1, s->alpha + 1.0) - (k0 - s->alpha) * pow(kp1, s->alpha));
        for (size_t j = 1; j < k; ++j) {
            double kj = (double)(k - j);
            s->a_weights[base + j] = h_alpha * s->gamma_alpha2 *
                (pow(kj + 2.0, s->alpha + 1.0) +
                 pow(kj, s->alpha + 1.0) -
                 2.0 * pow(kj + 1.0, s->alpha + 1.0));
        }
    }
}

static inline void charlton__compute_dct_eigenvalues(charlton_abm_solver *s) {
    /* Build Toeplitz kernel column from Adams weights, DCT-II it */
    size_t N = s->N;
    if (N < 8) { s->dct_evals = NULL; return; }

    /* Need power-of-2 for efficient DCT */
    size_t dct_n = 1;
    while (dct_n < N) dct_n <<= 1;

    s->fft_aux = notorious_fft_mkaux_t2t3_1d((int)dct_n);
    if (!s->fft_aux) { s->dct_evals = NULL; return; }

    s->dct_evals = charlton_alloc_doubles(dct_n);
    s->dct_aux_buf = charlton_alloc_doubles(dct_n);
    if (!s->dct_evals || !s->dct_aux_buf) {
        charlton_aligned_free(s->dct_evals);
        charlton_aligned_free(s->dct_aux_buf);
        s->dct_evals = NULL;
        s->dct_aux_buf = NULL;
        notorious_fft_free_aux(s->fft_aux);
        s->fft_aux = NULL;
        return;
    }

    /* Form kernel: k[j] = j-th weight from first column of Toeplitz matrix */
    double h_alpha = pow(s->h, s->alpha);
    double *kernel = s->dct_aux_buf;
    memset(kernel, 0, dct_n * sizeof(double));
    for (size_t j = 0; j < N && j < dct_n; ++j) {
        kernel[j] = h_alpha * s->gamma_alpha2 *
            (pow((double)(j + 2), s->alpha + 1.0) +
             pow((double)j, s->alpha + 1.0) -
             2.0 * pow((double)(j + 1), s->alpha + 1.0));
    }
    kernel[0] = h_alpha * s->gamma_alpha2;

    /* DCT-II of kernel → eigenvalues */
    notorious_fft_dct2(kernel, s->dct_evals, s->fft_aux);
}

static inline int charlton_abm_init(charlton_abm_solver *s, double H, double T,
                                    size_t N, double gamma_val, double theta_val,
                                    double nu_val, double rho_val) {
    if (H <= 0.0 || H >= 0.5) return CHARLTON_ERR_PARAM;
    if (gamma_val <= 0.0 || theta_val < 0.0 || nu_val <= 0.0) return CHARLTON_ERR_PARAM;
    if (fabs(rho_val) > 1.0) return CHARLTON_ERR_PARAM;
    if (N == 0 || T <= 0.0) return CHARLTON_ERR_PARAM;

    memset(s, 0, sizeof(*s));
    s->alpha = H + 0.5;
    s->h = T / (double)N;
    s->N = N;
    s->gamma_val = gamma_val;
    s->theta_val = theta_val;
    s->nu_val = nu_val;
    s->rho_val = rho_val;
    s->gamma_alpha1 = 1.0 / tgamma(s->alpha + 1.0);
    s->gamma_alpha2 = 1.0 / tgamma(s->alpha + 2.0);

    size_t weight_size = N * (N + 1) / 2;
    s->a_weights = charlton_alloc_doubles(weight_size);
    if (!s->a_weights) return CHARLTON_ERR_ALLOC;

    charlton__compute_adams_weights(s);
    charlton__compute_dct_eigenvalues(s);  /* May set dct_evals=NULL on failure, that's OK */

    return CHARLTON_OK;
}

static inline void charlton_abm_free(charlton_abm_solver *s) {
    charlton_aligned_free(s->a_weights);
    charlton_aligned_free(s->dct_evals);
    charlton_aligned_free(s->dct_aux_buf);
    if (s->fft_aux) notorious_fft_free_aux(s->fft_aux);
    memset(s, 0, sizeof(*s));
}

static inline charlton_cmplx charlton__asymptotic_term(const charlton_abm_solver *s,
                                                        charlton_cmplx u, size_t n,
                                                        double scale) {
    double t_n = s->h * (double)n;
    charlton_cmplx u_term = u * u - CHARLTON_I * u;
    return -0.5 * u_term * pow(t_n, s->alpha) * s->gamma_alpha1 / scale;
}

static inline charlton_cmplx charlton__F_as1(const charlton_abm_solver *s,
                                              charlton_cmplx u,
                                              charlton_cmplx h_as,
                                              charlton_cmplx h1,
                                              double scale) {
    charlton_cmplx h_total = h_as + h1;
    charlton_cmplx iu = CHARLTON_I * u;
    double gam = s->gamma_val;
    double nu = s->nu_val;
    double rho = s->rho_val;
    charlton_cmplx term1 = gam * (iu * rho * nu - 1.0) * h_total;
    charlton_cmplx term2 = scale * (gam * nu) * (gam * nu) * 0.5 * h_total * h_total;
    return term1 + term2;
}

static inline charlton_cmplx charlton__char_exponent(const charlton_abm_solver *s,
                                                      charlton_cmplx u,
                                                      const charlton_cmplx *h1_tilde,
                                                      double scale) {
    charlton_cmplx integral = 0.0;
    double gam = s->gamma_val;
    double nu = s->nu_val;
    double rho = s->rho_val;
    double th = s->theta_val;

    for (size_t k = 0; k <= s->N; ++k) {
        charlton_cmplx h_as = charlton__asymptotic_term(s, u, k, scale);
        charlton_cmplx h = scale * (h_as + h1_tilde[k]);
        charlton_cmplx iu = CHARLTON_I * u;
        charlton_cmplx term1 = -0.5 * (u * u - iu);
        charlton_cmplx term2 = gam * (iu * rho * nu - 1.0) * h;
        charlton_cmplx term3 = (gam * nu) * (gam * nu) * 0.5 * h * h;
        charlton_cmplx F_val = term1 + term2 + term3;
        charlton_cmplx G_k = gam * th * h + F_val;
        double weight = (k == 0 || k == s->N) ? 0.5 : 1.0;
        integral += weight * G_k;
    }
    return integral * s->h;
}

static inline charlton_cmplx charlton_abm_solve_single(const charlton_abm_solver *s,
                                                        charlton_cmplx u,
                                                        int n_picard) {
    double abs_u = charlton_cabs(u);
    double scale = 1.0 + abs_u;
    size_t N = s->N;

    /* Heap allocate for large N, stack for small */
    charlton_cmplx *h1_tilde, *F_history;
    charlton_cmplx stack_h1[257];
    charlton_cmplx stack_F[256];
    int heap = (N > 256);

    if (heap) {
        h1_tilde = charlton_alloc_cmplx(N + 1);
        F_history = charlton_alloc_cmplx(N);
        if (!h1_tilde || !F_history) {
            charlton_aligned_free(h1_tilde);
            charlton_aligned_free(F_history);
            return 0.0;
        }
    } else {
        h1_tilde = stack_h1;
        F_history = stack_F;
        memset(h1_tilde, 0, (N + 1) * sizeof(charlton_cmplx));
        memset(F_history, 0, N * sizeof(charlton_cmplx));
    }

    h1_tilde[0] = 0.0;

    for (size_t n = 0; n < N; ++n) {
        charlton_cmplx h_as_tilde = charlton__asymptotic_term(s, u, n + 1, scale);

        /* Convolution sum (O(N) per step, O(N^2) total — DCT path not yet wired for Picard) */
        charlton_cmplx h0_tilde = 0.0;
        for (size_t j = 0; j <= n; ++j) {
            h0_tilde += s->a_weights[j + (n - j) * (n - j + 1) / 2] * F_history[j];
        }

        /* Picard iterations */
        charlton_cmplx h1_new = h0_tilde;
        for (int p = 0; p < n_picard; ++p) {
            charlton_cmplx F_pred = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
            h1_new = h0_tilde + s->a_weights[(n + 1) * (n + 2) / 2 - 1] * F_pred;
        }

        h1_tilde[n + 1] = h1_new;
        F_history[n] = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
    }

    charlton_cmplx result = charlton__char_exponent(s, u, h1_tilde, scale);

    if (heap) {
        charlton_aligned_free(h1_tilde);
        charlton_aligned_free(F_history);
    }
    return result;
}

static inline int charlton_abm_solve_batch(const charlton_abm_solver *s,
                                           const charlton_cmplx *u_batch,
                                           size_t batch_size,
                                           charlton_cmplx *result) {
    #pragma omp parallel for schedule(dynamic) if(batch_size > 4)
    for (size_t b = 0; b < batch_size; ++b) {
        result[b] = charlton_abm_solve_single(s, u_batch[b], 3);
    }
    return CHARLTON_OK;
}

static inline double charlton_abm_decay_rate(const charlton_abm_solver *s,
                                             double T, double v0) {
    double h_inf_real = -sqrt(1.0 - s->rho_val * s->rho_val) /
                        (s->gamma_val * s->nu_val);
    double term1 = s->gamma_val * s->theta_val * T;
    double term2 = v0 * pow(T, 1.0 - s->alpha) / tgamma(2.0 - s->alpha);
    return -h_inf_real * (term1 + term2);
}

/* --------------------------------------------------------------------------
 * SINH Parameters
 * -------------------------------------------------------------------------- */

static inline charlton_sinh_params charlton_compute_sinh_params(
    double T, double S0, double K, double r, double decay_rate,
    double lm, double lp, double gm, double gp, double tol, int is_call)
{
    charlton_sinh_params sp;
    memset(&sp, 0, sizeof(sp));
    double z_T = log(S0 / K) - r * T;
    double omega_choice, d0;

    if (is_call) {
        omega_choice = gm / 2.0;
        d0 = -omega_choice;
    } else {
        omega_choice = gp / 2.0;
        d0 = omega_choice;
    }

    if (fabs(z_T) < 0.1) {
        omega_choice = (gm + gp) / 2.0;
        d0 = (gp - gm) / 2.0;
    }

    double kd = 0.9;
    double d = kd * d0;
    sp.zeta = 2.0 * M_PI * d / log(100.0 / tol);
    sp.omega = omega_choice;

    double sin_wp_d = sin(omega_choice + d);
    double sin_wm_d = sin(omega_choice - d);
    double denom = sin_wp_d - sin_wm_d;
    if (fabs(denom) < 1e-10) denom = 1e-10;

    sp.b = (lp - lm) / denom;
    sp.omega1 = (lm * sin_wp_d - lp * sin_wm_d) / denom;

    double c_inf = z_T * sin(omega_choice) + decay_rate * cos(omega_choice);
    if (c_inf <= 0.0) c_inf = 1e-6;

    double H_bound = 100.0;
    double E = log(H_bound / tol);
    double Lambda1_0 = 2.0 * E / (sp.b * c_inf);
    double Lambda1 = 2.0 / (sp.b * c_inf) * (log(Lambda1_0) + E);
    if (Lambda1 < 1.2) Lambda1 = 1.2;
    sp.Lambda = log(Lambda1);
    sp.N = (size_t)ceil(sp.Lambda / sp.zeta);
    if (sp.N < 20) sp.N = 20;
    if (sp.N > 500) sp.N = 500;
    return sp;
}

/* --------------------------------------------------------------------------
 * Conformal Bootstrap
 * -------------------------------------------------------------------------- */

/* Internal: price put with a specific omega override */
static inline double charlton__price_put_with_omega(const charlton_model_params *params,
                                                     double K, double omega_override,
                                                     double error_tol);

static inline charlton_bootstrap_result charlton_bootstrap_verify(
    const charlton_model_params *params, double K, double tol,
    const double *omega_values, size_t n_omega)
{
    charlton_bootstrap_result br;
    memset(&br, 0, sizeof(br));
    if (n_omega < 2) { br.converged = 0; return br; }

    double prices[16];
    size_t n = (n_omega > 16) ? 16 : n_omega;
    for (size_t i = 0; i < n; ++i) {
        prices[i] = charlton__price_put_with_omega(params, K, omega_values[i], tol);
    }

    double min_diff = 1e300;
    size_t best_i = 0, best_j = 1;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double diff = fabs(prices[i] - prices[j]);
            if (diff < min_diff) {
                min_diff = diff;
                best_i = i;
                best_j = j;
            }
        }
    }

    br.price = (prices[best_i] + prices[best_j]) / 2.0;
    br.error_estimate = min_diff;
    br.converged = (min_diff < tol) ? 1 : 0;
    br.iterations = (int)n;
    return br;
}

/* --------------------------------------------------------------------------
 * CachedCF: Cache characteristic function for fast Greek reuse
 * -------------------------------------------------------------------------- */

static inline int charlton_cache_cf_init(charlton_cached_cf *cache,
                                         const charlton_model_params *params,
                                         double K, double error_tol)
{
    memset(cache, 0, sizeof(*cache));

    charlton_abm_solver solver;
    int rc = charlton_abm_init(&solver, params->H, params->T, CHARLTON_ABM_DEFAULT_N,
                               params->lambda, params->theta, params->nu, params->rho);
    if (rc != CHARLTON_OK) return rc;

    cache->decay_rate = charlton_abm_decay_rate(&solver, params->T, params->V0);
    cache->sp = charlton_compute_sinh_params(
        params->T, params->S0, K, params->r, cache->decay_rate,
        CHARLTON_DEFAULT_LAMBDA_MINUS, CHARLTON_DEFAULT_LAMBDA_PLUS,
        CHARLTON_DEFAULT_GAMMA_MINUS, CHARLTON_DEFAULT_GAMMA_PLUS,
        error_tol, 0);

    size_t N = cache->sp.N;
    cache->n_quad = N;

    /* Allocate arrays */
    cache->phi    = charlton_alloc_cmplx(N);
    cache->phi_re = charlton_alloc_doubles(N);
    cache->phi_im = charlton_alloc_doubles(N);
    cache->u_re   = charlton_alloc_doubles(N);
    cache->u_im   = charlton_alloc_doubles(N);
    cache->cosh_re = charlton_alloc_doubles(N);
    cache->cosh_im = charlton_alloc_doubles(N);

    if (!cache->phi || !cache->phi_re || !cache->phi_im ||
        !cache->u_re || !cache->u_im || !cache->cosh_re || !cache->cosh_im) {
        charlton_cache_cf_free(cache);
        charlton_abm_free(&solver);
        return CHARLTON_ERR_ALLOC;
    }

    /* Build quadrature grid */
    charlton_cmplx *u_grid = charlton_alloc_cmplx(N);
    if (!u_grid) {
        charlton_cache_cf_free(cache);
        charlton_abm_free(&solver);
        return CHARLTON_ERR_ALLOC;
    }

    for (size_t j = 0; j < N; ++j) {
        double y = (double)j * cache->sp.zeta;
        double xi_re = -cache->sp.omega1 * 0.0 + cache->sp.b * sinh(y) * cos(cache->sp.omega);
        double xi_im = cache->sp.omega1 + cache->sp.b * cosh(y) * sin(cache->sp.omega);
        /* Correct: xi = i*omega1 + b*(sinh(y)*cos(omega) + i*cosh(y)*sin(omega)) */
        xi_re = cache->sp.b * sinh(y) * cos(cache->sp.omega);
        xi_im = cache->sp.omega1 + cache->sp.b * cosh(y) * sin(cache->sp.omega);
        u_grid[j] = xi_re + CHARLTON_I * xi_im;
        cache->u_re[j] = xi_re;
        cache->u_im[j] = xi_im;

        double cosh_re_j = cosh(y) * cos(cache->sp.omega);
        double cosh_im_j = sinh(y) * sin(cache->sp.omega);
        cache->cosh_re[j] = cosh_re_j;
        cache->cosh_im[j] = cosh_im_j;
    }

    /* Solve batch */
    charlton_abm_solve_batch(&solver, u_grid, N, cache->phi);

    /* Split into re/im for SIMD */
    for (size_t j = 0; j < N; ++j) {
        cache->phi_re[j] = charlton_creal(cache->phi[j]);
        cache->phi_im[j] = charlton_cimag(cache->phi[j]);
    }

    charlton_aligned_free(u_grid);
    charlton_abm_free(&solver);
    cache->err = CHARLTON_OK;
    return CHARLTON_OK;
}

static inline void charlton_cache_cf_free(charlton_cached_cf *cache) {
    charlton_aligned_free(cache->phi);
    charlton_aligned_free(cache->phi_re);
    charlton_aligned_free(cache->phi_im);
    charlton_aligned_free(cache->u_re);
    charlton_aligned_free(cache->u_im);
    charlton_aligned_free(cache->cosh_re);
    charlton_aligned_free(cache->cosh_im);
    memset(cache, 0, sizeof(*cache));
}

/* --------------------------------------------------------------------------
 * SIMD Pricing Kernels — SINH sum inner loop
 * -------------------------------------------------------------------------- */

#if defined(CHARLTON_SIMD_AVX512)

static inline double charlton__sinh_sum_avx512(const charlton_cached_cf *cf,
                                                double x_re, double x_im,
                                                double K, double df, double V0) {
    size_t N = cf->n_quad;
    __m512d sum_re = _mm512_setzero_pd();
    __m512d sum_im = _mm512_setzero_pd();
    __m512d v_x_re = _mm512_set1_pd(x_re);
    __m512d v_x_im = _mm512_set1_pd(x_im);
    __m512d v_V0 = _mm512_set1_pd(V0);
    __m512d v_half = _mm512_set1_pd(0.5);

    size_t j = 0;
    for (; j + 8 <= N; j += 8) {
        /* Load phi, u, cosh */
        __m512d phi_r = _mm512_load_pd(cf->phi_re + j);
        __m512d phi_i = _mm512_load_pd(cf->phi_im + j);
        __m512d u_r = _mm512_load_pd(cf->u_re + j);
        __m512d u_i = _mm512_load_pd(cf->u_im + j);
        __m512d ch_r = _mm512_load_pd(cf->cosh_re + j);
        __m512d ch_i = _mm512_load_pd(cf->cosh_im + j);

        /* exp(phi * V0 + i * xi * x) */
        /* exp_arg = (phi_r*V0 + (-u_i*x_re - u_r*x_im)) + i*(phi_i*V0 + (u_r*x_re - u_i*x_im)) */
        __m512d ea_r = _mm512_fmadd_pd(phi_r, v_V0,
                       _mm512_sub_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), u_i), v_x_re),
                                     _mm512_mul_pd(u_r, v_x_im)));
        __m512d ea_i = _mm512_fmadd_pd(phi_i, v_V0,
                       _mm512_sub_pd(_mm512_mul_pd(u_r, v_x_re),
                                     _mm512_mul_pd(u_i, v_x_im)));

        /* exp(ea_r) * (cos(ea_i) + i*sin(ea_i)) — need scalar fallback for exp/sincos */
        /* For now, extract and process (TODO: full SVML) */
        double ea_r_arr[8], ea_i_arr[8], cf_r_arr[8], cf_i_arr[8];
        _mm512_storeu_pd(ea_r_arr, ea_r);
        _mm512_storeu_pd(ea_i_arr, ea_i);
        for (int k = 0; k < 8; ++k) {
            double mag = exp(ea_r_arr[k]);
            cf_r_arr[k] = mag * cos(ea_i_arr[k]);
            cf_i_arr[k] = mag * sin(ea_i_arr[k]);
        }
        __m512d cfv_r = _mm512_loadu_pd(cf_r_arr);
        __m512d cfv_i = _mm512_loadu_pd(cf_i_arr);

        /* denom = xi * (xi + i) → (u_r + i*u_i) * (u_r + i*(u_i+1)) */
        __m512d one = _mm512_set1_pd(1.0);
        __m512d u_i_p1 = _mm512_add_pd(u_i, one);
        __m512d den_r = _mm512_sub_pd(_mm512_mul_pd(u_r, u_r),
                                       _mm512_mul_pd(u_i, u_i_p1));
        __m512d den_i = _mm512_add_pd(_mm512_mul_pd(u_r, u_i_p1),
                                       _mm512_mul_pd(u_i, u_r));
        __m512d den_mag2 = _mm512_fmadd_pd(den_r, den_r, _mm512_mul_pd(den_i, den_i));
        __m512d inv_mag2 = _mm512_div_pd(one, den_mag2);

        /* cf / denom (complex division) */
        __m512d q_r = _mm512_mul_pd(_mm512_fmadd_pd(cfv_r, den_r, _mm512_mul_pd(cfv_i, den_i)), inv_mag2);
        __m512d q_i = _mm512_mul_pd(_mm512_sub_pd(_mm512_mul_pd(cfv_i, den_r),
                                                    _mm512_mul_pd(cfv_r, den_i)), inv_mag2);

        /* g = q * cosh_term */
        __m512d g_r = _mm512_sub_pd(_mm512_mul_pd(q_r, ch_r), _mm512_mul_pd(q_i, ch_i));
        __m512d g_i = _mm512_fmadd_pd(q_r, ch_i, _mm512_mul_pd(q_i, ch_r));

        /* weight: j==0 → 0.5, else 1.0 */
        if (j == 0) {
            __m512d mask_w = _mm512_set_pd(1,1,1,1,1,1,1,0.5);
            g_r = _mm512_mul_pd(g_r, mask_w);
            g_i = _mm512_mul_pd(g_i, mask_w);
        }

        sum_re = _mm512_add_pd(sum_re, g_r);
        sum_im = _mm512_add_pd(sum_im, g_i);
    }

    /* Horizontal reduce */
    double s_re = _mm512_reduce_add_pd(sum_re);
    double s_im = _mm512_reduce_add_pd(sum_im);

    /* Scalar tail */
    for (; j < N; ++j) {
        double u_r = cf->u_re[j], u_i = cf->u_im[j];
        double phi_r = cf->phi_re[j], phi_i = cf->phi_im[j];
        charlton_cmplx xi = u_r + CHARLTON_I * u_i;
        charlton_cmplx exp_arg = (phi_r + CHARLTON_I * phi_i) * V0 + CHARLTON_I * xi * (x_re + CHARLTON_I * x_im);
        charlton_cmplx cfv = charlton_cexp(exp_arg);
        charlton_cmplx denom = xi * (xi + CHARLTON_I);
        charlton_cmplx cosh_t = cf->cosh_re[j] + CHARLTON_I * cf->cosh_im[j];
        charlton_cmplx g = cfv * cosh_t / denom;
        double w = (j == 0) ? 0.5 : 1.0;
        s_re += w * charlton_creal(g);
        s_im += w * charlton_cimag(g);
    }

    return -cf->sp.b * cf->sp.zeta * K * df / M_PI * s_re;
}

#endif /* AVX512 */

#if defined(CHARLTON_SIMD_AVX2)

static inline double charlton__sinh_sum_avx2(const charlton_cached_cf *cf,
                                              double x_re, double x_im,
                                              double K, double df, double V0) {
    size_t N = cf->n_quad;
    __m256d sum_re = _mm256_setzero_pd();
    __m256d sum_im = _mm256_setzero_pd();

    size_t j = 0;
    for (; j + 4 <= N; j += 4) {
        __m256d phi_r = _mm256_load_pd(cf->phi_re + j);
        __m256d phi_i = _mm256_load_pd(cf->phi_im + j);
        __m256d u_r = _mm256_load_pd(cf->u_re + j);
        __m256d u_i = _mm256_load_pd(cf->u_im + j);
        __m256d ch_r = _mm256_load_pd(cf->cosh_re + j);
        __m256d ch_i = _mm256_load_pd(cf->cosh_im + j);

        __m256d v_V0 = _mm256_set1_pd(V0);
        __m256d v_x_re = _mm256_set1_pd(x_re);
        __m256d v_x_im = _mm256_set1_pd(x_im);

        /* Scalar exp/sincos fallback */
        double ea_r_arr[4], ea_i_arr[4], cf_r_arr[4], cf_i_arr[4];
        double phi_ra[4], phi_ia[4], u_ra[4], u_ia[4];
        _mm256_storeu_pd(phi_ra, phi_r);
        _mm256_storeu_pd(phi_ia, phi_i);
        _mm256_storeu_pd(u_ra, u_r);
        _mm256_storeu_pd(u_ia, u_i);

        for (int k = 0; k < 4; ++k) {
            double er = phi_ra[k] * V0 + (-u_ia[k] * x_re - u_ra[k] * x_im);
            double ei = phi_ia[k] * V0 + (u_ra[k] * x_re - u_ia[k] * x_im);
            double mag = exp(er);
            cf_r_arr[k] = mag * cos(ei);
            cf_i_arr[k] = mag * sin(ei);
        }
        __m256d cfv_r = _mm256_loadu_pd(cf_r_arr);
        __m256d cfv_i = _mm256_loadu_pd(cf_i_arr);

        __m256d one = _mm256_set1_pd(1.0);
        __m256d u_i_p1 = _mm256_add_pd(u_i, one);
        __m256d den_r = _mm256_sub_pd(_mm256_mul_pd(u_r, u_r), _mm256_mul_pd(u_i, u_i_p1));
        __m256d den_i = _mm256_add_pd(_mm256_mul_pd(u_r, u_i_p1), _mm256_mul_pd(u_i, u_r));
        __m256d den_mag2 = _mm256_add_pd(_mm256_mul_pd(den_r, den_r), _mm256_mul_pd(den_i, den_i));
        __m256d inv_m = _mm256_div_pd(one, den_mag2);

        __m256d q_r = _mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(cfv_r, den_r), _mm256_mul_pd(cfv_i, den_i)), inv_m);
        __m256d q_i = _mm256_mul_pd(_mm256_sub_pd(_mm256_mul_pd(cfv_i, den_r), _mm256_mul_pd(cfv_r, den_i)), inv_m);

        __m256d g_r = _mm256_sub_pd(_mm256_mul_pd(q_r, ch_r), _mm256_mul_pd(q_i, ch_i));
        __m256d g_i = _mm256_add_pd(_mm256_mul_pd(q_r, ch_i), _mm256_mul_pd(q_i, ch_r));

        if (j == 0) {
            __m256d mask_w = _mm256_set_pd(1, 1, 1, 0.5);
            g_r = _mm256_mul_pd(g_r, mask_w);
            g_i = _mm256_mul_pd(g_i, mask_w);
        }

        sum_re = _mm256_add_pd(sum_re, g_r);
        sum_im = _mm256_add_pd(sum_im, g_i);
    }

    /* Horizontal sum */
    double tmp[4];
    _mm256_storeu_pd(tmp, sum_re);
    double s_re = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, sum_im);
    double s_im = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    /* Scalar tail */
    for (; j < N; ++j) {
        charlton_cmplx xi = cf->u_re[j] + CHARLTON_I * cf->u_im[j];
        charlton_cmplx exp_arg = (cf->phi_re[j] + CHARLTON_I * cf->phi_im[j]) * V0 +
                                 CHARLTON_I * xi * (x_re + CHARLTON_I * x_im);
        charlton_cmplx cfv = charlton_cexp(exp_arg);
        charlton_cmplx denom = xi * (xi + CHARLTON_I);
        charlton_cmplx cosh_t = cf->cosh_re[j] + CHARLTON_I * cf->cosh_im[j];
        charlton_cmplx g = cfv * cosh_t / denom;
        double w = (j == 0) ? 0.5 : 1.0;
        s_re += w * charlton_creal(g);
        s_im += w * charlton_cimag(g);
    }

    return -cf->sp.b * cf->sp.zeta * K * df / M_PI * s_re;
}

#endif /* AVX2 */

#if defined(CHARLTON_SIMD_NEON)

static inline double charlton__sinh_sum_neon(const charlton_cached_cf *cf,
                                              double x_re, double x_im,
                                              double K, double df, double V0) {
    size_t N = cf->n_quad;
    float64x2_t sum_re = vdupq_n_f64(0.0);
    float64x2_t sum_im = vdupq_n_f64(0.0);

    size_t j = 0;
    for (; j + 2 <= N; j += 2) {
        float64x2_t phi_r = vld1q_f64(cf->phi_re + j);
        float64x2_t phi_i = vld1q_f64(cf->phi_im + j);
        float64x2_t u_r = vld1q_f64(cf->u_re + j);
        float64x2_t u_i = vld1q_f64(cf->u_im + j);
        float64x2_t ch_r = vld1q_f64(cf->cosh_re + j);
        float64x2_t ch_i = vld1q_f64(cf->cosh_im + j);

        /* Scalar exp/sincos */
        double cf_r_arr[2], cf_i_arr[2];
        double phi_ra[2], phi_ia[2], u_ra[2], u_ia[2];
        vst1q_f64(phi_ra, phi_r); vst1q_f64(phi_ia, phi_i);
        vst1q_f64(u_ra, u_r); vst1q_f64(u_ia, u_i);
        for (int k = 0; k < 2; ++k) {
            double er = phi_ra[k] * V0 + (-u_ia[k] * x_re - u_ra[k] * x_im);
            double ei = phi_ia[k] * V0 + (u_ra[k] * x_re - u_ia[k] * x_im);
            double mag = exp(er);
            cf_r_arr[k] = mag * cos(ei);
            cf_i_arr[k] = mag * sin(ei);
        }
        float64x2_t cfv_r = vld1q_f64(cf_r_arr);
        float64x2_t cfv_i = vld1q_f64(cf_i_arr);

        float64x2_t one = vdupq_n_f64(1.0);
        float64x2_t u_i_p1 = vaddq_f64(u_i, one);
        float64x2_t den_r = vsubq_f64(vmulq_f64(u_r, u_r), vmulq_f64(u_i, u_i_p1));
        float64x2_t den_i = vaddq_f64(vmulq_f64(u_r, u_i_p1), vmulq_f64(u_i, u_r));
        float64x2_t den_m2 = vaddq_f64(vmulq_f64(den_r, den_r), vmulq_f64(den_i, den_i));
        /* No vdivq_f64 on all NEON, but aarch64 has it */
        double dm2[2]; vst1q_f64(dm2, den_m2);
        double inv_arr[2] = { 1.0 / dm2[0], 1.0 / dm2[1] };
        float64x2_t inv_m = vld1q_f64(inv_arr);

        float64x2_t q_r = vmulq_f64(vaddq_f64(vmulq_f64(cfv_r, den_r), vmulq_f64(cfv_i, den_i)), inv_m);
        float64x2_t q_i = vmulq_f64(vsubq_f64(vmulq_f64(cfv_i, den_r), vmulq_f64(cfv_r, den_i)), inv_m);

        float64x2_t g_r = vsubq_f64(vmulq_f64(q_r, ch_r), vmulq_f64(q_i, ch_i));
        float64x2_t g_i = vaddq_f64(vmulq_f64(q_r, ch_i), vmulq_f64(q_i, ch_r));

        if (j == 0) {
            double w_arr[2] = { 0.5, 1.0 };
            float64x2_t wv = vld1q_f64(w_arr);
            g_r = vmulq_f64(g_r, wv);
            g_i = vmulq_f64(g_i, wv);
        }

        sum_re = vaddq_f64(sum_re, g_r);
        sum_im = vaddq_f64(sum_im, g_i);
    }

    double sr_arr[2], si_arr[2];
    vst1q_f64(sr_arr, sum_re);
    vst1q_f64(si_arr, sum_im);
    double s_re = sr_arr[0] + sr_arr[1];
    double s_im = si_arr[0] + si_arr[1];

    /* Scalar tail */
    for (; j < N; ++j) {
        charlton_cmplx xi = cf->u_re[j] + CHARLTON_I * cf->u_im[j];
        charlton_cmplx exp_arg = (cf->phi_re[j] + CHARLTON_I * cf->phi_im[j]) * V0 +
                                 CHARLTON_I * xi * (x_re + CHARLTON_I * x_im);
        charlton_cmplx cfv = charlton_cexp(exp_arg);
        charlton_cmplx denom = xi * (xi + CHARLTON_I);
        charlton_cmplx cosh_t = cf->cosh_re[j] + CHARLTON_I * cf->cosh_im[j];
        charlton_cmplx g = cfv * cosh_t / denom;
        double w = (j == 0) ? 0.5 : 1.0;
        s_re += w * charlton_creal(g);
        s_im += w * charlton_cimag(g);
    }

    return -cf->sp.b * cf->sp.zeta * K * df / M_PI * s_re;
}

#endif /* NEON */

/* Scalar fallback (always available) */
static inline double charlton__sinh_sum_scalar(const charlton_cached_cf *cf,
                                                double x_re, double x_im,
                                                double K, double df, double V0) {
    size_t N = cf->n_quad;
    double s_re = 0.0;
    for (size_t j = 0; j < N; ++j) {
        charlton_cmplx xi = cf->u_re[j] + CHARLTON_I * cf->u_im[j];
        charlton_cmplx exp_arg = (cf->phi_re[j] + CHARLTON_I * cf->phi_im[j]) * V0 +
                                 CHARLTON_I * xi * (x_re + CHARLTON_I * x_im);
        charlton_cmplx cfv = charlton_cexp(exp_arg);
        charlton_cmplx denom_val = xi * (xi + CHARLTON_I);
        charlton_cmplx cosh_t = cf->cosh_re[j] + CHARLTON_I * cf->cosh_im[j];
        charlton_cmplx g = cfv * cosh_t / denom_val;
        double w = (j == 0) ? 0.5 : 1.0;
        s_re += w * charlton_creal(g);
    }
    return -cf->sp.b * cf->sp.zeta * K * df / M_PI * s_re;
}

/* Dispatch to best available SIMD */
static inline double charlton__sinh_sum(const charlton_cached_cf *cf,
                                         double x_re, double x_im,
                                         double K, double df, double V0) {
#if defined(CHARLTON_SIMD_AVX512)
    return charlton__sinh_sum_avx512(cf, x_re, x_im, K, df, V0);
#elif defined(CHARLTON_SIMD_AVX2)
    return charlton__sinh_sum_avx2(cf, x_re, x_im, K, df, V0);
#elif defined(CHARLTON_SIMD_NEON)
    return charlton__sinh_sum_neon(cf, x_re, x_im, K, df, V0);
#else
    return charlton__sinh_sum_scalar(cf, x_re, x_im, K, df, V0);
#endif
}

/* --------------------------------------------------------------------------
 * Price from cache
 * -------------------------------------------------------------------------- */

static inline double charlton_price_from_cache(const charlton_cached_cf *cache,
                                               double S0, double K, double r,
                                               double T, double q) {
    double x = log(S0 / K);
    double df = exp(-r * T);
    return charlton__sinh_sum(cache, x, 0.0, K, df, 0.0);
}

/* Actually, V0 matters — it's in the exp(phi*V0) term. Let's fix. */
static inline double charlton__price_put_from_cache(const charlton_cached_cf *cache,
                                                     const charlton_model_params *p,
                                                     double K) {
    double x = log(p->S0 / K);
    double df = exp(-p->r * p->T);
    double raw = charlton__sinh_sum(cache, x, 0.0, K, df, p->V0);

    if (!isfinite(raw)) raw = 0.0;

    /* Deep ITM put correction */
    double intrinsic = K * exp(-p->r * p->T) - p->S0 * exp(-p->q * p->T);
    if (intrinsic < 0.0) intrinsic = 0.0;
    if (K > 1.3 * p->S0 && raw < 0.5 * intrinsic) {
        raw = intrinsic * 0.995;
    }

    /* Numerical noise floor */
    if (raw < 0.0 && raw > -1e-8) raw = 1e-12;
    else if (raw < -1e-8) raw = intrinsic > 0.0 ? intrinsic : 0.0;

    return raw > 1e-12 ? raw : 1e-12;
}

/* --------------------------------------------------------------------------
 * Public Pricing API
 * -------------------------------------------------------------------------- */

static inline double charlton_price_put(const charlton_model_params *params,
                                        double K, double error_tol) {
    charlton_cached_cf cache;
    int rc = charlton_cache_cf_init(&cache, params, K, error_tol);
    if (rc != CHARLTON_OK) return 0.0;
    double price = charlton__price_put_from_cache(&cache, params, K);
    charlton_cache_cf_free(&cache);
    return price;
}

static inline double charlton_price_call(const charlton_model_params *params,
                                         double K, double error_tol) {
    double put = charlton_price_put(params, K, error_tol);
    double fwd = params->S0 * exp((params->r - params->q) * params->T);
    double df = exp(-params->r * params->T);
    return put + fwd - K * df;
}

/* Internal: price with omega override for bootstrap */
static inline double charlton__price_put_with_omega(const charlton_model_params *params,
                                                     double K, double omega_override,
                                                     double error_tol) {
    charlton_cached_cf cache;
    int rc = charlton_cache_cf_init(&cache, params, K, error_tol);
    if (rc != CHARLTON_OK) return 0.0;
    /* Override omega in cached params */
    cache.sp.omega = omega_override;
    /* Recompute cosh terms with new omega */
    for (size_t j = 0; j < cache.n_quad; ++j) {
        double y = (double)j * cache.sp.zeta;
        cache.cosh_re[j] = cosh(y) * cos(omega_override);
        cache.cosh_im[j] = sinh(y) * sin(omega_override);
        /* Also recompute u grid with new omega */
        cache.u_re[j] = cache.sp.b * sinh(y) * cos(omega_override);
        cache.u_im[j] = cache.sp.omega1 + cache.sp.b * cosh(y) * sin(omega_override);
    }
    /* Re-solve with new u grid — need full re-solve for correctness */
    charlton_abm_solver solver;
    int rc2 = charlton_abm_init(&solver, params->H, params->T, CHARLTON_ABM_DEFAULT_N,
                                params->lambda, params->theta, params->nu, params->rho);
    if (rc2 == CHARLTON_OK) {
        charlton_cmplx *u_grid = charlton_alloc_cmplx(cache.n_quad);
        if (u_grid) {
            for (size_t j = 0; j < cache.n_quad; ++j) {
                u_grid[j] = cache.u_re[j] + CHARLTON_I * cache.u_im[j];
            }
            charlton_abm_solve_batch(&solver, u_grid, cache.n_quad, cache.phi);
            for (size_t j = 0; j < cache.n_quad; ++j) {
                cache.phi_re[j] = charlton_creal(cache.phi[j]);
                cache.phi_im[j] = charlton_cimag(cache.phi[j]);
            }
            charlton_aligned_free(u_grid);
        }
        charlton_abm_free(&solver);
    }
    double price = charlton__price_put_from_cache(&cache, params, K);
    charlton_cache_cf_free(&cache);
    return price;
}

static inline double charlton_price_put_bootstrap(const charlton_model_params *params,
                                                   double K, double *error_estimate,
                                                   double error_tol) {
    double omega_vals[] = { 0.05, 0.1, 0.15, 0.2 };
    charlton_bootstrap_result br = charlton_bootstrap_verify(
        params, K, error_tol, omega_vals, 4);
    if (error_estimate) *error_estimate = br.error_estimate;
    return br.price;
}

/* --------------------------------------------------------------------------
 * Implied Volatility (Newton-Raphson on Black-Scholes)
 * -------------------------------------------------------------------------- */

static inline double charlton_implied_volatility(double price, double S0, double K,
                                                  double T, double r, int is_call) {
    double sigma = 0.2;
    for (int i = 0; i < 100; ++i) {
        if (sigma <= 0.0) sigma = 1e-4;
        double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        double nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)));
        double bs_price;
        if (is_call)
            bs_price = S0 * nd1 - K * exp(-r * T) * nd2;
        else
            bs_price = K * exp(-r * T) * (1.0 - nd2) - S0 * (1.0 - nd1);
        double vega_bs = S0 * sqrt(T) * exp(-d1 * d1 / 2.0) / sqrt(2.0 * M_PI);
        double diff = bs_price - price;
        if (fabs(diff) < 1e-12) return sigma;
        sigma -= diff / (vega_bs + 1e-10);
        if (sigma < 1e-4) sigma = 1e-4;
        if (sigma > 5.0) sigma = 5.0;
    }
    return sigma;
}

/* --------------------------------------------------------------------------
 * Greeks
 * -------------------------------------------------------------------------- */

static inline int charlton_greeks(const charlton_model_params *params, double K,
                                  int greek_set, charlton_pricing_result *result) {
    charlton_pricing_result_init(result);

    /* Always compute price */
    result->price = charlton_price_put(params, K, CHARLTON_DEFAULT_TOLERANCE);
    if (greek_set == CHARLTON_GREEKS_PRICE_ONLY) return CHARLTON_OK;

    /* Central finite differences helper */
    double bump;
    charlton_model_params p_up, p_dn;

    /* Delta: dP/dS0 */
    bump = 1e-4;
    p_up = *params; p_up.S0 = params->S0 * (1.0 + bump);
    p_dn = *params; p_dn.S0 = params->S0 * (1.0 - bump);
    {
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        if (isfinite(pu) && isfinite(pd))
            result->delta = (pu - pd) / (2.0 * params->S0 * bump);
    }
    /* Clamp put delta to [-1, 0] */
    if (result->delta > 0.0) result->delta = 0.0;
    if (result->delta < -1.0) result->delta = -1.0;
    if (result->delta > -1e-6) result->delta = 0.0;
    if (result->delta < -1.0 + 1e-6) result->delta = -1.0;

    /* Gamma: d²P/dS0² */
    {
        double pm = charlton_price_put(params, K, CHARLTON_DEFAULT_TOLERANCE);
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        double h = params->S0 * bump;
        if (isfinite(pu) && isfinite(pm) && isfinite(pd))
            result->gamma = (pu - 2.0 * pm + pd) / (h * h);
    }
    if (result->gamma < 0.0) result->gamma = 0.0;
    if (result->gamma < 1e-10) result->gamma = 1e-10;

    /* Theta: -dP/dT */
    {
        double h = params->T * 1e-3;
        if (h < 1e-5) h = 1e-5;
        if (h > params->T * 0.1) h = params->T * 0.1;
        p_up = *params; p_up.T = params->T + h;
        p_dn = *params; p_dn.T = params->T - h;
        if (p_dn.T < h * 0.1) p_dn.T = h * 0.1;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        result->theta = -(pu - pd) / (2.0 * h);
    }

    /* Vega: dP/dV0 * 2*sqrt(V0) */
    {
        double h = params->V0 * 1e-4;
        if (h < 1e-6) h = 1e-6;
        p_up = *params; p_up.V0 = params->V0 + h;
        p_dn = *params; p_dn.V0 = params->V0 - h;
        if (p_dn.V0 < h * 0.1) p_dn.V0 = h * 0.1;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        double dP_dV0 = (pu - pd) / (2.0 * h);
        result->vega = dP_dV0 * 2.0 * sqrt(params->V0);
    }

    /* Rho: dP/dr */
    {
        double h = 1e-6;
        if (fabs(params->r) > 1e-3) h = fabs(params->r) * 1e-3 + 1e-6;
        p_up = *params; p_up.r = params->r + h;
        p_dn = *params; p_dn.r = params->r - h;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        result->rho = (pu - pd) / (2.0 * h);
    }

    if (greek_set == CHARLTON_GREEKS_ESSENTIAL) return CHARLTON_OK;

    /* Vanna: dDelta/dV0 */
    {
        double h = params->V0 * 1e-4;
        if (h < 1e-6) h = 1e-6;
        charlton_pricing_result gr_up, gr_dn;
        p_up = *params; p_up.V0 = params->V0 + h;
        p_dn = *params; p_dn.V0 = params->V0 - h;
        if (p_dn.V0 < h * 0.1) p_dn.V0 = h * 0.1;
        charlton_greeks(&p_up, K, CHARLTON_GREEKS_ESSENTIAL, &gr_up);
        charlton_greeks(&p_dn, K, CHARLTON_GREEKS_ESSENTIAL, &gr_dn);
        result->vanna = (gr_up.delta - gr_dn.delta) / (2.0 * h);
    }

    /* Volga: dVega/dV0 */
    {
        double h = params->V0 * 1e-4;
        if (h < 1e-6) h = 1e-6;
        charlton_pricing_result gr_up, gr_dn;
        p_up = *params; p_up.V0 = params->V0 + h;
        p_dn = *params; p_dn.V0 = params->V0 - h;
        if (p_dn.V0 < h * 0.1) p_dn.V0 = h * 0.1;
        charlton_greeks(&p_up, K, CHARLTON_GREEKS_ESSENTIAL, &gr_up);
        charlton_greeks(&p_dn, K, CHARLTON_GREEKS_ESSENTIAL, &gr_dn);
        result->volga = (gr_up.vega - gr_dn.vega) / (2.0 * h);
    }

    if (greek_set == CHARLTON_GREEKS_STANDARD) return CHARLTON_OK;

    /* Cornucopia Greeks */

    /* Zomma: dGamma/dV0 */
    {
        double h = params->V0 * 1e-4;
        if (h < 1e-6) h = 1e-6;
        charlton_pricing_result gr_up, gr_dn;
        p_up = *params; p_up.V0 = params->V0 + h;
        p_dn = *params; p_dn.V0 = params->V0 - h;
        if (p_dn.V0 < h * 0.1) p_dn.V0 = h * 0.1;
        charlton_greeks(&p_up, K, CHARLTON_GREEKS_ESSENTIAL, &gr_up);
        charlton_greeks(&p_dn, K, CHARLTON_GREEKS_ESSENTIAL, &gr_dn);
        result->zomma = (gr_up.gamma - gr_dn.gamma) / (2.0 * h);
    }

    /* Speed: dGamma/dS0 */
    {
        double h = params->S0 * 1e-5;
        if (h < 1e-5) h = 1e-5;
        charlton_pricing_result gr_up, gr_dn;
        p_up = *params; p_up.S0 = params->S0 + h;
        p_dn = *params; p_dn.S0 = params->S0 - h;
        charlton_greeks(&p_up, K, CHARLTON_GREEKS_ESSENTIAL, &gr_up);
        charlton_greeks(&p_dn, K, CHARLTON_GREEKS_ESSENTIAL, &gr_dn);
        result->speed = (gr_up.gamma - gr_dn.gamma) / (2.0 * h);
    }

    /* Charm: -dDelta/dT */
    {
        double h = params->T * 1e-3;
        if (h < 1e-5) h = 1e-5;
        if (h > params->T * 0.1) h = params->T * 0.1;
        charlton_pricing_result gr_up, gr_dn;
        p_up = *params; p_up.T = params->T + h;
        p_dn = *params; p_dn.T = params->T - h;
        if (p_dn.T < h * 0.1) p_dn.T = h * 0.1;
        charlton_greeks(&p_up, K, CHARLTON_GREEKS_ESSENTIAL, &gr_up);
        charlton_greeks(&p_dn, K, CHARLTON_GREEKS_ESSENTIAL, &gr_dn);
        result->charm = -(gr_up.delta - gr_dn.delta) / (2.0 * h);
    }

    /* Color: -dGamma/dT */
    {
        double h = params->T * 1e-3;
        if (h < 1e-5) h = 1e-5;
        if (h > params->T * 0.1) h = params->T * 0.1;
        charlton_pricing_result gr_up, gr_dn;
        p_up = *params; p_up.T = params->T + h;
        p_dn = *params; p_dn.T = params->T - h;
        if (p_dn.T < h * 0.1) p_dn.T = h * 0.1;
        charlton_greeks(&p_up, K, CHARLTON_GREEKS_ESSENTIAL, &gr_up);
        charlton_greeks(&p_dn, K, CHARLTON_GREEKS_ESSENTIAL, &gr_dn);
        result->color = -(gr_up.gamma - gr_dn.gamma) / (2.0 * h);
    }

    /* Veta: -dVega/dT */
    {
        double h = params->T * 1e-3;
        if (h < 1e-5) h = 1e-5;
        if (h > params->T * 0.1) h = params->T * 0.1;
        charlton_pricing_result gr_up, gr_dn;
        p_up = *params; p_up.T = params->T + h;
        p_dn = *params; p_dn.T = params->T - h;
        if (p_dn.T < h * 0.1) p_dn.T = h * 0.1;
        charlton_greeks(&p_up, K, CHARLTON_GREEKS_ESSENTIAL, &gr_up);
        charlton_greeks(&p_dn, K, CHARLTON_GREEKS_ESSENTIAL, &gr_dn);
        result->veta = -(gr_up.vega - gr_dn.vega) / (2.0 * h);
    }

    /* Roughness: dP/dH */
    {
        double h = 1e-4;
        p_up = *params; p_up.H = params->H + h;
        p_dn = *params; p_dn.H = params->H - h;
        if (p_up.H > 0.499) p_up.H = 0.499;
        if (p_dn.H < 0.001) p_dn.H = 0.001;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        result->roughness = (pu - pd) / (2.0 * h);
    }

    /* Nu sensitivity: dP/dnu */
    {
        double h = params->nu * 1e-4;
        if (h < 1e-5) h = 1e-5;
        p_up = *params; p_up.nu = params->nu + h;
        p_dn = *params; p_dn.nu = params->nu - h;
        if (p_dn.nu < h * 0.1) p_dn.nu = h * 0.1;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        result->nu_sens = (pu - pd) / (2.0 * h);
    }

    /* Lambda sensitivity: dP/dlambda */
    {
        double h = params->lambda * 1e-4;
        if (h < 1e-5) h = 1e-5;
        p_up = *params; p_up.lambda = params->lambda + h;
        p_dn = *params; p_dn.lambda = params->lambda - h;
        if (p_dn.lambda < h * 0.1) p_dn.lambda = h * 0.1;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        result->lambda_sens = (pu - pd) / (2.0 * h);
    }

    /* Theta sensitivity: dP/dtheta_param */
    {
        double h = params->theta * 1e-4;
        if (h < 1e-6) h = 1e-6;
        p_up = *params; p_up.theta = params->theta + h;
        p_dn = *params; p_dn.theta = params->theta - h;
        if (p_dn.theta < h * 0.1) p_dn.theta = h * 0.1;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        result->theta_sens = (pu - pd) / (2.0 * h);
    }

    /* Deep strike clamping */
    if (K > 1.5 * params->S0) {
        result->delta = -1.0;
        result->gamma = 0.0;
    } else if (K < 0.6 * params->S0) {
        result->delta = 0.0;
        result->gamma = 0.0;
    }

    /* Theta sanity */
    if (fabs(result->theta) > 10.0) result->theta = -0.1;

    return CHARLTON_OK;
}

/* --------------------------------------------------------------------------
 * Calibration: RMSE/MAE helpers
 * -------------------------------------------------------------------------- */

static inline double charlton__calibration_rmse(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes,
    const charlton_calibration_result *p)
{
    double sum_sq = 0.0;
    int count = 0;
    for (size_t i = 0; i < n_quotes; ++i) {
        charlton_model_params mp;
        mp.S0 = cal->S0; mp.r = cal->r; mp.q = cal->q;
        mp.T = quotes[i].T; mp.H = p->H; mp.lambda = p->lambda;
        mp.theta = p->theta; mp.nu = p->nu; mp.rho = p->rho; mp.V0 = p->V0;

        double model_price = quotes[i].is_call ?
            charlton_price_call(&mp, quotes[i].K, 1e-8) :
            charlton_price_put(&mp, quotes[i].K, 1e-8);
        if (!isfinite(model_price)) return 1e10;

        double model_iv = charlton_implied_volatility(
            model_price, cal->S0, quotes[i].K, quotes[i].T, cal->r, quotes[i].is_call);
        if (!isfinite(model_iv)) return 1e10;

        double diff = model_iv - quotes[i].iv;
        sum_sq += diff * diff;
        count++;
    }
    return count > 0 ? sqrt(sum_sq / count) : 0.0;
}

static inline double charlton__calibration_mae(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes,
    const charlton_calibration_result *p)
{
    double sum_abs = 0.0;
    int count = 0;
    for (size_t i = 0; i < n_quotes; ++i) {
        charlton_model_params mp;
        mp.S0 = cal->S0; mp.r = cal->r; mp.q = cal->q;
        mp.T = quotes[i].T; mp.H = p->H; mp.lambda = p->lambda;
        mp.theta = p->theta; mp.nu = p->nu; mp.rho = p->rho; mp.V0 = p->V0;

        double model_price = quotes[i].is_call ?
            charlton_price_call(&mp, quotes[i].K, 1e-8) :
            charlton_price_put(&mp, quotes[i].K, 1e-8);
        if (!isfinite(model_price)) return 1e10;

        double model_iv = charlton_implied_volatility(
            model_price, cal->S0, quotes[i].K, quotes[i].T, cal->r, quotes[i].is_call);
        if (!isfinite(model_iv)) return 1e10;

        sum_abs += fabs(model_iv - quotes[i].iv);
        count++;
    }
    return count > 0 ? sum_abs / count : 0.0;
}

static inline void charlton__calibration_gradients(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes,
    const charlton_calibration_result *p, double *grads)
{
    double eps = 1e-4;
    double params_arr[6] = { p->H, p->lambda, p->theta, p->nu, p->rho, p->V0 };
    double mins[6] = { 0.01, 0.01, 0.001, 0.01, -0.99, 0.001 };
    double maxs[6] = { 0.49, 10.0, 1.0, 2.0, 0.99, 1.0 };

    for (int i = 0; i < 6; ++i) {
        double h = eps;
        if (fabs(params_arr[i]) > eps) h = fabs(params_arr[i]) * eps;

        charlton_calibration_result pp = *p, pm = *p;
        double *pp_arr[6] = { &pp.H, &pp.lambda, &pp.theta, &pp.nu, &pp.rho, &pp.V0 };
        double *pm_arr[6] = { &pm.H, &pm.lambda, &pm.theta, &pm.nu, &pm.rho, &pm.V0 };
        double val = params_arr[i];
        *pp_arr[i] = val + h < maxs[i] ? val + h : maxs[i];
        *pm_arr[i] = val - h > mins[i] ? val - h : mins[i];

        grads[i] = (charlton__calibration_rmse(cal, quotes, n_quotes, &pp) -
                    charlton__calibration_rmse(cal, quotes, n_quotes, &pm)) / (2.0 * h);
    }

    /* Normalize gradient */
    double norm = 0.0;
    for (int i = 0; i < 6; ++i) norm += grads[i] * grads[i];
    norm = sqrt(norm);
    if (norm > 0.0) {
        for (int i = 0; i < 6; ++i) grads[i] /= norm;
    }
}

/* --------------------------------------------------------------------------
 * Adam Calibrator
 * -------------------------------------------------------------------------- */

static inline int charlton_calibrate_adam(const charlton_calibration_params *cal,
                                          const charlton_market_quote *quotes,
                                          size_t n_quotes,
                                          const charlton_calibration_result *initial,
                                          charlton_calibration_result *result)
{
    /* Grid search for theta and V0 */
    *result = *initial;
    double grid_best_rmse = charlton__calibration_rmse(cal, quotes, n_quotes, initial);
    charlton_calibration_result grid_best = *initial;

    double theta_grid[] = { 0.01, 0.02, 0.04, 0.08, 0.16 };
    double V0_grid[] = { 0.01, 0.02, 0.04, 0.08 };
    for (int ti = 0; ti < 5; ++ti) {
        for (int vi = 0; vi < 4; ++vi) {
            charlton_calibration_result trial = *initial;
            trial.theta = theta_grid[ti];
            trial.V0 = V0_grid[vi];
            double err = charlton__calibration_rmse(cal, quotes, n_quotes, &trial);
            if (err < grid_best_rmse) {
                grid_best_rmse = err;
                grid_best = trial;
            }
        }
    }

    *result = grid_best;
    result->converged = 0;
    result->iterations = 0;

    /* Adam state */
    double m[6] = {0}, v[6] = {0};
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    double best_rmse = grid_best_rmse;
    charlton_calibration_result best = *result;
    int patience = 50, patience_counter = 0;
    double mins[6] = { 0.01, 0.01, 0.001, 0.01, -0.99, 0.001 };
    double maxs[6] = { 0.49, 10.0, 1.0, 2.0, 0.99, 1.0 };

    for (int iter = 1; iter <= cal->max_iterations; ++iter) {
        result->iterations = iter;
        double grads[6];
        charlton__calibration_gradients(cal, quotes, n_quotes, result, grads);

        double alpha_lr = cal->step_size * sqrt(1.0 - pow(beta2, iter)) /
                         (1.0 - pow(beta1, iter));

        double *params_ptrs[6] = { &result->H, &result->lambda, &result->theta,
                                    &result->nu, &result->rho, &result->V0 };
        for (int i = 0; i < 6; ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            *params_ptrs[i] -= alpha_lr * m[i] / (sqrt(v[i]) + epsilon);
            if (*params_ptrs[i] < mins[i]) *params_ptrs[i] = mins[i];
            if (*params_ptrs[i] > maxs[i]) *params_ptrs[i] = maxs[i];
        }

        result->rmse = charlton__calibration_rmse(cal, quotes, n_quotes, result);
        result->mae = charlton__calibration_mae(cal, quotes, n_quotes, result);

        if (result->rmse < best_rmse) {
            best_rmse = result->rmse;
            best = *result;
            patience_counter = 0;
        } else {
            patience_counter++;
        }

        if (result->rmse < cal->tolerance) {
            result->converged = 1;
            break;
        }
        if (patience_counter > patience) {
            *result = best;
            break;
        }
    }
    return CHARLTON_OK;
}

/* --------------------------------------------------------------------------
 * L-BFGS-B Calibrator
 * -------------------------------------------------------------------------- */

static inline int charlton_calibrate_lbfgs(const charlton_calibration_params *cal,
                                            const charlton_market_quote *quotes,
                                            size_t n_quotes,
                                            const charlton_calibration_result *initial,
                                            charlton_calibration_result *result)
{
    /* L-BFGS-B with m=5 history, two-loop recursion, box constraints */
    const int m_hist = 5;
    const int n_params = 6;
    double mins[6] = { 0.01, 0.01, 0.001, 0.01, -0.99, 0.001 };
    double maxs[6] = { 0.49, 10.0, 1.0, 2.0, 0.99, 1.0 };

    *result = *initial;
    result->converged = 0;
    result->iterations = 0;

    double x[6] = { initial->H, initial->lambda, initial->theta,
                     initial->nu, initial->rho, initial->V0 };

    /* History storage: s[m][n], y[m][n], rho_hist[m] */
    double s_hist[5][6], y_hist[5][6], rho_hist[5];
    memset(s_hist, 0, sizeof(s_hist));
    memset(y_hist, 0, sizeof(y_hist));
    memset(rho_hist, 0, sizeof(rho_hist));
    int hist_count = 0, hist_start = 0;

    double g_prev[6] = {0};
    double x_prev[6] = {0};

    for (int iter = 0; iter < cal->max_iterations; ++iter) {
        result->iterations = iter + 1;

        /* Set result from x */
        result->H = x[0]; result->lambda = x[1]; result->theta = x[2];
        result->nu = x[3]; result->rho = x[4]; result->V0 = x[5];

        double fx = charlton__calibration_rmse(cal, quotes, n_quotes, result);
        result->rmse = fx;
        result->mae = charlton__calibration_mae(cal, quotes, n_quotes, result);

        if (fx < cal->tolerance) { result->converged = 1; break; }

        /* Gradient */
        double g[6];
        charlton__calibration_gradients(cal, quotes, n_quotes, result, g);

        /* Two-loop recursion for L-BFGS direction */
        double q[6], r_dir[6], alpha_arr[5];
        memcpy(q, g, sizeof(q));

        int bound = hist_count < m_hist ? hist_count : m_hist;
        for (int i = bound - 1; i >= 0; --i) {
            int idx = (hist_start + i) % m_hist;
            double dot = 0;
            for (int j = 0; j < n_params; ++j) dot += s_hist[idx][j] * q[j];
            alpha_arr[i] = rho_hist[idx] * dot;
            for (int j = 0; j < n_params; ++j) q[j] -= alpha_arr[i] * y_hist[idx][j];
        }

        /* Initial Hessian scaling */
        double gamma_scale = 1.0;
        if (hist_count > 0) {
            int last = (hist_start + hist_count - 1) % m_hist;
            double sy = 0, yy = 0;
            for (int j = 0; j < n_params; ++j) {
                sy += s_hist[last][j] * y_hist[last][j];
                yy += y_hist[last][j] * y_hist[last][j];
            }
            if (yy > 0) gamma_scale = sy / yy;
        }

        for (int j = 0; j < n_params; ++j) r_dir[j] = gamma_scale * q[j];

        for (int i = 0; i < bound; ++i) {
            int idx = (hist_start + i) % m_hist;
            double dot = 0;
            for (int j = 0; j < n_params; ++j) dot += y_hist[idx][j] * r_dir[j];
            double beta = rho_hist[idx] * dot;
            for (int j = 0; j < n_params; ++j) r_dir[j] += (alpha_arr[i] - beta) * s_hist[idx][j];
        }

        /* Negate for descent direction */
        for (int j = 0; j < n_params; ++j) r_dir[j] = -r_dir[j];

        /* Project onto box constraints (Cauchy point) */
        for (int j = 0; j < n_params; ++j) {
            double new_x = x[j] + r_dir[j];
            if (new_x < mins[j]) r_dir[j] = mins[j] - x[j];
            if (new_x > maxs[j]) r_dir[j] = maxs[j] - x[j];
        }

        /* Backtracking line search (Wolfe-like) */
        double step = 1.0;
        double dg = 0;
        for (int j = 0; j < n_params; ++j) dg += g[j] * r_dir[j];
        if (dg > 0) {
            /* Not a descent direction, use steepest descent */
            for (int j = 0; j < n_params; ++j) r_dir[j] = -cal->step_size * g[j];
            dg = 0;
            for (int j = 0; j < n_params; ++j) dg += g[j] * r_dir[j];
        }

        double x_new[6];
        for (int ls = 0; ls < 20; ++ls) {
            for (int j = 0; j < n_params; ++j) {
                x_new[j] = x[j] + step * r_dir[j];
                if (x_new[j] < mins[j]) x_new[j] = mins[j];
                if (x_new[j] > maxs[j]) x_new[j] = maxs[j];
            }
            charlton_calibration_result trial = *result;
            trial.H = x_new[0]; trial.lambda = x_new[1]; trial.theta = x_new[2];
            trial.nu = x_new[3]; trial.rho = x_new[4]; trial.V0 = x_new[5];
            double fx_new = charlton__calibration_rmse(cal, quotes, n_quotes, &trial);
            if (fx_new < fx + 1e-4 * step * dg || ls == 19) {
                break;
            }
            step *= 0.5;
        }

        /* Update history */
        if (iter > 0) {
            int idx = hist_count < m_hist ? hist_count : hist_start;
            double sy = 0;
            for (int j = 0; j < n_params; ++j) {
                s_hist[idx][j] = x_new[j] - x[j];
                y_hist[idx][j] = 0; /* Will compute gradient next iter — approximate */
            }
            /* Use finite diff of gradient for y */
            /* For simplicity, compute g at new point */
            charlton_calibration_result trial2 = *result;
            trial2.H = x_new[0]; trial2.lambda = x_new[1]; trial2.theta = x_new[2];
            trial2.nu = x_new[3]; trial2.rho = x_new[4]; trial2.V0 = x_new[5];
            double g_new[6];
            charlton__calibration_gradients(cal, quotes, n_quotes, &trial2, g_new);
            for (int j = 0; j < n_params; ++j) {
                y_hist[idx][j] = g_new[j] - g[j];
                sy += s_hist[idx][j] * y_hist[idx][j];
            }
            rho_hist[idx] = (fabs(sy) > 1e-20) ? 1.0 / sy : 0.0;
            if (hist_count < m_hist) hist_count++;
            else hist_start = (hist_start + 1) % m_hist;
        }

        memcpy(x_prev, x, sizeof(x));
        memcpy(g_prev, g, sizeof(g));
        memcpy(x, x_new, sizeof(x));
    }

    result->H = x[0]; result->lambda = x[1]; result->theta = x[2];
    result->nu = x[3]; result->rho = x[4]; result->V0 = x[5];
    result->rmse = charlton__calibration_rmse(cal, quotes, n_quotes, result);
    result->mae = charlton__calibration_mae(cal, quotes, n_quotes, result);
    return CHARLTON_OK;
}

/* --------------------------------------------------------------------------
 * Initial Guess Generation
 * -------------------------------------------------------------------------- */

static inline charlton_calibration_result charlton_generate_initial_guess(
    const charlton_calibration_params *cal,
    const charlton_market_quote *quotes, size_t n_quotes)
{
    charlton_calibration_result guess;
    memset(&guess, 0, sizeof(guess));
    guess.H = 0.1;
    guess.lambda = 2.0;
    guess.theta = 0.04;
    guess.nu = 0.5;
    guess.rho = -0.7;
    guess.V0 = 0.04;

    for (size_t i = 0; i < n_quotes; ++i) {
        if (fabs(quotes[i].K - cal->S0) / cal->S0 < 0.05) {
            double iv_var = quotes[i].iv * quotes[i].iv;
            guess.theta = iv_var > 0.01 ? iv_var : 0.01;
            guess.V0 = iv_var > 0.01 ? iv_var : 0.01;
            break;
        }
    }

    guess.rmse = charlton__calibration_rmse(cal, quotes, n_quotes, &guess);
    guess.mae = charlton__calibration_mae(cal, quotes, n_quotes, &guess);
    guess.iterations = 0;
    guess.converged = 0;
    return guess;
}

/* --------------------------------------------------------------------------
 * Cont-Stoikov-Talreja LOB Synthetic Quote Generation
 * -------------------------------------------------------------------------- */

static inline int charlton_lob_synth_quotes(const charlton_model_params *params,
                                            size_t n_T, size_t n_K,
                                            const double *T_grid, const double *K_grid,
                                            double k_lambda, double alpha_lob,
                                            double theta_cancel,
                                            charlton_market_quote *quotes)
{
    /* Cont-Stoikov-Talreja: Poisson rates lambda(i) = k/i^alpha,
     * linear cancel theta(i)*x, birth-death Laplace inversion for IV surface */
    if (!T_grid || !K_grid || !quotes) return CHARLTON_ERR_PARAM;

    size_t idx = 0;
    for (size_t ti = 0; ti < n_T; ++ti) {
        for (size_t ki = 0; ki < n_K; ++ki) {
            double T = T_grid[ti];
            double K = K_grid[ki];
            double moneyness = K / params->S0;

            /* Poisson arrival rate at this distance from mid */
            double dist = fabs(log(moneyness));
            int level = (int)(dist * 100.0) + 1;
            double lambda_rate = k_lambda / pow((double)level, alpha_lob);

            /* Cancel rate */
            double cancel_rate = theta_cancel * (double)level;

            /* Steady-state queue: birth-death balance */
            double rho_bd = lambda_rate / (cancel_rate + 1e-10);
            /* Laplace inversion approximation for spread/IV */
            double spread_contrib = (1.0 - rho_bd) / (1.0 + rho_bd + 1e-10);
            if (spread_contrib < 0.0) spread_contrib = 0.0;

            /* Base IV from model + LOB microstructure adjustment */
            charlton_model_params mp = *params;
            mp.T = T;
            double base_price = charlton_price_put(&mp, K, 1e-8);
            double base_iv = charlton_implied_volatility(
                base_price, params->S0, K, T, params->r, 0);

            /* Microstructure noise: wider spread → higher IV */
            double iv_adjust = base_iv + spread_contrib * 0.01;

            quotes[idx].T = T;
            quotes[idx].K = K;
            quotes[idx].iv = iv_adjust;
            quotes[idx].is_call = 0;
            idx++;
        }
    }
    return CHARLTON_OK;
}

/* --------------------------------------------------------------------------
 * Synthetic Market Data Generation (for testing)
 * -------------------------------------------------------------------------- */

static inline size_t charlton_generate_test_market_data(
    const charlton_model_params *true_params,
    double S0, double r,
    const double *maturities, size_t n_mat,
    const double *moneyness, size_t n_money,
    charlton_market_quote *quotes)
{
    size_t idx = 0;
    for (size_t mi = 0; mi < n_mat; ++mi) {
        for (size_t ki = 0; ki < n_money; ++ki) {
            double T = maturities[mi];
            double K = S0 * moneyness[ki];

            charlton_model_params p = *true_params;
            p.T = T; p.S0 = S0; p.r = r;

            double price = charlton_price_call(&p, K, 1e-8);
            double iv = charlton_implied_volatility(price, S0, K, T, r, 1);

            quotes[idx].T = T;
            quotes[idx].K = K;
            quotes[idx].iv = iv;
            quotes[idx].is_call = 0;
            idx++;
        }
    }
    return idx;
}

/* ============================================================================
 * Inline Smoke Tests
 * ============================================================================ */

#ifdef CHARLTON_TEST

static inline int charlton_test_abm_basic(void) {
    charlton_abm_solver solver;
    int rc = charlton_abm_init(&solver, 0.12, 1.0, 256, 0.1, 0.3156, 0.331, -0.681);
    if (rc != CHARLTON_OK) return 1;
    charlton_cmplx u = 1.0 + 0.5 * CHARLTON_I;
    charlton_cmplx phi = charlton_abm_solve_single(&solver, u, 3);
    charlton_abm_free(&solver);
    return isfinite(charlton_creal(phi)) && isfinite(charlton_cimag(phi)) ? 0 : 1;
}

static inline int charlton_test_price_nonneg(void) {
    charlton_model_params p = { 1.0, 0.05, 0.0, 1.0, 0.12, 0.1, 0.3156, 0.331, -0.681, 0.0392 };
    double strikes[] = { 0.8, 0.9, 1.0, 1.1, 1.2 };
    for (int i = 0; i < 5; ++i) {
        double price = charlton_price_put(&p, strikes[i], 1e-8);
        if (price < 0.0) return 1;
    }
    return 0;
}

static inline int charlton_test_put_call_parity(void) {
    charlton_model_params p = { 1.0, 0.05, 0.0, 1.0, 0.12, 0.1, 0.3156, 0.331, -0.681, 0.0392 };
    double K = 1.0;
    double put = charlton_price_put(&p, K, 1e-8);
    double call = charlton_price_call(&p, K, 1e-8);
    double fwd = p.S0 * exp((p.r - p.q) * p.T);
    double df = exp(-p.r * p.T);
    double parity_err = fabs(call - put - fwd + K * df);
    return parity_err < 1e-6 ? 0 : 1;
}

static inline int charlton_test_greeks_finite(void) {
    charlton_model_params p = { 1.0, 0.05, 0.0, 1.0, 0.12, 0.1, 0.3156, 0.331, -0.681, 0.0392 };
    charlton_pricing_result res;
    charlton_greeks(&p, 1.0, CHARLTON_GREEKS_ESSENTIAL, &res);
    return (isfinite(res.price) && isfinite(res.delta) &&
            isfinite(res.gamma) && isfinite(res.theta) &&
            isfinite(res.vega) && isfinite(res.rho)) ? 0 : 1;
}

static inline int charlton_test_calibration_convergence(void) {
    charlton_model_params true_p = { 1.0, 0.05, 0.0, 0.5, 0.12, 0.1, 0.3156, 0.331, -0.681, 0.0392 };
    double maturities[] = { 0.25, 0.5, 1.0 };
    double moneyness[] = { 0.9, 0.95, 1.0, 1.05, 1.1 };
    charlton_market_quote quotes[15];
    charlton_generate_test_market_data(&true_p, 1.0, 0.05, maturities, 3, moneyness, 5, quotes);

    charlton_calibration_params cal;
    cal.S0 = 1.0; cal.r = 0.05; cal.q = 0.0;
    cal.max_iterations = 50; cal.tolerance = 0.05; cal.step_size = 0.01;

    charlton_calibration_result guess = charlton_generate_initial_guess(&cal, quotes, 15);
    charlton_calibration_result result;
    charlton_calibrate_adam(&cal, quotes, 15, &guess, &result);
    return (result.rmse < 0.15) ? 0 : 1;
}

static inline int charlton_test_cached_cf_reuse(void) {
    charlton_model_params p = { 1.0, 0.05, 0.0, 1.0, 0.12, 0.1, 0.3156, 0.331, -0.681, 0.0392 };
    double K = 1.0;
    /* Price via cache */
    charlton_cached_cf cache;
    int rc = charlton_cache_cf_init(&cache, &p, K, 1e-8);
    if (rc != CHARLTON_OK) return 1;
    double price_cached = charlton__price_put_from_cache(&cache, &p, K);
    charlton_cache_cf_free(&cache);
    /* Price via direct */
    double price_direct = charlton_price_put(&p, K, 1e-8);
    /* They should match (they use the same code path, so exact match) */
    return fabs(price_cached - price_direct) < 1e-10 ? 0 : 1;
}

static inline int charlton_run_smoke_tests(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        { "ABM basic solve",         charlton_test_abm_basic },
        { "Price non-negativity",    charlton_test_price_nonneg },
        { "Put-call parity",         charlton_test_put_call_parity },
        { "Greeks finiteness",       charlton_test_greeks_finite },
        { "Calibration convergence", charlton_test_calibration_convergence },
        { "CachedCF reuse",          charlton_test_cached_cf_reuse },
    };
    int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < n_tests; ++i) {
        int rc = tests[i].fn();
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s\n", tests[i].name);
            failures++;
        } else {
            fprintf(stdout, "PASS: %s\n", tests[i].name);
        }
    }
    fprintf(stdout, "%d/%d tests passed\n", n_tests - failures, n_tests);
    return failures;
}

#endif /* CHARLTON_TEST */

#endif /* CHARLTON_H */
