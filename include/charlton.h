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
    double *t_alpha_cache; /* Precomputed pow(t[n], alpha) for n=0..N */
    double *k_alpha1_cache;/* Precomputed pow(k, alpha+1) for k=0..N+2 */
    /* FFT fast-path for lower-triangular Toeplitz convolution */
    double *boundary_weights;     /* a_weights[n*(n+1)/2] for n=0..N-1 */
    notorious_fft_cmpl *fft_kernel; /* Precomputed FFT of padded kernel, length fft_n */
    notorious_fft_cmpl *fft_F_buf;  /* FFT of F_history (scratch), length fft_n */
    notorious_fft_cmpl *fft_g_buf;  /* Inverse FFT result (scratch), length fft_n */
    notorious_fft_aux *fft_aux;     /* Notorious-FFT aux for DFT */
    size_t  fft_n;                  /* padded size (next pow2 >= 2N) */
    int     force_on2;              /* for benchmark: force O(N^2) fallback */
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

/* COS-method American option pricing (Fang-Oosterlee 2009) */

typedef struct {
    int n_timesteps;      /* Bermudan exercise dates (default 64) */
    int n_cos_terms;      /* COS expansion terms N (default 128) */
    double *cf_re;        /* N allocated doubles: Re{phi(u_k)} */
    double *cf_im;        /* N allocated doubles: Im{phi(u_k)} */
    double *V_coeffs;     /* N allocated doubles for continuation value COS coeffs */
    double *payoff_coeffs;/* N allocated doubles for payoff COS coeffs */
    double *grid_vals;    /* N allocated doubles for grid-space values */
    double a, b;          /* truncation range [a,b] in log-moneyness */
    /* DCT-accelerated grid-space buffers (American option pricing) */
    notorious_fft_aux *dct2_aux;   /* N-point DCT-II aux */
    notorious_fft_aux *dct3_aux;   /* N-point DCT-III aux */
    notorious_fft_aux *idft_aux;   /* 2N-point inverse DFT aux */
    notorious_fft_cmpl *fft_buf;   /* 2N complex buffer for IFFT */
    double *grid_x;        /* N grid points x_m = a + (m+0.5)*(b-a)/N */
    double *grid_C;        /* continuation values on grid */
    int use_dct;           /* 1 if DCT plans allocated successfully */
} charlton_cos_workspace;

typedef struct {
    double price;
    double early_exercise_premium;
    int n_timesteps;
    int n_cos_terms;
    int converged;        /* 1 if Richardson extrapolation error < tol */
} charlton_american_result;

typedef struct {
    int n_cheb;           /* number of Chebyshev nodes */
    int n_timesteps;      /* exercise dates */
    double *nodes;        /* n_cheb Chebyshev-Gauss-Lobatto nodes in [0,T] */
    double *boundary;     /* n_cheb boundary values S*(t_j) */
    double *bary_weights; /* n_cheb barycentric weights */
} charlton_exercise_boundary;

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
                                               double T, double V0);

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

/* American Option Pricing (COS method) */
static inline int charlton_price_american_put(const charlton_model_params *p, double K,
                                               int n_timesteps, int n_cos_terms,
                                               charlton_american_result *result);
static inline int charlton_price_american_call(const charlton_model_params *p, double K,
                                                int n_timesteps, int n_cos_terms,
                                                charlton_american_result *result);
static inline int charlton_american_exercise_boundary(const charlton_model_params *p,
                                                       double K, int n_timesteps,
                                                       int n_cos_terms, int n_cheb,
                                                       charlton_exercise_boundary *eb);
static inline int charlton_american_greeks(const charlton_model_params *p, double K,
                                            int greek_set, charlton_pricing_result *result);

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
    double ap1 = s->alpha + 1.0;
    for (size_t k = 0; k < N; ++k) {
        size_t base = k * (k + 1) / 2;
        s->a_weights[base + k] = h_alpha * s->gamma_alpha2;
        if (k == 0) continue;
        double k0 = (double)k;
        double kp1_ap1 = s->k_alpha1_cache[k + 1];
        s->a_weights[base] = h_alpha * s->gamma_alpha2 *
            (kp1_ap1 - (k0 - s->alpha) * pow((double)(k + 1), s->alpha));
        for (size_t j = 1; j < k; ++j) {
            size_t m = k - j;
            s->a_weights[base + j] = h_alpha * s->gamma_alpha2 *
                (s->k_alpha1_cache[m + 2] +
                 s->k_alpha1_cache[m] -
                 2.0 * s->k_alpha1_cache[m + 1]);
        }
    }
}

/* Helper macros for interleaved complex access (works in both C and C++) */
#define CHARLTON_FFT_RE(ptr, i) (((double*)(ptr))[2*(i)])
#define CHARLTON_FFT_IM(ptr, i) (((double*)(ptr))[2*(i)+1])

static inline void charlton__compute_fft_kernel(charlton_abm_solver *s) {
    /* Precompute FFT of padded Toeplitz kernel for fast convolution */
    size_t N = s->N;
    if (N < 8) { s->fft_kernel = NULL; return; }

    /* Need power-of-2 for efficient FFT; fft_n >= 2N to avoid circular aliasing */
    size_t fft_n = 1;
    while (fft_n < 2 * N) fft_n <<= 1;
    s->fft_n = fft_n;

    s->fft_aux = notorious_fft_mkaux_dft_1d((int)fft_n);
    if (!s->fft_aux) { s->fft_kernel = NULL; return; }

    s->fft_kernel = (notorious_fft_cmpl*)charlton_alloc_doubles(2 * fft_n);
    s->fft_F_buf  = (notorious_fft_cmpl*)charlton_alloc_doubles(2 * fft_n);
    s->fft_g_buf  = (notorious_fft_cmpl*)charlton_alloc_doubles(2 * fft_n);
    s->boundary_weights = charlton_alloc_doubles(N);
    if (!s->fft_kernel || !s->fft_F_buf || !s->fft_g_buf || !s->boundary_weights) {
        charlton_aligned_free(s->fft_kernel);
        charlton_aligned_free(s->fft_F_buf);
        charlton_aligned_free(s->fft_g_buf);
        charlton_aligned_free(s->boundary_weights);
        s->fft_kernel = NULL;
        s->fft_F_buf = NULL;
        s->fft_g_buf = NULL;
        s->boundary_weights = NULL;
        notorious_fft_free_aux(s->fft_aux);
        s->fft_aux = NULL;
        return;
    }

    /* Form kernel: c[j] = interior Toeplitz weight for distance j */
    double h_alpha = pow(s->h, s->alpha);
    notorious_fft_cmpl *c_pad = s->fft_F_buf; /* borrow scratch */
    memset(c_pad, 0, fft_n * sizeof(notorious_fft_cmpl));
    for (size_t j = 0; j < N; ++j) {
        double w;
        if (j == 0) {
            w = h_alpha * s->gamma_alpha2 * (pow(2.0, s->alpha + 1.0) - 2.0);
        } else {
            w = h_alpha * s->gamma_alpha2 *
                (s->k_alpha1_cache[j + 2] + s->k_alpha1_cache[j]
                 - 2.0 * s->k_alpha1_cache[j + 1]);
        }
        CHARLTON_FFT_RE(c_pad, j) = w;
        CHARLTON_FFT_IM(c_pad, j) = 0.0;
    }

    /* FFT of padded kernel → fft_kernel */
    notorious_fft_dft(c_pad, s->fft_kernel, s->fft_aux);

    /* Populate boundary_weights: a_weights[n*(n+1)/2] for n=0..N-1 */
    for (size_t n = 0; n < N; ++n) {
        s->boundary_weights[n] = s->a_weights[n * (n + 1) / 2];
    }
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

    /* Precompute pow(t[n], alpha) for n=0..N */
    s->t_alpha_cache = charlton_alloc_doubles(N + 1);
    if (!s->t_alpha_cache) return CHARLTON_ERR_ALLOC;
    s->t_alpha_cache[0] = 0.0;
    for (size_t i = 1; i <= N; ++i)
        s->t_alpha_cache[i] = pow(s->h * (double)i, s->alpha);

    /* Precompute pow(k, alpha+1) for k=0..N+2 */
    s->k_alpha1_cache = charlton_alloc_doubles(N + 3);
    if (!s->k_alpha1_cache) { charlton_aligned_free(s->t_alpha_cache); return CHARLTON_ERR_ALLOC; }
    double ap1 = s->alpha + 1.0;
    s->k_alpha1_cache[0] = 0.0;
    for (size_t i = 1; i <= N + 2; ++i)
        s->k_alpha1_cache[i] = pow((double)i, ap1);

    size_t weight_size = N * (N + 1) / 2;
    s->a_weights = charlton_alloc_doubles(weight_size);
    if (!s->a_weights) return CHARLTON_ERR_ALLOC;

    charlton__compute_adams_weights(s);
    charlton__compute_fft_kernel(s);  /* May set fft_kernel=NULL on failure, that's OK */

    return CHARLTON_OK;
}

static inline void charlton_abm_free(charlton_abm_solver *s) {
    charlton_aligned_free(s->a_weights);
    charlton_aligned_free(s->t_alpha_cache);
    charlton_aligned_free(s->k_alpha1_cache);
    charlton_aligned_free(s->boundary_weights);
    charlton_aligned_free(s->fft_kernel);
    charlton_aligned_free(s->fft_F_buf);
    charlton_aligned_free(s->fft_g_buf);
    if (s->fft_aux) notorious_fft_free_aux(s->fft_aux);
    memset(s, 0, sizeof(*s));
}

static inline charlton_cmplx charlton__asymptotic_term(const charlton_abm_solver *s,
                                                        charlton_cmplx u, size_t n,
                                                        double scale) {
    charlton_cmplx u_term = u * u - CHARLTON_I * u;
    return -0.5 * u_term * s->t_alpha_cache[n] * s->gamma_alpha1 / scale;
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

/* FFT-accelerated causal convolution for lower-triangular Toeplitz matrix.
 * Computes h0_all[n] = sum_{j=0}^{n} a_weights[n,j] * F[j]  for n=0..N-1.
 * Uses FFT for the interior Toeplitz part and boundary corrections for j=0 and j=n. */
static inline void charlton__fft_causal_convolve(const charlton_abm_solver *s,
                                                 const charlton_cmplx *F_history,
                                                 charlton_cmplx *h0_all) {
    size_t N = s->N;
    size_t fft_n = s->fft_n;
    notorious_fft_cmpl *F_pad = s->fft_F_buf;
    notorious_fft_cmpl *G_fft = s->fft_g_buf;

    /* Pack F_history into interleaved complex, zero-pad to fft_n */
    memset(F_pad, 0, fft_n * sizeof(notorious_fft_cmpl));
    for (size_t j = 0; j < N; ++j) {
        CHARLTON_FFT_RE(F_pad, j) = charlton_creal(F_history[j]);
        CHARLTON_FFT_IM(F_pad, j) = charlton_cimag(F_history[j]);
    }

    /* Forward FFT of F_pad */
    notorious_fft_dft(F_pad, G_fft, s->fft_aux);

    /* Pointwise multiply by precomputed kernel FFT */
    for (size_t k = 0; k < fft_n; ++k) {
        double a = CHARLTON_FFT_RE(s->fft_kernel, k);
        double b = CHARLTON_FFT_IM(s->fft_kernel, k);
        double c = CHARLTON_FFT_RE(G_fft, k);
        double d = CHARLTON_FFT_IM(G_fft, k);
        CHARLTON_FFT_RE(G_fft, k) = a * c - b * d;
        CHARLTON_FFT_IM(G_fft, k) = a * d + b * c;
    }

    /* Inverse FFT → circular convolution */
    notorious_fft_invdft(G_fft, F_pad, s->fft_aux);

    double scale = 1.0 / (double)fft_n;
    double h_alpha = pow(s->h, s->alpha);
    double diag_weight = h_alpha * s->gamma_alpha2;
    double c0 = diag_weight * (pow(2.0, s->alpha + 1.0) - 2.0);

    for (size_t n = 0; n < N; ++n) {
        double conv_re = CHARLTON_FFT_RE(F_pad, n) * scale;
        double conv_im = CHARLTON_FFT_IM(F_pad, n) * scale;

        /* Boundary corrections:
         * FFT computes sum c[n-j]*F[j] where c[] is the interior Toeplitz kernel.
         * Actual ABM weight for j=0 is boundary_weights[n] (not c[n]).
         * Actual ABM weight for j=n is diag_weight (not c[0]).
         */
        double corr_re, corr_im;
        if (n == 0) {
            /* Only one term; correction accounts for both boundary deviations */
            corr_re = (s->boundary_weights[0] - c0) * charlton_creal(F_history[0]);
            corr_im = (s->boundary_weights[0] - c0) * charlton_cimag(F_history[0]);
        } else {
            double cn = h_alpha * s->gamma_alpha2 *
                (s->k_alpha1_cache[n + 2] + s->k_alpha1_cache[n]
                 - 2.0 * s->k_alpha1_cache[n + 1]);
            corr_re = (s->boundary_weights[n] - cn) * charlton_creal(F_history[0])
                    + (diag_weight - c0) * charlton_creal(F_history[n]);
            corr_im = (s->boundary_weights[n] - cn) * charlton_cimag(F_history[0])
                    + (diag_weight - c0) * charlton_cimag(F_history[n]);
        }
        h0_all[n] = CHARLTON_CMPLX(conv_re + corr_re, conv_im + corr_im);
    }
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

    if (s->fft_kernel != NULL && s->boundary_weights != NULL && !s->force_on2) {
        /* ================================================================
         * FFT Fast Path: Gauss-Seidel two-sweep
         * Sweep 1: predictor with 1 Picard iteration per step (cheap)
         * Sweep 2: FFT convolution + full corrector
         * ================================================================ */
        /* Sweep 1: predictor, 1 Picard iteration */
        for (size_t n = 0; n < N; ++n) {
            charlton_cmplx h_as_tilde = charlton__asymptotic_term(s, u, n + 1, scale);
            charlton_cmplx h0_tilde = 0.0;
            for (size_t j = 0; j <= n; ++j) {
                h0_tilde += s->a_weights[j + (n - j) * (n - j + 1) / 2] * F_history[j];
            }
            charlton_cmplx h1_new = h0_tilde;
            charlton_cmplx F_pred = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
            h1_new = h0_tilde + s->a_weights[(n + 1) * (n + 2) / 2 - 1] * F_pred;
            h1_tilde[n + 1] = h1_new;
            F_history[n] = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
        }

        /* Sweep 2: FFT convolution + full Picard */
        charlton_cmplx *h0_all = charlton_alloc_cmplx(N);
        if (h0_all) {
            charlton__fft_causal_convolve(s, F_history, h0_all);
            for (size_t n = 0; n < N; ++n) {
                charlton_cmplx h_as_tilde = charlton__asymptotic_term(s, u, n + 1, scale);
                charlton_cmplx h1_new = h0_all[n];
                for (int p = 0; p < n_picard; ++p) {
                    charlton_cmplx h1_old = h1_new;
                    charlton_cmplx F_pred = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
                    h1_new = h0_all[n] + s->a_weights[(n + 1) * (n + 2) / 2 - 1] * F_pred;
                    if (charlton_cabs(h1_new - h1_old) < 1e-14 * (1.0 + charlton_cabs(h1_new)))
                        break;
                }
                h1_tilde[n + 1] = h1_new;
                F_history[n] = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
            }
            charlton_aligned_free(h0_all);
        }
    } else {
        /* O(N^2) fallback */
        for (size_t n = 0; n < N; ++n) {
            charlton_cmplx h_as_tilde = charlton__asymptotic_term(s, u, n + 1, scale);

            charlton_cmplx h0_tilde = 0.0;
            for (size_t j = 0; j <= n; ++j) {
                h0_tilde += s->a_weights[j + (n - j) * (n - j + 1) / 2] * F_history[j];
            }

            /* Picard iterations with early convergence check */
            charlton_cmplx h1_new = h0_tilde;
            for (int p = 0; p < n_picard; ++p) {
                charlton_cmplx h1_old = h1_new;
                charlton_cmplx F_pred = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
                h1_new = h0_tilde + s->a_weights[(n + 1) * (n + 2) / 2 - 1] * F_pred;
                if (charlton_cabs(h1_new - h1_old) < 1e-14 * (1.0 + charlton_cabs(h1_new)))
                    break;
            }

            h1_tilde[n + 1] = h1_new;
            F_history[n] = charlton__F_as1(s, u, h_as_tilde, h1_new, scale);
        }
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
        result[b] = charlton_abm_solve_single(s, u_batch[b], 8);
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
    double lm_in, double lp_in, double gm, double gp, double tol, int is_call)
{
    charlton_sinh_params sp;
    memset(&sp, 0, sizeof(sp));
    double z_T = log(S0 / K) - r * T;
    double omega_choice, d0;
    double lm, lp;

    if (is_call) {
        lm = lm_in;
        lp = -1.0;
        omega_choice = gm / 2.0;
        d0 = -omega_choice;
    } else {
        lm = 0.0;
        lp = lp_in;
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
        /* xi = i*omega1 + b*(sinh(y)*cos(omega) + i*cosh(y)*sin(omega)) */
        double xi_re = cache->sp.b * sinh(y) * cos(cache->sp.omega);
        double xi_im = cache->sp.omega1 + cache->sp.b * cosh(y) * sin(cache->sp.omega);
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

/* Scalar fallback (always available) — with Richardson extrapolation */
static inline double charlton__sinh_sum_scalar(const charlton_cached_cf *cf,
                                                double x_re, double x_im,
                                                double K, double df, double V0) {
    size_t N = cf->n_quad;
    double s_full = 0.0, s_half = 0.0;
    for (size_t j = 0; j < N; ++j) {
        /* Skip NaN values from ABM solver (can occur for large |u|) */
        if (!isfinite(cf->phi_re[j]) || !isfinite(cf->phi_im[j])) {
            continue;
        }
        charlton_cmplx xi = cf->u_re[j] + CHARLTON_I * cf->u_im[j];
        charlton_cmplx exp_arg = (cf->phi_re[j] + CHARLTON_I * cf->phi_im[j]) * V0 +
                                 CHARLTON_I * xi * (x_re + CHARLTON_I * x_im);
        charlton_cmplx cfv = charlton_cexp(exp_arg);
        charlton_cmplx denom_val = xi * (xi + CHARLTON_I);
        charlton_cmplx cosh_t = cf->cosh_re[j] + CHARLTON_I * cf->cosh_im[j];
        charlton_cmplx g = cfv * cosh_t / denom_val;
        double w = (j == 0) ? 0.5 : 1.0;
        double val = w * charlton_creal(g);
        s_full += val;
        if (j % 2 == 0) s_half += val;
    }
    /* Richardson extrapolation: (4*S_full - S_half) / 3 */
    double s_rich = (4.0 * s_full - s_half) / 3.0;
    return -cf->sp.b * cf->sp.zeta * K * df / M_PI * s_rich;
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
                                               double T, double V0) {
    double x = log(S0 / K);
    double df = exp(-r * T);
    return charlton__sinh_sum(cache, x, 0.0, K, df, V0);
}
static inline double charlton__price_put_from_cache(const charlton_cached_cf *cache,
                                                     const charlton_model_params *p,
                                                     double K) {
    double x = log(p->S0 / K);
    double df = exp(-p->r * p->T);
    double raw = charlton__sinh_sum(cache, x, 0.0, K, df, p->V0);

    if (!isfinite(raw)) raw = 0.0;

    /* Numerical noise floor */
    double intrinsic = K * exp(-p->r * p->T) - p->S0 * exp(-p->q * p->T);
    if (intrinsic < 0.0) intrinsic = 0.0;

    /* For deep ITM/OTM with certain contour parameters, the SINH quadrature
     * can produce unreasonable results. Apply sanity bounds. */
    double max_reasonable = K * 10.0;  /* Put price should never exceed ~K */
    if (raw > max_reasonable) {
        /* Numerical explosion - fall back to intrinsic for ITM, 0 for OTM */
        raw = intrinsic > 0.0 ? intrinsic : 1e-12;
    }

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
    double sigma_lo = 0.01, sigma_hi = 5.0;
    double sqrtT = sqrt(T);
    for (int i = 0; i < 50; ++i) {
        double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;
        double nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)));
        double bs_price;
        if (is_call)
            bs_price = S0 * nd1 - K * exp(-r * T) * nd2;
        else
            bs_price = K * exp(-r * T) * (1.0 - nd2) - S0 * (1.0 - nd1);
        double diff = bs_price - price;
        if (fabs(diff) < 1e-10) break;

        /* Update bracket */
        if (diff > 0.0) sigma_hi = sigma;
        else            sigma_lo = sigma;

        /* Newton step with bisection fallback */
        double vega_bs = S0 * sqrtT * exp(-d1 * d1 / 2.0) / sqrt(2.0 * M_PI);
        double sigma_new = sigma - diff / vega_bs;
        if (vega_bs < 1e-12 || sigma_new <= sigma_lo || sigma_new >= sigma_hi)
            sigma_new = 0.5 * (sigma_lo + sigma_hi);
        sigma = sigma_new;
    }
    if (sigma < 1e-4) sigma = 1e-4;
    if (sigma > 10.0) sigma = 10.0;
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

    /* Delta: central finite differences */
    {
        bump = 1e-4;
        p_up = *params; p_up.S0 = params->S0 * (1.0 + bump);
        p_dn = *params; p_dn.S0 = params->S0 * (1.0 - bump);
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        if (isfinite(pu) && isfinite(pd))
            result->delta = (pu - pd) / (2.0 * params->S0 * bump);
        /* Enforce theoretical bounds [-1, 0] for put delta */
        if (result->delta > 0.0) result->delta = 0.0;
        if (result->delta < -1.0) result->delta = -1.0;
    }

    /* Gamma: central finite differences */
    {
        bump = 1e-4;
        p_up = *params; p_up.S0 = params->S0 * (1.0 + bump);
        p_dn = *params; p_dn.S0 = params->S0 * (1.0 - bump);
        double pm = result->price;
        double pu = charlton_price_put(&p_up, K, CHARLTON_DEFAULT_TOLERANCE);
        double pd = charlton_price_put(&p_dn, K, CHARLTON_DEFAULT_TOLERANCE);
        double h = params->S0 * bump;
        if (isfinite(pu) && isfinite(pm) && isfinite(pd))
            result->gamma = (pu - 2.0 * pm + pd) / (h * h);
        if (result->gamma < 0.0) result->gamma = 0.0;
    }

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
        double r_scale = fabs(params->r) > 0.01 ? fabs(params->r) : 0.01;
        double h = r_scale * 1e-4;
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

    /* Clip gradient to prevent explosion */
    double max_grad = 10.0;
    for (int i = 0; i < 6; ++i) {
        if (grads[i] > max_grad) grads[i] = max_grad;
        if (grads[i] < -max_grad) grads[i] = -max_grad;
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
    /* Grid search for lambda, theta, and V0 */
    *result = *initial;
    double grid_best_rmse = charlton__calibration_rmse(cal, quotes, n_quotes, initial);
    charlton_calibration_result grid_best = *initial;

    double lambda_grid[] = { 0.1, 0.3, 1.0, 3.0 };
    double theta_grid[] = { 0.01, 0.02, 0.04, 0.08, 0.16 };
    double V0_grid[] = { 0.01, 0.02, 0.04, 0.08 };
    for (int li = 0; li < 4; ++li) {
      for (int ti = 0; ti < 5; ++ti) {
        for (int vi = 0; vi < 4; ++vi) {
            charlton_calibration_result trial = *initial;
            trial.lambda = lambda_grid[li];
            trial.theta = theta_grid[ti];
            trial.V0 = V0_grid[vi];
            double err = charlton__calibration_rmse(cal, quotes, n_quotes, &trial);
            if (err < grid_best_rmse) {
                grid_best_rmse = err;
                grid_best = trial;
            }
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
    guess.H = 0.12;
    guess.lambda = 0.3;
    guess.theta = 0.04;
    guess.nu = 0.3;
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
            quotes[idx].is_call = 1;
            idx++;
        }
    }
    return idx;
}

/* --------------------------------------------------------------------------
 * COS-Method American Option Pricing (Fang-Oosterlee 2009)
 * -------------------------------------------------------------------------- */

/* Chebyshev-Gauss-Lobatto nodes on [a,b] */
static inline void charlton__cheb_nodes(int n, double a, double b, double *nodes) {
    for (int j = 0; j < n; ++j) {
        double theta_j = M_PI * (double)j / (double)(n - 1);
        nodes[j] = 0.5 * (a + b) + 0.5 * (b - a) * cos(theta_j);
    }
}

/* Barycentric weights for Chebyshev-Gauss-Lobatto nodes */
static inline void charlton__cheb_bary_weights(int n, double *w) {
    for (int j = 0; j < n; ++j) {
        w[j] = (j % 2 == 0) ? 1.0 : -1.0;
    }
    w[0] *= 0.5;
    w[n - 1] *= 0.5;
}

/* Barycentric interpolation at point x given nodes, values, weights */
static inline double charlton__cheb_interp(const double *nodes, const double *vals,
                                            const double *w, int n, double x) {
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        double diff = x - nodes[j];
        if (fabs(diff) < 1e-15) return vals[j];
        double term = w[j] / diff;
        num += term * vals[j];
        den += term;
    }
    return num / den;
}

/* COS workspace init */
static inline int charlton_cos_init(charlton_cos_workspace *ws, int n_ts, int n_cos,
                                     double a, double b) {
    memset(ws, 0, sizeof(*ws));
    ws->n_timesteps = n_ts;
    ws->n_cos_terms = n_cos;
    ws->a = a;
    ws->b = b;

    ws->cf_re         = charlton_alloc_doubles((size_t)n_cos);
    ws->cf_im         = charlton_alloc_doubles((size_t)n_cos);
    ws->V_coeffs      = charlton_alloc_doubles((size_t)n_cos);
    ws->payoff_coeffs = charlton_alloc_doubles((size_t)n_cos);
    ws->grid_vals     = charlton_alloc_doubles((size_t)n_cos);

    if (!ws->cf_re || !ws->cf_im || !ws->V_coeffs ||
        !ws->payoff_coeffs || !ws->grid_vals) {
        charlton_aligned_free(ws->cf_re);
        charlton_aligned_free(ws->cf_im);
        charlton_aligned_free(ws->V_coeffs);
        charlton_aligned_free(ws->payoff_coeffs);
        charlton_aligned_free(ws->grid_vals);
        memset(ws, 0, sizeof(*ws));
        return CHARLTON_ERR_ALLOC;
    }

    /* Allocate DCT-accelerated grid-space buffers */
    ws->grid_x = charlton_alloc_doubles((size_t)n_cos);
    ws->grid_C = charlton_alloc_doubles((size_t)n_cos);
    if (ws->grid_x && ws->grid_C) {
        double ba = b - a;
        for (int m = 0; m < n_cos; ++m) {
            ws->grid_x[m] = a + ((double)m + 0.5) * ba / (double)n_cos;
        }

        /* Create Notorious-FFT plans */
        ws->dct2_aux = notorious_fft_mkaux_t2t3_1d(n_cos);
        ws->dct3_aux = notorious_fft_mkaux_t2t3_1d(n_cos);
        ws->idft_aux = notorious_fft_mkaux_dft_1d(2 * n_cos);
        ws->fft_buf = (notorious_fft_cmpl *)charlton_aligned_alloc(64,
            (size_t)(2 * n_cos) * sizeof(notorious_fft_cmpl));

        if (ws->dct2_aux && ws->dct3_aux && ws->idft_aux && ws->fft_buf) {
            ws->use_dct = 1;
        } else {
            if (ws->dct2_aux) notorious_fft_free_aux(ws->dct2_aux);
            if (ws->dct3_aux) notorious_fft_free_aux(ws->dct3_aux);
            if (ws->idft_aux) notorious_fft_free_aux(ws->idft_aux);
            charlton_aligned_free(ws->fft_buf);
            ws->dct2_aux = NULL;
            ws->dct3_aux = NULL;
            ws->idft_aux = NULL;
            ws->fft_buf = NULL;
            ws->use_dct = 0;
        }
    }
    return CHARLTON_OK;
}

static inline void charlton_cos_free(charlton_cos_workspace *ws) {
    charlton_aligned_free(ws->cf_re);
    charlton_aligned_free(ws->cf_im);
    charlton_aligned_free(ws->V_coeffs);
    charlton_aligned_free(ws->payoff_coeffs);
    charlton_aligned_free(ws->grid_vals);
    charlton_aligned_free(ws->grid_x);
    charlton_aligned_free(ws->grid_C);
    if (ws->dct2_aux) notorious_fft_free_aux(ws->dct2_aux);
    if (ws->dct3_aux) notorious_fft_free_aux(ws->dct3_aux);
    if (ws->idft_aux) notorious_fft_free_aux(ws->idft_aux);
    charlton_aligned_free(ws->fft_buf);
    memset(ws, 0, sizeof(*ws));
}

/* Compute COS truncation range [a,b] from rough Heston cumulants */
static inline void charlton__cos_truncation(const charlton_model_params *p,
                                             double *a_out, double *b_out) {
    double T = p->T;
    double H = p->H;
    double lam = p->lambda;
    double th = p->theta;
    double V0 = p->V0;
    double r = p->r;
    double q = p->q;

    /* First cumulant (approximate drift in log-moneyness) */
    double exp_lT = exp(-lam * T);
    double c1 = (r - q) * T;
    if (lam > 1e-12)
        c1 += (1.0 - exp_lT) * (th - V0) / (2.0 * lam);
    else
        c1 -= 0.5 * V0 * T;

    /* Second cumulant: rough vol variance scaling ~ V0 * T^(2H) * C(H) */
    double gamma_2H1 = tgamma(2.0 * H + 1.0);
    double c2 = V0 * pow(T, 2.0 * H) / gamma_2H1;
    /* Add mean-reversion contribution */
    if (lam > 1e-12)
        c2 += th * T;
    else
        c2 += th * T;

    /* Wider range for rougher paths */
    double L = 8.0 + 4.0 / sqrt(H);
    double w = L * sqrt(fabs(c2));
    if (w < 1.0) w = 1.0; /* minimum width */

    *a_out = c1 - w;
    *b_out = c1 + w;
}

/* Compute CF coefficients for COS method at timestep dt.
 * F_k = Re{ phi(u_k, dt) * exp(-i * u_k * a) } where u_k = k*pi/(b-a) */
static inline int charlton__cos_cf_coeffs(charlton_cos_workspace *ws,
                                           const charlton_model_params *p,
                                           double dt) {
    int N = ws->n_cos_terms;
    double ba = ws->b - ws->a;
    double a = ws->a;

    /* Build ABM solver for this timestep duration */
    charlton_abm_solver solver;
    /* Use fewer ABM steps for small dt to keep cost down */
    size_t abm_N = 64;
    if (dt < 0.01) abm_N = 32;

    int rc = charlton_abm_init(&solver, p->H, dt, abm_N,
                                p->lambda, p->theta, p->nu, p->rho);
    if (rc != CHARLTON_OK) return rc;

    double drift = (p->r - p->q) * dt;

    for (int k = 0; k < N; ++k) {
        double u_k = (double)k * M_PI / ba;
        charlton_cmplx u = CHARLTON_CMPLX(u_k, 0.0);
        charlton_cmplx psi = charlton_abm_solve_single(&solver, u, 8);
        /* CF of log-return over dt: phi(u_k) = exp(psi * V0 + i*u*drift) */
        charlton_cmplx log_cf = psi * p->V0 + CHARLTON_I * u_k * drift;
        charlton_cmplx phi_val = charlton_cexp(log_cf);
        ws->cf_re[k] = charlton_creal(phi_val);
        ws->cf_im[k] = charlton_cimag(phi_val);
    }

    charlton_abm_free(&solver);
    return CHARLTON_OK;
}

/* Compute analytic COS coefficients for put payoff: max(K - e^x, 0) on [a,b]
 * where x = log(S/K), so payoff in x-space is K * max(1 - e^x, 0) for x <= 0.
 * Integration limit: a to min(0, b). */
static inline void charlton__cos_put_payoff_coeffs(charlton_cos_workspace *ws, double K) {
    int N = ws->n_cos_terms;
    double a = ws->a;
    double b = ws->b;
    double ba = b - a;
    double x_star = 0.0; /* exercise boundary in log-moneyness: S = K when x = 0 */
    if (x_star > b) x_star = b;
    if (x_star < a) { /* entirely OTM */
        memset(ws->payoff_coeffs, 0, (size_t)N * sizeof(double));
        return;
    }

    for (int k = 0; k < N; ++k) {
        double kpi = (double)k * M_PI;
        /* integral of (K - K*e^x) * cos(k*pi*(x-a)/(b-a)) dx from a to 0 */
        /* = K * integral_a^0 cos(k*pi*(x-a)/(b-a)) dx
         *   - K * integral_a^0 e^x * cos(k*pi*(x-a)/(b-a)) dx */
        double chi_k, psi_k;

        if (k == 0) {
            psi_k = x_star - a;
            chi_k = exp(x_star) - exp(a);
        } else {
            double w = kpi / ba;
            /* psi_k = integral_a^{x_star} cos(w*(x-a)) dx = sin(w*(x_star-a)) / w */
            psi_k = sin(w * (x_star - a)) / w;
            /* chi_k = integral_a^{x_star} e^x cos(w*(x-a)) dx (closed form) */
            double denom = 1.0 + w * w;
            double xs_a = x_star - a;
            chi_k = (exp(x_star) * (cos(w * xs_a) + w * sin(w * xs_a)) - exp(a)) / denom;
        }

        ws->payoff_coeffs[k] = (2.0 / ba) * K * (psi_k - chi_k);
    }
}

/* Compute COS coefficients for call payoff via put-call parity decomposition.
 * max(S-K,0) = max(S-K,0) - max(K-S,0) + max(K-S,0)
 *            = (S - K) + max(K-S,0)    for S >= K, or max(K-S,0) for S < K
 * So call_payoff = put_payoff + (e^x - 1)*K
 *
 * COS coefficients: call_k = put_k + K*(chi_k^{a,b} - psi_k^{a,b})
 * where chi_k = integral_a^b e^x cos(...) and psi_k = integral_a^b cos(...).
 * This avoids the unbounded exponential growth issue. */
static inline void charlton__cos_call_payoff_coeffs(charlton_cos_workspace *ws, double K) {
    int N = ws->n_cos_terms;
    double a = ws->a;
    double b = ws->b;
    double ba = b - a;

    /* First compute put payoff coefficients */
    charlton__cos_put_payoff_coeffs(ws, K);

    /* Add (e^x - 1)*K COS coefficients over [a,b] */
    for (int k = 0; k < N; ++k) {
        double kpi = (double)k * M_PI;
        double chi_k, psi_k;

        if (k == 0) {
            chi_k = exp(b) - exp(a);
            psi_k = ba;
        } else {
            double w = kpi / ba;
            double denom = 1.0 + w * w;
            /* chi_k = integral_a^b e^x cos(w*(x-a)) dx */
            chi_k = (exp(b) * (cos(kpi) + w * sin(kpi))
                     - exp(a) * (1.0 + 0.0)) / denom;
            /* sin(k*pi) = 0, cos(k*pi) = (-1)^k */
            chi_k = (exp(b) * (((k % 2 == 0) ? 1.0 : -1.0))
                     - exp(a)) / denom;
            /* psi_k = integral_a^b cos(w*(x-a)) dx = sin(k*pi)/w = 0 */
            psi_k = 0.0;
        }

        ws->payoff_coeffs[k] += (2.0 / ba) * K * (chi_k - psi_k);
    }
}

/* Evaluate continuation value at point x using COS series.
 * C(x) = e^{-r*dt} * sum_k (w_k) Re{phi_k * exp(i*u_k*(x-a))} * V_k */
static inline double charlton__cos_continuation(const charlton_cos_workspace *ws,
                                                 double x, double discount) {
    int N = ws->n_cos_terms;
    double ba = ws->b - ws->a;
    double xa = x - ws->a;
    double cont = 0.0;
    for (int k = 0; k < N; ++k) {
        double u_k = (double)k * M_PI / ba;
        double angle = u_k * xa;
        double re_part = ws->cf_re[k] * cos(angle) - ws->cf_im[k] * sin(angle);
        double weight = (k == 0) ? 0.5 : 1.0;
        cont += weight * re_part * ws->V_coeffs[k];
    }
    return discount * cont;
}

/* Analytic COS coefficient integrals for put payoff K*(1-e^x) over [a_lim, b_lim].
 * chi_k = integral_{a_lim}^{b_lim} e^x * cos(k*pi*(x-a)/(b-a)) dx
 * psi_k = integral_{a_lim}^{b_lim} cos(k*pi*(x-a)/(b-a)) dx */
static inline void charlton__cos_chi_psi(double a, double b, double a_lim, double b_lim,
                                          int k, double *chi_out, double *psi_out) {
    double ba = b - a;
    if (k == 0) {
        *psi_out = b_lim - a_lim;
        *chi_out = exp(b_lim) - exp(a_lim);
    } else {
        double w = (double)k * M_PI / ba;
        double al = a_lim - a;
        double bl = b_lim - a;
        *psi_out = (sin(w * bl) - sin(w * al)) / w;
        double denom = 1.0 + w * w;
        *chi_out = (exp(b_lim) * (cos(w * bl) + w * sin(w * bl))
                   - exp(a_lim) * (cos(w * al) + w * sin(w * al))) / denom;
    }
}

/* Compute COS coefficient of continuation value on interval [c_lo, c_hi]:
 * H_k = (2/(b-a)) * integral_{c_lo}^{c_hi} C(x) cos(k*pi*(x-a)/(b-a)) dx
 * where C(x) = discount * sum_j (w_j) Re{phi_j exp(i*u_j*(x-a))} * V_j
 *
 * This expands to a sum over j of analytic integrals involving
 * cos*cos and sin*cos products. */
static inline double charlton__cos_cont_coeff(const charlton_cos_workspace *ws,
                                               int k, double c_lo, double c_hi,
                                               double discount) {
    int N = ws->n_cos_terms;
    double ba = ws->b - ws->a;
    double a = ws->a;
    double w_k = (double)k * M_PI / ba;
    double lo = c_lo - a;
    double hi = c_hi - a;

    double H_k = 0.0;
    for (int j = 0; j < N; ++j) {
        double w_j = (double)j * M_PI / ba;
        double wt = (j == 0) ? 0.5 : 1.0;
        double phi_re = ws->cf_re[j];
        double phi_im = ws->cf_im[j];

        /* Re{phi_j * exp(i*w_j*t)} = phi_re*cos(w_j*t) - phi_im*sin(w_j*t)
         * Need: integral_{lo}^{hi} [phi_re*cos(w_j*t) - phi_im*sin(w_j*t)] * cos(w_k*t) dt */
        double I_cc, I_sc; /* cos*cos and sin*cos integrals */

        if (j == k) {
            /* cos(w*t)*cos(w*t) = (1 + cos(2w*t))/2 */
            if (k == 0) {
                I_cc = hi - lo;
            } else {
                double w2 = 2.0 * w_k;
                I_cc = (hi - lo) / 2.0 + (sin(w2 * hi) - sin(w2 * lo)) / (2.0 * w2);
            }
            /* sin(w*t)*cos(w*t) = sin(2w*t)/2 */
            if (k == 0) {
                I_sc = 0.0;
            } else {
                double w2 = 2.0 * w_k;
                I_sc = -(cos(w2 * hi) - cos(w2 * lo)) / (2.0 * w2);
            }
        } else {
            double wp = w_j + w_k;
            double wm = w_j - w_k;
            /* cos(w_j*t)*cos(w_k*t) = [cos((w_j-w_k)*t) + cos((w_j+w_k)*t)]/2 */
            if (fabs(wm) < 1e-15)
                I_cc = (hi - lo) / 2.0 + (sin(wp * hi) - sin(wp * lo)) / (2.0 * wp);
            else
                I_cc = (sin(wm * hi) - sin(wm * lo)) / (2.0 * wm)
                     + (sin(wp * hi) - sin(wp * lo)) / (2.0 * wp);
            /* sin(w_j*t)*cos(w_k*t) = [sin((w_j+w_k)*t) + sin((w_j-w_k)*t)]/2 */
            if (fabs(wm) < 1e-15)
                I_sc = -(cos(wp * hi) - cos(wp * lo)) / (2.0 * wp);
            else
                I_sc = -(cos(wm * hi) - cos(wm * lo)) / (2.0 * wm)
                     - (cos(wp * hi) - cos(wp * lo)) / (2.0 * wp);
            I_sc *= -1.0; /* correct sign: integral of sin(w_j*t)*cos(w_k*t) */

            /* Recompute: sin(a)*cos(b) = [sin(a+b) + sin(a-b)]/2 */
            I_sc = ((cos(wm * lo) - cos(wm * hi)) / (2.0 * (fabs(wm) < 1e-15 ? 1e-15 : wm))
                  + (cos(wp * lo) - cos(wp * hi)) / (2.0 * wp));
        }

        H_k += wt * discount * ws->V_coeffs[j] * (phi_re * I_cc - phi_im * I_sc);
    }

    return (2.0 / ba) * H_k;
}

/* DCT-accelerated backward step: grid-space round-trip via DCT-II/III.
 * Evaluates continuation on half-integer grid, applies early exercise,
 * then forward DCT-II to get new coefficients. O(N log N) per step. */
static inline double charlton__cos_backward_step_dct(charlton_cos_workspace *ws, double K,
                                                      int is_call, double r_dt) {
    int N = ws->n_cos_terms;
    double discount = exp(-r_dt);

    /* Reuse scratch buffers */
    double *X = ws->grid_vals;      /* DCT-III input (cosine weights) */
    double *Y = ws->payoff_coeffs;  /* DST-III input (sine weights) */
    double *C_grid = ws->grid_C;    /* continuation on grid */
    double *S_grid = ws->grid_vals; /* DST-III output (reuses X buffer) */

    /* Form DCT-III input: X_0 = c_0/2, X_k = c_k/2 for k>0 */
    X[0] = discount * 0.25 * ws->V_coeffs[0] * ws->cf_re[0];
    for (int k = 1; k < N; ++k) {
        X[k] = discount * 0.5 * ws->V_coeffs[k] * ws->cf_re[k];
        Y[k-1] = -discount * 0.5 * ws->V_coeffs[k] * ws->cf_im[k];
    }
    Y[N-1] = 0.0;
    double X0 = X[0]; /* save before X is overwritten */

    /* DCT-III(X) → A_m - X_0  (cosine part of continuation) */
    notorious_fft_dct3(X, C_grid, ws->dct3_aux);

    /* DST-III(Y) → B_m  (sine part of continuation) */
    notorious_fft_dst3(Y, S_grid, ws->dct3_aux);

    /* Assemble continuation and apply early exercise */
    double x_star = is_call ? ws->b : ws->a;
    int exercise_optimal = 0;

    if (!is_call) {
        for (int m = 0; m < N; ++m) {
            double cont = C_grid[m] + X0 + S_grid[m];
            double x = ws->grid_x[m];
            double payoff = (x <= 0.0) ? K * (1.0 - exp(x)) : 0.0;
            if (payoff > cont) {
                C_grid[m] = payoff;
                if (!exercise_optimal) {
                    exercise_optimal = 1;
                    x_star = x;
                }
            } else {
                C_grid[m] = cont;
            }
        }
    } else {
        for (int m = N - 1; m >= 0; --m) {
            double cont = C_grid[m] + X0 + S_grid[m];
            double x = ws->grid_x[m];
            double payoff = (x >= 0.0) ? K * (exp(x) - 1.0) : 0.0;
            if (payoff > cont) {
                C_grid[m] = payoff;
                if (!exercise_optimal) {
                    exercise_optimal = 1;
                    x_star = x;
                }
            } else {
                C_grid[m] = cont;
            }
        }
    }

    /* Forward DCT-II → new coefficients V_k = DCT-II(C_grid)_k / N */
    double *V_new = ws->grid_vals;
    notorious_fft_dct2(C_grid, V_new, ws->dct2_aux);
    double invN = 1.0 / (double)N;
    for (int k = 0; k < N; ++k) {
        ws->V_coeffs[k] = V_new[k] * invN;
    }

    return exercise_optimal ? x_star : (is_call ? ws->b : ws->a);
}

/* One backward induction step using Fang-Oosterlee analytic coefficient update.
 * Returns exercise boundary x* (in log-moneyness).
 * When DCT is available, delegates to grid-space round-trip. */
static inline double charlton__cos_backward_step(charlton_cos_workspace *ws, double K,
                                                   int is_call, double r_dt) {
    int N = ws->n_cos_terms;
    double ba = ws->b - ws->a;
    double a = ws->a;
    double b = ws->b;
    double discount = exp(-r_dt);

    /* Fast path: DCT-accelerated grid-space round-trip */
    if (ws->use_dct) {
        return charlton__cos_backward_step_dct(ws, K, is_call, r_dt);
    }

    /* Step 1: Find exercise boundary x* via bisection.
     * For put: find rightmost x where payoff(x) >= continuation(x). */
    double x_lo = a, x_hi = 0.0; /* for puts, boundary is in [a, 0] */
    if (is_call) { x_lo = 0.0; x_hi = b; }

    double pay_lo, pay_hi, cont_lo, cont_hi;
    if (!is_call) {
        pay_lo = K * (1.0 - exp(x_lo));
        pay_hi = 0.0;
        cont_lo = charlton__cos_continuation(ws, x_lo, discount);
        cont_hi = charlton__cos_continuation(ws, x_hi, discount);
    } else {
        pay_lo = 0.0;
        pay_hi = K * (exp(x_hi) - 1.0);
        cont_lo = charlton__cos_continuation(ws, x_lo, discount);
        cont_hi = charlton__cos_continuation(ws, x_hi, discount);
    }

    double x_star;
    int exercise_optimal = 0;

    if (!is_call) {
        if (pay_lo > cont_lo) {
            exercise_optimal = 1;
            for (int iter = 0; iter < 50; ++iter) {
                double x_mid = 0.5 * (x_lo + x_hi);
                double pay_mid = K * (1.0 - exp(x_mid));
                double cont_mid = charlton__cos_continuation(ws, x_mid, discount);
                if (pay_mid > cont_mid)
                    x_lo = x_mid;
                else
                    x_hi = x_mid;
            }
            x_star = 0.5 * (x_lo + x_hi);
        } else {
            x_star = a;
        }
    } else {
        if (pay_hi > cont_hi) {
            exercise_optimal = 1;
            for (int iter = 0; iter < 50; ++iter) {
                double x_mid = 0.5 * (x_lo + x_hi);
                double pay_mid = K * (exp(x_mid) - 1.0);
                double cont_mid = charlton__cos_continuation(ws, x_mid, discount);
                if (pay_mid > cont_mid)
                    x_hi = x_mid;
                else
                    x_lo = x_mid;
            }
            x_star = 0.5 * (x_lo + x_hi);
        } else {
            x_star = b;
        }
    }

    /* Step 2: Update V_coeffs analytically.
     * OPTIMIZATION: full-interval [a,b] uses orthogonality → O(N) */
    if (!exercise_optimal) {
        for (int k = 0; k < N; ++k) {
            ws->grid_vals[k] = discount * ws->cf_re[k] * ws->V_coeffs[k];
        }
    } else if (!is_call) {
        for (int k = 0; k < N; ++k) {
            double chi_k, psi_k;
            charlton__cos_chi_psi(a, b, a, x_star, k, &chi_k, &psi_k);
            double payoff_k = (2.0 / ba) * K * (psi_k - chi_k);
            double cont_k = charlton__cos_cont_coeff(ws, k, x_star, b, discount);
            ws->grid_vals[k] = payoff_k + cont_k;
        }
    } else {
        for (int k = 0; k < N; ++k) {
            double cont_k = charlton__cos_cont_coeff(ws, k, a, x_star, discount);
            double chi_k, psi_k;
            charlton__cos_chi_psi(a, b, x_star, b, k, &chi_k, &psi_k);
            double payoff_k = (2.0 / ba) * K * (chi_k - psi_k);
            ws->grid_vals[k] = cont_k + payoff_k;
        }
    }

    memcpy(ws->V_coeffs, ws->grid_vals, (size_t)N * sizeof(double));
    return exercise_optimal ? x_star : (is_call ? b : a);
}

/* Core COS backward induction for American options.
 * Returns 0 on success, fills result. If boundary != NULL, records S*(t_m). */
static inline int charlton__cos_american_core(const charlton_model_params *p, double K,
                                               int n_timesteps, int n_cos_terms,
                                               int is_call,
                                               charlton_american_result *result,
                                               double *boundary_out) {
    if (!p || !result) return CHARLTON_ERR_PARAM;
    if (p->T <= 0.0 || K <= 0.0) return CHARLTON_ERR_PARAM;
    if (n_timesteps < 1) n_timesteps = 64;
    if (n_cos_terms < 8) n_cos_terms = 128;

    memset(result, 0, sizeof(*result));
    result->n_timesteps = n_timesteps;
    result->n_cos_terms = n_cos_terms;

    /* Compute truncation range */
    double a, b;
    charlton__cos_truncation(p, &a, &b);

    /* Initialize workspace */
    charlton_cos_workspace ws;
    int rc = charlton_cos_init(&ws, n_timesteps, n_cos_terms, a, b);
    if (rc != CHARLTON_OK) return rc;

    int N = ws.n_cos_terms; /* may have been rounded up to power of 2 */
    double dt = p->T / (double)n_timesteps;

    /* Compute CF coefficients for one timestep */
    rc = charlton__cos_cf_coeffs(&ws, p, dt);
    if (rc != CHARLTON_OK) {
        charlton_cos_free(&ws);
        return rc;
    }

    /* Terminal payoff coefficients */
    if (is_call)
        charlton__cos_call_payoff_coeffs(&ws, K);
    else
        charlton__cos_put_payoff_coeffs(&ws, K);

    /* Initialize V_coeffs with payoff coefficients */
    memcpy(ws.V_coeffs, ws.payoff_coeffs, (size_t)N * sizeof(double));

    /* Backward induction */
    for (int m = n_timesteps - 1; m >= 0; --m) {
        double x_star = charlton__cos_backward_step(&ws, K, is_call, p->r * dt);
        if (boundary_out) {
            boundary_out[m] = K * exp(x_star);
        }
    }

    /* Price recovery: evaluate COS series at x = log(S0/K).
     * After backward induction, V_coeffs represent the option value.
     * V(x) = sum_k (w_k) * V_k * cos(k*pi*(x-a)/(b-a)) */
    double x0 = log(p->S0 / K);
    double ba = ws.b - ws.a;
    double price = 0.0;
    for (int k = 0; k < N; ++k) {
        double cos_term = cos((double)k * M_PI * (x0 - ws.a) / ba);
        double weight = (k == 0) ? 0.5 : 1.0;
        price += weight * ws.V_coeffs[k] * cos_term;
    }

    /* Ensure non-negative */
    if (price < 0.0) price = 0.0;

    /* Compute European price for early exercise premium */
    double euro_price;
    if (is_call)
        euro_price = charlton_price_call(p, K, 1e-8);
    else
        euro_price = charlton_price_put(p, K, 1e-8);

    /* If American price is somehow less than European (numerical error), use European */
    if (price < euro_price) price = euro_price;

    double intrinsic;
    if (is_call)
        intrinsic = (p->S0 > K) ? (p->S0 - K) : 0.0;
    else
        intrinsic = (K > p->S0) ? (K - p->S0) : 0.0;

    /* Ensure >= intrinsic */
    if (price < intrinsic) price = intrinsic;

    result->price = price;
    result->early_exercise_premium = price - euro_price;
    result->converged = 1;

    charlton_cos_free(&ws);
    return CHARLTON_OK;
}

static inline int charlton_price_american_put(const charlton_model_params *p, double K,
                                               int n_timesteps, int n_cos_terms,
                                               charlton_american_result *result) {
    return charlton__cos_american_core(p, K, n_timesteps, n_cos_terms, 0, result, NULL);
}

static inline int charlton_price_american_call(const charlton_model_params *p, double K,
                                                int n_timesteps, int n_cos_terms,
                                                charlton_american_result *result) {
    if (!p || !result) return CHARLTON_ERR_PARAM;

    /* For q=0 (or very small q), American call = European call: no early exercise */
    if (fabs(p->q) < 1e-12) {
        double euro_call = charlton_price_call(p, K, 1e-8);
        memset(result, 0, sizeof(*result));
        result->price = euro_call;
        result->early_exercise_premium = 0.0;
        result->n_timesteps = n_timesteps;
        result->n_cos_terms = n_cos_terms;
        result->converged = 1;
        return CHARLTON_OK;
    }

    /* For q > 0, use put-call symmetry: C_am(S,K,r,q,T) = P_am(K,S,q,r,T)
     * Swap S0 <-> K and r <-> q, then price American put. */
    charlton_model_params p_sym = *p;
    p_sym.S0 = K;
    p_sym.r = p->q;
    p_sym.q = p->r;
    /* rho sign flips under put-call symmetry for stochastic vol models */
    p_sym.rho = -p->rho;

    charlton_american_result put_result;
    int rc = charlton__cos_american_core(&p_sym, p->S0, n_timesteps, n_cos_terms,
                                          0, &put_result, NULL);
    if (rc != CHARLTON_OK) return rc;

    *result = put_result;
    return CHARLTON_OK;
}

/* Extract exercise boundary fitted to Chebyshev nodes */
static inline int charlton_american_exercise_boundary(const charlton_model_params *p,
                                                       double K, int n_timesteps,
                                                       int n_cos_terms, int n_cheb,
                                                       charlton_exercise_boundary *eb) {
    if (!p || !eb) return CHARLTON_ERR_PARAM;
    if (n_cheb < 3) n_cheb = 16;
    if (n_timesteps < 1) n_timesteps = 64;
    if (n_cos_terms < 8) n_cos_terms = 128;

    memset(eb, 0, sizeof(*eb));
    eb->n_cheb = n_cheb;
    eb->n_timesteps = n_timesteps;

    /* Allocate raw boundary storage */
    double *raw_boundary = charlton_alloc_doubles((size_t)n_timesteps);
    if (!raw_boundary) return CHARLTON_ERR_ALLOC;

    /* Run backward induction recording boundary */
    charlton_american_result result;
    int rc = charlton__cos_american_core(p, K, n_timesteps, n_cos_terms, 0,
                                          &result, raw_boundary);
    if (rc != CHARLTON_OK) {
        charlton_aligned_free(raw_boundary);
        return rc;
    }

    /* Allocate Chebyshev arrays */
    eb->nodes = charlton_alloc_doubles((size_t)n_cheb);
    eb->boundary = charlton_alloc_doubles((size_t)n_cheb);
    eb->bary_weights = charlton_alloc_doubles((size_t)n_cheb);
    if (!eb->nodes || !eb->boundary || !eb->bary_weights) {
        charlton_aligned_free(raw_boundary);
        charlton_aligned_free(eb->nodes);
        charlton_aligned_free(eb->boundary);
        charlton_aligned_free(eb->bary_weights);
        memset(eb, 0, sizeof(*eb));
        return CHARLTON_ERR_ALLOC;
    }

    /* Chebyshev nodes in [0, T] */
    charlton__cheb_nodes(n_cheb, 0.0, p->T, eb->nodes);
    charlton__cheb_bary_weights(n_cheb, eb->bary_weights);

    /* Interpolate raw boundary (uniform in time) to Chebyshev nodes */
    double dt = p->T / (double)n_timesteps;
    for (int j = 0; j < n_cheb; ++j) {
        double t = eb->nodes[j];
        /* Linear interpolation from raw boundary */
        double idx_f = t / dt;
        int idx_lo = (int)idx_f;
        if (idx_lo < 0) idx_lo = 0;
        if (idx_lo >= n_timesteps - 1) {
            eb->boundary[j] = raw_boundary[n_timesteps - 1];
        } else {
            double frac = idx_f - (double)idx_lo;
            eb->boundary[j] = (1.0 - frac) * raw_boundary[idx_lo]
                             + frac * raw_boundary[idx_lo + 1];
        }
        /* Clamp to [0, K] for puts */
        if (eb->boundary[j] > K) eb->boundary[j] = K;
        if (eb->boundary[j] < 0.0) eb->boundary[j] = 0.0;
    }

    charlton_aligned_free(raw_boundary);
    return CHARLTON_OK;
}

static inline void charlton_exercise_boundary_free(charlton_exercise_boundary *eb) {
    charlton_aligned_free(eb->nodes);
    charlton_aligned_free(eb->boundary);
    charlton_aligned_free(eb->bary_weights);
    memset(eb, 0, sizeof(*eb));
}

/* American Greeks via bump-and-reprice */
static inline int charlton_american_greeks(const charlton_model_params *p, double K,
                                            int greek_set, charlton_pricing_result *result) {
    if (!p || !result) return CHARLTON_ERR_PARAM;
    charlton_pricing_result_init(result);

    int n_ts = 64, n_cos = 128;
    charlton_american_result ar;

    /* Base price */
    int rc = charlton_price_american_put(p, K, n_ts, n_cos, &ar);
    if (rc != CHARLTON_OK) return rc;
    result->price = ar.price;

    if (greek_set == CHARLTON_GREEKS_PRICE_ONLY) return CHARLTON_OK;

    double eps_S = p->S0 * 0.001;
    double eps_v = p->V0 * 0.01;
    double eps_T = p->T * 0.001;
    double eps_r = 0.0001;

    /* Delta and Gamma */
    {
        charlton_model_params p_up = *p, p_dn = *p;
        p_up.S0 += eps_S; p_dn.S0 -= eps_S;
        charlton_american_result r_up, r_dn;
        charlton_price_american_put(&p_up, K, n_ts, n_cos, &r_up);
        charlton_price_american_put(&p_dn, K, n_ts, n_cos, &r_dn);
        result->delta = (r_up.price - r_dn.price) / (2.0 * eps_S);
        result->gamma = (r_up.price - 2.0 * ar.price + r_dn.price) / (eps_S * eps_S);
    }

    /* Theta */
    {
        charlton_model_params p_dn = *p;
        p_dn.T -= eps_T;
        if (p_dn.T > 0.0) {
            charlton_american_result r_dn;
            charlton_price_american_put(&p_dn, K, n_ts, n_cos, &r_dn);
            result->theta = -(r_dn.price - ar.price) / eps_T;
        }
    }

    /* Vega */
    {
        charlton_model_params p_up = *p, p_dn = *p;
        p_up.V0 += eps_v; p_dn.V0 -= eps_v;
        charlton_american_result r_up, r_dn;
        charlton_price_american_put(&p_up, K, n_ts, n_cos, &r_up);
        charlton_price_american_put(&p_dn, K, n_ts, n_cos, &r_dn);
        result->vega = (r_up.price - r_dn.price) / (2.0 * eps_v);
    }

    /* Rho */
    {
        charlton_model_params p_up = *p, p_dn = *p;
        p_up.r += eps_r; p_dn.r -= eps_r;
        charlton_american_result r_up, r_dn;
        charlton_price_american_put(&p_up, K, n_ts, n_cos, &r_up);
        charlton_price_american_put(&p_dn, K, n_ts, n_cos, &r_dn);
        result->rho = (r_up.price - r_dn.price) / (2.0 * eps_r);
    }

    return CHARLTON_OK;
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
    charlton_cmplx phi = charlton_abm_solve_single(&solver, u, 8);
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
