/**
 * CHARLTON Test Data (C99)
 *
 * Reference values and test parameters from the paper.
 */

#ifndef CHARLTON_TEST_DATA_H
#define CHARLTON_TEST_DATA_H

/* El Euch-Rosenbaum calibrated parameters for S&P 500 (Jan 7, 2010) */
#define CHARLTON_TEST_ALPHA  0.62      /* H + 0.5 */
#define CHARLTON_TEST_H      0.12
#define CHARLTON_TEST_GAMMA  0.1       /* Mean reversion speed */
#define CHARLTON_TEST_RHO    (-0.681)  /* Correlation */
#define CHARLTON_TEST_THETA  0.3156    /* Long-term variance */
#define CHARLTON_TEST_NU     0.331     /* Vol-of-vol */
#define CHARLTON_TEST_V0     0.0392    /* Initial variance */

/* TSLA calibrated parameters from Section 7.1 */
#define CHARLTON_TEST_TSLA_H       0.011913
#define CHARLTON_TEST_TSLA_LAMBDA  2.36609
#define CHARLTON_TEST_TSLA_THETA   0.424949
#define CHARLTON_TEST_TSLA_NU      0.5780
#define CHARLTON_TEST_TSLA_RHO     (-0.178493)
#define CHARLTON_TEST_TSLA_V0      0.527527

/* Test tolerances */
#define CHARLTON_TEST_PRICE_TOL   1e-6
#define CHARLTON_TEST_REL_TOL     1e-4

/* Benchmark prices from Table 1 (T = 1/252, S0 = 1) */
typedef struct {
    double K;
    double V;
} charlton_test_benchmark_price;

static const charlton_test_benchmark_price charlton_test_table1[] = {
    { 0.95,  2.4557955e-07 },
    { 0.975, 1.29117047e-04 },
    { 1.0,   5.0111443104e-03 },
    { 1.025, 9.16277402e-05 },
    { 1.05,  3.3118e-08 }
};
#define CHARLTON_TEST_TABLE1_SIZE 5

/* Benchmark prices from Table 2 (T = 1/52, S0 = 1) */
static const charlton_test_benchmark_price charlton_test_table2[] = {
    { 0.8,  1.78e-05 },
    { 0.85, 1.89042e-04 },
    { 0.9,  1.390943e-03 },
    { 0.95, 6.975898e-03 },
    { 1.0,  0.023896768 },
    { 1.05, 6.556374e-03 },
    { 1.1,  9.78149e-04 },
    { 1.15, 6.73e-05 }
};
#define CHARLTON_TEST_TABLE2_SIZE 8

#endif /* CHARLTON_TEST_DATA_H */
