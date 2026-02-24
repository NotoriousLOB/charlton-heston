/**
 * CHARLTON Conformal Bootstrap Example (Pure C99)
 *
 * Demonstrates error control using the Conformal Bootstrap principle.
 */

#define CHARLTON_IMPLEMENTATION
#include "charlton.h"

int main(void) {
    printf("========================================\n");
    printf("CHARLTON Conformal Bootstrap Example\n");
    printf("========================================\n\n");

    charlton_model_params params;
    params.S0 = 1.0;
    params.r = 0.0;
    params.q = 0.0;
    params.T = 1.0 / 252.0;  /* 1 day (challenging case) */
    params.H = 0.12;
    params.lambda = 0.1;
    params.theta = 0.3156;
    params.nu = 0.331;
    params.rho = -0.681;
    params.V0 = 0.0392;

    double K = 0.95;  /* Slightly OTM put */

    /* Standard price */
    double price = charlton_price_put(&params, K, 1e-10);
    printf("Standard Price (K=%.2f): %.10f\n\n", K, price);

    /* Bootstrap verification */
    double error_estimate;
    double bootstrap_price = charlton_price_put_bootstrap(&params, K, &error_estimate, 1e-10);

    printf("Conformal Bootstrap:\n");
    printf("----------------------------------------\n");
    printf("Price:          %.10f\n", bootstrap_price);
    printf("Error estimate: %.2e\n", error_estimate);
    printf("Rel error:      %.2e\n", error_estimate / bootstrap_price);

    /* Test multiple strikes */
    printf("\nBootstrap across strikes:\n");
    printf("----------------------------------------\n");
    printf("Strike    Price          Error Est\n");
    printf("----------------------------------------\n");

    double strikes[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    for (int i = 0; i < 5; ++i) {
        double err;
        double p = charlton_price_put_bootstrap(&params, strikes[i], &err, 1e-10);
        printf("%.4f    %.10f    %.2e\n", strikes[i], p, err);
    }

    /* Test convergence with different tolerances */
    printf("\nConvergence with tolerance:\n");
    printf("----------------------------------------\n");
    double tolerances[] = {1e-6, 1e-8, 1e-10};
    for (int i = 0; i < 3; ++i) {
        double err;
        double p = charlton_price_put_bootstrap(&params, 1.0, &err, tolerances[i]);
        printf("tol=%.0e  price=%.10f  err=%.2e\n", tolerances[i], p, err);
    }

    /* SINH parameters */
    printf("\nSINH Parameters (K=1.0):\n");
    printf("----------------------------------------\n");
    charlton_abm_solver solver;
    charlton_abm_init(&solver, params.H, params.T, 256,
                      params.lambda, params.theta, params.nu, params.rho);
    double decay = charlton_abm_decay_rate(&solver, params.T, params.V0);
    charlton_sinh_params sp = charlton_compute_sinh_params(
        params.T, params.S0, 1.0, params.r, decay,
        -2.0, 1.0, -M_PI/4, M_PI/4, 1e-10, 0);

    printf("omega1 = %.6f\n", sp.omega1);
    printf("b      = %.6f\n", sp.b);
    printf("omega  = %.6f\n", sp.omega);
    printf("zeta   = %.6f\n", sp.zeta);
    printf("N      = %zu\n", sp.N);
    printf("Lambda = %.6f\n", sp.Lambda);

    charlton_abm_free(&solver);

    printf("\n========================================\n");
    printf("Example completed successfully!\n");
    printf("========================================\n");

    return 0;
}
