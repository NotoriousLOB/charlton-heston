/**
 * CHARLTON Basic Pricing Example (Pure C99)
 *
 * Demonstrates basic usage of the Rough Heston pricer with SINH-acceleration.
 */

#define CHARLTON_IMPLEMENTATION
#include "charlton.h"

int main(void) {
    printf("========================================\n");
    printf("CHARLTON Basic Pricing Example\n");
    printf("========================================\n\n");

    /* Model parameters (El Euch-Rosenbaum calibrated to S&P 500) */
    charlton_model_params params;
    params.S0 = 1.0;
    params.r = 0.0;
    params.q = 0.0;
    params.T = 1.0 / 52.0;     /* 1 week maturity */
    params.H = 0.12;
    params.lambda = 0.1;
    params.theta = 0.3156;
    params.nu = 0.331;
    params.rho = -0.681;
    params.V0 = 0.0392;

    /* Price put options at various strikes */
    double strikes[] = {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2};
    int n_strikes = 7;

    printf("Put Option Prices (T = 1 week):\n");
    printf("----------------------------------------\n");
    printf("Strike    Price          IV (%%)\n");
    printf("----------------------------------------\n");

    for (int i = 0; i < n_strikes; ++i) {
        double K = strikes[i];
        double price = charlton_price_put(&params, K, 1e-10);
        double iv = charlton_implied_volatility(price, params.S0, K, params.T, params.r, 0);
        printf("%.4f    %.10f    %.2f%%\n", K, price, iv * 100.0);
    }

    printf("\n");

    /* Put-call parity check */
    double K = 1.0;
    double put_price = charlton_price_put(&params, K, 1e-10);
    double call_price = charlton_price_call(&params, K, 1e-10);

    printf("Put-Call Parity Check (K = %.2f):\n", K);
    printf("----------------------------------------\n");
    printf("Put Price:  %.10f\n", put_price);
    printf("Call Price: %.10f\n", call_price);

    double fwd = params.S0 * exp((params.r - params.q) * params.T);
    double df = exp(-params.r * params.T);
    double parity_lhs = call_price - put_price;
    double parity_rhs = fwd - K * df;

    printf("C - P = %.10f\n", parity_lhs);
    printf("S - K*e^(-rT) = %.10f\n", parity_rhs);
    printf("Difference: %.2e\n", fabs(parity_lhs - parity_rhs));

    printf("\n========================================\n");
    printf("Example completed successfully!\n");
    printf("========================================\n");

    return 0;
}
