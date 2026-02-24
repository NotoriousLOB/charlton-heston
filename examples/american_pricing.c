/**
 * CHARLTON American Option Pricing Example (Pure C99)
 *
 * Demonstrates COS-method American option pricing under rough Heston.
 */

#define CHARLTON_IMPLEMENTATION
#include "charlton.h"

int main(void) {
    printf("========================================\n");
    printf("CHARLTON American Option Pricing Example\n");
    printf("========================================\n\n");

    /* Model parameters (El Euch-Rosenbaum calibrated to S&P 500) */
    charlton_model_params params;
    params.S0 = 1.0;
    params.r = 0.05;
    params.q = 0.0;
    params.T = 0.5;
    params.H = 0.12;
    params.lambda = 0.1;
    params.theta = 0.3156;
    params.nu = 0.331;
    params.rho = -0.681;
    params.V0 = 0.0392;

    printf("Model: Rough Heston (H=%.2f, V0=%.4f, T=%.1f)\n",
           params.H, params.V0, params.T);
    printf("S0=%.2f, r=%.2f, q=%.2f\n\n", params.S0, params.r, params.q);

    /* Price American puts at various strikes */
    double strikes[] = {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15};
    int n_strikes = 7;

    printf("American Put Prices (COS method, 64 timesteps, 128 terms):\n");
    printf("----------------------------------------------------------\n");
    printf("Strike    American      European      Premium       Prem %%\n");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < n_strikes; ++i) {
        double K = strikes[i];

        charlton_american_result am;
        charlton_price_american_put(&params, K, 64, 128, &am);

        double euro = charlton_price_put(&params, K, 1e-8);
        double prem_pct = (euro > 1e-12) ? 100.0 * am.early_exercise_premium / euro : 0.0;

        printf("%.4f    %.8f  %.8f  %.8f  %6.2f%%\n",
               K, am.price, euro, am.early_exercise_premium, prem_pct);
    }

    /* Exercise boundary for ATM put */
    printf("\n\nExercise Boundary (K=1.0, put):\n");
    printf("-------------------------------\n");
    printf("Time      S*(t)\n");
    printf("-------------------------------\n");

    charlton_exercise_boundary eb;
    int rc = charlton_american_exercise_boundary(&params, 1.0, 64, 128, 12, &eb);
    if (rc == CHARLTON_OK) {
        for (int j = 0; j < eb.n_cheb; ++j) {
            printf("%.4f    %.6f\n", eb.nodes[j], eb.boundary[j]);
        }
        charlton_exercise_boundary_free(&eb);
    } else {
        printf("Error computing exercise boundary: %d\n", rc);
    }

    /* American call (should match European for q=0) */
    printf("\n\nAmerican vs European Call (q=0, no early exercise):\n");
    printf("---------------------------------------------------\n");

    charlton_american_result am_call;
    charlton_price_american_call(&params, 1.0, 64, 128, &am_call);
    double euro_call = charlton_price_call(&params, 1.0, 1e-8);

    printf("American call: %.8f\n", am_call.price);
    printf("European call: %.8f\n", euro_call);
    printf("Difference:    %.2e\n", am_call.price - euro_call);

    printf("\nDone.\n");
    return 0;
}
