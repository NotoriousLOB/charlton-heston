/**
 * CHARLTON Greeks Example (Pure C99)
 *
 * Demonstrates comprehensive Greek calculations.
 */

#define CHARLTON_IMPLEMENTATION
#include "charlton.h"

int main(void) {
    printf("========================================\n");
    printf("CHARLTON Greeks Example\n");
    printf("========================================\n\n");

    charlton_model_params params;
    params.S0 = 1.0;
    params.r = 0.0;
    params.q = 0.0;
    params.T = 1.0 / 52.0;
    params.H = 0.12;
    params.lambda = 0.1;
    params.theta = 0.3156;
    params.nu = 0.331;
    params.rho = -0.681;
    params.V0 = 0.0392;

    double K = 1.0;

    /* Essential Greeks */
    printf("Essential Greeks (ATM Put, T=1 week):\n");
    printf("----------------------------------------\n");
    charlton_pricing_result result;
    charlton_greeks(&params, K, CHARLTON_GREEKS_ESSENTIAL, &result);
    printf("Price:  %g\n", result.price);
    printf("Delta:  %g\n", result.delta);
    printf("Gamma:  %g\n", result.gamma);
    printf("Theta:  %g\n", result.theta);
    printf("Vega:   %g\n", result.vega);
    printf("Rho:    %g\n", result.rho);

    /* Standard Greeks */
    printf("\nStandard Greeks:\n");
    printf("----------------------------------------\n");
    charlton_greeks(&params, K, CHARLTON_GREEKS_STANDARD, &result);
    printf("Vanna:  %g\n", result.vanna);
    printf("Volga:  %g\n", result.volga);

    /* Cornucopia Greeks */
    printf("\nCornucopia Greeks:\n");
    printf("----------------------------------------\n");
    charlton_greeks(&params, K, CHARLTON_GREEKS_CORNUCOPIA, &result);
    printf("Zomma:      %g\n", result.zomma);
    printf("Speed:      %g\n", result.speed);
    printf("Charm:      %g\n", result.charm);
    printf("Color:      %g\n", result.color);
    printf("Veta:       %g\n", result.veta);
    printf("Roughness:  %g\n", result.roughness);
    printf("Nu sens:    %g\n", result.nu_sens);
    printf("Lambda s:   %g\n", result.lambda_sens);
    printf("Theta s:    %g\n", result.theta_sens);

    /* Delta profile across strikes */
    printf("\nDelta Profile:\n");
    printf("----------------------------------------\n");
    double strikes[] = {0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15};
    for (int i = 0; i < 7; ++i) {
        charlton_greeks(&params, strikes[i], CHARLTON_GREEKS_ESSENTIAL, &result);
        printf("K=%.2f  delta=%.4f  gamma=%.4f\n", strikes[i], result.delta, result.gamma);
    }

    printf("\n========================================\n");
    printf("Example completed successfully!\n");
    printf("========================================\n");

    return 0;
}
