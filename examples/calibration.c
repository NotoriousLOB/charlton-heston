/**
 * CHARLTON Calibration Example (Pure C99)
 *
 * Demonstrates model calibration to synthetic market data.
 */

#define CHARLTON_IMPLEMENTATION
#include "charlton.h"

int main(void) {
    printf("========================================\n");
    printf("CHARLTON Calibration Example\n");
    printf("========================================\n\n");

    /* True model parameters (what we'll try to recover) */
    charlton_model_params true_params;
    true_params.S0 = 1.0;
    true_params.r = 0.0;
    true_params.q = 0.0;
    true_params.T = 0.5;
    true_params.H = 0.1;
    true_params.lambda = 2.0;
    true_params.theta = 0.04;
    true_params.nu = 0.5;
    true_params.rho = -0.6;
    true_params.V0 = 0.04;

    printf("True Model Parameters:\n");
    printf("----------------------------------------\n");
    printf("H      = %.4f\n", true_params.H);
    printf("lambda = %.4f\n", true_params.lambda);
    printf("theta  = %.4f\n", true_params.theta);
    printf("nu     = %.4f\n", true_params.nu);
    printf("rho    = %.4f\n", true_params.rho);
    printf("V0     = %.4f\n", true_params.V0);

    /* Generate synthetic market data */
    double maturities[] = {1.0 / 52.0, 2.0 / 52.0, 1.0 / 12.0};
    double moneyness[] = {0.9, 0.95, 1.0, 1.05, 1.1};
    charlton_market_quote quotes[15];
    size_t n_quotes = charlton_generate_test_market_data(
        &true_params, 1.0, 0.0, maturities, 3, moneyness, 5, quotes);

    printf("\nSynthetic Market Data (%zu quotes):\n", n_quotes);
    printf("----------------------------------------\n");
    for (size_t i = 0; i < n_quotes; ++i) {
        printf("T=%.4f  K=%.2f  IV=%.4f\n", quotes[i].T, quotes[i].K, quotes[i].iv);
    }

    /* Calibrate */
    charlton_calibration_params cal;
    cal.S0 = 1.0;
    cal.r = 0.0;
    cal.q = 0.0;
    cal.max_iterations = 100;
    cal.tolerance = 1e-4;
    cal.step_size = 0.01;

    charlton_calibration_result guess = charlton_generate_initial_guess(&cal, quotes, n_quotes);
    printf("\nInitial Guess: H=%.4f lambda=%.4f theta=%.4f nu=%.4f rho=%.4f V0=%.4f\n",
           guess.H, guess.lambda, guess.theta, guess.nu, guess.rho, guess.V0);

    charlton_calibration_result result;
    charlton_calibrate_adam(&cal, quotes, n_quotes, &guess, &result);

    printf("\nCalibrated Parameters:\n");
    printf("----------------------------------------\n");
    printf("H      = %.4f  (true: %.4f, err: %.4f)\n", result.H, true_params.H, fabs(result.H - true_params.H));
    printf("lambda = %.4f  (true: %.4f, err: %.4f)\n", result.lambda, true_params.lambda, fabs(result.lambda - true_params.lambda));
    printf("theta  = %.4f  (true: %.4f, err: %.4f)\n", result.theta, true_params.theta, fabs(result.theta - true_params.theta));
    printf("nu     = %.4f  (true: %.4f, err: %.4f)\n", result.nu, true_params.nu, fabs(result.nu - true_params.nu));
    printf("rho    = %.4f  (true: %.4f, err: %.4f)\n", result.rho, true_params.rho, fabs(result.rho - true_params.rho));
    printf("V0     = %.4f  (true: %.4f, err: %.4f)\n", result.V0, true_params.V0, fabs(result.V0 - true_params.V0));
    printf("RMSE   = %.6f\n", result.rmse);
    printf("MAE    = %.6f\n", result.mae);
    printf("Iters  = %d\n", result.iterations);
    printf("Conv   = %s\n", result.converged ? "true" : "false");

    printf("\n========================================\n");
    printf("Example completed successfully!\n");
    printf("========================================\n");

    return 0;
}
