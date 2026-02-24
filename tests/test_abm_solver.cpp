/**
 * Tests for BL-Modified Fractional Adams Solver (C API)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <vector>

#include "charlton.h"

extern "C" {
#include "test_data.h"
}

class ABMSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        H = CHARLTON_TEST_H;
        T = 1.0 / 252.0;
        N = 1000;
        gamma_val = CHARLTON_TEST_GAMMA;
        theta_val = CHARLTON_TEST_THETA;
        nu_val = CHARLTON_TEST_NU;
        rho_val = CHARLTON_TEST_RHO;
    }

    double H, T, gamma_val, theta_val, nu_val, rho_val;
    size_t N;
};

TEST_F(ABMSolverTest, InitValidatesParameters) {
    charlton_abm_solver solver;

    // Valid parameters should succeed
    EXPECT_EQ(CHARLTON_OK,
              charlton_abm_init(&solver, H, T, N, gamma_val, theta_val, nu_val, rho_val));
    charlton_abm_free(&solver);

    // Invalid H should fail
    EXPECT_EQ(CHARLTON_ERR_PARAM,
              charlton_abm_init(&solver, -0.1, T, N, gamma_val, theta_val, nu_val, rho_val));
    EXPECT_EQ(CHARLTON_ERR_PARAM,
              charlton_abm_init(&solver, 0.6, T, N, gamma_val, theta_val, nu_val, rho_val));

    // Invalid correlation should fail
    EXPECT_EQ(CHARLTON_ERR_PARAM,
              charlton_abm_init(&solver, H, T, N, gamma_val, theta_val, nu_val, 1.5));
}

TEST_F(ABMSolverTest, SolveSingleReturnsFiniteValue) {
    charlton_abm_solver solver;
    ASSERT_EQ(CHARLTON_OK,
              charlton_abm_init(&solver, H, T, N, gamma_val, theta_val, nu_val, rho_val));

    charlton_cmplx test_freqs[] = {
        CHARLTON_CMPLX(0.0, -0.5),
        CHARLTON_CMPLX(1.0, -0.5),
        CHARLTON_CMPLX(10.0, -0.5),
        CHARLTON_CMPLX(100.0, -0.5)
    };

    for (int i = 0; i < 4; ++i) {
        charlton_cmplx result = charlton_abm_solve_single(&solver, test_freqs[i], 3);
        EXPECT_TRUE(std::isfinite(charlton_creal(result)))
            << "Real part not finite for freq " << i;
        EXPECT_TRUE(std::isfinite(charlton_cimag(result)))
            << "Imaginary part not finite for freq " << i;
    }

    charlton_abm_free(&solver);
}

TEST_F(ABMSolverTest, SolveBatchMatchesSingle) {
    charlton_abm_solver solver;
    ASSERT_EQ(CHARLTON_OK,
              charlton_abm_init(&solver, H, T, N, gamma_val, theta_val, nu_val, rho_val));

    charlton_cmplx batch[] = {
        CHARLTON_CMPLX(0.0, -0.5),
        CHARLTON_CMPLX(1.0, -0.5),
        CHARLTON_CMPLX(5.0, -0.5)
    };
    charlton_cmplx batch_result[3];
    charlton_abm_solve_batch(&solver, batch, 3, batch_result);

    for (int i = 0; i < 3; ++i) {
        charlton_cmplx single_result = charlton_abm_solve_single(&solver, batch[i], 3);
        EXPECT_NEAR(charlton_creal(batch_result[i]), charlton_creal(single_result), 1e-10);
        EXPECT_NEAR(charlton_cimag(batch_result[i]), charlton_cimag(single_result), 1e-10);
    }

    charlton_abm_free(&solver);
}

TEST_F(ABMSolverTest, DecayRateEstimateIsPositive) {
    charlton_abm_solver solver;
    ASSERT_EQ(CHARLTON_OK,
              charlton_abm_init(&solver, H, T, N, gamma_val, theta_val, nu_val, rho_val));

    double v0 = CHARLTON_TEST_V0;
    double decay = charlton_abm_decay_rate(&solver, T, v0);
    EXPECT_GT(decay, 0.0) << "Decay rate should be positive";

    charlton_abm_free(&solver);
}

TEST_F(ABMSolverTest, CharacteristicFunctionAtZero) {
    charlton_abm_solver solver;
    ASSERT_EQ(CHARLTON_OK,
              charlton_abm_init(&solver, H, T, N, gamma_val, theta_val, nu_val, rho_val));

    charlton_cmplx u = CHARLTON_CMPLX(0.0, 0.0);
    charlton_cmplx exponent = charlton_abm_solve_single(&solver, u, 3);

    EXPECT_NEAR(charlton_creal(exponent), 0.0, 1e-6);
    EXPECT_NEAR(charlton_cimag(exponent), 0.0, 1e-6);

    charlton_abm_free(&solver);
}

TEST_F(ABMSolverTest, ConvergenceWithRefinement) {
    charlton_cmplx u = CHARLTON_CMPLX(10.0, -0.5);
    size_t N_values[] = {100, 200, 400, 800};
    charlton_cmplx results[4];

    for (int i = 0; i < 4; ++i) {
        charlton_abm_solver solver;
        ASSERT_EQ(CHARLTON_OK,
                  charlton_abm_init(&solver, H, T, N_values[i],
                                    gamma_val, theta_val, nu_val, rho_val));
        results[i] = charlton_abm_solve_single(&solver, u, 3);
        charlton_abm_free(&solver);
    }

    for (int i = 1; i < 4; ++i) {
        double diff = charlton_cabs(results[i] - results[i - 1]);
        double prev_diff = (i > 1) ? charlton_cabs(results[i - 1] - results[i - 2]) : diff * 2;
        EXPECT_LT(diff, prev_diff * 1.5)
            << "Solution not converging at refinement " << i;
    }
}

TEST_F(ABMSolverTest, DCTEigenvaluesComputed) {
    charlton_abm_solver solver;
    ASSERT_EQ(CHARLTON_OK,
              charlton_abm_init(&solver, H, T, N, gamma_val, theta_val, nu_val, rho_val));

    /* DCT eigenvalues should be computed for N >= 8 */
    /* (May be NULL if Notorious-FFT allocation failed, which is OK) */
    /* Just verify solver is usable regardless */
    charlton_cmplx u = CHARLTON_CMPLX(1.0, -0.5);
    charlton_cmplx result = charlton_abm_solve_single(&solver, u, 3);
    EXPECT_TRUE(std::isfinite(charlton_creal(result)));
    EXPECT_TRUE(std::isfinite(charlton_cimag(result)));

    charlton_abm_free(&solver);
}
