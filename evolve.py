import numpy as np

from numba import njit

from integrators import heun_step

from noise import quantum_noise_field
from noise import quantum_noise_update
from noise import classical_noise_field


@njit
def compute_magnetisation(S):
    mx = 0.0
    my = 0.0
    mz = 0.0

    n_sites = S.shape[0]

    for i in range(n_sites):
        mx += S[i, 0]
        my += S[i, 1]
        mz += S[i, 2]

    mx /= n_sites
    my /= n_sites
    mz /= n_sites

    return np.sqrt(mx * mx + my * my + mz * mz)


@njit
def evolve_quantum_time_series(S, nn, nnn, z5, v5, z6, v6, t_series, M_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, q_pref, dt_q, amp5, amp6, iterations, stride):
    n_sites = S.shape[0]

    H_eff = np.empty((n_sites, 3), dtype=np.float64)
    H_eff_tilde = np.empty((n_sites, 3), dtype=np.float64)

    H_tot = np.empty((n_sites, 3), dtype=np.float64)
    H_tot_tilde = np.empty((n_sites, 3), dtype=np.float64)

    dS1 = np.empty((n_sites, 3), dtype=np.float64)
    dS2 = np.empty((n_sites, 3), dtype=np.float64)

    S_tilde = np.empty((n_sites, 3), dtype=np.float64)
    S_new = np.empty((n_sites, 3), dtype=np.float64)

    r = np.empty((n_sites, 3), dtype=np.float64)

    sample = 0

    for step in range(iterations):
        quantum_noise_field(z5, z6, r, q_pref)

        heun_step(S, S_new, S_tilde, r, nn, nnn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J1_mu, J2_mu, K_mu, h)

        S, S_new = S_new, S

        quantum_noise_update(z5, v5, z6, v6, dt_q, amp5, amp6)

        if step % stride == 0:
            t_series[sample] = step * dt
            M_series[sample] = compute_magnetisation(S)
            sample += 1


@njit
def evolve_classical_time_series(S, nn, nnn, t_series, M_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, c_pref, iterations, stride):
    n_sites = S.shape[0]

    H_eff = np.empty((n_sites, 3), dtype=np.float64)
    H_eff_tilde = np.empty((n_sites, 3), dtype=np.float64)

    H_tot = np.empty((n_sites, 3), dtype=np.float64)
    H_tot_tilde = np.empty((n_sites, 3), dtype=np.float64)

    dS1 = np.empty((n_sites, 3), dtype=np.float64)
    dS2 = np.empty((n_sites, 3), dtype=np.float64)

    S_tilde = np.empty((n_sites, 3), dtype=np.float64)
    S_new = np.empty((n_sites, 3), dtype=np.float64)

    r = np.empty((n_sites, 3), dtype=np.float64)

    sample = 0

    for step in range(iterations):
        classical_noise_field(r, c_pref)

        heun_step(S, S_new, S_tilde, r, nn, nnn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J1_mu, J2_mu, K_mu, h)

        S, S_new = S_new, S

        if step % stride == 0:
            t_series[sample] = step * dt
            M_series[sample] = compute_magnetisation(S)
            sample += 1


@njit
def evolve_deterministic_time_series(S, nn, nnn, t_series, M_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, iterations, stride):
    n_sites = S.shape[0]

    H_eff = np.empty((n_sites, 3), dtype=np.float64)
    H_eff_tilde = np.empty((n_sites, 3), dtype=np.float64)

    H_tot = np.empty((n_sites, 3), dtype=np.float64)
    H_tot_tilde = np.empty((n_sites, 3), dtype=np.float64)

    dS1 = np.empty((n_sites, 3), dtype=np.float64)
    dS2 = np.empty((n_sites, 3), dtype=np.float64)

    S_tilde = np.empty((n_sites, 3), dtype=np.float64)
    S_new = np.empty((n_sites, 3), dtype=np.float64)

    r = np.zeros((n_sites, 3), dtype=np.float64)

    sample = 0

    for step in range(iterations):
        heun_step(S, S_new, S_tilde, r, nn, nnn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J1_mu, J2_mu, K_mu, h)

        S, S_new = S_new, S

        if step % stride == 0:
            t_series[sample] = step * dt
            M_series[sample] = compute_magnetisation(S)
            sample += 1