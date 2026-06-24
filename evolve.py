import numpy as np

from numba import njit

from integrators import heun_step
from magnetisation import store_magnetisation_sample
from magnetisation import store_energy_sample
from noise import quantum_noise_field
from noise import quantum_noise_update
from noise import classical_noise_field


@njit
def store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp):
    n_sites = S.shape[0]

    for i in range(n_sites):
        Sx_samp[sample, i] = S[i, 0]
        Sy_samp[sample, i] = S[i, 1]
        Sz_samp[sample, i] = S[i, 2]


@njit
def evolve_quantum_time_series(S, nn, nnn, z5, v5, z6, v6, t_series, Mx_series, My_series, Mz_series, M_series, E_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, q_pref, dt_q, amp5, amp6, iterations, stride):
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
            store_magnetisation_sample(S, sample, Mx_series, My_series, Mz_series, M_series)
            store_energy_sample(S, nn, nnn, sample, E_series, J1_mu, J2_mu, K_mu, h)
            sample += 1


@njit
def evolve_classical_time_series(S, nn, nnn, t_series, Mx_series, My_series, Mz_series, M_series, E_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, c_pref, iterations, stride):
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
            store_magnetisation_sample(S, sample, Mx_series, My_series, Mz_series, M_series)
            store_energy_sample(S, nn, nnn, sample, E_series, J1_mu, J2_mu, K_mu, h)
            sample += 1


@njit
def evolve_deterministic_time_series(S, nn, nnn, t_series, Mx_series, My_series, Mz_series, M_series, E_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, iterations, stride):
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
            store_magnetisation_sample(S, sample, Mx_series, My_series, Mz_series, M_series)
            store_energy_sample(S, nn, nnn, sample, E_series, J1_mu, J2_mu, K_mu, h)
            sample += 1


@njit
def evolve_quantum_spin_samples(S, nn, nnn, z5, v5, z6, v6, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, q_pref, dt_q, amp5, amp6, iterations, burn_in_steps, stride):
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

        if step >= burn_in_steps and ((step - burn_in_steps) % stride == 0):
            if sample < Sx_samp.shape[0]:
                store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp)
                sample += 1


@njit
def evolve_classical_spin_samples(S, nn, nnn, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, c_pref, iterations, burn_in_steps, stride):
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

        if step >= burn_in_steps and ((step - burn_in_steps) % stride == 0):
            if sample < Sx_samp.shape[0]:
                store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp)
                sample += 1


@njit
def evolve_deterministic_spin_samples(S, nn, nnn, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J1_mu, J2_mu, K_mu, h, iterations, burn_in_steps, stride):
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

        if step >= burn_in_steps and ((step - burn_in_steps) % stride == 0):
            if sample < Sx_samp.shape[0]:
                store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp)
                sample += 1