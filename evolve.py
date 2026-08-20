import numpy as np

from numba import njit

from integrators import heun_step
from observables import compute_energy, compute_magnetisation_components
from noise import quantum_noise_field
from noise import quantum_noise_update
from noise import classical_noise_field
from integrators import classical_noise_rodrigues_step


@njit
def store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp):
    n_sites = S.shape[0]

    for i in range(n_sites):
        Sx_samp[sample, i] = S[i, 0]
        Sy_samp[sample, i] = S[i, 1]
        Sz_samp[sample, i] = S[i, 2]


@njit
def record_observables(S, nn, step, dt, sample, save_magnetisation, save_energy,
                       t_series, Mx_series, My_series, Mz_series, M_series,
                       E_series, moments, J, K, h):
    mx, my, mz, magnitude = compute_magnetisation_components(S)
    values = (mx, my, mz, magnitude)
    moments[0] += 1.0
    for i in range(4):
        moments[1 + i] += values[i]
        moments[5 + i] += values[i] * values[i]
    if save_magnetisation or save_energy:
        t_series[sample] = (step + 1) * dt
    if save_magnetisation:
        Mx_series[sample] = mx
        My_series[sample] = my
        Mz_series[sample] = mz
        M_series[sample] = magnitude
    if save_energy:
        E_series[sample] = compute_energy(S, nn, J, K, h)
    return sample + 1


@njit
def evolve_quantum_time_series(S, nn, z5, v5, z6, v6, t_series, Mx_series, My_series, Mz_series, M_series, E_series, moments, save_magnetisation, save_energy, dt, gamma, lam, J_mu, K_mu, h_mu, J, K, h, q_pref, dt_q, amp5, amp6, iterations, burn_in_steps, stride):
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
        heun_step(S, S_new, S_tilde, r, nn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu, h_mu)

        S, S_new = S_new, S

        quantum_noise_update(z5, v5, z6, v6, dt_q, amp5, amp6)

        if step >= burn_in_steps and (step - burn_in_steps) % stride == 0:
            sample = record_observables(S, nn, step, dt, sample, save_magnetisation,
                save_energy, t_series, Mx_series, My_series, Mz_series, M_series,
                E_series, moments, J, K, h)

    return S


@njit
def evolve_classical_time_series(S, nn, t_series, Mx_series, My_series, Mz_series, M_series, E_series, moments, save_magnetisation, save_energy, dt, gamma, lam, J_mu, K_mu, h_mu, J, K, h, c_pref, iterations, burn_in_steps, stride):
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
        heun_step(S, S_new, S_tilde, r, nn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu, h_mu)

        S, S_new = S_new, S

        if step >= burn_in_steps and (step - burn_in_steps) % stride == 0:
            sample = record_observables(S, nn, step, dt, sample, save_magnetisation,
                save_energy, t_series, Mx_series, My_series, Mz_series, M_series,
                E_series, moments, J, K, h)

    return S


@njit
def evolve_deterministic_heun_time_series(S, nn, t_series, Mx_series, My_series, Mz_series, M_series, E_series, moments, save_magnetisation, save_energy, dt, gamma, lam, J_mu, K_mu, h_mu, J, K, h, iterations, burn_in_steps, stride):
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
        heun_step(S, S_new, S_tilde, r, nn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu, h_mu)

        S, S_new = S_new, S

        if step >= burn_in_steps and (step - burn_in_steps) % stride == 0:
            sample = record_observables(S, nn, step, dt, sample, save_magnetisation,
                save_energy, t_series, Mx_series, My_series, Mz_series, M_series,
                E_series, moments, J, K, h)

    return S


@njit
def evolve_quantum_spin_samples(S, nn, z5, v5, z6, v6, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J_mu, K_mu, h, q_pref, dt_q, amp5, amp6, iterations, burn_in_steps, stride):
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
        heun_step(S, S_new, S_tilde, r, nn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu, h)

        S, S_new = S_new, S

        quantum_noise_update(z5, v5, z6, v6, dt_q, amp5, amp6)

        if step >= burn_in_steps and ((step - burn_in_steps) % stride == 0):
            if sample < Sx_samp.shape[0]:
                store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp)
                sample += 1


@njit
def evolve_classical_spin_samples(S, nn, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J_mu, K_mu, h, c_pref, iterations, burn_in_steps, stride):
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
        heun_step(S, S_new, S_tilde, r, nn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu, h)

        S, S_new = S_new, S

        if step >= burn_in_steps and ((step - burn_in_steps) % stride == 0):
            if sample < Sx_samp.shape[0]:
                store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp)
                sample += 1


@njit
def evolve_deterministic_spin_samples(S, nn, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J_mu, K_mu, h, iterations, burn_in_steps, stride):
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
        heun_step(S, S_new, S_tilde, r, nn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu, h)

        S, S_new = S_new, S

        if step >= burn_in_steps and ((step - burn_in_steps) % stride == 0):
            if sample < Sx_samp.shape[0]:
                store_spin_sample(S, sample, Sx_samp, Sy_samp, Sz_samp)
                sample += 1
