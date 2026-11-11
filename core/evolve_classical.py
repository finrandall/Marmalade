import numpy as np
from numba import njit

from .damping import damping_at_time, classical_noise_factor
from .integrators import heun_step_shells, heun_step_shells_observe
from .rk4 import rk4_step_shells
from .thermostats import classical_noise_field
from .sampling import record_observables, store_observables, store_spin_sample


@njit(cache=True)
def evolve_classical_time_series(
    S, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    D_mu, anisotropy_axis, field_tesla, rng_states,
    time_series, Mx_series, My_series, Mz_series, M_series, E_series,
    moments, save_magnetisation, save_energy, dt, gamma, lam, J_shell,
    lambda_shell, D, magnetic_moment, classical_prefactor,
    classical_noise_scale, damping_initial, damping_decay_time,
    damping_schedule_end_time, time_offset, iterations, burn_in_steps, stride,
    integrator_code=0,
):
    number_of_sites = S.shape[0]
    dS1 = np.empty((number_of_sites, 3), dtype=np.float64)
    S_tilde = np.empty((number_of_sites, 3), dtype=np.float64)
    S_new = np.empty((number_of_sites, 3), dtype=np.float64)
    random_field = np.empty((number_of_sites, 3), dtype=np.float64)
    rk4_work = np.empty((3 if integrator_code == 1 else 0, number_of_sites, 3))
    sample = 0
    defer_observables = save_energy and integrator_code == 0
    pending_sample = -1

    for step in range(iterations):
        step_damping = lam
        step_noise_prefactor = classical_prefactor
        schedule_time = time_offset + step * dt
        if (
            damping_decay_time > 0.0
            and (
                damping_schedule_end_time < 0.0
                or schedule_time < damping_schedule_end_time
            )
        ):
            step_damping = damping_at_time(
                schedule_time, damping_initial, lam, damping_decay_time,
            )
            step_noise_prefactor = (
                classical_noise_scale * classical_noise_factor(step_damping)
            )
        classical_noise_field(rng_states, random_field, step_noise_prefactor)
        if integrator_code == 1:
            rk4_step_shells(
                S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
                J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
                dS1, rk4_work[0], rk4_work[1], rk4_work[2], dt, gamma, step_damping,
            )
        elif pending_sample >= 0:
            mx, my, mz, magnitude, energy = heun_step_shells_observe(
                S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
                J_mu_shell, lambda_mu_shell, D_mu,
                anisotropy_axis, field_tesla, dS1, dt, gamma, step_damping,
                J_shell, lambda_shell, D, magnetic_moment,
            )
            sample = store_observables(
                mx, my, mz, magnitude, energy, step - 1, dt, pending_sample,
                save_magnetisation, save_energy, time_series, Mx_series,
                My_series, Mz_series, M_series, E_series, moments,
            )
            pending_sample = -1
        else:
            heun_step_shells(
                S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
                J_mu_shell, lambda_mu_shell, D_mu,
                anisotropy_axis, field_tesla, dS1, dt, gamma, step_damping,
            )
        S, S_new = S_new, S
        if step >= burn_in_steps and (step - burn_in_steps) % stride == 0:
            if defer_observables and step + 1 < iterations:
                pending_sample = sample
            else:
                sample = record_observables(
                    S, row_ptr, neighbours, bond_shell, J_shell, lambda_shell,
                    D, anisotropy_axis, field_tesla, magnetic_moment,
                    step, dt, sample, save_magnetisation, save_energy,
                    time_series, Mx_series, My_series, Mz_series, M_series,
                    E_series, moments,
                )
    return S


@njit(cache=True)
def evolve_classical_spin_samples(
    S, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    D_mu, anisotropy_axis, field_tesla, rng_states,
    Sx_samples, Sy_samples, Sz_samples, dt, gamma, lam,
    classical_prefactor, classical_noise_scale, damping_initial,
    damping_decay_time, damping_schedule_end_time, time_offset, iterations,
    burn_in_steps, stride, integrator_code=0,
):
    number_of_sites = S.shape[0]
    dS1 = np.empty((number_of_sites, 3), dtype=np.float64)
    S_tilde = np.empty((number_of_sites, 3), dtype=np.float64)
    S_new = np.empty((number_of_sites, 3), dtype=np.float64)
    random_field = np.empty((number_of_sites, 3), dtype=np.float64)
    rk4_work = np.empty((3 if integrator_code == 1 else 0, number_of_sites, 3))
    sample = 0

    for step in range(iterations):
        step_damping = lam
        step_noise_prefactor = classical_prefactor
        schedule_time = time_offset + step * dt
        if (
            damping_decay_time > 0.0
            and (
                damping_schedule_end_time < 0.0
                or schedule_time < damping_schedule_end_time
            )
        ):
            step_damping = damping_at_time(
                schedule_time, damping_initial, lam, damping_decay_time,
            )
            step_noise_prefactor = (
                classical_noise_scale * classical_noise_factor(step_damping)
            )
        classical_noise_field(rng_states, random_field, step_noise_prefactor)
        if integrator_code == 1:
            rk4_step_shells(
                S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
                J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
                dS1, rk4_work[0], rk4_work[1], rk4_work[2], dt, gamma, step_damping,
            )
        else:
            heun_step_shells(
                S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
                J_mu_shell, lambda_mu_shell, D_mu,
                anisotropy_axis, field_tesla, dS1, dt, gamma, step_damping,
            )
        S, S_new = S_new, S
        if step >= burn_in_steps and (step - burn_in_steps) % stride == 0:
            store_spin_sample(S, sample, Sx_samples, Sy_samples, Sz_samples)
            sample += 1
    return S
