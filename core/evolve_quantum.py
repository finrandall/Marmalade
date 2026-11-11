import numpy as np
from numba import njit, prange

from .damping import damping_at_time, damping_noise_factor
from .integrators import heun_step_shells, heun_step_shells_observe
from .rk4 import rk4_step_shells
from .thermostats import Gamma5, Gamma6, Omega5, Omega6, quantum_noise_field
from .thermostats import quantum_noise_update
from .sampling import record_observables, store_observables, store_spin_sample


@njit(parallel=True, cache=True)
def add_local_field(field, site_weights, pulse_field_tesla, envelope):
    for site in prange(field.shape[0]):
        weight = envelope * site_weights[site]
        field[site, 0] += weight * pulse_field_tesla[0]
        field[site, 1] += weight * pulse_field_tesla[1]
        field[site, 2] += weight * pulse_field_tesla[2]


@njit(cache=True)
def evolve_quantum_time_series(
    S, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    D_mu, anisotropy_axis, field_tesla,
    z5, v5, z6, v6, rng_states, time_series, Mx_series, My_series, Mz_series,
    M_series, E_series, moments, save_magnetisation, save_energy, dt,
    gamma, lam, J_shell, lambda_shell, D, magnetic_moment, q_pref,
    quantum_noise_scale, damping_initial, damping_decay_time,
    damping_schedule_end_time, time_offset, dt_q, amp5, amp6, iterations,
    burn_in_steps, stride, integrator_code=0,
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
        step_noise_prefactor = q_pref
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
                quantum_noise_scale * damping_noise_factor(step_damping)
            )
        quantum_noise_field(z5, z6, random_field, step_noise_prefactor)
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
        quantum_noise_update(
            rng_states, z5, v5, z6, v6, dt_q, amp5, amp6,
            Omega5, Omega6, Gamma5, Gamma6,
        )
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
def evolve_quantum_spin_samples(
    S, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    D_mu, anisotropy_axis, field_tesla,
    z5, v5, z6, v6, rng_states, Sx_samples, Sy_samples, Sz_samples, dt,
    gamma, lam,
    q_pref, quantum_noise_scale, damping_initial, damping_decay_time,
    damping_schedule_end_time, time_offset, dt_q, amp5, amp6, iterations,
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
        step_noise_prefactor = q_pref
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
                quantum_noise_scale * damping_noise_factor(step_damping)
            )
        quantum_noise_field(z5, z6, random_field, step_noise_prefactor)
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
        quantum_noise_update(
            rng_states, z5, v5, z6, v6, dt_q, amp5, amp6,
            Omega5, Omega6, Gamma5, Gamma6,
        )
        if step >= burn_in_steps and (step - burn_in_steps) % stride == 0:
            store_spin_sample(S, sample, Sx_samples, Sy_samples, Sz_samples)
            sample += 1
    return S


@njit(cache=True)
def evolve_quantum_spin_observable_samples(
    S, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    D_mu, anisotropy_axis, field_tesla,
    z5, v5, z6, v6, rng_states, time_series, Mx_series, My_series, Mz_series,
    M_series, E_series, moments, Sx_samples, Sy_samples, Sz_samples, dt,
    gamma, lam, J_shell, lambda_shell, D, magnetic_moment, q_pref,
    quantum_noise_scale, damping_initial, damping_decay_time,
    damping_schedule_end_time, time_offset, dt_q, amp5, amp6, iterations,
    sampling_step_offset, stride, integrator_code=0,
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
        step_noise_prefactor = q_pref
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
                quantum_noise_scale * damping_noise_factor(step_damping)
            )
        quantum_noise_field(z5, z6, random_field, step_noise_prefactor)
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
        quantum_noise_update(
            rng_states, z5, v5, z6, v6, dt_q, amp5, amp6,
            Omega5, Omega6, Gamma5, Gamma6,
        )
        if (sampling_step_offset + step + 1) % stride == 0:
            sample = record_observables(
                S, row_ptr, neighbours, bond_shell, J_shell, lambda_shell,
                D, anisotropy_axis, field_tesla, magnetic_moment,
                step, dt, sample, True, True, time_series, Mx_series,
                My_series, Mz_series, M_series, E_series, moments,
            )
            store_spin_sample(
                S, sample - 1, Sx_samples, Sy_samples, Sz_samples,
            )
    return S


@njit(cache=True)
def evolve_quantum_local_field_spin_samples(
    S, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    D_mu, anisotropy_axis, field_tesla,
    z5, v5, z6, v6, rng_states, site_weights, pulse_field_tesla,
    pulse_duration, pulse_cycles, Sx_samples, Sy_samples, Sz_samples, dt,
    gamma, lam,
    q_pref, dt_q, amp5, amp6, iterations, stride, integrator_code=0,
):
    number_of_sites = S.shape[0]
    dS1 = np.empty((number_of_sites, 3), dtype=np.float64)
    S_tilde = np.empty((number_of_sites, 3), dtype=np.float64)
    S_new = np.empty((number_of_sites, 3), dtype=np.float64)
    random_field = np.empty((number_of_sites, 3), dtype=np.float64)
    rk4_work = np.empty((3 if integrator_code == 1 else 0, number_of_sites, 3))
    sample = 0

    for step in range(iterations):
        quantum_noise_field(z5, z6, random_field, q_pref)
        pulse_time = (step + 0.5) * dt
        waveform = 0.0
        if pulse_time < pulse_duration:
            phase = np.pi * pulse_time / pulse_duration
            envelope = np.sin(phase) ** 2
            carrier = np.sin(2.0 * pulse_cycles * phase)
            waveform = envelope * carrier
        add_local_field(
            random_field, site_weights, pulse_field_tesla, waveform,
        )
        if integrator_code == 1:
            rk4_step_shells(
                S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
                J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
                dS1, rk4_work[0], rk4_work[1], rk4_work[2], dt, gamma, lam,
            )
        else:
            heun_step_shells(
                S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
                J_mu_shell, lambda_mu_shell, D_mu,
                anisotropy_axis, field_tesla, dS1, dt, gamma, lam,
            )
        S, S_new = S_new, S
        quantum_noise_update(
            rng_states, z5, v5, z6, v6, dt_q, amp5, amp6,
            Omega5, Omega6, Gamma5, Gamma6,
        )
        if (step + 1) % stride == 0:
            store_spin_sample(S, sample, Sx_samples, Sy_samples, Sz_samples)
            sample += 1
    return S
