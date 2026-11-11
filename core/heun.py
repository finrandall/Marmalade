import numpy as np
from numba import njit, prange

from .hamiltonian import effective_field, effective_field_at_site
from .llg import llg_rhs, llg_rhs_at_site as _llg_rhs_at_site
from .thermostats import (
    _classical_noise_field_at_site, _quantum_noise_field_at_site,
    _quantum_noise_map_at_site, quantum_oscillator_map,
)


@njit(cache=True, parallel=True)
def heun_step_shells(
    S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
    field_tesla, dS1, dt, gamma, damping,
):
    prefactor = -gamma / (1.0 + damping * damping)

    for site in prange(S.shape[0]):
        hx, hy, hz = effective_field_at_site(
            S, site, row_ptr, neighbours, bond_shell, J_mu_shell,
            lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla,
        )
        hx += random_field[site, 0]
        hy += random_field[site, 1]
        hz += random_field[site, 2]
        sx = S[site, 0]
        sy = S[site, 1]
        sz = S[site, 2]
        dx, dy, dz = _llg_rhs_at_site(
            sx, sy, sz, hx, hy, hz, prefactor, damping
        )
        dS1[site, 0] = dx
        dS1[site, 1] = dy
        dS1[site, 2] = dz

        x = sx + dt * dx
        y = sy + dt * dy
        z = sz + dt * dz
        inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
        S_tilde[site, 0] = x * inverse_norm
        S_tilde[site, 1] = y * inverse_norm
        S_tilde[site, 2] = z * inverse_norm

    for site in prange(S.shape[0]):
        hx, hy, hz = effective_field_at_site(
            S_tilde, site, row_ptr, neighbours, bond_shell, J_mu_shell,
            lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla,
        )
        hx += random_field[site, 0]
        hy += random_field[site, 1]
        hz += random_field[site, 2]
        sx_tilde = S_tilde[site, 0]
        sy_tilde = S_tilde[site, 1]
        sz_tilde = S_tilde[site, 2]
        dx, dy, dz = _llg_rhs_at_site(
            sx_tilde, sy_tilde, sz_tilde, hx, hy, hz, prefactor, damping
        )

        x = S[site, 0] + 0.5 * dt * (dS1[site, 0] + dx)
        y = S[site, 1] + 0.5 * dt * (dS1[site, 1] + dy)
        z = S[site, 2] + 0.5 * dt * (dS1[site, 2] + dz)
        inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
        S_new[site, 0] = x * inverse_norm
        S_new[site, 1] = y * inverse_norm
        S_new[site, 2] = z * inverse_norm


@njit(cache=True, inline="always")
def _effective_field_and_energy_at_site(
    S, site, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    D_mu, anisotropy_axis, field_tesla, J_shell, lambda_shell, D,
    magnetic_moment,
):
    hx = field_tesla[0]
    hy = field_tesla[1]
    hz = field_tesla[2]
    sx = S[site, 0]
    sy = S[site, 1]
    sz = S[site, 2]
    site_energy = 0.0

    for edge in range(row_ptr[site], row_ptr[site + 1]):
        neighbour = neighbours[edge]
        shell = bond_shell[edge]
        neighbour_x = S[neighbour, 0]
        neighbour_y = S[neighbour, 1]
        neighbour_z = S[neighbour, 2]
        coupling = J_mu_shell[shell]
        hx -= coupling * neighbour_x
        hy -= coupling * neighbour_y
        hz -= coupling * neighbour_z
        hz -= lambda_mu_shell[shell] * neighbour_z
        dot = sx * neighbour_x + sy * neighbour_y + sz * neighbour_z
        site_energy += 0.5 * J_shell[shell] * dot
        site_energy += 0.5 * lambda_shell[shell] * sz * neighbour_z

    nx = anisotropy_axis[0]
    ny = anisotropy_axis[1]
    nz = anisotropy_axis[2]
    projection = sx * nx + sy * ny + sz * nz
    anisotropy_field = -2.0 * D_mu * projection
    site_energy += D * projection * projection
    site_energy -= magnetic_moment * (
        field_tesla[0] * sx + field_tesla[1] * sy + field_tesla[2] * sz
    )
    return (
        hx + anisotropy_field * nx,
        hy + anisotropy_field * ny,
        hz + anisotropy_field * nz,
        site_energy,
    )


@njit(cache=True, parallel=True)
def heun_step_shells_observe(
    S, S_new, S_tilde, random_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
    field_tesla, dS1, dt, gamma, damping, J_shell, lambda_shell, D,
    magnetic_moment,
):
    prefactor = -gamma / (1.0 + damping * damping)
    mx = 0.0
    my = 0.0
    mz = 0.0
    energy = 0.0

    for site in prange(S.shape[0]):
        hx, hy, hz, site_energy = _effective_field_and_energy_at_site(
            S, site, row_ptr, neighbours, bond_shell, J_mu_shell,
            lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
            J_shell, lambda_shell, D, magnetic_moment,
        )
        energy += site_energy
        hx += random_field[site, 0]
        hy += random_field[site, 1]
        hz += random_field[site, 2]
        sx = S[site, 0]
        sy = S[site, 1]
        sz = S[site, 2]
        mx += sx
        my += sy
        mz += sz
        dx, dy, dz = _llg_rhs_at_site(
            sx, sy, sz, hx, hy, hz, prefactor, damping
        )
        dS1[site, 0] = dx
        dS1[site, 1] = dy
        dS1[site, 2] = dz

        x = sx + dt * dx
        y = sy + dt * dy
        z = sz + dt * dz
        inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
        S_tilde[site, 0] = x * inverse_norm
        S_tilde[site, 1] = y * inverse_norm
        S_tilde[site, 2] = z * inverse_norm

    for site in prange(S.shape[0]):
        hx, hy, hz = effective_field_at_site(
            S_tilde, site, row_ptr, neighbours, bond_shell, J_mu_shell,
            lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla,
        )
        hx += random_field[site, 0]
        hy += random_field[site, 1]
        hz += random_field[site, 2]
        sx_tilde = S_tilde[site, 0]
        sy_tilde = S_tilde[site, 1]
        sz_tilde = S_tilde[site, 2]
        dx, dy, dz = _llg_rhs_at_site(
            sx_tilde, sy_tilde, sz_tilde, hx, hy, hz, prefactor, damping
        )

        x = S[site, 0] + 0.5 * dt * (dS1[site, 0] + dx)
        y = S[site, 1] + 0.5 * dt * (dS1[site, 1] + dy)
        z = S[site, 2] + 0.5 * dt * (dS1[site, 2] + dz)
        inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
        S_new[site, 0] = x * inverse_norm
        S_new[site, 1] = y * inverse_norm
        S_new[site, 2] = z * inverse_norm

    number_of_sites = S.shape[0]
    mx /= number_of_sites
    my /= number_of_sites
    mz /= number_of_sites
    return mx, my, mz, np.sqrt(mx * mx + my * my + mz * mz), energy


@njit(cache=True, inline="always")
def _heun_predictor_shells_at_site(
    site, prefactor, S, S_tilde, dS1, random_field, dipolar_field, row_ptr, neighbours,
    bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma,
    damping,
):
    hx, hy, hz = effective_field_at_site(
        S, site, row_ptr, neighbours, bond_shell, J_mu_shell,
        lambda_mu_shell, D_mu, anisotropy_axis,
        field_tesla,
    )
    hx += random_field[site, 0] + dipolar_field[site, 0]
    hy += random_field[site, 1] + dipolar_field[site, 1]
    hz += random_field[site, 2] + dipolar_field[site, 2]
    sx = S[site, 0]
    sy = S[site, 1]
    sz = S[site, 2]
    dx, dy, dz = _llg_rhs_at_site(
        sx, sy, sz, hx, hy, hz, prefactor, damping
    )
    dS1[site, 0] = dx
    dS1[site, 1] = dy
    dS1[site, 2] = dz

    x = sx + dt * dx
    y = sy + dt * dy
    z = sz + dt * dz
    inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
    S_tilde[site, 0] = x * inverse_norm
    S_tilde[site, 1] = y * inverse_norm
    S_tilde[site, 2] = z * inverse_norm


@njit(cache=True, parallel=True)
def heun_predictor_shells(
    S, S_tilde, dS1, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma, damping,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in prange(S.shape[0]):
        _heun_predictor_shells_at_site(
            site, prefactor, S, S_tilde, dS1, random_field, dipolar_field, row_ptr,
            neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, dt, gamma, damping,
        )


@njit(cache=True)
def heun_predictor_shells_serial(
    S, S_tilde, dS1, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma, damping,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in range(S.shape[0]):
        _heun_predictor_shells_at_site(
            site, prefactor, S, S_tilde, dS1, random_field, dipolar_field, row_ptr,
            neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, dt, gamma, damping,
        )


@njit(cache=True, inline="always")
def _heun_corrector_shells_at_site(
    site, prefactor, S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde, row_ptr,
    neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
    dt, gamma, damping,
):
    hx, hy, hz = effective_field_at_site(
        S_tilde, site, row_ptr, neighbours, bond_shell, J_mu_shell,
        lambda_mu_shell, D_mu, anisotropy_axis,
        field_tesla,
    )
    hx += random_field[site, 0] + dipolar_field_tilde[site, 0]
    hy += random_field[site, 1] + dipolar_field_tilde[site, 1]
    hz += random_field[site, 2] + dipolar_field_tilde[site, 2]
    sx = S_tilde[site, 0]
    sy = S_tilde[site, 1]
    sz = S_tilde[site, 2]
    dx, dy, dz = _llg_rhs_at_site(
        sx, sy, sz, hx, hy, hz, prefactor, damping
    )

    x = S[site, 0] + 0.5 * dt * (dS1[site, 0] + dx)
    y = S[site, 1] + 0.5 * dt * (dS1[site, 1] + dy)
    z = S[site, 2] + 0.5 * dt * (dS1[site, 2] + dz)
    inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
    S_new[site, 0] = x * inverse_norm
    S_new[site, 1] = y * inverse_norm
    S_new[site, 2] = z * inverse_norm


@njit(cache=True, parallel=True)
def heun_corrector_shells(
    S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde, row_ptr, neighbours,
    bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma,
    damping,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in prange(S.shape[0]):
        _heun_corrector_shells_at_site(
            site, prefactor, S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde,
            row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu,
            anisotropy_axis, field_tesla, dt, gamma, damping,
        )


@njit(cache=True)
def heun_corrector_shells_serial(
    S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde, row_ptr, neighbours,
    bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma,
    damping,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in range(S.shape[0]):
        _heun_corrector_shells_at_site(
            site, prefactor, S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde,
            row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu,
            anisotropy_axis, field_tesla, dt, gamma, damping,
        )


@njit(cache=True, parallel=True)
def heun_predictor_classical_shells(
    S, S_tilde, dS1, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma, damping,
    rng_states, noise_prefactor,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in prange(S.shape[0]):
        _classical_noise_field_at_site(site, rng_states, random_field, noise_prefactor)
        _heun_predictor_shells_at_site(
            site, prefactor, S, S_tilde, dS1, random_field, dipolar_field, row_ptr,
            neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, dt, gamma, damping,
        )


@njit(cache=True)
def heun_predictor_classical_shells_serial(
    S, S_tilde, dS1, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma, damping,
    rng_states, noise_prefactor,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in range(S.shape[0]):
        _classical_noise_field_at_site(site, rng_states, random_field, noise_prefactor)
        _heun_predictor_shells_at_site(
            site, prefactor, S, S_tilde, dS1, random_field, dipolar_field, row_ptr,
            neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, dt, gamma, damping,
        )


@njit(cache=True, parallel=True)
def heun_predictor_quantum_shells(
    S, S_tilde, dS1, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma, damping,
    z5, z6, noise_prefactor,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in prange(S.shape[0]):
        _quantum_noise_field_at_site(site, z5, z6, random_field, noise_prefactor)
        _heun_predictor_shells_at_site(
            site, prefactor, S, S_tilde, dS1, random_field, dipolar_field, row_ptr,
            neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, dt, gamma, damping,
        )


@njit(cache=True)
def heun_predictor_quantum_shells_serial(
    S, S_tilde, dS1, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma, damping,
    z5, z6, noise_prefactor,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in range(S.shape[0]):
        _quantum_noise_field_at_site(site, z5, z6, random_field, noise_prefactor)
        _heun_predictor_shells_at_site(
            site, prefactor, S, S_tilde, dS1, random_field, dipolar_field, row_ptr,
            neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, dt, gamma, damping,
        )


@njit(cache=True, parallel=True)
def heun_corrector_quantum_shells(
    S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde, row_ptr, neighbours,
    bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma,
    damping, rng_states, z5, v5, z6, v6, dt_q, amp5, amp6, omega5, omega6, gamma5, gamma6,
):
    prefactor = -gamma / (1.0 + damping * damping)
    zz5, zv5, zi5, vz5, vv5, vi5 = quantum_oscillator_map(dt_q, amp5, omega5, gamma5)
    zz6, zv6, zi6, vz6, vv6, vi6 = quantum_oscillator_map(dt_q, amp6, omega6, gamma6)
    for site in prange(S.shape[0]):
        _heun_corrector_shells_at_site(
            site, prefactor, S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde,
            row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu,
            anisotropy_axis, field_tesla, dt, gamma, damping,
        )
        _quantum_noise_map_at_site(
            site, rng_states, z5, v5, z6, v6,
            zz5, zv5, zi5, vz5, vv5, vi5, zz6, zv6, zi6, vz6, vv6, vi6,
        )


@njit(cache=True)
def heun_corrector_quantum_shells_serial(
    S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde, row_ptr, neighbours,
    bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, dt, gamma,
    damping, rng_states, z5, v5, z6, v6, dt_q, amp5, amp6, omega5, omega6, gamma5, gamma6,
):
    prefactor = -gamma / (1.0 + damping * damping)
    zz5, zv5, zi5, vz5, vv5, vi5 = quantum_oscillator_map(dt_q, amp5, omega5, gamma5)
    zz6, zv6, zi6, vz6, vv6, vi6 = quantum_oscillator_map(dt_q, amp6, omega6, gamma6)
    for site in range(S.shape[0]):
        _heun_corrector_shells_at_site(
            site, prefactor, S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde,
            row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu,
            anisotropy_axis, field_tesla, dt, gamma, damping,
        )
        _quantum_noise_map_at_site(
            site, rng_states, z5, v5, z6, v6,
            zz5, zv5, zi5, vz5, vv5, vi5, zz6, zv6, zi6, vz6, vv6, vi6,
        )


@njit(cache=True, parallel=True)
def heun_step(S, S_new, S_tilde, r, row_ptr, neighbours, J_mu, lambda_mu,
              D_mu, anisotropy_axis, field_tesla, H_eff,
              H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam):
    number_of_sites = S.shape[0]
    effective_field(
        S, row_ptr, neighbours, J_mu, lambda_mu, H_eff, D_mu,
        anisotropy_axis, field_tesla,
    )

    for i in prange(number_of_sites):
        H_tot[i, 0] = H_eff[i, 0] + r[i, 0]
        H_tot[i, 1] = H_eff[i, 1] + r[i, 1]
        H_tot[i, 2] = H_eff[i, 2] + r[i, 2]

    llg_rhs(S, H_tot, dS1, gamma, lam)

    for i in prange(number_of_sites):
        x = S[i, 0] + dt * dS1[i, 0]
        y = S[i, 1] + dt * dS1[i, 1]
        z = S[i, 2] + dt * dS1[i, 2]
        inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
        S_tilde[i, 0] = x * inverse_norm
        S_tilde[i, 1] = y * inverse_norm
        S_tilde[i, 2] = z * inverse_norm

    effective_field(
        S_tilde, row_ptr, neighbours, J_mu, lambda_mu,
        H_eff_tilde, D_mu,
        anisotropy_axis, field_tesla,
    )

    for i in prange(number_of_sites):
        H_tot_tilde[i, 0] = H_eff_tilde[i, 0] + r[i, 0]
        H_tot_tilde[i, 1] = H_eff_tilde[i, 1] + r[i, 1]
        H_tot_tilde[i, 2] = H_eff_tilde[i, 2] + r[i, 2]

    llg_rhs(S_tilde, H_tot_tilde, dS2, gamma, lam)

    for i in prange(number_of_sites):
        x = S[i, 0] + 0.5 * dt * (dS1[i, 0] + dS2[i, 0])
        y = S[i, 1] + 0.5 * dt * (dS1[i, 1] + dS2[i, 1])
        z = S[i, 2] + 0.5 * dt * (dS1[i, 2] + dS2[i, 2])
        inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
        S_new[i, 0] = x * inverse_norm
        S_new[i, 1] = y * inverse_norm
        S_new[i, 2] = z * inverse_norm
