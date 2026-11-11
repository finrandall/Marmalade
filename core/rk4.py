import numpy as np
from numba import njit, prange

from .hamiltonian import effective_field_at_site
from .llg import llg_rhs_at_site as _llg_rhs_at_site


@njit(cache=True, inline="always")
def _rk4_rhs_at_site(
    site, S, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
    out, prefactor, damping,
):
    hx, hy, hz = effective_field_at_site(
        S, site, row_ptr, neighbours, bond_shell, J_mu_shell,
        lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
    )
    hx += random_field[site, 0]
    hy += random_field[site, 1]
    hz += random_field[site, 2]
    if dipolar_field is not None:
        hx += dipolar_field[site, 0]
        hy += dipolar_field[site, 1]
        hz += dipolar_field[site, 2]
    dx, dy, dz = _llg_rhs_at_site(
        S[site, 0], S[site, 1], S[site, 2], hx, hy, hz, prefactor, damping,
    )
    out[site, 0] = dx
    out[site, 1] = dy
    out[site, 2] = dz


@njit(cache=True, parallel=True)
def rk4_rhs_shells(
    S, random_field, dipolar_field, out, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, gamma, damping,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in prange(S.shape[0]):
        _rk4_rhs_at_site(
            site, S, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
            J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
            out, prefactor, damping,
        )


@njit(cache=True)
def rk4_rhs_shells_serial(
    S, random_field, dipolar_field, out, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla, gamma, damping,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in range(S.shape[0]):
        _rk4_rhs_at_site(
            site, S, random_field, dipolar_field, row_ptr, neighbours, bond_shell,
            J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
            out, prefactor, damping,
        )


@njit(cache=True, parallel=True)
def rk4_stage(S, derivative, stage, scale):
    for site in prange(S.shape[0]):
        for component in range(3):
            stage[site, component] = S[site, component] + scale * derivative[site, component]


@njit(cache=True)
def rk4_stage_serial(S, derivative, stage, scale):
    for site in range(S.shape[0]):
        for component in range(3):
            stage[site, component] = S[site, component] + scale * derivative[site, component]


@njit(cache=True, inline="always")
def _rk4_finish_at_site(site, S, S_new, k1, k2, k3, k4, dt):
    x = S[site, 0] + dt / 6.0 * (k1[site, 0] + 2*k2[site, 0] + 2*k3[site, 0] + k4[site, 0])
    y = S[site, 1] + dt / 6.0 * (k1[site, 1] + 2*k2[site, 1] + 2*k3[site, 1] + k4[site, 1])
    z = S[site, 2] + dt / 6.0 * (k1[site, 2] + 2*k2[site, 2] + 2*k3[site, 2] + k4[site, 2])
    inverse_norm = 1.0 / np.sqrt(x*x + y*y + z*z)
    S_new[site, 0] = x * inverse_norm
    S_new[site, 1] = y * inverse_norm
    S_new[site, 2] = z * inverse_norm


@njit(cache=True, parallel=True)
def rk4_finish(S, S_new, k1, k2, k3, k4, dt):
    for site in prange(S.shape[0]):
        _rk4_finish_at_site(site, S, S_new, k1, k2, k3, k4, dt)


@njit(cache=True)
def rk4_finish_serial(S, S_new, k1, k2, k3, k4, dt):
    for site in range(S.shape[0]):
        _rk4_finish_at_site(site, S, S_new, k1, k2, k3, k4, dt)


@njit(cache=True, parallel=True)
def rk4_rhs_stage(
    S, S_eval, random_field, dipolar_field, out, stage, row_ptr, neighbours,
    bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
    gamma, damping, scale,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in prange(S.shape[0]):
        _rk4_rhs_at_site(
            site, S_eval, random_field, dipolar_field, row_ptr, neighbours,
            bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, out, prefactor, damping,
        )
        for component in range(3):
            stage[site, component] = S[site, component] + scale * out[site, component]


@njit(cache=True)
def rk4_rhs_stage_serial(
    S, S_eval, random_field, dipolar_field, out, stage, row_ptr, neighbours,
    bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
    gamma, damping, scale,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in range(S.shape[0]):
        _rk4_rhs_at_site(
            site, S_eval, random_field, dipolar_field, row_ptr, neighbours,
            bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, out, prefactor, damping,
        )
        for component in range(3):
            stage[site, component] = S[site, component] + scale * out[site, component]


@njit(cache=True, parallel=True)
def rk4_rhs_finish(
    S, S_eval, random_field, dipolar_field, k1, k2, k3, k4, S_new, row_ptr,
    neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
    field_tesla, gamma, damping, dt,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in prange(S.shape[0]):
        _rk4_rhs_at_site(
            site, S_eval, random_field, dipolar_field, row_ptr, neighbours,
            bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, k4, prefactor, damping,
        )
        _rk4_finish_at_site(site, S, S_new, k1, k2, k3, k4, dt)


@njit(cache=True)
def rk4_rhs_finish_serial(
    S, S_eval, random_field, dipolar_field, k1, k2, k3, k4, S_new, row_ptr,
    neighbours, bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
    field_tesla, gamma, damping, dt,
):
    prefactor = -gamma / (1.0 + damping * damping)
    for site in range(S.shape[0]):
        _rk4_rhs_at_site(
            site, S_eval, random_field, dipolar_field, row_ptr, neighbours,
            bond_shell, J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla, k4, prefactor, damping,
        )
        _rk4_finish_at_site(site, S, S_new, k1, k2, k3, k4, dt)


@njit(cache=True)
def rk4_step_shells(
    S, S_new, S_stage, random_field, row_ptr, neighbours, bond_shell,
    J_mu_shell, lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
    k1, k2, k3, k4, dt, gamma, damping,
):
    common = (row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
              D_mu, anisotropy_axis, field_tesla, gamma, damping)
    rk4_rhs_stage(S, S, random_field, None, k1, S_stage, *common, 0.5*dt)
    rk4_rhs_stage(S, S_stage, random_field, None, k2, S_new, *common, 0.5*dt)
    rk4_rhs_stage(S, S_new, random_field, None, k3, S_stage, *common, dt)
    rk4_rhs_finish(S, S_stage, random_field, None, k1, k2, k3, k4, S_new,
                   *common, dt)


