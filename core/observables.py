import numpy as np
from numba import njit, prange


@njit(cache=True)
def compute_magnetisation_components(S):
    mx = 0.0
    my = 0.0
    mz = 0.0
    number_of_sites = S.shape[0]
    for i in range(number_of_sites):
        mx += S[i, 0]
        my += S[i, 1]
        mz += S[i, 2]
    mx /= number_of_sites
    my /= number_of_sites
    mz /= number_of_sites
    return mx, my, mz, np.sqrt(mx * mx + my * my + mz * mz)


@njit(cache=True)
def compute_energy(S, row_ptr, neighbours, J_edge, lambda_edge,
                   D, anisotropy_axis, field_tesla, magnetic_moment):
    energy = 0.0
    nx = anisotropy_axis[0]
    ny = anisotropy_axis[1]
    nz = anisotropy_axis[2]

    for i in range(S.shape[0]):
        for edge in range(row_ptr[i], row_ptr[i + 1]):
            j = neighbours[edge]
            dot = (
                S[i, 0] * S[j, 0]
                + S[i, 1] * S[j, 1]
                + S[i, 2] * S[j, 2]
            )
            energy += 0.5 * J_edge[edge] * dot
            energy += 0.5 * lambda_edge[edge] * S[i, 2] * S[j, 2]

        projection = S[i, 0] * nx + S[i, 1] * ny + S[i, 2] * nz
        energy += D * projection * projection
        energy -= magnetic_moment * (
            field_tesla[0] * S[i, 0]
            + field_tesla[1] * S[i, 1]
            + field_tesla[2] * S[i, 2]
        )
    return energy


@njit(parallel=True, cache=True)
def compute_energy_shells(
    S, row_ptr, neighbours, bond_shell, J_shell, lambda_shell,
    D, anisotropy_axis, field_tesla, magnetic_moment,
):
    energy = 0.0
    nx = anisotropy_axis[0]
    ny = anisotropy_axis[1]
    nz = anisotropy_axis[2]

    for i in prange(S.shape[0]):
        site_energy = 0.0
        for edge in range(row_ptr[i], row_ptr[i + 1]):
            j = neighbours[edge]
            shell = bond_shell[edge]
            dot = (
                S[i, 0] * S[j, 0]
                + S[i, 1] * S[j, 1]
                + S[i, 2] * S[j, 2]
            )
            site_energy += 0.5 * J_shell[shell] * dot
            site_energy += 0.5 * lambda_shell[shell] * S[i, 2] * S[j, 2]

        projection = S[i, 0] * nx + S[i, 1] * ny + S[i, 2] * nz
        site_energy += D * projection * projection
        site_energy -= magnetic_moment * (
            field_tesla[0] * S[i, 0]
            + field_tesla[1] * S[i, 1]
            + field_tesla[2] * S[i, 2]
        )
        energy += site_energy
    return energy


@njit(parallel=True, cache=True)
def compute_observables_shells(
    S, row_ptr, neighbours, bond_shell, J_shell, lambda_shell,
    D, anisotropy_axis, field_tesla, magnetic_moment, include_energy,
):
    mx = 0.0
    my = 0.0
    mz = 0.0
    energy = 0.0
    nx = anisotropy_axis[0]
    ny = anisotropy_axis[1]
    nz = anisotropy_axis[2]
    number_of_sites = S.shape[0]

    for i in prange(number_of_sites):
        sx = S[i, 0]
        sy = S[i, 1]
        sz = S[i, 2]
        mx += sx
        my += sy
        mz += sz
        site_energy = 0.0
        if include_energy:
            for edge in range(row_ptr[i], row_ptr[i + 1]):
                j = neighbours[edge]
                shell = bond_shell[edge]
                dot = (
                    S[i, 0] * S[j, 0]
                    + S[i, 1] * S[j, 1]
                    + S[i, 2] * S[j, 2]
                )
                site_energy += 0.5 * J_shell[shell] * dot
                site_energy += 0.5 * lambda_shell[shell] * S[i, 2] * S[j, 2]
            projection = sx * nx + sy * ny + sz * nz
            site_energy += D * projection * projection
            site_energy -= magnetic_moment * (
                field_tesla[0] * sx + field_tesla[1] * sy + field_tesla[2] * sz
            )
        energy += site_energy
    mx /= number_of_sites
    my /= number_of_sites
    mz /= number_of_sites
    return mx, my, mz, np.sqrt(mx * mx + my * my + mz * mz), energy


@njit(parallel=True, cache=True)
def energy_of_spin_samples(
    spin_x, spin_y, spin_z, row_ptr, neighbours, bond_shell, J_shell,
    lambda_shell, D, anisotropy_axis, field_tesla, magnetic_moment,
):
    count = spin_x.shape[0]
    result = np.empty(count, dtype=np.float64)
    nx = anisotropy_axis[0]
    ny = anisotropy_axis[1]
    nz = anisotropy_axis[2]
    for sample in prange(count):
        energy = 0.0
        for i in range(spin_x.shape[1]):
            sx = spin_x[sample, i]
            sy = spin_y[sample, i]
            sz = spin_z[sample, i]
            site_energy = 0.0
            for edge in range(row_ptr[i], row_ptr[i + 1]):
                j = neighbours[edge]
                shell = bond_shell[edge]
                tx = spin_x[sample, j]
                ty = spin_y[sample, j]
                tz = spin_z[sample, j]
                dot = sx * tx + sy * ty + sz * tz
                site_energy += 0.5 * J_shell[shell] * dot
                site_energy += 0.5 * lambda_shell[shell] * sz * tz
            projection = sx * nx + sy * ny + sz * nz
            site_energy += D * projection * projection
            site_energy -= magnetic_moment * (
                field_tesla[0] * sx + field_tesla[1] * sy + field_tesla[2] * sz
            )
            energy += site_energy
        result[sample] = energy
    return result


def sublattice_magnetisation(S, basis_indices):
    spins = np.asarray(S)
    basis_indices = np.asarray(basis_indices)
    return np.vstack(
        (spins[basis_indices == 0].mean(axis=0),
         spins[basis_indices == 1].mean(axis=0))
    )
