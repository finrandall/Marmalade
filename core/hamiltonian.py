from numba import njit, prange


@njit(inline="always", cache=True)
def effective_field_at_site(
    S, site, row_ptr, neighbours, bond_shell, J_mu_shell,
    lambda_mu_shell, D_mu, anisotropy_axis, field_tesla,
):
    hx = field_tesla[0]
    hy = field_tesla[1]
    hz = field_tesla[2]
    sx = S[site, 0]
    sy = S[site, 1]
    sz = S[site, 2]

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

    nx = anisotropy_axis[0]
    ny = anisotropy_axis[1]
    nz = anisotropy_axis[2]
    projection = sx * nx + sy * ny + sz * nz
    anisotropy_field = -2.0 * D_mu * projection
    return (
        hx + anisotropy_field * nx,
        hy + anisotropy_field * ny,
        hz + anisotropy_field * nz,
    )


@njit(parallel=True, cache=True)
def effective_field_shells(
    S, row_ptr, neighbours, bond_shell, J_mu_shell, lambda_mu_shell,
    H, D_mu, anisotropy_axis, field_tesla,
):
    for site in prange(S.shape[0]):
        hx, hy, hz = effective_field_at_site(
            S, site, row_ptr, neighbours, bond_shell, J_mu_shell,
            lambda_mu_shell, D_mu, anisotropy_axis,
            field_tesla,
        )
        H[site, 0] = hx
        H[site, 1] = hy
        H[site, 2] = hz


@njit(parallel=True, cache=True)
def effective_field(S, row_ptr, neighbours, J_mu, lambda_mu, H,
                    D_mu, anisotropy_axis, field_tesla):
    number_of_sites = S.shape[0]
    nx = anisotropy_axis[0]
    ny = anisotropy_axis[1]
    nz = anisotropy_axis[2]

    for i in prange(number_of_sites):
        hx = field_tesla[0]
        hy = field_tesla[1]
        hz = field_tesla[2]

        for edge in range(row_ptr[i], row_ptr[i + 1]):
            j = neighbours[edge]
            coupling = J_mu[edge]
            hx -= coupling * S[j, 0]
            hy -= coupling * S[j, 1]
            hz -= coupling * S[j, 2]
            hz -= lambda_mu[edge] * S[j, 2]

        projection = S[i, 0] * nx + S[i, 1] * ny + S[i, 2] * nz
        anisotropy_field = -2.0 * D_mu * projection
        H[i, 0] = hx + anisotropy_field * nx
        H[i, 1] = hy + anisotropy_field * ny
        H[i, 2] = hz + anisotropy_field * nz
