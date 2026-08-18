import numpy as np
from numba import njit


@njit
def site_index(ix, iy, iz, Lx, Ly):
    return ix + Lx * (iy + Ly * iz)


@njit
def boundary_index(ix, iy, iz, Lx, Ly, Lz, pbc_x, pbc_y, pbc_z):
    if pbc_x:
        ix = ix % Lx
    elif ix < 0 or ix >= Lx:
        return -1

    if pbc_y:
        iy = iy % Ly
    elif iy < 0 or iy >= Ly:
        return -1

    if pbc_z:
        iz = iz % Lz
    elif iz < 0 or iz >= Lz:
        return -1

    return site_index(ix, iy, iz, Lx, Ly)


@njit
def build_neighbour_list(Lx, Ly, Lz=1, pbc_x=True, pbc_y=True, pbc_z=True):
    N = Lx * Ly * Lz

    if Lz == 1:
        nn = np.empty((N, 4), dtype=np.int32)
    else:
        nn = np.empty((N, 6), dtype=np.int32)

    for iz in range(Lz):
        for iy in range(Ly):
            for ix in range(Lx):
                i = site_index(ix, iy, iz, Lx, Ly)

                nn[i, 0] = boundary_index(ix + 1, iy, iz, Lx, Ly, Lz, pbc_x, pbc_y, pbc_z)
                nn[i, 1] = boundary_index(ix - 1, iy, iz, Lx, Ly, Lz, pbc_x, pbc_y, pbc_z)
                nn[i, 2] = boundary_index(ix, iy + 1, iz, Lx, Ly, Lz, pbc_x, pbc_y, pbc_z)
                nn[i, 3] = boundary_index(ix, iy - 1, iz, Lx, Ly, Lz, pbc_x, pbc_y, pbc_z)

                if Lz != 1:
                    nn[i, 4] = boundary_index(ix, iy, iz + 1, Lx, Ly, Lz, pbc_x, pbc_y, pbc_z)
                    nn[i, 5] = boundary_index(ix, iy, iz - 1, Lx, Ly, Lz, pbc_x, pbc_y, pbc_z)

    return nn


def neighbour_statistics(nn):
    nn_entries = np.sum(nn != -1)

    return {
        "nn_entries": nn_entries,
        "nn_bonds": nn_entries // 2,
    }
