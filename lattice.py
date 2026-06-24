import numpy as np
from numba import njit



@njit
def site_index(ix, iy, Lx):
    return ix + Lx * iy


@njit
def boundary_index(ix, iy, Lx, Ly, pbc_x, pbc_y):
    if pbc_x:
        ix = ix % Lx
    elif ix < 0 or ix >= Lx:
        return -1

    if pbc_y:
        iy = iy % Ly
    elif iy < 0 or iy >= Ly:
        return -1

    return site_index(ix, iy, Lx)


@njit
def build_neighbour_list(Lx, Ly, pbc_x=True, pbc_y=True):
    N = Lx * Ly
    nn = np.empty((N, 4), dtype=np.int32)
    nnn = np.empty((N, 4), dtype=np.int32)

    for iy in range(Ly):
        for ix in range(Lx):
            i = site_index(ix, iy, Lx)

            nn[i, 0] = boundary_index(ix + 1, iy, Lx, Ly, pbc_x, pbc_y)
            nn[i, 1] = boundary_index(ix - 1, iy, Lx, Ly, pbc_x, pbc_y)
            nn[i, 2] = boundary_index(ix, iy + 1, Lx, Ly, pbc_x, pbc_y)
            nn[i, 3] = boundary_index(ix, iy - 1, Lx, Ly, pbc_x, pbc_y)

            nnn[i, 0] = boundary_index(ix + 1, iy + 1, Lx, Ly, pbc_x, pbc_y)
            nnn[i, 1] = boundary_index(ix + 1, iy - 1, Lx, Ly, pbc_x, pbc_y)
            nnn[i, 2] = boundary_index(ix - 1, iy + 1, Lx, Ly, pbc_x, pbc_y)
            nnn[i, 3] = boundary_index(ix - 1, iy - 1, Lx, Ly, pbc_x, pbc_y)

    return nn, nnn

def neighbour_statistics(nn, nnn):
    nn_entries = np.sum(nn != -1)
    nnn_entries = np.sum(nnn != -1)

    return {
        "nn_entries": nn_entries,
        "nn_bonds": nn_entries // 2,
        "nnn_entries": nnn_entries,
        "nnn_bonds": nnn_entries // 2}