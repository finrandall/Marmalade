import numpy as np

from numba import njit, prange


@njit(parallel=True)
def llg_rhs(S, H, out, gamma, lam):
    n_sites = S.shape[0]
    pref = -gamma / (1.0 + lam * lam)

    for i in prange(n_sites):
        sx = S[i, 0]
        sy = S[i, 1]
        sz = S[i, 2]

        hx = H[i, 0]
        hy = H[i, 1]
        hz = H[i, 2]

        ax = sy * hz - sz * hy
        ay = sz * hx - sx * hz
        az = sx * hy - sy * hx

        bx = sy * az - sz * ay
        by = sz * ax - sx * az
        bz = sx * ay - sy * ax

        out[i, 0] = pref * (ax + lam * bx)
        out[i, 1] = pref * (ay + lam * by)
        out[i, 2] = pref * (az + lam * bz)