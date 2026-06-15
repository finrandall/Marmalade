import numpy as np
from numba import njit, prange


@njit(parallel=True)
def heun_step(S, S_new, S_tilde, r, neighbour_list, H_eff, H_eff_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu):

    n_sites = S.shape[0]
    n_neigh = neighbour_list.shape[1]
    pref = -gamma / (1.0 + lam * lam)

    for i in prange(n_sites):
        hx = 0.0
        hy = 0.0
        hz = 0.0

        for k in range(n_neigh):
            j = neighbour_list[i, k]
            hx += S[j, 0]
            hy += S[j, 1]
            hz += S[j, 2]

        H_eff[i, 0] = J_mu * hx
        H_eff[i, 1] = J_mu * hy
        H_eff[i, 2] = J_mu * hz + 2.0 * K_mu * S[i, 2]

    for i in prange(n_sites):
        hx = H_eff[i, 0] + r[i, 0]
        hy = H_eff[i, 1] + r[i, 1]
        hz = H_eff[i, 2] + r[i, 2]

        sx = S[i, 0]
        sy = S[i, 1]
        sz = S[i, 2]

        ax = sy * hz - sz * hy
        ay = sz * hx - sx * hz
        az = sx * hy - sy * hx

        bx = sy * az - sz * ay
        by = sz * ax - sx * az
        bz = sx * ay - sy * ax

        dS1[i, 0] = pref * (ax + lam * bx)
        dS1[i, 1] = pref * (ay + lam * by)
        dS1[i, 2] = pref * (az + lam * bz)

    for i in prange(n_sites):
        x = S[i, 0] + dt * dS1[i, 0]
        y = S[i, 1] + dt * dS1[i, 1]
        z = S[i, 2] + dt * dS1[i, 2]

        inv = 1.0 / np.sqrt(x * x + y * y + z * z)

        S_tilde[i, 0] = x * inv
        S_tilde[i, 1] = y * inv
        S_tilde[i, 2] = z * inv

    for i in prange(n_sites):
        hx = 0.0
        hy = 0.0
        hz = 0.0

        for k in range(n_neigh):
            j = neighbour_list[i, k]
            hx += S_tilde[j, 0]
            hy += S_tilde[j, 1]
            hz += S_tilde[j, 2]

        H_eff_tilde[i, 0] = J_mu * hx
        H_eff_tilde[i, 1] = J_mu * hy
        H_eff_tilde[i, 2] = J_mu * hz + 2.0 * K_mu * S_tilde[i, 2]

    for i in prange(n_sites):
        hx = H_eff_tilde[i, 0] + r[i, 0]
        hy = H_eff_tilde[i, 1] + r[i, 1]
        hz = H_eff_tilde[i, 2] + r[i, 2]

        sx = S_tilde[i, 0]
        sy = S_tilde[i, 1]
        sz = S_tilde[i, 2]

        ax = sy * hz - sz * hy
        ay = sz * hx - sx * hz
        az = sx * hy - sy * hx

        bx = sy * az - sz * ay
        by = sz * ax - sx * az
        bz = sx * ay - sy * ax

        dS2[i, 0] = pref * (ax + lam * bx)
        dS2[i, 1] = pref * (ay + lam * by)
        dS2[i, 2] = pref * (az + lam * bz)

    for i in prange(n_sites):
        x = S[i, 0] + 0.5 * dt * (dS1[i, 0] + dS2[i, 0])
        y = S[i, 1] + 0.5 * dt * (dS1[i, 1] + dS2[i, 1])
        z = S[i, 2] + 0.5 * dt * (dS1[i, 2] + dS2[i, 2])

        inv = 1.0 / np.sqrt(x * x + y * y + z * z)

        S_new[i, 0] = x * inv
        S_new[i, 1] = y * inv
        S_new[i, 2] = z * inv