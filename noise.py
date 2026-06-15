import numpy as np
from numba import njit, prange


c5 = 1.8315
c6 = 0.3429

Omega5 = 2.7189
Omega6 = 1.2223

Gamma5 = 5.0142
Gamma6 = 3.2974


def init_quantum_noise(N, seed=None):
    rng = np.random.default_rng(seed)

    z5 = rng.normal(0.0, 1.0 / Omega5, size=(N, 3)).astype(np.float64)
    v5 = rng.normal(0.0, 1.0, size=(N, 3)).astype(np.float64)

    z6 = rng.normal(0.0, 1.0 / Omega6, size=(N, 3)).astype(np.float64)
    v6 = rng.normal(0.0, 1.0, size=(N, 3)).astype(np.float64)

    return z5, v5, z6, v6


def quantum_noise_amplitudes(dt_q):
    amp5 = np.sqrt(2.0 * Gamma5 * dt_q)
    amp6 = np.sqrt(2.0 * Gamma6 * dt_q)

    return amp5, amp6


@njit(parallel=True)
def quantum_noise_field(z5, z6, r, q_pref):
    n_sites = z5.shape[0]

    for i in prange(n_sites):
        for a in range(3):
            r[i, a] = q_pref * (c5 * z5[i, a] + c6 * z6[i, a])


@njit(parallel=True)
def quantum_noise_update(z5, v5, z6, v6, dt_q, amp5, amp6):
    n_sites = z5.shape[0]

    for i in prange(n_sites):
        for a in range(3):
            xi5 = np.random.normal(0.0, 1.0)
            xi6 = np.random.normal(0.0, 1.0)

            z5_old = z5[i, a]
            v5_old = v5[i, a]

            z6_old = z6[i, a]
            v6_old = v6[i, a]

            z5[i, a] = z5_old + dt_q * v5_old
            v5[i, a] = v5_old - dt_q * (Omega5 * Omega5 * z5_old + Gamma5 * v5_old) + amp5 * xi5

            z6[i, a] = z6_old + dt_q * v6_old
            v6[i, a] = v6_old - dt_q * (Omega6 * Omega6 * z6_old + Gamma6 * v6_old) + amp6 * xi6


@njit(parallel=True)
def classical_noise_field(r, c_pref):
    n_sites = r.shape[0]

    for i in prange(n_sites):
        for a in range(3):
            r[i, a] = c_pref * np.random.normal(0.0, 1.0)