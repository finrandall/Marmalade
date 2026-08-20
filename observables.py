import numpy as np

from numba import njit


@njit
def compute_magnetisation_components(S):
    mx = 0.0
    my = 0.0
    mz = 0.0

    n_sites = S.shape[0]

    for i in range(n_sites):
        mx += S[i, 0]
        my += S[i, 1]
        mz += S[i, 2]

    mx /= n_sites
    my /= n_sites
    mz /= n_sites

    m_abs = np.sqrt(mx * mx + my * my + mz * mz)

    return mx, my, mz, m_abs


@njit
def store_magnetisation_sample(S, sample, Mx_series, My_series, Mz_series, M_series):
    mx, my, mz, m_abs = compute_magnetisation_components(S)

    Mx_series[sample] = mx
    My_series[sample] = my
    Mz_series[sample] = mz
    M_series[sample] = m_abs


@njit
def compute_energy(S, nn, J, K, h):
    E = 0.0

    for i in range(S.shape[0]):
        for a in range(nn.shape[1]):
            j = nn[i, a]
            if j != -1:
                E -= 0.5 * J * (
                    S[i, 0] * S[j, 0]
                    + S[i, 1] * S[j, 1]
                    + S[i, 2] * S[j, 2]
                )
        E -= 0.5 * K * S[i, 2] * S[i, 2]
        E -= h * S[i, 2]

    return E


def normalised_energy(E, n_sites, J):
    return np.asarray(E) / (n_sites * J)


def energy_drift(E, n_sites, J):
    E = np.asarray(E)
    return (E - E[0]) / (n_sites * J)


def analytic_1d_heisenberg_energy(T, n_sites, J, pbc, kB):
    if T <= 0.0:
        raise ValueError("T must be greater than zero.")
    if J <= 0.0:
        raise ValueError("This benchmark assumes ferromagnetic J > 0.")

    n_bonds = n_sites if pbc else n_sites - 1
    x = J / (kB * T)
    E_per_bond = J * (1.0 / x - 1.0 / np.tanh(x))
    E = n_bonds * E_per_bond
    return E / (n_sites * J), E


def energy(S, nn, J, h, mu=None):
    E = 0.0

    counted = set()

    for i in range(S.shape[0]):
        for j in nn[i]:
            if j >= 0 and (j, i) not in counted:
                E += -J * np.dot(S[i], S[j])
                counted.add((i, j))

    if mu is None:
        raise ValueError("mu must be provided because h is now a magnetic field in tesla.")

    E += -mu * h * np.sum(S[:, 2])

    return E
