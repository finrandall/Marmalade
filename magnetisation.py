import matplotlib.pyplot as plt
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


def plot_magnetisation_components(t_series, Mx_series, My_series, Mz_series, M_series):
    plt.figure(figsize=(7, 4), dpi=150)

    plt.plot(t_series * 1e12, Mx_series, label=r"$M_x$")
    plt.plot(t_series * 1e12, My_series, label=r"$M_y$")
    plt.plot(t_series * 1e12, Mz_series, label=r"$M_z$")
    plt.plot(t_series * 1e12, M_series, label=r"$|\mathbf{M}|$", linewidth=1.5)

    plt.xlabel("Time (ps)")
    plt.ylabel("Magnetisation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_magnetisation_magnitude(t_series, M_series):
    plt.figure(figsize=(7, 4), dpi=150)

    plt.plot(t_series * 1e12, M_series)

    plt.xlabel("Time (ps)")
    plt.ylabel(r"$|M|$")
    plt.tight_layout()


def plot_transverse_magnetisation(t_series, Mx_series, My_series):
    Mxy_series = np.sqrt(Mx_series * Mx_series + My_series * My_series)

    plt.figure(figsize=(7, 4), dpi=150)

    plt.plot(t_series * 1e12, Mxy_series)

    plt.xlabel("Time (ps)")
    plt.ylabel(r"$M_\perp$")
    plt.tight_layout()


@njit
def compute_total_energy(S, nn, nnn, J1, J2, K, h):
    E = 0.0
    n_sites = S.shape[0]

    for i in range(n_sites):
        for a in range(nn.shape[1]):
            j = nn[i, a]

            if j != -1 and j > i:
                E -= J1 * (S[i, 0] * S[j, 0] + S[i, 1] * S[j, 1] + S[i, 2] * S[j, 2])

        for a in range(nnn.shape[1]):
            j = nnn[i, a]

            if j != -1 and j > i:
                E -= J2 * (S[i, 0] * S[j, 0] + S[i, 1] * S[j, 1] + S[i, 2] * S[j, 2])

        E -= K * S[i, 2] * S[i, 2]
        E -= h * S[i, 2]

    return E


@njit
def store_energy_sample(S, nn, nnn, sample, E_series, J1, J2, K, h):
    E_series[sample] = compute_total_energy(S, nn, nnn, J1, J2, K, h)


def plot_energy(t_series, E_series):
    plt.figure(figsize=(7, 4), dpi=150)

    plt.plot(t_series * 1e12, E_series)

    plt.xlabel("Time (ps)")
    plt.ylabel("Energy")
    plt.tight_layout()