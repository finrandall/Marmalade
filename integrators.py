import numpy as np

from numba import njit, prange

from hamiltonian import effective_field
from llg import llg_rhs


@njit(parallel=True)
def heun_step(S, S_new, S_tilde, r, nn, nnn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J1_mu, J2_mu, K_mu, h):
    n_sites = S.shape[0]

    effective_field(S, nn, nnn, H_eff, J1_mu, J2_mu, K_mu, h)

    for i in prange(n_sites):
        H_tot[i, 0] = H_eff[i, 0] + r[i, 0]
        H_tot[i, 1] = H_eff[i, 1] + r[i, 1]
        H_tot[i, 2] = H_eff[i, 2] + r[i, 2]

    llg_rhs(S, H_tot, dS1, gamma, lam)

    for i in prange(n_sites):
        x = S[i, 0] + dt * dS1[i, 0]
        y = S[i, 1] + dt * dS1[i, 1]
        z = S[i, 2] + dt * dS1[i, 2]

        inv = 1.0 / np.sqrt(x * x + y * y + z * z)

        S_tilde[i, 0] = x * inv
        S_tilde[i, 1] = y * inv
        S_tilde[i, 2] = z * inv

    effective_field(S_tilde, nn, nnn, H_eff_tilde, J1_mu, J2_mu, K_mu, h)

    for i in prange(n_sites):
        H_tot_tilde[i, 0] = H_eff_tilde[i, 0] + r[i, 0]
        H_tot_tilde[i, 1] = H_eff_tilde[i, 1] + r[i, 1]
        H_tot_tilde[i, 2] = H_eff_tilde[i, 2] + r[i, 2]

    llg_rhs(S_tilde, H_tot_tilde, dS2, gamma, lam)

    for i in prange(n_sites):
        x = S[i, 0] + 0.5 * dt * (dS1[i, 0] + dS2[i, 0])
        y = S[i, 1] + 0.5 * dt * (dS1[i, 1] + dS2[i, 1])
        z = S[i, 2] + 0.5 * dt * (dS1[i, 2] + dS2[i, 2])

        inv = 1.0 / np.sqrt(x * x + y * y + z * z)

        S_new[i, 0] = x * inv
        S_new[i, 1] = y * inv
        S_new[i, 2] = z * inv