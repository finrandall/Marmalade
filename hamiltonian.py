import numpy as np

from numba import njit, prange


@njit(parallel=True)
def effective_field(S, nn, nnn, H, J1_mu, J2_mu, K_mu, h):
    n_sites = S.shape[0]

    for i in prange(n_sites):
        hx = 0.0
        hy = 0.0
        hz = 0.0

        for k in range(nn.shape[1]):
            j = nn[i, k]
            hx += S[j, 0]
            hy += S[j, 1]
            hz += S[j, 2]

        hx *= J1_mu
        hy *= J1_mu
        hz *= J1_mu

        for k in range(nnn.shape[1]):
            j = nnn[i, k]
            hx += J2_mu * S[j, 0]
            hy += J2_mu * S[j, 1]
            hz += J2_mu * S[j, 2]

        H[i, 0] = hx
        H[i, 1] = hy
        H[i, 2] = hz + K_mu * S[i, 2] + h