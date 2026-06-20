import numpy as np

from numba import njit, prange


@njit(parallel=True)
def effective_field(S, nn, nnn, H, J1_mu, J2_mu, K_mu, h_mu):
    N_local = S.shape[0]

    for i in prange(N_local):
        hx1 = 0.0
        hy1 = 0.0
        hz1 = 0.0

        hx2 = 0.0
        hy2 = 0.0
        hz2 = 0.0

        for a in range(nn.shape[1]):
            j = nn[i, a]
            hx1 += S[j, 0]
            hy1 += S[j, 1]
            hz1 += S[j, 2]

        for a in range(nnn.shape[1]):
            j = nnn[i, a]
            hx2 += S[j, 0]
            hy2 += S[j, 1]
            hz2 += S[j, 2]

        H[i, 0] = J1_mu * hx1 + J2_mu * hx2
        H[i, 1] = J1_mu * hy1 + J2_mu * hy2
        H[i, 2] = J1_mu * hz1 + J2_mu * hz2 + K_mu * S[i, 2] + h_mu