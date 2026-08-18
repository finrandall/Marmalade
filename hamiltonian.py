import numpy as np

from numba import njit, prange


@njit(parallel=True)
def effective_field(S, nn, H, J_mu, K_mu, h_mu):
    N_local = S.shape[0]

    for i in prange(N_local):
        hx1 = 0.0
        hy1 = 0.0
        hz1 = 0.0

        for a in range(nn.shape[1]):
            j = nn[i, a]
            if j != -1:
                hx1 += S[j, 0]
                hy1 += S[j, 1]
                hz1 += S[j, 2]

        H[i, 0] = J_mu * hx1
        H[i, 1] = J_mu * hy1
        H[i, 2] = J_mu * hz1 + K_mu * S[i, 2] + h_mu