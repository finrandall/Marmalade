import numpy as np

from numba import njit, prange

from hamiltonian import effective_field
from llg import llg_rhs


@njit(parallel=True)
def heun_step(S, S_new, S_tilde, r, nn, H_eff, H_eff_tilde, H_tot, H_tot_tilde, dS1, dS2, dt, gamma, lam, J_mu, K_mu, h):
    n_sites = S.shape[0]

    effective_field(S, nn, H_eff, J_mu, K_mu, h)

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

    effective_field(S_tilde, nn, H_eff_tilde, J_mu, K_mu, h)

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



@njit
def classical_noise_rodrigues_step(S, r, gamma, lam, dt_noise):
    n_sites = S.shape[0]
    pref = gamma / (1.0 + lam * lam)

    for i in range(n_sites):
        sx = S[i, 0]
        sy = S[i, 1]
        sz = S[i, 2]

        hx = r[i, 0]
        hy = r[i, 1]
        hz = r[i, 2]

        cx = sy * hz - sz * hy
        cy = sz * hx - sx * hz
        cz = sx * hy - sy * hx

        phi_x = dt_noise * pref * (hx + lam * cx)
        phi_y = dt_noise * pref * (hy + lam * cy)
        phi_z = dt_noise * pref * (hz + lam * cz)

        angle = np.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)

        if angle < 1e-14:
            out_x = sx + phi_y * sz - phi_z * sy
            out_y = sy + phi_z * sx - phi_x * sz
            out_z = sz + phi_x * sy - phi_y * sx
        else:
            ux = phi_x / angle
            uy = phi_y / angle
            uz = phi_z / angle

            c = np.cos(angle)
            q = np.sin(angle)
            dot = ux * sx + uy * sy + uz * sz

            out_x = sx * c + (uy * sz - uz * sy) * q + ux * dot * (1.0 - c)
            out_y = sy * c + (uz * sx - ux * sz) * q + uy * dot * (1.0 - c)
            out_z = sz * c + (ux * sy - uy * sx) * q + uz * dot * (1.0 - c)

        inv = 1.0 / np.sqrt(out_x * out_x + out_y * out_y + out_z * out_z)

        S[i, 0] = out_x * inv
        S[i, 1] = out_y * inv
        S[i, 2] = out_z * inv