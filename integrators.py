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


@njit
def rodrigues_rotate(phi, s, out):
    angle = np.sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2])

    if angle < 1e-14:
        out[0] = s[0] + phi[1] * s[2] - phi[2] * s[1]
        out[1] = s[1] + phi[2] * s[0] - phi[0] * s[2]
        out[2] = s[2] + phi[0] * s[1] - phi[1] * s[0]
        return

    ux = phi[0] / angle
    uy = phi[1] / angle
    uz = phi[2] / angle

    c = np.cos(angle)
    q = np.sin(angle)
    dot = ux * s[0] + uy * s[1] + uz * s[2]

    out[0] = s[0] * c + (uy * s[2] - uz * s[1]) * q + ux * dot * (1.0 - c)
    out[1] = s[1] * c + (uz * s[0] - ux * s[2]) * q + uy * dot * (1.0 - c)
    out[2] = s[2] * c + (ux * s[1] - uy * s[0]) * q + uz * dot * (1.0 - c)


@njit
def dexp_inv_so3(phi, v, k):
    px = phi[0]
    py = phi[1]
    pz = phi[2]

    vx = v[0]
    vy = v[1]
    vz = v[2]

    cx = py * vz - pz * vy
    cy = pz * vx - px * vz
    cz = px * vy - py * vx

    c2x = py * cz - pz * cy
    c2y = pz * cx - px * cz
    c2z = px * cy - py * cx

    k[0] = vx - 0.5 * cx + (1.0 / 12.0) * c2x
    k[1] = vy - 0.5 * cy + (1.0 / 12.0) * c2y
    k[2] = vz - 0.5 * cz + (1.0 / 12.0) * c2z


@njit
def llg_omega(S, H, gamma, lam, omega):
    n_sites = S.shape[0]

    pref = gamma / (1.0 + lam * lam)

    for i in range(n_sites):
        sx = S[i, 0]
        sy = S[i, 1]
        sz = S[i, 2]

        hx = H[i, 0]
        hy = H[i, 1]
        hz = H[i, 2]

        cx = sy * hz - sz * hy
        cy = sz * hx - sx * hz
        cz = sx * hy - sy * hx

        omega[i, 0] = pref * (hx + lam * cx)
        omega[i, 1] = pref * (hy + lam * cy)
        omega[i, 2] = pref * (hz + lam * cz)


@njit
def rkmk2_step(S, S_new, S_mid, S_init, phi, k, nn, nnn, H_eff, H_mid, omega, omega_mid, dt, gamma, lam, J1_mu, J2_mu, K_mu, h_mu):
    n_sites = S.shape[0]

    effective_field(S, nn, nnn, H_eff, J1_mu, J2_mu, K_mu, h_mu)
    llg_omega(S, H_eff, gamma, lam, omega)

    for i in range(n_sites):
        S_init[i, 0] = S[i, 0]
        S_init[i, 1] = S[i, 1]
        S_init[i, 2] = S[i, 2]

        phi[i, 0] = 0.5 * dt * omega[i, 0]
        phi[i, 1] = 0.5 * dt * omega[i, 1]
        phi[i, 2] = 0.5 * dt * omega[i, 2]

        rodrigues_rotate(phi[i], S[i], S_mid[i])

    effective_field(S_mid, nn, nnn, H_mid, J1_mu, J2_mu, K_mu, h_mu)
    llg_omega(S_mid, H_mid, gamma, lam, omega_mid)

    for i in range(n_sites):
        k[i, 0] = dt * omega_mid[i, 0]
        k[i, 1] = dt * omega_mid[i, 1]
        k[i, 2] = dt * omega_mid[i, 2]

        dexp_inv_so3(phi[i], k[i], k[i])
        rodrigues_rotate(k[i], S_init[i], S_new[i])

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