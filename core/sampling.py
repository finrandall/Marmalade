from numba import njit, prange

from .observables import compute_observables_shells


@njit(cache=True, parallel=True)
def store_spin_sample(S, sample, Sx_samples, Sy_samples, Sz_samples):
    for i in prange(S.shape[0]):
        Sx_samples[sample, i] = S[i, 0]
        Sy_samples[sample, i] = S[i, 1]
        Sz_samples[sample, i] = S[i, 2]


@njit(cache=True)
def record_observables(
    S, row_ptr, neighbours, bond_shell, J_shell, lambda_shell,
    D, anisotropy_axis, field_tesla, magnetic_moment, step, dt, sample,
    save_magnetisation, save_energy, time_series, Mx_series, My_series,
    Mz_series, M_series, E_series, moments,
):
    mx, my, mz, magnitude, energy = compute_observables_shells(
        S, row_ptr, neighbours, bond_shell, J_shell, lambda_shell,
        D, anisotropy_axis, field_tesla, magnetic_moment, save_energy,
    )
    return store_observables(
        mx, my, mz, magnitude, energy, step, dt, sample,
        save_magnetisation, save_energy, time_series, Mx_series, My_series,
        Mz_series, M_series, E_series, moments,
    )


@njit(cache=True)
def store_observables(
    mx, my, mz, magnitude, energy, step, dt, sample,
    save_magnetisation, save_energy, time_series, Mx_series, My_series,
    Mz_series, M_series, E_series, moments,
):
    values = (mx, my, mz, magnitude)
    moments[0] += 1.0
    for component in range(4):
        moments[1 + component] += values[component]
        moments[5 + component] += values[component] * values[component]

    if save_magnetisation or save_energy:
        time_series[sample] = (step + 1) * dt
    if save_magnetisation:
        Mx_series[sample] = mx
        My_series[sample] = my
        Mz_series[sample] = mz
        M_series[sample] = magnitude
    if save_energy:
        E_series[sample] = energy
    return sample + 1
