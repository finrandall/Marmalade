import numpy as np
from numba import njit


@njit(cache=True)
def damping_at_time(time, damping_initial, damping_final, decay_time):
    if decay_time <= 0.0 or damping_initial == damping_final:
        return damping_final
    return damping_final + (damping_initial - damping_final) * np.exp(
        -time / decay_time
    )


@njit(cache=True)
def damping_noise_factor(damping):
    return np.sqrt(damping)


@njit(cache=True)
def classical_noise_factor(damping):
    return np.sqrt(damping)


@njit(cache=True)
def damping_at_times(times, damping_initial, damping_final, decay_time):
    result = np.empty(times.shape[0], dtype=np.float64)
    for index in range(times.shape[0]):
        result[index] = damping_at_time(
            times[index], damping_initial, damping_final, decay_time
        )
    return result
