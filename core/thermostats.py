import numpy as np
from numba import njit, prange


c5 = 1.8315
c6 = 0.3429

Omega5 = 2.7189
Omega6 = 1.2223

Gamma5 = 5.0142
Gamma6 = 3.2974

QUANTUM_THERMOSTAT_INTEGRATOR = "rk4"
QUANTUM_THERMOSTAT_SCHEME = "rk4_constant_gaussian_drive_v1"


def quantum_thermostat_metadata():
    return {
        "quantum_thermostat_integrator": np.array(QUANTUM_THERMOSTAT_INTEGRATOR),
        "quantum_thermostat_scheme": np.array(QUANTUM_THERMOSTAT_SCHEME),
    }


def init_quantum_noise(N, seed=None):
    rng = np.random.default_rng(seed)

    z5 = rng.normal(0.0, 1.0 / Omega5, size=(N, 3)).astype(np.float64)
    v5 = rng.normal(0.0, 1.0, size=(N, 3)).astype(np.float64)

    z6 = rng.normal(0.0, 1.0 / Omega6, size=(N, 3)).astype(np.float64)
    v6 = rng.normal(0.0, 1.0, size=(N, 3)).astype(np.float64)

    return z5, v5, z6, v6


def quantum_noise_amplitudes(dt_q):
    amp5 = np.sqrt(2.0 * Gamma5 * dt_q)
    amp6 = np.sqrt(2.0 * Gamma6 * dt_q)

    return amp5, amp6


@njit(cache=True, inline="always")
def _quantum_noise_field_at_site(
    i, z5, z6, r, q_pref,
):
    for a in range(3):
        r[i, a] = q_pref * (c5 * z5[i, a] + c6 * z6[i, a])


@njit(cache=True, parallel=True)
def quantum_noise_field(
    z5, z6, r, q_pref,
):
    n_sites = z5.shape[0]
    for i in prange(n_sites):
        _quantum_noise_field_at_site(
            i, z5, z6, r, q_pref,
        )


@njit(cache=True)
def quantum_noise_field_serial(
    z5, z6, r, q_pref,
):
    n_sites = z5.shape[0]
    for i in range(n_sites):
        _quantum_noise_field_at_site(
            i, z5, z6, r, q_pref,
        )


_TWO_PI = 2.0 * np.pi
_TWO_NEGATIVE_53 = 1.0 / 9007199254740992.0


def initialise_rng_states(number_of_sites, seed=None):
    rng = np.random.default_rng(seed)
    states = rng.integers(
        0, np.iinfo(np.uint64).max, size=(number_of_sites, 2), dtype=np.uint64
    )
    both_zero = (states[:, 0] == 0) & (states[:, 1] == 0)
    states[both_zero, 1] = np.uint64(1)
    return states


@njit(cache=True, inline="always")
def _rotate_left(value, shift):
    return (value << shift) | (value >> (64 - shift))


@njit(cache=True, inline="always")
def _next_uint64(state_0, state_1):
    sample = state_0 + state_1
    state_1 ^= state_0
    state_0 = _rotate_left(state_0, 55) ^ state_1 ^ (state_1 << 14)
    state_1 = _rotate_left(state_1, 36)
    return sample, state_0, state_1


@njit(cache=True, inline="always")
def _uniform_open(sample):
    return ((sample >> 11) + 0.5) * _TWO_NEGATIVE_53


@njit(cache=True, inline="always")
def _normal_pair(state_0, state_1):
    sample_1, state_0, state_1 = _next_uint64(state_0, state_1)
    sample_2, state_0, state_1 = _next_uint64(state_0, state_1)
    radius = np.sqrt(-2.0 * np.log(_uniform_open(sample_1)))
    angle = _TWO_PI * _uniform_open(sample_2)
    return (
        radius * np.cos(angle), radius * np.sin(angle), state_0, state_1
    )


@njit(cache=True, inline="always")
def _classical_noise_field_at_site(
    site, rng_states, field, prefactor,
):
    state_0 = rng_states[site, 0]
    state_1 = rng_states[site, 1]

    normal_0, normal_1, state_0, state_1 = _normal_pair(
        state_0, state_1
    )
    normal_2, _, state_0, state_1 = _normal_pair(state_0, state_1)
    field[site, 0] = prefactor * normal_0
    field[site, 1] = prefactor * normal_1
    field[site, 2] = prefactor * normal_2

    rng_states[site, 0] = state_0
    rng_states[site, 1] = state_1


@njit(cache=True, parallel=True)
def classical_noise_field(
    rng_states, field, prefactor,
):
    for site in prange(field.shape[0]):
        _classical_noise_field_at_site(
            site, rng_states, field, prefactor,
        )


@njit(cache=True)
def classical_noise_field_serial(
    rng_states, field, prefactor,
):
    for site in range(field.shape[0]):
        _classical_noise_field_at_site(
            site, rng_states, field, prefactor,
        )


@njit(cache=True, inline="always")
def quantum_oscillator_rk4(z, v, impulse, dt_q, omega, damping):
    omega_squared = omega * omega
    k1_z = dt_q * v
    k1_v = impulse - dt_q * (omega_squared * z + damping * v)
    k2_z = dt_q * (v + 0.5 * k1_v)
    k2_v = impulse - dt_q * (
        omega_squared * (z + 0.5 * k1_z) + damping * (v + 0.5 * k1_v)
    )
    k3_z = dt_q * (v + 0.5 * k2_v)
    k3_v = impulse - dt_q * (
        omega_squared * (z + 0.5 * k2_z) + damping * (v + 0.5 * k2_v)
    )
    k4_z = dt_q * (v + k3_v)
    k4_v = impulse - dt_q * (
        omega_squared * (z + k3_z) + damping * (v + k3_v)
    )
    return (
        z + (k1_z + 2.0*k2_z + 2.0*k3_z + k4_z) / 6.0,
        v + (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v) / 6.0,
    )


@njit(cache=True)
def quantum_oscillator_map(dt_q, amplitude, omega, damping):
    zz, vz = quantum_oscillator_rk4(1.0, 0.0, 0.0, dt_q, omega, damping)
    zv, vv = quantum_oscillator_rk4(0.0, 1.0, 0.0, dt_q, omega, damping)
    zi, vi = quantum_oscillator_rk4(0.0, 0.0, amplitude, dt_q, omega, damping)
    return zz, zv, zi, vz, vv, vi


@njit(cache=True, inline="always")
def _quantum_noise_map_at_site(
    site, rng_states, z5, v5, z6, v6,
    zz5, zv5, zi5, vz5, vv5, vi5, zz6, zv6, zi6, vz6, vv6, vi6,
):
    state_0 = rng_states[site, 0]
    state_1 = rng_states[site, 1]
    for component in range(3):
        xi5, xi6, state_0, state_1 = _normal_pair(state_0, state_1)
        z = z5[site, component]
        v = v5[site, component]
        z5[site, component] = zz5 * z + zv5 * v + zi5 * xi5
        v5[site, component] = vz5 * z + vv5 * v + vi5 * xi5
        z = z6[site, component]
        v = v6[site, component]
        z6[site, component] = zz6 * z + zv6 * v + zi6 * xi6
        v6[site, component] = vz6 * z + vv6 * v + vi6 * xi6

    rng_states[site, 0] = state_0
    rng_states[site, 1] = state_1


@njit(cache=True, inline="always")
def _quantum_noise_update_at_site(
    site, rng_states, z5, v5, z6, v6, dt_q, amp5, amp6, omega5, omega6, gamma5, gamma6,
):
    state_0 = rng_states[site, 0]
    state_1 = rng_states[site, 1]
    for component in range(3):
        xi5, xi6, state_0, state_1 = _normal_pair(state_0, state_1)

        z5[site, component], v5[site, component] = quantum_oscillator_rk4(
            z5[site, component], v5[site, component], amp5 * xi5,
            dt_q, omega5, gamma5,
        )
        z6[site, component], v6[site, component] = quantum_oscillator_rk4(
            z6[site, component], v6[site, component], amp6 * xi6,
            dt_q, omega6, gamma6,
        )

    rng_states[site, 0] = state_0
    rng_states[site, 1] = state_1


@njit(cache=True, parallel=True)
def quantum_noise_update(
    rng_states, z5, v5, z6, v6, dt_q, amp5, amp6, omega5, omega6, gamma5, gamma6,
):
    zz5, zv5, zi5, vz5, vv5, vi5 = quantum_oscillator_map(dt_q, amp5, omega5, gamma5)
    zz6, zv6, zi6, vz6, vv6, vi6 = quantum_oscillator_map(dt_q, amp6, omega6, gamma6)
    for site in prange(z5.shape[0]):
        _quantum_noise_map_at_site(
            site, rng_states, z5, v5, z6, v6,
            zz5, zv5, zi5, vz5, vv5, vi5, zz6, zv6, zi6, vz6, vv6, vi6,
        )


@njit(cache=True)
def quantum_noise_update_serial(
    rng_states, z5, v5, z6, v6, dt_q, amp5, amp6, omega5, omega6, gamma5, gamma6,
):
    zz5, zv5, zi5, vz5, vv5, vi5 = quantum_oscillator_map(dt_q, amp5, omega5, gamma5)
    zz6, zv6, zi6, vz6, vv6, vi6 = quantum_oscillator_map(dt_q, amp6, omega6, gamma6)
    for site in range(z5.shape[0]):
        _quantum_noise_map_at_site(
            site, rng_states, z5, v5, z6, v6,
            zz5, zv5, zi5, vz5, vv5, vi5, zz6, zv6, zi6, vz6, vv6, vi6,
        )
