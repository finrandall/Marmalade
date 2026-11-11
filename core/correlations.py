import operator

import numpy as np


_WORKSPACE_BYTES = 32 * 1024 * 1024


def _batch_length(batch_size, count, bytes_per_item):
    if batch_size is None:
        size = max(1, _WORKSPACE_BYTES // max(1, bytes_per_item))
    else:
        try:
            size = operator.index(batch_size)
        except TypeError as error:
            raise ValueError("batch_size must be a positive integer") from error
        if size < 1:
            raise ValueError("batch_size must be a positive integer")
    return max(1, min(count, size))


def _require_finite(values, message):
    row_size = max(1, int(np.prod(values.shape[1:], dtype=np.int64)))
    batch_size = _batch_length(None, values.shape[0], row_size)
    for start in range(0, values.shape[0], batch_size):
        if not np.all(np.isfinite(values[start:start + batch_size])):
            raise ValueError(message)


def _lag_count(number_of_samples, maximum_lag):
    if maximum_lag is None:
        maximum_lag = number_of_samples - 1
    maximum_lag = int(maximum_lag)
    if not 0 <= maximum_lag < number_of_samples:
        raise ValueError("maximum_lag must lie between 0 and time_samples - 1")
    return maximum_lag + 1


def _column_batch(values, start, stop):
    if values.ndim == 1:
        return values[:, None]
    if values.flags.c_contiguous:
        return values.reshape(values.shape[0], -1)[:, start:stop]
    indices = np.unravel_index(np.arange(start, stop), values.shape[1:])
    return values[(slice(None),) + indices]


def _spin_components(spin_x, spin_y, spin_z):
    components = tuple(
        np.asarray(component) for component in (spin_x, spin_y, spin_z)
    )
    shape = components[0].shape
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 1:
        raise ValueError("spin components must have shape (time, site)")
    if any(component.shape != shape for component in components[1:]):
        raise ValueError("spin components must have identical shapes")
    for component in components:
        _require_finite(component, "spin components must contain only finite values")
    return components


def equilibrium_axis(spin_x, spin_y, spin_z):
    components = _spin_components(spin_x, spin_y, spin_z)
    magnetisation = np.array(
        [component.mean(dtype=np.float64) for component in components]
    )
    magnitude = np.linalg.norm(magnetisation)
    if magnitude <= 100.0 * np.finfo(np.float64).eps:
        raise ValueError(
            "mean magnetisation is too small to define a longitudinal axis"
        )
    return magnetisation / magnitude


def reciprocal_supercell_vectors(supercell_vectors_angstrom):
    supercell = np.asarray(supercell_vectors_angstrom, dtype=np.float64)
    if supercell.shape not in ((2, 2), (2, 3)):
        raise ValueError("supercell vectors must have shape (2, 2) or (2, 3)")
    in_plane = supercell[:, :2]
    if abs(np.linalg.det(in_plane)) <= np.finfo(np.float64).eps:
        raise ValueError("in-plane supercell vectors must be linearly independent")
    reciprocal_xy = 2.0 * np.pi * np.linalg.inv(in_plane).T
    reciprocal = np.zeros((2, 3), dtype=np.float64)
    reciprocal[:, :2] = reciprocal_xy
    return reciprocal


def wavevectors_from_indices(supercell_vectors_angstrom, wavevector_indices):
    indices = np.asarray(wavevector_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 2 or indices.shape[0] < 1:
        raise ValueError("wavevector indices must have shape (number_of_q, 2)")
    reciprocal = reciprocal_supercell_vectors(supercell_vectors_angstrom)
    return indices @ reciprocal


def spin_fourier_modes(
    spin_x, spin_y, spin_z, positions_angstrom, wavevectors_inverse_angstrom,
    batch_size=None,
):
    components = _spin_components(spin_x, spin_y, spin_z)
    number_of_sites = components[0].shape[1]
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    wavevectors = np.asarray(wavevectors_inverse_angstrom, dtype=np.float64)
    if positions.shape != (number_of_sites, 3):
        raise ValueError("positions must have shape (number_of_sites, 3)")
    if wavevectors.ndim != 2 or wavevectors.shape[1] != 3:
        raise ValueError("wavevectors must have shape (number_of_q, 3)")
    if wavevectors.shape[0] < 1:
        raise ValueError("at least one wavevector is required")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(wavevectors)):
        raise ValueError("positions and wavevectors must contain finite values")

    modes = np.empty(
        (components[0].shape[0], wavevectors.shape[0], 3),
        dtype=np.complex128,
    )
    q_batch = _batch_length(batch_size, wavevectors.shape[0], 32 * number_of_sites)
    time_batch = _batch_length(
        None, components[0].shape[0], 16 * (number_of_sites + q_batch)
    )
    normalisation = np.sqrt(number_of_sites)
    for start in range(0, wavevectors.shape[0], q_batch):
        stop = min(start + q_batch, wavevectors.shape[0])
        angles = positions @ wavevectors[start:stop].T
        phases = np.empty(angles.shape, dtype=np.complex128)
        np.multiply(angles, -1j, out=phases)
        np.exp(phases, out=phases)
        phases /= normalisation
        del angles
        for cartesian, component in enumerate(components):
            for time_start in range(0, component.shape[0], time_batch):
                time_stop = min(time_start + time_batch, component.shape[0])
                modes[time_start:time_stop, start:stop, cartesian] = (
                    component[time_start:time_stop] @ phases
                )
    return modes


def mode_autocorrelation(modes, maximum_lag=None, connected=True, batch_size=None):
    values = np.asarray(modes)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("modes must contain at least two time samples")
    _require_finite(values, "modes must contain only finite values")
    number_of_samples = values.shape[0]
    lag_count = _lag_count(number_of_samples, maximum_lag)
    number_of_modes = int(np.prod(values.shape[1:], dtype=np.int64))
    columns = _batch_length(batch_size, number_of_modes, 64 * number_of_samples)
    result = np.empty((lag_count,) + values.shape[1:], dtype=np.complex128)
    result_columns = result.reshape(lag_count, number_of_modes)
    pairs = (number_of_samples - np.arange(lag_count)).reshape(-1, 1)
    for start in range(0, number_of_modes, columns):
        stop = min(start + columns, number_of_modes)
        work = np.array(_column_batch(values, start, stop), dtype=np.complex128)
        if connected:
            work -= work.mean(axis=0, keepdims=True)
        transform = np.fft.fft(work, n=2 * number_of_samples, axis=0)
        del work
        np.multiply(transform, transform.conj(), out=transform)
        correlation = np.fft.ifft(transform, axis=0)
        del transform
        np.divide(correlation[:lag_count], pairs, out=result_columns[:, start:stop])
        del correlation
    return result


def longitudinal_transverse_correlations(
    cartesian_modes, axis, maximum_lag=None, connected=True, batch_size=None,
):
    modes = np.asarray(cartesian_modes)
    if modes.ndim != 3 or modes.shape[2] != 3:
        raise ValueError("cartesian_modes must have shape (time, q, 3)")
    direction = np.asarray(axis, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("axis must be a finite three-component vector")
    norm = np.linalg.norm(direction)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("axis must be non-zero")
    direction /= norm

    if modes.shape[0] < 2:
        raise ValueError("modes must contain at least two time samples")
    _require_finite(modes, "modes must contain only finite values")
    lag_count = _lag_count(modes.shape[0], maximum_lag)
    q_batch = _batch_length(batch_size, modes.shape[1], 96 * modes.shape[0])
    longitudinal = np.empty((lag_count, modes.shape[1]), dtype=np.complex128)
    trace = np.empty_like(longitudinal)
    for start in range(0, modes.shape[1], q_batch):
        stop = min(start + q_batch, modes.shape[1])
        work = modes[:, start:stop].astype(np.complex128, copy=True)
        if connected:
            work -= work.mean(axis=0, keepdims=True)
        longitudinal_modes = np.einsum("tqc,c->tq", work, direction)
        longitudinal[:, start:stop] = mode_autocorrelation(
            longitudinal_modes, maximum_lag, connected=False, batch_size=batch_size
        )
        cartesian = mode_autocorrelation(
            work, maximum_lag, connected=False, batch_size=batch_size
        )
        trace[:, start:stop] = cartesian.sum(axis=2)
        del work, longitudinal_modes, cartesian
    transverse = trace - longitudinal
    return {
        "longitudinal": longitudinal,
        "transverse": transverse,
        "trace": trace,
    }


def spin_mode_correlations(
    spin_x,
    spin_y,
    spin_z,
    positions_angstrom,
    wavevectors_inverse_angstrom,
    axis=None,
    maximum_lag=None,
    connected=True,
    batch_size=None,
):
    if axis is None:
        axis = equilibrium_axis(spin_x, spin_y, spin_z)
    else:
        direction = np.asarray(axis, dtype=np.float64)
        if direction.shape != (3,) or np.linalg.norm(direction) == 0.0:
            raise ValueError("axis must be a non-zero three-component vector")
        axis = direction / np.linalg.norm(direction)
    modes = spin_fourier_modes(
        spin_x,
        spin_y,
        spin_z,
        positions_angstrom,
        wavevectors_inverse_angstrom,
        batch_size=batch_size,
    )
    result = longitudinal_transverse_correlations(
        modes, axis, maximum_lag=maximum_lag, connected=connected,
        batch_size=batch_size,
    )
    result["axis"] = np.asarray(axis)
    return result


def periodic_space_time_correlation(
    field, maximum_lag=None, connected=True, batch_size=None,
):
    values = np.asarray(field)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("field must have time and at least one spatial axis")
    _require_finite(values, "field must contain only finite values")
    number_of_samples = values.shape[0]
    lag_count = _lag_count(number_of_samples, maximum_lag)
    spatial_size = int(np.prod(values.shape[1:], dtype=np.int64))
    spatial_axes = tuple(range(1, values.ndim))
    time_batch = _batch_length(batch_size, number_of_samples, 32 * spatial_size)
    mean = values.mean(dtype=np.complex128) if connected else 0.0
    spatial_modes = np.empty(values.shape, dtype=np.complex128)
    for start in range(0, number_of_samples, time_batch):
        stop = min(start + time_batch, number_of_samples)
        work = np.array(values[start:stop], dtype=np.complex128)
        if connected:
            work -= mean
        spatial_modes[start:stop] = np.fft.fftn(work, axes=spatial_axes)
    del work
    correlation = mode_autocorrelation(
        spatial_modes, maximum_lag=lag_count - 1, connected=False,
        batch_size=batch_size,
    )
    del spatial_modes
    lag_batch = _batch_length(batch_size, lag_count, 32 * spatial_size)
    result_is_real = True
    for start in range(0, lag_count, lag_batch):
        stop = min(start + lag_batch, lag_count)
        work = np.fft.ifftn(correlation[start:stop], axes=spatial_axes)
        work /= spatial_size
        if np.iscomplexobj(np.real_if_close(work)):
            result_is_real = False
        correlation[start:stop] = np.fft.fftshift(work, axes=spatial_axes)
        del work
    return correlation.real if result_is_real else correlation
