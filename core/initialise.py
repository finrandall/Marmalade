import numpy as np


INITIAL_STATES = ("fm_x", "fm_y", "fm_z", "neel_z", "random")


def init_spins(number_of_sites, basis_indices, mode="fm_x", seed=None):
    rng = np.random.default_rng(seed)

    if mode == "random":
        spins = rng.normal(size=(number_of_sites, 3)).astype(np.float64)
        spins /= np.linalg.norm(spins, axis=1, keepdims=True)
        return spins

    directions = {
        "fm_x": (1.0, 0.0, 0.0),
        "fm_y": (0.0, 1.0, 0.0),
        "fm_z": (0.0, 0.0, 1.0),
    }
    if mode in directions:
        return np.tile(
            np.asarray(directions[mode], dtype=np.float64),
            (number_of_sites, 1),
        )

    if mode == "neel_z":
        spins = np.zeros((number_of_sites, 3), dtype=np.float64)
        spins[:, 2] = np.where(np.asarray(basis_indices) == 0, 1.0, -1.0)
        return spins

    raise ValueError(f"initial state must be one of {INITIAL_STATES}")
