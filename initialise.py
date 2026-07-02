import numpy as np


def init_spins(Lx, Ly, Lz=1, mode="afm", seed=None):
    N = Lx * Ly * Lz
    rng = np.random.default_rng(seed)

    if mode == "random":
        S = rng.normal(size=(N, 3)).astype(np.float64)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        return S

    if mode == "fm":
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (N, 1))

    if mode == "afm":
        S = np.empty((N, 3), dtype=np.float64)

        for iz in range(Lz):
            for iy in range(Ly):
                for ix in range(Lx):
                    i = ix + Lx * (iy + Ly * iz)
                    S[i, 0] = 0.0
                    S[i, 1] = 0.0

                    if (ix + iy + iz) % 2 == 0:
                        S[i, 2] = 1.0
                    else:
                        S[i, 2] = -1.0
        return S

    raise ValueError("mode must be 'random', 'fm' or 'afm'")
