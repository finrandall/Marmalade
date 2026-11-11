import argparse

import numpy as np

if __package__ in (None, ""):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "core"

from .storage import data_path
from .simulation import OUTPUT_CHOICES
from .integrators import INTEGRATORS
from .fft_thermostat import FFT_PARAMETER_DEFAULTS, add_fft_arguments, fft_parameters


def default_parameters():
    params = {
        "lattice": np.array(["cubic"]),
        "dipolar_enabled": np.array([False]),
        "noise_mode": np.array(["classical"]),
        "integrator": np.array(["heun"]),
        "initial_state": np.array(["fm"]),
        "T": np.array([30.0]),
        "Lx": np.array([32]),
        "Ly": np.array([32]),
        "Lz": np.array([32]),
        "pbc_x": np.array([True]),
        "pbc_y": np.array([True]),
        "pbc_z": np.array([True]),
        "dt": np.array([1e-15]),
        "end_time": np.array([1e-11]),
        "burn_in_time": np.array([0.0]),
        "stride": np.array([1]),
        "J": np.array([-1e-2]),
        "D": np.array([0.0]),
        "h": np.array([0.0]),
        "lam": np.array([0.01]),
        **{k: np.asarray([v]) for k, v in FFT_PARAMETER_DEFAULTS.items()},
    }
    settings = {
        "outputs": ("magnetisation_timeseries", "energy_timeseries"),
        "output_directory": str(data_path("results")),
        "threads_per_rank": None,
        "fft_workers": None,
        "blas_threads": 1,
        "compressed": True,
        "compression_level": 1,
    }
    return params, settings


def _boolean(value):
    value = value.lower()
    if value in ("true", "yes", "1", "on"):
        return True
    if value in ("false", "no", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def get_parameters(args=None):
    params, settings = default_parameters()
    parser = argparse.ArgumentParser(description="Run the nearest-neighbour 3D cubic Heisenberg model. J, anisotropy and field energies are in eV; temperature is in K. Unit spins with +J and +D: negative J is ferromagnetic; positive D is a hard axis.")
    parser.add_argument("--noise-mode", nargs="+", choices=("classical", "quantum", "quantum_fft", "none"), default=params["noise_mode"])
    parser.add_argument("--integrator", nargs="+", choices=INTEGRATORS, default=params["integrator"])
    parser.add_argument("--initial-state", nargs="+", choices=("fm", "afm", "random"), default=params["initial_state"])
    parser.add_argument("--temperature", type=float, nargs="+", default=params["T"])
    parser.add_argument("--lattice-x", type=int, nargs="+", default=params["Lx"])
    parser.add_argument("--lattice-y", type=int, nargs="+", default=params["Ly"])
    parser.add_argument("--lattice-z", type=int, nargs="+", default=params["Lz"])
    parser.add_argument("--pbc-x", type=_boolean, nargs="+", default=params["pbc_x"])
    parser.add_argument("--pbc-y", type=_boolean, nargs="+", default=params["pbc_y"])
    parser.add_argument("--pbc-z", type=_boolean, nargs="+", default=params["pbc_z"])
    parser.add_argument("--time-step", type=float, nargs="+", default=params["dt"])
    parser.add_argument("--end-time", type=float, nargs="+", default=params["end_time"])
    parser.add_argument("--burn-in-time", type=float, nargs="+", default=params["burn_in_time"])
    parser.add_argument("--stride", type=int, nargs="+", default=params["stride"])
    parser.add_argument("--J", type=float, nargs="+", default=params["J"])
    parser.add_argument("--D", "--anisotropy", dest="anisotropy", type=float, nargs="+", default=params["D"])
    parser.add_argument("--field", type=float, nargs="+", default=params["h"])
    parser.add_argument("--damping", type=float, nargs="+", default=params["lam"])
    parser.add_argument("--outputs", nargs="+", choices=OUTPUT_CHOICES, default=settings["outputs"])
    parser.add_argument("--output-directory", default=settings["output_directory"])
    parser.add_argument("--threads-per-rank", type=int, default=settings["threads_per_rank"])

    parser.add_argument("--fft-workers", type=int, default=settings["fft_workers"])
    parser.add_argument("--blas-threads", type=int, default=settings["blas_threads"])
    parser.add_argument("--compressed", type=_boolean, default=settings["compressed"])
    parser.add_argument("--compression-level", type=int, choices=range(1, 10),
                        default=settings["compression_level"],
                        help="lossless deflate level for NPZ output (1 fastest, 9 smallest)")
    parser.add_argument("--random-seed", type=int, nargs="+")

    add_fft_arguments(parser, sweep=True)
    options = parser.parse_args(args)
    params.update(fft_parameters(options, sweep=True))
    option_names = {
        "noise_mode": "noise_mode", "integrator": "integrator", "initial_state": "initial_state",
        "T": "temperature", "Lx": "lattice_x", "Ly": "lattice_y", "Lz": "lattice_z",
        "pbc_x": "pbc_x", "pbc_y": "pbc_y", "pbc_z": "pbc_z", "dt": "time_step",
        "end_time": "end_time", "burn_in_time": "burn_in_time", "stride": "stride",
        "J": "J", "D": "anisotropy", "h": "field", "lam": "damping",
    }
    for parameter_name, option_name in option_names.items():
        params[parameter_name] = np.asarray(getattr(options, option_name))
    if options.random_seed is not None:
        params["random_seed"] = np.asarray(options.random_seed)
    settings["outputs"] = tuple(dict.fromkeys(options.outputs))
    settings["output_directory"] = str(data_path(options.output_directory))
    settings["threads_per_rank"] = options.threads_per_rank
    settings["fft_workers"] = options.fft_workers
    settings["blas_threads"] = options.blas_threads
    settings["compressed"] = options.compressed
    settings["compression_level"] = options.compression_level
    return params, settings


if __name__ == "__main__":
    from .main import main
    main(*get_parameters())
