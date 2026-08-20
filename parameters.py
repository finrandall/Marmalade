import argparse

import numpy as np


OUTPUT_CHOICES = ("magnetisation_mean", "magnetisation_timeseries", "energy_timeseries", "spin_trajectory")


def default_parameters():
    """Return sweepable physical parameters and non-sweep run settings."""
    params = {
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
        "J": np.array([1e-2]),
        "K": np.array([0.0]),
        "h": np.array([0.0]),
        "lam": np.array([0.01]),
    }
    settings = {
        "outputs": ("magnetisation_timeseries", "energy_timeseries"),
        "output_directory": "results",
        "threads_per_rank": None,
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
    parser = argparse.ArgumentParser(description="Run ASD atomistic spin dynamics simulations.")
    parser.add_argument("--noise-mode", nargs="+", choices=("classical", "quantum", "none"), default=params["noise_mode"])
    parser.add_argument("--integrator", nargs="+", choices=("heun",), default=params["integrator"])
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
    parser.add_argument("--anisotropy", type=float, nargs="+", default=params["K"])
    parser.add_argument("--field", type=float, nargs="+", default=params["h"])
    parser.add_argument("--damping", type=float, nargs="+", default=params["lam"])
    parser.add_argument("--outputs", nargs="+", choices=OUTPUT_CHOICES, default=settings["outputs"])
    parser.add_argument("--output-directory", default=settings["output_directory"])
    parser.add_argument("--threads-per-rank", type=int, default=settings["threads_per_rank"])

    options = parser.parse_args(args)
    option_names = {
        "noise_mode": "noise_mode", "integrator": "integrator", "initial_state": "initial_state",
        "T": "temperature", "Lx": "lattice_x", "Ly": "lattice_y", "Lz": "lattice_z",
        "pbc_x": "pbc_x", "pbc_y": "pbc_y", "pbc_z": "pbc_z", "dt": "time_step",
        "end_time": "end_time", "burn_in_time": "burn_in_time", "stride": "stride",
        "J": "J", "K": "anisotropy", "h": "field", "lam": "damping",
    }
    for parameter_name, option_name in option_names.items():
        params[parameter_name] = np.asarray(getattr(options, option_name))
    settings["outputs"] = tuple(dict.fromkeys(options.outputs))
    settings["output_directory"] = options.output_directory
    settings["threads_per_rank"] = options.threads_per_rank
    return params, settings


if __name__ == "__main__":
    from main import main
    main(*get_parameters())
