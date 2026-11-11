import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.storage import data_root

EXCLUDE = ("final_spins", "spin_snapshots", "fourier_modes", "maps_b2", "correlations", "tracks", "checkpoint",
           "equilibrated", "legacy_reference", "projections", "excitation", "result", "reference", "spec_", "merons")
QUANTITIES = {
    "magnitude": ("magnetisation_magnitude", None),
    "plane": ("magnetisation_plane", "mplane"),
    "x": ("magnetisation_x", "mx"),
    "y": ("magnetisation_y", "my"),
    "z": ("magnetisation_z", "mz"),
    "energy": ("energy", "E"),
    "dipolar_energy": ("dipolar_energy", "Edip"),
    "damping": ("damping_lambda", None),
}
MILLIELECTRON_VOLT = 1.602176634e-22
PARAMETER_SETS = {-2.1375: "chen2022", -1.7775: "lu2020", -0.79: "lu2020"}


@dataclass
class Run:
    path: Path
    temperature: float
    time_ns: np.ndarray
    values: dict
    info: dict


def resolve(name):
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    root = data_root()
    if root is not None and (root / path).exists():
        return root / path
    return (Path.cwd() / path).resolve()


def files(name):
    path = resolve(name)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.npz") if "_archive" not in p.parts and not any(x in p.name for x in EXCLUDE))


def _index(folder):
    for directory in (folder, folder.parent):
        index = directory / "index.csv"
        if index.is_file():
            with index.open() as handle:
                return {row["file"]: row for row in csv.DictReader(handle)}
    return {}


def _json(folder):
    for directory in (folder, folder.parent):
        path = directory / "parameters.json"
        if path.is_file():
            data = json.loads(path.read_text())
            return data.get("parameters", data) if isinstance(data, dict) else {}
    return {}


def _scalar(data, *keys):
    for key in keys:
        if key in data:
            value = data[key]
            value = value.item() if hasattr(value, "item") else value
            try:
                return float(value) if not isinstance(value, str) else value
            except (TypeError, ValueError):
                return value
    return None


def _describe(scalars):
    j1 = scalars.get("J1_meV")
    parameter_set = scalars.get("parameter_set") or PARAMETER_SETS.get(round(j1, 4) if j1 is not None else None, "unknown")
    noise = scalars.get("noise_mode") or scalars.get("thermostat") or "unknown"
    noise = {"quantum": "quantum_aux"}.get(noise, noise)
    start = {"fm_x": "xstart", "fm_y": "ystart", "fm_z": "zstart"}.get(scalars.get("initial_state"), scalars.get("initial_state"))
    size = scalars.get("Lx")
    return dict(parameter_set=str(parameter_set).split("_")[0], noise=noise, size=int(size) if size else None,
                damping=scalars.get("lam"), start=start, seed=scalars.get("seed"), dt_fs=scalars.get("dt_fs"))


def read(path, quantities=("magnitude",)):
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    names = set(data.files)
    if not ({"time", "t"} & names and {"magnetisation_x", "mx"} & names):
        return None
    if "time" in names:
        time = data["time"] * 1e9
        moment = _scalar(data, "magnetic_moment_muB", "moment_muB", "parameter_moment_muB") or 1.0
        spins = _scalar(data, "number_of_spins", "spatial_number_of_spins", "N")
        temperature = _scalar(data, "temperature_kelvin", "temperature", "temperature_K", "parameter_T")
        if temperature is None:
            match = re.search(r"T(\d+(?:\.\d+)?)K", str(path))
            temperature = float(match.group(1)) if match else float("nan")
        scalars = {key: _scalar(data, key, "parameter_" + key) for key in ("J1_meV", "lam", "initial_state", "Lx")}
        scalars.update(parameter_set=_scalar(data, "parameter_set", "parameter_parameter_set"),
                       noise_mode=_scalar(data, "parameter_noise_mode"), thermostat=_scalar(data, "thermostat"),
                       seed=_scalar(data, "random_seed", "parameter_random_seed", "seed"))
        dt = _scalar(data, "dt_fs")
        step = _scalar(data, "parameter_dt", "time_step")
        scalars["dt_fs"] = dt if dt is not None else (step * 1e15 if step is not None else None)
        scalars["lam"] = scalars["lam"] if scalars["lam"] is not None else _scalar(data, "damping")
        scalars["Lx"] = scalars["Lx"] if scalars["Lx"] is not None else _scalar(data, "lattice_x")
        if scalars["Lx"] is None and "spatial_lattice_shape" in names:
            scalars["Lx"] = float(data["spatial_lattice_shape"][0])
        row_dt = row.get("time_step") if (row := _index(path.parent).get(path.name, {})) else None
        if scalars["dt_fs"] is None and row_dt:
            scalars["dt_fs"] = float(row_dt) * 1e15
        row = _index(path.parent).get(path.name, {})
        for key, column in (("noise_mode", "thermostat"), ("initial_state", "initial_state")):
            scalars[key] = scalars[key] or row.get(column)
    else:
        time = data["t"] * 1e9
        number = re.search(r"simulation_\d+", path.name)
        row = _index(path.parent).get(number.group(0) + ".npz", {}) if number else {}
        extra = _json(path.parent)
        moment = float(row.get("moment_muB") or extra.get("moment_muB") or 3.0)
        spins = float(data["N"]) if "N" in names else float(row.get("number_of_spins") or 0) or None
        match = re.search(r"T([\d.]+)K", path.name)
        temperature = float(row.get("temperature") or (match.group(1) if match else "nan"))
        seed = re.search(r"seed(\d+)", path.name)
        scalars = dict(J1_meV=float(row.get("J1_meV") or extra.get("J1_meV") or 0) or None,
                       lam=float(row.get("damping") or extra.get("lam") or 0) or None,
                       initial_state=row.get("initial_state") or extra.get("initial_state"),
                       Lx=row.get("lattice_x") or extra.get("Lx"), parameter_set=extra.get("parameter_set"),
                       noise_mode=row.get("thermostat") or extra.get("noise_mode"),
                       seed=int(seed.group(1)) if seed else None,
                       dt_fs=float(row["time_step"]) * 1e15 if row.get("time_step") else None)
    values = {}
    for quantity in quantities:
        long, short = QUANTITIES[quantity]
        if long in names:
            array = data[long]
        elif short in names:
            array = data[short]
        elif quantity in ("magnitude", "plane"):
            parts = [next(data[k] for k in pair if k in names) for pair in (("magnetisation_x", "mx"), ("magnetisation_y", "my"), ("magnetisation_z", "mz"))]
            array = np.sqrt(sum(c ** 2 for c in (parts if quantity == "magnitude" else parts[:2])))
        else:
            raise KeyError(f"{quantity} not stored in {path}")
        if quantity in ("energy", "dipolar_energy"):
            values[quantity] = array / (spins * MILLIELECTRON_VOLT) if spins else array
        elif quantity == "damping":
            values[quantity] = array
        else:
            values[quantity] = array / moment
    info = _describe(scalars)
    info.update(file=str(path), duration_ns=float(time[-1]) if time.size else 0.0,
                kind="nve" if "_nve_" in path.name else "thermostat", spins=spins, moment=moment)
    if info["kind"] == "nve":
        info["damping"] = 0.0
    return Run(path, float(temperature), time, values, info)


def runs(name, quantities=("magnitude",), temperatures=None):
    seen = set()
    for path in files(name):
        run = read(path, quantities)
        if run is None:
            continue
        if temperatures is not None and not any(abs(run.temperature - t) < 1e-6 for t in temperatures):
            continue
        key = (run.temperature, run.info["seed"], run.info["dt_fs"], run.info["damping"], run.info["kind"], round(run.info["duration_ns"], 6))
        if run.info["seed"] is not None and key in seen:
            continue
        seen.add(key)
        yield run


def block_average(time_ns, values, start_ns=0.0, end_ns=None, block_ns=1.0):
    mask = (time_ns >= start_ns) & (time_ns <= (time_ns[-1] if end_ns is None else end_ns))
    selected, window = values[mask], time_ns[mask]
    if selected.size < 2:
        return np.nan, np.nan
    blocks = max(1, int((window[-1] - window[0]) // block_ns))
    means = np.array([block.mean() for block in np.array_split(selected, blocks)])
    return selected.mean(), means.std(ddof=1) / np.sqrt(blocks) if blocks > 1 else np.nan
