import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from .storage import data_path, prepare_output_directory


def json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    return value


def run_name(params):
    noise = {"quantum": "quantum_aux"}.get(params.get("noise_mode"), params.get("noise_mode"))
    start = {"fm_x": "xstart", "fm_y": "ystart", "fm_z": "zstart"}.get(params.get("initial_state"), params.get("initial_state"))
    try:
        temperature = float(params["T"])
        label = f"{int(temperature):02d}" if temperature.is_integer() else f"{temperature:g}"
        return (f"{noise}_L{int(params['Lx'])}_lam{float(params['lam']):g}_{start}_T{label}K_"
                f"dt{float(params['dt']) * 1e15:g}fs_t{float(params['end_time']) * 1e9:g}ns_")
    except (KeyError, TypeError, ValueError):
        return ""


def create_output_directory(parent, prefix=""):
    parent = prepare_output_directory(parent)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = prefix + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = parent / timestamp
    suffix = 1
    while output.exists():
        output = parent / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output.mkdir()
    return output


def write_npz(stream, arrays, compressed=True, compression_level=1, *, allow_pickle=True):
    if (isinstance(compression_level, (bool, np.bool_))
            or not isinstance(compression_level, (int, np.integer))
            or not 1 <= compression_level <= 9):
        raise ValueError("compression_level must be an integer from 1 to 9")
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    level = int(compression_level) if compressed else None
    with zipfile.ZipFile(stream, "w", compression=compression,
                         compresslevel=level, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=allow_pickle)


def save_result(params, settings, data):
    settings = dict(settings)
    settings.setdefault("compressed", True)
    settings.setdefault("compression_level", 1)
    output = create_output_directory(settings["output_directory"], run_name(params))
    archive = dict(data)
    for name, value in params.items():
        if value is not None:
            archive[f"parameter_{name}"] = np.asarray(value)
    archive["outputs"] = np.asarray(settings["outputs"])
    temporary = output / ".simulation.npz.tmp"
    with temporary.open("wb") as stream:
        write_npz(stream, archive, settings.get("compressed", True),
                  settings.get("compression_level", 1))
    temporary.replace(output / "simulation.npz")

    payload = {
        "parameters": json_compatible(params),
        "settings": json_compatible(settings),
        "data_file": "simulation.npz",
    }
    with open(output / "parameters.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    return output


def load_checkpoint(path):
    from pathlib import Path
    path = data_path(path)
    if path.is_dir():
        path = path / "simulation.npz"
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name] for name in archive.files if name.startswith("checkpoint_")}
    if "checkpoint_spins" not in state:
        raise KeyError(f"{path} has no checkpoint (written before checkpoints were added)")
    state["source"] = str(path)
    return state
