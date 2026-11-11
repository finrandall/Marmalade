from pathlib import Path
import argparse
import os
import shutil
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
MAC_DATA_ROOT = Path.home() / "SimulationData" / "ASD"
MAC_BACKUP_ROOT = Path("/Volumes/b0069/Users/ed20f2r/ASD")


def _absolute(path):
    return Path(os.path.abspath(Path(path).expanduser()))


def _in_onedrive(path):
    return any(part.lower().startswith("onedrive") for part in path.parts)


def data_root():
    configured = os.environ.get("ASD_DATA_ROOT")
    if configured is not None:
        root = Path(configured).expanduser()
        if not configured or not root.is_absolute():
            raise ValueError("ASD_DATA_ROOT must be an absolute path outside OneDrive")
    elif sys.platform == "darwin":
        root = MAC_DATA_ROOT
    else:
        return None
    root = _absolute(root)
    if _in_onedrive(root.resolve()) or root.resolve().is_relative_to(REPOSITORY):
        raise ValueError("ASD_DATA_ROOT must be outside OneDrive and the code checkout")
    return root


def data_path(value, base=None):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path if base is None else Path(base) / path
    path = _absolute(path)
    root = data_root()
    if root is not None and path.is_relative_to(REPOSITORY):
        path = root / path.relative_to(REPOSITORY)
    if root is not None and path.is_relative_to(root / "downloads"):
        path = root / path.relative_to(root / "downloads")
    return path


def backup_root():
    root = Path(os.environ.get("ASD_BACKUP_ROOT", str(MAC_BACKUP_ROOT))).expanduser()
    if not root.is_absolute() or _in_onedrive(root):
        raise ValueError("ASD_BACKUP_ROOT must be an absolute path outside OneDrive")
    root = _absolute(root)
    local = data_root()
    if local is not None and (root.is_relative_to(local)
                              or local.is_relative_to(root)):
        raise ValueError("Working data and backup directories must be separate")
    return root


IMAGES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".eps", ".tif", ".tiff", ".webp", ".gif"}
TOPICS = (
    ("tests", "01_engine_validation", False),
    ("crcl3/runs/standard", "02_crcl3_single_runs", False),
    ("crcl3/runs/verification", "02_crcl3_single_runs", False),
    ("crcl3/runs/sweep", "03_crcl3_temperature_sweeps", False),
    ("crcl3/runs/calder_fft_sweep", "04_crcl3_L512_magnons_merons", False),
    ("crcl3_L512", "04_crcl3_L512_magnons_merons", True),
    ("crcl3/runs/local_lowdamping_dt", "05_timestep_damping", False),
    ("crcl3/parameter_sets/lu2020_merons", "06_lu2020_comparison", False),
    ("lu2020", "06_lu2020_comparison", True),
    ("_archive/hydrodynamics_pilot", "07_hydrodynamics", False),
    ("crcl3/runs/benchmarks", "08_benchmarks", False),
)


def figures_root():
    configured = os.environ.get("ASD_PROCESSED_ROOT")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("ASD_PROCESSED_ROOT must be an absolute path")
        return _absolute(root)
    if sys.platform == "darwin" or data_root() is not None:
        return REPOSITORY.parent / "ASD Figures"
    return None


def topic_path(relative):
    text = relative.as_posix()
    for prefix, topic, keep in TOPICS:
        if keep and text.startswith(prefix):
            return Path(topic) / relative
        if text == prefix or text.startswith(prefix + "/"):
            return Path(topic) / text[len(prefix):].lstrip("/")
    return Path("00_unsorted") / relative


def processed_path(value):
    path = _absolute(value)
    root = figures_root()
    if root is None or path.is_relative_to(root):
        return path
    if path.suffix.lower() not in IMAGES:
        return data_path(path)
    for origin in (data_root(), backup_root(), REPOSITORY):
        if origin is not None and path.is_relative_to(origin):
            return root / topic_path(path.relative_to(origin))
    return path


def prepare_processed_directory(value):
    path = data_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_file(value):
    path = processed_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def publish_processed_file(value):
    source = _absolute(value)
    destination = processed_path(source)
    if destination == source or source.suffix.lower() not in IMAGES:
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def check_storage(value):
    path = _absolute(value)
    resolved = path.resolve()
    if (sys.platform == "darwin" or data_root() is not None) and _in_onedrive(resolved):
        raise ValueError(f"Simulation data must be outside OneDrive: {path}")
    volume_path = resolved if len(resolved.parts) >= 3 and resolved.parts[1] == "Volumes" else path
    if len(volume_path.parts) >= 3 and volume_path.parts[1] == "Volumes":
        mount = Path(*volume_path.parts[:3])
        if not mount.is_mount():
            raise FileNotFoundError(
                f"Data volume is not mounted: {mount}. Connect to the Leeds VPN, "
                "then in Finder press Cmd+K and connect to "
                "smb://svmt003.leeds.ac.uk/b0069 as ds\\ed20f2r. "
                "If Finder uses another mount name, set ASD_DATA_ROOT to its "
                "absolute data folder. No local fallback will be used."
            )
    return path


def prepare_output_directory(value, base=None):
    path = check_storage(data_path(value, base))
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--prepare", action="store_true",
                        help="also check the mount and create a writable folder")
    options = parser.parse_args()
    try:
        path = (prepare_output_directory(options.path) if options.prepare
                else data_path(options.path))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(path)


if __name__ == "__main__":
    main()
