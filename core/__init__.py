import os
import sys
from pathlib import Path

_cache = Path.home() / ("Library/Caches/ASD" if sys.platform == "darwin" else ".cache/ASD")
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(_cache / "pycache")
