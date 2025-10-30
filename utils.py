import hashlib
import json
import os
from pathlib import Path

import cmocean
import matplotlib
import matplotlib.pyplot as plt

_file_dir = Path(__file__).resolve().parent
_src_dir = _file_dir / "cmaps" / "data"
_meta_file = _file_dir / ".cmap_meta.json"
_cmap_file = _file_dir / "plot_cmaps.py"

ipcc_cmap_list = [f.stem for f in _src_dir.glob("*.txt")]
plt_cmap_list = plt.colormaps()
cmocean_cmap_list = list(cmocean.cm.cmapnames)
all_cmaps = ipcc_cmap_list + plt_cmap_list + cmocean_cmap_list


imports = """
from dataclasses import dataclass, field

import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from .cmap_funcs import *
\n
"""

container_class = """
@dataclass
class ColorMap:

"""

for name in all_cmaps:
    if not (name.endswith("_r") or "cmo" in name):
        container_class += f"    {name.lower()} = get_cm('{name}')\n"

colormaps_class = """
@dataclass
class ColorMaps:
    cm: ColorMap = field(default_factory=ColorMap)

    @staticmethod
    def new(colors: list[str], N: int = 25, *, discrete: bool = True):
        return blend(colors, N=N, discrete=discrete)
    \n
        """

for name in all_cmaps:
    if name.endswith("_r") or "cmo" in name.lower():
        continue
    colormaps_class += f"""
    @staticmethod
    def {name.lower()}(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("{name}", N, reverse, split, add_colors, discrete)
        \n
    """

init = """
cmaps: ColorMaps = ColorMaps()
cm : ColorMaps = cmaps
"""


def _compute_hash():
    """Compute a hash from versions and src file metadata (names + modification times)."""
    src_files = sorted(_src_dir.glob("*.txt"))
    src_state = {f.name: os.path.getmtime(f) for f in src_files}

    data = {
        "matplotlib_version": matplotlib.__version__,
        "cmocean_version": cmocean.__version__,
        "src_state": src_state,
    }

    hash_str = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return hash_str, data


def _load_meta():
    if _meta_file.exists():
        try:
            with open(_meta_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _write_meta(data, hash_str):
    data["_hash"] = hash_str
    with open(_meta_file, "w") as f:
        json.dump(data, f, indent=2)


def gen_cmap_file():
    """Generate plot_cmaps.py only if versions or src files changed (added, removed, or modified)."""
    new_hash, meta_data = _compute_hash()
    old_meta = _load_meta()

    if _cmap_file.exists() and old_meta.get("_hash") == new_hash:
        return  # Up to date
    with open(_cmap_file, "w") as f:
        f.write(imports)
        f.write(container_class)
        f.write(colormaps_class)
        f.write(init)

    _write_meta(meta_data, new_hash)
