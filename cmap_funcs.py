import hashlib
import json
import os
import textwrap
from pathlib import Path

import cmocean
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex

_file_dir = Path(__file__).resolve().parent
_src_dir = _file_dir / "data" / "cmaps"

files = _src_dir.glob("*.txt")
ipcc_cmap_list = [f.stem for f in files]
plt_cmap_list = plt.colormaps()
cmocean_cmap_list = list(cmocean.cm.cmapnames)


def build_cm(name):
    names = [name, name.lower(), name.capitalize(), name.upper()]
    for c in names:
        ipcc_file = _src_dir / f"{c}.txt"
        if ipcc_file.exists():
            return LinearSegmentedColormap.from_list(c, np.loadtxt(ipcc_file), N=256)
        if c in plt_cmap_list:
            return plt.colormaps[c]
        if c in cmocean_cmap_list:
            return getattr(cmocean.cm, c)
    raise KeyError(f"Colormap '{name}' is not valid.")


def get_colormap(name, N, reverse, split, add_colors, discrete, as_colors):
    return adjust_cmap(
        cmap=build_cm(name),
        N=N,
        split=split,
        add_colors=add_colors,
        reverse=reverse,
        discrete=discrete,
        as_colors=as_colors,
    )


def add_cmap_colors(
    obj,
    cmap,
    N=256,
    idx=256,
    reverse: bool = True,
):
    """
    Add custom colors into an existing matplotlib colormap or combine multiple colormaps.

    Parameters
    ----------
    obj : str | list[str]
        Hex color(s) or CSS4 color name(s) to insert.
    cmap : Colormap
        Existing matplotlib colormap.
    N : int, default=256
        Number of colors to sample from the colormap.
    idx : int, default=256
        Index at which to insert the new colors.
    reverse : bool, default=True
        Whether to reverse the colormap before adding colors.



    """
    if reverse:
        cmap = cmap.reversed()

    # Clamp index within [0, N]
    idx = max(0, min(idx, N))

    # Ensure `objs` is a list of strings
    if isinstance(obj, str):
        objs = [obj]
    elif isinstance(obj, (list, tuple)):
        objs = list(obj)
    else:
        raise TypeError(
            "Invalid colors specified. Please provide a list of valid color names or hex values."
        )

    colors_to_add = []
    for color in objs:
        if isinstance(color, str):
            if color.startswith("#"):
                colors_to_add.append(color)
            elif mcolors.CSS4_COLORS.get(color) is not None:
                colors_to_add.append(to_hex(mcolors.CSS4_COLORS[color]))
            else:
                raise ValueError(
                    f"Invalid color '{color}'. Must be a hex code or a named CSS4 color."
                )
        else:
            raise TypeError("Color must be a string (hex or named CSS4 color).")

    # Extract colors from cmap
    colors = [to_hex(tuple(c), keep_alpha=True) for c in cmap(np.linspace(0, 1, N))]

    # Insert at position `where`
    new_colors = colors[:idx] + colors_to_add + colors[idx:]
    new_colors = np.array(new_colors)
    N += len(colors_to_add)
    cmap = LinearSegmentedColormap.from_list(cmap.name, new_colors, N=N)

    return cmap


def get_colors(cmap, N):
    colors = cmap(np.linspace(0, 1, N))
    return [to_hex(c) for c in colors]


def adjust_cmap(
    cmap=None,
    N: int = 25,
    *,
    split: tuple[float, float] = (0, 1),
    add_colors: dict[int, str | list[str]] = None,
    reverse: bool = False,
    discrete: bool = True,
    as_colors: bool = False,
):
    """
    Retrieve and modify a matplotlib colormap with optional slicing, color insertion,
    reversal, and discretization.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap, optional
        Name or colormap object to adjust.

    N : int, default=25
        Number of colors to sample or bins to discretize when ``discrete=True``.
        Has no effect when ``discrete=False``.

    split : tuple of float, default=(0.0, 1.0)
        Fractional range of the colormap to retain, specified as ``(start, end)`` within [0.0, 1.0].
        Useful for truncating or isolating a subrange of a continuous colormap.

    add_colors : dict of {int: str or list[str]}, optional
        Mapping of integer indices to one or more colors to insert into the colormap.
        Keys represent insertion positions (0 ≤ index ≤ N), and values can be either
        a single color (hex string or CSS4 name) or a list of colors.
        For example: ``{0: "black", 128: ["white", "#ff0000"]}``.

    reverse : bool, default=False
        If ``True``, reverse the colormap before applying other adjustments.

    discrete : bool, default=True
        If ``True``, return a discrete ``ListedColormap`` with ``N`` bins.
        If ``False``, return a continuous ``LinearSegmentedColormap``.
    as_colors : bool, default=False
        If ``True``, return a list of colors instead of a cmap



    Returns
    -------
    matplotlib.colors.Colormap or list of colors
        The adjusted colormap after applying all transformations.

    Notes
    -----
    - This function provides fine control over colormap structure for visual encoding.
    - The ``split`` parameter is applied before color insertion.
    - Indices in ``add_colors`` are automatically clamped to the valid range [0, N].
    - The function is typically used as a utility method within a plotting toolkit or visualization class.

    Examples
    --------
    >>> cmap = adjust("plasma", N=256, split=(0.2, 0.8), reverse=True)
    >>> cmap = adjust("viridis", add_colors={128: ["#ffffff", "red"]}, discrete=False)
    """

    if not isinstance(split, tuple) or len(split) != 2:
        raise ValueError("`split` must be a tuple of two floats (start, end).")

    range_values = np.linspace(split[0], split[1], N)
    colors = [cmap(value) for value in range_values]

    if discrete:
        res = ListedColormap(colors, N=N, name=cmap.name)
    else:
        res = LinearSegmentedColormap.from_list(cmap.name, colors, N=N)

    if add_colors:

        if not isinstance(add_colors, dict):
            raise TypeError(
                f"`add_colors` must be a dictionary of dict[int, str | list[str]]."
            )

        for k, v in add_colors.items():
            if not isinstance(v, (list, tuple, str)):
                raise TypeError(
                    "Values in `add_colors` must be a string or list of strings."
                )
            if not isinstance(k, int):
                raise TypeError(
                    "Keys in `add_colors` must be integers representing position to insert colors."
                )

            res = add_cmap_colors(
                obj=v,
                idx=k,
                cmap=res,
                N=N,
                reverse=False,
            )
    if reverse:
        res = res.reversed()

    if as_colors:
        return get_colors(res, N)
    return res


def blend(colors: list[str], N: int = 25, *, discrete: bool = True):
    valid = lambda c: isinstance(c, str) and (
        c.startswith("#") or c in mcolors.CSS4_COLORS
    )
    if not all(map(valid, colors)):
        raise ValueError("All colors must be valid hex codes or CSS4 color names.")

    range_values = np.linspace(0, 1, N)
    cmap = sns.blend_palette(colors, as_cmap=True, input="hex", n_colors=N)
    color_list = [cmap(value) for value in range_values]

    if discrete:
        return ListedColormap(color_list, N=N, name=f"blend_{len(colors)}")
    return LinearSegmentedColormap.from_list(f"blend_{len(colors)}", color_list, N=N)


def gen_cmap_file():
    _meta_file = _file_dir / ".cmap_meta.json"
    _cmap_file = _file_dir / "cmaps_inventory.py"

    ipcc_cmap_list = [f.stem for f in _src_dir.glob("*.txt")]
    plt_cmap_list = plt.colormaps()
    cmocean_cmap_list = list(cmocean.cm.cmapnames)
    all_cmaps = ipcc_cmap_list + plt_cmap_list + cmocean_cmap_list

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

    def _cmap_file_contents():

        imports = """
        from dataclasses import dataclass
        from matplotlib.colors import Colormap

        from .cmap_funcs import *
        \n
        """
        imports = textwrap.dedent(imports)

        body = """
        @dataclass
        class ColorMaps:

            @staticmethod
            def new(colors: list[str], N: int = 25, *, discrete: bool = True):
                return blend(colors, N=N, discrete=discrete)
            \n
                """

        for name in all_cmaps:
            if name.endswith("_r") or "cmo" in name.lower():
                continue
            body = f""" {body}
            
            @staticmethod
            def {name.lower()}(
                N: int = 25,
                reverse: bool = False,
                split: tuple[float, float] = (0, 1),
                add_colors: dict[int, str | list[str]] = None,
                discrete: bool = True,
                as_colors : bool =False
            ) -> Colormap:
                return get_colormap("{name}", N, reverse, split, add_colors, discrete,as_colors)
                \n
            """
        body = textwrap.dedent(body)

        init = """
        cmaps: ColorMaps = ColorMaps()
        cm : ColorMaps = cmaps
        """
        init = textwrap.dedent(init)
        return imports, body, init

    def _generate():
        """Generate plot_cmaps.py only if versions or src files changed (added, removed, or modified)."""
        new_hash, meta_data = _compute_hash()
        old_meta = _load_meta()

        if _cmap_file.exists() and old_meta.get("_hash") == new_hash:
            return  # Up to date
        imports, body, init = _cmap_file_contents()
        with open(_cmap_file, "w") as f:
            f.write(imports)
            f.write(body)
            f.write(init)

        _write_meta(meta_data, new_hash)

    _generate()
