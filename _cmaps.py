import hashlib
import textwrap
import uuid
from pathlib import Path

import cmocean
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex

_file_dir = Path(__file__).resolve().parent
_src_dir = _file_dir / "data" / "cmaps"

files = _src_dir.glob("*.txt")
ipcc_cmap_list = [f.stem for f in files]
plt_cmap_list = plt.colormaps()
cmocean_cmap_list = list(cmocean.cm.cmapnames)


def build_cm(name: str) -> LinearSegmentedColormap | ListedColormap:
    names = [name, name.lower(), name.capitalize(), name.upper()]
    for c in names:
        cmap_file = _src_dir / f"{c}.txt"
        if cmap_file.exists():
            return LinearSegmentedColormap.from_list(c, np.loadtxt(cmap_file), N=256)
        if c in plt_cmap_list:
            return plt.colormaps[c]
        if c in cmocean_cmap_list:
            return getattr(cmocean.cm, c)
    raise KeyError(f"Colormap '{name}' is not valid.")


def get_colormap(
    name, N, r, split, add_colors, discrete, as_colors, gamma=1.0
) -> ListedColormap | LinearSegmentedColormap | list[str]:
    return adjust_cmap(
        cmap=build_cm(name),
        N=N,
        split=split,
        add_colors=add_colors,
        r=r,
        discrete=discrete,
        as_colors=as_colors,
        gamma=gamma,
    )


def add_colors_to_cmap(
    obj: str | list[str],
    cmap: ListedColormap | LinearSegmentedColormap,
    idx: int = 256,
    N: int = 256,
    gamma: float = 1.0,
    cmap_name: str = None,
) -> ListedColormap | LinearSegmentedColormap:
    """
    Add custom colors into an existing matplotlib colormap or combine multiple colormaps.

    Parameters
    ----------
    obj : str | list[str]
        Hex color(s) or CSS4 color name(s) to insert.
    cmap : ListedColormap | LinearSegmentedColormap
        Existing matplotlib colormap.
    N : int, default=256
        Number of colors to sample from the colormap.
    idx : int, default=256
        Index at which to insert the new colors.
    gamma : float, default=1.0
        Gamma correction factor for the colormap.
    cmap_name : str, optional
        Name of the new colormap to create. If not provided, the original colormap's name is used.



    """

    N = cmap.N

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
    cmap = LinearSegmentedColormap.from_list(cmap_name, new_colors, N, gamma=gamma)

    return cmap


def get_colors(cmap: ListedColormap | LinearSegmentedColormap, N: int) -> list[str]:
    colors = cmap(np.linspace(0, 1, N))
    return [to_hex(c) for c in colors]


def adjust_cmap(
    cmap: str | ListedColormap | LinearSegmentedColormap,
    N: int = 25,
    *,
    split: tuple[float, float] = (0, 1),
    add_colors: dict[int, str | list[str]] = None,
    r: bool = False,
    discrete: bool = False,
    as_colors: bool = False,
    gamma: float = 1.0,
) -> ListedColormap | LinearSegmentedColormap | list[str]:
    """
    Retrieve and modify a matplotlib colormap with optional slicing, color insertion,
    reversal, and discretization.

    Parameters
    ----------
    cmap : str | ListedColormap | LinearSegmentedColormap
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

    r : bool, default=False
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
    >>> cmap = adjust("plasma", N=256, split=(0.2, 0.8), r=True)
    >>> cmap = adjust("viridis", add_colors={128: ["#ffffff", "red"]}, discrete=False)
    """

    if not isinstance(split, tuple) or len(split) != 2:
        raise ValueError("`split` must be a tuple of two floats (start, end).")

    cmap_name = cmap.name
    if r:
        cmap = cmap.reversed()
        cmap_name = f"reversed_{cmap_name}"

    range_values = np.linspace(split[0], split[1], N)
    colors = [cmap(value) for value in range_values]

    if discrete:
        res = ListedColormap(colors, N=N, name=cmap_name)
    else:
        res = LinearSegmentedColormap.from_list(cmap_name, colors, N=N, gamma=gamma)

    if add_colors:
        cmap_name = f"{uuid.uuid4().hex[:6]}"
        if not isinstance(add_colors, dict):
            raise TypeError(
                "`add_colors` must be a dictionary of dict[int, str | list[str]]."
            )

        for k in sorted(add_colors):
            v = add_colors[k]
            if k == -1:
                k = res.N

            if not isinstance(v, (list, tuple, str)):
                raise TypeError(
                    "Values in `add_colors` must be a string or list of strings."
                )
            if not isinstance(k, int):
                raise TypeError(
                    "Keys in `add_colors` must be integers representing position to insert colors."
                )

            res = add_colors_to_cmap(
                obj=v,
                idx=k,
                cmap=res,
                N=N,
                gamma=gamma,
                cmap_name=cmap_name,
            )

    if as_colors:
        return get_colors(res, res.N)
    return res


def concat(
    cmap1, cmap2, N: int = 32, *, discrete: bool = False, gamma: float = 1.0
) -> ListedColormap | LinearSegmentedColormap:
    def _colors(cmap):
        range_values = np.linspace(0, 1, N // 2)
        colors = [to_hex(cmap(value)) for value in range_values]
        return colors

    colors1 = _colors(cmap1)
    colors2 = _colors(cmap2)
    return create(colors1 + colors2, N=N, discrete=discrete, gamma=gamma)


def add_or_subtract(
    cmap1,
    cmap2,
    operator: str,
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
) -> ListedColormap | LinearSegmentedColormap:
    def _colors(cmap):
        range_values = np.linspace(0, 1, N)
        colors = [cmap(value) for value in range_values]
        return np.asarray(colors, dtype=float)

    c1 = _colors(cmap1)
    c2 = _colors(cmap2)

    if operator == "+":
        c = np.clip(c1 + c2, 0.0, 1.0)
    elif operator == "-":
        c = np.clip(c1 - c2, 0.0, 1.0)

    c = [to_hex(tuple(row)) for row in c]

    return create(c, N=N, discrete=discrete, gamma=gamma)


def create(
    colors: list[str],
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
    name: str = None,
    save: bool = False,
) -> ListedColormap | LinearSegmentedColormap:
    def valid(c: str) -> bool:
        return isinstance(c, str) and (c.startswith("#") or c in mcolors.CSS4_COLORS)

    if not all(map(valid, colors)):
        raise ValueError("All colors must be valid hex codes or CSS4 color names.")

    if save and not name:
        raise ValueError("A name must be provided when save = True.")

    if name is None:
        name = uuid.uuid4().hex[:6]

    if discrete:
        cmap = ListedColormap(colors, N=N, name=name)
    else:
        cmap = LinearSegmentedColormap.from_list(
            name,
            colors,
            N=N,
            gamma=gamma,
        )

    if save:
        rgb = cmap(np.linspace(0.0, 1.0, 256))[:, :3]
        np.savetxt(
            f"{_src_dir}/{name}.txt",
            rgb,
            fmt="%.6f",
        )

    return cmap


def gen_cmap_file():
    _cmap_file = _file_dir / "cmaps.py"

    ipcc_cmap_list = [f.stem for f in _src_dir.glob("*.txt")]
    plt_cmap_list = plt.colormaps()
    cmocean_cmap_list = list(cmocean.cm.cmapnames)
    all_cmaps = ipcc_cmap_list + plt_cmap_list + cmocean_cmap_list

    src_files = sorted(_src_dir.glob("*.txt"))

    src_state = []
    for s in src_files:
        with open(s, "r") as f:
            src_state.append(f.read())

    src_state = "".join(src_state)

    src_checksum = hashlib.sha256(src_state.encode("utf-8")).hexdigest().upper()

    checksum = None

    try:
        from ._cmaps import checksum

        if checksum == src_checksum:
            return
    except Exception:
        checksum = src_checksum

    def _cmap_file_contents():
        imports = f"""
        '''
        Fancy custom colormap utilities for creating, modifying, and combining matplotlib colormaps.
        '''
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap

        from ._cmaps import *

        checksum = "{checksum}"
        \n
        """
        imports = textwrap.dedent(imports)

        operators_signatures = """
cmap1: ListedColormap | LinearSegmentedColormap,
cmap2: ListedColormap | LinearSegmentedColormap,
N: int = 32,
*,
discrete: bool = False,
gamma: float = 1.0
"""

        body = f"""
       

def new(colors: list[str], N: int = 32, *, discrete: bool = False, gamma: float = 1.0, name: str = None, save: bool = False):
    ''' Create a new colormap from a list of colors '''
    return create(colors, N=N, discrete=discrete, gamma=gamma, name=name, save=save)


def concat({operators_signatures}):
    ''' Concat two colormaps together '''
    return concat(cmap1, cmap2, N=N, discrete=discrete, gamma=gamma)


def add({operators_signatures}):
    ''' Add two colormaps together '''
    return add_or_subtract(cmap1, cmap2, operator="+", N=N, discrete=discrete, gamma=gamma)



def subtract({operators_signatures}):
    ''' Subtract two colormaps '''
    return add_or_subtract(cmap1, cmap2, operator="-", N=N, discrete=discrete, gamma=gamma)
            \n
                """

        processed_cmaps = []
        for name in all_cmaps:
            if name.endswith("_r") or "cmo" in name.lower():
                continue

            if name.lower() in processed_cmaps:
                continue

            processed_cmaps.append(name.lower())

            body = f""" {body}

def {name.lower()}(
    N: int = 32,
    r: bool = False,
    split: tuple[float, float] = (0, 1),
    add_colors: dict[int, str | list[str]] = None,
    discrete: bool = False,
    as_colors : bool = False,
    gamma: float = 1.0
) -> ListedColormap | LinearSegmentedColormap:
    ''' Get the '{name}' colormap '''
    
    return get_colormap("{name}", N, r, split, add_colors, discrete,as_colors, gamma)
    \n
"""
        body = textwrap.dedent(body)

        return imports, body

    def _generate():
        """Generate plot_cmaps.py only if versions or src files changed (added, removed, or modified)."""

        if _cmap_file.exists():
            _cmap_file.unlink()
        (imports, body) = _cmap_file_contents()
        with open(_cmap_file, "w") as f:
            f.write(imports)
            f.write(body)

    _generate()


gen_cmap_file()
