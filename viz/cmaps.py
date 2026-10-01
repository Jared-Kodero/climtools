"""
Colormap toolkit for climtools.

Each registered colormap is exposed as a module level callable, for example::

    from xgeo import cmaps
    cmaps.low_high(span=(0, 0.5)).reversed()

Callables are produced on demand by ``__getattr__`` (PEP 562), so the user
facing access is unchanged while no Python source is generated at import time.
Editor autocomplete and static type checking are served by the companion stub
``cmaps.pyi``, which contains only typed signatures and is therefore never
executed. Regenerate the stub after the set of colormaps changes with::

    python -m xgeo.viz.cmaps

or by calling :func:`write_stub`.

Colormap names are drawn from three backends, in this precedence:
    1. Local IPCC style colormaps stored as plain text RGB tables.
    2. Built in matplotlib colormaps.
    3. cmocean colormaps.
"""

from __future__ import annotations

import hashlib
import os
import sys
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cmocean
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex
from skimage.color import rgb2lab

if TYPE_CHECKING:
    from IPython.display import DisplayHandle

type ColorMap = ListedColormap | LinearSegmentedColormap

# fmt: off
_FILE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _FILE_DIR / "data" / "cmaps"

# Backend colormap names resolved once at import. ``build_cm`` consults these.
_PLT_CMAPS = mpl.colormaps  # public matplotlib ColormapRegistry
_PLT_CMAP_LIST = list(_PLT_CMAPS)
_CMOCEAN_CMAP_LIST = list(cmocean.cm.cmapnames)
_PUBLIC = {
    "new",
    "concat",
    "available",
    "slice_cmap",
    "get_cmap",
    "get_colors",
    "classify_cmap",
    "write_stub",
}

_EQ_ATOL = 1e-6  # tolerance consistent with the %.6f text colormap format

_PREDEFINED_SEQUENTIAL = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu",
    "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "gray", "bone", "pink", "spring", "summer", "autumn", "winter",
    "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper",
]

_PREDEFINED_DIVERGING = [
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu",
    "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic",
    "berlin", "managua", "vanimo",
]

_PREDEFINED_QUALITATIVE = [
    "Pastel1", "Pastel2", "Paired", "Accent", "okabe_ito",
    "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20",
    "tab20b", "tab20c",
]

_PREDEFINED_CYCLIC = [
    "twilight", "twilight_shifted", "hsv",
]

_PREDEFINED_MISCELLANEOUS = [
    "flag", "prism", "ocean", "gist_earth", "terrain",
    "gist_stern", "gnuplot", "gnuplot2", "CMRmap",
    "cubehelix", "brg", "gist_rainbow", "rainbow", "jet",
    "turbo", "nipy_spectral", "gist_ncar",
]
# fmt: on


_CMAP_CLASSIFICATIONS = {
    **{name: "sequential" for name in _PREDEFINED_SEQUENTIAL},
    **{name: "diverging" for name in _PREDEFINED_DIVERGING},
    **{name: "qualitative" for name in _PREDEFINED_QUALITATIVE},
    **{name: "cyclic" for name in _PREDEFINED_CYCLIC},
    **{name: "miscellaneous" for name in _PREDEFINED_MISCELLANEOUS},
}


_CMAP_CLASSIFICATIONS = {
    **{name: "sequential" for name in _PREDEFINED_SEQUENTIAL},
    **{name: "diverging" for name in _PREDEFINED_DIVERGING},
    **{name: "qualitative" for name in _PREDEFINED_QUALITATIVE},
    **{name: "cyclic" for name in _PREDEFINED_CYCLIC},
    **{name: "miscellaneous" for name in _PREDEFINED_MISCELLANEOUS},
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _registry() -> dict[str, str]:
    """
    Map each public function name to its canonical backend name.

    The public name preserves the actual colormap name if it is a valid Python
    identifier, otherwise it falls back to lowercase or a valid form.
    """
    text_names = [f.stem for f in _SRC_DIR.glob("*.txt")]
    all_names = text_names + _PLT_CMAP_LIST + _CMOCEAN_CMAP_LIST
    mapping: dict[str, str] = {}
    for name in all_names:
        if name.endswith("_r") or "cmo" in name.lower():
            continue

        # Use the actual name if it's a valid identifier, otherwise try lower()
        key = name if name.isidentifier() else name.lower().replace("-", "_")
        if not key.isidentifier():
            continue

        mapping.setdefault(key, name)
    return dict(sorted(mapping.items()))


# ---------------------------------------------------------------------------
# Dynamic per-colormap callables
# ---------------------------------------------------------------------------
@cache
def create(public_name: str, source_name: str):
    def cmap(
        N: int | None = None,
        *,
        span: tuple[float, float] = (0.0, 1.0),
        add_colors: dict[int, str | list[str]] | None = None,
        format: Literal["linear", "listed", "hex"] = "linear",
        gamma: float = 1.0,
    ):
        return get_cmap(source_name, N, span, add_colors, format, gamma)

    cmap.__name__ = public_name
    cmap.__qualname__ = public_name
    cmap.__doc__ = f"Return the '{source_name}' colormap."
    return cmap


def __getattr__(name: str):
    registry = _registry()
    if name in registry:
        return create(name, registry[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get(name: str) -> ColorMap:
    """Return the colormap corresponding to the given name."""
    if not name.endswith("_r"):
        return load_cmap(name)
    return load_cmap(name.replace("_r", "")).reversed()


def __dir__():
    return sorted(_PUBLIC | cmap_index())


def classify_cmap(cmap: str | ColorMap) -> str:
    """Return the colormap class.

    Predefined colormaps use their known classification. Unknown colormaps
    are classified perceptually from their CIELAB lightness profile.
    """
    if isinstance(cmap, str):
        name = cmap.removesuffix("_r")
        if name in _CMAP_CLASSIFICATIONS:
            return _CMAP_CLASSIFICATIONS[name]
        cmap = load_cmap(cmap)
    else:
        name = getattr(cmap, "name", "").removesuffix("_r")
        if name in _CMAP_CLASSIFICATIONS:
            return _CMAP_CLASSIFICATIONS[name]

    rgb = cmap(np.linspace(0.0, 1.0, 256))[:, :3]
    lab = rgb2lab(rgb.reshape(1, -1, 3))[0]
    lightness = lab[:, 0]

    diffs = np.diff(lightness)
    if np.all(diffs >= -0.5) or np.all(diffs <= 0.5):
        return "sequential"

    mid = len(lightness) // 2
    if lightness[mid] > max(lightness[0], lightness[-1]) or lightness[mid] < min(
        lightness[0], lightness[-1]
    ):
        return "diverging"

    # A cyclic map should return perceptually close to its starting color.
    endpoint_distance = np.linalg.norm(lab[0] - lab[-1])
    if endpoint_distance < 10.0:
        return "cyclic"

    return "unclassified"


# ---------------------------------------------------------------------------
# Colormap construction and modification
# ---------------------------------------------------------------------------


def get_colors(cmap: ColorMap, N: int | None = None) -> list[str]:
    """Sample ``N`` evenly spaced colors from ``cmap`` and return them as hex strings."""
    n_colors = cmap.N if N is None else N
    return [to_hex(c) for c in cmap(np.linspace(0, 1, n_colors))]


def array_rep(cmap: ColorMap) -> np.ndarray:
    """Return the 256 point RGB sampling used for colormap equality tests."""
    return cmap(np.linspace(0.0, 1.0, 256))[:, :3]


def load_cmap(name: str) -> ColorMap:
    """Resolve a colormap by name across the text, matplotlib and cmocean backends."""
    for candidate in (name, name.lower(), name.capitalize(), name.upper()):
        cmap_file = _SRC_DIR / f"{candidate}.txt"
        if cmap_file.exists():
            data = np.loadtxt(cmap_file)
            return LinearSegmentedColormap.from_list(candidate, data, N=data.shape[0])

        if candidate in _PLT_CMAP_LIST:
            return _PLT_CMAPS[candidate]
        if candidate in _CMOCEAN_CMAP_LIST:
            return getattr(cmocean.cm, candidate)
    raise KeyError(f"Colormap '{name}' is not valid.")


def slice_cmap(
    cmap: str | ColorMap,
    span: tuple[float, float] = (0.0, 1.0),
    N: int | None = None,
    *,
    format: Literal["linear", "listed", "hex"] | None = None,
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """
    Extract a subset of colors from a colormap based on a span range and return in the specified format.
    """
    if not isinstance(span, tuple) or len(span) != 2:
        raise ValueError("`span` must be a tuple of two floats (start, end).")

    if isinstance(cmap, str):
        cmap = load_cmap(cmap)

    n_colors = cmap.N if N is None else N
    cmap_name = cmap.name
    colors = [cmap(value) for value in np.linspace(span[0], span[1], n_colors)]

    if format is None:
        format = "listed" if isinstance(cmap, ListedColormap) else "linear"

    if format == "hex":
        res = ListedColormap(colors, name=cmap_name)
        return get_colors(res, res.N)
    elif format == "listed":
        return ListedColormap(colors, name=cmap_name)
    else:
        return LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_colors, gamma=gamma
        )


def add_colors(
    obj: str | list[str],
    cmap: ColorMap,
    idx: int | None = None,
    N: int | None = None,
    gamma: float = 1.0,
    cmap_name: str | None = None,
    format: Literal["linear", "listed", "hex"] = "linear",
) -> ColorMap | list[str]:

    if format not in {"linear", "listed", "hex"}:
        raise ValueError("`format` must be 'linear', 'listed', or 'hex'.")

    n_colors = cmap.N if N is None else N
    idx = n_colors if idx is None else max(0, min(idx, n_colors))

    if isinstance(obj, str):
        objs = [obj]
    elif isinstance(obj, (list, tuple)):
        objs = list(obj)
    else:
        raise TypeError(
            "Invalid colors specified. Provide a list of CSS4 names or hex values."
        )

    colors_to_add = []
    for color in objs:
        if not isinstance(color, str):
            raise TypeError("Color must be a string (hex or named CSS4 color).")
        if color.startswith("#"):
            colors_to_add.append(color)
        elif mcolors.CSS4_COLORS.get(color) is not None:
            colors_to_add.append(to_hex(mcolors.CSS4_COLORS[color]))
        else:
            raise ValueError(
                f"Invalid color '{color}'. Must be a hex code or a named CSS4 color."
            )

    colors = [
        to_hex(tuple(c), keep_alpha=True) for c in cmap(np.linspace(0, 1, n_colors))
    ]
    new_colors = colors[:idx] + colors_to_add + colors[idx:]

    if format in {"listed", "hex"}:
        res = ListedColormap(new_colors, N=len(new_colors), name=cmap_name)
    else:
        res = LinearSegmentedColormap.from_list(
            cmap_name, new_colors, N=n_colors, gamma=gamma
        )

    if format == "hex":
        return get_colors(res, res.N)
    return res


def get_cmap(
    name: str,
    N: int | None,
    span: tuple[float, float],
    add_colors: dict[int, str | list[str]] | None,
    format: Literal["linear", "listed", "hex"],
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """Resolve ``name`` to a colormap and apply the requested adjustments."""
    cmap = load_cmap(name)
    n_colors = cmap.N if N is None else N

    """
    Modify a colormap by slicing, color insertion and output format.
    """
    if format not in {"linear", "listed", "hex"}:
        raise ValueError("`format` must be 'linear', 'listed', or 'hex'.")

    cmap_name = cmap.name if not isinstance(cmap, str) else cmap
    n_colors = cmap.N if N is None else N
    res = slice_cmap(cmap, span=span, N=n_colors, format=format, gamma=gamma)

    if add_colors:
        if not isinstance(add_colors, dict):
            raise TypeError("`add_colors` must be a dict[int, str | list[str]].")

        if isinstance(res, list):
            res = ListedColormap(res, name=cmap_name)

        cmap_name = "added"
        internal_format: Literal["linear", "listed", "hex"] = (
            "listed" if format == "hex" else format
        )
        for k in sorted(add_colors):
            v = add_colors[k]
            if k == -1:
                k = res.N
            if not isinstance(k, int):
                raise TypeError("Keys in `add_colors` must be integers.")
            if not isinstance(v, (list, tuple, str)):
                raise TypeError("Values in `add_colors` must be str or list[str].")
            adjusted = add_colors(
                obj=v,
                idx=k,
                cmap=res,
                N=n_colors,
                gamma=gamma,
                cmap_name=cmap_name,
                format=internal_format,
            )
            if isinstance(adjusted, list):
                raise TypeError("Internal colormap conversion returned a color list.")
            res = adjusted

    if format == "hex" and not isinstance(res, list):
        return get_colors(res, res.N)
    return res


def new(
    colors: list[str],
    N: int | None = None,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
    name: str | None = None,
    save: bool = False,
) -> ColorMap | list[str]:
    """Create a colormap from a list of hex codes or CSS4 names."""

    def valid(c: str) -> bool:
        return isinstance(c, str) and (c.startswith("#") or c in mcolors.CSS4_COLORS)

    if not all(map(valid, colors)):
        raise ValueError("All colors must be valid hex codes or CSS4 names.")
    if save and not name:
        raise ValueError("A name must be provided when save=True.")
    if format not in {"linear", "listed", "hex"}:
        raise ValueError("`format` must be 'linear', 'listed', or 'hex'.")
    if name is None:
        name = "custom"

    n_colors = len(colors) if N is None else N

    if format in {"listed", "hex"}:
        cmap: ColorMap = ListedColormap(colors, name=name)
    else:
        cmap = LinearSegmentedColormap.from_list(name, colors, N=n_colors, gamma=gamma)

    if save:
        dup = find_duplicate(cmap)
        if dup is not None:
            match_name, is_reversed = dup
            kind = "reversed " if is_reversed else ""
            print(
                f"Creation skipped: identical to existing {kind} colormap {match_name!r}'."
            )
            existing = load_cmap(match_name)
            cmap = existing.reversed() if is_reversed else existing
        else:
            rgb = cmap(range(cmap.N))[:, :3]
            _SRC_DIR.mkdir(parents=True, exist_ok=True)
            np.savetxt(Path(_SRC_DIR / name).with_suffix(".txt"), rgb, fmt="%.6f")
            _registry.cache_clear()
            cmap_index.cache_clear()
            write_stub(force=True)

    if format == "hex":
        return get_colors(cmap, cmap.N)
    return cmap


def concat(
    cmap1: ColorMap | str,
    cmap2: ColorMap | str,
    N: int | None = None,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """Concatenate two colormaps."""

    if isinstance(cmap1, str):
        cmap1 = load_cmap(cmap1)
    if isinstance(cmap2, str):
        cmap2 = load_cmap(cmap2)

    return new(
        [to_hex(cmap1(v)) for v in np.linspace(0, 1, cmap1.N)]
        + [to_hex(cmap2(v)) for v in np.linspace(0, 1, cmap2.N)],
        N=N or 256,
        format=format,
        gamma=gamma,
    )


@lru_cache(maxsize=1)
def cmap_index() -> frozenset[str]:
    """Return the set of public colormap names for fast membership tests."""
    return frozenset(_registry())


def find_duplicate(cmap: ColorMap) -> tuple[str, bool] | None:
    """
    Return ``(name, reversed)`` if ``cmap`` matches a registered colormap.

    Matching is evaluated on the 256 point RGB sampling. ``reversed`` is True
    when the match is against the reversed form. Returns ``None`` otherwise.
    """
    target = array_rep(cmap)
    for public_name, source_name in _registry().items():
        existing = array_rep(load_cmap(source_name))
        if np.allclose(target, existing, atol=_EQ_ATOL):
            return public_name, False
        if np.allclose(target, existing[::-1], atol=_EQ_ATOL):
            return public_name, True
    return None


def available(show: bool = True) -> list[str] | DisplayHandle:
    """List or show available colormaps"""
    if "ipykernel" in sys.modules and show:
        for source_name in _registry().values():
            display(get_cmap(source_name))

        return

    return list(_registry().keys())


# ---------------------------------------------------------------------------
# Type stub generation (cmaps.pyi)
# ---------------------------------------------------------------------------
_CMAP_SIGNATURE = (
    "(N: int | None = None, *, "
    "span: tuple[float, float] = ..., "
    "add_colors: dict[int, str | list[str]] | None = None, "
    'format: Literal["linear", "listed", "hex"] = "linear", '
    "gamma: float = 1.0) -> ListedColormap | LinearSegmentedColormap | list[str]: ..."
)

_STUB_HEADER = """from typing import Literal
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from IPython.display import DisplayHandle
type ColorMap = ListedColormap | LinearSegmentedColormap

def get(name: str) -> ColorMap: ...
def new(colors: list[str], N: int | None = None, *, format: Literal["linear", "listed", "hex"] = "linear", gamma: float = 1.0, name: str | None = None, save: bool = False) -> ColorMap | list[str]: ...
def concat(cmap1: ColorMap, cmap2: ColorMap, N: int | None = None, *, format: Literal["linear", "listed", "hex"] = "linear", gamma: float = 1.0) -> ColorMap | list[str]: ...
def available(show: bool = True) -> list[str] | DisplayHandle: ...
"""


def build_stub_text() -> str:
    """Return the full text of the ``cmaps.pyi`` type stub."""
    lines = [_STUB_HEADER]
    for name in list(_registry().keys()):
        lines.append(f"def {name}{_CMAP_SIGNATURE}")
    return "\n".join(lines) + "\n"


def _src_checksum() -> str:
    """Checksum the text colormaps, the resolved name set, and this source file."""
    h = hashlib.sha256()
    for f in sorted(_SRC_DIR.glob("*.txt")):
        h.update(f.read_bytes())
    h.update(",".join(list(_registry().keys())).encode("utf-8"))
    try:
        h.update(Path(__file__).read_bytes())
    except Exception:
        pass
    return h.hexdigest()


def write_stub(force: bool = False) -> bool:
    """
    Write ``cmaps.pyi`` if the colormap set changed or ``force`` is set.

    Returns ``True`` when the file was written. The checksum is stored on the
    first line so a stale stub is detected without an external sidecar. The file
    is replaced atomically, which removes the need for lock files.
    """
    pyi = _FILE_DIR / "cmaps.pyi"
    marker = f"# checksum: {_src_checksum()}\n"
    if not force and pyi.exists():
        try:
            if pyi.read_text().startswith(marker):
                return False
        except OSError:
            pass
    TMP = pyi.with_suffix(".pyi.TMP")
    TMP.write_text(marker + build_stub_text())
    os.replace(TMP, pyi)
    return True


try:  # pragma: no cover
    write_stub()
except OSError:  # pragma: no cover
    pass
