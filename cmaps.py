"""
Colormap toolkit for climtools.

Each registered colormap is exposed as a module level callable, for example::

    from climtools import cmaps
    cmaps.low_high(r=True)

Callables are produced on demand by ``__getattr__`` (PEP 562), so the user
facing access is unchanged while no Python source is generated at import time.
Editor autocomplete and static type checking are served by the companion stub
``cmaps.pyi``, which contains only typed signatures and is therefore never
executed. Regenerate the stub after the set of colormaps changes with::

    python -m climtools.cmaps

or by calling :func:`write_stub`.

Colormap names are drawn from three backends, in this precedence:
    1. Local IPCC style colormaps stored as plain text RGB tables.
    2. Built in matplotlib colormaps.
    3. cmocean colormaps.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import cmocean
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex

_file_dir = Path(__file__).resolve().parent
_src_dir = _file_dir / "data" / "cmaps"

# Backend colormap names resolved once at import. ``build_cm`` consults these.
_plt_registry = mpl.colormaps  # public matplotlib ColormapRegistry
_plt_cmap_list = list(_plt_registry)
_cmocean_cmap_list = list(cmocean.cm.cmapnames)

_Cmap = ListedColormap | LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Colormap construction and modification
# ---------------------------------------------------------------------------
def build_cm(name: str) -> _Cmap:
    """Resolve a colormap by name across the text, matplotlib and cmocean backends."""
    for candidate in (name, name.lower(), name.capitalize(), name.upper()):
        cmap_file = _src_dir / f"{candidate}.txt"
        if cmap_file.exists():
            return LinearSegmentedColormap.from_list(
                candidate, np.loadtxt(cmap_file), N=256
            )
        if candidate in _plt_cmap_list:
            return _plt_registry[candidate]
        if candidate in _cmocean_cmap_list:
            return getattr(cmocean.cm, candidate)
    raise KeyError(f"Colormap '{name}' is not valid.")


def get_colors(cmap: _Cmap, N: int) -> list[str]:
    """Sample ``N`` evenly spaced colors from ``cmap`` and return them as hex strings."""
    return [to_hex(c) for c in cmap(np.linspace(0, 1, N))]


def add_colors_to_cmap(
    obj: str | list[str],
    cmap: _Cmap,
    idx: int = 256,
    N: int = 256,
    gamma: float = 1.0,
    cmap_name: str | None = None,
) -> LinearSegmentedColormap:
    """
    Insert one or more colors into an existing colormap at a given index.

    Parameters
    ----------
    obj : str or list of str
        Hex codes or CSS4 color names to insert.
    cmap : ListedColormap or LinearSegmentedColormap
        Source colormap.
    idx : int, default 256
        Insertion position, clamped to ``[0, cmap.N]``.
    N : int, default 256
        Retained for signature compatibility. The number of sampled colors is
        taken from ``cmap.N``.
    gamma : float, default 1.0
        Gamma applied when rebuilding the colormap.
    cmap_name : str, optional
        Name for the returned colormap.
    """
    N = cmap.N
    idx = max(0, min(idx, N))

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

    colors = [to_hex(tuple(c), keep_alpha=True) for c in cmap(np.linspace(0, 1, N))]
    new_colors = np.array(colors[:idx] + colors_to_add + colors[idx:])
    return LinearSegmentedColormap.from_list(cmap_name, new_colors, N, gamma=gamma)


def adjust_cmap(
    cmap: str | _Cmap,
    N: int = 25,
    *,
    split: tuple[float, float] = (0.0, 1.0),
    add_colors: dict[int, str | list[str]] | None = None,
    r: bool = False,
    discrete: bool = False,
    as_colors: bool = False,
    gamma: float = 1.0,
) -> _Cmap | list[str]:
    """
    Modify a colormap by slicing, reversal, color insertion and discretization.

    Parameters
    ----------
    cmap : str or Colormap
        Colormap to adjust.
    N : int, default 25
        Number of sampled colors, or bins when ``discrete=True``.
    split : tuple of float, default (0.0, 1.0)
        Fractional sub range of the colormap to retain. Applied before insertion.
    add_colors : dict of {int: str or list of str}, optional
        Insertion positions mapped to one or more colors. A key of ``-1`` maps to
        the end of the colormap.
    r : bool, default False
        Reverse the colormap before other operations.
    discrete : bool, default False
        Return a ``ListedColormap`` with ``N`` bins instead of a continuous map.
    as_colors : bool, default False
        Return a list of hex colors instead of a colormap.
    gamma : float, default 1.0
        Gamma applied when building a continuous colormap.
    """
    if not isinstance(split, tuple) or len(split) != 2:
        raise ValueError("`split` must be a tuple of two floats (start, end).")

    cmap_name = cmap.name
    if r:
        cmap = cmap.reversed()
        cmap_name = f"reversed_{cmap_name}"

    colors = [cmap(value) for value in np.linspace(split[0], split[1], N)]

    if discrete:
        res = ListedColormap(colors, N=N, name=cmap_name)
    else:
        res = LinearSegmentedColormap.from_list(cmap_name, colors, N=N, gamma=gamma)

    if add_colors:
        if not isinstance(add_colors, dict):
            raise TypeError("`add_colors` must be a dict[int, str | list[str]].")
        cmap_name = "added"
        for k in sorted(add_colors):
            v = add_colors[k]
            if k == -1:
                k = res.N
            if not isinstance(k, int):
                raise TypeError("Keys in `add_colors` must be integers.")
            if not isinstance(v, (list, tuple, str)):
                raise TypeError("Values in `add_colors` must be str or list[str].")
            res = add_colors_to_cmap(
                obj=v, idx=k, cmap=res, N=N, gamma=gamma, cmap_name=cmap_name
            )

    if as_colors:
        return get_colors(res, res.N)
    return res


def get_colormap(
    name: str,
    N: int,
    r: bool,
    split: tuple[float, float],
    add_colors: dict[int, str | list[str]] | None,
    discrete: bool,
    as_colors: bool,
    gamma: float = 1.0,
) -> _Cmap | list[str]:
    """Resolve ``name`` to a colormap and apply the requested adjustments."""
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


def create(
    colors: list[str],
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
    name: str | None = None,
    save: bool = False,
) -> _Cmap:
    """Create a colormap from a list of hex codes or CSS4 names."""

    def valid(c: str) -> bool:
        return isinstance(c, str) and (c.startswith("#") or c in mcolors.CSS4_COLORS)

    if not all(map(valid, colors)):
        raise ValueError("All colors must be valid hex codes or CSS4 names.")
    if save and not name:
        raise ValueError("A name must be provided when save=True.")
    if name is None:
        name = "custom"

    if discrete:
        cmap = ListedColormap(colors, N=N, name=name)
    else:
        cmap = LinearSegmentedColormap.from_list(name, colors, N=N, gamma=gamma)

    if save:
        rgb = cmap(np.linspace(0.0, 1.0, 256))[:, :3]
        _src_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(_src_dir / f"{name}.txt", rgb, fmt="%.6f")
        _registry.cache_clear()
        list_cmaps.cache_clear()
        cmap_index.cache_clear()

    return cmap


def _concat_cmaps(
    cmap1: _Cmap,
    cmap2: _Cmap,
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
) -> _Cmap:
    """Concatenate two colormaps into a single colormap of ``N`` colors."""

    def _colors(cmap):
        return [to_hex(cmap(v)) for v in np.linspace(0, 1, N // 2)]

    return create(_colors(cmap1) + _colors(cmap2), N=N, discrete=discrete, gamma=gamma)


def add_or_subtract(
    cmap1: _Cmap,
    cmap2: _Cmap,
    operator: str,
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
) -> _Cmap:
    """Add or subtract two colormaps channel wise in RGBA space."""

    def _colors(cmap):
        return np.asarray([cmap(v) for v in np.linspace(0, 1, N)], dtype=float)

    c1, c2 = _colors(cmap1), _colors(cmap2)
    if operator == "+":
        c = np.clip(c1 + c2, 0.0, 1.0)
    elif operator == "-":
        c = np.clip(c1 - c2, 0.0, 1.0)
    else:
        raise ValueError("`operator` must be '+' or '-'.")

    return create(
        [to_hex(tuple(row)) for row in c], N=N, discrete=discrete, gamma=gamma
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _registry() -> dict[str, str]:
    """
    Map each public function name to its canonical backend name.

    The public name is the lowercased backend name and must be a valid Python
    identifier so it can be exposed as an attribute and written into the stub.
    The first occurrence of a name wins, matching the precedence text, then
    matplotlib, then cmocean. Reversed (``_r``) and namespaced cmocean entries
    are excluded.
    """
    text_names = [f.stem for f in _src_dir.glob("*.txt")]
    all_names = text_names + _plt_cmap_list + _cmocean_cmap_list

    mapping: dict[str, str] = {}
    for name in all_names:
        if name.endswith("_r") or "cmo" in name.lower():
            continue
        key = name.lower()
        if not key.isidentifier():
            continue
        mapping.setdefault(key, name)
    return dict(sorted(mapping.items()))


@lru_cache(maxsize=1)
def list_cmaps() -> tuple[str, ...]:
    """Return the sorted tuple of public colormap names."""
    return tuple(_registry().keys())


@lru_cache(maxsize=1)
def cmap_index() -> frozenset[str]:
    """Return the set of public colormap names for fast membership tests."""
    return frozenset(_registry())


def available() -> list[str]:
    """Return the sorted list of registered colormap names."""
    return list(list_cmaps())


# ---------------------------------------------------------------------------
# Public combination helpers (documented, importable, no recursion)
# ---------------------------------------------------------------------------
def new(
    colors: list[str],
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
    name: str | None = None,
    save: bool = False,
) -> _Cmap:
    """Create a colormap from a list of colors."""
    return create(colors, N=N, discrete=discrete, gamma=gamma, name=name, save=save)


def concat(
    cmap1: _Cmap,
    cmap2: _Cmap,
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
) -> _Cmap:
    """Concatenate two colormaps."""
    return _concat_cmaps(cmap1, cmap2, N=N, discrete=discrete, gamma=gamma)


def add(
    cmap1: _Cmap,
    cmap2: _Cmap,
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
) -> _Cmap:
    """Add two colormaps channel wise."""
    return add_or_subtract(cmap1, cmap2, "+", N=N, discrete=discrete, gamma=gamma)


def subtract(
    cmap1: _Cmap,
    cmap2: _Cmap,
    N: int = 32,
    *,
    discrete: bool = False,
    gamma: float = 1.0,
) -> _Cmap:
    """Subtract two colormaps channel wise."""
    return add_or_subtract(cmap1, cmap2, "-", N=N, discrete=discrete, gamma=gamma)


# ---------------------------------------------------------------------------
# Dynamic per-colormap callables
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _make(public_name: str, source_name: str):
    def cmap(
        N: int = 32,
        r: bool = False,
        *,
        split: tuple[float, float] = (0.0, 1.0),
        add_colors: dict[int, str | list[str]] | None = None,
        discrete: bool = False,
        as_colors: bool = False,
        gamma: float = 1.0,
    ):
        return get_colormap(
            source_name, N, r, split, add_colors, discrete, as_colors, gamma
        )

    cmap.__name__ = public_name
    cmap.__qualname__ = public_name
    cmap.__doc__ = f"Return the '{source_name}' colormap."
    return cmap


_PUBLIC = {
    "new",
    "concat",
    "add",
    "subtract",
    "create",
    "available",
    "write_stub",
    "build_cm",
    "get_colormap",
    "adjust_cmap",
    "get_colors",
    "add_colors_to_cmap",
    "add_or_subtract",
    "list_cmaps",
}


def _default_cmap(source_name: str):
    """Return the colormap for ``source_name`` built with default options."""
    return get_colormap(source_name, 32, False, (0.0, 1.0), None, False, False, 1.0)


def __getattr__(name: str):
    registry = _registry()
    if name in registry:
        return _make(name, registry[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_PUBLIC | cmap_index())


# Subscript access. Attribute access (cmaps.low_high) returns the factory to
# call; subscript access (cmaps["low_high"]) returns the colormap directly,
# matching matplotlib.colormaps["viridis"]. Modules do not support __getitem__
# through a module level function, so the module object is promoted to a
# ModuleType subclass that defines it. The existing __getattr__ is unaffected.
import sys as _sys
from types import ModuleType as _ModuleType


class _CmapsModule(_ModuleType):
    def __getitem__(self, name: str):
        registry = _registry()
        if name not in registry:
            raise KeyError(name)
        return _default_cmap(registry[name])

    def __contains__(self, name: str) -> bool:
        return name in cmap_index()


_sys.modules[__name__].__class__ = _CmapsModule


# ---------------------------------------------------------------------------
# Type stub generation (cmaps.pyi)
# ---------------------------------------------------------------------------
_CMAP_SIGNATURE = (
    "(N: int = 32, r: bool = False, *, "
    "split: tuple[float, float] = ..., "
    "add_colors: dict[int, str | list[str]] | None = None, "
    "discrete: bool = False, as_colors: bool = False, "
    "gamma: float = 1.0) -> ListedColormap | LinearSegmentedColormap | list[str]: ..."
)

_STUB_HEADER = """from matplotlib.colors import LinearSegmentedColormap, ListedColormap

_Cmap = ListedColormap | LinearSegmentedColormap

def new(colors: list[str], N: int = 32, *, discrete: bool = False, gamma: float = 1.0, name: str | None = None, save: bool = False) -> _Cmap: ...
def create(colors: list[str], N: int = 32, *, discrete: bool = False, gamma: float = 1.0, name: str | None = None, save: bool = False) -> _Cmap: ...
def concat(cmap1: _Cmap, cmap2: _Cmap, N: int = 32, *, discrete: bool = False, gamma: float = 1.0) -> _Cmap: ...
def add(cmap1: _Cmap, cmap2: _Cmap, N: int = 32, *, discrete: bool = False, gamma: float = 1.0) -> _Cmap: ...
def subtract(cmap1: _Cmap, cmap2: _Cmap, N: int = 32, *, discrete: bool = False, gamma: float = 1.0) -> _Cmap: ...
def available() -> list[str]: ...
def write_stub(force: bool = False) -> bool: ...
"""


def build_stub_text() -> str:
    """Return the full text of the ``cmaps.pyi`` type stub."""
    lines = [_STUB_HEADER]
    for name in list_cmaps():
        lines.append(f"def {name}{_CMAP_SIGNATURE}")
    return "\n".join(lines) + "\n"


def _src_checksum() -> str:
    """Checksum the text colormaps and the resolved name set."""
    h = hashlib.sha256()
    for f in sorted(_src_dir.glob("*.txt")):
        h.update(f.read_bytes())
    h.update(",".join(list_cmaps()).encode("utf-8"))
    return h.hexdigest()


def write_stub(force: bool = False) -> bool:
    """
    Write ``cmaps.pyi`` if the colormap set changed or ``force`` is set.

    Returns ``True`` when the file was written. The checksum is stored on the
    first line so a stale stub is detected without an external sidecar. The file
    is replaced atomically, which removes the need for lock files.
    """
    pyi = _file_dir / "cmaps.pyi"
    marker = f"# checksum: {_src_checksum()}\n"
    if not force and pyi.exists():
        try:
            if pyi.read_text().startswith(marker):
                return False
        except OSError:
            pass
    tmp = pyi.with_suffix(".pyi.tmp")
    tmp.write_text(marker + build_stub_text())
    os.replace(tmp, pyi)
    return True


# Best effort stub refresh for editor support. Silent if the package directory
# is read only, as in a standard installed environment.
try:  # pragma: no cover
    write_stub()
except OSError:  # pragma: no cover
    pass
