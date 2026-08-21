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

if TYPE_CHECKING:
    from IPython.display import DisplayHandle

type ColorMap = ListedColormap | LinearSegmentedColormap

CMAP_N: int = 25


_file_dir = Path(__file__).resolve().parent
_src_dir = _file_dir / "data" / "cmaps"

# Backend colormap names resolved once at import. ``build_cm`` consults these.
_plt_registry = mpl.colormaps  # public matplotlib ColormapRegistry
_plt_cmap_list = list(_plt_registry)
_cmocean_cmap_list = list(cmocean.cm.cmapnames)


# ---------------------------------------------------------------------------
# Colormap construction and modification
# ---------------------------------------------------------------------------
def build_cm(name: str) -> ColorMap:
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


def get_colors(cmap: ColorMap, N: int) -> list[str]:
    """Sample ``N`` evenly spaced colors from ``cmap`` and return them as hex strings."""
    return [to_hex(c) for c in cmap(np.linspace(0, 1, N))]


_EQ_ATOL = 1e-6  # tolerance consistent with the %.6f text colormap format


def _signature(cmap: ColorMap) -> np.ndarray:
    """Return the 256 point RGB sampling used for colormap equality tests."""
    return cmap(np.linspace(0.0, 1.0, 256))[:, :3]


def add_colors_to_cmap(
    obj: str | list[str],
    cmap: ColorMap,
    idx: int = 256,
    N: int = 256,
    gamma: float = 1.0,
    cmap_name: str | None = None,
    format: Literal["linear", "listed", "hex"] = "linear",
) -> ColorMap | list[str]:
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
        Gamma applied when rebuilding a linear segmented colormap.
    cmap_name : str, optional
        Name for the returned colormap.
    format : {"linear", "listed", "hex"}, default "linear"
        Output format. ``"hex"`` uses listed colors internally and returns
        hexadecimal color strings.
    """
    if format not in {"linear", "listed", "hex"}:
        raise ValueError("`format` must be 'linear', 'listed', or 'hex'.")

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
    new_colors = colors[:idx] + colors_to_add + colors[idx:]

    if format in {"listed", "hex"}:
        res = ListedColormap(new_colors, N=len(new_colors), name=cmap_name)
    else:
        res = LinearSegmentedColormap.from_list(cmap_name, new_colors, N=N, gamma=gamma)

    if format == "hex":
        return get_colors(res, res.N)
    return res


def adjust_cmap(
    cmap: str | ColorMap,
    N: int | None = None,
    *,
    split: tuple[float, float] = (0.0, 1.0),
    add_colors: dict[int, str | list[str]] | None = None,
    r: bool = False,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """
    Modify a colormap by slicing, reversal, color insertion and output format.
    # ... (docstrings omitted for brevity)
    """
    if not isinstance(split, tuple) or len(split) != 2:
        raise ValueError("`split` must be a tuple of two floats (start, end).")
    if format not in {"linear", "listed", "hex"}:
        raise ValueError("`format` must be 'linear', 'listed', or 'hex'.")

    if N is None:
        N = cmap.N if format == "hex" else CMAP_N

    cmap_name = cmap.name
    if r:
        cmap = cmap.reversed()
        cmap_name = f"reversed_{cmap_name}"

    colors = [cmap(value) for value in np.linspace(split[0], split[1], N)]

    if format in {"listed", "hex"}:
        res = ListedColormap(colors, name=cmap_name)
    else:
        res = LinearSegmentedColormap.from_list(cmap_name, colors, N=N, gamma=gamma)

    if add_colors:
        if not isinstance(add_colors, dict):
            raise TypeError("`add_colors` must be a dict[int, str | list[str]].")
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
            adjusted = add_colors_to_cmap(
                obj=v,
                idx=k,
                cmap=res,
                N=N,
                gamma=gamma,
                cmap_name=cmap_name,
                format=internal_format,
            )
            if isinstance(adjusted, list):
                raise TypeError("Internal colormap conversion returned a color list.")
            res = adjusted

    if format == "hex":
        return get_colors(res, res.N)
    return res


def get_colormap(
    name: str,
    N: int | None,
    r: bool,
    split: tuple[float, float],
    add_colors: dict[int, str | list[str]] | None,
    format: Literal["linear", "listed", "hex"],
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """Resolve ``name`` to a colormap and apply the requested adjustments."""
    cmap = build_cm(name)
    if N is None:
        N = cmap.N if format == "hex" else CMAP_N

    return adjust_cmap(
        cmap=cmap,
        N=N,
        split=split,
        add_colors=add_colors,
        r=r,
        format=format,
        gamma=gamma,
    )


def create(
    colors: list[str],
    N: int = CMAP_N,
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

    if format in {"listed", "hex"}:
        cmap: ColorMap = ListedColormap(colors, N=N, name=name)
    else:
        cmap = LinearSegmentedColormap.from_list(name, colors, N=N, gamma=gamma)

    if save:
        dup = find_duplicate(cmap)
        if dup is not None:
            match_name, is_reversed = dup
            kind = "reversed " if is_reversed else ""
            print(
                f"Creation skipped: identical to existing {kind}colormap {match_name!r}'."
            )
            existing = build_cm(match_name)
            cmap = existing.reversed() if is_reversed else existing
        else:
            rgb = cmap(np.linspace(0.0, 1.0, 256))[:, :3]
            _src_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(_src_dir / f"{name}.txt", rgb, fmt="%.6f")
            _registry.cache_clear()
            list_cmaps.cache_clear()
            cmap_index.cache_clear()

    if format == "hex":
        return get_colors(cmap, cmap.N)
    return cmap


def _concat_cmaps(
    cmap1: ColorMap,
    cmap2: ColorMap,
    N: int = CMAP_N,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """Concatenate two colormaps into a single colormap of ``N`` colors."""

    def _colors(cmap):
        return [to_hex(cmap(v)) for v in np.linspace(0, 1, N // 2)]

    return create(
        _colors(cmap1) + _colors(cmap2),
        N=N,
        format=format,
        gamma=gamma,
    )


def add_or_subtract(
    cmap1: ColorMap,
    cmap2: ColorMap,
    operator: str,
    N: int = CMAP_N,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
) -> ColorMap | list[str]:
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
        [to_hex(tuple(row)) for row in c],
        N=N,
        format=format,
        gamma=gamma,
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


def find_duplicate(cmap: ColorMap) -> tuple[str, bool] | None:
    """
    Return ``(name, reversed)`` if ``cmap`` matches a registered colormap.

    Matching is evaluated on the 256 point RGB sampling. ``reversed`` is True
    when the match is against the reversed form. Returns ``None`` otherwise.
    """
    target = _signature(cmap)
    for public_name, source_name in _registry().items():
        existing = _signature(build_cm(source_name))
        if np.allclose(target, existing, atol=_EQ_ATOL):
            return public_name, False
        if np.allclose(target, existing[::-1], atol=_EQ_ATOL):
            return public_name, True
    return None


def available(show: bool = True) -> list[str] | DisplayHandle:
    """List or show available colormaps"""
    if "ipykernel" in sys.modules and show:
        for source_name in _registry().values():
            display(_default_cmap(source_name))

        return

    return list(list_cmaps())


# ---------------------------------------------------------------------------
# Public combination helpers (documented, importable, no recursion)
# ---------------------------------------------------------------------------
def new(
    colors: list[str],
    N: int = CMAP_N,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
    name: str | None = None,
    save: bool = False,
) -> ColorMap | list[str]:
    """Create a colormap from a list of colors."""
    return create(
        colors,
        N=N,
        format=format,
        gamma=gamma,
        name=name,
        save=save,
    )


def concat(
    cmap1: ColorMap,
    cmap2: ColorMap,
    N: int = CMAP_N,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """Concatenate two colormaps."""
    return _concat_cmaps(cmap1, cmap2, N=N, format=format, gamma=gamma)


def add(
    cmap1: ColorMap,
    cmap2: ColorMap,
    N: int = CMAP_N,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """Add two colormaps channel wise."""
    return add_or_subtract(cmap1, cmap2, "+", N=N, format=format, gamma=gamma)


def subtract(
    cmap1: ColorMap,
    cmap2: ColorMap,
    N: int = CMAP_N,
    *,
    format: Literal["linear", "listed", "hex"] = "linear",
    gamma: float = 1.0,
) -> ColorMap | list[str]:
    """Subtract two colormaps channel wise."""
    return add_or_subtract(cmap1, cmap2, "-", N=N, format=format, gamma=gamma)


# ---------------------------------------------------------------------------
# Dynamic per-colormap callables
# ---------------------------------------------------------------------------
@cache
def _make(public_name: str, source_name: str):
    def cmap(
        N: int | None = None,
        r: bool = False,
        *,
        split: tuple[float, float] = (0.0, 1.0),
        add_colors: dict[int, str | list[str]] | None = None,
        format: Literal["linear", "listed", "hex"] = "linear",
        gamma: float = 1.0,
    ):
        return get_colormap(source_name, N, r, split, add_colors, format, gamma)

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
    return get_colormap(source_name, None, False, (0.0, 1.0), None, "linear", 1.0)


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
    def __getitem__(self, name: str) -> LinearSegmentedColormap | ListedColormap:
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
    "(N: int | None = None, r: bool = False, *, "
    "split: tuple[float, float] = ..., "
    "add_colors: dict[int, str | list[str]] | None = None, "
    'format: Literal["linear", "listed", "hex"] = "linear", '
    "gamma: float = 1.0) -> ListedColormap | LinearSegmentedColormap | list[str]: ..."
)

_STUB_HEADER = """from typing import Literal
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from IPython.display import DisplayHandle
type ColorMap = ListedColormap | LinearSegmentedColormap


def new(colors: list[str], N: int | None = None, *, format: Literal["linear", "listed", "hex"] = \"linear\", gamma: float = 1.0, name: str | None = None, save: bool = False) ->  ColorMap | list[str]: ...
def create(colors: list[str], N: int | None = None , *, format: Literal["linear", "listed", "hex"] = \"linear\", gamma: float = 1.0, name: str | None = None, save: bool = False) ->  ColorMap | list[str]: ...
def concat(cmap1:  ColorMap, cmap2:  ColorMap, N: int | None = None, *, format: Literal["linear", "listed", "hex"] = \"linear\", gamma: float = 1.0) ->  ColorMap | list[str]: ...
def add(cmap1:  ColorMap, cmap2:  ColorMap, N: int | None = None, *, format: Literal["linear", "listed", "hex"] = \"linear\", gamma: float = 1.0) ->  ColorMap | list[str]: ...
def subtract(cmap1:  ColorMap, cmap2:  ColorMap, N: int | None = None, *, format: Literal["linear", "listed", "hex"] = \"linear\", gamma: float = 1.0) ->  ColorMap | list[str]: ...
def available(show:bool=True) -> list[str] | DisplayHandle: ...
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
