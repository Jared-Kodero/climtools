"""
cdo
===

Python wrapper for the Climate Data Operators (CDO) command-line tool.

This module provides a thin, xarray-aware façade over a subset of CDO
operators relevant to climate-model post-processing: horizontal
interpolation onto regular lon-lat grids, temporal concatenation, and
arbitrary pass-through execution. NetCDF file paths and xarray objects are
accepted interchangeably as inputs, and either may be returned as output.

Design notes
------------
- Temporary files are isolated to a per-process directory created lazily on
  first use and removed at interpreter exit via ``atexit``.
- CDO failures propagate as ``RuntimeError`` with the captured stderr.
- Environment variables that influence CDO behaviour (e.g.
  ``REMAP_EXTRAPOLATE``) are set inside a context manager and restored on
  exit, so they cannot leak across calls.
- The default output overwrite flag ``-O`` is always passed, removing the
  need for callers to clear stale output files.

Usage
-----
    >>> import cdo
    >>> cdo.remapbil("input.nc", "output.nc",
    ...            resolution=0.25,
    ...            bbox=(-180, -90, 180, 90))
    >>> cdo.mergetime(["file1.nc", "file2.nc"], "merged.nc")
    >>> cdo.run(["-sellonlatbox,-10,40,30,70", "in.nc", "out.nc"])

Reference
---------
CDO user guide: https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Iterable, Iterator, Literal

import xarray as xr

from .tools import n_cpus

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------


__all__ = [
    "run",
    "remap",
    "remapbil",
    "remapbic",
    "remapnn",
    "remapdis",
    "remapcon",
    "remaplaf",
    "mergetime",
    "RemapMethod",
]

os.environ.setdefault("CDO_VERSION_INFO", "false")
os.environ.setdefault("CDO_HISTORY_INFO", "false")

RemapMethod = Literal[
    "remapbil",
    "remapbic",
    "remapnn",
    "remapdis",
    "remapcon",
    "remaplaf",
]
_VALID_METHODS: tuple[str, ...] = (
    "remapbil",
    "remapbic",
    "remapnn",
    "remapdis",
    "remapcon",
    "remaplaf",
)
_EXTRAPOLATING_METHODS: frozenset[str] = frozenset(
    {"remapbil", "remapbic", "remapnn", "remaplaf"}
)
_DEFAULT_BBOX: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)

_N_CPUS: int = n_cpus


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


class _TmpDir:
    """Process-wide temporary directory, created lazily and cleaned at exit."""

    _path: Path | None = None

    @classmethod
    def get(cls) -> Path:
        if cls._path is None:
            cls._path = Path(tempfile.mkdtemp(prefix="cdo_py_"))
            atexit.register(cls._cleanup)
        return cls._path

    @classmethod
    def new_file(cls, suffix: str = ".nc") -> Path:
        return cls.get() / f"{uuid.uuid4().hex}{suffix}"

    @classmethod
    def _cleanup(cls) -> None:
        if cls._path is not None and cls._path.exists():
            shutil.rmtree(cls._path, ignore_errors=True)


@contextmanager
def _env(**kwargs: str) -> Iterator[None]:
    """Temporarily set environment variables, restoring prior values on exit."""
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in kwargs}
    os.environ.update(kwargs)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _check_cdo() -> None:
    if not shutil.which("cdo"):
        raise RuntimeError(
            "CDO executable not found in PATH. Install CDO and ensure it is accessible."
        )


def _run(args: list[str], n_threads: int | None = None) -> subprocess.CompletedProcess:
    """Execute a CDO command. Raise RuntimeError on non-zero exit."""
    _check_cdo()
    threads = n_threads if n_threads is not None else min(_N_CPUS, 32)
    cmd = ["cdo", "-s", "-O", "-P", str(threads), *args]
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"CDO command failed: {exc.stderr.strip()}") from exc


def _to_netcdf(
    obj: Path | str | xr.DataArray | xr.Dataset,
) -> tuple[Path, str | None]:
    """
    Resolve an input to a NetCDF file path.

    Returns the path and, if the input was a DataArray, the original variable
    name so the caller can re-extract the same field after CDO writes back a
    Dataset.
    """
    if isinstance(obj, xr.DataArray):
        da_name = obj.name or "var"
        path = _TmpDir.new_file()
        obj.to_dataset(name=da_name).to_netcdf(path)
        return path, da_name

    if isinstance(obj, xr.Dataset):
        path = _TmpDir.new_file()
        obj.to_netcdf(path)
        return path, None

    p = Path(obj)
    if not p.exists():
        raise FileNotFoundError(f"Input file does not exist: {p}")
    return p, None


def _write_lonlat_grid(
    bbox: tuple[float, float, float, float],
    resolution: float,
) -> Path:
    """Write a CDO lon-lat grid description file and return its path."""
    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_max <= lon_min or lat_max <= lat_min:
        raise ValueError(
            f"Invalid bbox {bbox}. Expected (lon_min, lat_min, lon_max, lat_max) with lon_min<lon_max and lat_min<lat_max."
        )
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    xsize = int(round((lon_max - lon_min) / resolution)) + 1
    ysize = int(round((lat_max - lat_min) / resolution)) + 1

    lines = [
        "gridtype = lonlat",
        f"xsize    = {xsize}",
        f"ysize    = {ysize}",
        f"xfirst   = {lon_min}",
        f"xinc     = {resolution}",
        f"yfirst   = {lat_min}",
        f"yinc     = {resolution}",
    ]
    grid_file = _TmpDir.new_file(suffix=".grid")
    grid_file.write_text("\n".join(lines))
    return grid_file


def _open_result(path: Path, da_name: str | None) -> xr.DataArray | xr.Dataset:
    ds = xr.open_dataset(path, chunks="auto")
    if da_name and da_name in ds:
        return ds[da_name]
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    cmd: list[str],
    n_threads: int | None = None,
) -> subprocess.CompletedProcess:
    """
    Execute an arbitrary CDO command.

    Parameters
    ----------
    cmd : list of str
        CDO operator chain and arguments, excluding the ``cdo`` executable.
    n_threads : int, optional
        Override the default OpenMP thread count passed via ``-P``. Defaults
        to ``min(os.cpu_count(), 32)``.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process object containing stdout, stderr, and return code.

    Raises
    ------
    TypeError
        If ``cmd`` is not a list of strings.
    RuntimeError
        If CDO exits with a non-zero status.

    Examples
    --------
    >>> run(["remapbil,gridfile", "infile.nc", "outfile.nc"])
    """
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        raise TypeError("cmd must be a list of strings.")
    return _run(cmd, n_threads=n_threads)


def remap(
    obj: Path | str | xr.DataArray | xr.Dataset,
    outfile: Path | str | None = None,
    *,
    method: RemapMethod = "remapbil",
    resolution: float = 0.25,
    bbox: tuple[float, float, float, float] = _DEFAULT_BBOX,
    extrapolate: bool = False,
    as_xarray: bool = False,
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.DataArray | xr.Dataset:
    """
    Horizontally interpolate a NetCDF dataset onto a regular lon-lat grid.

    Parameters
    ----------
    obj : Path, str, xr.DataArray, or xr.Dataset
        Input data. xarray objects are written to a temporary NetCDF first.
    outfile : Path or str, optional
        Output NetCDF path. A temporary file is used if omitted.
    method : {'remapbil','remapbic','remapnn','remapdis','remapcon','remaplaf'}
        CDO remap operator.
    resolution : float
        Target grid spacing in degrees.
    bbox : tuple of float
        Target grid extent as ``(lon_min, lat_min, lon_max, lat_max)``.
        Default is global.
    extrapolate : bool
        If True, set ``REMAP_EXTRAPOLATE=on`` for the duration of the call.
        Only valid for ``remapbil``, ``remapbic``, ``remapnn``, ``remaplaf``.
    as_xarray : bool
        If True, open the output as xarray and return it.
    compression : str
        CDO ``-z`` value (e.g. ``"zip"``, ``"zip_5"``, ``"zstd_3"``).
    datatype : str
        CDO ``-b`` value (e.g. ``"F32"``, ``"F64"``).
    n_threads : int, optional
        OpenMP threads to pass via ``-P``.

    Returns
    -------
    Path or xr.DataArray or xr.Dataset
        Output path, or the opened xarray object if ``as_xarray`` is True.

    Raises
    ------
    ValueError
        If ``method`` is unknown, if ``extrapolate`` is requested for a method
        that does not support it, or if ``bbox`` or ``resolution`` are invalid.
    FileNotFoundError
        If ``obj`` is a path that does not exist.
    RuntimeError
        If CDO exits with a non-zero status.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}.")
    if extrapolate and method not in _EXTRAPOLATING_METHODS:
        raise ValueError(
            f"extrapolate is not supported for method {method!r}. Use one of {sorted(_EXTRAPOLATING_METHODS)}."
        )

    input_path, da_name = _to_netcdf(obj)
    grid_file = _write_lonlat_grid(bbox, resolution)
    output_path = Path(outfile) if outfile is not None else _TmpDir.new_file()

    args = [
        "-b",
        datatype,
        "-z",
        compression,
        f"{method},{grid_file}",
        str(input_path),
        str(output_path),
    ]

    env = {"REMAP_EXTRAPOLATE": "on"} if extrapolate else {}
    with _env(**env):
        _run(args, n_threads=n_threads)

    if as_xarray:
        return _open_result(output_path, da_name)
    return output_path


# Backwards-compatible method-specific wrappers.


@wraps(remap)
def remapbil(obj, outfile=None, **kwargs):
    """Bilinear interpolation. See :func:`remap` for parameters."""
    return remap(obj, outfile, method="remapbil", **kwargs)


@wraps(remap)
def remapbic(obj, outfile=None, **kwargs):
    """Bicubic interpolation. See :func:`remap` for parameters."""
    return remap(obj, outfile, method="remapbic", **kwargs)


@wraps(remap)
def remapnn(obj, outfile=None, **kwargs):
    """Nearest-neighbour interpolation. See :func:`remap` for parameters."""
    return remap(obj, outfile, method="remapnn", **kwargs)


@wraps(remap)
def remapdis(obj, outfile=None, **kwargs):
    """Distance-weighted average of nearest neighbours. See :func:`remap`."""
    return remap(obj, outfile, method="remapdis", **kwargs)


@wraps(remap)
def remapcon(obj, outfile=None, **kwargs):
    """First-order conservative remapping. See :func:`remap`."""
    return remap(obj, outfile, method="remapcon", **kwargs)


@wraps(remap)
def remaplaf(obj, outfile=None, **kwargs):
    """Largest-area-fraction remapping, suited to categorical fields. See :func:`remap`."""
    return remap(obj, outfile, method="remaplaf", **kwargs)


def mergetime(
    infiles: Iterable[Path | str],
    outfile: Path | str,
    *,
    as_xarray: bool = False,
    delete_input: bool = False,
    compression: str = "zip",
    datatype: str = "F32",
    n_threads: int | None = None,
) -> Path | xr.Dataset:
    """
    Concatenate multiple NetCDF files along the time dimension.

    CDO's ``-mergetime`` orders records by their internal time coordinate, so
    the order in which paths are supplied does not affect the result.

    Parameters
    ----------
    infiles : iterable of Path or str
        Input NetCDF files.
    outfile : Path or str
        Output NetCDF path.
    as_xarray : bool
        If True, return the merged file opened as an xarray Dataset.
    delete_input : bool
        Remove input files after a successful merge.
    compression : str
        CDO ``-z`` value.
    datatype : str
        CDO ``-b`` value.
    n_threads : int, optional
        OpenMP threads to pass via ``-P``.

    Returns
    -------
    Path or xr.Dataset
        Output path, or the opened Dataset if ``as_xarray`` is True.

    Raises
    ------
    ValueError
        If no input files are provided.
    FileNotFoundError
        If any input file does not exist.
    RuntimeError
        If CDO exits with a non-zero status.
    """
    paths = [Path(f).resolve() for f in infiles]
    if not paths:
        raise ValueError("No input files provided.")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file does not exist: {p}")
    paths.sort()  # purely for reproducible logging

    args = [
        "-b",
        datatype,
        "-z",
        compression,
        "-mergetime",
        *(str(p) for p in paths),
        str(outfile),
    ]
    _run(args, n_threads=n_threads)

    if delete_input:
        for p in paths:
            try:
                p.unlink()
            except OSError as exc:
                raise RuntimeError(f"Failed to delete input file {p}: {exc}") from exc

    if as_xarray:
        return xr.open_dataset(outfile, chunks="auto")
    return Path(outfile)
