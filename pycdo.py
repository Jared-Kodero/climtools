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
    Run a Climate Data Operators command and return the completed process.

    This function is a thin wrapper around the internal command runner. It
    validates that ``cmd`` is a list of strings, then delegates execution to
    ``_run(cmd, n_threads=n_threads)``.

    Parameters
    ----------
    cmd : list[str]
        Command arguments passed to CDO. The first element should usually be
        ``"cdo"``, followed by CDO options, operators, input files, and output
        files. Each element must be a string.
    n_threads : int | None, optional
        Number of OpenMP threads to use for CDO execution. If ``None``, the
        default thread configuration is used.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process object returned by the underlying command runner.

    Raises
    ------
    TypeError
        If ``cmd`` is not a list of strings.

    Examples
    --------
    >>> run(["cdo", "-O", "timmean", "input.nc", "output.nc"])
    >>> run(["cdo", "-O", "remapbil,target_grid.txt", "input.nc", "output.nc"], n_threads=4)

    CDO command-line reference
    --------------------------
    Usage
        cdo [Options] Operator1 [-Operator2 [-OperatorN]]

    Info
        --attribs <arbitrary|filesOnly|onlyFirst|noOutput|obase>
            Lists all operators with selected features or the attributes of
            given operator(s). The argument can be an operator name or a
            combination of arbitrary, filesOnly, onlyFirst, noOutput, and obase.
        --config <all|all-json|<specific_feature_name>>
            Prints all features and the enabled status. Use ``all`` to show
            explicit feature names.
        --envvars
            Prints the environment variables of CDO.
        --module_info <module name>
            Prints the list of operators for a module.
        --operators
            Prints the list of operators.
        --operators_no_output
            Prints all operators that produce no output.
        --rusage
            Prints information about resource utilization.
        --settings
            Prints the settings of CDO.
        -V, --version
            Prints the version number.

    Output
        -C, --color <auto|no|all>
            Sets behavior of colorized output messages.
        -d, --debug
            Prints all available debug messages.
        -w, --disable_warnings
            Disables warning messages.
        -D, --scoped_debug <comma-separated scopes>
            Enables debug messages for selected scopes. Multiple scopes are
            allowed. Use this option without arguments to list available scopes.
        -s, --silent
            Enables silent mode.

    Multi-threading
        -P, --num_threads <nthreads>
            Sets the number of OpenMP threads.
        --worker <num>
            Sets the number of workers used to decode or decompress GRIB records.

    Search methods
        --gridsearchradius <degrees[0..180]>
            Sets the grid search radius in degrees.

    Format specific
        --chunksize <size>
            Sets the NetCDF4 chunk size.
        -k, --chunktype <auto|grid|lines>
            Sets the NetCDF4 chunk type.
        --eccodes
            Uses ecCodes to decode or encode GRIB1 messages.
        -Q, --sortname
            Sorts NetCDF parameter names alphanumerically.

    CGRIBEX
        -R, --regular
            Converts GRIB1 data from global reduced Gaussian grid to regular
            Gaussian grid. Applies only to CGRIBEX.
        -t, --table <codetab>
            Sets the GRIB1 default parameter code table name or file. Applies
            only to CGRIBEX. Predefined tables include echam4, echam5, echam6,
            mpiom1, ecmwf, remo, cosmo002, cosmo201, cosmo202, cosmo203,
            cosmo205, and cosmo250.

    Numeric
        -b, --default_datatype <nbits>
            Sets the number of bits for the output precision. Supported values
            include I8, I16, I32, F32, F64 for nc1, nc2, nc4, nc4c, nc5, and
            nczarr; U8, U16, U32 for nc4, nc4c, and nc5; F32 and F64 for grb2,
            srv, ext, and ieg; and P1 through P24 for grb1 and grb2.
        --double
            Uses double-precision floats for data in memory.
        --enableexcept <except>
            Enables individual floating-point traps. Supported values include
            DIVBYZERO, INEXACT, INVALID, OVERFLOW, UNDERFLOW, and ALL_EXCEPT.
        --float
            Uses single-precision floats for data in memory.
        --percentile <method>
            Sets the percentile method. Supported methods include nrank, nist,
            rtype8, and NumPy-style methods such as linear, lower, higher, and
            nearest.
        --precision <float_digits[,double_digits]>
            Sets precision for displaying floating-point data. The default is
            7 digits for floats and 15 digits for doubles.
        --seed <seed>
            Sets the seed for a new sequence of pseudo-random numbers. The seed
            must be greater than or equal to 0.
        --single
            Uses single-precision floats for data in memory.

    History
        --disable_history <true|false>
            Overrides CDO_DISABLE_HISTORY. See the corresponding environment
            variable.
        --history
            Appends to the NetCDF ``history`` global attribute.
        --history_info <true|false>
            Overrides CDO_HISTORY_INFO. See the corresponding environment
            variable.
        --no_history
            Does not append to the NetCDF ``history`` global attribute.
        --reset_history <true|false>
            Overrides CDO_RESET_HISTORY. See the corresponding environment
            variable.

    Compression
        -Z, --compress
            Enables compression. The default is SZIP.
        -z, --compression_type <aec|jpeg|zip[_1-9]|zstd[1-19]>
            Sets the compression type. ``aec`` applies AEC compression to GRIB2
            records; ``jpeg`` applies JPEG compression to GRIB2 records;
            ``zip[_1-9]`` applies Deflate compression to NetCDF4 variables; and
            ``zstd[_1-19]`` applies Zstandard compression to NetCDF4 variables.
        -F, --filter <filterspec>
            Sets the NetCDF4 filter specification.
        --shuffle
            Enables shuffling of variable data bytes before NetCDF compression.

    Options
        -a, --absolute_taxis
            Generates an absolute time axis.
        -S, --cdo_diagnostic
            Creates an extra output stream for the TIMSTAT module containing the
            number of non-missing values for each output period.
        -c, --check_data_range
            Enables checks for data overflow.
        --cmor
            Enables CMOR-conformant NetCDF output.
        --disable_file_suffix <true|false>
            Overrides CDO_DISABLE_FILE_SUFFIX. See the corresponding environment
            variable.
        --download_path <path>
            Overrides CDO_DOWNLOAD_PATH. See the corresponding environment
            variable.
        -A, --dryrun
            Performs a dry run and prints the processed CDO call.
        --file_suffix <suffix>
            Overrides CDO_FILE_SUFFIX. See the corresponding environment
            variable.
        --force
            Forces a CDO process.
        -f, --format <grb1|grb2|nc1|nc2|nc4|nc4c|nc5|nczarr|srv|ext|ieg>
            Sets the output file format.
        -g, --grid <grid>
            Sets the default grid name or file. Available grids include
            F<XXX>, t<RES>, tl<RES>, r<NX>x<NY>, global_<DXY>, zonal_<DY>,
            gme<NI>, lon=<LON>/lat=<LAT>, and hpz<ZOOM>.
        -M, --has_missval
            Sets HAS_MISSVAL to true.
        --icon_grids <path>
            Overrides CDO_ICON_GRIDS. See the corresponding environment
            variable.
        --ignore_time_bounds
            Ignores time bounds for time-range statistics.
        -i, --institution <institute_name>
            Sets the institution name.
        -u, --interactive
            Enables CDO interactive mode.
        -L, --lock_io
            Locks I/O for sequential access.
        --netcdf_hdr_pad <nbr>
            Pads the NetCDF output header with ``nbr`` bytes.
        --no_remap_weights
            Switches off generation of remap weights.
        -O, --overwrite
            Overwrites an existing output file, if checked.
        --pedantic
            Treats warnings as errors.
        --reduce_dim
            Reduces NetCDF dimensions.
        -r, --relative_taxis
            Generates a relative time axis.
        --remap_weights <0|1>
            Enables or disables generation of remap weights. The default is 1.
        -m, --set_missval <missval>
            Sets the missing value of non-NetCDF files. The default is -9e+33.
        --sortparam
            Sorts parameters.
        --test <true|false>
            Overrides CDO_TEST. See the corresponding environment variable.
        -T, --timer
            Enables the timer.
        --timestat_date <srcdate>
            Sets the target timestamp for temporal statistics. Supported values
            are first, middle, midhigh, and last source timestep.
        --use_fftw <true|false>
            Sets FFTW usage.
        --use_time_bounds
            Enables use of time bounds.
        -v, --verbose
            Prints extra details for some operators.
        --version_info <true|false>
            Overrides CDO_VERSION_INFO. See the corresponding environment
            variable.
        -l, --zaxis <zaxis>
            Sets the default z-axis name or file.

    Help
        --apply
            Shows explanation and examples for ``-apply`` syntax.
        --argument_groups
            Shows explanation and examples for subgrouping operators with
            bracket syntax.
        -h, --help <operator>
            Shows help information for the given operator or the general CDO
            usage message.

    Environment variables
        CDO_CORESIZE <max. core dump size>
            Largest size, in bytes, of a core file that may be created.
        CDO_DISABLE_FILE_SUFFIX <true|false>
            If true, disables file suffixes.
        CDO_DISABLE_HISTORY <true|false>
            If true, disables the history attribute.
        CDO_DOWNLOAD_PATH <path>
            Path where CDO can store downloads.
        CDO_FILE_SUFFIX <suffix>
            Default filename suffix.
        CDO_HISTORY_INFO <true|false>
            If false, does not write information to the global history attribute.
            The default is true.
        CDO_ICON_GRIDS <path>
            Root directory of installed ICON grids, for example
            ``/pool/data/ICON``.
        CDO_RESET_HISTORY <true|false>
            If true, resets the global history attribute. The default is false.
        CDO_TEST <true|false>
            If true, enables new features for testing. The default is false.
        CDO_VERSION_INFO <true|false>
            If false, disables the global NetCDF attribute CDO. The default is
            true.

    Notes
    -----
    This reference reflects CDO version 2.5.0, copyright 2002-2024
    MPI für Meteorologie. CDO is free software and comes with no warranty.
    Bugs may be reported to the CDO maintainers.
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
