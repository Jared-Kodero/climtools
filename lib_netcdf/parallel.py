"""Collective NetCDF-4 output from a distributed xarray Dataset.

Model: every rank holds a contiguous, non-overlapping slab of one dimension of
the same logical dataset, and identical copies of everything else. The slab
boundaries are recovered at write time from an all-gather of the local lengths
along that dimension, so no rank has to be told its global offset.

This module is the NetCDF half of the parallel writer. Process coordination
and the C ABI live in :mod:`climtools.lib_mpi`.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import math
import warnings
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..lib_mpi import native
from ..lib_mpi.native import NativeLibraryError
from ..lib_mpi.runtime import mpi

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from os import PathLike
    from typing import Any

_LOGGER = logging.getLogger(__name__)


class InconsistentRanksError(NativeLibraryError):
    """Raised when the ranks do not agree about the dataset they are writing.

    Every rank detects this at the same all-gather, so it is safe to raise it
    as an ordinary Python exception; no rank is left waiting in a collective.
    """


_NC_FROM_NUMPY = {
    "float64": native.NC_DOUBLE,
    "float32": native.NC_FLOAT,
    "int64": native.NC_INT64,
    "uint64": native.NC_UINT64,
    "int32": native.NC_INT,
    "uint32": native.NC_UINT,
    "int16": native.NC_SHORT,
    "uint16": native.NC_USHORT,
    "int8": native.NC_BYTE,
    "uint8": native.NC_UBYTE,
}

_TARGET_CHUNK_BYTES = 4 * 1024 * 1024


def _fingerprint(value: str | bytes) -> int:
    """Return a stable positive 63-bit fingerprint for MPI all-gather."""
    payload = value.encode("utf-8") if isinstance(value, str) else value
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def _array_fingerprint(array: np.ndarray) -> int:
    """Return a stable fingerprint including an array's dtype and shape."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return int.from_bytes(digest.digest(), "little") & 0x7FFFFFFFFFFFFFFF


class Encoded:
    """A variable ready to be written: native-endian buffer plus metadata."""

    __slots__ = ("attrs", "data", "dims", "itemsize", "name", "xtype")

    def __init__(
        self,
        name: str,
        dims: Sequence[str],
        data: np.ndarray | None,
        xtype: int,
        attrs: Mapping[str, Any],
    ) -> None:
        self.name = name
        self.dims = tuple(dims)
        self.data = data
        self.xtype = xtype
        self.attrs = attrs
        # NC_CHAR buffers are built later, once the global string width is
        # known; one byte per element either way.
        self.itemsize = 1 if data is None else data.dtype.itemsize


# ------------------------------------------------------------------ encoding


def encode_strings(values: Any, name: str) -> tuple[list[bytes], int, tuple[int, ...]]:
    """Turn a str/bytes/object array into a contiguous NC_CHAR buffer."""
    flat = np.asarray(values, dtype=object).ravel()
    raw = []
    for item in flat:
        if item is None:
            raw.append(b"")
        elif isinstance(item, bytes):
            raw.append(item)
        elif isinstance(item, str):
            raw.append(item.encode("utf-8"))
        else:
            raise NativeLibraryError(
                f"Variable '{name}' contains a non-string object of type "
                + f"{type(item).__name__}."
            )
        if b"\0" in raw[-1]:
            raise NativeLibraryError(
                f"Variable '{name}' contains an embedded NUL byte."
            )
    width = max((len(r) for r in raw), default=1) or 1
    return raw, width, np.shape(values)


def pack_chars(raw: Sequence[bytes], width: int, shape: tuple[int, ...]) -> np.ndarray:
    packed = np.array(raw, dtype=f"S{width}").reshape(shape)
    return np.ascontiguousarray(packed).view("S1").reshape(shape + (width,))


def native_array(array: Any) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype.kind == "b":
        array = array.astype("int8")
    if array.dtype.byteorder not in ("=", "|"):
        array = array.astype(array.dtype.newbyteorder("="))
    return np.ascontiguousarray(array)


_TIME_UNITS = (
    ("days", 86_400_000_000_000),
    ("seconds", 1_000_000_000),
    ("milliseconds", 1_000_000),
    ("microseconds", 1_000),
    ("nanoseconds", 1),
)
_EPOCH = "1970-01-01T00:00:00"


def _time_resolution_code(values: np.ndarray) -> int:
    """Coarsest exact representation of these datetimes/timedeltas."""
    as_int = (
        np.asarray(values)
        .astype("datetime64[ns]" if values.dtype.kind == "M" else "timedelta64[ns]")
        .astype("int64")
    )
    if as_int.size == 0:
        return 1
    for code, (_, divisor) in enumerate(_TIME_UNITS):
        if np.all(as_int % divisor == 0):
            return code
    return len(_TIME_UNITS) - 1


def harmonise_time_encoding(
    variables: dict[str, Any], comm: WorldComm
) -> dict[str, Any]:
    """Give every rank the same CF units for datetime and timedelta axes.

    Left to itself, xarray derives `units` from the values it can see, so each
    rank would choose a different epoch and the file would contain four
    mutually inconsistent time axes under one `units` attribute. This is the
    single most damaging failure mode of a naive distributed writer, so the
    epoch is fixed and only the resolution is negotiated.
    """
    names: list[str] = []
    error: BaseException | None = None
    try:
        if any(not isinstance(name, str) or not name for name in variables):
            raise TypeError("NetCDF variable names must be non-empty strings.")
        names = sorted(
            name
            for name, variable in variables.items()
            if variable.dtype.kind in "Mm" and "units" not in variable.encoding
        )
    except BaseException as exc:
        error = exc
    comm.raise_if_error(error, "time-variable discovery")

    _agree(len(names), comm, "the number of time variables")
    _agree(
        _fingerprint("|".join(names)),
        comm,
        "the names of the time variables",
    )

    for name in names:
        var = variables[name]
        values: np.ndarray | None = None
        error = None
        try:
            values = np.asarray(var.data)
            if values.size and np.isnat(values).any():
                raise NativeLibraryError(
                    f"Variable '{name}' contains NaT values. Configure an "
                    + "explicit encoding with units, dtype, and _FillValue."
                )
        except BaseException as exc:
            error = exc
        comm.raise_if_error(error, f"time encoding for variable '{name}'")

        if values is None:
            raise AssertionError("synchronized time encoding produced no values")
        resolution = 0
        error = None
        try:
            resolution = _time_resolution_code(values)
        except BaseException as exc:
            error = exc
        comm.raise_if_error(error, f"time resolution for variable '{name}'")
        code = max(comm.allgather_i64(resolution))
        unit = _TIME_UNITS[code][0]
        error = None
        try:
            new = var.copy(deep=False)
            encoding = dict(var.encoding)
            encoding["units"] = (
                f"{unit} since {_EPOCH}" if var.dtype.kind == "M" else unit
            )
            encoding["dtype"] = "int64"
            if var.dtype.kind == "M":
                encoding.setdefault("calendar", "proleptic_gregorian")
            new.encoding = encoding
            variables[name] = new
        except BaseException as exc:
            error = exc
        comm.raise_if_error(error, f"time metadata for variable '{name}'")
    return variables


def encode_dataset(
    ds: Any, comm: WorldComm
) -> tuple[dict[str, Encoded], dict[str, Any], dict[str, Any]]:
    """Apply the CF encoders xarray would apply, then flatten to buffers."""
    variables: dict[Any, Any] = {}
    global_attrs: dict[Any, Any] = {}
    xr: Any = None
    error: BaseException | None = None
    try:
        import xarray as xr_module

        xr = xr_module
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"ds must be an xarray.Dataset, got {type(ds).__name__}.")
        try:
            variables, global_attrs = xr.conventions.encode_dataset_coordinates(ds)
        except (AttributeError, TypeError):
            variables, global_attrs = dict(ds.variables), dict(ds.attrs)
    except BaseException as exc:
        error = exc
    comm.raise_if_error(error, "xarray dataset encoding")

    if xr is None:
        raise AssertionError("synchronized dataset encoding did not import xarray")

    variables = harmonise_time_encoding(dict(variables), comm)

    out: dict[str, Encoded] = {}
    strings: dict[str, Any] = {}
    error = None
    try:
        for name, var in variables.items():
            if not isinstance(name, str) or not name:
                raise TypeError("NetCDF variable names must be non-empty strings.")
            encoded = xr.conventions.encode_cf_variable(var, name=name)
            if any(not isinstance(dim, str) or not dim for dim in encoded.dims):
                raise TypeError(
                    f"Dimensions of variable '{name}' must have non-empty "
                    + "string names."
                )
            values = np.asarray(encoded.data)
            attrs = dict(encoded.attrs)
            if values.dtype.kind in ("U", "S", "O"):
                raw, width, shape = encode_strings(values, name)
                strings[name] = (raw, width, shape)
                out[name] = Encoded(name, encoded.dims, None, native.NC_CHAR, attrs)
            else:
                buf = native_array(values)
                key = buf.dtype.name
                if key not in _NC_FROM_NUMPY:
                    raise NativeLibraryError(
                        f"Variable '{name}' with data type {buf.dtype} has no "
                        + "NetCDF-4 classic equivalent. Set an explicit target "
                        + "encoding data type."
                    )
                out[name] = Encoded(name, encoded.dims, buf, _NC_FROM_NUMPY[key], attrs)
    except BaseException as exc:
        error = exc
    comm.raise_if_error(error, "CF variable encoding")
    return out, dict(global_attrs), strings


# ------------------------------------------------------------ communicators


class WorldComm:
    """MPI_COMM_WORLD: every rank contributes a slab."""

    parallel = True

    def __init__(self, rank: int, size: int) -> None:
        self.rank = rank
        self.size = size

    def allgather_i64(self, value: int) -> list[int]:
        return native.allgather_i64(int(value), self.size)

    def raise_if_error(
        self,
        error: BaseException | None,
        phase: str,
    ) -> None:
        """Raise the same synchronized validation error on every rank.

        Parameters
        ----------
        error : BaseException or None
            Failure observed locally, or ``None``.
        phase : str
            Name of the validation phase, used in the message.

        Raises
        ------
        BaseException
            The local exception, unchanged, when every rank failed. A bad
            argument is not a disagreement between ranks, so its type and
            message must survive.
        InconsistentRanksError
            When only some ranks failed.
        """
        failures = self.allgather_i64(1 if error is not None else 0)
        if not any(failures):
            return

        if error is not None and all(failures):
            raise error

        failed_rank = failures.index(1)
        detail: tuple[str, str] | None = None
        if self.rank == failed_rank and error is not None:
            detail = (type(error).__name__, str(error))
        name, message = native.bcast_obj(detail, failed_rank)
        raise InconsistentRanksError(
            f"Rank {failed_rank} failed during {phase} with {name}: {message}"
        )


# ---------------------------------------------------------------- agreement


def _agree(value: int, comm: WorldComm, what: str) -> int:
    values = comm.allgather_i64(int(value))
    if min(values) != max(values):
        raise InconsistentRanksError(
            f"Ranks disagree about {what}: {values}. Every rank must hold matching variables, dimensions, dtypes, and attributes."
        )
    return values[0]


def _attrs_key(attrs: Mapping[str, Any]) -> str:
    """Attribute names and values. Values matter: a per-rank `units` string
    would otherwise be written by rank 0 and silently misdescribe the rest."""
    items = []
    for original_key, value in sorted(attrs.items(), key=lambda item: str(item[0])):
        key = str(original_key)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        items.append(f"{key}={value!r}")
    return ";".join(items)


def _schema_fingerprint(
    encvars: Mapping[str, Encoded], global_attrs: Mapping[str, Any]
) -> int:
    parts = []
    for name in sorted(encvars):
        v = encvars[name]
        parts.append(f"{name}|{v.xtype}|{','.join(v.dims)}|{_attrs_key(v.attrs)}")
    parts.append("GLOBAL:" + _attrs_key(global_attrs))
    return _fingerprint("\n".join(parts))


# ------------------------------------------------------------ decomposition


def _local_dim_sizes(
    encvars: Mapping[str, Encoded], strings: Mapping[str, Any]
) -> dict[str, int]:
    sizes = {}
    for name, v in encvars.items():
        shape = strings[name][2] if v.data is None else v.data.shape
        for d, n in zip(v.dims, shape):
            if d in sizes and sizes[d] != n:
                raise NativeLibraryError(
                    f"Dimension '{d}' length mismatch in variable '{name}': existing length {sizes[d]} conflicts with local size {n}."
                )
            sizes[d] = int(n)
    return sizes


def coord_fingerprint(
    encvars: Mapping[str, Encoded], strings: Mapping[str, Any], dim: str
) -> int:
    """Fingerprint the local index coordinate of ``dim``."""
    v = encvars.get(dim)
    if v is not None and v.dims == (dim,):
        if v.data is not None:
            return _fingerprint(v.data.tobytes())
        return _fingerprint(b"".join(strings[dim][0]))
    return -1


def infer_parallel_dim(
    encvars: Mapping[str, Encoded],
    strings: Mapping[str, Any],
    local_sizes: Mapping[str, int],
    comm: WorldComm,
) -> str:
    """Find the one dimension that is partitioned across ranks."""
    candidates = []
    for dim in sorted(local_sizes):
        lengths = comm.allgather_i64(local_sizes[dim])
        if min(lengths) != max(lengths):
            candidates.append(dim)
            continue
        fp = coord_fingerprint(encvars, strings, dim)
        fps = comm.allgather_i64(fp)
        if fp != -1 and len(set(fps)) > 1:
            candidates.append(dim)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        msg = (
            "No distributed dimension could be inferred across ranks. Every rank reports identical "
            + "dimension lengths and coordinate arrays. Specify dim='<name>' explicitly or execute "
            + "from a single rank."
        )
        raise InconsistentRanksError(msg)
    raise InconsistentRanksError(
        f"Multiple candidate distributed dimensions detected across ranks: {candidates}. "
        + "Specify dim='<name>' explicitly to declare the partitioned dimension."
    )


# ------------------------------------------------------------------ chunking


def default_chunks(
    shape: Sequence[int], axis: int | None, block: int, itemsize: int
) -> tuple[int, ...] | None:
    if not shape or any(n == 0 for n in shape):
        return None
    chunk = [int(n) for n in shape]
    if axis is not None:
        chunk[axis] = max(1, min(int(block), shape[axis]))
    guard = 0
    while math.prod(chunk) * itemsize > _TARGET_CHUNK_BYTES and guard < 1024:
        candidates = [index for index, length in enumerate(chunk) if length > 1]
        if not candidates:
            break
        non_distributed = [index for index in candidates if index != axis]
        index = max(non_distributed or candidates, key=chunk.__getitem__)
        chunk[index] = max(1, chunk[index] // 2)
        guard += 1
    return tuple(chunk)


# -------------------------------------------------------------- attributes


def put_attrs(
    handle: Any,
    varname: str,
    attrs: Mapping[str, Any],
    xtype: int | None,
) -> None:
    for original_key, value in sorted(attrs.items(), key=lambda item: str(item[0])):
        key = str(original_key)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            raw_value = value if isinstance(value, bytes) else value.encode("utf-8")
            if b"\0" in raw_value:
                raise NativeLibraryError(
                    f"Attribute '{key}' contains an embedded NUL byte."
                )
            native.check(
                native.lib.mpi_netcdf_put_att_text(
                    handle,
                    native.b(varname),
                    native.b(key),
                    raw_value,
                ),
                f"attribute '{key}' of '{varname or 'global'}'",
            )
            continue
        arr = np.atleast_1d(np.asarray(value))
        if arr.dtype.kind in ("U", "S", "O"):
            text = " ".join(str(x) for x in arr.ravel())
            native.check(
                native.lib.mpi_netcdf_put_att_text(
                    handle,
                    native.b(varname),
                    native.b(key),
                    native.b(text),
                ),
                f"attribute '{key}' of '{varname or 'global'}'",
            )
            continue
        if key in ("_FillValue", "missing_value") and xtype is not None:
            arr = arr.astype(_numpy_for(xtype), copy=False)
        arr = native_array(arr)
        att_type = _NC_FROM_NUMPY.get(arr.dtype.name)
        if att_type is None:
            raise NativeLibraryError(
                f"Attribute '{key}' of target '{varname or 'global'}' has unsupported data type {arr.dtype}."
            )
        native.check(
            native.lib.mpi_netcdf_put_att_num(
                handle,
                native.b(varname),
                native.b(key),
                att_type,
                arr.size,
                arr.ctypes.data_as(ctypes.c_void_p),
            ),
            f"attribute '{key}' of '{varname or 'global'}'",
        )


def _numpy_for(xtype: int) -> np.dtype[Any]:
    for name, code in _NC_FROM_NUMPY.items():
        if code == xtype:
            return np.dtype(name)
    return np.dtype("float64")


def _normalize_options(
    path: str | PathLike[str],
    partition_dim: str | None,
    deflate: int | None,
    shuffle: bool,
    chunks: Mapping[str, Iterable[int]] | None,
    unlimited_dim: Iterable[str],
    hints: str | None,
    nofill: bool,
    allow_serial: bool,
) -> tuple[str, int | None, dict[str, tuple[int, ...]], set[str]]:
    if partition_dim is not None and (
        not isinstance(partition_dim, str) or not partition_dim
    ):
        raise TypeError("partition_dim must be a non-empty string or None.")
    if isinstance(partition_dim, str) and "\0" in partition_dim:
        raise ValueError("partition_dim cannot contain a NUL character.")

    level: int | None = None
    if deflate is not None:
        if isinstance(deflate, bool) or not isinstance(deflate, Integral):
            raise TypeError("deflate must be an integer in range [0, 9].")
        level = int(deflate)
        if not 0 <= level <= 9:
            raise ValueError("deflate must be an integer in range [0, 9].")

    for name, value in (
        ("shuffle", shuffle),
        ("nofill", nofill),
        ("allow_serial", allow_serial),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool.")

    if chunks is not None and not isinstance(chunks, Mapping):
        raise TypeError("chunks must be a mapping from variable names to shapes.")
    chunk_map: dict[str, tuple[int, ...]] = {}
    for name, shape in (chunks or {}).items():
        if not isinstance(name, str) or not name:
            raise TypeError("Every chunks key must be a non-empty string.")
        if isinstance(shape, (str, bytes)):
            raise TypeError(f"Chunk shape for variable '{name}' must be iterable.")
        normalized_shape = []
        for value in shape:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    f"Chunk lengths for variable '{name}' must be integers."
                )
            length = int(value)
            if length <= 0:
                raise ValueError("Chunk lengths must be positive integers.")
            normalized_shape.append(length)
        chunk_map[name] = tuple(normalized_shape)

    unlimited_items = (
        (unlimited_dim,) if isinstance(unlimited_dim, str) else unlimited_dim
    )
    unlimited: set[str] = set()
    for name in unlimited_items:
        if not isinstance(name, str) or not name:
            raise TypeError("Unlimited-dimension names must be non-empty strings.")
        if "\0" in name:
            raise ValueError("Unlimited-dimension names cannot contain NUL characters.")
        unlimited.add(name)

    if hints is not None and not isinstance(hints, str):
        raise TypeError("hints must be a string or None.")
    if hints is not None and "\0" in hints:
        raise ValueError("hints cannot contain a NUL character.")

    output_path = str(Path(path).expanduser().resolve(strict=False))
    if "\0" in output_path:
        raise ValueError("path cannot contain a NUL character.")
    return output_path, level, chunk_map, unlimited


def _validate_schema_text(
    encvars: Mapping[str, Encoded],
    global_attrs: Mapping[str, Any],
) -> None:
    attr_keys = list(global_attrs)
    attr_keys.extend(name for variable in encvars.values() for name in variable.attrs)
    if any(not isinstance(name, str) or not name for name in attr_keys):
        raise TypeError("NetCDF attribute names must be non-empty strings.")
    names = list(encvars)
    names.extend(dim for variable in encvars.values() for dim in variable.dims)
    names.extend(attr_keys)
    if any("\0" in name for name in names):
        raise ValueError("NetCDF names cannot contain NUL characters.")


# ----------------------------------------------------------------- main API


@mpi(all_ranks=True)
def to_netcdf(
    ds: Any,
    path: str | PathLike[str],
    partition_dim: str | None = None,
    deflate: int | None = None,
    shuffle: bool = True,
    chunks: Mapping[str, Iterable[int]] | None = None,
    unlimited_dim: Iterable[str] = (),
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
    strict_compression: bool = False,
) -> str:
    """Write a distributed xarray dataset to one NetCDF-4 file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset slab owned by the current rank. All ranks must supply matching
        variable names, dtypes, attributes, and dimension names.
    path : str or os.PathLike
        Output path visible to every rank.
    partition_dim : str or None, optional
        Partitioned dimension. The writer infers it when omitted.
    deflate : int or None, optional
        Deflate level from 0 to 9. Compression requires parallel filter support
        in the active NetCDF-C/HDF5 stack.
    shuffle : bool, optional
        Enable the HDF5 shuffle filter when compression is requested.
    chunks : mapping of str to iterable of int or None, optional
        Explicit chunk shape for selected variables.
    unlimited_dim : iterable of str, optional
        Dimensions to define as unlimited record dimensions.
    hints : str or None, optional
        Semicolon-separated MPI-IO hints in ``key=value`` form.
    nofill : bool, optional
        Disable NetCDF pre-filling when ``True``.
    allow_serial : bool, optional
        Permit execution with a one-rank MPI world.
    strict_compression : bool, optional
        Fail when ``deflate`` is requested but the linked NetCDF-C and HDF5
        lack parallel filter support. When ``False``, compression is disabled
        and a warning is issued instead.

    Returns
    -------
    str
        Output path after the collective write completes.

    Raises
    ------
    TypeError
        If ``deflate`` is not an integer.
    ValueError
        If ``deflate`` or a chunk length is outside its valid range.
    NativeLibraryError
        If MPI or NetCDF-C reports an error, or parallel requirements are not
        satisfied.
    InconsistentRanksError
        If ranks disagree about writer options or dataset schema.

    Notes
    -----
    Every rank in ``MPI_COMM_WORLD`` must call this function in the same order.
    """
    rank, size = native.init()
    require_parallel(size, allow_serial)
    comm = WorldComm(rank, size)
    dropped_deflate = False

    normalized: (
        tuple[
            str,
            int | None,
            dict[str, tuple[int, ...]],
            set[str],
        ]
        | None
    ) = None
    error: BaseException | None = None
    try:
        if not isinstance(strict_compression, bool):
            raise TypeError("strict_compression must be a bool.")
        normalized = _normalize_options(
            path,
            partition_dim,
            deflate,
            shuffle,
            chunks,
            unlimited_dim,
            hints,
            nofill,
            allow_serial,
        )
        if normalized[1] is not None and not bool(
            native.lib.mpi_netcdf_has_parallel_filters()
        ):
            if strict_compression:
                raise NativeLibraryError(
                    "Deflate compression requires NetCDF-C and HDF5 built "
                    + "with parallel filter support. Pass deflate=None, or "
                    + "strict_compression=False to write uncompressed."
                )
            # Filter support is a property of the linked library, so this
            # decision is identical on every rank and cannot desynchronize
            # them. Downgrading here keeps the documented default call
            # working on the many stacks built without parallel filters.
            normalized = (normalized[0], None, normalized[2], normalized[3])
            dropped_deflate = True
    except BaseException as exc:
        error = exc
    comm.raise_if_error(error, "writer option validation")
    if normalized is None:
        raise AssertionError("synchronized option validation produced no result")
    output_path, deflate_level, chunk_map, unlimited = normalized

    if dropped_deflate and rank == 0:
        warnings.warn(
            "Deflate compression was requested but the linked NetCDF-C and "
            + "HDF5 lack parallel filter support; writing uncompressed. Pass "
            + "strict_compression=True to make this an error.",
            RuntimeWarning,
            stacklevel=2,
        )

    option_key = repr(
        (
            output_path,
            partition_dim,
            -1 if deflate_level is None else deflate_level,
            shuffle,
            sorted(chunk_map.items()),
            sorted(unlimited),
            hints,
            nofill,
            allow_serial,
            strict_compression,
        )
    )
    _agree(_fingerprint(option_key), comm, "the writer options")

    handle = None
    try:
        encvars, global_attrs, strings = encode_dataset(ds, comm)
        error = (
            None
            if encvars
            else NativeLibraryError("Dataset contains no variables to write.")
        )
        comm.raise_if_error(error, "dataset validation")

        schema_key = 0
        error = None
        try:
            _validate_schema_text(encvars, global_attrs)
            schema_key = _schema_fingerprint(encvars, global_attrs)
        except BaseException as exc:
            error = exc
        comm.raise_if_error(error, "schema validation")
        _agree(schema_key, comm, "the schema")

        local_sizes: dict[str, int] = {}
        error = None
        try:
            local_sizes = _local_dim_sizes(encvars, strings)
            if not local_sizes:
                raise NativeLibraryError(
                    "Dataset contains no dimensions; a partitioned dimension "
                    + "is required."
                )
        except BaseException as exc:
            error = exc
        comm.raise_if_error(error, "dimension validation")
        _agree(len(local_sizes), comm, "the number of dimensions")

        pdim = partition_dim
        if pdim is None:
            pdim = (
                infer_parallel_dim(encvars, strings, local_sizes, comm)
                if size > 1
                else next(iter(sorted(local_sizes)))
            )
        _agree(_fingerprint(pdim), comm, "the partitioned dimension")
        error = None
        if pdim not in local_sizes:
            error = NativeLibraryError(
                f"Partitioned dimension '{pdim}' is not present in this dataset."
            )
        comm.raise_if_error(error, "partitioned-dimension validation")

        lengths = comm.allgather_i64(local_sizes[pdim])
        offset = int(sum(lengths[: comm.rank]))
        global_sizes = dict(local_sizes)
        global_sizes[pdim] = int(sum(lengths))
        for name in sorted(local_sizes):
            if name != pdim:
                _agree(local_sizes[name], comm, f"the length of dimension '{name}'")

        # Character variables need a global maximum string width.
        string_widths: dict[str, int] = {}
        for name in sorted(strings):
            _, width, _ = strings[name]
            widths = comm.allgather_i64(width)
            string_widths[name] = max(widths)

        error = None
        try:
            for name, width in string_widths.items():
                raw, _, shape = strings[name]
                encvars[name].data = pack_chars(raw, width, shape)
                strlen_dim = f"{name}_strlen"
                if strlen_dim in global_sizes:
                    raise NativeLibraryError(
                        f"Generated string dimension '{strlen_dim}' conflicts "
                        + "with an existing dataset dimension."
                    )
                encvars[name].dims = encvars[name].dims + (strlen_dim,)
                global_sizes[strlen_dim] = width
        except BaseException as exc:
            error = exc
        comm.raise_if_error(error, "string packing")

        error = None
        try:
            unknown_chunks = sorted(set(chunk_map) - set(encvars))
            if unknown_chunks:
                raise NativeLibraryError(
                    "Chunk specifications reference non-existent variables: "
                    + f"{unknown_chunks}."
                )
            unknown_unlimited = sorted(unlimited - set(global_sizes))
            if unknown_unlimited:
                raise NativeLibraryError(
                    "Unlimited dimensions are absent from the dataset: "
                    + f"{unknown_unlimited}."
                )
            for name, chunkshape in chunk_map.items():
                variable = encvars[name]
                if len(chunkshape) != len(variable.dims):
                    raise NativeLibraryError(
                        f"Variable '{name}' has rank {len(variable.dims)}, but "
                        + f"its chunk shape has length {len(chunkshape)}."
                    )
                for dim, chunk_length in zip(variable.dims, chunkshape):
                    dim_length = global_sizes[dim]
                    if (
                        dim not in unlimited
                        and dim_length > 0
                        and chunk_length > dim_length
                    ):
                        raise NativeLibraryError(
                            f"Chunk length {chunk_length} for dimension '{dim}' "
                            + f"exceeds its fixed length {dim_length}."
                        )
        except BaseException as exc:
            error = exc
        comm.raise_if_error(error, "chunk and unlimited-dimension validation")

        for name in sorted(encvars):
            variable = encvars[name]
            if pdim not in variable.dims:
                if variable.data is None:
                    raise AssertionError(f"Variable '{name}' has no encoded buffer.")
                _agree(
                    _array_fingerprint(variable.data),
                    comm,
                    f"the replicated values of variable '{name}'",
                )

        handle = native.lib.mpi_netcdf_create(
            native.b(output_path),
            1 if nofill else 0,
            native.b(hints) if hints else None,
        )
        if not handle:
            raise NativeLibraryError(f"mpi_netcdf_create failed: {native.last_error()}")

        for name in sorted(global_sizes):
            native.check(
                native.lib.mpi_netcdf_def_dim(
                    handle,
                    native.b(name),
                    global_sizes[name],
                    1 if name in unlimited else 0,
                ),
                f"defining dimension '{name}'",
            )

        block = max(lengths)
        plan: dict[str, tuple[int | None, tuple[int, ...]]] = {}
        for name in sorted(encvars):
            variable = encvars[name]
            shape = tuple(global_sizes[dim] for dim in variable.dims)
            axis = variable.dims.index(pdim) if pdim in variable.dims else None
            use_deflate = (
                -1 if deflate_level is None or not variable.dims else deflate_level
            )
            if name in chunk_map:
                chunkshape = chunk_map[name]
            elif use_deflate >= 0:
                chunkshape = default_chunks(
                    shape,
                    axis,
                    block,
                    variable.itemsize,
                )
            else:
                chunkshape = None
            native.check(
                native.lib.mpi_netcdf_def_var(
                    handle,
                    native.b(name),
                    variable.xtype,
                    len(variable.dims),
                    native.str_array(variable.dims),
                    use_deflate,
                    1 if shuffle else 0,
                    native.size_array(chunkshape) if chunkshape else None,
                ),
                f"defining variable '{name}'",
            )
            put_attrs(handle, name, variable.attrs, variable.xtype)
            plan[name] = (axis, shape)

        put_attrs(handle, "", global_attrs, None)
        native.check(
            native.lib.mpi_netcdf_enddef(handle, native.COLLECTIVE),
            "leaving define mode",
        )

        for name in sorted(encvars):
            axis, _ = plan[name]
            if axis is None and not encvars[name].dims:
                native.check(
                    native.lib.mpi_netcdf_set_access(
                        handle,
                        native.b(name),
                        native.INDEPENDENT,
                    ),
                    f"setting independent access on '{name}'",
                )

        for name in sorted(encvars):
            variable = encvars[name]
            axis, shape = plan[name]
            if variable.data is None:
                raise AssertionError(f"Variable '{name}' has no encoded buffer.")
            start = [0] * len(shape)
            count = list(shape)
            if axis is not None:
                start[axis] = offset
                count[axis] = variable.data.shape[axis]
            elif comm.rank != 0:
                # Replicated arrays are written by rank zero. Other ranks
                # enter collective I/O with an empty selection. Scalars use
                # independent access and are written only by rank zero.
                if not shape:
                    continue
                count[0] = 0
            buffer = (
                variable.data.ctypes.data_as(ctypes.c_void_p)
                if variable.data.size
                else None
            )
            native.check(
                native.lib.mpi_netcdf_write(
                    handle,
                    native.b(name),
                    native.size_array(start),
                    native.size_array(count),
                    buffer,
                ),
                f"writing variable '{name}'",
            )

        close_status = native.lib.mpi_netcdf_close(handle)
        handle = None
        native.check(close_status, "closing the file")
        if size > 1:
            native.check(native.lib.mpi_netcdf_barrier(), "MPI barrier")
        return output_path

    except InconsistentRanksError:
        # Synchronized validation failures occur on every rank at the same
        # collective, so the caller may catch them without terminating MPI.
        if handle is not None:
            native.lib.mpi_netcdf_close(handle)
            handle = None
        raise

    except Exception:
        _LOGGER.exception("rank %d failed during parallel NetCDF output", rank)
        if size > 1:
            # An asymmetric Python or native error can leave other ranks in a
            # collective call. Aborting prevents an indefinite MPI job hang.
            native.abort(1)
        if handle is not None:
            native.lib.mpi_netcdf_close(handle)
            handle = None
        raise


def require_parallel(size: int, allow_serial: bool) -> None:
    if size > 1 or allow_serial:
        return

    msg = (
        "MPI_COMM_WORLD is initialized with a single rank. Launch execution using\n"
        + "`srun --ntasks=N --mpi=pmix python ...` or `mpirun -n N python ...`,\n"
        + "or pass allow_serial=True to allow a single-rank parallel-NetCDF write."
    )
    raise NativeLibraryError(msg)


def to_netcdf_parallel(
    data: xr.Dataset | xr.DataArray,
    path: str | PathLike[str],
    partition_dim: str | None = None,
    deflate: int | None = None,
    shuffle: bool = True,
    chunks: Mapping[str, Iterable[int]] | None = None,
    unlimited_dim: str | Iterable[str] = (),
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
    strict_compression: bool = False,
) -> str:
    """Write a distributed Dataset or DataArray to one NetCDF-4 file.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Slab owned by the current rank. A DataArray must be named.
    path : str or os.PathLike
        Output path visible to every rank.
    partition_dim : str or None, optional
        Partitioned dimension. The writer infers it when omitted.
    deflate : int or None, optional
        Deflate compression level from 0 to 9.
    shuffle : bool, default True
        Enable the HDF5 shuffle filter.
    chunks : mapping of str to iterable of int, optional
        Explicit chunk shape for selected variables.
    unlimited_dim : str or iterable of str, default ()
        Dimensions defined as unlimited record dimensions.
    hints : str or None, optional
        Semicolon-separated MPI-IO hints in key=value form.
    nofill : bool, default True
        Disable NetCDF pre-filling.
    allow_serial : bool, default False
        Permit execution with a one-rank MPI world.
    strict_compression : bool, default False
        Fail rather than warn when compression is unavailable in parallel.

    Returns
    -------
    str
        Output path after the collective write completes.

    Notes
    -----
    Time variables are deliberately not encoded here. The writer negotiates a
    single set of CF units across all ranks before applying xarray's encoders;
    encoding first would fix each rank's units from the values that rank
    happens to hold, and the ranks would then disagree about the schema.
    """
    if isinstance(data, xr.DataArray):
        if data.name is None:
            raise ValueError("DataArray must have a name for parallel output.")
        dataset = data.to_dataset()
    elif isinstance(data, xr.Dataset):
        dataset = data
    else:
        raise TypeError(
            "data must be an xarray.Dataset or xarray.DataArray, got "
            + f"{type(data).__name__}."
        )

    if isinstance(unlimited_dim, str):
        unlimited: tuple[str, ...] = (unlimited_dim,)
    elif unlimited_dim:
        unlimited = tuple(unlimited_dim)
    else:
        unlimited = ()

    return to_netcdf(
        dataset,
        path,
        partition_dim=partition_dim,
        deflate=deflate,
        shuffle=shuffle,
        chunks=chunks,
        unlimited_dim=unlimited,
        hints=hints,
        nofill=nofill,
        allow_serial=allow_serial,
        strict_compression=strict_compression,
    )


__all__ = [
    "InconsistentRanksError",
    "NativeLibraryError",
    "to_netcdf",
    "to_netcdf_parallel",
]
