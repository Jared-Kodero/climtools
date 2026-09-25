"""Provide xarray I/O and redistribution across MPI ranks."""

from __future__ import annotations

import atexit
import json
import math
import shutil
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from multiprocessing import shared_memory
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np

import xarray as xr

from ..mpi.context import MPIContext
from ..mpi.mpi_init import MPI
from ..mpp.ext_collectives import gather_v

if TYPE_CHECKING:
    from .core import MPIXarray

from ..mpp.mpp import mpp_sync
from ..mpp.mpp_domains_define import mpp_define_domains
from .chunks import (
    compute_save_chunks,
    get_chunk_bounds,
    get_chunk_info,
    get_effective_chunk_size,
)
from .distribute import (
    _as_partition_dims,
    mpp_create_dataarray,
    mpp_create_dataset,
    mpp_partition,
)
from .meta import (
    choose_partition_dim,
    mpp_get_meta,
    mpp_log_partition_report,
    mpp_should_log_partitions,
    mpp_update_meta,
    set_save_chunks,
)
from .netcdf import mpp_to_netcdf_parallel, nc_append, to_netcdf_serial

__all__ = [
    "empty_distributed_dataset",
    "is_distributed_empty",
    "nc_append",
    "to_netcdf",
]

_NO_DATA_ATTR = "_climtools_no_data"


def _open_partitioned(
    mpi_context: MPIContext,
    filename_or_obj: Any,
    dims: tuple[Hashable, ...],
    open_fn: Callable[..., xr.Dataset],
    chunks: Any,
    log_partitions: bool,
    automatic: bool,
    kwargs: dict[str, Any],
) -> xr.Dataset:
    """Open a Dataset lazily and keep only this rank's slice.

    Rank 0 reads the header alone and broadcasts the global shape, so the
    other ranks never touch the file until they open their own slice. As in
    FMS, one axis and several differ only in how the per-axis bounds are
    chosen: a single axis follows the file's chunk boundaries, a process grid
    divides each axis evenly.
    """
    comm = mpi_context.comm

    plan: dict[str, Any] | None = None
    error: BaseException | None = None
    if mpi_context.is_root():
        try:
            with open_fn(filename_or_obj, chunks=None, **kwargs) as metadata:
                resolved = dims
                if automatic:
                    resolved = (
                        choose_partition_dim(metadata.sizes, comm.size, rank=comm.rank),
                    )
                for d in resolved:
                    if d not in metadata.dims:
                        raise ValueError(f"Unknown partition dimension {d!r}.")
                plan = {
                    "dims": resolved,
                    "global_sizes": {d: int(metadata.sizes[d]) for d in resolved},
                    "chunk_info": get_chunk_info(metadata, comm.size),
                }
        except BaseException as exc:
            error = exc

    mpi_context.raise_if_error(error, "open_dataset planning")
    plan = mpi_context.broadcast(plan, root=0)
    dims = plan["dims"]
    global_sizes = plan["global_sizes"]
    chunk_info = plan["chunk_info"]

    cart = None
    if len(dims) == 1:
        dim = dims[0]
        bounds = {
            dim: get_chunk_bounds(
                global_sizes[dim], chunk_info[str(dim)], comm.rank, comm.size
            )
        }
    else:
        domain = mpp_define_domains(mpi_context, global_sizes, dims)
        bounds = {d: (domain.starts[d], domain.stops[d]) for d in dims}
        cart = domain.cart

    # Hold every rank here so none starts reading before the plan is settled.
    mpp_sync(comm)

    data: xr.Dataset = open_fn(filename_or_obj, chunks=chunks, **kwargs)
    data = data.isel({d: slice(*bounds[d]) for d in dims})

    if cart is not None:
        chunk_info = {
            str(name): get_effective_chunk_size(int(length), None, comm.size)
            for name, length in data.sizes.items()
        }

    starts = {d: bounds[d][0] for d in dims}
    stops = {d: bounds[d][1] for d in dims}
    single = len(dims) == 1 and cart is None
    mpp_update_meta(
        data,
        dim=dims[0] if single else dims,
        global_size=global_sizes[dims[0]] if single else global_sizes,
        start=starts[dims[0]] if single else starts,
        stop=stops[dims[0]] if single else stops,
        chunk_info=chunk_info,
        cart=cart,
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        mpp_log_partition_report(
            mpi_context,
            data,
            dims[0] if single else dims,
            origin="open_dataset",
            global_size=global_sizes[dims[0]] if single else global_sizes,
            start=starts[dims[0]] if single else starts,
            stop=stops[dims[0]] if single else stops,
            grid_shape=None if cart is None else cart["grid_shape"],
            coords=None if cart is None else cart["coords"],
            automatic=automatic,
        )
    return data


# mpi4py point-to-point tag for mpp_partition(); arbitrary but fixed so a
# stray message from unrelated code can never be mistaken for a piece
# this call is expecting.


def mpp_attach_save_chunks(
    mpi_context: MPIContext, value: xr.Dataset | xr.DataArray
) -> xr.Dataset | xr.DataArray:
    """Attach write-time chunk metadata to a distributed object.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed rank-local object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        ``value`` with ``mpi_meta["save_chunks"]`` attached.

    Raises
    ------
    ValueError
        If required partition chunk metadata are missing.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        return value

    save_chunks: dict[str, tuple[int, ...]] | None = None
    error: BaseException | None = None
    if mpi_context.is_root():
        try:
            save_chunks = compute_save_chunks(value, meta, mpi_context.comm.size)
        except BaseException as exc:
            error = exc
    mpi_context.raise_if_error(error, "attach_save_chunks planning")

    save_chunks = mpi_context.broadcast(save_chunks, root=0)
    set_save_chunks(value, cast("dict[str, tuple[int, ...]]", save_chunks))
    return value


def open_distributed_dataset(
    filename: Path | str | PathLike,
    mpi_context: MPIContext | MPI.Intracomm,
    *,
    partition_dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    chunks: Any = None,
    log_partitions: bool = True,
    **kwargs: Any,
) -> MPIXarray:
    """Open a Dataset lazily and partition it across MPI ranks.

    Parameters
    ----------
    filename : str, path-like, file-like, or list of these
        Input accepted by ``xarray.open_dataset``/``xarray.open_mfdataset``.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime whose communicator the result is bound to.
    partition_dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Dimension(s) to partition.
    chunks : int, dict, "auto" or None, optional
        Passed unchanged to xarray.
    log_partitions : bool, optional
        Print one aligned table showing which global interval each rank received.
    **kwargs : Any
        Additional arguments passed unchanged to ``xarray.open_dataset``/ ``xarray.open_mfdataset`` (e.g.

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.

    """

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    xr.set_options(keep_attrs=True)

    use_mfdataset = (isinstance(filename, str) and "*" in filename) or isinstance(
        filename, (list, tuple)
    )
    open_fn: Callable[..., xr.Dataset] = (
        xr.open_mfdataset if use_mfdataset else xr.open_dataset
    )

    requested_dims = _as_partition_dims(partition_dim)
    automatic = requested_dims == "auto"
    # "auto" leaves the axis for rank 0 to choose from the file header.
    dims: tuple[Hashable, ...] = (
        requested_dims
        if isinstance(requested_dims, tuple)
        else ()
        if automatic
        else (requested_dims,)
    )
    data = _open_partitioned(
        mpi_context,
        filename,
        dims,
        open_fn,
        chunks,
        log_partitions,
        automatic,
        kwargs,
    )

    from .core import MPIXarray

    return MPIXarray(data, mpi_context)


def create_distributed_dataarray(
    mpi_context: MPIContext | MPI.Intracomm,
    fill: Callable[..., Any],
    dims: Sequence[Hashable],
    *,
    shape: Sequence[int] | Mapping[Hashable, int] | None = None,
    dim: Hashable | int | Sequence[Hashable] = 0,
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    name: Hashable | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = False,
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> MPIXarray:
    """Create an :class:`MPIXarray` DataArray from a fill function.

    Parameters
    ----------
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        MPI context or communicator.
    fill : callable
        Function producing rank-local values.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes.
    dim : Hashable, int, or sequence of Hashable
        Partition dimension or dimensions.
    dtype : Any, optional
        Fill-function output dtype.
    coords : mapping, optional
        DataArray coordinates.
    name : Hashable, optional
        DataArray name.
    attrs : mapping, optional
        DataArray attributes.
    log_partitions : bool, optional
        Log the rank layout.
    min_partition_size : int or mapping, optional
        Minimum non-empty local extent per partition dimension.

    Returns
    -------
    MPIXarray
        Distributed DataArray wrapper.
    """
    from .core import MPIXarray

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_create_dataarray(
        mpi_context,
        fill,
        dims,
        shape=shape,
        dim=dim,
        dtype=dtype,
        coords=coords,
        name=name,
        attrs=attrs,
        log_partitions=log_partitions,
        min_partition_size=min_partition_size,
    )
    return MPIXarray(data, mpi_context)


def create_distributed_dataset(
    mpi_context: MPIContext | MPI.Intracomm,
    data_vars: Mapping[
        Hashable, xr.DataArray | tuple[Sequence[Hashable], Callable[..., Any]]
    ],
    sizes: Mapping[Hashable, int] | None = None,
    *,
    dim: Hashable | Sequence[Hashable],
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = True,
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> MPIXarray:
    """Create an :class:`MPIXarray` Dataset from rank-local variables.

    Parameters
    ----------
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        MPI context or communicator.
    data_vars : mapping
        DataArrays or ``(dims, fill)`` variable specifications.
    sizes : mapping, optional
        Global dimension sizes.
    dim : Hashable or sequence of Hashable
        Partition dimension or dimensions.
    dtype : Any or mapping, optional
        Default or per-variable fill dtype.
    coords, attrs : mapping, optional
        Dataset coordinates and attributes.
    log_partitions : bool, optional
        Log the rank layout.
    min_partition_size : int or mapping, optional
        Minimum non-empty local extent per partition dimension.

    Returns
    -------
    MPIXarray
        Distributed Dataset wrapper.
    """
    from .core import MPIXarray

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_create_dataset(
        mpi_context,
        data_vars,
        sizes,
        dim=dim,
        dtype=dtype,
        coords=coords,
        attrs=attrs,
        log_partitions=log_partitions,
        min_partition_size=min_partition_size,
    )
    return MPIXarray(data, mpi_context)


def distribute_data(
    value: MPIXarray | xr.Dataset | xr.DataArray | None,
    mpi_context: MPIContext | MPI.Intracomm,
    dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    *,
    root: int = 0,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> MPIXarray:
    """Partition a root-owned xarray object across MPI ranks.

    Parameters
    ----------
    value : MPIXarray, xarray.Dataset, xarray.DataArray, or None
        Complete object on ``root``; non-root ranks must pass None.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime or communicator the result is bound to.
    dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Partition dimension(s).
    root : int, optional
        Rank that owns ``value``.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Rank-local slice with ``.meta`` set.

    """
    from .core import MPIXarray, unwrap

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_partition(
        mpi_context,
        unwrap(value),
        dim,
        root=root,
        chunk_info=chunk_info,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, mpi_context)


def empty_distributed_dataset() -> xr.Dataset:
    """Return a placeholder Dataset for a non-root MPI rank.

    Returns
    -------
    xarray.Dataset
        Dataset marked as containing no rank-local data.

    """
    return xr.Dataset(attrs={_NO_DATA_ATTR: True})


def is_distributed_empty(data: xr.Dataset | xr.DataArray) -> bool:
    """Return whether an object is a non-root MPI placeholder.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to inspect.

    Returns
    -------
    bool
        True when ``data`` is an MPI placeholder Dataset.

    """
    return isinstance(data, xr.Dataset) and data.attrs.get(_NO_DATA_ATTR) is True


def to_netcdf(
    data: xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    mpi_context: MPIContext | MPI.Intracomm | None = None,
    unlimited_dim: str | Iterable[str] | None = None,
    partition_dim: str | None = None,
    *,
    parallel: bool = False,
    batch_size: int = 24,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
    chunks: Mapping[str, Iterable[int]] | None = None,
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> None:
    """Write an xarray object to NetCDF.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to write.
    file : str or os.PathLike
        Output path.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm, optional
        MPI context or communicator.
    unlimited_dim : str or iterable of str, optional
        Unlimited dimension names.
    partition_dim : str, optional
        MPI partition dimension.
    parallel : bool, default False
        Use MPI-parallel NetCDF-4 output.
    batch_size : int, default 24
        Slices written per serial append.
    format : str, default "NETCDF4"
        NetCDF format for serial output.
    shuffle, zlib : bool, default True
        HDF5 filters.
    complevel : int, default 4
        Compression level.
    show_progress : bool, default True
        Display serial write progress.
    stdout : Any, optional
        Progress output stream.
    chunks : mapping, optional
        Explicit NetCDF chunk shapes.
    hints : str, optional
        Semicolon-separated MPI-IO ``key=value`` hints.
    nofill : bool, default True
        Disable NetCDF pre-filling in parallel mode.
    allow_serial : bool, default False
        Permit the parallel writer with one MPI rank.
    """

    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError("data must be an xarray.Dataset or xarray.DataArray")

    target_path = Path(file)

    if parallel:
        if not mpi_context:
            from ..mpi.context import get_mpi_ctx

            mpi_context = get_mpi_ctx()
        if not isinstance(mpi_context, MPIContext):
            mpi_context = MPIContext(mpi_context)

        mpi_meta = mpp_get_meta(data)
        distributed = mpi_meta is not None

        # Ranks must agree on the write path. If one rank saw valid mpi_meta
        # and another did not, the two paths post different collectives and
        # the writer would block instead of reporting the inconsistency.
        agreed = gather_v(distributed, mpi_context.comm)
        if any(agreed) and not all(agreed):
            disagreeing = [
                rank for rank, state in enumerate(agreed) if state != agreed[0]
            ]
            raise mpi_context.MPIError(
                f"MPI ranks disagree on distribution state: {disagreeing}."
            )

        if distributed:
            distributed_dim = str(mpi_meta["dim"])
            if partition_dim is not None and partition_dim != distributed_dim:
                raise ValueError(
                    f"partition_dim={partition_dim!r} differs from distributed "
                    + f"dim={distributed_dim!r}."
                )
            partition_dim = distributed_dim
        elif mpi_context.comm.rank != 0:
            data = empty_distributed_dataset()

        mpp_to_netcdf_parallel(
            mpi_context,
            data,
            target_path,
            partition_dim=partition_dim,
            deflate=complevel if zlib else None,
            shuffle=shuffle,
            chunks=chunks,
            unlimited_dim=unlimited_dim if unlimited_dim is not None else (),
            hints=hints,
            nofill=nofill,
            allow_serial=allow_serial,
        )
        return

    to_netcdf_serial(
        data=data,
        file=target_path,
        unlimited_dim=unlimited_dim,
        batch_size=batch_size,
        format=format,
        shuffle=shuffle,
        zlib=zlib,
        complevel=complevel,
        show_progress=show_progress,
        stdout=stdout,
    )


def to_xnpy(
    obj: np.ndarray | xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a NumPy or xarray object to a memory-mappable XNpy store.

    Parameters
    ----------
    obj : numpy.ndarray or xarray.DataArray or xarray.Dataset
        Object to store.
    path : str or pathlib.Path
        Store path. The ``.xnpy`` suffix is appended if absent.
    overwrite : bool, default False
        Remove an existing store before writing when True.

    Returns
    -------
    pathlib.Path
        Path to the completed store.
    """
    return XNpyStore(path).save(obj, overwrite=overwrite)


def open_xnpy(
    path: str | Path,
    variable: str | None = None,
    *,
    mmap_mode: str | None = "r",
) -> np.ndarray | xr.DataArray | xr.Dataset:
    """Open a NumPy or xarray object from a memory-mappable XNpy store.

    Parameters
    ----------
    path : str or pathlib.Path
        Store path. The ``.xnpy`` suffix is appended if absent.
    variable : str, optional
        Dataset variable to reconstruct. None reconstructs the complete stored
        object. For a DataArray, the stored variable name may also be supplied.
    mmap_mode : {"r", "r+", "w+", "c"} or None, default "r"
        Memory-map mode passed to :func:`numpy.load`. None loads payloads into
        ordinary in-memory arrays.

    Returns
    -------
    numpy.ndarray or xarray.DataArray or xarray.Dataset
        Reconstructed object.
    """
    return XNpyStore(path).load(variable=variable, mmap_mode=mmap_mode)


class XNpyStore:
    """Internal directory store for memory-mappable NumPy and xarray objects.

    Numerical payloads are stored as uncompressed ``.npy`` files and metadata
    are stored in a versioned JSON manifest. No pickle data are written.
    """

    SUFFIX = ".xnpy"
    META_FILE = "metadata.json"
    COORD_DIR = "_coords"
    FORMAT = "xnpy"
    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != self.SUFFIX:
            path = Path(f"{path}{self.SUFFIX}")
        self.path = path
        self.meta_path = self.path / self.META_FILE

    def save(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
        *,
        overwrite: bool = False,
    ) -> None:
        """Save an array, DataArray, or Dataset."""
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)

        if isinstance(obj, xr.Dataset):
            metadata = self._save_dataset(obj)
        elif isinstance(obj, xr.DataArray):
            metadata = self._save_dataarray(obj)
        elif isinstance(obj, np.ndarray):
            metadata = self._save_ndarray(obj)
        else:
            raise TypeError(
                "XNpy supports numpy.ndarray, xarray.DataArray, and "
                "xarray.Dataset only."
            )

        # Written last: its presence marks the store complete.
        with self.meta_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"format": self.FORMAT, "version": self.VERSION, **metadata},
                file,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )

    def metadata(self) -> dict[str, Any]:
        """Read and validate the JSON manifest without opening payloads."""
        with self.meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("format") != self.FORMAT:
            raise ValueError(f"Not an {self.FORMAT!r} store: {self.path}.")
        if metadata.get("version") != self.VERSION:
            raise ValueError(
                f"Unsupported {self.FORMAT} version: {metadata.get('version')!r}."
            )
        return metadata

    def load(
        self,
        variable: str | None = None,
        *,
        mmap_mode: str | None = "r",
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Load a stored object or one Dataset variable."""
        metadata = self.metadata()
        kind = metadata["kind"]

        if kind == "dataset":
            if variable is None:
                return self._load_dataset(metadata, mmap_mode)
            return self._load_dataset_variable(metadata, variable, mmap_mode)

        if kind == "dataarray":
            if variable is not None and variable != metadata["variable_name"]:
                raise KeyError(variable)
            return self._load_dataarray(metadata, mmap_mode)

        if kind == "ndarray":
            if variable is not None:
                raise ValueError("variable= is valid only for xarray stores.")
            return self._load_array(metadata["array"], mmap_mode)

        raise ValueError(f"Unknown store kind: {kind!r}")

    def _save_dataset(self, dataset: xr.Dataset) -> dict[str, Any]:
        """Save every data variable and coordinate of a Dataset."""
        variables = {}
        for name, variable in dataset.data_vars.items():
            info = self._save_xarray_variable(
                variable,
                self._make_dir(self.path, name),
            )
            info["coords"] = list(dataset[name].coords)
            variables[name] = info

        return {
            "kind": "dataset",
            "attrs": self._json_value(dict(dataset.attrs)),
            "variables": variables,
            "coords": self._save_coords(dataset),
        }

    def _save_dataarray(self, array: xr.DataArray) -> dict[str, Any]:
        """Save a DataArray and its coordinates."""
        name = str(array.name) if array.name is not None else "data"
        variable = self._save_xarray_variable(
            array,
            self._make_dir(self.path, name),
        )
        variable["coords"] = list(array.coords)
        return {
            "kind": "dataarray",
            "name": array.name,
            "variable_name": name,
            "variable": variable,
            "coords": self._save_coords(array),
        }

    def _save_coords(self, obj: xr.Dataset | xr.DataArray) -> dict[str, Any]:
        """Save every coordinate under the shared coordinate directory."""
        root = self.path / self.COORD_DIR
        root.mkdir()
        return {
            name: self._save_xarray_variable(coord, self._make_dir(root, name))
            for name, coord in obj.coords.items()
        }

    def _save_ndarray(self, array: np.ndarray) -> dict[str, Any]:
        """Save a bare NumPy array."""
        directory = self.path / "array"
        directory.mkdir()
        return {"kind": "ndarray", "array": self._save_array(directory, array)}

    def _save_xarray_variable(
        self,
        variable: xr.Variable | xr.DataArray,
        directory: Path,
    ) -> dict[str, Any]:
        """Save one variable payload and reconstruction metadata."""
        return {
            "dims": list(variable.dims),
            "attrs": self._json_value(dict(variable.attrs)),
            "array": self._save_array(directory, np.asarray(variable.data)),
        }

    def _save_array(self, directory: Path, array: np.ndarray) -> dict[str, Any]:
        """Write one payload as ``.npy``."""
        if array.dtype.hasobject:
            raise TypeError(
                "object-dtype arrays are not supported because XNpy does not "
                "use pickle."
            )
        path = directory / "data.npy"
        np.save(path, array, allow_pickle=False)
        return {
            "file": str(path.relative_to(self.path)),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    def _load_dataset(
        self,
        metadata: dict[str, Any],
        mmap_mode: str | None,
    ) -> xr.Dataset:
        """Reconstruct a complete Dataset."""
        return xr.Dataset(
            data_vars={
                name: self._as_variable(info, mmap_mode)
                for name, info in metadata["variables"].items()
            },
            coords=self._load_coords(
                metadata["coords"],
                metadata["coords"],
                mmap_mode,
            ),
            attrs=metadata["attrs"],
        )

    def _load_dataset_variable(
        self,
        metadata: dict[str, Any],
        name: str,
        mmap_mode: str | None,
    ) -> xr.DataArray:
        """Reconstruct one Dataset variable and the coordinates it uses."""
        variables = metadata["variables"]
        if name not in variables:
            raise KeyError(
                f"{name!r} not found; available variables: {tuple(variables)!r}"
            )
        return self._build_dataarray(
            variables[name],
            metadata["coords"],
            name,
            mmap_mode,
        )

    def _load_dataarray(
        self,
        metadata: dict[str, Any],
        mmap_mode: str | None,
    ) -> xr.DataArray:
        """Reconstruct a stored DataArray."""
        return self._build_dataarray(
            metadata["variable"],
            metadata["coords"],
            metadata["name"],
            mmap_mode,
        )

    def _build_dataarray(
        self,
        info: dict[str, Any],
        coord_metadata: dict[str, Any],
        name: Any,
        mmap_mode: str | None,
    ) -> xr.DataArray:
        """Assemble a DataArray from its payload and recorded coordinates."""
        return xr.DataArray(
            self._load_array(info["array"], mmap_mode),
            dims=info["dims"],
            coords=self._load_coords(coord_metadata, info["coords"], mmap_mode),
            name=name,
            attrs=info["attrs"],
        )

    def _load_coords(
        self,
        metadata: dict[str, Any],
        names: Any,
        mmap_mode: str | None,
    ) -> dict[str, tuple[Any, ...]]:
        """Load the named coordinates."""
        return {name: self._as_variable(metadata[name], mmap_mode) for name in names}

    def _as_variable(
        self,
        info: dict[str, Any],
        mmap_mode: str | None,
    ) -> tuple[Any, ...]:
        """Return the ``(dims, data, attrs)`` triple xarray builds from."""
        return (
            info["dims"],
            self._load_array(info["array"], mmap_mode),
            info["attrs"],
        )

    def _load_array(
        self,
        info: dict[str, Any],
        mmap_mode: str | None,
    ) -> np.ndarray:
        """Load a payload and validate its recorded dtype and shape."""
        path = self.path / info["file"]
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

        expected_dtype = np.dtype(info["dtype"])
        expected_shape = tuple(info["shape"])
        if array.dtype != expected_dtype:
            raise TypeError(f"{path}: dtype {array.dtype} != {expected_dtype}")
        if array.shape != expected_shape:
            raise ValueError(f"{path}: shape {array.shape} != {expected_shape}")
        return array

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        """Convert metadata to JSON-safe values without lossy coercion."""
        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TypeError("Non-finite float metadata are not supported.")
            return value
        if isinstance(value, np.generic):
            return cls._json_value(value.item())
        if isinstance(value, np.ndarray):
            return cls._json_value(value.tolist())
        if isinstance(value, list | tuple):
            return [cls._json_value(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("JSON metadata dictionaries require string keys.")
            return {key: cls._json_value(item) for key, item in value.items()}
        raise TypeError(f"Unsupported JSON metadata type: {type(value).__name__}.")

    @classmethod
    def _make_dir(cls, parent: Path, name: Any) -> Path:
        """Create one payload directory, rejecting reserved or unsafe names."""
        value = str(name)
        if not value or value in {cls.COORD_DIR, cls.META_FILE}:
            raise ValueError(f"Reserved or empty array name: {value!r}.")
        if "/" in value or "\\" in value:
            raise ValueError(f"Array names cannot contain path separators: {value!r}.")
        directory = parent / value
        directory.mkdir()
        return directory


class SharedMemoryObject:
    """Shared-memory transport for NumPy and xarray objects.

    Supported
    ---------
    - numpy.ndarray
    - xarray.DataArray
    - xarray.Dataset

    When this object is pickled, for example by multiprocessing.Queue,
    Pool, Process, or ProcessPoolExecutor, the receiving process gets the
    original ndarray/DataArray/Dataset rather than SharedMemoryObject.

    Pointer-free NumPy buffers are placed in shared memory. Object-dtype
    arrays fall back to ordinary pickle transport because their memory
    contains process-local Python pointers.

    Parameters
    ----------
    obj
        Object to place into shared memory.
    readonly
        If True, reconstructed NumPy buffers are marked non-writeable.

    Notes
    -----
    The creating process owns the shared-memory segments. It must keep this
    object alive until all workers have finished using the reconstructed
    objects.

    Use as a context manager whenever possible.
    """

    _attachments: ClassVar[dict[str, shared_memory.SharedMemory]] = {}

    _atexit_registered: ClassVar[bool] = False

    def __init__(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
        *,
        readonly: bool = True,
    ) -> None:
        self._owners: list[shared_memory.SharedMemory] = []
        self._closed = False
        self._readonly = readonly

        self._ensure_atexit()

        try:
            self._spec = self._encode_object(obj)
        except Exception:
            self.close()
            raise

    @classmethod
    def _ensure_atexit(cls) -> None:
        if cls._atexit_registered:
            return

        atexit.register(cls.close_attachments)
        cls._atexit_registered = True

    @staticmethod
    def _open_attachment(
        name: str,
    ) -> shared_memory.SharedMemory:
        # Python >= 3.13 supports track=False. This prevents receiver
        # processes from independently trying to unlink memory owned by
        # the creating process.
        try:
            return shared_memory.SharedMemory(
                name=name,
                track=False,
            )
        except TypeError:
            # Python <= 3.12
            return shared_memory.SharedMemory(name=name)

    @classmethod
    def _attach(
        cls,
        name: str,
    ) -> shared_memory.SharedMemory:
        cls._ensure_atexit()

        shm = cls._attachments.get(name)

        if shm is None:
            shm = cls._open_attachment(name)
            cls._attachments[name] = shm

        return shm

    def _encode_array(
        self,
        array: np.ndarray,
    ) -> dict[str, Any]:
        array = np.asarray(array)

        order: Literal["C", "F"]

        if array.flags.f_contiguous and not array.flags.c_contiguous:
            order = "F"
        else:
            order = "C"

        # Raw shared memory cannot safely represent object-containing
        # dtypes because the buffer contains CPython pointers.
        if array.dtype.hasobject:
            return {
                "shape": array.shape,
                "dtype": array.dtype,
                "order": order,
                "shm_name": None,
                "inline": np.array(
                    array,
                    copy=True,
                    order=order,
                ),
            }

        contiguous = np.array(
            array,
            copy=True,
            order=order,
        )

        shm = shared_memory.SharedMemory(
            create=True,
            size=max(contiguous.nbytes, 1),
        )

        self._owners.append(shm)

        target = np.ndarray(
            contiguous.shape,
            dtype=contiguous.dtype,
            buffer=shm.buf,
            order=order,
        )

        target[...] = contiguous

        return {
            "shape": contiguous.shape,
            "dtype": contiguous.dtype,
            "order": order,
            "shm_name": shm.name,
            "inline": None,
        }

    def _encode_variable(
        self,
        name: Any,
        role: Literal["data", "coord"],
        variable: xr.Variable,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "role": role,
            "dims": tuple(variable.dims),
            "attrs": dict(variable.attrs),
            "encoding": dict(variable.encoding),
            "array": self._encode_array(np.asarray(variable.data)),
        }

    def _encode_object(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
    ) -> dict[str, Any]:
        if isinstance(obj, xr.Dataset):
            variables = [
                self._encode_variable(name, "data", variable)
                for name, variable in obj.data_vars.items()
            ]

            variables.extend(
                self._encode_variable(
                    name,
                    "coord",
                    variable,
                )
                for name, variable in obj.coords.items()
            )

            return {
                "kind": "dataset",
                "attrs": dict(obj.attrs),
                "encoding": dict(obj.encoding),
                "variables": variables,
                "readonly": self._readonly,
            }

        if isinstance(obj, xr.DataArray):
            variables = [self._encode_variable(None, "data", obj.variable)]

            variables.extend(
                self._encode_variable(
                    name,
                    "coord",
                    variable,
                )
                for name, variable in obj.coords.items()
            )

            return {
                "kind": "dataarray",
                "name": obj.name,
                "variables": variables,
                "readonly": self._readonly,
            }

        if isinstance(obj, np.ndarray):
            return {
                "kind": "ndarray",
                "array": self._encode_array(obj),
                "readonly": self._readonly,
            }

        raise TypeError(
            f"Expected np.ndarray, xr.DataArray, or xr.Dataset; got {type(obj)!r}."
        )

    @classmethod
    def _decode_array(cls, spec: dict[str, Any], *, readonly: bool) -> np.ndarray:
        shm_name = spec["shm_name"]

        if shm_name is None:
            array = spec["inline"]
        else:
            shm = cls._attach(shm_name)

            array = np.ndarray(
                spec["shape"], dtype=spec["dtype"], buffer=shm.buf, order=spec["order"]
            )

        if readonly:
            array.flags.writeable = False

        return array

    @classmethod
    def _decode_variable(cls, spec: dict[str, Any], *, readonly: bool) -> xr.Variable:
        variable = xr.Variable(
            spec["dims"],
            cls._decode_array(
                spec["array"],
                readonly=readonly,
            ),
            attrs=dict(spec["attrs"]),
        )

        variable.encoding.update(spec["encoding"])

        return variable

    @classmethod
    def _rebuild(
        cls,
        spec: dict[str, Any],
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        readonly = spec["readonly"]
        kind = spec["kind"]

        if kind == "ndarray":
            return cls._decode_array(
                spec["array"],
                readonly=readonly,
            )

        decoded = [
            (
                variable_spec,
                cls._decode_variable(variable_spec, readonly=readonly),
            )
            for variable_spec in spec["variables"]
        ]

        if kind == "dataset":
            data_vars = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "data"
            }

            coords = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "coord"
            }

            dataset = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs=dict(spec["attrs"]),
            )

            dataset.encoding.update(spec["encoding"])

            return dataset

        if kind == "dataarray":
            data_variable = next(
                variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "data"
            )

            coords = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "coord"
            }

            return xr.DataArray(data_variable, coords=coords, name=spec["name"])

        raise RuntimeError(f"Unknown object kind: {kind!r}")

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Control multiprocessing/pickle reconstruction."""
        if self._closed:
            raise RuntimeError("Cannot serialize a closed SharedMemoryObject.")

        return self._rebuild, (self._spec,)

    def get(self) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Return a local view of the shared object."""
        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self._rebuild(self._spec)

    @property
    def nbytes(self) -> int:
        """Total allocated shared-memory bytes."""
        return sum(shm.size for shm in self._owners)

    @property
    def closed(self) -> bool:
        """Whether the owner has been closed."""
        return self._closed

    def close(self) -> None:
        """Unlink and close owned shared-memory segments."""
        if self._closed:
            return

        for shm in self._owners:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

            shm.close()

        self._owners.clear()
        self._closed = True

    @classmethod
    def close_attachments(cls) -> None:
        """Close receiver-side shared-memory handles."""
        for name, shm in list(cls._attachments.items()):
            try:
                shm.close()
            except (BufferError, OSError):
                # A live NumPy array may still export the
                # underlying mmap. The OS will clean it up
                # when this process terminates.
                continue

            del cls._attachments[name]

    def __enter__(self):
        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self

    def __exit__(self, *_: object) -> None:
        self.close()
