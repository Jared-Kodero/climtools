"""Distribution and I/O for MPI-backed xarray objects.

Opening, distributing, repartitioning, and creating datasets/data arrays
across ranks: everything that establishes or changes ``mpi_meta`` without
performing a numerical reduction.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

import xarray as xr

from .chunks import (
    compute_save_chunks,
    get_balanced_bounds,
    get_chunk_bounds,
    get_chunk_info,
    get_effective_chunk_size,
    prune_chunk_info,
)
from .meta import (
    _delayed_local,
    _localize_coord,
    _resolve_sizes,
    choose_partition_dim,
    get_mpi_meta,
    log_partition_report,
    set_mpi_meta,
    set_save_chunks,
    should_log_partitions,
    strip_mpi_meta,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..mpi.runtime import MPIRuntime


class IO:
    """Dataset/DataArray I/O and (re)distribution across MPI ranks.

    Requires a ``self._runtime`` attribute set by :class:`~.mpi.XarrayMPI`.
    """

    _runtime: MPIRuntime

    def open_xr_dataset(
        self,
        filename_or_obj: Any,
        *,
        partition_dim: Hashable | Literal["auto"] = "auto",
        chunks: Any = None,
        log_partitions: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open a Dataset lazily and distribute one dimension across ranks.

        Parameters
        ----------
        filename_or_obj : str, path-like, file-like, or list of these
            Input accepted by :func:`xarray.open_dataset` or :func:`xarray.open_mfdataset`.
            Strings containing a wildcard ("*") or sequences (e.g., list, tuple) will
            automatically trigger multi-file loading.
        partition_dim : Hashable or {"auto"}, optional
            Dimension to distribute. ``"auto"`` selects the longest dimension,
            which is the choice that leaves the fewest ranks idle. Selection is
            deterministic and identical on every rank. Default is "auto".
        chunks : int, dict, "auto" or None, optional
            Passed unchanged to xarray. ``None`` keeps single-file reads
            backend-lazy without Dask; explicit chunking enables Dask
            according to xarray semantics.
        log_partitions : bool, optional
            Print one aligned table showing which global interval each rank
            received. Default is True.
        engine : str, optional
            Engine to use for reading files. Options include 'netcdf4', 'h5netcdf',
            'scipy', 'cfgrib', 'zarr', etc. Passed via ``**kwargs``.
        concat_dim : str, DataArray, Index or list thereof, optional
            (Multi-file only) Dimension(s) over which to concatenate datasets. Passed
            via ``**kwargs``.
        combine : {"by_coords", "nested"}, optional
            (Multi-file only) Whether to combine datasets by matching coordinates or
            by their nested structure. Passed via ``**kwargs``.
        preprocess : callable, optional
            (Multi-file only) If provided, call this function on each dataset prior to
            concatenation. Passed via ``**kwargs``.
        parallel : bool, optional
            (Multi-file only) If True, the open and preprocess steps will be performed
            in parallel using ``dask.delayed``. Passed via ``**kwargs``.
        decode_cf : bool, optional
            Whether to decode these variables, assuming they were saved according to
            CF conventions (e.g., ``mask_and_scale``, ``decode_times``). Passed via ``**kwargs``.
        **kwargs : Any
            Any additional standard arguments passed unchanged to
            :func:`xarray.open_dataset` or :func:`xarray.open_mfdataset` (e.g.,
            ``decode_times``, ``drop_variables``, ``compat``, ``data_vars``).

        Returns
        -------
        xarray.Dataset
            Lazy rank-local Dataset carrying ``mpi_meta``.
        """

        xr.set_options(keep_attrs=True)

        use_mfdataset = (
            isinstance(filename_or_obj, str) and "*" in filename_or_obj
        ) or isinstance(filename_or_obj, (list, tuple))

        open_dataset: Callable = xr.open_mfdataset if use_mfdataset else xr.open_dataset

        automatic = partition_dim == "auto"

        # Build the metadata plan on rank 0.
        plan: dict[str, Any] | None = None
        error: BaseException | None = None
        if self._runtime.is_root():
            try:
                with open_dataset(filename_or_obj, chunks=None, **kwargs) as metadata:
                    if automatic:
                        partition_dim = choose_partition_dim(
                            metadata.sizes,
                            self._runtime.comm.size,
                            rank=self._runtime.comm.rank,
                        )
                    if partition_dim not in metadata.dims:
                        raise ValueError(
                            f"partition_dim {partition_dim!r} is not in "
                            + f"{list(metadata.dims)!r}."
                        )
                    chunk_info = get_chunk_info(metadata, self._runtime.comm.size)
                    global_size = int(metadata.sizes[partition_dim])
                    longest_size = max(
                        int(length) for length in metadata.sizes.values()
                    )

                    if not automatic and global_size < longest_size:
                        longest_dims = [
                            str(dim)
                            for dim, length in metadata.sizes.items()
                            if int(length) == longest_size
                        ]
                        warnings.warn(
                            f"partition_dim {partition_dim!r} has length "
                            + f"{global_size}, but it should be a longest "
                            + "dataset dimension. Longest dimension(s) "
                            + f"{longest_dims!r} have length {longest_size}.",
                            UserWarning,
                            stacklevel=2,
                        )

                    # Pack the plan into a dictionary for broadcasting
                    plan = {
                        "partition_dim": partition_dim,
                        "chunk_info": chunk_info,
                        "global_size": global_size,
                    }
            except BaseException as exc:
                error = exc

        # Synchronize rank-0 planning failures before broadcasting the plan.
        self._runtime.raise_if_error(error, "mpi.xarray.open_dataset planning")

        # Broadcast the plan.
        plan = self._runtime.broadcast(plan, root=0)

        partition_dim = plan["partition_dim"]
        chunk_info = plan["chunk_info"]
        global_size = plan["global_size"]

        # Compute this rank's bounds.
        partition_chunk = chunk_info[str(partition_dim)]
        start, stop = get_chunk_bounds(
            global_size,
            partition_chunk,
            self._runtime.comm.rank,
            self._runtime.comm.size,
        )

        # Synchronize before opening the dataset.
        self._runtime.comm.Barrier()

        # Open this rank's lazy slice.
        data: xr.Dataset = open_dataset(filename_or_obj, chunks=chunks, **kwargs)
        data = data.isel({partition_dim: slice(start, stop)})

        set_mpi_meta(
            data,
            dim=partition_dim,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info=chunk_info,
        )
        if should_log_partitions(self._runtime, log_partitions):
            log_partition_report(
                self._runtime,
                data,
                partition_dim,
                origin="mpi.xarray.open_dataset",
                global_size=global_size,
                start=start,
                stop=stop,
                automatic=automatic,
            )
        return data

    # mpi4py point-to-point tag for distribute(); arbitrary but fixed so a
    # stray message from unrelated code can never be mistaken for a piece
    # this call is expecting.
    _DISTRIBUTE_TAG = 0x6469_7374  # b"dist" as an int, easy to spot in a trace

    def distribute(
        self,
        value: xr.Dataset | xr.DataArray | None,
        dim: Hashable | Literal["auto"] = "auto",
        *,
        root: int = 0,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """Distribute a root-owned xarray object across MPI ranks.

        The root slices the object along ``dim`` and sends each rank only its local
        piece. Use :meth:`repartition` when the full object already exists on every
        rank.

        Parameters
        ----------
        value : xarray.Dataset, xarray.DataArray, or None
            Complete object on ``root``; non-root ranks must pass None.
        dim : Hashable or {"auto"}, optional
            Partition dimension. ``"auto"`` selects the largest dimension.
            Default is ``"auto"``.
        root : int, optional
            Rank that owns ``value``. Default is 0.
        chunk_info : mapping of str to int, optional
            Effective chunk-size hints.
        log_partitions : bool, optional
            Log the resulting rank layout. Default is False.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Rank-local slice carrying ``mpi_meta``.

        Raises
        ------
        ValueError
            If ownership, metadata, or ``dim`` is invalid.

        Notes
        -----
        Dask-backed inputs remain lazy: the root sends sliced task graphs rather
        than materializing the full array."""
        comm = self._runtime.comm
        is_root = self._runtime.is_root(root)

        # Prepare every slice before communication so a root-side failure is
        # synchronized before any rank can block in send/receive.
        error: BaseException | None = None
        pieces: list[Any] | None = None
        replicated_value: xr.Dataset | xr.DataArray | None = None
        try:
            if is_root:
                if value is None:
                    raise ValueError(
                        f"Rank {root} (root) must provide a value, not None."
                    )
                if get_mpi_meta(value) is not None:
                    raise ValueError(
                        "Cannot distribute an already distributed object. "
                        + "Reduce or gather its distributed dimension first."
                    )
                stripped = strip_mpi_meta(value)

                if not stripped.dims:
                    # Nothing to partition: send the (necessarily small)
                    # whole object to every rank as replicated data,
                    # mirroring repartition's handling of the same case.
                    replicated_value = stripped
                else:
                    automatic = dim == "auto"
                    resolved_dim = (
                        choose_partition_dim(stripped.sizes, comm.size, rank=comm.rank)
                        if automatic
                        else dim
                    )
                    if resolved_dim not in stripped.dims:
                        raise ValueError(
                            f"Distribution dimension {resolved_dim!r} does not exist."
                        )

                    length = int(stripped.sizes[resolved_dim])
                    info = dict(chunk_info or {})
                    chunk_size = int(
                        info.get(
                            str(resolved_dim),
                            get_effective_chunk_size(length, None, comm.size),
                        )
                    )
                    chunk_size = get_effective_chunk_size(length, chunk_size, comm.size)
                    info[str(resolved_dim)] = chunk_size

                    pieces = []
                    for rank in range(comm.size):
                        start, stop = get_chunk_bounds(
                            length, chunk_size, rank, comm.size
                        )
                        piece = stripped.isel({resolved_dim: slice(start, stop)})
                        # Break shallow-copy attribute sharing before adding
                        # rank metadata.
                        piece.attrs = dict(piece.attrs)
                        if isinstance(piece, xr.Dataset):
                            for variable in piece.variables.values():
                                variable.attrs = dict(variable.attrs)
                        piece_info = prune_chunk_info(info, piece)
                        for other_dim, other_length in piece.sizes.items():
                            piece_info.setdefault(
                                str(other_dim),
                                get_effective_chunk_size(
                                    int(other_length), None, comm.size
                                ),
                            )
                        set_mpi_meta(
                            piece,
                            dim=resolved_dim,
                            global_size=length,
                            start=start,
                            stop=stop,
                            chunk_info=piece_info,
                        )
                        pieces.append(piece)
            elif value is not None:
                raise ValueError(
                    f"Only rank {root} (root) may provide a value; "
                    + f"got one on rank {comm.rank}."
                )
        except BaseException as exc:
            error = exc
        self._runtime.raise_if_error(error, "mpi.xarray.distribute")

        # Broadcast which transfer path root prepared.
        dimensionless = self._runtime.broadcast(
            replicated_value is not None if is_root else None, root=root
        )

        # Transfer the validated pieces.
        if dimensionless:
            # Nothing to partition: same small object broadcast to every
            # rank, no per-rank slicing or point-to-point send needed.
            output = self._runtime.broadcast(
                replicated_value if is_root else None, root=root
            )
            return cast("xr.Dataset | xr.DataArray", output)

        if is_root:
            assert pieces is not None
            for rank, piece in enumerate(pieces):
                if rank == root:
                    output = piece
                else:
                    self._runtime.send(piece, dest=rank, tag=self._DISTRIBUTE_TAG)
        else:
            output = self._runtime.receive(source=root, tag=self._DISTRIBUTE_TAG)

        if should_log_partitions(self._runtime, log_partitions):
            meta = get_mpi_meta(output)
            if meta is not None:
                log_partition_report(
                    self._runtime,
                    output,
                    meta["dim"],
                    origin="mpi.xarray.distribute",
                    global_size=meta["global_size"],
                    start=meta["start"],
                    stop=meta["stop"],
                    automatic=(dim == "auto"),
                )
        return output

    def create_dataarray(
        self,
        fill: Callable[[int, int], Any],
        dims: Sequence[Hashable],
        *,
        shape: Sequence[int] | Mapping[Hashable, int] | None = None,
        dim: Hashable | int = 0,
        dtype: Any = np.float64,
        coords: Mapping[Hashable, Any] | None = None,
        name: Hashable | None = None,
        attrs: Mapping[str, Any] | None = None,
        log_partitions: bool = False,
    ) -> xr.DataArray:
        """Create a distributed DataArray from a rank-local fill function.

        Parameters
        ----------
        fill : callable
            Function called as ``fill(start, stop)`` for this rank's bounds.
        dims : sequence of Hashable
            Dimension names.
        shape : sequence of int, mapping, or None, optional
            Global dimension sizes. Missing sizes may be inferred from ``coords``.
        dim : Hashable or int, optional
            Dimension or axis to partition. Default is 0.
        dtype : Any, optional
            Data type returned by ``fill``. Default is ``numpy.float64``.
        coords : mapping, optional
            Coordinates passed to :class:`xarray.DataArray`.
        name : Hashable, optional
            DataArray name.
        attrs : mapping, optional
            DataArray attributes.
        log_partitions : bool, optional
            Log the resulting rank layout. Default is False.

        Returns
        -------
        xarray.DataArray
            Lazy rank-local DataArray carrying ``mpi_meta``.

        Raises
        ------
        ValueError
            If ``dim`` is invalid or global sizes cannot be resolved."""
        axis = dims.index(dim) if not isinstance(dim, Integral) else int(dim)
        if not 0 <= axis < len(dims):
            raise ValueError(f"dim {dim!r} is not in dims {tuple(dims)!r}.")
        dim_name = dims[axis]

        if shape is None or isinstance(shape, Mapping):
            explicit_sizes = dict(shape) if shape else None
        else:
            if len(shape) != len(dims):
                raise ValueError(
                    f"shape has {len(shape)} entries but dims has {len(dims)}."
                )
            explicit_sizes = dict(zip(dims, shape, strict=True))
        resolved_sizes = _resolve_sizes(dims, explicit_sizes, coords)

        comm = self._runtime.comm
        global_size = int(resolved_sizes[dim_name])
        start, stop = get_balanced_bounds(global_size, comm.rank, comm.size)
        local_shape = tuple(
            stop - start if name == dim_name else int(resolved_sizes[name])
            for name in dims
        )

        local_data = _delayed_local(fill, (start, stop), local_shape, dtype)

        local_coords = dict(coords) if coords else {}
        if dim_name in local_coords:
            local_coords[dim_name] = _localize_coord(
                local_coords[dim_name], global_size, start, stop
            )

        da = xr.DataArray(
            local_data, dims=tuple(dims), coords=local_coords, name=name, attrs=attrs
        )
        set_mpi_meta(
            da,
            dim=dim_name,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info={str(dim_name): stop - start},
        )
        if should_log_partitions(self._runtime, log_partitions):
            log_partition_report(
                self._runtime,
                da,
                dim_name,
                origin="mpi.xarray.create_dataarray",
                global_size=global_size,
                start=start,
                stop=stop,
            )
        return da

    def create_dataset(
        self,
        data_vars: Mapping[
            Hashable,
            xr.DataArray | tuple[Sequence[Hashable], Callable[[int, int], Any]],
        ],
        sizes: Mapping[Hashable, int] | None = None,
        *,
        dim: Hashable,
        dtype: Any = np.float64,
        coords: Mapping[Hashable, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        log_partitions: bool = True,
    ) -> xr.Dataset:
        """Create a distributed Dataset from rank-local variables.

        Parameters
        ----------
        data_vars : mapping
            Variables as DataArrays or ``(dims, fill)`` pairs. Partitioned fill
            functions receive ``(start, stop)``; unpartitioned fills take no arguments.
        sizes : mapping, optional
            Global dimension sizes. Missing sizes may be inferred from ``coords``.
        dim : Hashable
            Dimension to partition.
        dtype : Any or mapping, optional
            Default or per-variable fill dtype. Default is ``numpy.float64``.
        coords : mapping, optional
            Coordinates passed to :class:`xarray.Dataset`.
        attrs : mapping, optional
            Dataset attributes.
        log_partitions : bool, optional
            Log the resulting rank layout. Default is True.

        Returns
        -------
        xarray.Dataset
            Lazy rank-local Dataset carrying ``mpi_meta``.

        Raises
        ------
        ValueError
            If sizes cannot be resolved or a partitioned DataArray has the wrong
            local length."""
        required_dims: set[Hashable] = {dim}
        for spec in data_vars.values():
            if not isinstance(spec, xr.DataArray):
                var_dims, _ = spec
                required_dims.update(var_dims)
        resolved_sizes = _resolve_sizes(required_dims, sizes, coords)

        comm = self._runtime.comm
        global_size = int(resolved_sizes[dim])
        start, stop = get_balanced_bounds(global_size, comm.rank, comm.size)

        dtype_map = dtype if isinstance(dtype, Mapping) else None

        built_vars: dict[Hashable, Any] = {}
        for var_name, spec in data_vars.items():
            if isinstance(spec, xr.DataArray):
                if dim in spec.dims and int(spec.sizes[dim]) != stop - start:
                    raise ValueError(
                        f"data_vars[{var_name!r}] is a DataArray of length "
                        + f"{spec.sizes[dim]} along {dim!r}, but this rank "
                        + f"owns [{start}:{stop}) ({stop - start} elements). "
                        + "Pass a DataArray already sized to this rank's own "
                        + "bounds (e.g. from create_dataarray), not the full "
                        + "global array."
                    )
                built_vars[var_name] = spec
                continue

            var_dims, var_fill = spec
            var_dtype = (
                dtype_map.get(var_name, np.float64) if dtype_map is not None else dtype
            )
            if dim in var_dims:
                local_shape = tuple(
                    stop - start if name == dim else int(resolved_sizes[name])
                    for name in var_dims
                )
                local_data = _delayed_local(
                    var_fill, (start, stop), local_shape, var_dtype
                )
            elif callable(var_fill):
                # Not partitioned: identical on every rank, so there is no
                # (start, stop) to give -- fill() takes no arguments and
                # closes over whatever sizes it needs itself.
                local_shape = tuple(int(resolved_sizes[name]) for name in var_dims)
                local_data = _delayed_local(var_fill, (), local_shape, var_dtype)
            else:
                local_data = var_fill
            built_vars[var_name] = (tuple(var_dims), local_data)

        local_coords = dict(coords) if coords else {}
        if dim in local_coords:
            local_coords[dim] = _localize_coord(
                local_coords[dim], global_size, start, stop
            )

        ds = xr.Dataset(built_vars, coords=local_coords, attrs=attrs)
        set_mpi_meta(
            ds,
            dim=dim,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info={str(dim): stop - start},
        )
        if should_log_partitions(self._runtime, log_partitions):
            log_partition_report(
                self._runtime,
                ds,
                dim,
                origin="mpi.xarray.create_dataset",
                global_size=global_size,
                start=start,
                stop=stop,
            )
        return ds

    def repartition(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable | Literal["auto"] = "auto",
        *,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """Partition a replicated xarray object across MPI ranks.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Complete object present on every rank.
        dim : Hashable or {"auto"}, optional
            New partition dimension. ``"auto"`` selects the largest dimension.
            Default is ``"auto"``.
        chunk_info : mapping of str to int, optional
            Effective chunk-size hints.
        log_partitions : bool, optional
            Log the resulting rank layout. Default is False.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Rank-local slice carrying ``mpi_meta``.

        Raises
        ------
        ValueError
            If ``value`` is already distributed or ``dim`` is invalid."""
        if get_mpi_meta(value) is not None:
            raise ValueError(
                "Cannot repartition an already distributed object. "
                + "Reduce or gather its distributed dimension first."
            )

        automatic = dim == "auto"
        if automatic:
            if not value.dims:
                return strip_mpi_meta(value)
            dim = choose_partition_dim(
                value.sizes, self._runtime.comm.size, rank=self._runtime.comm.rank
            )

        if dim not in value.dims:
            raise ValueError(f"Repartition dimension {dim!r} does not exist.")

        info = dict(chunk_info or {})
        length = int(value.sizes[dim])
        chunk_size = int(
            info.get(
                str(dim),
                get_effective_chunk_size(length, None, self._runtime.comm.size),
            )
        )
        chunk_size = get_effective_chunk_size(
            length, chunk_size, self._runtime.comm.size
        )
        info[str(dim)] = chunk_size

        start, stop = get_chunk_bounds(
            length, chunk_size, self._runtime.comm.rank, self._runtime.comm.size
        )
        output = strip_mpi_meta(value).isel({dim: slice(start, stop)})
        info = prune_chunk_info(info, output)
        for other_dim, other_length in output.sizes.items():
            info.setdefault(
                str(other_dim),
                get_effective_chunk_size(
                    int(other_length), None, self._runtime.comm.size
                ),
            )

        set_mpi_meta(
            output, dim=dim, global_size=length, start=start, stop=stop, chunk_info=info
        )
        if should_log_partitions(self._runtime, log_partitions):
            log_partition_report(
                self._runtime,
                output,
                dim,
                origin="mpi.xarray.repartition",
                global_size=length,
                start=start,
                stop=stop,
                automatic=automatic,
            )
        return output

    def attach_save_chunks(
        self, value: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """Attach write-time chunk metadata to a distributed object.

        The save-chunk plan is computed on rank 0 from distribution metadata and
        broadcast to all ranks. No data are materialized.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Distributed rank-local object.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            ``value`` with ``mpi_meta["save_chunks"]`` attached. Undistributed input
            is returned unchanged.

        Raises
        ------
        ValueError
            If required partition chunk metadata are missing."""
        meta = get_mpi_meta(value)
        if meta is None:
            return value

        save_chunks: dict[str, tuple[int, ...]] | None = None
        error: BaseException | None = None
        if self._runtime.is_root():
            try:
                save_chunks = compute_save_chunks(value, meta, self._runtime.comm.size)
            except BaseException as exc:
                error = exc
        self._runtime.raise_if_error(error, "mpi.xarray.attach_save_chunks planning")

        save_chunks = self._runtime.broadcast(save_chunks, root=0)
        set_save_chunks(value, cast("dict[str, tuple[int, ...]]", save_chunks))
        return value
