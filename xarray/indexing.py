"""Provide global-coordinate indexing for distributed xarray objects."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .cartesian import mpp_dim_comm as _dim_comm
from .chunks import get_chunk_bounds, get_effective_chunk_size, prune_chunk_info
from .meta import (
    choose_partition_dim,
    indexer_is_scalar,
    mpp_get_meta,
    mpp_update_meta,
    reattach_meta_after_collapse,
    strip_mpi_meta,
)
from .mpp import mpp_slice_compute_domain


def _select_partition_dim(
    meta: Mapping[str, Any], supplied: Mapping[Any, Any], *, caller: str
) -> Hashable | None:
    """Return the sole active partition dimension present in ``supplied``."""
    hit = tuple(dim for dim in meta["dims"] if dim in supplied)
    if not hit:
        return None
    if len(hit) > 1:
        raise NotImplementedError(
            f"{caller} supports one partition dimension per call; got {hit!r}."
        )
    return hit[0]


def _merge_partition_meta(
    output: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    dim: Hashable,
    *,
    global_size: int,
    start: int,
    stop: int,
    chunk_info: Mapping[str, int],
) -> None:
    """Update metadata bounds for ``dim`` while preserving other partition axes."""
    global_sizes = dict(meta["global_sizes"])
    starts = dict(meta["starts"])
    stops = dict(meta["stops"])
    global_sizes[dim] = global_size
    starts[dim] = start
    stops[dim] = stop
    mpp_update_meta(
        output,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=chunk_info,
        cart=meta.get("cart"),
    )


def mpp_isel(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    indexers: Mapping[Any, Any] | None = None,
    *,
    partition_dim: Hashable | Literal["auto"] | None = None,
    **indexers_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Index a distributed object with global integer coordinates.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to index.
    indexers : mapping, optional
        Integer indexers using global coordinates on the partition dimension.
    partition_dim : Hashable or {"auto"} or None, optional
        Only consulted when a *slice* on the partition dimension leaves a single global element behind (a scalar indexer already collapses the dimension entirely and broadcasts, so this does not apply there).
    **indexers_kwargs : Any
        Additional indexers passed by dimension name.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Indexed object with updated distribution metadata.

    """
    supplied = dict(indexers or {})
    supplied.update(indexers_kwargs)
    meta = mpp_get_meta(value)
    if meta is None:
        return value.isel(supplied)

    dim = _select_partition_dim(meta, supplied, caller="isel")
    if dim is None:
        return value.isel(supplied)

    distributed_indexer = supplied.pop(dim)
    if indexer_is_scalar(distributed_indexer):
        return mpp_isel_scalar(
            mpi_context, value, dim, int(distributed_indexer), supplied
        )

    if not isinstance(distributed_indexer, slice):
        raise NotImplementedError("Distributed isel supports only slices or scalar indices.")
    if distributed_indexer.step not in (None, 1):
        raise NotImplementedError("Distributed isel currently requires slice step 1.")

    global_size = int(meta["global_sizes"][dim])
    requested_start, requested_stop, _ = distributed_indexer.indices(global_size)
    requested_stop = max(requested_start, requested_stop)

    # Compute slice offsets from local contiguous bounds; no cross-rank metadata
    # exchange is needed.
    local_start, local_stop, new_start = mpp_slice_compute_domain(
        int(meta["starts"][dim]),
        int(meta["stops"][dim]),
        requested_start,
        requested_stop,
    )
    local_indexers = dict(supplied)
    local_indexers[dim] = slice(local_start, local_stop)
    output = value.isel(local_indexers)

    new_global_size = requested_stop - requested_start
    if new_global_size == 1 and partition_dim is not None:
        # Rare enough not to be worth deriving every rank's share locally;
        # the branch itself is taken identically on every rank, so the
        # collective below stays consistent.
        dim_comm = _dim_comm(mpi_context, meta, dim)
        counts = dim_comm.allgather(int(output.sizes[dim]))
        if len(meta["dims"]) > 1:
            raise NotImplementedError(f"Cannot redistribute collapsed partition dimension {dim!r}.")
        return _repartition_singleton(mpi_context, output, dim, counts, partition_dim)

    new_stop = new_start + (local_stop - local_start)
    chunk_info = prune_chunk_info(meta["chunk_info"], output)
    _merge_partition_meta(
        output,
        meta,
        dim,
        global_size=new_global_size,
        start=new_start,
        stop=new_stop,
        chunk_info=chunk_info,
    )
    return output


def mpp_isel_scalar(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    index: int,
    other_indexers: Mapping[Any, Any],
) -> xr.Dataset | xr.DataArray:
    """Select one global integer index from the partition dimension.

    Raises
    ------
    IndexError
        If ``index`` is outside the global dimension.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        return value.isel({dim: index, **other_indexers})

    global_size = int(meta["global_sizes"][dim])
    normalized = index + global_size if index < 0 else index
    if normalized < 0 or normalized >= global_size:
        raise IndexError(f"Index {index} is out of bounds for {dim!r} (size {global_size}).")

    dim_comm = _dim_comm(mpi_context, meta, dim)
    # Find a scalar index owner with a fixed-size reduction instead of gathering rank
    # bounds.
    claim = np.array(
        [
            dim_comm.rank
            if int(meta["starts"][dim]) <= normalized < int(meta["stops"][dim])
            else -1
        ],
        dtype=np.int64,
    )
    elected = np.empty_like(claim)
    dim_comm.Allreduce(claim, elected, op=MPI.MAX)
    owner = int(elected[0])
    if owner < 0:
        raise RuntimeError("Distributed partitions do not own the requested index.")

    result = None
    if dim_comm.rank == owner:
        local_index = normalized - int(meta["starts"][dim])
        result = strip_mpi_meta(value).isel({dim: local_index, **other_indexers})
        # Materialize scalar selections before broadcast to avoid pickling lazy Dask
        # graphs.
        result = result.load()
    result = dim_comm.bcast(result, root=owner)
    result = reattach_meta_after_collapse(result, meta, dim)
    return cast("xr.Dataset | xr.DataArray", result)


def mpp_sel(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    indexers: Mapping[Any, Any] | None = None,
    method: str | None = None,
    tolerance: Any = None,
    drop: bool = False,
    *,
    partition_dim: Hashable | Literal["auto"] | None = None,
    **indexers_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Index a distributed object with global coordinate labels.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to index.
    indexers : mapping, optional
        Label indexers using global semantics on the partition dimension.
    method : str, optional
        Inexact matching method passed to xarray.
    tolerance : Any, optional
        Maximum distance for inexact matches.
    drop : bool, optional
        Drop selected coordinate variables.
    partition_dim : Hashable or {"auto"} or None, optional
        Only consulted when a label *slice* on the partition dimension leaves a single global element behind (a scalar label already collapses the dimension entirely and broadcasts, so this does not apply there).
    **indexers_kwargs : Any
        Additional indexers passed by dimension name.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Indexed object with updated distribution metadata.

    """
    supplied = dict(indexers or {})
    supplied.update(indexers_kwargs)
    meta = mpp_get_meta(value)
    if meta is None:
        return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

    dim = _select_partition_dim(meta, supplied, caller="sel")
    if dim is None:
        return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

    distributed_indexer = supplied.pop(dim)
    if indexer_is_scalar(distributed_indexer):
        return mpp_sel_scalar(
            mpi_context,
            value,
            dim,
            distributed_indexer,
            supplied,
            method=method,
            tolerance=tolerance,
            drop=drop,
        )

    if not isinstance(distributed_indexer, slice):
        raise NotImplementedError("Distributed sel supports only slices or scalar labels.")

    local_indexers = dict(supplied)
    local_indexers[dim] = distributed_indexer
    output = value.sel(local_indexers, method=method, tolerance=tolerance, drop=drop)
    dim_comm = _dim_comm(mpi_context, meta, dim)

    # Use a global sum and exclusive prefix sum to resolve label-slice sizes and
    # offsets.
    local_length = np.array([int(output.sizes[dim])], dtype=np.int64)
    total = np.empty_like(local_length)
    dim_comm.Allreduce(local_length, total, op=MPI.SUM)
    prefix = np.zeros_like(local_length)
    dim_comm.Exscan(local_length, prefix, op=MPI.SUM)
    if dim_comm.rank == 0:
        prefix[0] = 0  # Exscan leaves rank 0's receive buffer undefined.

    new_global_size = int(total[0])
    new_start = int(prefix[0])
    if new_global_size == 1 and partition_dim is not None:
        counts = dim_comm.allgather(int(local_length[0]))
        if len(meta["dims"]) > 1:
            raise NotImplementedError(f"Cannot redistribute collapsed partition dimension {dim!r}.")
        return _repartition_singleton(mpi_context, output, dim, counts, partition_dim)

    new_stop = new_start + int(local_length[0])
    chunk_info = prune_chunk_info(meta["chunk_info"], output)
    _merge_partition_meta(
        output,
        meta,
        dim,
        global_size=new_global_size,
        start=new_start,
        stop=new_stop,
        chunk_info=chunk_info,
    )
    return output


def mpp_sel_scalar(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    label: Any,
    other_indexers: Mapping[Any, Any],
    *,
    method: str | None,
    tolerance: Any,
    drop: bool,
) -> xr.Dataset | xr.DataArray:
    """Select one global label from the partition dimension."""
    if method is not None:
        meta = mpp_get_meta(value)
        if meta is None:
            return value.sel(
                {dim: label, **other_indexers},
                method=method,
                tolerance=tolerance,
                drop=drop,
            )

        dim_comm = _dim_comm(mpi_context, meta, dim)
        if dim in value.coords:
            local_coord = np.asarray(value[dim].values)
        else:
            local_coord = np.arange(int(meta["starts"][dim]), int(meta["stops"][dim]))
        local_start = int(meta["starts"][dim])

        # Resolve each rank's best inexact match locally; reduce only the candidate
        # metadata globally.
        if method in ("nearest",):
            rank_fn = min
        elif method in ("pad", "ffill"):
            rank_fn = max
        elif method in ("backfill", "bfill"):
            rank_fn = min
        else:
            raise NotImplementedError(f"Distributed sel does not support method={method!r}.")

        candidate: tuple[int, Any] | None = None
        if local_coord.size:
            locator = xr.DataArray(
                np.arange(local_coord.size, dtype=np.int64),
                dims=(dim,),
                coords={dim: local_coord},
            )
            try:
                selected = locator.sel({dim: label}, method=method, tolerance=tolerance)
            except (KeyError, IndexError):
                selected = None
            if selected is not None:
                if selected.ndim != 0:
                    raise NotImplementedError("Inexact sel requires a unique 1-D index.")
                local_index = int(selected.item())
                matched_coord = local_coord[local_index]
                key = (
                    abs(matched_coord - label) if method == "nearest" else matched_coord
                )
                candidate = (local_start + local_index, key)

        # Exchange only one candidate tuple per rank to choose the global match.
        candidates = [c for c in dim_comm.allgather(candidate) if c is not None]
        if not candidates:
            raise KeyError(f"No match for label {label!r} on {dim!r}.")
        global_index = rank_fn(candidates, key=lambda pair: pair[1])[0]

        bounds = dim_comm.allgather((int(meta["starts"][dim]), int(meta["stops"][dim])))
        owner = next(
            rank
            for rank, (start, stop) in enumerate(bounds)
            if start <= global_index < stop
        )

        result = None
        error: BaseException | None = None
        if dim_comm.rank == owner:
            try:
                local_index = global_index - int(meta["starts"][dim])
                result = strip_mpi_meta(value).isel({dim: local_index}, drop=drop)
                if other_indexers:
                    result = result.sel(
                        other_indexers,
                        method=method,
                        tolerance=tolerance,
                        drop=drop,
                    )
                # Materialize before it gets pickled by bcast below (same
                # reasoning as the sibling scalar-selection function above).
                result = result.load()
            except BaseException as exc:
                error = exc
        mpi_context.raise_if_error(error, "distributed scalar selection", comm=dim_comm)
        result = dim_comm.bcast(result, root=owner)
        result = reattach_meta_after_collapse(result, meta, dim)
        return cast("xr.Dataset | xr.DataArray", result)

    result = None
    found = False
    try:
        result = strip_mpi_meta(value).sel(
            {dim: label, **other_indexers},
            method=method,
            tolerance=tolerance,
            drop=drop,
        )
        found = True
    except (KeyError, IndexError):
        pass

    meta = mpp_get_meta(value)
    dim_comm = mpi_context.comm if meta is None else _dim_comm(mpi_context, meta, dim)
    # Use one fixed-size sum to detect and identify a unique matching rank.
    claim = np.array([int(found), dim_comm.rank if found else 0], dtype=np.int64)
    tally = np.empty_like(claim)
    dim_comm.Allreduce(claim, tally, op=MPI.SUM)
    owner_count = int(tally[0])
    if owner_count == 0:
        raise KeyError(f"No rank contains label {label!r} on {dim!r}.")
    if owner_count > 1:
        raise NotImplementedError("Scalar sel requires a unique owning rank.")
    owner = int(tally[1])
    payload = result if dim_comm.rank == owner else None
    # Materialize before it gets pickled by bcast below (same reasoning as
    # the sibling scalar-selection functions above).
    if payload is not None:
        payload = payload.load()
    result = dim_comm.bcast(payload, root=owner)
    if meta is not None:
        result = reattach_meta_after_collapse(result, meta, dim)
    return cast("xr.Dataset | xr.DataArray", result)


def _repartition_singleton(
    mpi_context: MPIContext,
    output: xr.Dataset | xr.DataArray,
    old_dim: Hashable,
    counts: list[int],
    partition_dim: Hashable | Literal["auto"],
) -> xr.Dataset | xr.DataArray:
    """Scatter a slice-``isel``/``sel`` result stranded on one rank."""
    owner = counts.index(1)
    stripped = strip_mpi_meta(output)
    comm = mpi_context.comm

    def _keep_single_owner() -> xr.Dataset | xr.DataArray:
        """Keep the singleton result on one owning rank."""
        new_start = sum(counts[: comm.rank])
        new_stop = new_start + counts[comm.rank]
        chunk_info = prune_chunk_info({str(old_dim): 1}, output)
        mpp_update_meta(
            output,
            dim=old_dim,
            global_size=1,
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
        return output

    candidates = {
        name: int(length) for name, length in stripped.sizes.items() if name != old_dim
    }
    target = partition_dim
    if target == "auto":
        if not candidates or not any(n > 1 for n in candidates.values()):
            return _keep_single_owner()
        target = choose_partition_dim(candidates, comm.size, rank=comm.rank)
    elif target not in candidates:
        raise ValueError(f"partition_dim={target!r} is absent after selection.")

    target_length = candidates[target]
    chunk_size = get_effective_chunk_size(target_length, None, comm.size)

    # Guard owner-side slicing before scatter so root failures cannot strand other
    # ranks.
    error: BaseException | None = None
    parts: list[xr.Dataset | xr.DataArray] | None = None
    if comm.rank == owner:
        try:
            parts = [
                stripped.isel(
                    {
                        target: slice(
                            *get_chunk_bounds(target_length, chunk_size, r, comm.size)
                        )
                    }
                )
                for r in range(comm.size)
            ]
        except BaseException as exc:
            error = exc
    mpi_context.raise_if_error(error, "isel/sel partition_dim scatter")

    local = mpi_context.scatter(parts if comm.rank == owner else None, root=owner)

    start, stop = get_chunk_bounds(target_length, chunk_size, comm.rank, comm.size)
    info = {str(target): chunk_size}
    info = prune_chunk_info(info, local)
    for other_dim, other_length in local.sizes.items():
        info.setdefault(
            str(other_dim),
            get_effective_chunk_size(int(other_length), None, comm.size),
        )
    mpp_update_meta(
        local,
        dim=target,
        global_size=target_length,
        start=start,
        stop=stop,
        chunk_info=info,
    )
    return cast("xr.Dataset | xr.DataArray", local)
