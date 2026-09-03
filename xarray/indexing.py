"""Provide global-coordinate indexing for distributed xarray objects."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

import xarray as xr

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .cartesian import dim_comm as _dim_comm
from .chunks import get_chunk_bounds, get_effective_chunk_size, prune_chunk_info
from .meta import (
    choose_partition_dim,
    get_mpi_meta,
    indexer_is_scalar,
    set_mpi_meta,
    strip_mpi_meta,
)


def _select_partition_dim(
    meta: Mapping[str, Any], supplied: Mapping[Any, Any], *, caller: str
) -> Hashable | None:
    """Return the sole active partition dimension present in ``supplied``."""
    hit = tuple(dim for dim in meta["dims"] if dim in supplied)
    if not hit:
        return None
    if len(hit) > 1:
        raise NotImplementedError(
            f"{caller} cannot yet index more than one active partition "
            + f"dimension in a single call: {hit!r}"
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
    set_mpi_meta(
        output,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=chunk_info,
        cart=meta.get("cart"),
    )


def isel(
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
    meta = get_mpi_meta(value)
    if meta is None:
        return value.isel(supplied)

    dim = _select_partition_dim(meta, supplied, caller="isel")
    if dim is None:
        return value.isel(supplied)

    distributed_indexer = supplied.pop(dim)
    if indexer_is_scalar(distributed_indexer):
        return isel_scalar(mpi_context, value, dim, int(distributed_indexer), supplied)

    if not isinstance(distributed_indexer, slice):
        raise NotImplementedError(
            "Distributed isel currently supports slices and scalar indices."
        )
    if distributed_indexer.step not in (None, 1):
        raise NotImplementedError("Distributed isel currently requires slice step 1.")

    global_size = int(meta["global_sizes"][dim])
    requested_start, requested_stop, _ = distributed_indexer.indices(global_size)
    local_global_start = max(requested_start, int(meta["starts"][dim]))
    local_global_stop = min(requested_stop, int(meta["stops"][dim]))
    local_global_stop = max(local_global_start, local_global_stop)

    local_start = local_global_start - int(meta["starts"][dim])
    local_stop = local_global_stop - int(meta["starts"][dim])
    local_indexers = dict(supplied)
    local_indexers[dim] = slice(local_start, local_stop)
    output = value.isel(local_indexers)

    dim_comm = _dim_comm(mpi_context, meta, dim)
    counts = dim_comm.allgather(int(output.sizes[dim]))
    new_global_size = sum(counts)
    if new_global_size == 1 and partition_dim is not None:
        if len(meta["dims"]) > 1:
            raise NotImplementedError(
                "cannot yet redistribute a partition slice "
                + f"({dim!r}) that collapsed to a single global element "
                + "under a multi-dimensional partition; pass "
                + "partition_dim=None to keep it where it landed."
            )
        return _repartition_singleton(mpi_context, output, dim, counts, partition_dim)

    new_start = sum(counts[: dim_comm.rank])
    new_stop = new_start + counts[dim_comm.rank]
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


def isel_scalar(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    index: int,
    other_indexers: Mapping[Any, Any],
) -> xr.Dataset | xr.DataArray:
    """Select one global integer index from the partition dimension.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed object.
    dim : Hashable
        Partition dimension.
    index : int
        Global integer index.
    other_indexers : mapping
        Additional local ``isel`` indexers.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Replicated selected slice.

    Raises
    ------
    IndexError
        If ``index`` is outside the global dimension.
    """
    meta = get_mpi_meta(value)
    if meta is None:
        return value.isel({dim: index, **other_indexers})

    global_size = int(meta["global_sizes"][dim])
    normalized = index + global_size if index < 0 else index
    if normalized < 0 or normalized >= global_size:
        raise IndexError(
            f"index {index} is out of bounds for dimension {dim!r} "
            + f"with size {global_size}."
        )

    dim_comm = _dim_comm(mpi_context, meta, dim)
    owner = None
    parts = dim_comm.allgather((int(meta["starts"][dim]), int(meta["stops"][dim])))
    for candidate_rank, (start, stop) in enumerate(parts):
        if start <= normalized < stop:
            owner = candidate_rank
            break
    if owner is None:
        raise RuntimeError("Distributed partitions do not own the requested index.")

    result = None
    if dim_comm.rank == owner:
        local_index = normalized - int(meta["starts"][dim])
        result = strip_mpi_meta(value).isel({dim: local_index, **other_indexers})
    result = dim_comm.bcast(result, root=owner)
    remaining_dims = tuple(
        d for d in meta["dims"] if d != dim and d in getattr(result, "dims", ())
    )
    if remaining_dims:
        # `dim` just collapsed away; every other active partition axis
        # is untouched, so its existing bounds are simply carried
        # forward. The Cartesian "cart" descriptor is only carried
        # forward when every one of its axes survived (i.e. `dim` was
        # the only one to begin with, which can't happen here since
        # `remaining_dims` is non-empty) -- otherwise it no longer
        # describes every axis it was built for, and a fresh, smaller
        # topology is built lazily on demand instead (see `_finish`'s
        # identical handling for a reduction).
        set_mpi_meta(
            result,
            dim=remaining_dims,
            global_size={d: int(meta["global_sizes"][d]) for d in remaining_dims},
            start={d: int(meta["starts"][d]) for d in remaining_dims},
            stop={d: int(meta["stops"][d]) for d in remaining_dims},
            chunk_info=prune_chunk_info(meta["chunk_info"], result),
            cart=meta.get("cart") if len(remaining_dims) == len(meta["dims"]) else None,
        )
    return cast("xr.Dataset | xr.DataArray", result)


def sel(
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
    meta = get_mpi_meta(value)
    if meta is None:
        return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

    dim = _select_partition_dim(meta, supplied, caller="sel")
    if dim is None:
        return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

    distributed_indexer = supplied.pop(dim)
    if indexer_is_scalar(distributed_indexer):
        return sel_scalar(
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
        raise NotImplementedError(
            "Distributed sel currently supports slices and scalar labels."
        )

    local_indexers = dict(supplied)
    local_indexers[dim] = distributed_indexer
    output = value.sel(local_indexers, method=method, tolerance=tolerance, drop=drop)
    dim_comm = _dim_comm(mpi_context, meta, dim)
    counts = dim_comm.allgather(int(output.sizes[dim]))
    new_global_size = sum(counts)
    if new_global_size == 1 and partition_dim is not None:
        if len(meta["dims"]) > 1:
            raise NotImplementedError(
                "cannot yet redistribute a partition slice "
                + f"({dim!r}) that collapsed to a single global element "
                + "under a multi-dimensional partition; pass "
                + "partition_dim=None to keep it where it landed."
            )
        return _repartition_singleton(mpi_context, output, dim, counts, partition_dim)

    new_start = sum(counts[: dim_comm.rank])
    new_stop = new_start + counts[dim_comm.rank]
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


def sel_scalar(
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
    """Select one global label from the partition dimension.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed object.
    dim : Hashable
        Partition dimension.
    label : Any
        Global coordinate label.
    other_indexers : mapping
        Additional non-partition ``sel`` indexers.
    method : str or None
        Inexact matching method.
    tolerance : Any
        Maximum distance for inexact matches.
    drop : bool
        Whether to drop selected coordinates.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Replicated selected slice.
    """
    if method is not None:
        meta = get_mpi_meta(value)
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

        # Resolve the match using only this rank's own local coordinate
        # slice -- no other rank's coordinate values are needed to know
        # how good this rank's own candidate is, so this is entirely
        # rank-local (zero communication). "nearest" ranks candidates by
        # distance to `label`; "pad"/"ffill" and "backfill"/"bfill" rank
        # by the matched coordinate value itself, since xarray's own
        # per-rank `.sel(method=...)` already finds this rank's tightest
        # local bound (largest local coord <= label, or smallest local
        # coord >= label respectively) -- the true global bound is then
        # just the max (ffill) or min (bfill) of those local bounds
        # across ranks, exactly like a bounded reduction rather than a
        # full-object allgather.
        if method in ("nearest",):
            rank_fn = min
        elif method in ("pad", "ffill"):
            rank_fn = max
        elif method in ("backfill", "bfill"):
            rank_fn = min
        else:
            raise NotImplementedError(
                f"Distributed inexact sel does not support method={method!r}."
            )

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
                    raise NotImplementedError(
                        "Inexact distributed sel requires a unique one-dimensional index."
                    )
                local_index = int(selected.item())
                matched_coord = local_coord[local_index]
                key = (
                    abs(matched_coord - label) if method == "nearest" else matched_coord
                )
                candidate = (local_start + local_index, key)

        # A small, fixed-size (one tuple per rank) collective -- not the
        # coordinate data itself -- is all that is genuinely required to
        # pick the global winner among each rank's already-resolved local
        # candidate.
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
            except BaseException as exc:
                error = exc
        mpi_context.raise_if_error(error, "distributed scalar selection", comm=dim_comm)
        result = dim_comm.bcast(result, root=owner)
        remaining_dims = tuple(
            d for d in meta["dims"] if d != dim and d in getattr(result, "dims", ())
        )
        if remaining_dims:
            set_mpi_meta(
                result,
                dim=remaining_dims,
                global_size={d: int(meta["global_sizes"][d]) for d in remaining_dims},
                start={d: int(meta["starts"][d]) for d in remaining_dims},
                stop={d: int(meta["stops"][d]) for d in remaining_dims},
                chunk_info=prune_chunk_info(meta["chunk_info"], result),
                cart=meta.get("cart")
                if len(remaining_dims) == len(meta["dims"])
                else None,
            )
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

    meta = get_mpi_meta(value)
    dim_comm = mpi_context.comm if meta is None else _dim_comm(mpi_context, meta, dim)
    found_ranks = dim_comm.allgather(found)
    owners = [rank for rank, state in enumerate(found_ranks) if state]
    if not owners:
        raise KeyError(f"No rank contains label {label!r} on {dim!r}.")
    if len(owners) > 1:
        raise NotImplementedError(
            "Distributed scalar sel requires labels to be owned by one rank."
        )
    owner = owners[0]
    payload = result if dim_comm.rank == owner else None
    result = dim_comm.bcast(payload, root=owner)
    if meta is not None:
        remaining_dims = tuple(
            d for d in meta["dims"] if d != dim and d in getattr(result, "dims", ())
        )
        if remaining_dims:
            set_mpi_meta(
                result,
                dim=remaining_dims,
                global_size={d: int(meta["global_sizes"][d]) for d in remaining_dims},
                start={d: int(meta["starts"][d]) for d in remaining_dims},
                stop={d: int(meta["stops"][d]) for d in remaining_dims},
                chunk_info=prune_chunk_info(meta["chunk_info"], result),
                cart=meta.get("cart")
                if len(remaining_dims) == len(meta["dims"])
                else None,
            )
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
        set_mpi_meta(
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
        raise ValueError(
            f"partition_dim={target!r} is not a surviving dimension of "
            + f"the selection result (old partition dimension {old_dim!r} "
            + "has already collapsed to a single global element and "
            + "cannot be reused)."
        )

    target_length = candidates[target]
    chunk_size = get_effective_chunk_size(target_length, None, comm.size)

    # Only the owner rank does any real work here (slicing its local
    # data into comm.size pieces); every other rank just receives.
    # Guard the owner's slicing so a failure there can't strand every
    # other rank blocked forever inside scatter() waiting on a root
    # that already raised and never called it -- the same hazard
    # IOMixin.partition() guards against for its own root-side prep.
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
    set_mpi_meta(
        local,
        dim=target,
        global_size=target_length,
        start=start,
        stop=stop,
        chunk_info=info,
    )
    return cast("xr.Dataset | xr.DataArray", local)
