"""Provide global-coordinate indexing for distributed xarray objects."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

import xarray as xr

from .cartesian import dim_comm as _dim_comm
from .chunks import get_chunk_bounds, get_effective_chunk_size, prune_chunk_info
from .meta import (
    choose_partition_dim,
    get_mpi_meta,
    indexer_is_scalar,
    set_mpi_meta,
    strip_mpi_meta,
)

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime


def _select_partition_dim(
    meta: Mapping[str, Any], supplied: Mapping[Any, Any], *, caller: str
) -> Hashable | None:
    """Return the sole active partition dimension present in ``supplied``.

    None when the indexer touches no partition dimension at all (the
    caller then falls straight through to a plain, communication-free
    local ``isel``/``sel``, exactly as before this helper existed).

    Raises
    ------
    NotImplementedError
        If ``supplied`` indexes more than one partition dimension in the
        same call -- each partition axis needs its own cross-rank
        bookkeeping pass, which is not yet fused into a single one.
    """
    hit = tuple(dim for dim in meta["dims"] if dim in supplied)
    if not hit:
        return None
    if len(hit) > 1:
        raise NotImplementedError(
            f"Distributed {caller} cannot yet index more than one active "
            + f"partition dimension in a single call ({hit!r} under this "
            + f"multi-dimensional partition); call {caller}() once per "
            + "partition dimension instead."
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
    """Attach updated bounds for ``dim`` onto ``output`` in place, while
    preserving every other active partition dimension's metadata
    unchanged.

    A single-dimension partition (or a multi-dimensional one updated on
    every one of its axes at once) can just call :func:`set_mpi_meta`
    directly; this exists for the case unique to a multi-dimensional
    partition -- indexing changed the bounds of *one* axis while every
    other one is untouched, so those other axes' ``global_size``/
    ``start``/``stop`` must be carried forward exactly as they were, and
    the Cartesian topology descriptor (structural: grid shape and this
    rank's coordinates, neither of which indexing changes) carried
    forward too.
    """
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


class Indexing:
    """Provide global-coordinate indexing for distributed xarray objects.

    The host class must provide ``self._runtime``.
    """

    _runtime: MPIRuntime

    def isel(
        self,
        value: xr.Dataset | xr.DataArray,
        indexers: Mapping[Any, Any] | None = None,
        *,
        partition_dim: Hashable | Literal["auto"] | None = None,
        **indexers_kwargs: Any,
    ) -> xr.Dataset | xr.DataArray:
        """Index a distributed object with global integer coordinates.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to index.
        indexers : mapping, optional
            Integer indexers using global coordinates on the partition dimension.
        partition_dim : Hashable or {"auto"} or None, optional
            Only consulted when a *slice* on the partition dimension leaves a
            single global element behind (a scalar indexer already collapses
            the dimension entirely and broadcasts, so this does not apply
            there). Left at the default ``None``, that single element stays
            where it landed: one rank holds it, every other rank holds a
            length-0 slice on that dimension. Passing a dimension name (or
            ``"auto"`` to pick the largest remaining dimension) instead
            scatters that one rank's local data across all ranks along
            ``partition_dim``, so the object stays evenly spread out rather
            than parked on a single rank. ``"auto"`` is a no-op if no other
            dimension has more than one element. See
            :meth:`Indexing._repartition_singleton`.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Indexed object with updated distribution metadata. A scalar selection on
            the partition dimension is replicated on every rank."""
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
            return self.isel_scalar(value, dim, int(distributed_indexer), supplied)

        if not isinstance(distributed_indexer, slice):
            raise NotImplementedError(
                "Distributed isel currently supports slices and scalar indices."
            )
        if distributed_indexer.step not in (None, 1):
            raise NotImplementedError(
                "Distributed isel currently requires slice step 1."
            )

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

        dim_comm = _dim_comm(self._runtime, meta, dim)
        counts = dim_comm.allgather(int(output.sizes[dim]))
        new_global_size = sum(counts)
        if new_global_size == 1 and partition_dim is not None:
            if len(meta["dims"]) > 1:
                raise NotImplementedError(
                    "Distributed isel() cannot yet redistribute a "
                    + f"partition-dimension slice ({dim!r}) that collapsed "
                    + "to a single global element under a multi-dimensional "
                    + "partition; pass partition_dim=None to keep the "
                    + "single-element result where it landed instead."
                )
            return self._repartition_singleton(output, dim, counts, partition_dim)

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
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable,
        index: int,
        other_indexers: Mapping[Any, Any],
    ) -> xr.Dataset | xr.DataArray:
        """Select one global integer index from the partition dimension.

        Parameters
        ----------
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
            If ``index`` is outside the global dimension."""
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

        dim_comm = _dim_comm(self._runtime, meta, dim)
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
                cart=meta.get("cart")
                if len(remaining_dims) == len(meta["dims"])
                else None,
            )
        return cast("xr.Dataset | xr.DataArray", result)

    def sel(
        self,
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
        value : xarray.Dataset or xarray.DataArray
            Object to index.
        indexers : mapping, optional
            Label indexers using global semantics on the partition dimension.
        method : str, optional
            Inexact matching method passed to xarray.
        tolerance : Any, optional
            Maximum distance for inexact matches.
        drop : bool, optional
            Drop selected coordinate variables. Default is False.
        partition_dim : Hashable or {"auto"} or None, optional
            Only consulted when a label *slice* on the partition dimension
            leaves a single global element behind (a scalar label already
            collapses the dimension entirely and broadcasts, so this does
            not apply there). See :meth:`isel` and
            :meth:`Indexing._repartition_singleton` for the exact semantics.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Indexed object with updated distribution metadata. A scalar selection on
            the partition dimension is replicated on every rank."""
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
            return self.sel_scalar(
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
        output = value.sel(
            local_indexers, method=method, tolerance=tolerance, drop=drop
        )
        dim_comm = _dim_comm(self._runtime, meta, dim)
        counts = dim_comm.allgather(int(output.sizes[dim]))
        new_global_size = sum(counts)
        if new_global_size == 1 and partition_dim is not None:
            if len(meta["dims"]) > 1:
                raise NotImplementedError(
                    "Distributed sel() cannot yet redistribute a "
                    + f"partition-dimension slice ({dim!r}) that collapsed "
                    + "to a single global element under a multi-dimensional "
                    + "partition; pass partition_dim=None to keep the "
                    + "single-element result where it landed instead."
                )
            return self._repartition_singleton(output, dim, counts, partition_dim)

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
        self,
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
            Replicated selected slice."""
        if method is not None:
            meta = get_mpi_meta(value)
            if meta is None:
                return value.sel(
                    {dim: label, **other_indexers},
                    method=method,
                    tolerance=tolerance,
                    drop=drop,
                )

            dim_comm = _dim_comm(self._runtime, meta, dim)
            if dim in value.coords:
                local_coord = np.asarray(value[dim].values)
            else:
                local_coord = np.arange(
                    int(meta["starts"][dim]), int(meta["stops"][dim])
                )
            coord_parts = dim_comm.allgather(local_coord)
            global_coord = np.concatenate(coord_parts)
            locator = xr.DataArray(
                np.arange(global_coord.size, dtype=np.int64),
                dims=(dim,),
                coords={dim: global_coord},
            )
            selected = locator.sel({dim: label}, method=method, tolerance=tolerance)
            if selected.ndim != 0:
                raise NotImplementedError(
                    "Inexact distributed sel requires a unique one-dimensional index."
                )
            global_index = int(selected.item())

            bounds = dim_comm.allgather(
                (int(meta["starts"][dim]), int(meta["stops"][dim]))
            )
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
            self._runtime.raise_if_error(
                error, "distributed scalar selection", comm=dim_comm
            )
            result = dim_comm.bcast(result, root=owner)
            remaining_dims = tuple(
                d for d in meta["dims"] if d != dim and d in getattr(result, "dims", ())
            )
            if remaining_dims:
                set_mpi_meta(
                    result,
                    dim=remaining_dims,
                    global_size={
                        d: int(meta["global_sizes"][d]) for d in remaining_dims
                    },
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
        dim_comm = (
            self._runtime.comm if meta is None else _dim_comm(self._runtime, meta, dim)
        )
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
                    global_size={
                        d: int(meta["global_sizes"][d]) for d in remaining_dims
                    },
                    start={d: int(meta["starts"][d]) for d in remaining_dims},
                    stop={d: int(meta["stops"][d]) for d in remaining_dims},
                    chunk_info=prune_chunk_info(meta["chunk_info"], result),
                    cart=meta.get("cart")
                    if len(remaining_dims) == len(meta["dims"])
                    else None,
                )
        return cast("xr.Dataset | xr.DataArray", result)

    def _repartition_singleton(
        self,
        output: xr.Dataset | xr.DataArray,
        old_dim: Hashable,
        counts: list[int],
        partition_dim: Hashable | Literal["auto"],
    ) -> xr.Dataset | xr.DataArray:
        """Scatter a slice-``isel``/``sel`` result stranded on one rank.

        Called only when a *slice* selection on the partition dimension
        leaves exactly one global element behind: ``counts`` (each rank's
        local size on ``old_dim`` after the local ``isel``/``sel``) then
        has a single ``1`` and the rest ``0`` -- every other dimension's
        full local extent still sits on that one rank alongside it.
        Rather than leaving the object that lopsided, this picks a
        surviving dimension and scatters the owning rank's data across
        every rank along it with one point-to-point transfer per rank (via
        :meth:`~..mpi.runtime.MPIRuntime.scatter`), which is the least
        data movement a redistribution of that single owned chunk can do:
        each rank receives exactly the slice it ends up keeping, nothing
        more.

        Parameters
        ----------
        output : xarray.Dataset or xarray.DataArray
            Local slice result; ``mpi_meta`` is stripped before use.
        old_dim : Hashable
            The now globally length-1 former partition dimension.
        counts : list of int
            Each rank's local size on ``old_dim``, from ``allgather``.
        partition_dim : Hashable or {"auto"}
            Dimension to distribute across ranks. ``"auto"`` selects the
            largest remaining dimension; if none has more than one
            element there is nothing to spread out, and the existing
            single-owner layout on ``old_dim`` is kept unchanged.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Rank-local slice carrying fresh ``mpi_meta`` on the chosen
            dimension, or on ``old_dim`` if there was nothing to
            repartition onto.

        Raises
        ------
        ValueError
            If ``partition_dim`` names a dimension that is not present in
            the result, or is ``old_dim`` itself (already collapsed to one
            element, so it cannot be reused as the new partition
            dimension)."""
        owner = counts.index(1)
        stripped = strip_mpi_meta(output)
        comm = self._runtime.comm

        def _keep_single_owner() -> xr.Dataset | xr.DataArray:
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
            name: int(length)
            for name, length in stripped.sizes.items()
            if name != old_dim
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
        # IOMixin.distribute() guards against for its own root-side prep.
        error: BaseException | None = None
        parts: list[xr.Dataset | xr.DataArray] | None = None
        if comm.rank == owner:
            try:
                parts = [
                    stripped.isel(
                        {
                            target: slice(
                                *get_chunk_bounds(
                                    target_length, chunk_size, r, comm.size
                                )
                            )
                        }
                    )
                    for r in range(comm.size)
                ]
            except BaseException as exc:
                error = exc
        self._runtime.raise_if_error(error, "isel/sel partition_dim scatter")

        local = self._runtime.scatter(parts if comm.rank == owner else None, root=owner)

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
