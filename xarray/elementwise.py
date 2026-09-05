"""Provide MPI-aware elementwise, scan, and order-statistic operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from mpi4py import MPI

import xarray as xr

from .arithmetic import (
    mpp_check_operands_distribution,
    check_partition_preserved,
    mpp_halo_exchange,
    reattach_meta,
)
from .cartesian import mpp_dim_comm as _dim_comm
from .chunks import prune_chunk_info
from .meta import mpp_get_meta, mpp_update_meta, strip_mpi_meta
from .planning import _agree, guarded

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from ..mpi.context import MPIContext

#: Sentinel distinguishing "no fill value given" from a genuine ``other=None``.
_UNSET = object()


def mpp_where(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    cond: Any,
    other: Any = np.nan,
    *,
    drop: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Elementwise selection (``value.where(cond, other)``), MPI-safe.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to select from.
    cond : Any
        Boolean condition, following ``xarray.DataArray.where``.
    other : Any, optional
        Fill value where ``cond`` is False.
    drop : bool, optional
        Must be False for a distributed object.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The selected object, with ``.meta`` preserved unchanged.

    Raises
    ------
    ValueError
        If ``drop=True`` is requested on a distributed object, or the operands are distributed over incompatible partitions (see :meth:`~.arithmetic.Arithmetic.apply`).
    """
    operands = (value, cond, other)
    meta, reference = mpp_check_operands_distribution(mpi_context, operands)
    if meta is not None and drop:
        raise ValueError(
            "drop=True is not supported on a distributed object "
            + "(result length could differ across ranks)"
        )

    _agree(
        mpi_context,
        (
            "where",
            None if meta is None else (str(meta["dim"]), int(meta["global_size"])),
        ),
    )
    result = value.where(cond) if other is _UNSET else value.where(cond, other)
    if meta is None:
        return result
    check_partition_preserved(result, meta, reference)
    return reattach_meta(result, meta)


def mpp_cumsum(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Cumulative sum along ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to accumulate.
    dim : Hashable
        Dimension to accumulate along.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes on the rank-local cumulative sum step; lost by the subsequent addition of the cross-rank prefix.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Cumulative sum with the same local length and ``.meta`` as ``value``.
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.cumsum(dim, skipna=skipna, keep_attrs=keep_attrs)

    _agree(mpi_context, ("cumsum", str(dim), int(meta["global_size"])))

    if isinstance(value, xr.Dataset):
        # Only data variables that actually carry `dim` may enter the
        # cross-rank prefix scan below. xarray's own `.sum(dim)` (and
        # `.cumsum(dim)`) silently leave a variable lacking `dim`
        # completely unchanged rather than reducing it to a scalar --
        # every rank's "total" for such a variable is really just its
        # own already-identical, unreduced, replicated array. Feeding
        # that into the same gather/scatter prefix machinery as the
        # genuinely-reduced variables would silently add rank_index
        # extra copies of that array onto itself (rank r's exclusive
        # prefix becomes the sum of r copies of the same array, not the
        # additive identity 0 it needs to be for a variable with
        # nothing to accumulate), corrupting a variable that `dim`
        # never touched at all.
        touched = [name for name, var in value.data_vars.items() if dim in var.dims]
        if not touched:
            return strip_mpi_meta(value.copy(deep=False))
        untouched = [name for name in value.data_vars if name not in touched]
        scanned = _cumsum_scan(
            mpi_context, value[touched], dim, meta, skipna=skipna, keep_attrs=keep_attrs
        )
        result = (
            xr.merge([scanned, value[untouched]], combine_attrs="no_conflicts")
            if untouched
            else scanned
        )
        result.attrs = dict(value.attrs)
        return reattach_meta(result, meta)

    return reattach_meta(
        _cumsum_scan(
            mpi_context, value, dim, meta, skipna=skipna, keep_attrs=keep_attrs
        ),
        meta,
    )


def _cumsum_scan(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    meta: Mapping[str, Any],
    *,
    skipna: bool | None,
    keep_attrs: bool | None,
) -> xr.Dataset | xr.DataArray:
    """Cross-rank prefix-sum core of :meth:`cumsum`."""

    def _locals() -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
        """Return this rank's local cumulative sum and total."""
        local_cumsum = value.cumsum(dim, skipna=skipna, keep_attrs=keep_attrs)
        local_total = value.sum(dim, skipna=skipna)
        return local_cumsum, local_total

    locals_or_none, error = guarded(_locals)
    mpi_context.raise_if_error(
        error, "MPI xarray cumsum", signature=("cumsum", str(dim))
    )
    local_cumsum, local_total = locals_or_none
    # Materialize before gathering, for the same reason mpp_median() does:
    # `.sum(dim)` alone does not force a still-lazy dask-backed `value` to
    # compute, so `local_total` can still be lazy here, and `comm.gather`
    # pickles it as-is -- a lazy graph is not guaranteed picklable (e.g. a
    # local-closure `fill` passed to `mpi_create_dataarray`, as in this
    # project's own test fixtures). `local_cumsum` does not need the same
    # treatment: it is never pickled, only added to `exclusive_prefix`
    # below, and whatever laziness it still carries is resolved by
    # whatever later materializes the final returned result.
    local_total = local_total.load()

    comm = _dim_comm(mpi_context, meta, dim)
    # MPI_EXSCAN computes exactly the exclusive running total this needs
    # (rank r's prefix = sum of every totals[0..r-1]) directly, via a
    # proper distributed scan algorithm (mpi4py's object-based
    # `exscan`, which pickles `local_total` and reduces with Python's
    # own `+` -- xr.Dataset/DataArray both support it, so no manual
    # tree logic is needed here). The previous implementation instead
    # gathered every rank's total onto rank 0, computed every prefix
    # there in a serial Python loop, and scattered them back out --
    # correct, but a gather+scatter round trip through a single root is
    # the one thing MPI's own MPI_EXSCAN exists specifically to avoid,
    # and it does not scale as favorably to a large rank count.
    #
    # MPI_EXSCAN leaves rank 0's result undefined (there is no
    # predecessor to sum), which mpi4py surfaces as `None`; the correct
    # exclusive prefix there is a genuine zero, not `local_total * 0` --
    # if any rank's own total contains +-inf (routine in real
    # geophysical fields, e.g. log of a non-positive value), `inf * 0`
    # is NaN, corrupting what should be a clean +-inf result.
    exclusive_prefix = comm.exscan(local_total, op=MPI.SUM)
    if exclusive_prefix is None:
        exclusive_prefix = xr.zeros_like(local_total)

    return local_cumsum + exclusive_prefix


def mpp_ffill(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    limit: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Forward-fill along ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to fill.
    dim : Hashable
        Dimension to fill along.
    limit : int or None, optional
        As in ``xarray.DataArray.ffill``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The forward-filled object, same shape and distribution as the input.
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.ffill(dim, limit=limit)

    if limit is not None:
        _agree(mpi_context, ("ffill", str(dim), int(limit)))
        padded, left_pad, _right_pad = mpp_halo_exchange(
            mpi_context, value, dim, before=limit, after=0
        )
        filled = padded.ffill(dim, limit=limit)
        local_len = int(value.sizes[dim])
        trimmed = filled.isel({dim: slice(left_pad, left_pad + local_len)})
        return reattach_meta(trimmed, meta)

    _agree(mpi_context, ("ffill", str(dim), None))
    return reattach_meta(_fill_scan(mpi_context, value, dim, meta, forward=True), meta)


def mpp_bfill(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    limit: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Backward-fill along ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to fill.
    dim : Hashable
        Dimension to fill along.
    limit : int or None, optional
        As in ``xarray.DataArray.bfill``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The backward-filled object, same shape and distribution as the input.
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.bfill(dim, limit=limit)

    if limit is not None:
        _agree(mpi_context, ("bfill", str(dim), int(limit)))
        padded, _left_pad, right_pad = mpp_halo_exchange(
            mpi_context, value, dim, before=0, after=limit
        )
        filled = padded.bfill(dim, limit=limit)
        local_len = int(value.sizes[dim])
        trimmed = filled.isel({dim: slice(0, local_len)})
        return reattach_meta(trimmed, meta)

    _agree(mpi_context, ("bfill", str(dim), None))
    return reattach_meta(_fill_scan(mpi_context, value, dim, meta, forward=False), meta)


def _fill_scan(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    meta: Mapping[str, Any],
    *,
    forward: bool,
) -> xr.Dataset | xr.DataArray:
    """Unbounded ffill/bfill core: an exclusive-scan last-value-seen carry."""
    comm = _dim_comm(mpi_context, meta, dim)
    edge_index = -1 if forward else 0

    def _local() -> tuple[xr.Dataset | xr.DataArray, Any, bool]:
        """Return this rank's locally filled array and boundary value."""
        local_filled = value.ffill(dim) if forward else value.bfill(dim)
        edge_slice = local_filled.isel({dim: edge_index}, drop=True)
        has_valid = bool(edge_slice.notnull().all())
        # Materialize before it gets pickled by comm.exscan() below, for
        # the same reason mpp_median()/mpp_cumsum() do: computing has_valid above
        # forces *that* particular reduction, not edge_slice itself, which
        # can still carry a lazy dask graph when `value` does.
        return local_filled, edge_slice.load(), has_valid

    local_or_none, error = guarded(_local)
    mpi_context.raise_if_error(
        error,
        "MPI xarray ffill/bfill",
        signature=("fill_scan", str(dim), forward),
        comm=comm,
    )
    local_filled, edge_slice, has_valid = local_or_none

    def _last_valid(carry: Any, current: Any) -> Any:
        """Combine two (has_valid, edge_slice) pairs, keeping the more recent valid one."""
        return current if current[0] else carry

    # An exclusive scan of "last valid value seen so far" -- MPI_EXSCAN's
    # own textbook use case, just with a custom combine instead of SUM.
    # Ascending rank order is `forward`'s fill direction directly; `bfill`
    # needs the same scan walked from the *last* rank toward the first,
    # which MPI_EXSCAN has no "reverse" mode for, so it runs on a
    # same-ranks-different-numbering sub-communicator instead: `Split`
    # with `key = size - 1 - rank` relabels rank `size-1` as scan-rank 0
    # and rank 0 as scan-rank `size-1`, without moving any data or
    # posting any extra messages beyond the scan itself. This replaces
    # the previous gather-every-edge-value-to-rank-0 -> serial Python
    # loop -> scatter round trip (every rank waiting on one root for
    # both halves of that trip) with the same proper distributed
    # algorithm mpp_cumsum uses.
    scan_comm = comm if forward else comm.Split(0, comm.size - 1 - comm.rank)
    try:
        carry_in_pair = scan_comm.exscan((has_valid, edge_slice), op=_last_valid)
    finally:
        if scan_comm is not comm:
            scan_comm.Free()
    carry_in = None if carry_in_pair is None else carry_in_pair[1]

    if carry_in is None:
        return local_filled
    return local_filled.fillna(carry_in)


def mpp_interp(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    new_coord: Any,
    method: str = "linear",
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Interpolate onto ``new_coord`` along ``dim``, correct when distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to interpolate.
    dim : Hashable
        Dimension to interpolate along.
    new_coord : array-like
        This rank's own local slice of the new target coordinate along ``dim`` (not the global target grid -- exactly as this rank's own local ``value`` is its slice of the source, not the global source).
    method : str, optional
        As in ``xarray.DataArray.interp``.
    **kwargs : Any
        Forwarded to ``xarray.DataArray.interp``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Interpolated onto this rank's ``new_coord``, with ``.meta`` recomputed for the new length along ``dim`` (an allgather of each rank's own new local length, the same mechanism :func:`diff`/:func:`~.arithmetic.coarsen_reduce` use for their own length-changing case).
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.interp({dim: new_coord}, method=method, **kwargs)

    _agree(mpi_context, ("interp", str(dim), method))

    comm = _dim_comm(mpi_context, meta, dim)
    pieces = comm.allgather(value)
    full = (
        xr.concat(pieces, dim=dim, data_vars="minimal")
        if isinstance(value, xr.Dataset)
        else xr.concat(pieces, dim=dim)
    )
    result = full.interp({dim: new_coord}, method=method, **kwargs)

    counts = comm.allgather(int(result.sizes[dim]))
    new_global_size = sum(counts)
    new_start = sum(counts[: comm.rank])
    new_stop = new_start + counts[comm.rank]
    chunk_info = prune_chunk_info(meta["chunk_info"], result)
    global_sizes = dict(meta["global_sizes"])
    starts = dict(meta["starts"])
    stops = dict(meta["stops"])
    global_sizes[dim] = new_global_size
    starts[dim] = new_start
    stops[dim] = new_stop
    mpp_update_meta(
        result,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=chunk_info,
        cart=meta.get("cart"),
    )
    return result


def mpp_median(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Median over ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : Hashable
        Dimension to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object. Under a single partition dimension, fully
        replicated (``.meta`` is None) since nothing remains
        distributed. Under a multi-dimensional partition, metadata is
        reattached for whichever dimension(s) survive ``dim`` being
        reduced away, with no duplicated ownership: exactly one rank
        per distinct surviving range keeps the real result; every
        other rank that shared that range before the reduction is left
        with a genuinely empty (``start == stop``) slice instead of a
        redundant copy.
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.median(dim, skipna=skipna, keep_attrs=keep_attrs)

    _agree(mpi_context, ("median", str(dim), int(meta["global_size"])))
    comm = _dim_comm(mpi_context, meta, dim)
    # Materialize before gathering: `comm.gather` (unlike the small-scalar
    # Allreduce every other reduction in this module uses) pickles `value`
    # itself, and a still-lazy dask-backed array's graph is not guaranteed
    # picklable in general -- in particular, `mpi_create_dataarray`'s public
    # `fill` callable is commonly a local closure (as in this project's own
    # test fixtures), which the standard `pickle` module cannot serialize at
    # all. `.load()` forces exactly the same local computation every other
    # reduction here already forces implicitly when it extracts a plain
    # scalar/small array from `value` before its own Allreduce, so this
    # costs nothing extra other operations don't already pay, and it makes
    # mpp_median() robust to any construction pattern rather than only ones
    # whose graph happens to be picklable.
    value = value.load()
    pieces = comm.gather(value, root=0)

    def _reduce_on_root() -> xr.Dataset | xr.DataArray:
        """Compute the requested median on the root rank."""
        full = (
            xr.concat(pieces, dim=dim, data_vars="minimal")
            if isinstance(value, xr.Dataset)
            else xr.concat(pieces, dim=dim)
        )
        return full.median(dim, skipna=skipna, keep_attrs=keep_attrs)

    result, error = guarded(_reduce_on_root) if comm.rank == 0 else (None, None)
    mpi_context.raise_if_error(
        error, "MPI xarray median", signature=("median", str(dim)), comm=comm
    )
    # Every member of this sub-communicator shares identical bounds along
    # every surviving dimension (they differ only along `dim`, which is
    # now reduced away), so broadcasting the small, already-reduced
    # result to all of them and letting every rank keep its own full
    # copy would leave `comm.size`-many redundant, byte-identical copies
    # of the same slice -- multiple ranks claiming ownership of the same
    # range, violating the no-overlap partition invariant every other
    # operation in this package relies on. Only rank 0 of this
    # sub-communicator keeps the real data; every other member is
    # assigned a genuinely empty (start == stop) slice instead, exactly
    # like a rank `get_balanced_bounds` leaves idle when a dimension is
    # shorter than the rank count -- not a second copy of someone else's
    # data. The broadcast itself still reaches everyone, since a
    # non-owner still needs the result's dtype/other-dims shape to build
    # its own correctly-typed empty slice.
    result = comm.bcast(result, root=0)
    result = strip_mpi_meta(result)

    remaining_dims = tuple(d for d in meta["dims"] if d != dim)
    if not remaining_dims:
        return result

    start = {d: int(meta["starts"][d]) for d in remaining_dims}
    stop = {d: int(meta["stops"][d]) for d in remaining_dims}
    if comm.rank != 0:
        empty_dim = remaining_dims[0]
        result = result.isel({empty_dim: slice(0, 0)})
        stop[empty_dim] = start[empty_dim]

    mpp_update_meta(
        result,
        dim=remaining_dims,
        global_size={d: int(meta["global_sizes"][d]) for d in remaining_dims},
        start=start,
        stop=stop,
        chunk_info=prune_chunk_info(meta["chunk_info"], result),
        cart=None,
    )
    return result


def mpp_diff(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    n: int = 1,
    *,
    label: Literal["upper", "lower"] = "upper",
) -> xr.Dataset | xr.DataArray:
    """``n``-th order difference along ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to difference.
    dim : Hashable
        Dimension to difference along.
    n : int, optional
        Order of the difference.
    label : {"upper", "lower"}, optional
        As in ``xarray.DataArray.diff``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The differenced object, ``n`` elements shorter along ``dim`` globally -- and, when ``dim`` is the partition dimension, at exactly one rank (0 for "upper", the last rank for "lower") locally; every other rank's local length is unchanged.

    Raises
    ------
    ValueError
        If ``n`` is negative, ``label`` is not "upper"/"lower", or any rank's local length along ``dim`` is shorter than ``n`` (this last case is caught by :meth:`~.arithmetic.mpp_halo_exchange` itself, which checks every rank's local length together via a synchronized ``allgather`` before raising, so the error is consistent and every rank raises together rather than some hanging).
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.diff(dim, n=n, label=label)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n!r}")
    if label not in ("upper", "lower"):
        raise ValueError(f"label must be 'upper' or 'lower', got {label!r}")
    if n == 0:
        return reattach_meta(value.diff(dim, n=0, label=label), meta)

    before, after = (n, 0) if label == "upper" else (0, n)
    padded, _left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after
    )
    diffed = padded.diff(dim, n=n, label=label)

    comm = _dim_comm(mpi_context, meta, dim)
    counts = comm.allgather(int(diffed.sizes[dim]))
    new_global_size = sum(counts)
    new_start = sum(counts[: comm.rank])
    new_stop = new_start + counts[comm.rank]
    chunk_info = prune_chunk_info(meta["chunk_info"], diffed)
    global_sizes = dict(meta["global_sizes"])
    starts = dict(meta["starts"])
    stops = dict(meta["stops"])
    global_sizes[dim] = new_global_size
    starts[dim] = new_start
    stops[dim] = new_stop
    mpp_update_meta(
        diffed,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=chunk_info,
        cart=meta.get("cart"),
    )
    return diffed


def mpp_shift(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    periods: int = 1,
    *,
    fill_value: Any = _UNSET,
) -> xr.Dataset | xr.DataArray:
    """Shift ``value`` by ``periods`` along ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to shift.
    dim : Hashable
        Dimension to shift along.
    periods : int, optional
        Number of positions to shift by; positive shifts values toward higher indices (as in ``xarray.DataArray.shift``).
    fill_value : Any, optional
        As in ``xarray.DataArray.shift``; defaults to xarray's own dtype-aware NA fill when omitted.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The shifted object, same shape and distribution as the input.
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        kwargs = {} if fill_value is _UNSET else {"fill_value": fill_value}
        return value.shift({dim: periods}, **kwargs)
    if periods == 0:
        return value

    before, after = (periods, 0) if periods > 0 else (0, -periods)
    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after
    )
    kwargs = {} if fill_value is _UNSET else {"fill_value": fill_value}
    shifted = padded.shift({dim: periods}, **kwargs)

    local_len = int(value.sizes[dim])
    trimmed = shifted.isel({dim: slice(left_pad, left_pad + local_len)})
    return reattach_meta(trimmed, meta)


def mpp_roll(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    shift: int,
) -> xr.Dataset | xr.DataArray:
    """Circularly shift ``value`` by ``shift`` along ``dim``, wrapping at the edge.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to roll.
    dim : Hashable
        Dimension to roll along.
    shift : int
        Number of positions to roll by; positive rolls toward higher indices.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rolled object, same shape and distribution as the input.
    """
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.roll({dim: shift}, roll_coords=False)

    global_size = int(meta["global_sizes"][dim])
    if global_size > 0:
        # Normalize to the smallest-magnitude equivalent shift, not
        # just into [0, global_size): e.g. shift=-2 on a length-8 array
        # is mathematically identical to shift=6, but would request an
        # unnecessarily wide (6-element) halo instead of the genuinely
        # sufficient 2-element one -- exactly the "don't exchange more
        # than the operation actually needs" rule applied to the wrap
        # case too.
        shift = shift % global_size
        if shift > global_size // 2:
            shift -= global_size
    if shift == 0:
        return value

    before, after = (shift, 0) if shift > 0 else (0, -shift)
    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after, periodic=True
    )
    shifted = padded.shift({dim: shift})

    local_len = int(value.sizes[dim])
    trimmed = shifted.isel({dim: slice(left_pad, left_pad + local_len)})
    # `.shift()` unconditionally reserves a float NaN fill value for the
    # boundary it introduces, upcasting any integer/bool variable to
    # float even though, by construction, that boundary is never
    # actually missing here: `mpp_halo_exchange(..., periodic=True)` already
    # padded with genuine neighbor data (wrapping at the true global
    # edge), so every position `trimmed` keeps is real, borrowed data,
    # never a fill value. Restore each variable's original dtype now
    # that the shift itself is done -- safe precisely because nothing
    # in `trimmed` can be NaN from this operation.
    if isinstance(value, xr.Dataset):
        original_dtypes = {name: var.dtype for name, var in value.variables.items()}
        for name, dtype in original_dtypes.items():
            if name in trimmed.variables and trimmed.variables[name].dtype != dtype:
                trimmed[name] = trimmed[name].astype(dtype)
    elif trimmed.dtype != value.dtype:
        trimmed = trimmed.astype(value.dtype)
    return reattach_meta(trimmed, meta)


def mpp_differentiate(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    coord: Hashable,
    edge_order: Literal[1, 2] = 1,
    datetime_unit: Any = None,
) -> xr.Dataset | xr.DataArray:
    """Differentiate ``value`` along ``coord``, correct when ``coord`` is distributed.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to differentiate.
    coord : Hashable
        Coordinate to differentiate along.
    edge_order : {1, 2}, optional
        As in ``xarray.DataArray.differentiate``.
    datetime_unit : Any, optional
        As in ``xarray.DataArray.differentiate``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The derivative, same shape and distribution as the input.

    Raises
    ------
    ValueError
        If any rank's local length along ``coord`` is shorter than 1 (see ``mpp_halo_exchange``'s own synchronized length check) or too short overall for ``edge_order`` (raised by xarray itself).
    """
    meta = mpp_get_meta(value)
    if meta is None or coord not in meta["dims"]:
        return value.differentiate(
            coord, edge_order=edge_order, datetime_unit=datetime_unit
        )

    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, coord, before=1, after=1
    )
    # dask's gradient (unlike every other mpp_halo_exchange consumer -- shift,
    # diff, rolling_reduce, coarsen_reduce, ffill/bfill) requires every
    # individual chunk along the differentiated axis, not just the total
    # local length, to exceed edge_order + 1. mpp_halo_exchange's padding
    # arrives as its own separate, unconsolidated 1-element chunk (e.g.
    # local shape 125000 pads to chunks (125000, 1), not one (125002,)
    # chunk), which is too small on its own regardless of how large the
    # rank's real local data is. Consolidating to a single chunk here
    # only touches this local, already-fully-materialized-by-mpp_halo_exchange
    # piece -- it does not change mpp_halo_exchange's own chunking for any of
    # its other, unaffected callers.
    if padded.chunks:
        padded = padded.chunk({coord: -1})
    derivative = padded.differentiate(
        coord, edge_order=edge_order, datetime_unit=datetime_unit
    )

    local_len = int(value.sizes[coord])
    trimmed = derivative.isel({coord: slice(left_pad, left_pad + local_len)})
    return reattach_meta(trimmed, meta)
