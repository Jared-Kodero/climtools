"""Provide MPI-aware elementwise, scan, and order-statistic operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from .arithmetic import (
    check_operands_distribution,
    check_partition_preserved,
    halo_exchange,
    reattach_meta,
)
from .cartesian import dim_comm as _dim_comm
from .chunks import prune_chunk_info
from .meta import get_mpi_meta, set_mpi_meta, strip_mpi_meta
from .planning import _agree, guarded

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

#: Sentinel distinguishing "no fill value given" from a genuine ``other=None``.
_UNSET = object()


def where(
    runtime,
    value: xr.Dataset | xr.DataArray,
    cond: Any,
    other: Any = _UNSET,
    *,
    drop: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Elementwise selection (``value.where(cond, other)``), MPI-safe.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to select from.
    cond : Any
        Boolean condition, following ``xarray.DataArray.where``.
    other : Any, optional
        Fill value where ``cond`` is False. Omit for xarray's default
        (NaN).
    drop : bool, optional
        Must be False for a distributed object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The selected object, with ``.meta`` preserved unchanged.

    Raises
    ------
    ValueError
        If ``drop=True`` is requested on a distributed object, or the
        operands are distributed over incompatible partitions (see
        :meth:`~.arithmetic.Arithmetic.apply`).
    """
    operands = (value, cond) if other is _UNSET else (value, cond, other)
    meta, reference = check_operands_distribution(runtime, operands)
    if meta is not None and drop:
        raise ValueError(
            "where(): drop=True is not supported on a distributed "
            "object; it can remove a different number of positions on "
            "different ranks and desynchronize the partition. Select "
            "with isel()/sel() first, or repartition afterwards."
        )

    _agree(
        runtime,
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


def cumsum(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Cumulative sum along ``dim``, correct when ``dim`` is distributed.

    When ``dim`` is the active partition dimension, each rank's running
    total must include every earlier rank's total. This gathers every
    rank's local total onto rank 0, which computes each rank's
    *exclusive* prefix (the sum of every rank before it) and scatters
    one prefix back to each rank; every rank then adds its prefix onto
    its own local cumulative sum. No new MPI collective: just the
    ``gather``/``scatter`` pair already used elsewhere in this package
    (see :func:`~.io.IO.attach_save_chunks`).

    The rank-local cumulative-sum/total computation happens on every
    rank independently before the first collective; it is guarded the
    same way :meth:`~.reduction_planning.ReductionPlanningMixin._comm_reduce` guards
    its own local step, so a local failure on one rank (e.g. an
    unsupported dtype) is reported consistently on every rank via
    ``raise_if_error`` instead of leaving the other ranks blocked
    waiting at ``gather`` for a rank that already raised.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to accumulate.
    dim : Hashable
        Dimension to accumulate along.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics. Applied
        consistently to both the local cumulative sum and the local
        total that feeds the cross-rank prefix, so a rank's NaNs never
        change another rank's prefix.
    keep_attrs : bool or None, optional
        Whether to preserve attributes on the rank-local cumulative sum
        step; lost by the subsequent addition of the cross-rank prefix.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Cumulative sum with the same local length and ``.meta`` as
        ``value``.
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.cumsum(dim, skipna=skipna, keep_attrs=keep_attrs)

    _agree(runtime, ("cumsum", str(dim), int(meta["global_size"])))

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
            runtime, value[touched], dim, meta, skipna=skipna, keep_attrs=keep_attrs
        )
        result = (
            xr.merge([scanned, value[untouched]], combine_attrs="no_conflicts")
            if untouched
            else scanned
        )
        result.attrs = dict(value.attrs)
        return reattach_meta(result, meta)

    return reattach_meta(
        _cumsum_scan(runtime, value, dim, meta, skipna=skipna, keep_attrs=keep_attrs),
        meta,
    )


def _cumsum_scan(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    meta: Mapping[str, Any],
    *,
    skipna: bool | None,
    keep_attrs: bool | None,
) -> xr.Dataset | xr.DataArray:
    """Cross-rank prefix-sum core of :meth:`cumsum`.

    Every variable in ``value`` (a DataArray, or a Dataset already
    filtered to only the variables that carry ``dim``) is assumed to
    actually contain ``dim``, so ``value.sum(dim)`` genuinely reduces
    every one of them to a same-shaped, rank-local total -- the
    precondition the gather/scatter exclusive-prefix computation below
    requires.

    Scoped to :func:`~.cartesian.dim_comm`'s dimension-only
    communicator, not the full runtime communicator: under a
    multi-dimensional partition, ranks that vary along a *different*
    partition axis (e.g. a different ``lon`` column, when scanning
    ``lat``) must run entirely independent prefix scans rather than
    being folded into one flat, axis-blind rank ordering -- the same
    reasoning :func:`~.cartesian.dim_comm`'s own docstring gives for
    :meth:`diff`/:meth:`isel`.
    """

    def _locals() -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
        local_cumsum = value.cumsum(dim, skipna=skipna, keep_attrs=keep_attrs)
        local_total = value.sum(dim, skipna=skipna)
        return local_cumsum, local_total

    locals_or_none, error = guarded(_locals)
    runtime.raise_if_error(error, "MPI xarray cumsum", signature=("cumsum", str(dim)))
    local_cumsum, local_total = locals_or_none

    comm = _dim_comm(runtime, meta, dim)
    totals = comm.gather(local_total, root=0)
    prefixes = None
    if comm.rank == 0:
        prefixes = []
        # The additive identity must be a genuine zero, not `totals[0] * 0`:
        # if any rank's local total contains +-inf (routine in real
        # geophysical fields -- e.g. log of a non-positive value), `inf * 0`
        # is NaN, and that single NaN becomes every rank's exclusive prefix
        # at that position (each `prefixes[i]` derives from this same
        # `running` seed), silently turning a correct +-inf cumsum result
        # into NaN everywhere, not just on the rank that produced the inf.
        running = xr.zeros_like(totals[0])
        for total in totals:
            prefixes.append(running)
            running = running + total
    exclusive_prefix = comm.scatter(prefixes, root=0)

    return local_cumsum + exclusive_prefix


def ffill(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    limit: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Forward-fill along ``dim``, correct when ``dim`` is distributed.

    Two genuinely different strategies, chosen by whether ``limit``
    bounds the dependency (per the routing rules in the module's
    top-level design notes: bounded gets a halo, unbounded gets a
    scan):

    ``limit`` given: the fill can only ever reach back ``limit``
    positions, so this borrows exactly ``limit`` elements from the
    left neighbor via :func:`~.arithmetic.halo_exchange` -- the same
    single-hop, fixed-width structure as :func:`shift` -- fills
    locally, and trims the borrowed prefix back off.

    ``limit=None`` (default, matching ``xarray``): the last valid
    value can originate arbitrarily many ranks back (an entire empty
    rank must still receive it), which no fixed-width halo can bound.
    Uses the same gather-on-root/scatter-back exclusive-prefix scan
    :func:`cumsum` already uses for its own unbounded cross-rank
    dependency, with "carry the last value seen so far" in place of
    "carry the running sum": every rank computes its own local
    ``ffill`` (fills everything reachable from data it already owns)
    and reports one small (dim-collapsed) slice -- its own last valid
    value, or nothing if it has none -- to root; root's sequential
    scan turns those into each rank's *incoming* carry (the nearest
    earlier rank's last valid value, skipping over any rank with none
    at all); every rank then fills whatever its own local ``ffill``
    could not with that one broadcast-shaped value. A rank with no
    valid value anywhere before it (including itself) is left with
    genuine leading NaNs, exactly matching plain ``xarray.ffill`` at
    the true start of the array.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to fill.
    dim : Hashable
        Dimension to fill along.
    limit : int or None, optional
        As in ``xarray.DataArray.ffill``. When given and ``dim`` is
        the partition dimension, must not exceed any rank's own local
        length along ``dim`` (see ``halo_exchange``'s single-hop
        limit).

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The forward-filled object, same shape and distribution as the
        input.
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.ffill(dim, limit=limit)

    if limit is not None:
        _agree(runtime, ("ffill", str(dim), int(limit)))
        padded, left_pad, _right_pad = halo_exchange(
            runtime, value, dim, before=limit, after=0
        )
        filled = padded.ffill(dim, limit=limit)
        local_len = int(value.sizes[dim])
        trimmed = filled.isel({dim: slice(left_pad, left_pad + local_len)})
        return reattach_meta(trimmed, meta)

    _agree(runtime, ("ffill", str(dim), None))
    return reattach_meta(_fill_scan(runtime, value, dim, meta, forward=True), meta)


def bfill(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    limit: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Backward-fill along ``dim``, correct when ``dim`` is distributed.

    Exact mirror image of :func:`ffill`: bounded ``limit`` borrows
    from the *right* neighbor instead of the left; unbounded runs the
    same carry scan in reverse rank order, carrying each rank's own
    *first* valid value backward instead of its last valid value
    forward. See :func:`ffill` for the full rationale.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to fill.
    dim : Hashable
        Dimension to fill along.
    limit : int or None, optional
        As in ``xarray.DataArray.bfill``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The backward-filled object, same shape and distribution as
        the input.
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.bfill(dim, limit=limit)

    if limit is not None:
        _agree(runtime, ("bfill", str(dim), int(limit)))
        padded, _left_pad, right_pad = halo_exchange(
            runtime, value, dim, before=0, after=limit
        )
        filled = padded.bfill(dim, limit=limit)
        local_len = int(value.sizes[dim])
        trimmed = filled.isel({dim: slice(0, local_len)})
        return reattach_meta(trimmed, meta)

    _agree(runtime, ("bfill", str(dim), None))
    return reattach_meta(_fill_scan(runtime, value, dim, meta, forward=False), meta)


def _fill_scan(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    meta: Mapping[str, Any],
    *,
    forward: bool,
) -> xr.Dataset | xr.DataArray:
    """Unbounded ffill/bfill core: a gather/scatter last-value-seen scan.

    ``forward=True`` scans ranks 0..size-1 carrying each rank's own
    last valid value forward (ffill); ``forward=False`` scans ranks
    size-1..0 carrying each rank's own first valid value backward
    (bfill). Both are the same "override with whatever was most
    recently seen" associative scan as :func:`~.ffill`'s docstring
    describes, just walked in opposite directions.

    Scoped to :func:`~.cartesian.dim_comm`'s dimension-only
    communicator, not the full runtime communicator: under a
    multi-dimensional partition, ranks that vary along a *different*
    partition axis (e.g. a different ``lon`` column, when filling
    along ``lat``) must run entirely independent scans rather than
    being folded into one flat, axis-blind rank ordering -- the same
    reasoning :func:`~.cartesian.dim_comm`'s own docstring gives for
    :meth:`diff`/:meth:`isel`, and :func:`_cumsum_scan` already
    applies to its own gather/scatter prefix scan.
    """
    comm = _dim_comm(runtime, meta, dim)
    edge_index = -1 if forward else 0

    def _local() -> tuple[xr.Dataset | xr.DataArray, Any, bool]:
        local_filled = value.ffill(dim) if forward else value.bfill(dim)
        edge_slice = local_filled.isel({dim: edge_index}, drop=True)
        has_valid = bool(edge_slice.notnull().all())
        return local_filled, edge_slice, has_valid

    local_or_none, error = guarded(_local)
    runtime.raise_if_error(
        error,
        "MPI xarray ffill/bfill",
        signature=("fill_scan", str(dim), forward),
        comm=comm,
    )
    local_filled, edge_slice, has_valid = local_or_none

    rank_order = range(comm.size) if forward else range(comm.size - 1, -1, -1)
    gathered = comm.gather((has_valid, edge_slice), root=0)
    carries = None
    if comm.rank == 0:
        carries = [None] * comm.size
        running = None
        for r in rank_order:
            carries[r] = running
            rank_has_valid, rank_edge = gathered[r]
            if rank_has_valid:
                running = rank_edge
    carry_in = comm.scatter(carries, root=0)

    if carry_in is None:
        return local_filled
    return local_filled.fillna(carry_in)


def interp(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    new_coord: Any,
    method: str = "linear",
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Interpolate onto ``new_coord`` along ``dim``, correct when distributed.

    Unlike :func:`rolling_reduce`/:func:`diff`/:func:`shift`, an
    interpolation target point has no fixed-width dependency on the
    source data: depending on how ``new_coord`` relates to the
    original grid, the two source points bracketing a given target
    point could be owned by any rank, not just an immediate neighbor
    (the spec's own distinction: interp "may require targeted source
    points rather than a fixed-width halo"). Building genuine
    point-to-point targeted delivery for arbitrary target grids is
    real future work; this instead takes the explicitly-sanctioned
    fallback for exactly this situation -- "global reconstruction,
    only when genuinely required" -- but as an ``Allgather`` (every
    rank ends up with the full source along ``dim``) rather than a
    gather-to-root: unlike :func:`median`'s reduction (whose *output*
    is small and identical on every rank, so a single root computes it
    once and broadcasts), every rank here interpolates onto its own,
    generally different, slice of ``new_coord`` and so must each end
    up with a real, independently-usable result -- there is no single
    small answer to broadcast.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to interpolate.
    dim : Hashable
        Dimension to interpolate along.
    new_coord : array-like
        This rank's own local slice of the new target coordinate
        along ``dim`` (not the global target grid -- exactly as this
        rank's own local ``value`` is its slice of the source, not
        the global source).
    method : str, optional
        As in ``xarray.DataArray.interp``. Default ``"linear"``.
    **kwargs : Any
        Forwarded to ``xarray.DataArray.interp``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Interpolated onto this rank's ``new_coord``, with ``.meta``
        recomputed for the new length along ``dim`` (an allgather of
        each rank's own new local length, the same mechanism
        :func:`diff`/:func:`~.arithmetic.coarsen_reduce` use for their
        own length-changing case).
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.interp({dim: new_coord}, method=method, **kwargs)

    _agree(runtime, ("interp", str(dim), method))

    comm = _dim_comm(runtime, meta, dim)
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
    set_mpi_meta(
        result,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=chunk_info,
        cart=meta.get("cart"),
    )
    return result


def median(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
) -> xr.Dataset | xr.DataArray:
    """Median over ``dim``, correct when ``dim`` is distributed.

    Median has no MPI reduction operator (unlike sum/min/max), so when
    ``dim`` is the active partition dimension this gathers every
    rank's slice onto rank 0, which reconstructs the full ``dim`` and
    takes xarray's own median locally, then broadcasts the (already
    reduced, small) result back to every rank. Only rank 0 ever
    materializes the full ``dim`` -- unlike an ``Allgather``, which
    would replicate it onto every rank.

    The reconstruct-and-reduce step runs on rank 0 only, immediately
    before every other rank is already waiting at the final
    ``broadcast`` -- the case :meth:`cumsum`'s equivalent guarding
    note describes as most dangerous to leave unguarded, since a
    rank-0-only failure there (e.g. an ``xr.concat`` dtype mismatch)
    would otherwise leave every other rank blocked forever. Guarded
    the same way: the root's attempt is wrapped and any exception
    deferred, then every rank (root or not) calls ``raise_if_error``
    together so the failure -- if any -- is reported consistently
    everywhere instead of only on rank 0.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce. Unlike :meth:`~.core.MPIXarray.mean`,
        :meth:`~.core.MPIXarray.sum`, etc., only a single dimension is
        supported (not an iterable, ``None``, or ``...``).
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
        reattached for whichever partition dimension(s) survive ``dim``
        being reduced away, with no duplication: exactly one rank per
        distinct surviving-dimension range keeps the real result
        (the sub-communicator's own rank 0, from
        :func:`~.cartesian.dim_comm`); every other rank that shared
        that same range before the reduction -- differing only along
        the now-reduced ``dim`` -- is left with a genuinely empty
        (``start == stop``) slice instead of a redundant copy, exactly
        like a rank :func:`~.chunks.get_balanced_bounds` already leaves
        idle when a dimension is shorter than the rank count.
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.median(dim, skipna=skipna, keep_attrs=keep_attrs)

    _agree(runtime, ("median", str(dim), int(meta["global_size"])))
    comm = _dim_comm(runtime, meta, dim)
    pieces = comm.gather(value, root=0)

    def _reduce_on_root() -> xr.Dataset | xr.DataArray:
        full = (
            xr.concat(pieces, dim=dim, data_vars="minimal")
            if isinstance(value, xr.Dataset)
            else xr.concat(pieces, dim=dim)
        )
        return full.median(dim, skipna=skipna, keep_attrs=keep_attrs)

    result, error = guarded(_reduce_on_root) if comm.rank == 0 else (None, None)
    runtime.raise_if_error(
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

    set_mpi_meta(
        result,
        dim=remaining_dims,
        global_size={d: int(meta["global_sizes"][d]) for d in remaining_dims},
        start=start,
        stop=stop,
        chunk_info=prune_chunk_info(meta["chunk_info"], result),
        cart=None,
    )
    return result


def diff(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    n: int = 1,
    *,
    label: Literal["upper", "lower"] = "upper",
) -> xr.Dataset | xr.DataArray:
    """``n``-th order difference along ``dim``, correct when ``dim`` is distributed.

    When ``dim`` is the active partition dimension: ``label="upper"``
    drops the global *first* ``n`` elements (xarray labels each
    difference with the later/"upper" of the two positions it came
    from), so every rank except rank 0 can compute its output at full
    local length by borrowing ``n`` elements from its left neighbor
    (:meth:`~.arithmetic.Arithmetic.halo_exchange`); rank 0 has no left
    neighbor and is genuinely ``n`` shorter, which is exactly where the
    global array actually lost those ``n`` elements. ``label="lower"``
    is the mirror image: drops the global *last* ``n``, borrows from
    the right neighbor instead, and only the last rank comes up short.
    Either way, every rank's new ``start``/``stop``/``global_size`` is
    then recomputed from an ``allgather`` of each rank's new local
    length -- the same mechanism :meth:`~.indexing.Indexing.isel`
    already uses for its own length-changing slice case.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to difference.
    dim : Hashable
        Dimension to difference along.
    n : int, optional
        Order of the difference. Must be less than every rank's local
        length along ``dim`` when ``dim`` is the partition dimension
        (see ``halo_exchange``'s own limit: a rank can only forward
        data it owns, so a wider request would need a multi-hop relay
        this does not perform).
    label : {"upper", "lower"}, optional
        As in ``xarray.DataArray.diff``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The differenced object, ``n`` elements shorter along ``dim``
        globally -- and, when ``dim`` is the partition dimension, at
        exactly one rank (0 for "upper", the last rank for "lower")
        locally; every other rank's local length is unchanged.

    Raises
    ------
    ValueError
        If ``n`` is negative, ``label`` is not "upper"/"lower", or any
        rank's local length along ``dim`` is shorter than ``n`` (this
        last case is caught by :meth:`~.arithmetic.Arithmetic.halo_exchange`
        itself, which checks every rank's local length together via a
        synchronized ``allgather`` before raising, so the error is
        consistent and every rank raises together rather than some
        hanging).
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        return value.diff(dim, n=n, label=label)
    if n < 0:
        raise ValueError(f"diff(): n must be >= 0, got {n!r}.")
    if label not in ("upper", "lower"):
        raise ValueError(f"diff(): label must be 'upper' or 'lower', got {label!r}.")
    if n == 0:
        return reattach_meta(value.diff(dim, n=0, label=label), meta)

    before, after = (n, 0) if label == "upper" else (0, n)
    padded, _left_pad, _right_pad = halo_exchange(
        runtime, value, dim, before=before, after=after
    )
    diffed = padded.diff(dim, n=n, label=label)

    comm = _dim_comm(runtime, meta, dim)
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
    set_mpi_meta(
        diffed,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=chunk_info,
        cart=meta.get("cart"),
    )
    return diffed


def shift(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    periods: int = 1,
    *,
    fill_value: Any = _UNSET,
) -> xr.Dataset | xr.DataArray:
    """Shift ``value`` by ``periods`` along ``dim``, correct when ``dim`` is distributed.

    A stencil operation: shifting by ``periods`` moves each position's
    value ``periods`` slots along ``dim`` without changing length, so a
    rank whose local window would otherwise pull in a neighbor's data
    (any rank except the one at the true edge the shift moves away
    from) needs ``|periods|`` boundary elements from that neighbor
    (:meth:`~.arithmetic.Arithmetic.halo_exchange`) before shifting
    locally. ``xarray.shift``'s own fill-value semantics fall out for
    free: shifting the *padded* array only introduces ``fill_value``
    at the padded array's own edges, and ``halo_exchange`` only leaves
    an edge unpadded (0 elements) at the true global boundary -- so a
    fill value appears exactly where the global array actually runs
    out of data, and nowhere else.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to shift.
    dim : Hashable
        Dimension to shift along.
    periods : int, optional
        Number of positions to shift by; positive shifts values toward
        higher indices (as in ``xarray.DataArray.shift``). Its
        magnitude must not exceed any rank's local length along
        ``dim`` when ``dim`` is the partition dimension (see
        ``halo_exchange``'s own limit).
    fill_value : Any, optional
        As in ``xarray.DataArray.shift``; defaults to xarray's own
        dtype-aware NA fill when omitted.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The shifted object, same shape and distribution as the input.
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        kwargs = {} if fill_value is _UNSET else {"fill_value": fill_value}
        return value.shift({dim: periods}, **kwargs)
    if periods == 0:
        return value

    before, after = (periods, 0) if periods > 0 else (0, -periods)
    padded, left_pad, _right_pad = halo_exchange(
        runtime, value, dim, before=before, after=after
    )
    kwargs = {} if fill_value is _UNSET else {"fill_value": fill_value}
    shifted = padded.shift({dim: periods}, **kwargs)

    local_len = int(value.sizes[dim])
    trimmed = shifted.isel({dim: slice(left_pad, left_pad + local_len)})
    return reattach_meta(trimmed, meta)


def roll(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    shift: int,
) -> xr.Dataset | xr.DataArray:
    """Circularly shift ``value`` by ``shift`` along ``dim``, wrapping at the edge.

    Same shift-amount and sign convention as :func:`shift` (and as
    ``xarray.DataArray.roll``: a positive ``shift`` moves each value
    toward higher indices) and built the same way -- borrow
    ``|shift|`` boundary elements from the neighbor via
    :func:`~.arithmetic.halo_exchange` and shift the padded local
    array -- but with ``periodic=True``, so the rank at the true
    global edge borrows from the rank at the *opposite* edge instead
    of getting no neighbor at all. That is the one and only
    difference from ``shift()``: once every rank's padding is real
    data (never "missing"), a plain windowed ``.shift()`` on the
    padded array already reproduces circular-roll semantics exactly,
    with no fill value anywhere -- mirroring FMS/``mpp_domains``,
    where periodicity is likewise just a boundary condition on a
    *bounded* halo exchange's neighbor lookup, not a distinct
    general-purpose data-movement primitive. Coordinates are not
    rolled (``roll_coords=False`` in ``xarray`` terms): under MPI,
    "which rank owns which global index" is fixed distribution
    metadata, not something a data-only circular shift should
    perturb.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to roll.
    dim : Hashable
        Dimension to roll along.
    shift : int
        Number of positions to roll by; positive rolls toward higher
        indices. Its magnitude must not exceed any rank's local
        length along ``dim`` when ``dim`` is the partition dimension
        (see ``halo_exchange``'s own single-hop limit) -- repartition
        to fewer, larger chunks first for a roll wider than that.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rolled object, same shape and distribution as the input.
    """
    meta = get_mpi_meta(value)
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
    padded, left_pad, _right_pad = halo_exchange(
        runtime, value, dim, before=before, after=after, periodic=True
    )
    shifted = padded.shift({dim: shift})

    local_len = int(value.sizes[dim])
    trimmed = shifted.isel({dim: slice(left_pad, left_pad + local_len)})
    # `.shift()` unconditionally reserves a float NaN fill value for the
    # boundary it introduces, upcasting any integer/bool variable to
    # float even though, by construction, that boundary is never
    # actually missing here: `halo_exchange(..., periodic=True)` already
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


def differentiate(
    runtime,
    value: xr.Dataset | xr.DataArray,
    coord: Hashable,
    edge_order: Literal[1, 2] = 1,
    datetime_unit: Any = None,
) -> xr.Dataset | xr.DataArray:
    """Differentiate ``value`` along ``coord``, correct when ``coord`` is distributed.

    A stencil operation: every interior point's derivative is a
    centered difference needing exactly one neighbor on each side
    regardless of ``edge_order`` (``edge_order`` only changes the
    one-sided formula used at the *true* global first/last position,
    which never needs another rank's data -- it is computed from
    this rank's own later/earlier local points exactly as plain
    ``xarray`` would). So a fixed one-element halo
    (:meth:`~.arithmetic.Arithmetic.halo_exchange`) on each side
    suffices for any ``edge_order``: it supplies genuine neighbor
    values at every rank boundary and, at the true global edge (where
    ``halo_exchange`` returns 0 padding), differentiating the
    unpadded-there array reproduces xarray's own edge-order-specific
    boundary stencil unchanged.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to differentiate.
    coord : Hashable
        Coordinate to differentiate along.
    edge_order : {1, 2}, optional
        As in ``xarray.DataArray.differentiate``. Default 1.
    datetime_unit : Any, optional
        As in ``xarray.DataArray.differentiate``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The derivative, same shape and distribution as the input.

    Raises
    ------
    ValueError
        If any rank's local length along ``coord`` is shorter than 1
        (see ``halo_exchange``'s own synchronized length check) or
        too short overall for ``edge_order`` (raised by xarray itself).
    """
    meta = get_mpi_meta(value)
    if meta is None or coord not in meta["dims"]:
        return value.differentiate(
            coord, edge_order=edge_order, datetime_unit=datetime_unit
        )

    padded, left_pad, _right_pad = halo_exchange(
        runtime, value, coord, before=1, after=1
    )
    # dask's gradient (unlike every other halo_exchange consumer -- shift,
    # diff, rolling_reduce, coarsen_reduce, ffill/bfill) requires every
    # individual chunk along the differentiated axis, not just the total
    # local length, to exceed edge_order + 1. halo_exchange's padding
    # arrives as its own separate, unconsolidated 1-element chunk (e.g.
    # local shape 125000 pads to chunks (125000, 1), not one (125002,)
    # chunk), which is too small on its own regardless of how large the
    # rank's real local data is. Consolidating to a single chunk here
    # only touches this local, already-fully-materialized-by-halo_exchange
    # piece -- it does not change halo_exchange's own chunking for any of
    # its other, unaffected callers.
    if padded.chunks:
        padded = padded.chunk({coord: -1})
    derivative = padded.differentiate(
        coord, edge_order=edge_order, datetime_unit=datetime_unit
    )

    local_len = int(value.sizes[coord])
    trimmed = derivative.isel({coord: slice(left_pad, left_pad + local_len)})
    return reattach_meta(trimmed, meta)
