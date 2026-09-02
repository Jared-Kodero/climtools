"""Provide MPI-aware alignment and arithmetic for distributed xarray objects."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from mpi4py import MPI

import xarray as xr

from .cartesian import dim_comm as _dim_comm
from .cartesian import get_cartesian_topology
from .chunks import get_balanced_bounds, prune_chunk_info
from .meta import _partitions_match, get_mpi_meta, set_mpi_meta, strip_mpi_meta
from .planning import _agree, comm_reduce, resolve_comm

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from ..mpi.runtime import MPIRuntime

# Callables apply() recognizes and transparently redirects to their
# dedicated implementation, so apply() is MPI-aware for them the same way
# evaluate() is: apply(operator.matmul, a, b) computes the same correct,
# MPI-reduced result as evaluate("a @ b", a=a, b=b) and matmul(a, b),
# instead of running the plain rank-local matmul and failing the post-call
# partition check whenever the distributed dimension gets contracted away.
_MATMUL_CALLABLES: frozenset[Callable[..., Any]] = frozenset(
    {operator.matmul, np.matmul}
)
# ast.MatMult ('@') is deliberately absent: whether matrix multiplication is
# rank-local depends on which dimension gets contracted, so it is routed to
# the dedicated Arithmetic.matmul() implementation in _eval_ast_node()
# instead of the generic apply(operator.matmul, ...) table below.
_AST_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

_AST_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_AST_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}


_AST_BOOL_OPS: dict[type[ast.boolop], Callable[[list[Any]], Any]] = {
    ast.And: lambda values: all(values),
    ast.Or: lambda values: any(values),
}


def _fill_chunk(
    template: xr.Dataset | xr.DataArray, dim: str, n: int, fill_value: Any
) -> xr.Dataset | xr.DataArray:
    """Build an ``n``-long, all-``fill_value`` chunk along ``dim``."""
    shaped = template.isel({dim: [0] * n})
    if isinstance(shaped, xr.Dataset):
        filled = shaped.copy(deep=False)
        for name, var in shaped.data_vars.items():
            if dim in var.dims:
                filled[name] = xr.full_like(
                    var,
                    fill_value,
                    # Pass fill_value itself (not np.asarray(fill_value).dtype)
                    # so NumPy's value-based scalar promotion applies: a
                    # Python-float nan against float32 stays float32 (nan is
                    # representable), matching native xr.Dataset.reindex.
                    # np.asarray(nan).dtype is float64, which would force
                    # every float32 variable up to float64 regardless of
                    # whether the fill value actually needs the extra
                    # precision/range -- exactly the promotion this
                    # implementation must not introduce.
                    dtype=np.result_type(var.dtype, fill_value),
                )
        return filled
    return xr.full_like(
        shaped,
        fill_value,
        # See the matching comment in the Dataset branch above: use
        # fill_value's value-based promotion, not its array dtype.
        dtype=np.result_type(shaped.dtype, fill_value),
    )


def _haloed_variable_names(
    value: xr.Dataset | xr.DataArray, partition_dim: Hashable
) -> tuple[Hashable, ...]:
    """Return variables that vary along ``partition_dim``."""
    if isinstance(value, xr.Dataset):
        return tuple(
            name for name, var in value.variables.items() if partition_dim in var.dims
        )
    names = [value.name] if partition_dim in value.dims else []
    names.extend(
        name
        for name, coord in value.coords.items()
        if partition_dim in coord.dims and name != value.name
    )
    return tuple(names)


def _exchange_halo_blocks(
    value: xr.Dataset | xr.DataArray,
    partition_dim: Hashable,
    before: int,
    after: int,
    *,
    comm: MPI.Comm,
    local_len: int,
    left_rank: int | None,
    right_rank: int | None,
) -> tuple[xr.Dataset | xr.DataArray | None, xr.Dataset | xr.DataArray | None]:
    """Exchange boundary slabs with adjacent ranks using nonblocking MPI buffers."""
    haloed = _haloed_variable_names(value, partition_dim)

    def _local_array(name: Hashable) -> np.ndarray:
        """Return a contiguous local array for one variable."""
        if isinstance(value, xr.Dataset):
            return value[name].variable
        if name == value.name:
            return value.variable
        return value.coords[name].variable

    def _mpi_buffer_view(arr: np.ndarray) -> np.ndarray:
        """View ``arr`` as a dtype the raw MPI buffer protocol accepts."""
        return arr.view(np.int64) if arr.dtype.kind in "mM" else arr

    # This rank's own boundary slabs to send: the tail (size `before`) goes
    # to right_rank, who will use it as their own "before" padding; the
    # head (size `after`) goes to left_rank, who will use it as their own
    # "after" padding -- symmetric with what this rank expects back below.
    #
    # Guarded on before>0/after>0, matching the receive side's guard
    # exactly (not just on left_rank/right_rank being non-None): every
    # rank agrees on the same before/after (see _agree() in the caller),
    # so before==0 means *no* rank anywhere posts a recv_before this
    # call. An unconditional send here would still fire a zero-byte
    # Isend with no matching Irecv anywhere in this call -- MPI does not
    # require a receive to exist for a send to "complete" locally, so
    # that stray message is left unmatched and queued for the next
    # message from the same (source, tag). The very next call that
    # *does* request a halo on this axis then has its own legitimate
    # Irecv silently satisfied by that old, empty, unrelated message
    # instead of the new data -- a real cross-call corruption bug,
    # confirmed by reproduction: a `before=0` or `after=0` call
    # immediately followed by one with a nonzero request on the same
    # side reads back an uninitialized (`np.empty`) buffer, not the
    # true neighbor data.
    send_to_right: dict[Hashable, np.ndarray] = {}
    send_to_left: dict[Hashable, np.ndarray] = {}
    for name in haloed:
        var = _local_array(name)
        axis = var.dims.index(partition_dim)
        if right_rank is not None and before > 0:
            send_to_right[name] = _mpi_buffer_view(
                np.ascontiguousarray(
                    var.isel(
                        {partition_dim: slice(local_len - before, local_len)}
                    ).values
                )
            )
        if left_rank is not None and after > 0:
            send_to_left[name] = _mpi_buffer_view(
                np.ascontiguousarray(var.isel({partition_dim: slice(0, after)}).values)
            )

    recv_before: dict[Hashable, np.ndarray] = {}
    recv_after: dict[Hashable, np.ndarray] = {}
    recv_before_bufs: dict[Hashable, np.ndarray] = {}
    recv_after_bufs: dict[Hashable, np.ndarray] = {}
    if left_rank is not None and before > 0:
        for name in haloed:
            var = _local_array(name)
            axis = var.dims.index(partition_dim)
            shape = list(var.shape)
            shape[axis] = before
            recv_before[name] = np.empty(shape, dtype=var.dtype)
            recv_before_bufs[name] = _mpi_buffer_view(recv_before[name])
    if right_rank is not None and after > 0:
        for name in haloed:
            var = _local_array(name)
            axis = var.dims.index(partition_dim)
            shape = list(var.shape)
            shape[axis] = after
            recv_after[name] = np.empty(shape, dtype=var.dtype)
            recv_after_bufs[name] = _mpi_buffer_view(recv_after[name])

    recv_reqs: list[MPI.Request] = [
        comm.Irecv(buf, source=left_rank) for buf in recv_before_bufs.values()
    ] + [comm.Irecv(buf, source=right_rank) for buf in recv_after_bufs.values()]
    send_reqs: list[MPI.Request] = [
        comm.Isend(arr, dest=right_rank) for arr in send_to_right.values()
    ] + [comm.Isend(arr, dest=left_rank) for arr in send_to_left.values()]

    MPI.Request.Waitall(recv_reqs)
    MPI.Request.Waitall(send_reqs)

    def _reconstruct(
        received: dict[Hashable, np.ndarray],
    ) -> xr.Dataset | xr.DataArray | None:
        """Reconstruct an xarray object from exchanged arrays."""
        if not received:
            return None
        if isinstance(value, xr.Dataset):
            pieces = {}
            for name, var in value.variables.items():
                if name in received:
                    pieces[name] = xr.Variable(
                        var.dims, received[name], attrs=var.attrs
                    )
                else:
                    pieces[name] = var
            return xr.Dataset(pieces, attrs=value.attrs)
        data_var = xr.Variable(value.dims, received[value.name], attrs=value.attrs)
        new_coords = {}
        for coord_name, coord in value.coords.items():
            if coord_name in received:
                new_coords[coord_name] = xr.Variable(
                    coord.dims, received[coord_name], attrs=coord.attrs
                )
            else:
                new_coords[coord_name] = coord.variable
        return xr.DataArray(data_var, coords=new_coords, name=value.name)

    return _reconstruct(recv_before), _reconstruct(recv_after)


def _gather_full(
    runtime: MPIRuntime, value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]
) -> xr.Dataset | xr.DataArray:
    """Reconstruct ``value``'s full, replicated extent on every rank."""
    dim = meta["dim"]
    if len(meta["dims"]) > 1:
        raise NotImplementedError(
            "align() cannot yet gather a multi-dimensionally partitioned "
            + f"object onto every rank (dims={meta['dims']!r}); this is a "
            + "genuine structural redistribution across a Cartesian "
            + "process grid, not yet implemented. Reduce or reconcile "
            + "one of the partition dimensions first, or align operands "
            + "sharing an identical partition (see _partitions_match), "
            + "which needs no data movement and is unaffected by this."
        )
    pieces = runtime.comm.allgather(value)
    full = (
        xr.concat(pieces, dim=dim, data_vars="minimal")
        if isinstance(value, xr.Dataset)
        else xr.concat(pieces, dim=dim)
    )
    return strip_mpi_meta(full)


def _align_replicated(
    runtime: MPIRuntime,
    other: Any,
    meta: dict[str, Any],
    partner: xr.Dataset | xr.DataArray | None = None,
) -> Any:
    """Slice a replicated operand onto an already-distributed partner's bounds."""
    if not isinstance(other, (xr.Dataset, xr.DataArray)):
        return other
    shared_dims = tuple(dim for dim in meta["dims"] if dim in other.dims)
    if not shared_dims:
        return other

    indexers: dict[Hashable, slice] = {}
    for dim in shared_dims:
        length = int(other.sizes[dim])
        global_size = int(meta["global_sizes"][dim])
        if length != global_size:
            raise ValueError(
                f"Cannot align: operand carries dimension {dim!r} at length "
                + f"{length}, but the distributed partner's global size "
                + f"along {dim!r} is {global_size}. align() only slices a "
                + "replicated (full-length) operand onto an existing "
                + "partition; lengths must match the whole distributed "
                + "dimension."
            )
        indexers[dim] = slice(meta["starts"][dim], meta["stops"][dim])
    sliced = other.isel(indexers)

    if partner is not None:
        for dim in shared_dims:
            if dim not in getattr(partner, "indexes", {}) or dim not in getattr(
                sliced, "indexes", {}
            ):
                continue
            try:
                xr.align(partner, sliced, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Cannot align: the replicated operand's {dim!r} labels "
                    + "do not match the distributed partner's labels for "
                    + "this rank's slice, even though both have length "
                    + f"{meta['stops'][dim] - meta['starts'][dim]}. "
                    + f"xarray.align(..., join='exact') reports: {exc}"
                ) from exc

    return reattach_meta(sliced, meta)


def align(
    runtime: MPIRuntime,
    left: xr.Dataset | xr.DataArray,
    right: xr.Dataset | xr.DataArray,
    dim: Hashable | Literal["auto"] | None = None,
    *,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
    """Return ``(left, right)`` partitioned identically across ranks.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    left : xarray.Dataset or xarray.DataArray
        Left operand to align.
    right : xarray.Dataset or xarray.DataArray
        Right operand to align.
    dim : hashable or {"auto"}, optional
        Dimension to partition both operands along when neither is currently distributed, or the shared dimension to reconcile onto when both are already distributed differently.
    chunk_info : mapping, optional
        Forwarded to ``repartition``.
    log_partitions : bool, optional
        Forwarded to ``repartition``.
    Returns
    -------
    tuple of xarray.Dataset or xarray.DataArray
        ``(left, right)``, each carrying matching distribution metadata (or neither carrying any, if both remain replicated).

    Raises
    ------
    ValueError
        If neither operand is distributed and ``dim`` is omitted.
    """
    from .io import repartition

    left_meta = _operand_meta(left)
    right_meta = _operand_meta(right)

    if left_meta is not None and right_meta is not None:
        if _partitions_match(left_meta, right_meta):
            return left, right
        target_dim = dim if dim is not None else left_meta["dim"]
        full_left = _gather_full(runtime, left, left_meta)
        full_right = _gather_full(runtime, right, right_meta)
        return (
            repartition(
                runtime,
                full_left,
                target_dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            repartition(
                runtime,
                full_right,
                target_dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
        )

    if left_meta is not None:
        return left, _align_replicated(runtime, right, left_meta, partner=left)

    if right_meta is not None:
        return _align_replicated(runtime, left, right_meta, partner=right), right

    if dim is None:
        return left, right

    if (
        isinstance(left, (xr.Dataset, xr.DataArray))
        and isinstance(right, (xr.Dataset, xr.DataArray))
        and dim in getattr(left, "indexes", {})
        and dim in getattr(right, "indexes", {})
    ):
        try:
            xr.align(left, right, join="exact")
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Cannot align: left and right disagree on {dim!r} "
                + "coordinate labels, so distributing each "
                + "independently would silently combine mismatched "
                + f"slices. xarray.align(..., join='exact') reports: {exc}"
            ) from exc

    return (
        repartition(
            runtime, left, dim, chunk_info=chunk_info, log_partitions=log_partitions
        ),
        repartition(
            runtime, right, dim, chunk_info=chunk_info, log_partitions=log_partitions
        ),
    )


#
# reindex() and sortby() are xarray's own coordinate-label operations
# (unlike align() above, which reconciles rank *ownership* rather than
# labels) -- either can move an element to a different rank whenever the
# partition dimension itself is reindexed/reordered. Routing follows the
# same "local unless communication is structurally required" rule as
# everywhere else in this module: if none of the touched dimensions are
# currently partitioned, native xarray runs rank-locally and ownership
# is provably unaffected (see _local_reduction_meta's identical
# reasoning for reductions).
#
# When the partition dimension itself IS touched, this does a genuine
# personalized shuffle (see _shuffle_by_position below), not a full
# MPI_Allgather: only the (small) coordinate/key values along `dim` are
# ever gathered onto every rank -- ~O(global_size(dim)) numbers, the
# same order of magnitude bookkeeping already costs elsewhere in this
# package -- never the bulk data. Every rank then redundantly computes
# the identical global new-position -> old-position mapping from those
# small gathered arrays (cheap, no further communication needed to
# agree on it), and the bulk payload moves rank-to-rank with
# point-to-point, non-blocking sends: exactly one message per (source,
# destination) pair that actually has data to move, receives posted
# before sends (the same order FMS's mpp_do_update_ posts them in),
# and no message at all for a rank's own self-contribution or for
# newly-filled positions. Peak memory on any one rank is its own old
# local slice plus its own new local slice -- never the global array --
# so this scales to a partition dimension far larger than any single
# rank could hold, unlike a full gather.


def _shuffle_by_position(
    runtime: MPIRuntime,
    value: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    dim: str,
    *,
    new_coord: np.ndarray[Any, Any],
    old_pos: np.ndarray[Any, Any],
    fill_value: Any,
) -> xr.Dataset | xr.DataArray:
    """Redistribute ``value`` along ``dim`` to match ``old_pos``."""
    comm = _dim_comm(runtime, meta, dim)
    rank, size = comm.rank, comm.size

    old_start = int(meta["starts"][dim])
    old_stop = int(meta["stops"][dim])
    old_starts, _old_stops = zip(*comm.allgather((old_start, old_stop)), strict=True)
    old_starts_arr = np.asarray(old_starts, dtype=np.int64)

    new_length = int(new_coord.shape[0])
    new_starts_all = np.fromiter(
        (get_balanced_bounds(new_length, r, size)[0] for r in range(size)),
        dtype=np.int64,
        count=size,
    )
    new_start, new_stop = get_balanced_bounds(new_length, rank, size)

    def _owner_of(
        global_positions: np.ndarray[Any, Any], starts: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Return the rank owning a global position."""
        return np.searchsorted(starts, global_positions, side="right") - 1

    owned_mask = (old_pos >= old_start) & (old_pos < old_stop)
    p_owned = np.nonzero(owned_mask)[0]  # ascending new positions I feed
    g_owned = old_pos[p_owned]  # corresponding old global positions (mine)
    dest_of_p_owned = _owner_of(p_owned, new_starts_all)

    self_payload: xr.Dataset | xr.DataArray | None = None
    send_requests: list[MPI.Request] = []

    # What I must receive, computed independently but symmetrically
    # with every source's own view of the same global mapping above --
    # see the module note: both sides iterate the same predicate over
    # the same universe in the same ascending order, so no metadata
    # about position/order needs to travel alongside the payload.
    my_local_p = np.arange(new_start, new_stop, dtype=np.int64)
    my_local_g = old_pos[new_start:new_stop] if new_length > 0 else my_local_p
    my_is_fill = my_local_g == -1
    my_owner = np.full(my_local_p.shape, -1, dtype=np.int64)
    if (~my_is_fill).any():
        my_owner[~my_is_fill] = _owner_of(my_local_g[~my_is_fill], old_starts_arr)

    incoming_sources = sorted(
        {int(s) for s in np.unique(my_owner) if s >= 0 and s != rank}
    )

    for dest in range(size):
        mask = dest_of_p_owned == dest
        if not mask.any():
            continue
        local_old_idx = g_owned[mask] - old_start
        payload = value.isel({dim: local_old_idx})
        if dest == rank:
            self_payload = payload
        else:
            send_requests.append(comm.isend(payload, dest=dest))

    # Blocking recv, not irecv: mpi4py's pickle irecv needs an accurate
    # buffer-size guess up front and can silently corrupt memory once
    # a payload exceeds it (exactly the risk a genuinely large,
    # OOM-motivated shuffle would hit). recv() self-sizes via an
    # internal probe first, so it stays correct at any payload size;
    # the isend side above needs no such guess, since the sender
    # already knows its own pickled size exactly. This trades away
    # "receives posted before sends" latency-hiding for that
    # correctness guarantee -- sends are still posted non-blocking so
    # this rank's own sends never stall waiting on a slow receiver.
    received = {source: comm.recv(source=source) for source in incoming_sources}
    MPI.Request.Waitall(send_requests)

    if new_stop <= new_start:
        empty = value.isel({dim: slice(0, 0)})
        result = empty.assign_coords({dim: new_coord[new_start:new_stop]})
    else:
        pieces: list[xr.Dataset | xr.DataArray] = []
        slot_pieces: list[np.ndarray[Any, Any]] = []

        self_mask = my_owner == rank
        if self_mask.any():
            if self_payload is None:
                raise AssertionError(
                    "Shuffle planning mismatch: expected a "
                    + "self-contribution that was never built."
                )
            pieces.append(self_payload)
            slot_pieces.append(np.nonzero(self_mask)[0])

        if my_is_fill.any():
            n_fill = int(my_is_fill.sum())
            pieces.append(_fill_chunk(value, dim, n_fill, fill_value))
            slot_pieces.append(np.nonzero(my_is_fill)[0])

        for source in incoming_sources:
            mask = my_owner == source
            pieces.append(received[source])
            slot_pieces.append(np.nonzero(mask)[0])

        combined = (
            xr.concat(pieces, dim=dim, data_vars="minimal")
            if isinstance(value, xr.Dataset)
            else xr.concat(pieces, dim=dim)
        )
        slots = np.concatenate(slot_pieces)
        final_order = np.argsort(slots, kind="stable")
        result = combined.isel({dim: final_order})
        result = result.assign_coords({dim: new_coord[new_start:new_stop]})

    result = strip_mpi_meta(result)
    chunk_info = prune_chunk_info(meta["chunk_info"], result)
    remaining_dims = tuple(d for d in meta["dims"] if d != dim)
    if remaining_dims:
        all_dims = meta["dims"]
        global_size = {d: int(meta["global_sizes"][d]) for d in remaining_dims}
        start = {d: int(meta["starts"][d]) for d in remaining_dims}
        stop = {d: int(meta["stops"][d]) for d in remaining_dims}
        global_size[dim] = new_length
        start[dim] = new_start
        stop[dim] = new_stop
        set_mpi_meta(
            result,
            dim=all_dims,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info=chunk_info,
            cart=meta.get("cart"),
        )
    else:
        set_mpi_meta(
            result,
            dim=dim,
            global_size=new_length,
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
    return result


def reindex(
    runtime: MPIRuntime,
    value: xr.Dataset | xr.DataArray,
    indexers: Mapping[Hashable, Any] | None = None,
    *,
    method: str | None = None,
    tolerance: float | Iterable[float] | None = None,
    fill_value: Any = np.nan,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
    **indexers_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Reindex ``value`` onto new coordinate labels, redistributing if needed.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reindex; distributed or replicated.
    indexers : mapping, optional
        New coordinate labels per dimension, exactly as ``xarray.Dataset.reindex``/``DataArray.reindex`` accepts.
    method : str, optional
        Forwarded to ``pandas.Index.get_indexer`` when the partition dimension is reindexed (``None``, ``"nearest"``, ``"ffill"``/ ``"pad"``, ``"bfill"``/``"backfill"``); forwarded to xarray's own ``reindex`` otherwise.
    tolerance : float or iterable of float, optional
        Forwarded to ``pandas.Index.get_indexer``/xarray's ``reindex``.
    fill_value : Any, optional
        Value used for labels with no match in ``value``.
    chunk_info : mapping, optional
        Reserved for parity with ``repartition``'s signature; not consulted by the redistributing path, which always balances.
    log_partitions : bool, optional
        Currently unused by the redistributing path.
    **indexers_kwargs : Any
        Additional indexers given as keywords, merged with ``indexers``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The reindexed object: rank-local (metadata preserved) if no partitioned dimension was touched; freshly, memory-scalably redistributed (new bounds, possibly a new global length) otherwise -- see ``_shuffle_by_position``.

    Raises
    ------
    ValueError
        If no indexers are given.
    NotImplementedError
        If more than one active partition dimension is reindexed at once, or a reindexed partition dimension's new coordinate is not one-dimensional.
    """
    indexers = {**(indexers or {}), **indexers_kwargs}
    if not indexers:
        raise ValueError("reindex() requires at least one indexer.")

    meta = get_mpi_meta(value)
    if meta is None:
        return value.reindex(
            indexers, method=method, tolerance=tolerance, fill_value=fill_value
        )

    partition_dims = meta["dims"]
    touched = tuple(str(d) for d in partition_dims if d in indexers)

    if not touched:
        result = strip_mpi_meta(value).reindex(
            indexers, method=method, tolerance=tolerance, fill_value=fill_value
        )
        set_mpi_meta(
            result,
            dim=meta["dims"],
            global_size=meta["global_sizes"],
            start=meta["starts"],
            stop=meta["stops"],
            chunk_info=prune_chunk_info(meta["chunk_info"], result),
            cart=meta.get("cart"),
        )
        return result

    if len(touched) > 1:
        raise NotImplementedError(
            "reindex() cannot yet redistribute more than one active "
            + f"partition dimension at once (touched={touched!r}); each "
            + "call may reindex at most one of them. Reindexing a "
            + "single active partition dimension already works, even "
            + "under a multi-dimensional partition, as does reindexing "
            + "any dimension that is not currently partitioned."
        )

    dim = touched[0]
    new_labels = np.asarray(indexers[dim])
    if new_labels.ndim != 1:
        raise NotImplementedError(
            f"reindex(): the new {dim!r} labels must be one-dimensional "
            + f"to redistribute; got shape {new_labels.shape!r}."
        )
    _agree(
        runtime,
        (
            "reindex",
            dim,
            int(new_labels.shape[0]),
            str(method),
            str(tolerance),
        ),
    )

    comm = _dim_comm(runtime, meta, dim)
    old_coord_local = np.asarray(value[dim].values)
    old_full_coord = np.concatenate(comm.allgather(old_coord_local))
    old_index = pd.Index(old_full_coord)
    old_pos = old_index.get_indexer(new_labels, method=method, tolerance=tolerance)
    old_pos = old_pos.astype(np.int64)

    return _shuffle_by_position(
        runtime,
        value,
        meta,
        dim,
        new_coord=new_labels,
        old_pos=old_pos,
        fill_value=fill_value,
    )


def sortby(
    runtime: MPIRuntime,
    value: xr.Dataset | xr.DataArray,
    by: Hashable | xr.DataArray | Sequence[Hashable | xr.DataArray],
    *,
    ascending: bool = True,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Sort ``value`` by one or more keys, redistributing if needed.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to sort; distributed or replicated.
    by : Hashable, DataArray, or sequence of these
        Sort key(s): variable/coordinate name(s) or explicit DataArray(s), exactly as ``xarray.Dataset.sortby``/ ``DataArray.sortby`` accepts.
    ascending : bool, optional
        Sort order.
    chunk_info : mapping, optional
        Reserved for parity with ``repartition``'s signature; not consulted by the redistributing path, which always balances.
    log_partitions : bool, optional
        Currently unused by the redistributing path.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The sorted object: rank-local (metadata preserved) if no sort key varies along a partitioned dimension; freshly, memory-scalably redistributed otherwise -- see ``_shuffle_by_position``.

    Raises
    ------
    NotImplementedError
        If the sort key(s) together vary along more than one active partition dimension under a multi-dimensional (Cartesian) partition, or a key is not one-dimensional along the partition dimension it varies along.
    """
    meta = get_mpi_meta(value)
    if meta is None:
        return value.sortby(by, ascending=ascending)

    keys = list(by) if isinstance(by, (list, tuple)) else [by]
    touched_dims: set[str] = set()
    for key in keys:
        if isinstance(key, xr.DataArray):
            touched_dims.update(str(d) for d in key.dims)
            continue
        try:
            touched_dims.update(str(d) for d in value[key].dims)
        except (KeyError, TypeError):
            continue

    partition_dims = meta["dims"]
    touched = tuple(str(d) for d in partition_dims if d in touched_dims)

    if not touched:
        result = strip_mpi_meta(value).sortby(by, ascending=ascending)
        set_mpi_meta(
            result,
            dim=meta["dims"],
            global_size=meta["global_sizes"],
            start=meta["starts"],
            stop=meta["stops"],
            chunk_info=prune_chunk_info(meta["chunk_info"], result),
            cart=meta.get("cart"),
        )
        return result

    if len(touched) > 1:
        raise NotImplementedError(
            "sortby() cannot yet redistribute when the sort key(s) "
            + "together vary along more than one active partition "
            + f"dimension ({touched!r}); each key must vary along at "
            + "most one of them. Sorting by a key that varies along a "
            + "single active partition dimension already works, even "
            + "under a multi-dimensional partition."
        )

    dim = touched[0]
    local_len = int(value.sizes[dim])
    key_arrays_local: list[np.ndarray[Any, Any]] = []
    for key in keys:
        arr = np.asarray(
            key.values if isinstance(key, xr.DataArray) else value[key].values
        )
        if arr.ndim != 1 or arr.shape[0] != local_len:
            raise NotImplementedError(
                f"sortby(): key {key!r} is not one-dimensional along the "
                + f"partition dimension {dim!r} (shape {arr.shape!r} vs. "
                + f"local length {local_len!r}); redistribution needs "
                + "every key to give exactly one sort value per element "
                + "along the partition dimension."
            )
        key_arrays_local.append(arr)

    key_signature = tuple(
        "<dataarray>" if isinstance(key, xr.DataArray) else str(key) for key in keys
    )
    _agree(runtime, ("sortby", dim, key_signature, bool(ascending)))

    comm = _dim_comm(runtime, meta, dim)
    full_keys = [np.concatenate(comm.allgather(arr)) for arr in key_arrays_local]
    old_full_coord = np.concatenate(comm.allgather(np.asarray(value[dim].values)))
    # np.lexsort sorts by the *last* array primarily; reverse so the
    # first key in `by` is primary, matching xarray.sortby's own order.
    order = np.lexsort(tuple(reversed(full_keys)))
    if not ascending:
        order = order[::-1]
    old_pos = order.astype(np.int64)
    new_coord = old_full_coord[order]

    return _shuffle_by_position(
        runtime,
        value,
        meta,
        dim,
        new_coord=new_coord,
        old_pos=old_pos,
        fill_value=np.nan,
    )




def _operand_meta(operand: Any) -> dict[str, Any] | None:
    """Return ``operand``'s MPI distribution metadata, if any."""
    if isinstance(operand, (xr.Dataset, xr.DataArray)):
        return get_mpi_meta(operand)
    return None


def reattach_meta(result: Any, meta: dict[str, Any]) -> Any:
    """Tag ``result`` with ``meta`` if it is an xarray object.

    Parameters
    ----------
    result : Any
        The computation result to be tagged.
    meta : dict[str, Any]
        The distribution metadata dictionary to reattach.
    Returns
    -------
    Any
        The tagged result object if it is an xarray dataset or dataarray, otherwise returned unmodified.
    """
    if isinstance(result, (xr.Dataset, xr.DataArray)):
        set_mpi_meta(
            result,
            dim=meta["dims"],
            global_size=meta["global_sizes"],
            start=meta["starts"],
            stop=meta["stops"],
            chunk_info=meta["chunk_info"],
            cart=meta.get("cart"),
        )
    return result


def check_operands_distribution(
    runtime: MPIRuntime, operands: Iterable[Any]
) -> tuple[dict[str, Any] | None, Any]:
    """Return the mpi_meta to attach to a multi-operand call's result.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    operands : iterable of Any
        Every positional and keyword argument passed to :meth:`apply`.
    Returns
    -------
    tuple[dict[str, Any] | None, Any]
        ``(meta, reference)``: metadata to reattach to the result (or None when no operand is distributed) together with the first distributed operand itself, used by :meth:`apply` as the coordinate baseline for post-call validation.

    Raises
    ------
    ValueError
        If two operands are distributed over different partitions, if a replicated operand carries the distributed dimension at a different length than the partition owns, if a replicated operand's coordinate labels along the distributed dimension do not match the distributed partition's labels for this rank's slice (equal length alone does not imply equal coordinates), or (on more than one rank) if that coordinate check cannot even run because either side has no coordinate for the distributed dimension -- equal length alone is not enough evidence the operand is genuinely this rank's own data rather than another rank's same-length slice by coincidence.
    """
    operands = list(operands)
    metas = [_operand_meta(item) for item in operands]

    ref_index = next((i for i, item in enumerate(metas) if item is not None), None)
    if ref_index is None:
        return None, None
    meta = metas[ref_index]
    reference = operands[ref_index]

    for other, other_meta in zip(operands, metas, strict=True):
        if other_meta is not None:
            if not _partitions_match(meta, other_meta):
                raise ValueError(
                    "Cannot combine operands distributed over "
                    + f"different partitions: dims={meta['dims']!r} "
                    + f"bounds={ {d: (meta['starts'][d], meta['stops'][d]) for d in meta['dims']} } vs "
                    + f"dims={other_meta['dims']!r} "
                    + f"bounds={ {d: (other_meta['starts'][d], other_meta['stops'][d]) for d in other_meta['dims']} }. "
                    + "Call align(...) first."
                )
            continue

        for dim in meta["dims"]:
            if not (
                isinstance(other, (xr.Dataset, xr.DataArray)) and dim in other.dims
            ):
                continue
            owned = meta["stops"][dim] - meta["starts"][dim]
            local = int(other.sizes[dim])
            if local != owned:
                raise ValueError(
                    f"Operand carries dimension {dim!r} at length "
                    + f"{local}, which does not match this rank's "
                    + f"owned partition length {owned}. Call "
                    + "align(...) first."
                )
            reference_indexed = dim in getattr(reference, "indexes", {})
            other_indexed = dim in getattr(other, "indexes", {})
            if reference_indexed and other_indexed:
                try:
                    xr.align(reference, other, join="exact")
                except (ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Operand carries dimension {dim!r} at the "
                        + f"expected local length ({owned}), but its "
                        + "coordinate labels do not match the "
                        + "distributed partition's labels for this "
                        + "rank's slice; equal length does not imply "
                        + "equal coordinates. Call align(...) "
                        + "first. xarray.align(..., join='exact') "
                        + f"reports: {exc}"
                    ) from exc
            elif runtime.comm.size > 1:
                # Equal length is necessary but not sufficient: without a
                # coordinate on dim to check exactly (the branch above),
                # there is no way to tell this rank's own correctly
                # aligned slice apart from, say, a different rank's
                # slice of the same length -- a silently wrong answer
                # that would otherwise pass unnoticed. Refuse rather
                # than trust length alone once more than one rank makes
                # that ambiguity possible.
                missing = [
                    name
                    for name, indexed in (
                        ("the distributed side", reference_indexed),
                        ("the operand", other_indexed),
                    )
                    if not indexed
                ]
                raise ValueError(
                    f"Operand carries dimension {dim!r} at the "
                    + f"expected local length ({owned}), but its "
                    + "alignment with this rank's own owned slice "
                    + f"cannot be verified: {' and '.join(missing)} "
                    + f"has no coordinate for {dim!r}, so equal length "
                    + "alone is not enough evidence this is actually "
                    + "this rank's own data rather than, say, another "
                    + "rank's same-length slice by coincidence. Add a "
                    + f"coordinate for {dim!r} to both sides so it can "
                    + "be checked exactly, or build the operand from "
                    + "the distributed side directly (e.g. via "
                    + "isel()/apply() on it) instead of a separately "
                    + "constructed array."
                )
    return meta, reference


def check_partition_preserved(
    result: Any, meta: Mapping[str, Any], reference: Any
) -> None:
    """Verify ``result`` still owns the same partition-dimension slice.

    Parameters
    ----------
    result : Any
        The value returned by the callable.
    meta : Mapping[str, Any]
        The distribution metadata captured before the call.
    reference : Any
        The distributed operand the metadata was taken from, used as the coordinate baseline for the label check below.
    Raises
    ------
    ValueError
        If the distributed dimension is missing from ``result``, its local length changed, or its coordinate labels no longer match this rank's owned interval.
    """
    if not isinstance(result, (xr.Dataset, xr.DataArray)):
        return

    for dim in meta["dims"]:
        owned = meta["stops"][dim] - meta["starts"][dim]

        if dim not in result.dims:
            raise ValueError(
                "apply(): the callable removed or renamed the distributed "
                + f"dimension {dim!r} (result dims: {tuple(result.dims)!r}). "
                + "apply() only supports partition-preserving rank-local "
                + "callables; use the corresponding mpi.xarray reduction, "
                + "indexing, or groupby method for operations that change "
                + "the partition dimension."
            )

        local = int(result.sizes[dim])
        if local != owned:
            raise ValueError(
                "apply(): the callable changed the local length of the "
                + f"distributed dimension {dim!r} from {owned} to {local} "
                + "on this rank. apply() only supports partition-preserving "
                + "rank-local callables that leave every rank's owned "
                + "slice the same length; operations such as slicing, "
                + "dropping, or windowed reductions along the partition "
                + "dimension require values from neighboring ranks and "
                + "must not be done inside apply()."
            )

        if (
            isinstance(reference, (xr.Dataset, xr.DataArray))
            and dim in getattr(reference, "indexes", {})
            and dim in getattr(result, "indexes", {})
        ):
            try:
                xr.align(reference, result, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"apply(): the callable changed the {dim!r} coordinate "
                    + "labels on this rank, even though the local length "
                    + f"({local}) is unchanged. apply() only supports "
                    + "partition-preserving rank-local callables that leave "
                    + f"each rank's owned {dim!r} interval untouched. "
                    + f"xarray.align(..., join='exact') reports: {exc}"
                ) from exc


def apply(
    runtime: MPIRuntime, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    func : callable
        Any partition-preserving, rank-local function of the given ``args`` and ``kwargs``.
    *args : Any
        Positional arguments to ``func``: xarray Datasets or DataArrays (distributed or not) or plain scalars and arrays, in any mix.
    **kwargs : Any
        Keyword arguments to ``func``, checked for distribution metadata exactly like ``args``.
    Returns
    -------
    Any
        The result of ``func(*args, **kwargs)``.

    Raises
    ------
    ValueError
        If the xarray arguments are distributed over incompatible partitions or their coordinates disagree, or if the callable's result no longer represents the same owned partition (missing dimension, changed local length, or changed coordinate labels).
    """
    if func in _MATMUL_CALLABLES and not kwargs and len(args) == 2:
        return matmul(runtime, *args)

    return _apply_generic(runtime, func, args, kwargs)


def _apply_generic(
    runtime: MPIRuntime, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Run the shared partition-preserving callable path."""
    meta, reference = check_operands_distribution(runtime, (*args, *kwargs.values()))

    _agree(
        runtime,
        (
            "apply",
            getattr(func, "__name__", repr(func)),
            None
            if meta is None
            else (
                tuple(str(d) for d in meta["dims"]),
                tuple(int(meta["global_sizes"][d]) for d in meta["dims"]),
            ),
        ),
    )

    result = func(*args, **kwargs)
    if meta is None:
        return result
    check_partition_preserved(result, meta, reference)
    return reattach_meta(result, meta)


#
# apply() only accepts callables that leave the partition dimension
# untouched. The two methods below are the "dedicated implementations"
# for the classes of operation that genuinely need to reduce or
# communicate across it: matrix multiplication that contracts the
# partition dimension (needs an MPI reduction), and windowed/rolling
# reductions along the partition dimension (need boundary values owned
# by a neighboring rank). Both compute the mathematically correct
# distributed result instead of refusing outright.


def matmul(runtime: MPIRuntime, left: xr.DataArray, right: Any) -> xr.DataArray:
    """Matrix multiplication (``left @ right``), correct under MPI.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    left : xarray.DataArray
        Left operand.
    right : Any
        Right operand: an ``xarray.DataArray`` (distributed or not) or a plain array/scalar ``left`` can be matrix-multiplied with.
    Returns
    -------
    xarray.DataArray
        The matrix product.

    Raises
    ------
    ValueError
        If ``left``/``right`` are distributed over incompatible partitions (see :meth:`apply`).
    TypeError
        If the dtype involved has no MPI reduction datatype, when the distributed dimension is contracted.
    """
    meta, _reference = check_operands_distribution(runtime, (left, right))
    if meta is None:
        return _apply_generic(runtime, operator.matmul, (left, right), {})

    contracted = tuple(
        d
        for d in meta["dims"]
        if d in getattr(left, "dims", ()) and d in getattr(right, "dims", ())
    )
    if not contracted:
        # None of the partition dimensions are among the dot product's
        # common dimensions, so none are ever contracted: the operation
        # only reads this rank's own owned slice and apply()'s post-call
        # check confirms it.
        return _apply_generic(runtime, operator.matmul, (left, right), {})
    if len(contracted) > 1:
        raise NotImplementedError(
            "matmul() cannot yet contract more than one partition "
            + f"dimension at once (both {contracted!r} are common to "
            + "left and right under this multi-dimensional partition). "
            + "Reduce one of them first (e.g. via a prior matmul/sum "
            + "restricted to a single-dimension partition), or restructure "
            + "the contraction to touch only one partition axis."
        )
    dim = contracted[0]
    other_axes = tuple(d for d in meta["dims"] if d != dim)
    replicated = tuple(
        d
        for d in other_axes
        if not (d in getattr(left, "dims", ()) and d in getattr(right, "dims", ()))
    )
    if replicated:
        raise NotImplementedError(
            "matmul() cannot yet contract dimension "
            + f"{dim!r} while an operand is replicated along "
            + f"{replicated!r} under this multi-dimensional partition "
            + "(the contraction sum would need to be de-duplicated "
            + "across that replicated axis, which is not yet "
            + "implemented for matmul -- see sum()/mean(), which do "
            + "handle this)."
        )

    _agree(runtime, ("matmul", str(dim), int(meta["global_sizes"][dim])))

    partial = operator.matmul(left, right)
    total = comm_reduce(
        runtime,
        partial,
        MPI.SUM,
        phase="MPI xarray distributed matrix multiplication",
        comm=resolve_comm(runtime, meta, (dim,)),
    )
    return strip_mpi_meta(total)


def halo_exchange(
    runtime: MPIRuntime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable | None = None,
    *,
    before: int,
    after: int,
    periodic: bool = False,
) -> tuple[xr.Dataset | xr.DataArray, int, int]:
    """Pad ``value`` with boundary slices from the adjacent ranks.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed object to pad.
    dim : Hashable, optional
        The partition axis to exchange along.
    before, after : int
        Number of elements requested from the neighbor below/above along ``dim``.
    periodic : bool, optional
        Wrap the neighbor lookup at the global boundary instead of leaving that side unpadded -- rank 0's "left" neighbor becomes the last rank and vice versa (and symmetrically on the right), mirroring how FMS/``mpp_domains`` treats periodicity purely as a boundary condition on which rank a *bounded* halo exchange talks to (``cyclic``/``x_cyclic_offset``), not as a general block-move primitive.
    Returns
    -------
    tuple[xarray.Dataset or xarray.DataArray, int, int]
        ``(padded, left_pad, right_pad)``: the padded object (replicated metadata stripped, since it is no longer a clean partition) and the number of elements actually prepended/appended (equal to ``before``/``after`` except at a global edge, where it is 0).

    Raises
    ------
    ValueError
        If ``value`` is not distributed, ``dim`` is missing or disagrees with an active partition dimension, ``before``/``after`` are negative, or any rank's local partition along ``dim`` is shorter than ``before``/``after``.
    """
    meta = _operand_meta(value)
    if meta is None:
        raise ValueError(
            "halo_exchange() requires a distributed xarray object; "
            + "call repartition(...) first."
        )
    partition_dims = meta["dims"]
    if dim is None:
        if len(partition_dims) > 1:
            raise ValueError(
                "halo_exchange(): dim is required (no default) once more "
                + "than one dimension is partitioned; pick one of "
                + f"{tuple(str(d) for d in partition_dims)!r}."
            )
        partition_dim = partition_dims[0]
    elif dim not in partition_dims:
        raise ValueError(
            f"halo_exchange(): dim={dim!r} is not one of the object's "
            + f"active partition dimensions {tuple(str(d) for d in partition_dims)!r}."
        )
    else:
        partition_dim = dim
    if before < 0 or after < 0:
        raise ValueError("halo_exchange(): before and after must be >= 0.")

    _agree(
        runtime,
        ("halo_exchange", str(partition_dim), int(before), int(after), bool(periodic)),
    )

    if before == 0 and after == 0:
        # No boundary data requested on either side: every rank's own
        # local slice is already the complete answer, so this is a
        # local operation and should communicate nothing at all (the
        # same "no MPI traffic when no communication is structurally
        # required" rule the routing model applies everywhere else) --
        # skip the length allgather and every point-to-point call
        # below, which would otherwise still post/complete 2-4 messages
        # per rank carrying zero-length payloads for no benefit. Callers
        # that pass before=after=0 unconditionally (e.g. diff()/shift()
        # with n=0/periods=0, or an edge_order that needs no interior
        # halo) get correct output at zero communication cost instead.
        return value, 0, 0

    comm = runtime.comm
    rank = comm.rank
    if len(partition_dims) > 1:
        # Multi-dimensional partition: the rank below/above along
        # `partition_dim` is not `rank - 1`/`rank + 1` (that is a
        # neighbor in the flattened Cartesian rank numbering, which
        # only coincides with a neighbor along one particular axis
        # when every other axis has exactly one division). Look it up
        # from the Cartesian topology's `Cartcomm.Shift`-derived face
        # neighbors instead -- built once and cached, not repeated
        # per call; see `get_cartesian_topology`.
        topology = get_cartesian_topology(comm, partition_dims, meta["global_sizes"])
        if periodic:
            # `topology.cart_comm` was itself built non-periodic (its
            # `Shift`-derived `neighbors` always stop at a true edge,
            # which every *other* caller of this function needs), so
            # periodic wrapping is done by hand here rather than by
            # asking Cart_create for a periodic communicator: wrap
            # this rank's own coordinate on the target axis and look
            # up the owning rank directly with `Get_cart_rank`, which
            # (unlike `Shift`) does not consult the communicator's own
            # `periods` flag at all -- it just maps a coordinate tuple
            # to a rank, so this works regardless of how the
            # communicator itself was created. Safe to feed straight
            # back into `comm` (not just `cart_comm`) because
            # `Create_cart` above used `reorder=False`, which
            # `CartesianTopology.cart_comm`'s own docstring documents
            # as keeping the two rank numberings identical.
            axis = partition_dims.index(partition_dim)
            axis_size = topology.grid_shape[axis]
            coords = list(topology.coords)
            coords[axis] = (topology.coords[axis] - 1) % axis_size
            left_rank = topology.cart_comm.Get_cart_rank(coords)
            coords[axis] = (topology.coords[axis] + 1) % axis_size
            right_rank = topology.cart_comm.Get_cart_rank(coords)
        else:
            left_rank, right_rank = topology.neighbors[partition_dim]
    else:
        size = comm.size
        if periodic:
            left_rank = (rank - 1) % size
            right_rank = (rank + 1) % size
        else:
            left_rank = rank - 1 if rank > 0 else None
            right_rank = rank + 1 if rank < size - 1 else None

    local_len = int(value.sizes[partition_dim])
    lengths = comm.allgather(local_len)
    deficient = [
        (r, length)
        for r, length in enumerate(lengths)
        if length < before or length < after
    ]
    if deficient:
        raise ValueError(
            f"halo_exchange(): rank(s) {deficient} ([rank, local_length]) "
            + f"have a local partition along {partition_dim!r} shorter "
            + f"than the requested halo (before={before}, after={after}). "
            + "Each rank can only forward data it owns; repartition "
            + "with fewer, larger chunks (or a coarser process grid) "
            + "before requesting this wide a halo."
        )

    before_block, after_block = _exchange_halo_blocks(
        value,
        partition_dim,
        before,
        after,
        comm=comm,
        local_len=local_len,
        left_rank=left_rank,
        right_rank=right_rank,
    )

    pieces = [
        piece for piece in (before_block, value, after_block) if piece is not None
    ]
    if len(pieces) <= 1:
        padded = value
    elif isinstance(value, xr.Dataset):
        # data_vars="minimal": only concatenate variables that actually
        # vary along partition_dim. The default ("all") broadcasts every
        # *other* variable along it too, silently turning a static
        # (y, x) variable into a bogus (partition_dim, y, x) one
        # duplicated across before_block/value/after_block -- those three
        # pieces already agree exactly on any variable that lacks
        # partition_dim (each is this rank's or a neighbor's full,
        # untouched copy), so "minimal" is not just faster but the only
        # option that leaves such variables unchanged.
        padded = xr.concat(pieces, dim=partition_dim, data_vars="minimal")
    else:
        padded = xr.concat(pieces, dim=partition_dim)
    return (
        strip_mpi_meta(padded),
        before if before_block is not None else 0,
        after if after_block is not None else 0,
    )


def rolling_reduce(
    runtime: MPIRuntime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    window: int,
    reduce: str = "mean",
    *,
    center: bool = True,
    min_periods: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Windowed reduction along ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to roll over.
    dim : Hashable
        Dimension to roll over.
    window : int
        Window size, as in ``xarray.DataArray.rolling``.
    reduce : str, optional
        Name of the reduction to call on the rolling object (e.g.
    center : bool, optional
        As in ``xarray.DataArray.rolling``.
    min_periods : int or None, optional
        As in ``xarray.DataArray.rolling``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rolled-and-reduced result, with the same local length and distribution metadata as the input when ``dim`` is the partition dimension.
    """
    meta = _operand_meta(value)
    if meta is None or dim not in meta["dims"]:
        rolled = value.rolling({dim: window}, center=center, min_periods=min_periods)
        return getattr(rolled, reduce)()

    # xarray's own centered-window convention (verified against
    # DataArray.rolling(..., center=True) directly): for an odd window
    # the split is symmetric either way, but for an *even* window the
    # extra element goes on the left, i.e. before=window//2, not
    # (window-1)//2 -- the two only differ (by one, in the wrong
    # direction) when window is even.
    before = window // 2 if center else window - 1
    after = (window - 1) - before if center else 0

    padded, left_pad, _right_pad = halo_exchange(
        runtime, value, dim, before=before, after=after
    )
    rolled = padded.rolling({dim: window}, center=center, min_periods=min_periods)
    reduced = getattr(rolled, reduce)()

    local_len = int(value.sizes[dim])
    trimmed = reduced.isel({dim: slice(left_pad, left_pad + local_len)})
    return reattach_meta(trimmed, meta)


def coarsen_reduce(
    runtime: MPIRuntime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    window: int,
    reduce: str = "mean",
    *,
    boundary: str = "exact",
    side: str = "left",
    coord_func: str = "mean",
) -> xr.Dataset | xr.DataArray:
    """Block reduction along ``dim``, correct when ``dim`` is distributed.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to coarsen.
    dim : Hashable
        Dimension to coarsen along.
    window : int
        Block size, as in ``xarray.DataArray.coarsen``.
    reduce : str, optional
        Name of the reduction to call on the coarsen object (e.g.
    boundary : {"exact", "trim", "pad"}, optional
        As in ``xarray.DataArray.coarsen``.
    side : {"left"}, optional
        As in ``xarray.DataArray.coarsen``.
    coord_func : str, optional
        As in ``xarray.DataArray.coarsen``.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The coarsened-and-reduced result, correctly distributed along the now block-reduced ``dim``.

    Raises
    ------
    ValueError
        If ``boundary="exact"`` and the global size is not evenly divisible by ``window``.
    NotImplementedError
        If ``side="right"`` is requested on a distributed ``dim``.
    """
    meta = get_mpi_meta(value)
    if meta is None or dim not in meta["dims"]:
        coarsened = value.coarsen(
            {dim: window}, boundary=boundary, side=side, coord_func=coord_func
        )
        return getattr(coarsened, reduce)()

    if side != "left":
        raise NotImplementedError(
            "coarsen_reduce(): side='right' is not yet implemented for a "
            + "distributed dimension (only the default side='left' is); "
            + "gather/repartition onto a single rank along this dimension "
            + "first if side='right' is required."
        )

    _agree(
        runtime,
        ("coarsen_reduce", str(dim), int(window), boundary, side),
    )

    global_size = int(meta["global_sizes"][dim])
    start = int(meta["starts"][dim])
    stop = int(meta["stops"][dim])
    remainder = global_size % window

    if boundary == "exact" and remainder != 0:
        raise ValueError(
            f"Could not coarsen a distributed dimension of size {global_size} "
            + f"with window {window} and boundary='exact'. Try boundary="
            + "'trim' or 'pad'."
        )

    is_left_edge = start == 0
    is_right_edge = stop == global_size

    before_needed = 0 if is_left_edge else start % window
    after_needed = 0 if is_right_edge else (window - stop % window) % window

    # halo_exchange() requires every rank to request the *same*
    # before/after width (enforced by its own internal _agree(), which
    # exists to catch genuine cross-rank call mismatches elsewhere and
    # should not be weakened here) -- but each rank's own alignment
    # offset (before_needed/after_needed above) is a function of its
    # own start/stop, so it is not, and must not be forced to be, the
    # same on every rank. Request the single largest width any rank
    # could ever need (window - 1, an O(window) bound, not an
    # O(global_size) one) uniformly instead, then trim the excess back
    # off below once each rank knows what it actually received -- the
    # same "ask for an upper bound, trim locally" trick, just applied
    # to a collective-agreement constraint rather than to the maximum
    # halo width itself.
    request = max(window - 1, 0)
    padded, left_pad, right_pad = halo_exchange(
        runtime, value, dim, before=request, after=request
    )
    # left_pad/right_pad are what was actually fetched (0 at a true
    # global edge, `request` everywhere else); keep only the slice
    # closest to this rank's own data on each side.
    padded = padded.isel(
        {
            dim: slice(
                left_pad - before_needed,
                left_pad + int(value.sizes[dim]) + after_needed,
            )
        }
    )

    local_boundary = "exact"
    if is_right_edge and remainder != 0:
        if boundary == "trim":
            trim_len = int(padded.sizes[dim]) - remainder
            padded = padded.isel({dim: slice(0, trim_len)})
        else:  # "pad": only the true global edge ever needs a synthetic
            # (non-neighbor-sourced) pad -- every interior boundary block
            # already got real data from halo_exchange above.
            local_boundary = "pad"

    coarsened = getattr(
        padded.coarsen(
            {dim: window}, boundary=local_boundary, side="left", coord_func=coord_func
        ),
        reduce,
    )()

    if before_needed > 0:
        # This rank's own first block started inside the left neighbor's
        # unpadded range (see the ownership rule in the docstring); the
        # left neighbor computes and reports the identical block itself.
        coarsened = coarsened.isel({dim: slice(1, None)})

    # coarsen changes the dimension's length, so -- exactly like diff()'s
    # own length-changing case -- start/stop/global_size are recomputed
    # from an allgather of each rank's new local length, not carried
    # over from the (now stale) pre-coarsen meta.
    comm = _dim_comm(runtime, meta, dim)
    counts = comm.allgather(int(coarsened.sizes[dim]))
    new_global_size = sum(counts)
    new_start = sum(counts[: comm.rank])
    new_stop = new_start + counts[comm.rank]
    new_chunk_info = prune_chunk_info(meta["chunk_info"], coarsened)
    global_sizes = dict(meta["global_sizes"])
    starts = dict(meta["starts"])
    stops = dict(meta["stops"])
    global_sizes[dim] = new_global_size
    starts[dim] = new_start
    stops[dim] = new_stop
    set_mpi_meta(
        coarsened,
        dim=meta["dims"],
        global_size=global_sizes,
        start=starts,
        stop=stops,
        chunk_info=new_chunk_info,
        cart=meta.get("cart"),
    )
    return coarsened


def _eval_ast_node(
    runtime: MPIRuntime, node: ast.expr, variables: Mapping[str, Any]
) -> Any:
    """Recursively evaluate one parsed expression node."""
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.MatMult):
            left = _eval_ast_node(runtime, node.left, variables)
            right = _eval_ast_node(runtime, node.right, variables)
            return matmul(runtime, left, right)

        function = _AST_BINARY_OPS.get(type(node.op))
        if function is None:
            raise ValueError(
                f"Unsupported operator {type(node.op).__name__!r} in " + "expression."
            )
        left = _eval_ast_node(runtime, node.left, variables)
        right = _eval_ast_node(runtime, node.right, variables)
        return apply(runtime, function, left, right)

    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        last_val = None
        for val_node in node.values:
            last_val = _eval_ast_node(runtime, val_node, variables)
            if isinstance(last_val, (xr.Dataset, xr.DataArray)):
                raise TypeError(
                    "evaluate(): 'and'/'or' use Python truth-value "
                    + "checks, which are not defined for xarray "
                    + "Datasets/DataArrays (no single element is "
                    + "'the' truth value of a multi-element array). "
                    + "Use the elementwise bitwise forms instead: '&' "
                    + "for 'and', '|' for 'or', e.g. "
                    + '"(a > 0) & (b < 1)".'
                )
            if is_and and not last_val:
                return last_val
            if not is_and and last_val:
                return last_val
        return last_val

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError(
                "Chained comparisons (e.g. 'a < b < c') are not "
                + "supported; write them as separate comparisons."
            )
        function = _AST_COMPARE_OPS.get(type(node.ops[0]))
        if function is None:
            raise ValueError(
                f"Unsupported comparison {type(node.ops[0]).__name__!r} "
                + "in expression."
            )
        left = _eval_ast_node(runtime, node.left, variables)
        right = _eval_ast_node(runtime, node.comparators[0], variables)
        return apply(runtime, function, left, right)

    if isinstance(node, ast.UnaryOp):
        function = _AST_UNARY_OPS.get(type(node.op))
        if function is None:
            raise ValueError(
                f"Unsupported unary operator {type(node.op).__name__!r} "
                + "in expression."
            )
        operand = _eval_ast_node(runtime, node.operand, variables)
        return apply(runtime, function, operand)

    if isinstance(node, ast.Name):
        try:
            return variables[node.id]
        except KeyError:
            raise NameError(
                f"Name {node.id!r} is not defined; pass it as "
                + f"evaluate(..., {node.id}=...)."
            ) from None

    if isinstance(node, ast.Constant):
        return node.value

    raise ValueError(
        f"Unsupported expression element {type(node).__name__!r}; "
        + "evaluate() only accepts variable names, numeric literals, "
        + "parentheses, and the arithmetic/comparison/bitwise/boolean "
        + "operators."
    )


def evaluate(runtime: MPIRuntime, expression: str, /, **variables: Any) -> Any:
    """Evaluate a string expression, respecting normal operator precedence.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime used for communication.
    expression : str
        A Python expression referencing ``variables`` by name, for example ``"(a + b) * c - d / e"``.
    **variables : Any
        Values bound to the names used in ``expression``: xarray Datasets/DataArrays (distributed or not) or plain scalars.
    Returns
    -------
    Any
        The expression's value.

    Raises
    ------
    ValueError
        If ``expression`` fails to parse, uses an unsupported operator or expression element, or chains comparisons.
    NameError
        If ``expression`` references a name not present in ``variables``.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse expression {expression!r}: {exc}") from exc
    return _eval_ast_node(runtime, tree.body, variables)
