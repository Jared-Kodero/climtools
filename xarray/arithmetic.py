"""Provide MPI-aware alignment and arithmetic for distributed xarray objects."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..mpi.mpi_init import MPI
from .cartesian import mpp_dim_comm as _dim_comm
from .chunks import get_balanced_bounds, prune_chunk_info
from .meta import _partitions_match, mpp_get_meta, mpp_update_meta, strip_mpi_meta
from .planning import _agree, mpp_comm_reduce, mpp_resolve_comm

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from ..mpi.context import MPIContext

from .mpp import (
    Domain,
    mpp_complete_update_domains,
    mpp_get_neighbor_pe,
    mpp_partition_offsets,
    mpp_start_update_domains,
)

# Route matrix multiplication callables to the MPI-aware implementation.
_MATMUL_CALLABLES: frozenset[Callable[..., Any]] = frozenset(
    {operator.matmul, np.matmul}
)
# Handle ``@`` separately because contracting a partition dimension requires MPI.
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
                    # Pass the fill value itself so NumPy preserves xarray's
                    # scalar-promotion rules.
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
    domain: Domain,
    left_rank: int | None,
    right_rank: int | None,
) -> tuple[xr.Dataset | xr.DataArray | None, xr.Dataset | xr.DataArray | None]:
    """Exchange boundary slabs with adjacent ranks."""
    haloed = _haloed_variable_names(value, partition_dim)

    def _local_array(name: Hashable) -> xr.Variable:
        if isinstance(value, xr.Dataset):
            return value[name].variable
        if name == value.name:
            return value.variable
        return value.coords[name].variable

    # Move each partition axis to axis 0 so mixed variable layouts share one halo
    # kernel.
    axes = {name: _local_array(name).dims.index(partition_dim) for name in haloed}
    fields = {
        name: np.moveaxis(np.asarray(_local_array(name).values), axes[name], 0)
        for name in haloed
    }

    # Use start/complete to consume only halo slabs and avoid full-array copies.
    update = mpp_start_update_domains(
        fields,
        domain,
        str(partition_dim),
        0,
        before=before,
        after=after,
        left_rank=left_rank,
        right_rank=right_rank,
    )
    recv_before, recv_after, left_pad, right_pad = mpp_complete_update_domains(update)

    def _received(name: Hashable, side: str) -> np.ndarray[Any, Any] | None:
        """This name's exchanged slab, moved back to its original axis, or None."""
        pad = left_pad if side == "before" else right_pad
        if pad == 0:
            return None
        slab = (recv_before if side == "before" else recv_after)[name]
        return np.moveaxis(slab, 0, axes[name])

    def _reconstruct(side: str) -> xr.Dataset | xr.DataArray | None:
        """Reconstruct an xarray object from the exchanged arrays, or None if unpadded."""
        if (left_pad if side == "before" else right_pad) == 0:
            return None
        if isinstance(value, xr.Dataset):
            pieces = {}
            for name, var in value.variables.items():
                received = _received(name, side) if name in haloed else None
                pieces[name] = (
                    var
                    if received is None
                    else xr.Variable(var.dims, received, attrs=var.attrs)
                )
            return xr.Dataset(pieces, attrs=value.attrs)
        data_var = xr.Variable(
            value.dims, _received(value.name, side), attrs=value.attrs
        )
        new_coords = {}
        for coord_name, coord in value.coords.items():
            received = _received(coord_name, side) if coord_name in haloed else None
            new_coords[coord_name] = (
                coord.variable
                if received is None
                else xr.Variable(coord.dims, received, attrs=coord.attrs)
            )
        return xr.DataArray(data_var, coords=new_coords, name=value.name)

    return _reconstruct("before"), _reconstruct("after")


def _gather_full(
    mpi_context: MPIContext, value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]
) -> xr.Dataset | xr.DataArray:
    """Reconstruct ``value``'s full, replicated extent on every rank."""
    dim = meta["dim"]
    if len(meta["dims"]) > 1:
        raise NotImplementedError(f"Gathering partition dims {meta['dims']!r} is unsupported.")
    pieces = mpi_context.comm.allgather(value)
    full = (
        xr.concat(pieces, dim=dim, data_vars="minimal")
        if isinstance(value, xr.Dataset)
        else xr.concat(pieces, dim=dim)
    )
    return strip_mpi_meta(full)


def _align_replicated(
    mpi_context: MPIContext,
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
            raise ValueError(f"Cannot align {dim!r}: length {length}, expected {global_size}.")
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
                    f"Operand {dim!r} coordinates do not match this rank."
                ) from exc

    return reattach_meta(sliced, meta)


def mpp_align(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
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
    from .io import mpp_repartition

    left_meta = _operand_meta(left)
    right_meta = _operand_meta(right)

    if left_meta is not None and right_meta is not None:
        if _partitions_match(left_meta, right_meta):
            return left, right
        target_dim = dim if dim is not None else left_meta["dim"]
        full_left = _gather_full(mpi_context, left, left_meta)
        full_right = _gather_full(mpi_context, right, right_meta)
        return (
            mpp_repartition(
                mpi_context,
                full_left,
                target_dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            mpp_repartition(
                mpi_context,
                full_right,
                target_dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
        )

    if left_meta is not None:
        return left, _align_replicated(mpi_context, right, left_meta, partner=left)

    if right_meta is not None:
        return _align_replicated(mpi_context, left, right_meta, partner=right), right

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
            raise ValueError(f"Cannot align {dim!r}: coordinate labels differ.") from exc

    return (
        mpp_repartition(
            mpi_context, left, dim, chunk_info=chunk_info, log_partitions=log_partitions
        ),
        mpp_repartition(
            mpi_context,
            right,
            dim,
            chunk_info=chunk_info,
            log_partitions=log_partitions,
        ),
    )


# Reindex and sort stay local unless they touch a partition dimension.
# Partition-axis changes gather only coordinate metadata, then shuffle bulk data
# point-to-point.


def _shuffle_by_position(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    dim: str,
    *,
    new_coord: np.ndarray[Any, Any],
    old_pos: np.ndarray[Any, Any],
    fill_value: Any,
) -> xr.Dataset | xr.DataArray:
    """Redistribute ``value`` along ``dim`` to match ``old_pos``."""
    comm = _dim_comm(mpi_context, meta, dim)
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

    # Source and destination ranks derive the same position map, so payload metadata is
    # unnecessary.
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

    # Use ``recv`` for pickled payloads because it probes size; ``irecv`` requires a
    # buffer-size guess.
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
                raise AssertionError("Missing planned self-contribution.")
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
        mpp_update_meta(
            result,
            dim=all_dims,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info=chunk_info,
            cart=meta.get("cart"),
        )
    else:
        mpp_update_meta(
            result,
            dim=dim,
            global_size=new_length,
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
    return result


def mpp_reindex(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
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
        raise ValueError("requires at least one indexer")

    meta = mpp_get_meta(value)
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
        mpp_update_meta(
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
        raise NotImplementedError(f"Cannot redistribute multiple partition dims: {touched!r}.")

    dim = touched[0]
    new_labels = np.asarray(indexers[dim])
    if new_labels.ndim != 1:
        raise NotImplementedError(f"New {dim!r} labels must be 1-D; got {new_labels.shape!r}.")
    _agree(
        mpi_context,
        (
            "reindex",
            dim,
            int(new_labels.shape[0]),
            str(method),
            str(tolerance),
        ),
    )

    comm = _dim_comm(mpi_context, meta, dim)
    old_coord_local = np.asarray(value[dim].values)
    old_full_coord = np.concatenate(comm.allgather(old_coord_local))
    old_index = pd.Index(old_full_coord)
    old_pos = old_index.get_indexer(new_labels, method=method, tolerance=tolerance)
    old_pos = old_pos.astype(np.int64)

    return _shuffle_by_position(
        mpi_context,
        value,
        meta,
        dim,
        new_coord=new_labels,
        old_pos=old_pos,
        fill_value=fill_value,
    )


def mpp_sortby(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
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
    meta = mpp_get_meta(value)
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
        mpp_update_meta(
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
        raise NotImplementedError(f"Sort keys span multiple partition dims: {touched!r}.")

    dim = touched[0]
    local_len = int(value.sizes[dim])
    key_arrays_local: list[np.ndarray[Any, Any]] = []
    for key in keys:
        arr = np.asarray(
            key.values if isinstance(key, xr.DataArray) else value[key].values
        )
        if arr.ndim != 1 or arr.shape[0] != local_len:
            raise NotImplementedError(
                f"Sort key {key!r} must be 1-D along {dim!r}; got {arr.shape!r}."
            )
        key_arrays_local.append(arr)

    key_signature = tuple(
        "<dataarray>" if isinstance(key, xr.DataArray) else str(key) for key in keys
    )
    _agree(mpi_context, ("sortby", dim, key_signature, bool(ascending)))

    comm = _dim_comm(mpi_context, meta, dim)
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
        mpi_context,
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
        return mpp_get_meta(operand)
    return None


def reattach_meta(result: Any, meta: dict[str, Any]) -> Any:
    """Tag ``result`` with ``meta`` if it is an xarray object.

    Returns
    -------
    Any
        The tagged result object if it is an xarray dataset or dataarray, otherwise returned unmodified.

    """
    if isinstance(result, (xr.Dataset, xr.DataArray)):
        mpp_update_meta(
            result,
            dim=meta["dims"],
            global_size=meta["global_sizes"],
            start=meta["starts"],
            stop=meta["stops"],
            chunk_info=meta["chunk_info"],
            cart=meta.get("cart"),
        )
    return result


def mpp_check_operands_distribution(
    mpi_context: MPIContext, operands: Iterable[Any]
) -> tuple[dict[str, Any] | None, Any]:
    """Return the mpi_meta to attach to a multi-operand call's result.

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
                raise ValueError("Operands have different partition ownership.")
            continue

        for dim in meta["dims"]:
            if not (
                isinstance(other, (xr.Dataset, xr.DataArray)) and dim in other.dims
            ):
                continue
            owned = meta["stops"][dim] - meta["starts"][dim]
            local = int(other.sizes[dim])
            if local != owned:
                raise ValueError(f"Operand {dim!r} length is {local}; expected {owned}.")
            reference_indexed = dim in getattr(reference, "indexes", {})
            other_indexed = dim in getattr(other, "indexes", {})
            if reference_indexed and other_indexed:
                try:
                    xr.align(reference, other, join="exact")
                except (ValueError, KeyError) as exc:
                    raise ValueError(
                    f"Operand {dim!r} coordinates do not match this rank."
                ) from exc
            elif mpi_context.comm.size > 1:
                # Without coordinates, equal local lengths cannot prove cross-rank
                # alignment; reject the ambiguous case.
                missing = [
                    name
                    for name, indexed in (
                        ("the distributed side", reference_indexed),
                        ("the operand", other_indexed),
                    )
                    if not indexed
                ]
                raise ValueError(
                    f"Cannot verify {dim!r} alignment: missing coordinate on "
                    + f"{' and '.join(missing)}."
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
            raise ValueError(f"Callable removed distributed dimension {dim!r}.")

        local = int(result.sizes[dim])
        if local != owned:
            raise ValueError(f"Callable changed local {dim!r} length from {owned} to {local}.")

        if (
            isinstance(reference, (xr.Dataset, xr.DataArray))
            and dim in getattr(reference, "indexes", {})
            and dim in getattr(result, "indexes", {})
        ):
            try:
                xr.align(reference, result, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Callable changed {dim!r} coordinates.") from exc


def mpp_apply(
    mpi_context: MPIContext, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
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
        return mpp_matmul(mpi_context, *args)

    return _apply_generic(mpi_context, func, args, kwargs)


def _apply_generic(
    mpi_context: MPIContext,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run the shared partition-preserving callable path."""
    meta, reference = mpp_check_operands_distribution(
        mpi_context, (*args, *kwargs.values())
    )

    _agree(
        mpi_context,
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


# Cross-partition matmul and rolling operations use dedicated MPI-aware paths.


def mpp_matmul(mpi_context: MPIContext, left: xr.DataArray, right: Any) -> xr.DataArray:
    """Matrix multiplication (``left @ right``), correct under MPI.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
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
    meta, _reference = mpp_check_operands_distribution(mpi_context, (left, right))
    if meta is None:
        return _apply_generic(mpi_context, operator.matmul, (left, right), {})

    contracted = tuple(
        d
        for d in meta["dims"]
        if d in getattr(left, "dims", ()) and d in getattr(right, "dims", ())
    )
    if not contracted:
        # No partition dimension is contracted, so matrix multiplication is rank-local.
        return _apply_generic(mpi_context, operator.matmul, (left, right), {})
    if len(contracted) > 1:
        raise NotImplementedError(f"Cannot contract multiple partition dims: {contracted!r}.")
    dim = contracted[0]
    other_axes = tuple(d for d in meta["dims"] if d != dim)
    replicated = tuple(
        d
        for d in other_axes
        if not (d in getattr(left, "dims", ()) and d in getattr(right, "dims", ()))
    )
    if replicated:
        raise NotImplementedError(
            f"Cannot contract {dim!r}; operand is replicated over {replicated!r}."
        )

    _agree(mpi_context, ("matmul", str(dim), int(meta["global_sizes"][dim])))

    partial = operator.matmul(left, right)
    total = mpp_comm_reduce(
        mpi_context,
        partial,
        MPI.SUM,
        phase="MPI xarray distributed matrix multiplication",
        comm=mpp_resolve_comm(mpi_context, meta, (dim,)),
    )
    return strip_mpi_meta(total)


class HaloWidthError(ValueError):
    """A rank's local partition is shorter than the halo an op asked for.

    Its own type rather than a bare ``ValueError`` because callers have to
    tell this architectural refusal apart from a genuine failure -- the test
    suite reports it as a skip, not a failure. That classification used to
    match a substring of the message, so shortening the message silently
    turned every one of those skips into a failure. Subclasses ``ValueError``
    so existing ``except ValueError`` handlers are unaffected.
    """


def mpp_halo_exchange(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable | None = None,
    *,
    before: int,
    after: int,
    periodic: bool = False,
    exchange_coords: bool = True,
) -> tuple[xr.Dataset | xr.DataArray, int, int]:
    """Pad ``value`` with boundary slices from the adjacent ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed object to pad.
    dim : Hashable, optional
        The partition axis to exchange along.
    before, after : int
        Number of elements requested from the neighbor below/above along ``dim``.
    periodic : bool, optional
        Wrap the neighbor lookup at the global boundary instead of
        leaving that side unpadded (rank 0's lower neighbor becomes the
        last rank, and symmetrically on the upper side).
    exchange_coords : bool, optional
        Whether coordinates varying along ``dim`` take part. An operation
        that reads coordinate *values* across the rank boundary needs them
        -- ``differentiate`` divides by a spacing that straddles it -- but
        one that reads only data values and then trims back to its own
        compute domain does not, which is most of them. Exchanging them
        anyway costs far more than their size suggests: joining an index
        coordinate makes xarray rebuild a pandas Index over the padded
        extent, measured here at roughly four times the cost of joining the
        data alone. Pass False when the caller restores the coordinate
        itself; the padded object then carries none along ``dim``.

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
        raise ValueError("requires a distributed xarray object")
    partition_dims = meta["dims"]
    if dim is None:
        if len(partition_dims) > 1:
            raise ValueError(
                "dim is required for partition dimensions "
                + f"{tuple(str(d) for d in partition_dims)!r}."
            )
        partition_dim = partition_dims[0]
    elif dim not in partition_dims:
        raise ValueError(
            f"dim={dim!r} is not active; choose from "
            + f"{tuple(str(d) for d in partition_dims)!r}."
        )
    else:
        partition_dim = dim
    if before < 0 or after < 0:
        raise ValueError("before and after must be >= 0")

    _agree(
        mpi_context,
        (
            "mpp_halo_exchange",
            str(partition_dim),
            int(before),
            int(after),
            bool(periodic),
            bool(exchange_coords),
        ),
    )

    if before == 0 and after == 0:
        # A zero-width halo is purely local; skip all communication.
        return value, 0, 0

    if not exchange_coords:
        along_dim = [
            name for name, coord in value.coords.items() if partition_dim in coord.dims
        ]
        if along_dim:
            value = value.drop_vars(along_dim)

    comm = mpi_context.comm
    # Resolve halo neighbors through the Cartesian-aware domain helper.
    domain = Domain.from_meta(meta, comm)
    left_rank, right_rank = mpp_get_neighbor_pe(
        domain, str(partition_dim), periodic=periodic
    )

    local_len = int(value.sizes[partition_dim])
    # Use a fixed-size reduction for the common pass case; gather rank details only on
    # failure.
    shortest = np.empty(1, dtype=np.int64)
    comm.Allreduce(np.array([local_len], dtype=np.int64), shortest, op=MPI.MIN)
    if int(shortest[0]) < max(before, after):
        lengths = comm.allgather(local_len)
        deficient = [
            (r, length)
            for r, length in enumerate(lengths)
            if length < before or length < after
        ]
        raise HaloWidthError(
            f"Halo ({before}, {after}) exceeds local {partition_dim!r} size "
            + f"on ranks {deficient}."
        )

    before_block, after_block = _exchange_halo_blocks(
        value,
        partition_dim,
        before,
        after,
        domain=domain,
        left_rank=left_rank,
        right_rank=right_rank,
    )

    pieces = [
        piece for piece in (before_block, value, after_block) if piece is not None
    ]
    if len(pieces) <= 1:
        padded = value
    elif isinstance(value, xr.Dataset):
        # Concatenate only variables that vary along the partition dimension.
        padded = xr.concat(pieces, dim=partition_dim, data_vars="minimal")
    else:
        padded = xr.concat(pieces, dim=partition_dim)
    return (
        strip_mpi_meta(padded),
        before if before_block is not None else 0,
        after if after_block is not None else 0,
    )


def mpp_rolling_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    window: int,
    reduce: str = "mean",
    *,
    center: bool = True,
    min_periods: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """Windowed reduction along ``dim``, correct when ``dim`` is distributed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The rolled-and-reduced result, with the same local length and distribution metadata as the input when ``dim`` is the partition dimension.

    """
    meta = _operand_meta(value)
    if meta is None or dim not in meta["dims"]:
        rolled = value.rolling({dim: window}, center=center, min_periods=min_periods)
        return getattr(rolled, reduce)()

    # Match xarray centered windows: even windows place the extra cell on the left.
    before = window // 2 if center else window - 1
    after = (window - 1) - before if center else 0

    # Halo coordinates are unused; restore the original compute-domain coordinate after
    # trimming.
    dim_coords = {
        name: coord for name, coord in value.coords.items() if dim in coord.dims
    }
    padded, left_pad, _right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=before, after=after, exchange_coords=False
    )
    rolled = padded.rolling({dim: window}, center=center, min_periods=min_periods)
    reduced = getattr(rolled, reduce)()

    local_len = int(value.sizes[dim])
    trimmed = reduced.isel({dim: slice(left_pad, left_pad + local_len)})
    if dim_coords:
        trimmed = trimmed.assign_coords(dim_coords)
    return reattach_meta(trimmed, meta)


def mpp_coarsen_reduce(
    mpi_context: MPIContext,
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
    meta = mpp_get_meta(value)
    if meta is None or dim not in meta["dims"]:
        coarsened = value.coarsen(
            {dim: window}, boundary=boundary, side=side, coord_func=coord_func
        )
        return getattr(coarsened, reduce)()

    if side != "left":
        raise NotImplementedError("Distributed searchsorted supports only side='left'.")

    _agree(
        mpi_context,
        ("coarsen_reduce", str(dim), int(window), boundary, side),
    )

    global_size = int(meta["global_sizes"][dim])
    start = int(meta["starts"][dim])
    stop = int(meta["stops"][dim])
    remainder = global_size % window

    if boundary == "exact" and remainder != 0:
        raise ValueError(
            f"Size {global_size} is not divisible by window {window} "
            + "with boundary='exact'."
        )

    is_left_edge = start == 0
    is_right_edge = stop == global_size

    before_needed = 0 if is_left_edge else start % window
    after_needed = 0 if is_right_edge else (window - stop % window) % window

    # Request the common upper-bound halo ``window - 1`` on all ranks, then trim
    # locally.
    request = max(window - 1, 0)
    padded, left_pad, right_pad = mpp_halo_exchange(
        mpi_context, value, dim, before=request, after=request
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
            # already got real data from mpp_halo_exchange above.
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

    # Recompute global bounds after coarsen because the distributed dimension length
    # changes.
    comm = _dim_comm(mpi_context, meta, dim)
    new_global_size, new_start, new_stop = mpp_partition_offsets(
        comm, int(coarsened.sizes[dim])
    )
    new_chunk_info = prune_chunk_info(meta["chunk_info"], coarsened)
    global_sizes = dict(meta["global_sizes"])
    starts = dict(meta["starts"])
    stops = dict(meta["stops"])
    global_sizes[dim] = new_global_size
    starts[dim] = new_start
    stops[dim] = new_stop
    mpp_update_meta(
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
    mpi_context: MPIContext, node: ast.expr, variables: Mapping[str, Any]
) -> Any:
    """Recursively evaluate one parsed expression node."""
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.MatMult):
            left = _eval_ast_node(mpi_context, node.left, variables)
            right = _eval_ast_node(mpi_context, node.right, variables)
            return mpp_matmul(mpi_context, left, right)

        function = _AST_BINARY_OPS.get(type(node.op))
        if function is None:
            raise ValueError(f"Unsupported expression operator: {type(node.op).__name__}.")
        left = _eval_ast_node(mpi_context, node.left, variables)
        right = _eval_ast_node(mpi_context, node.right, variables)
        return mpp_apply(mpi_context, function, left, right)

    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        last_val = None
        for val_node in node.values:
            last_val = _eval_ast_node(mpi_context, val_node, variables)
            if isinstance(last_val, (xr.Dataset, xr.DataArray)):
                raise TypeError("Use '&' or '|' for array boolean expressions.")
            if is_and and not last_val:
                return last_val
            if not is_and and last_val:
                return last_val
        return last_val

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Chained comparisons are unsupported; combine separate comparisons.")
        function = _AST_COMPARE_OPS.get(type(node.ops[0]))
        if function is None:
            raise ValueError(f"Unsupported comparison operator: {type(node.ops[0]).__name__}.")
        left = _eval_ast_node(mpi_context, node.left, variables)
        right = _eval_ast_node(mpi_context, node.comparators[0], variables)
        return mpp_apply(mpi_context, function, left, right)

    if isinstance(node, ast.UnaryOp):
        function = _AST_UNARY_OPS.get(type(node.op))
        if function is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}.")
        operand = _eval_ast_node(mpi_context, node.operand, variables)
        return mpp_apply(mpi_context, function, operand)

    if isinstance(node, ast.Name):
        try:
            return variables[node.id]
        except KeyError:
            raise NameError(f"Undefined expression name {node.id!r}.") from None

    if isinstance(node, ast.Constant):
        return node.value

    raise ValueError(f"Unsupported expression node: {type(node).__name__}.")


def mpp_evaluate(mpi_context: MPIContext, expression: str, /, **variables: Any) -> Any:
    """Evaluate a string expression, respecting normal operator precedence.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
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
    return _eval_ast_node(mpi_context, tree.body, variables)
