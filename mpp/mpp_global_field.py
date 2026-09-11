"""Assemble and move whole fields between decompositions.

Mirrors FMS ``mpp/include/mpp_global_field.fh`` and ``mpp_domains_misc.inc``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

import xarray as xr

from ..mpi.mpi_init import MPI
from ..xarray.chunks import get_balanced_bounds, prune_chunk_info
from ..xarray.meta import mpp_update_meta, strip_mpi_meta
from .mpp_do_update import _fill_chunk
from .mpp_domains import DomainMismatchError
from .mpp_domains_define import mpp_dim_comm

if TYPE_CHECKING:
    from ..mpi.context import MPIContext


def mpp_global_field(
    mpi_context: MPIContext,
    coordinate: xr.DataArray,
    dim: str,
    comm: MPI.Comm,
    *,
    start: int,
    stop: int,
    global_size: int,
) -> xr.DataArray | None:
    """Gather a coordinate distributed along ``dim`` into its global form.

    Follows FMS ``mpp_global_field``: each rank contributes its compute-domain
    slice and the root reassembles the whole axis, verifying that the slices
    tile it exactly with no gap or overlap.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    coordinate : xarray.DataArray
        This rank's slice of the coordinate.
    dim : str
        Partitioned dimension.
    comm : mpi4py.MPI.Comm
        Communicator varying along ``dim``.
    start, stop : int
        This rank's half-open bounds along ``dim``.
    global_size : int
        Global length of ``dim``.

    Returns
    -------
    xarray.DataArray or None
        The reassembled coordinate on rank 0 of ``comm``, None elsewhere.

    Raises
    ------
    DomainMismatchError
        If the gathered slices do not tile the axis exactly.
    """
    axis = coordinate.get_axis_num(dim)
    pieces = comm.gather((start, stop, np.asarray(coordinate.values)), root=0)
    if comm.rank != 0 or pieces is None:
        return None

    cursor = 0
    ordered = sorted(pieces, key=lambda item: item[0])
    for piece_start, piece_stop, values in ordered:
        if piece_start != cursor:
            raise DomainMismatchError(
                f"Coordinate {coordinate.name!r}: expected start {cursor}, "
                + f"got {piece_start}."
            )
        if values.shape[axis] != piece_stop - piece_start:
            raise DomainMismatchError(
                f"Coordinate {coordinate.name!r} slice length "
                + f"{values.shape[axis]} != {piece_stop - piece_start}."
            )
        cursor = piece_stop
    if cursor != global_size:
        raise DomainMismatchError(
            f"Coordinate {coordinate.name!r} covers {cursor}/{global_size} elements."
        )

    rebuilt = xr.DataArray(
        np.concatenate([values for _, _, values in ordered], axis=axis),
        dims=coordinate.dims,
        name=coordinate.name,
        attrs=dict(coordinate.attrs),
    )
    rebuilt.encoding = dict(coordinate.encoding)
    return rebuilt


def mpp_redistribute(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    meta: Mapping[str, Any],
    dim: str,
    *,
    new_coord: np.ndarray[Any, Any],
    old_pos: np.ndarray[Any, Any],
    fill_value: Any,
) -> xr.Dataset | xr.DataArray:
    """Move ``value`` onto a new decomposition of ``dim``.

    Follows FMS ``mpp_redistribute``: each rank derives, from the shared
    position map alone, which of its elements every other rank needs and which
    it must receive, so no layout metadata travels with the payload.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.Dataset or xarray.DataArray
        Object to redistribute.
    meta : mapping
        Current distribution metadata.
    dim : str
        Dimension being redistributed.
    new_coord : numpy.ndarray
        Coordinate values of the target decomposition.
    old_pos : numpy.ndarray
        For each new global position, the old global position feeding it, or
        -1 where the target has no source and takes ``fill_value``.
    fill_value : Any
        Value for target positions with no source.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        This rank's slice of the redistributed object.
    """
    comm = mpp_dim_comm(mpi_context, meta, dim)
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
