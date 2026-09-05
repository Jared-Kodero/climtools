"""FMS/mpp-style distributed-memory primitives, adapted from GFDL FMS."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..mpi.context import MPIContext

__all__ = [
    "Domain",
    "mpp_define_domains",
    "mpp_define_layout",
    "mpp_get_neighbor_pe",
    "mpp_max",
    "mpp_min",
    "mpp_reduce_scatter",
    "mpp_sum",
    "mpp_update_domains",
]


@dataclass(frozen=True)
class Domain:
    """This rank's partition: FMS's ``domain2D``/``domain1D``, no data or labels.

    dims : partitioned dimension names.
    global_sizes, starts, stops : global length and this rank's
        half-open ``[start, stop)`` interval per dim (FMS's ``isc:iec``).
    comm : communicator owning the full global array.
    cart : ``{grid_shape, coords, periods}`` for a multi-dim partition,
        else ``None``.
    """

    dims: tuple[str, ...]
    global_sizes: dict[str, int]
    starts: dict[str, int]
    stops: dict[str, int]
    comm: MPI.Comm
    cart: dict[str, Any] | None = field(default=None)

    def local_size(self, dim: str) -> int:
        return self.stops[dim] - self.starts[dim]

    def is_global_edge(self, dim: str, side: str) -> bool:
        """``side`` is "lower" or "upper"."""
        if side == "lower":
            return self.starts[dim] == 0
        if side == "upper":
            return self.stops[dim] == self.global_sizes[dim]
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}.")

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any], comm: MPI.Comm) -> Domain:
        """Build from climtools' ``.meta`` dict (see ``xarray/meta.py``)."""
        dims = tuple(str(d) for d in meta["dims"])
        return cls(
            dims=dims,
            global_sizes={d: int(meta["global_sizes"][d]) for d in dims},
            starts={d: int(meta["starts"][d]) for d in dims},
            stops={d: int(meta["stops"][d]) for d in dims},
            comm=comm,
            cart=dict(meta["cart"]) if meta.get("cart") is not None else None,
        )

    def to_meta(self, *, chunk_info: Mapping[str, int]) -> dict[str, Any]:
        """Inverse of :meth:`from_meta`."""
        meta: dict[str, Any] = {
            "dims": self.dims,
            "global_sizes": dict(self.global_sizes),
            "starts": dict(self.starts),
            "stops": dict(self.stops),
            "dim": self.dims[0],
            "global_size": self.global_sizes[self.dims[0]],
            "start": self.starts[self.dims[0]],
            "stop": self.stops[self.dims[0]],
            "chunk_info": {
                str(name): int(size) for name, size in chunk_info.items() if size > 0
            },
        }
        if self.cart is not None:
            meta["cart"] = dict(self.cart)
        return meta


def mpp_reduce_scatter(
    local: np.ndarray,
    op: MPI.Op,
    comm: MPI.Comm,
    recvcounts: Sequence[int],
    *,
    axis: int = 0,
) -> np.ndarray:
    """Elementwise-reduce ``local`` (identical shape on every rank) so each
    rank keeps only its own contiguous slice along ``axis`` -- MPI's
    ``Reduce_scatter``, generalized to any axis via ``moveaxis`` (it
    natively splits a flat buffer, so ``axis`` is brought to position 0,
    split by element count, then moved back). Where :func:`_mpp_reduce`'s
    ``Allreduce`` gives every rank the full result (needed when every rank
    genuinely wants a full copy), this gives each rank only the part it
    will end up owning -- for a result about to be redistributed anyway,
    that's the same answer without ever materializing the full array on
    every rank. ``recvcounts[r]`` is rank ``r``'s share of ``axis``, in
    elements along that axis (not flattened count); their sum must equal
    ``local.shape[axis]``.
    """
    moved = np.ascontiguousarray(np.moveaxis(local, axis, 0))
    per_slice = moved[0].size if moved.ndim > 1 else 1
    flat_counts = [c * per_slice for c in recvcounts]
    recvbuf = np.empty(flat_counts[comm.rank], dtype=moved.dtype)
    comm.Reduce_scatter(moved.reshape(-1), recvbuf, recvcounts=flat_counts, op=op)
    my_len = recvcounts[comm.rank]
    shape = (my_len, *moved.shape[1:]) if moved.ndim > 1 else (my_len,)
    return np.moveaxis(recvbuf.reshape(shape), 0, axis)


def mpp_define_layout(extent0: int, extent1: int, ndivs: int) -> tuple[int, int]:
    """FMS's ``mpp_define_layout2D`` (``mpp_domains_define.inc``): guess the
    aspect-ratio-matching divisor, walk down until it evenly divides
    ``ndivs``. 2D only -- the only case climtools partitions.
    """
    idiv = max(round(math.sqrt(ndivs * extent0 / extent1)), 1)
    while ndivs % idiv != 0:
        idiv -= 1
    return idiv, ndivs // idiv


def mpp_define_domains(
    mpi_context: MPIContext,
    global_sizes: Mapping[str, int],
    dims: str | Sequence[str],
    *,
    min_partition_size: int | Mapping[str, int] | None = None,
    rank: int | None = None,
) -> Domain:
    """FMS's ``mpp_define_domains``. One dim: balanced ``[start, stop)``
    slabs (``chunks.get_balanced_bounds``). Two or more: a Cartesian
    process grid (``cartesian.get_cartesian_topology``). ``min_partition_size``
    is ``get_balanced_bounds``'s ``min_chunk``, per dimension for either case.

    ``rank`` defaults to the caller's own rank (the common case: "what is
    my own domain"). Pass another rank's number to compute *its* domain
    instead -- e.g. a root building every rank's piece for a scatter,
    the way FMS itself keeps every PE's domain in one shared table rather
    than each PE only knowing its own. For the multi-dim case this bypasses
    ``get_cartesian_topology``'s cached, calling-rank-only Cartcomm lookup
    in favor of computing that rank's grid coordinates directly (the same
    row-major mapping ``get_cartesian_topology`` itself relies on for a
    non-reordered communicator, confirmed to agree with it across many
    rank counts before this was ever used for a rank other than one's own).
    """
    from .chunks import get_balanced_bounds

    comm = mpi_context.comm
    target_rank = comm.rank if rank is None else rank
    dim_tuple = (dims,) if isinstance(dims, str) else tuple(dims)

    def _min_chunk(d: str) -> int | None:
        return (
            min_partition_size
            if not isinstance(min_partition_size, Mapping)
            else min_partition_size.get(d)
        )

    if len(dim_tuple) == 1:
        dim = dim_tuple[0]
        length = int(global_sizes[dim])
        start, stop = get_balanced_bounds(length, target_rank, comm.size, _min_chunk(dim))
        return Domain(
            dims=dim_tuple,
            global_sizes={dim: length},
            starts={dim: start},
            stops={dim: stop},
            comm=comm,
        )

    sizes = {d: int(global_sizes[d]) for d in dim_tuple}

    if target_rank == comm.rank:
        from .cartesian import get_cartesian_topology

        topology = get_cartesian_topology(comm, dim_tuple, sizes)
        grid_shape = topology.grid_shape
        starts = {d: topology.bounds[d][0] for d in dim_tuple}
        stops = {d: topology.bounds[d][1] for d in dim_tuple}
        cart = topology.as_meta_cart()
    else:
        grid_shape = mpp_define_layout(sizes[dim_tuple[0]], sizes[dim_tuple[1]], comm.size)
        coords = tuple(int(c) for c in np.unravel_index(target_rank, grid_shape))
        starts, stops = {}, {}
        for axis, d in enumerate(dim_tuple):
            s, e = get_balanced_bounds(
                sizes[d], coords[axis], grid_shape[axis], _min_chunk(d)
            )
            starts[d], stops[d] = s, e
        cart = {"grid_shape": grid_shape, "coords": coords, "periods": (False,) * len(dim_tuple)}

    return Domain(
        dims=dim_tuple,
        global_sizes=sizes,
        starts=starts,
        stops=stops,
        comm=comm,
        cart=cart,
    )


def _mpp_reduce(
    local: np.ndarray, op: MPI.Op, comm: MPI.Comm | None, domain: Domain | None = None
) -> np.ndarray:
    """Global elementwise reduction under any MPI op (generic kernel behind
    ``mpp_sum``/``mpp_max``/``mpp_min``): a single ``Allreduce``, as FMS's
    ``MPP_REDUCE_`` (``mpp_reduce_mpi.fh``) does. No agreement handshake --
    unlike ``mpp_comm_reduce`` above this, FMS trusts SPMD by construction, and
    so does this: every rank must reach it with the same op and a
    compatible ``local`` shape/dtype. Always calls ``Allreduce`` even for a
    size-1 comm (cheap no-op in MPI; a short-circuit here broke
    ``mpp_comm_reduce``'s replicated-axis path, which can hand different ranks
    different-sized comms).

    ``comm`` takes priority over ``domain`` (used only for its ``.comm``)
    when both are given.
    """
    active_comm = comm if comm is not None else (domain.comm if domain else MPI.COMM_WORLD)
    recv = np.empty_like(local)
    active_comm.Allreduce(local, recv, op=op)
    return recv


def mpp_sum(
    local: np.ndarray, domain: Domain | None = None, *, comm: MPI.Comm | None = None
) -> np.ndarray:
    """FMS's ``mpp_sum``. Pass ``domain`` or ``comm``."""
    return _mpp_reduce(local, MPI.SUM, comm, domain)


def mpp_max(
    local: np.ndarray, domain: Domain | None = None, *, comm: MPI.Comm | None = None
) -> np.ndarray:
    """FMS's ``mpp_max``."""
    return _mpp_reduce(local, MPI.MAX, comm, domain)


def mpp_min(
    local: np.ndarray, domain: Domain | None = None, *, comm: MPI.Comm | None = None
) -> np.ndarray:
    """FMS's ``mpp_min``."""
    return _mpp_reduce(local, MPI.MIN, comm, domain)


def mpp_get_neighbor_pe(
    domain: Domain, dim: str, *, periodic: bool = False
) -> tuple[int | None, int | None]:
    """FMS's ``mpp_get_neighbor_pe``, adapted to return both directions in
    one call (FMS takes a ``direction`` argument and returns one PE per
    call) rather than one call per side -- ``Domain`` only has two
    directions per axis, so there's no direction enum to take. Single
    dim: linear ``rank -+ 1``. Multi-dim: Cartesian face neighbor (not
    ``rank -+ 1`` in general). ``None`` on a non-periodic edge.
    """
    comm = domain.comm
    rank = comm.rank

    if len(domain.dims) > 1:
        from .cartesian import get_cartesian_topology

        topology = get_cartesian_topology(comm, domain.dims, domain.global_sizes)
        if periodic:
            axis = domain.dims.index(dim)
            axis_size = topology.grid_shape[axis]
            coords = list(topology.coords)
            coords[axis] = (topology.coords[axis] - 1) % axis_size
            left_rank = topology.cart_comm.Get_cart_rank(coords)
            coords[axis] = (topology.coords[axis] + 1) % axis_size
            right_rank = topology.cart_comm.Get_cart_rank(coords)
            return left_rank, right_rank
        return topology.neighbors[dim]

    size = comm.size
    if periodic:
        return (rank - 1) % size, (rank + 1) % size
    left_rank = rank - 1 if rank > 0 else None
    right_rank = rank + 1 if rank < size - 1 else None
    return left_rank, right_rank


def mpp_update_domains(
    fields: np.ndarray | Mapping[str, np.ndarray],
    domain: Domain,
    dim: str,
    axis: int,
    *,
    before: int,
    after: int,
    periodic: bool = False,
    left_rank: int | None = None,
    right_rank: int | None = None,
) -> tuple[np.ndarray | dict[str, np.ndarray], int, int]:
    """FMS's ``mpp_update_domains``: halo exchange with the neighbors along
    ``dim``, via nonblocking ``Isend``/``Irecv`` + one shared ``Waitall``.
    A ``Mapping`` of same-``axis``-length fields is FMS's *group update*
    (all messages batched into one ``Waitall``, not one round trip per
    field); a single array is the plain case, and the return shape
    mirrors whichever was passed. ``before``/``after`` are the halo width
    from the lower/upper neighbor; at a non-periodic global edge that
    side is left unpadded (``left_pad``/``right_pad`` report 0 there
    instead of raising). ``left_rank``/``right_rank`` default to
    :func:`mpp_get_neighbor_pe`; a Cartesian caller should pass them
    explicitly (its neighbors are not ``rank -+ 1``).
    """
    single = isinstance(fields, np.ndarray)
    items: dict[str, np.ndarray] = {"": fields} if single else dict(fields)

    comm = domain.comm
    if left_rank is None or right_rank is None:
        default_left, default_right = mpp_get_neighbor_pe(domain, dim, periodic=periodic)
        left_rank = default_left if left_rank is None else left_rank
        right_rank = default_right if right_rank is None else right_rank

    def _view(arr: np.ndarray) -> np.ndarray:
        """View as a dtype the raw MPI buffer protocol accepts."""
        return arr.view(np.int64) if arr.dtype.kind in "mM" else arr

    def _slab(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return np.ascontiguousarray(arr[tuple(idx)])

    send_right = {
        name: _view(_slab(arr, arr.shape[axis] - before, arr.shape[axis]))
        for name, arr in items.items()
        if right_rank is not None and before > 0
    }
    send_left = {
        name: _view(_slab(arr, 0, after))
        for name, arr in items.items()
        if left_rank is not None and after > 0
    }
    recv_before = {
        name: np.empty(
            [*arr.shape[:axis], before, *arr.shape[axis + 1 :]], dtype=arr.dtype
        )
        for name, arr in items.items()
        if left_rank is not None and before > 0
    }
    recv_after = {
        name: np.empty(
            [*arr.shape[:axis], after, *arr.shape[axis + 1 :]], dtype=arr.dtype
        )
        for name, arr in items.items()
        if right_rank is not None and after > 0
    }

    # Post every message for every field, then wait once (the group update).
    recv_reqs = [comm.Irecv(_view(buf), source=left_rank) for buf in recv_before.values()]
    recv_reqs += [comm.Irecv(_view(buf), source=right_rank) for buf in recv_after.values()]
    send_reqs = [comm.Isend(arr, dest=right_rank) for arr in send_right.values()]
    send_reqs += [comm.Isend(arr, dest=left_rank) for arr in send_left.values()]

    MPI.Request.Waitall(recv_reqs)
    MPI.Request.Waitall(send_reqs)

    padded = {
        name: np.concatenate(
            [p for p in (recv_before.get(name), arr, recv_after.get(name)) if p is not None],
            axis=axis,
        )
        for name, arr in items.items()
    }
    left_pad = before if recv_before else 0
    right_pad = after if recv_after else 0
    return (padded[""] if single else padded), left_pad, right_pad
