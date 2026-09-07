"""FMS/mpp-style distributed-memory primitives, adapted from GFDL FMS.

Covers three FMS layers: the domain decomposition and domain table of
``mpp_domains_mod`` (``mpp_define_domains``, ``mpp_get_compute_domains``,
``mpp_get_neighbor_pe``, ``mpp_update_domains``), the plain collectives of
``mpp_mod`` (``mpp_sum``/``mpp_max``/``mpp_min``), and the extended
fixed-point reductions of ``mpp_efp_mod`` (``mpp_reproducing_sum``, plus the
product analogue ``mpp_reproducing_prod`` that FMS has no counterpart for).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..mpi.context import MPIContext

__all__ = [
    "MAX_EFP_RANKS",
    "MAX_PROD_RANKS",
    "PROD_EXPONENT",
    "PROD_INF",
    "PROD_NAN",
    "PROD_NEGATIVE",
    "PROD_ZERO",
    "Domain",
    "DomainUpdate",
    "mpp_chksum",
    "mpp_complete_update_domains",
    "mpp_define_domains",
    "mpp_define_layout",
    "mpp_get_compute_domains",
    "mpp_get_neighbor_pe",
    "mpp_max",
    "mpp_min",
    "mpp_partition_offsets",
    "mpp_prod_decompose",
    "mpp_prod_recombine",
    "mpp_reduce_scatter",
    "mpp_reproducing_prod",
    "mpp_reproducing_sum",
    "mpp_slice_compute_domain",
    "mpp_start_update_domains",
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
    """Choose a 2D process grid for ``ndivs`` ranks over ``extent0 x extent1``.

    FMS's ``mpp_define_layout2D`` guesses the aspect-ratio-matching divisor,
    ``nint(sqrt(ndivs*isz/jsz))``, then walks it *down* until it divides
    ``ndivs``. That search is one-directional, so it never reconsiders a
    divisor above the guess and can land on a needlessly lopsided grid: for
    721x1440 on 4 ranks it returns 1x4, giving 721x360 subdomains, where 2x2
    gives 360x720 and a slightly smaller halo perimeter. FMS's own comment on
    the neighbouring balancing routine concedes the point -- "It is very hard
    to make it balance for all the situation. Hopefully some smart idea will
    come up someday."

    ``ndivs`` has few divisors, so the exact search FMS approximates is
    affordable: enumerate every factor pair and take the one minimising the
    per-subdomain halo perimeter ``extent0/rows + extent1/cols``, which is the
    quantity the aspect-ratio heuristic is a proxy for. Ties go to the more
    square grid.

    Layouts giving every rank points are preferred. FMS treats an empty
    subdomain as fatal (``mpp_compute_extent``: "domain extents must be
    positive definite"), but climtools tolerates them deliberately --
    ``get_balanced_bounds`` hands out empty ``(length, length)`` slabs when a
    dimension is shorter than the rank count -- so when no factor pair fits,
    the one leaving the fewest ranks idle is returned rather than raising.
    """
    if ndivs < 1:
        raise ValueError(f"ndivs must be positive, got {ndivs}.")

    pairs = [(rows, ndivs // rows) for rows in range(1, ndivs + 1) if ndivs % rows == 0]

    def cost(layout: tuple[int, int]) -> tuple[int, float, int]:
        rows, cols = layout
        # Ranks left with nothing dominate; then the halo perimeter of one
        # subdomain; then squareness, purely to make ties deterministic.
        idle = max(0, rows - extent0) * cols + max(0, cols - extent1) * rows
        perimeter = extent0 / rows + extent1 / cols
        return idle, perimeter, abs(rows - cols)

    return min(pairs, key=cost)


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
        start, stop = get_balanced_bounds(
            length, target_rank, comm.size, _min_chunk(dim)
        )
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
        grid_shape = mpp_define_layout(
            sizes[dim_tuple[0]], sizes[dim_tuple[1]], comm.size
        )
        coords = tuple(int(c) for c in np.unravel_index(target_rank, grid_shape))
        starts, stops = {}, {}
        for axis, d in enumerate(dim_tuple):
            s, e = get_balanced_bounds(
                sizes[d], coords[axis], grid_shape[axis], _min_chunk(d)
            )
            starts[d], stops[d] = s, e
        cart = {
            "grid_shape": grid_shape,
            "coords": coords,
            "periods": (False,) * len(dim_tuple),
        }

    return Domain(
        dims=dim_tuple,
        global_sizes=sizes,
        starts=starts,
        stops=stops,
        comm=comm,
        cart=cart,
    )


def mpp_get_compute_domains(
    global_size: int,
    dim_size: int,
    *,
    min_partition_size: int | None = None,
) -> list[tuple[int, int]]:
    """FMS's ``mpp_get_compute_domains``: every rank's ``[begin, end)`` on ``dim``.

    FMS fills ``domain%list(0:ndivs-1)`` inside ``mpp_define_domains`` and every
    PE keeps the whole table, so ``mpp_get_compute_domains`` answers "which PE
    owns which indices" by a local array read rather than a collective. The
    decomposition rule is deterministic, so this reconstructs the same table
    from ``global_size`` and the number of divisions alone -- no communication,
    and no dependence on any rank's own position.

    ``dim_size`` is the number of divisions along the dimension: the
    communicator size for a 1-D partition, or that axis's extent in the
    Cartesian process grid.
    """
    from .chunks import get_balanced_bounds

    return [
        get_balanced_bounds(int(global_size), rank, int(dim_size), min_partition_size)
        for rank in range(int(dim_size))
    ]


def mpp_slice_compute_domain(
    start: int,
    stop: int,
    requested_start: int,
    requested_stop: int,
) -> tuple[int, int, int]:
    """Intersect one rank's compute domain with a global slice, without communicating.

    Returns ``(local_start, local_stop, new_global_start)``: the half-open
    interval to take from this rank's local array, and the offset this rank's
    surviving elements occupy in the sliced global array.

    The third value is what a naive implementation reaches for an ``allgather``
    to obtain. It needs no communication because a compute-domain partition is
    contiguous and ordered by rank: everything owned by lower ranks is exactly
    the global interval ``[0, start)``, so the number of lower-rank elements
    surviving the slice is ``|[0, start) & [requested_start, requested_stop)|``,
    which this rank can evaluate from its own bounds alone. This is the same
    property FMS relies on when it reads ``domain%list(:)`` locally instead of
    querying other PEs.
    """
    lower = max(requested_start, start)
    upper = max(lower, min(requested_stop, stop))
    below = max(0, min(requested_stop, start) - requested_start)
    return lower - start, upper - start, below


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
    active_comm = (
        comm if comm is not None else (domain.comm if domain else MPI.COMM_WORLD)
    )
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


@dataclass
class DomainUpdate:
    """In-flight halo exchange, as returned by :func:`mpp_start_update_domains`.

    FMS splits ``mpp_update_domains`` into a ``start``/``complete`` pair so a
    PE can compute on its interior -- which needs no neighbour data -- while
    the boundary exchange is still on the wire. This carries the state
    between the two halves: the posted requests, the receive buffers, and
    enough layout to unpack them.
    """

    items: dict[str, np.ndarray]
    groups: dict[Any, list[str]]
    recv_bufs: dict[tuple[Any, str], np.ndarray]
    recv_reqs: list[Any]
    send_reqs: list[Any]
    axis: int
    before: int
    after: int
    single: bool
    unpack: Any


def mpp_start_update_domains(
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
) -> DomainUpdate:
    """FMS's ``mpp_start_update_domains``: post a halo exchange with the neighbors along
    ``dim``, via nonblocking ``Isend``/``Irecv`` + one shared ``Waitall``.
    A ``Mapping`` of same-``axis``-length fields is FMS's *group update*:
    every field's halo slab is packed into one contiguous buffer per
    neighbor per dtype -- FMS's own ``buffer_pos`` accumulation in
    ``mpp_do_update.fh`` -- so message count per exchange is
    ``2 * (distinct dtypes present)``, not ``2 * (fields present)``. FMS
    groups this same way, not across dtypes: its group-update routines
    are compiled per Fortran kind (``mpp_do_update_r8_3d``/``_r4_3d``/
    ``_i4_3d``, ...), so a Fortran group update never mixes dtypes into
    one buffer either -- this mirrors that constraint rather than forcing
    heterogeneous dtypes into one raw byte buffer, which FMS has no
    equivalent for. A single array is the plain case (one field, one
    dtype, same code path), and the return shape mirrors whichever was
    passed. ``before``/``after`` are the halo width from the lower/upper
    neighbor; at a non-periodic global edge that side is left unpadded
    (``left_pad``/``right_pad`` report 0 there instead of raising).
    ``left_rank``/``right_rank`` default to :func:`mpp_get_neighbor_pe`;
    a Cartesian caller should pass them explicitly (its neighbors are not
    ``rank -+ 1``).
    """
    single = isinstance(fields, np.ndarray)
    items: dict[str, np.ndarray] = {"": fields} if single else dict(fields)

    comm = domain.comm
    if left_rank is None or right_rank is None:
        default_left, default_right = mpp_get_neighbor_pe(
            domain, dim, periodic=periodic
        )
        left_rank = default_left if left_rank is None else left_rank
        right_rank = default_right if right_rank is None else right_rank

    def _view(arr: np.ndarray) -> np.ndarray:
        """View as a dtype the raw MPI buffer protocol accepts."""
        return arr.view(np.int64) if arr.dtype.kind in "mM" else arr

    def _slab(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return np.ascontiguousarray(arr[tuple(idx)])

    def _halo_shape(name: str, width: int) -> tuple[int, ...]:
        arr = items[name]
        return (*arr.shape[:axis], width, *arr.shape[axis + 1 :])

    # Group by wire dtype, sorted for a deterministic pack/unpack order
    # every rank derives independently from its own, structurally
    # identical copy of `items` -- no layout information travels over
    # the wire, only each dtype group's one packed buffer.
    groups: dict[np.dtype, list[str]] = {}
    for name, arr in items.items():
        groups.setdefault(_view(arr).dtype, []).append(name)
    for names in groups.values():
        names.sort()

    def _pack(names: list[str], side: str) -> np.ndarray:
        pieces = []
        for name in names:
            arr = items[name]
            slab = (
                _slab(arr, arr.shape[axis] - before, arr.shape[axis])
                if side == "right"
                else _slab(arr, 0, after)
            )
            pieces.append(_view(slab).reshape(-1))
        return np.concatenate(pieces)

    def _unpack(
        flat: np.ndarray, names: list[str], width: int
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        pos = 0
        for name in names:
            shape = _halo_shape(name, width)
            count = int(np.prod(shape)) if shape else 1
            # `flat` carries the group's wire dtype (e.g. int64 for a
            # datetime64/timedelta64 field, per `_view`); reinterpret each
            # field's slice back to its own original dtype before handing
            # it back -- `padded`'s concatenate below needs it to match
            # `arr`'s dtype, not the wire view's.
            out[name] = flat[pos : pos + count].reshape(shape).view(items[name].dtype)
            pos += count
        return out

    can_send_right = right_rank is not None and before > 0
    can_send_left = left_rank is not None and after > 0
    can_recv_before = left_rank is not None and before > 0
    can_recv_after = right_rank is not None and after > 0

    recv_bufs: dict[tuple[np.dtype, str], np.ndarray] = {}
    recv_reqs = []
    for dtype, names in groups.items():
        if can_recv_before:
            count = sum(int(np.prod(_halo_shape(name, before))) for name in names)
            buf = np.empty(count, dtype=dtype)
            recv_bufs[dtype, "before"] = buf
            recv_reqs.append(comm.Irecv(buf, source=left_rank))
        if can_recv_after:
            count = sum(int(np.prod(_halo_shape(name, after))) for name in names)
            buf = np.empty(count, dtype=dtype)
            recv_bufs[dtype, "after"] = buf
            recv_reqs.append(comm.Irecv(buf, source=right_rank))

    send_reqs = []
    for dtype, names in groups.items():
        if can_send_right:
            send_reqs.append(comm.Isend(_pack(names, "right"), dest=right_rank))
        if can_send_left:
            send_reqs.append(comm.Isend(_pack(names, "left"), dest=left_rank))

    return DomainUpdate(
        items=items,
        groups=groups,
        recv_bufs=recv_bufs,
        recv_reqs=recv_reqs,
        send_reqs=send_reqs,
        axis=axis,
        before=before,
        after=after,
        single=single,
        unpack=_unpack,
    )


def mpp_complete_update_domains(
    update: DomainUpdate,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int]:
    """FMS's ``mpp_complete_update_domains``: wait, then hand back the halos.

    Returns ``(recv_before, recv_after, left_pad, right_pad)``. Deliberately
    *not* the padded array: callers that only need the neighbours' slabs --
    which is every caller in climtools, since the xarray layer re-joins the
    pieces itself -- would otherwise pay a full copy of their own local array
    to build a padded buffer whose interior is then discarded. In FMS terms
    this is the difference between writing into the halo ring of an existing
    data domain and reallocating the whole data domain per exchange.
    """
    MPI.Request.Waitall(update.recv_reqs)
    MPI.Request.Waitall(update.send_reqs)

    recv_before: dict[str, np.ndarray] = {}
    recv_after: dict[str, np.ndarray] = {}
    for dtype, names in update.groups.items():
        if (dtype, "before") in update.recv_bufs:
            recv_before.update(
                update.unpack(update.recv_bufs[dtype, "before"], names, update.before)
            )
        if (dtype, "after") in update.recv_bufs:
            recv_after.update(
                update.unpack(update.recv_bufs[dtype, "after"], names, update.after)
            )

    return (
        recv_before,
        recv_after,
        update.before if recv_before else 0,
        update.after if recv_after else 0,
    )


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
    """FMS's blocking ``mpp_update_domains``: start, complete, and join.

    Kept for callers that want the padded array in one call. Anything that
    only needs the neighbours' slabs, or that has interior work to overlap
    with the exchange, should use :func:`mpp_start_update_domains` and
    :func:`mpp_complete_update_domains` directly and skip the concatenate
    below -- which copies the entire local array.
    """
    update = mpp_start_update_domains(
        fields,
        domain,
        dim,
        axis,
        before=before,
        after=after,
        periodic=periodic,
        left_rank=left_rank,
        right_rank=right_rank,
    )
    recv_before, recv_after, left_pad, right_pad = mpp_complete_update_domains(update)

    padded = {
        name: np.concatenate(
            [
                piece
                for piece in (recv_before.get(name), arr, recv_after.get(name))
                if piece is not None
            ],
            axis=axis,
        )
        for name, arr in update.items.items()
    }
    return (padded[""] if update.single else padded), left_pad, right_pad


# FMS: NUMBIT = 46, NUMINT = 6 (mpp_efp.F90). Six base-2**46 digits span
# 2**(2*46) down to 2**(-3*46), i.e. ~1e27 down to ~1e-42 in magnitude,
# which covers the dynamic range of geophysical fields in float64.
_NUMBIT = 46
_NUMINT = 6
_PREC = float(2**_NUMBIT)

# FMS: max_count_prec = 2**(63-NUMBIT)-1. Number of terms that can be added
# into one int64 digit before the accumulator itself can overflow.
MAX_EFP_RANKS = 2 ** (63 - _NUMBIT) - 1

# Scale of digit n, n = 0..NUMINT-1 (FMS's `pr` array).
_SCALES = np.array([_PREC ** (2 - n) for n in range(_NUMINT)], dtype=np.float64)

# The rank-combining product below multiplies one mantissa in [0.5, 1) per
# rank, so the reduced mantissa is >= 2**-nranks. Keeping it a normal
# float64 (>= 2**-1022) bounds the usable rank count.
MAX_PROD_RANKS = 1000

# Longest run of mantissas multiplied before renormalising. Each is in
# [0.5, 1), so a block product is >= 2**-_PROD_BLOCK, still normal in
# float64. frexp renormalisation is exact, so blocking changes nothing
# except the exponent bookkeeping.
_PROD_BLOCK = 512


def _to_digits(array: np.ndarray, axis: int) -> np.ndarray:
    """Split ``array`` into ``_NUMINT`` signed integer digits along a new axis 0.

    The NumPy analogue of FMS's ``real_to_ints`` + ``increment_ints``: the
    digits are extracted in descending scale order and each is exact, so
    summing them as int64 reproduces the real sum to within the truncation
    of the smallest digit, independent of order.
    """
    values = np.asarray(array, dtype=np.float64)
    sign = np.where(values < 0.0, -1.0, 1.0)
    residual = np.abs(values)
    digits = np.empty((_NUMINT, *values.shape), dtype=np.int64)
    for n, scale in enumerate(_SCALES):
        digit = np.floor(residual / scale)
        digits[n] = (sign * digit).astype(np.int64)
        residual -= digit * scale
    return digits.sum(axis=axis + 1 if axis >= 0 else axis, dtype=np.int64)


def _from_digits(digits: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_to_digits`, summing smallest scale first."""
    total = np.zeros(digits.shape[1:], dtype=np.float64)
    for n in range(_NUMINT - 1, -1, -1):
        total += digits[n].astype(np.float64) * _SCALES[n]
    return total


def mpp_reproducing_sum(
    local: np.ndarray,
    comm: MPI.Comm,
    *,
    axis: int | None = None,
) -> np.ndarray:
    """FMS's ``mpp_reproducing_sum``: a sum invariant to rank count and order.

    ``local`` is this rank's slab. When ``axis`` is given the local slab is
    first reduced along that axis; the remaining shape must then match on
    every rank. The reduction itself is a single integer ``Allreduce``, so
    the result is bitwise identical for any partition of the same global
    array across any number of ranks.

    Raises ``ValueError`` when the communicator is large enough that the
    int64 digit accumulators could overflow, mirroring FMS's
    ``max_count_prec`` check.
    """
    if comm.size > MAX_EFP_RANKS:
        raise ValueError(
            f"mpp_reproducing_sum: {comm.size} ranks exceeds the "
            f"{MAX_EFP_RANKS}-rank limit set by the {_NUMBIT}-bit EFP digit "
            "width; the integer accumulators could overflow."
        )
    if not np.all(np.isfinite(local)):
        raise ValueError(
            "mpp_reproducing_sum: input contains NaN or infinity, which have "
            "no extended-fixed-point representation."
        )
    flat = np.asarray(local).reshape(-1) if axis is None else local
    digits = _to_digits(flat, 0 if axis is None else axis)
    total = np.empty_like(digits)
    comm.Allreduce(np.ascontiguousarray(digits), total, op=MPI.SUM)
    return _from_digits(total)


# Order of the integer companions packed alongside the mantissa, so that the
# exponent and every exceptional-value tally travel in one int64 Allreduce
# instead of four.
PROD_EXPONENT, PROD_NAN, PROD_INF, PROD_ZERO, PROD_NEGATIVE = range(5)
_PROD_FIELDS = 5


def _moved_to_front(values: np.ndarray, axes: Sequence[int]) -> np.ndarray:
    """Collapse ``axes`` into a single leading axis, preserving the rest."""
    ordered = tuple(a % values.ndim for a in axes)
    moved = np.moveaxis(values, ordered, range(len(ordered)))
    kept = moved.shape[len(ordered) :]
    return moved.reshape((-1, *kept))


def mpp_prod_decompose(
    local: np.ndarray, axes: int | Sequence[int] = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``local`` along ``axes`` into an over/underflow-free partial product.

    Returns ``(mantissa, companions)``. ``mantissa`` is float64 in ``[0.5, 1)``
    and ``companions`` is an int64 array whose leading axis is indexed by the
    ``PROD_*`` constants: the base-2 exponent, and the counts of NaN, infinite,
    zero and negative entries. Both have the shape of the reduced result, so
    combining two partials is a ``PROD`` on the mantissa and a ``SUM`` on the
    companions -- both associative and commutative, which is what makes the
    distributed product agree with the serial one whatever the rank count.

    The mantissa is renormalised with ``frexp`` every ``_PROD_BLOCK`` terms.
    ``frexp`` only moves bits between the significand and the exponent field,
    so blocking is exact: the rounding is identical to multiplying the same
    values directly in float64, minus the overflow.
    """
    axis_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
    work = _moved_to_front(np.asarray(local).astype(np.float64, copy=False), axis_tuple)

    is_nan = np.isnan(work)
    is_inf = np.isinf(work)
    is_zero = work == 0.0
    ordinary = ~(is_nan | is_inf | is_zero)
    magnitude = np.where(ordinary, np.abs(work), 1.0)

    mantissa = np.ones(work.shape[1:], dtype=np.float64)
    exponent = np.zeros(work.shape[1:], dtype=np.int64)
    for start in range(0, work.shape[0], _PROD_BLOCK):
        block_mantissa, block_exponent = np.frexp(
            magnitude[start : start + _PROD_BLOCK]
        )
        mantissa *= block_mantissa.prod(axis=0)
        exponent += block_exponent.sum(axis=0, dtype=np.int64)
        mantissa, extra = np.frexp(mantissa)
        exponent += extra

    companions = np.empty((_PROD_FIELDS, *exponent.shape), dtype=np.int64)
    companions[PROD_EXPONENT] = exponent
    companions[PROD_NAN] = np.count_nonzero(is_nan, axis=0)
    companions[PROD_INF] = np.count_nonzero(is_inf, axis=0)
    companions[PROD_ZERO] = np.count_nonzero(is_zero, axis=0)
    companions[PROD_NEGATIVE] = np.count_nonzero(np.signbit(work) & ~is_nan, axis=0)
    return mantissa, companions


def mpp_prod_recombine(
    mantissa: np.ndarray,
    companions: np.ndarray,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Rebuild a product from the reduced output of :func:`mpp_prod_decompose`.

    The exceptional cases are decided from the exact global tallies rather than
    from floating-point accidents, which is the whole point of carrying them:
    a plain distributed product evaluates ``inf * 0`` whenever overflow and a
    zero land on different ranks, so its answer depends on the rank count.

    * any NaN input, or a zero and an infinity anywhere in the global array,
      gives NaN;
    * otherwise any zero gives a correctly signed zero, and any infinity a
      correctly signed infinity;
    * otherwise ``ldexp(mantissa, exponent)``, which is infinite only when the
      true product genuinely overflows ``dtype``.
    """
    n_nan = companions[PROD_NAN]
    n_inf = companions[PROD_INF]
    n_zero = companions[PROD_ZERO]
    sign = np.where(companions[PROD_NEGATIVE] % 2 == 1, -1.0, 1.0)

    # ldexp takes a C int exponent; clipping is safe because anything beyond
    # this range has already saturated the float64 result either way.
    exponent = np.clip(companions[PROD_EXPONENT], -32768, 32768).astype(np.int32)
    with np.errstate(over="ignore"):
        result = sign * np.ldexp(mantissa, exponent)

    result = np.where(n_zero > 0, sign * 0.0, result)
    result = np.where(n_inf > 0, sign * np.inf, result)
    result = np.where((n_zero > 0) & (n_inf > 0), np.nan, result)
    result = np.where(n_nan > 0, np.nan, result)
    return result if dtype is None else result.astype(dtype, copy=False)


def mpp_reproducing_prod(
    local: np.ndarray,
    comm: MPI.Comm,
    *,
    axis: int | Sequence[int] = 0,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Product along ``axis`` of a distributed array, invariant to rank count.

    The multiplicative counterpart of :func:`mpp_reproducing_sum`. FMS has no
    equivalent -- ``mpp_efp_mod`` only covers sums -- but the requirement is
    the same one FMS states for reductions, that the answer must not depend on
    how the global array was divided among PEs.

    Two collectives: a ``PROD`` on the mantissa and a ``SUM`` on the packed
    integer companions. See :func:`mpp_prod_decompose` for the representation
    and :func:`mpp_prod_recombine` for the reassembly.
    """
    if comm.size > MAX_PROD_RANKS:
        raise ValueError(
            f"mpp_reproducing_prod: {comm.size} ranks exceeds the "
            f"{MAX_PROD_RANKS}-rank limit; the combined mantissa would "
            "become subnormal and lose precision."
        )
    values = np.asarray(local)
    out_dtype = np.dtype(dtype) if dtype is not None else values.dtype
    mantissa, companions = mpp_prod_decompose(values, axis)

    global_mantissa = np.empty_like(mantissa)
    comm.Allreduce(np.ascontiguousarray(mantissa), global_mantissa, op=MPI.PROD)
    global_companions = np.empty_like(companions)
    comm.Allreduce(np.ascontiguousarray(companions), global_companions, op=MPI.SUM)
    return mpp_prod_recombine(global_mantissa, global_companions, out_dtype)


def mpp_chksum(
    local: np.ndarray,
    comm: MPI.Comm | None = None,
    *,
    mask_val: float | None = None,
) -> int:
    """FMS's ``mpp_chksum``: a bitwise checksum of a distributed field.

    FMS reinterprets each element's bits as an integer and sums those with
    ``mpp_sum`` (``mpp/include/mpp_chksum.fh``). Integer addition is exact and
    commutative, so the checksum depends only on the set of values in the
    global field, never on how it was divided among PEs or in what order the
    ranks were combined. Two runs at different rank counts that agree bitwise
    give the same number; two differing in one bit of one element almost
    certainly do not.

    That is the cheap validation the correctness suite otherwise lacks: one
    integer per field, rather than an elementwise comparison against a
    gathered reference that needs the whole field on one rank.

    ``mask_val`` excludes a sentinel (a fill value or a land mask) from the
    sum, as FMS's own ``mask_val`` argument does. NaN is matched as a NaN
    rather than by equality, since NaN never equals itself.
    """
    values = np.asarray(local)
    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    if mask_val is not None:
        keep = (
            ~np.isnan(values)
            if isinstance(mask_val, float) and np.isnan(mask_val)
            else values != mask_val
        )
        values = values[keep]

    # Reinterpret the bits and widen to int64 so the sum cannot overflow the
    # element type: FMS's TRANSFER, without a copy where NumPy allows it.
    width = values.dtype.itemsize
    if width not in (1, 2, 4, 8):
        raise TypeError(f"mpp_chksum: unsupported dtype {values.dtype}.")
    as_int = np.ascontiguousarray(values).view(f"i{width}")
    local_sum = np.int64(as_int.sum(dtype=np.int64))

    if comm is None or comm.size == 1:
        return int(local_sum)
    total = np.empty(1, dtype=np.int64)
    comm.Allreduce(np.array([local_sum], dtype=np.int64), total, op=MPI.SUM)
    return int(total[0])


def mpp_partition_offsets(comm: MPI.Comm, local_length: int) -> tuple[int, int, int]:
    """New ``(global_size, start, stop)`` after an op changed the local length.

    Length-changing operations -- ``coarsen``, ``diff``, anything that drops
    or adds elements along the partition dimension -- have to rebuild the
    partition metadata from each rank's new length. The obvious way is an
    ``allgather`` of that one integer, but the result is only ever used as a
    total and an exclusive prefix sum, both of which are fixed-size
    collectives: ``Allreduce`` and ``Exscan``. The allgather moves a pickled
    object per rank and grows with rank count; this does not.

    Requires the partition to be contiguous and ordered by rank along the
    dimension, which is what makes the prefix sum this rank's own offset --
    the same property FMS relies on throughout ``mpp_domains``.
    """
    length = np.array([int(local_length)], dtype=np.int64)
    total = np.empty_like(length)
    comm.Allreduce(length, total, op=MPI.SUM)
    prefix = np.zeros_like(length)
    comm.Exscan(length, prefix, op=MPI.SUM)
    if comm.rank == 0:
        prefix[0] = 0  # Exscan leaves rank 0's receive buffer undefined.
    start = int(prefix[0])
    return int(total[0]), start, start + int(length[0])
