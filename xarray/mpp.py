"""Provide FMS-style MPI domain, halo, reduction, and checksum primitives."""

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
    """Describe one rank's distributed compute domain.

    Attributes
    ----------
    dims : tuple[str, ...]
        Partitioned dimension names.
    global_sizes : dict[str, int]
        Global size of each partitioned dimension.
    starts, stops : dict[str, int]
        Rank-local half-open ownership bounds.
    comm : mpi4py.MPI.Comm
        Communicator owning the global array.
    cart : dict or None
        Cartesian topology descriptor for multi-dimensional partitions.
    """

    dims: tuple[str, ...]
    global_sizes: dict[str, int]
    starts: dict[str, int]
    stops: dict[str, int]
    comm: MPI.Comm
    cart: dict[str, Any] | None = field(default=None)

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any], comm: MPI.Comm) -> Domain:
        """Build a domain from climtools MPI metadata.

        Parameters
        ----------
        meta : mapping
            Canonical MPI metadata.
        comm : mpi4py.MPI.Comm
            Owning communicator.

        Returns
        -------
        Domain
            Rank-local domain descriptor.
        """
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
    local: np.ndarray[Any, Any],
    op: MPI.Op,
    comm: MPI.Comm,
    recvcounts: Sequence[int],
    *,
    axis: int = 0,
) -> np.ndarray[Any, Any]:
    """Reduce an array and retain each rank's contiguous slice.

    Parameters
    ----------
    local : numpy.ndarray
        Equal-shaped local reduction buffer on every rank.
    op : mpi4py.MPI.Op
        Reduction operator.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    recvcounts : sequence of int
        Elements retained by each rank along ``axis``.
    axis : int, default 0
        Axis split among ranks.

    Returns
    -------
    numpy.ndarray
        This rank's reduced slice.
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
    """Choose a two-dimensional process-grid layout.

    Parameters
    ----------
    extent0, extent1 : int
        Global grid extents.
    ndivs : int
        Number of MPI ranks.

    Returns
    -------
    tuple[int, int]
        Process-grid shape minimizing idle ranks, then halo perimeter.

    Raises
    ------
    ValueError
        If ``ndivs`` is not positive.
    """
    if ndivs < 1:
        raise ValueError(f"ndivs must be positive, got {ndivs}.")

    pairs = [(rows, ndivs // rows) for rows in range(1, ndivs + 1) if ndivs % rows == 0]

    def cost(layout: tuple[int, int]) -> tuple[int, float, int]:
        """Return idle-rank, halo-perimeter, and aspect-ratio costs."""
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
    """Define balanced rank-local compute domains.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    global_sizes : mapping[str, int]
        Global sizes of partitioned dimensions.
    dims : str or sequence of str
        Partition dimensions.
    min_partition_size : int or mapping, optional
        Minimum non-empty local extent.
    rank : int, optional
        Rank whose domain to compute; defaults to the caller.

    Returns
    -------
    Domain
        Rank-local domain descriptor.
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
    """Return balanced ownership bounds for every division of a dimension.

    Parameters
    ----------
    global_size : int
        Global dimension length.
    dim_size : int
        Number of divisions along the dimension.
    min_partition_size : int, optional
        Minimum non-empty local extent.

    Returns
    -------
    list[tuple[int, int]]
        Half-open bounds for each division.
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
    """Intersect one compute domain with a global slice.

    Parameters
    ----------
    start, stop : int
        Rank-local global ownership bounds.
    requested_start, requested_stop : int
        Requested global half-open slice.

    Returns
    -------
    tuple[int, int, int]
        Local slice bounds and the surviving global start offset.
    """
    lower = max(requested_start, start)
    upper = max(lower, min(requested_stop, stop))
    below = max(0, min(requested_stop, start) - requested_start)
    return lower - start, upper - start, below


def _mpp_reduce(
    local: np.ndarray[Any, Any], op: MPI.Op, comm: MPI.Comm | None, domain: Domain | None = None
) -> np.ndarray[Any, Any]:
    """Reduce rank-local arrays with an MPI reduction operator."""
    active_comm = (
        comm if comm is not None else (domain.comm if domain else MPI.COMM_WORLD)
    )
    recv = np.empty_like(local)
    active_comm.Allreduce(local, recv, op=op)
    return recv


def mpp_sum(
    local: np.ndarray[Any, Any], domain: Domain | None = None, *, comm: MPI.Comm | None = None
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_sum``. Pass ``domain`` or ``comm``."""
    return _mpp_reduce(local, MPI.SUM, comm, domain)


def mpp_max(
    local: np.ndarray[Any, Any], domain: Domain | None = None, *, comm: MPI.Comm | None = None
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_max``."""
    return _mpp_reduce(local, MPI.MAX, comm, domain)


def mpp_min(
    local: np.ndarray[Any, Any], domain: Domain | None = None, *, comm: MPI.Comm | None = None
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_min``."""
    return _mpp_reduce(local, MPI.MIN, comm, domain)


def mpp_get_neighbor_pe(
    domain: Domain, dim: str, *, periodic: bool = False
) -> tuple[int | None, int | None]:
    """Return neighboring ranks along one partition dimension.

    Parameters
    ----------
    domain : Domain
        Rank-local domain descriptor.
    dim : str
        Partition dimension.
    periodic : bool, default False
        Wrap neighbors across global edges.

    Returns
    -------
    tuple[int or None, int or None]
        Lower- and upper-side neighbor ranks.
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
    """Store state for an in-flight halo exchange.

    Attributes
    ----------
    items : dict[str, numpy.ndarray]
        Fields being exchanged.
    groups : dict
        Fields grouped by wire dtype.
    recv_bufs : dict
        Receive buffers keyed by dtype and side.
    recv_reqs, send_reqs : list
        Outstanding MPI requests.
    axis : int
        Exchanged array axis.
    before, after : int
        Requested halo widths.
    single : bool
        Whether the input was a single array.
    unpack : Any
        Callable restoring wire representations.
    """

    items: dict[str, np.ndarray[Any, Any]]
    groups: dict[Any, list[str]]
    recv_bufs: dict[tuple[Any, str], np.ndarray[Any, Any]]
    recv_reqs: list[Any]
    send_reqs: list[Any]
    axis: int
    before: int
    after: int
    single: bool
    unpack: Any


def mpp_start_update_domains(
    fields: np.ndarray[Any, Any] | Mapping[str, np.ndarray[Any, Any]],
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
    """Start a nonblocking halo exchange.

    Parameters
    ----------
    fields : numpy.ndarray or mapping[str, numpy.ndarray]
        Field or fields sharing the exchanged axis.
    domain : Domain
        Rank-local domain descriptor.
    dim : str
        Partition dimension.
    axis : int
        Array axis corresponding to ``dim``.
    before, after : int
        Lower and upper halo widths.
    periodic : bool, default False
        Wrap across global edges.
    left_rank, right_rank : int or None, optional
        Explicit neighboring ranks.

    Returns
    -------
    DomainUpdate
        In-flight exchange state.
    """
    single = isinstance(fields, np.ndarray)
    items: dict[str, np.ndarray[Any, Any]] = {"": fields} if single else dict(fields)

    comm = domain.comm
    if left_rank is None or right_rank is None:
        default_left, default_right = mpp_get_neighbor_pe(
            domain, dim, periodic=periodic
        )
        left_rank = default_left if left_rank is None else left_rank
        right_rank = default_right if right_rank is None else right_rank

    def _view(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """View as a dtype the raw MPI buffer protocol accepts."""
        return arr.view(np.int64) if arr.dtype.kind in "mM" else arr

    def _slab(arr: np.ndarray[Any, Any], start: int, stop: int) -> np.ndarray[Any, Any]:
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return np.ascontiguousarray(arr[tuple(idx)])

    def _halo_shape(name: str, width: int) -> tuple[int, ...]:
        arr = items[name]
        return (*arr.shape[:axis], width, *arr.shape[axis + 1 :])

    # Pack fields by wire dtype in deterministic order; no layout metadata is
    # transmitted.
    groups: dict[np.dtype[Any], list[str]] = {}
    for name, arr in items.items():
        groups.setdefault(_view(arr).dtype, []).append(name)
    for names in groups.values():
        names.sort()

    def _pack(names: list[str], side: str) -> np.ndarray[Any, Any]:
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
        flat: np.ndarray[Any, Any], names: list[str], width: int
    ) -> dict[str, np.ndarray[Any, Any]]:
        out: dict[str, np.ndarray[Any, Any]] = {}
        pos = 0
        for name in names:
            shape = _halo_shape(name, width)
            count = int(np.prod(shape)) if shape else 1
            # Restore each field's original dtype after unpacking the wire
            # representation.
            out[name] = flat[pos : pos + count].reshape(shape).view(items[name].dtype)
            pos += count
        return out

    can_send_right = right_rank is not None and before > 0
    can_send_left = left_rank is not None and after > 0
    can_recv_before = left_rank is not None and before > 0
    can_recv_after = right_rank is not None and after > 0

    recv_bufs: dict[tuple[np.dtype[Any], str], np.ndarray[Any, Any]] = {}
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
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]], int, int]:
    """Complete a halo exchange and return received slabs.

    Parameters
    ----------
    update : DomainUpdate
        In-flight exchange state.

    Returns
    -------
    tuple[dict, dict, int, int]
        Lower halos, upper halos, and realized lower/upper pad widths.
    """
    MPI.Request.Waitall(update.recv_reqs)
    MPI.Request.Waitall(update.send_reqs)

    recv_before: dict[str, np.ndarray[Any, Any]] = {}
    recv_after: dict[str, np.ndarray[Any, Any]] = {}
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
    fields: np.ndarray[Any, Any] | Mapping[str, np.ndarray[Any, Any]],
    domain: Domain,
    dim: str,
    axis: int,
    *,
    before: int,
    after: int,
    periodic: bool = False,
    left_rank: int | None = None,
    right_rank: int | None = None,
) -> tuple[np.ndarray[Any, Any] | dict[str, np.ndarray[Any, Any]], int, int]:
    """Exchange halos and return padded local fields.

    Parameters
    ----------
    fields : numpy.ndarray or mapping[str, numpy.ndarray]
        Field or fields to exchange.
    domain : Domain
        Rank-local domain descriptor.
    dim : str
        Partition dimension.
    axis : int
        Array axis corresponding to ``dim``.
    before, after : int
        Lower and upper halo widths.
    periodic : bool, default False
        Wrap across global edges.
    left_rank, right_rank : int or None, optional
        Explicit neighboring ranks.

    Returns
    -------
    tuple[numpy.ndarray or dict, int, int]
        Padded field(s) and realized lower/upper pad widths.
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

# Renormalize mantissa products before they can become subnormal.
_PROD_BLOCK = 512


def _to_digits(array: np.ndarray[Any, Any], axis: int) -> np.ndarray[Any, Any]:
    """Split values into signed integer digits along a new leading axis."""
    values = np.asarray(array, dtype=np.float64)
    sign = np.where(values < 0.0, -1.0, 1.0)
    residual = np.abs(values)
    digits = np.empty((_NUMINT, *values.shape), dtype=np.int64)
    for n, scale in enumerate(_SCALES):
        digit = np.floor(residual / scale)
        digits[n] = (sign * digit).astype(np.int64)
        residual -= digit * scale
    return digits.sum(axis=axis + 1 if axis >= 0 else axis, dtype=np.int64)


def _from_digits(digits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Inverse of :func:`_to_digits`, summing smallest scale first."""
    total = np.zeros(digits.shape[1:], dtype=np.float64)
    for n in range(_NUMINT - 1, -1, -1):
        total += digits[n].astype(np.float64) * _SCALES[n]
    return total


def mpp_reproducing_sum(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm,
    *,
    axis: int | None = None,
) -> np.ndarray[Any, Any]:
    """Compute a rank-count-invariant distributed sum.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    axis : int or None, optional
        Local reduction axis before the global sum.

    Returns
    -------
    numpy.ndarray
        Reproducible global sum.

    Raises
    ------
    ValueError
        If the rank count is unsupported or input contains non-finite values.
    """
    if comm.size > MAX_EFP_RANKS:
        raise ValueError(f"mpp_reproducing_sum supports at most {MAX_EFP_RANKS} ranks.")
    if not np.all(np.isfinite(local)):
        raise ValueError("mpp_reproducing_sum requires finite input.")
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


def _moved_to_front(values: np.ndarray[Any, Any], axes: Sequence[int]) -> np.ndarray[Any, Any]:
    """Collapse ``axes`` into a single leading axis, preserving the rest."""
    ordered = tuple(a % values.ndim for a in axes)
    moved = np.moveaxis(values, ordered, range(len(ordered)))
    kept = moved.shape[len(ordered) :]
    return moved.reshape((-1, *kept))


def mpp_prod_decompose(
    local: np.ndarray[Any, Any], axes: int | Sequence[int] = 0
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Decompose a product into a stable mantissa and integer companions.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    axes : int or sequence of int, default 0
        Local reduction axes.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Mantissa and packed exponent/exception tallies.
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
    mantissa: np.ndarray[Any, Any],
    companions: np.ndarray[Any, Any],
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Reconstruct a product from reduced decomposition fields.

    Parameters
    ----------
    mantissa : numpy.ndarray
        Reduced mantissa.
    companions : numpy.ndarray
        Reduced exponent and exception tallies.
    dtype : numpy.dtype, optional
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Reconstructed product with signed zero/inf and NaN handling.
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
    local: np.ndarray[Any, Any],
    comm: MPI.Comm,
    *,
    axis: int | Sequence[int] = 0,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Compute a rank-count-invariant distributed product.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    axis : int or sequence of int, default 0
        Local reduction axes.
    dtype : numpy.dtype, optional
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Reproducible global product.

    Raises
    ------
    ValueError
        If the communicator exceeds the supported rank limit.
    """
    if comm.size > MAX_PROD_RANKS:
        raise ValueError(f"mpp_reproducing_prod supports at most {MAX_PROD_RANKS} ranks.")
    values = np.asarray(local)
    out_dtype = np.dtype(dtype) if dtype is not None else values.dtype
    mantissa, companions = mpp_prod_decompose(values, axis)

    global_mantissa = np.empty_like(mantissa)
    comm.Allreduce(np.ascontiguousarray(mantissa), global_mantissa, op=MPI.PROD)
    global_companions = np.empty_like(companions)
    comm.Allreduce(np.ascontiguousarray(companions), global_companions, op=MPI.SUM)
    return mpp_prod_recombine(global_mantissa, global_companions, out_dtype)


def mpp_chksum(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm | None = None,
    *,
    mask_val: float | None = None,
) -> int:
    """Compute a rank-order-independent bitwise checksum.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local field.
    comm : mpi4py.MPI.Comm, optional
        Reduction communicator.
    mask_val : float, optional
        Sentinel excluded from the checksum.

    Returns
    -------
    int
        Global checksum.

    Raises
    ------
    TypeError
        If the element width is unsupported.
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
    """Recompute distributed offsets after a local length change.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Partition communicator.
    local_length : int
        This rank's new local length.

    Returns
    -------
    tuple[int, int, int]
        Global size and this rank's half-open ownership bounds.
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
