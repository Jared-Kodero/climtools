"""Domain decomposition, halo updates and redistribution.

Mirrors FMS ``mpp/mpp_domains.F90``: the compute domain each rank owns, the
process grid it sits in, and every routine that moves data between
decompositions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

from collections.abc import Mapping

from ..mpi.diagnostics import MPIError
from ..mpi.mpi_init import MPI
from ..xarray.chunks import get_balanced_bounds, prune_chunk_info
from ..xarray.meta import mpp_operand_meta, mpp_update_meta, strip_mpi_meta
from .mpp import mpp_max, mpp_min, mpp_sum

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from mpi4py.MPI import Cartcomm, Comm

    from ..mpi.context import MPIContext

_TOPOLOGY_KEYVAL = MPI.Comm.Create_keyval()


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
    halo : dict[str, tuple[int, int]]
        Halo widths held before and after the compute domain on each
        dimension. Empty means the data domain equals the compute domain.

    Notes
    -----
    FMS distinguishes the *compute* domain, which a rank owns and is
    responsible for updating, from the *data* domain, which additionally
    covers the halo points it holds copies of. ``starts``/``stops`` describe
    the compute domain; :func:`mpp_get_data_domain` applies ``halo``.
    """

    dims: tuple[str, ...]
    global_sizes: dict[str, int]
    starts: dict[str, int]
    stops: dict[str, int]
    comm: MPI.Comm
    cart: dict[str, Any] | None = field(default=None)
    halo: dict[str, tuple[int, int]] = field(default_factory=dict)

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


def mpp_get_compute_domain(domain: Domain, dim: str) -> tuple[int, int]:
    """Return the half-open bounds this rank owns along ``dim``.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple[int, int]
        ``(start, stop)`` of the compute domain.
    """
    return domain.starts[dim], domain.stops[dim]


def mpp_get_data_domain(domain: Domain, dim: str) -> tuple[int, int]:
    """Return the bounds this rank holds along ``dim``, halo included.

    The data domain is the compute domain widened by the halo and clipped to
    the global domain, since an edge rank has no neighbour to receive from.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple[int, int]
        ``(start, stop)`` of the data domain.
    """
    before, after = domain.halo.get(dim, (0, 0))
    start, stop = domain.starts[dim], domain.stops[dim]
    return max(0, start - before), min(domain.global_sizes[dim], stop + after)


def mpp_get_global_domain(domain: Domain, dim: str) -> tuple[int, int]:
    """Return the global bounds of ``dim``.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple[int, int]
        ``(0, global_size)``.
    """
    return 0, domain.global_sizes[dim]


def mpp_get_layout(domain: Domain) -> tuple[int, ...]:
    """Return the process-grid shape the domain is divided over.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.

    Returns
    -------
    tuple of int
        Divisions along each partitioned dimension.
    """
    if domain.cart is not None:
        return tuple(int(n) for n in domain.cart["shape"])
    return (domain.comm.size,)


def mpp_get_pelist(domain: Domain) -> tuple[int, ...]:
    """Return the ranks the domain is distributed over.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.

    Returns
    -------
    tuple of int
        Ranks in the owning communicator.
    """
    return tuple(range(domain.comm.size))


def mpp_get_domain_extents(domain: Domain, dim: str) -> tuple[tuple[int, int], ...]:
    """Return every rank's compute-domain bounds along ``dim``.

    Parameters
    ----------
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.

    Returns
    -------
    tuple of tuple[int, int]
        ``(start, stop)`` per rank, in rank order.
    """
    return tuple(domain.comm.allgather((domain.starts[dim], domain.stops[dim])))


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

    comm = mpi_context.comm
    target_rank = comm.rank if rank is None else rank
    dim_tuple = (dims,) if isinstance(dims, str) else tuple(dims)

    def _min_chunk(d: str) -> int | None:
        """Return the minimum partition size requested for one dimension."""
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
        topology = mpp_get_cartesian_domain(comm, dim_tuple, sizes)
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


def _compute_slice(
    field: np.ndarray[Any, Any], domain: Domain, dims: Sequence[str]
) -> np.ndarray[Any, Any]:
    """Trim halo points off ``field`` so only compute-domain values remain."""
    if not domain.halo:
        return field
    index: list[slice] = [slice(None)] * field.ndim
    for axis, dim in enumerate(dims):
        before, after = domain.halo.get(dim, (0, 0))
        if before or after:
            index[axis] = slice(before, field.shape[axis] - after or None)
    return field[tuple(index)]


def mpp_global_sum(
    field: np.ndarray[Any, Any],
    domain: Domain,
    dims: Sequence[str],
    *,
    bitwise_exact: bool = False,
) -> Any:
    """Sum a distributed field over its whole global domain.

    Halo points are excluded, so a value shared by two ranks is counted once.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's data-domain values.
    domain : Domain
        Rank-local domain describing ``field``.
    dims : sequence of str
        Dimension name of each axis of ``field``.
    bitwise_exact : bool, default False
        Sum in extended fixed point, giving a result independent of the rank
        count. Mirrors the FMS ``BITWISE_EXACT_SUM`` flag.

    Returns
    -------
    Any
        Global sum.
    """
    owned = _compute_slice(np.asarray(field), domain, dims)
    if bitwise_exact:
        from .mpp_efp import mpp_reproducing_sum

        return mpp_reproducing_sum(owned.reshape(-1), domain.comm)
    return mpp_sum(np.asarray(owned.sum(), dtype=np.float64), comm=domain.comm)


def mpp_global_max(
    field: np.ndarray[Any, Any], domain: Domain, dims: Sequence[str]
) -> Any:
    """Return the maximum of a distributed field over its global domain.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's data-domain values.
    domain : Domain
        Rank-local domain describing ``field``.
    dims : sequence of str
        Dimension name of each axis of ``field``.

    Returns
    -------
    Any
        Global maximum, over compute-domain points only.
    """
    owned = _compute_slice(np.asarray(field), domain, dims)
    return mpp_max(np.asarray(owned.max()), comm=domain.comm)


def mpp_global_min(
    field: np.ndarray[Any, Any], domain: Domain, dims: Sequence[str]
) -> Any:
    """Return the minimum of a distributed field over its global domain.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's data-domain values.
    domain : Domain
        Rank-local domain describing ``field``.
    dims : sequence of str
        Dimension name of each axis of ``field``.

    Returns
    -------
    Any
        Global minimum, over compute-domain points only.
    """
    owned = _compute_slice(np.asarray(field), domain, dims)
    return mpp_min(np.asarray(owned.min()), comm=domain.comm)


def mpp_check_field(
    field: np.ndarray[Any, Any], domain: Domain, *, label: str = "field"
) -> None:
    """Verify every rank holds the same values where their domains overlap.

    FMS ``mpp_check_field`` is a debugging aid: it catches a halo update that
    silently failed to propagate, which otherwise shows up much later as a
    wrong answer.

    Parameters
    ----------
    field : numpy.ndarray
        Values to compare.
    domain : Domain
        Rank-local domain describing ``field``.
    label : str, default "field"
        Name used in the error message.

    Raises
    ------
    DomainMismatchError
        If any rank disagrees.
    """
    from .mpp import mpp_chksum

    local = mpp_chksum(np.asarray(field))
    every = domain.comm.allgather(local)
    if len(set(every)) != 1:
        raise DomainMismatchError(
            f"{label} differs across ranks: {len(set(every))} distinct checksums."
        )


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
        topology = mpp_get_cartesian_domain(comm, domain.dims, domain.global_sizes)
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
        """Return a contiguous copy of ``arr[start:stop]`` along the halo axis."""
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return np.ascontiguousarray(arr[tuple(idx)])

    def _halo_shape(name: str, width: int) -> tuple[int, ...]:
        """Return the shape of a ``width``-wide halo slab of one field."""
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
        """Flatten one edge of every named field into a single send buffer."""
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
        """Split a received buffer back into per-field halo slabs."""
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


def _define_layout_nd(extents: Sequence[int], ndivs: int) -> tuple[int, ...]:
    """Choose a process-grid shape for more than two partition dimensions.

    Assigns the prime factors of ``ndivs``, largest first, to whichever axis
    currently carries the most work per rank. Two-dimensional layouts use
    :func:`mpp_define_layout` instead, which follows FMS exactly.

    Parameters
    ----------
    extents : sequence of int
        Global length of each partitioned dimension.
    ndivs : int
        Number of ranks to divide among.

    Returns
    -------
    tuple of int
        Number of divisions along each axis.

    Raises
    ------
    ValueError
        If ``extents`` is empty, any extent is not positive, or ``ndivs`` is
        not positive.
    """
    if not extents:
        raise ValueError("requires at least one extent")
    if any(extent <= 0 for extent in extents):
        raise ValueError(f"All extents must be positive; got {tuple(extents)!r}.")
    if ndivs <= 0:
        raise ValueError(f"ndivs must be positive; got {ndivs}.")
    if len(extents) == 2:
        return mpp_define_layout(extents[0], extents[1], ndivs)

    factors: list[int] = []
    remaining, factor = ndivs, 2
    while factor * factor <= remaining:
        while remaining % factor == 0:
            factors.append(factor)
            remaining //= factor
        factor += 1
    if remaining > 1:
        factors.append(remaining)

    shape = [1] * len(extents)
    for f in sorted(factors, reverse=True):
        axis = max(range(len(extents)), key=lambda i: extents[i] / shape[i])
        shape[axis] *= f
    return tuple(shape)


@dataclass(frozen=True)
class CartesianDomain:
    """One rank's view of a multi-dimensional Cartesian process grid.

    Attributes
    ----------
    dims : tuple of str
        Partition dimension names, in Cartesian-axis order.
    grid_shape : tuple of int
        Number of process-grid divisions along each axis.
    coords : tuple of int
        This rank's position in the process grid, one entry per axis.
    cart_comm : mpi4py.MPI.Cartcomm
        The underlying Cartesian communicator. Rank order matches
        ``comm`` (``reorder=False``), so ``cart_comm.rank`` and the
        originating communicator's rank agree.
    bounds : dict of str to (int, int)
        Global half-open ``[start, stop)`` interval owned by this rank,
        per dimension.
    neighbors : dict of str to (int or None, int or None)
        Per-dimension ``(lower_rank, upper_rank)`` face neighbors in the
        *original* (non-Cartesian) communicator's rank numbering. None at
        a non-periodic global boundary.

    """

    dims: tuple[str, ...]
    grid_shape: tuple[int, ...]
    coords: tuple[int, ...]
    cart_comm: Cartcomm
    bounds: dict[str, tuple[int, int]]
    neighbors: dict[str, tuple[int | None, int | None]]
    _sub_comm_cache: dict[frozenset[str], Comm] = field(
        default_factory=dict, repr=False, compare=False
    )

    def as_meta_cart(self) -> dict[str, Any]:
        """Return the ``meta["cart"]`` descriptor for this topology.

        Returns
        -------
        dict[str, Any]
            Cartesian topology metadata descriptor.

        """
        return {
            "grid_shape": self.grid_shape,
            "coords": self.coords,
            "periods": (False,) * len(self.dims),
        }

    def sub_comm(self, merge_axes: Sequence[str]) -> Comm:
        """Return the communicator grouping ranks for a partial collective.

        Parameters
        ----------
        merge_axes : sequence of str
            Subset of :attr:`dims` to group ranks across.

        Returns
        -------
        mpi4py.MPI.Comm
            The (possibly cached) sub-communicator.

        """
        key = frozenset(merge_axes)
        cached = self._sub_comm_cache.get(key)
        if cached is not None:
            return cached
        remain = [dim in key for dim in self.dims]
        sub = self.cart_comm.Sub(remain)
        self._sub_comm_cache[key] = sub
        return sub


def _no_proc_null(rank: int) -> int | None:
    """Map ``MPI.PROC_NULL`` (no neighbor) to None."""
    return None if rank == MPI.PROC_NULL else int(rank)


def mpp_define_cartesian_domain(
    comm: MPI.Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianDomain:
    """Build a rank's Cartesian topology for a multi-dimensional partition.

    Raises
    ------
    ValueError
        If fewer than two dimensions are given.

    """
    if len(dims) < 2:
        raise ValueError(
            "requires at least two partition dimensions; got " + f"{tuple(dims)!r}"
        )

    extents = [int(sizes[dim]) for dim in dims]
    grid_shape = _define_layout_nd(extents, comm.size)

    cart_comm = comm.Create_cart(
        dims=list(grid_shape),
        periods=[False] * len(dims),
        reorder=False,
    )
    coords = tuple(cart_comm.Get_coords(cart_comm.rank))

    bounds: dict[str, tuple[int, int]] = {}
    neighbors: dict[str, tuple[int | None, int | None]] = {}
    for axis, dim in enumerate(dims):
        bounds[dim] = get_balanced_bounds(extents[axis], coords[axis], grid_shape[axis])
        lower, upper = cart_comm.Shift(axis, 1)
        neighbors[dim] = (_no_proc_null(lower), _no_proc_null(upper))

    return CartesianDomain(
        dims=tuple(dims),
        grid_shape=grid_shape,
        coords=coords,
        cart_comm=cart_comm,
        bounds=bounds,
        neighbors=neighbors,
    )


def mpp_get_cartesian_domain(
    comm: MPI.Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianDomain:
    """Return (building and caching once) a rank's Cartesian topology."""
    dims = tuple(dims)
    # Include sizes in the cache key so same-named dimensions with different extents
    # cannot collide.
    cache_key = (dims, tuple(int(sizes[d]) for d in dims))
    cache = comm.Get_attr(_TOPOLOGY_KEYVAL)
    if cache is None:
        cache = {}
        comm.Set_attr(_TOPOLOGY_KEYVAL, cache)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    topology = mpp_define_cartesian_domain(comm, dims, sizes)
    cache[cache_key] = topology
    return topology


class DomainMismatchError(MPIError):
    """Raised when rank-local domains do not tile the global domain."""


class HaloWidthError(ValueError):
    """A rank's local partition is shorter than the halo an op asked for.

    Its own type rather than a bare ``ValueError`` because callers have to
    tell this architectural refusal apart from a genuine failure -- the test
    suite reports it as a skip, not a failure. That classification used to
    match a substring of the message, so shortening the message silently
    turned every one of those skips into a failure. Subclasses ``ValueError``
    so existing ``except ValueError`` handlers are unaffected.
    """


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
        """Return the rank-local variable behind a data or coordinate name."""
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
        """Reconstruct an xarray object from the exchanged arrays, or None if
        unpadded."""
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
        ``(padded, left_pad, right_pad)``: the padded object (replicated metadata
        stripped, since it is no longer a clean partition) and the number of elements
        actually prepended/appended (equal to ``before``/``after`` except at a global
        edge, where it is 0).

    Raises
    ------
    ValueError
        If ``value`` is not distributed, ``dim`` is missing or disagrees with an active
        partition dimension, ``before``/``after`` are negative, or any rank's local
        partition along ``dim`` is shorter than ``before``/``after``.

    """
    meta = mpp_operand_meta(value)
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

    from ..xarray.planning import _agree

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


def mpp_dim_comm(mpi_context: MPIContext, meta: Mapping[str, Any], dim: str) -> Comm:
    """Return the communicator varying only along one partition dimension.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    meta : mapping
        Canonical MPI metadata.
    dim : str
        Partition dimension.

    Returns
    -------
    mpi4py.MPI.Comm
        Full communicator for 1-D partitions or the corresponding Cartesian
        subcommunicator.
    """
    dims = meta["dims"]
    if len(dims) <= 1 or "cart" not in meta:
        return cast("Comm", mpi_context.comm)
    topology = mpp_get_cartesian_domain(mpi_context.comm, dims, meta["global_sizes"])
    return topology.sub_comm((dim,))
