"""FMS/mpp-style distributed-memory primitives.

Named and organized after GFDL's Flexible Modeling System (FMS)
``mpp``/``mpp_domains`` modules (NOAA-GFDL/FMS: ``mpp/mpp.F90``,
``mpp/mpp_domains.F90``): a :class:`Domain` is FMS's ``domain2D``/
``domain1D`` equivalent -- partition bounds, halo topology, and a
communicator, carrying no data and no labels of its own. It is built once
via :func:`mpp_define_domains` and passed alongside plain NumPy buffers to
:func:`mpp_sum`/:func:`mpp_max`/:func:`mpp_min` (global reductions, FMS's
``MPP_REDUCE_`` family) and :func:`mpp_update_domains` (halo exchange,
FMS's ``mpp_update_domains``).

This module is the actual communication kernel underneath climtools'
xarray-facing layer: ``xarray/planning.py``'s ``comm_reduce``
and ``xarray/arithmetic.py``'s ``halo_exchange`` both delegate their real
MPI traffic here. Every function below takes and returns plain
``numpy.ndarray`` buffers plus a :class:`Domain` (or a raw
``mpi4py.MPI.Comm``) -- never an xarray object -- so it is usable, and
independently testable, with no xarray dependency at all. xarray's role
above this module is exactly what FMS's ``diag_manager`` layer's is above
``mpp``: labels, coordinates, and access, not communication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..mpi.context import MPIContext

__all__ = [
    "Domain",
    "mpp_define_domains",
    "mpp_define_layout",
    "mpp_get_neighbors",
    "mpp_max",
    "mpp_min",
    "mpp_reduce",
    "mpp_sum",
    "mpp_update_domains",
]


@dataclass(frozen=True)
class Domain:
    """A rank's local partition of one or more distributed dimensions.

    The FMS equivalent of ``domain2D``/``domain1D``: describes *where*
    this rank's data sits in the global array (compute-domain bounds) and
    *how* ranks are laid out relative to one another (the communicator,
    and -- for a multi-dimensional partition -- the Cartesian process
    grid), but holds no field data and no coordinate/units/attrs labels.
    Those stay entirely in xarray, one layer up.

    Attributes
    ----------
    dims : tuple of str
        Partitioned dimension names, in a fixed order.
    global_sizes : mapping of str to int
        Global length of each dimension in ``dims``.
    starts, stops : mapping of str to int
        This rank's half-open ``[start, stop)`` compute-domain interval
        along each dimension in ``dims`` (FMS's ``isc:iec``/``jsc:jec``).
    comm : mpi4py.MPI.Comm
        Communicator whose ranks jointly own the full global array.
    cart : mapping, optional
        Cartesian process-grid descriptor (``grid_shape``, ``coords``,
        ``periods``) for a multi-dimensional partition; ``None`` for a
        single partitioned dimension. See
        :func:`~.xarray.cartesian.get_cartesian_topology`.
    """

    dims: tuple[str, ...]
    global_sizes: dict[str, int]
    starts: dict[str, int]
    stops: dict[str, int]
    comm: MPI.Comm
    cart: dict[str, Any] | None = field(default=None)

    def local_size(self, dim: str) -> int:
        """This rank's local extent along ``dim``."""
        return self.stops[dim] - self.starts[dim]

    def is_global_edge(self, dim: str, side: str) -> bool:
        """True if this rank's slab touches the global boundary of ``dim``.

        Parameters
        ----------
        dim : str
            Partitioned dimension to check.
        side : {"lower", "upper"}
            Which edge of ``dim``.
        """
        if side == "lower":
            return self.starts[dim] == 0
        if side == "upper":
            return self.stops[dim] == self.global_sizes[dim]
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}.")

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any], comm: MPI.Comm) -> Domain:
        """Build a :class:`Domain` from climtools' existing ``.meta`` dict.

        ``.meta`` (see ``xarray/meta.py``) remains the system-of-record
        threaded through the xarray-facing layer (every ``MPIXarray`` op
        reads/writes it via ``.attrs``); this is the bridge that lets the
        buffer-only communication kernel below consume it without the
        rest of the codebase having to be rewritten around
        :class:`Domain` all at once.
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

    def to_meta(self, *, chunk_info: Mapping[str, int]) -> dict[str, Any]:
        """Return the ``.meta`` dict form of this domain (see :func:`from_meta`)."""
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


def mpp_define_layout(extent0: int, extent1: int, ndivs: int) -> tuple[int, int]:
    """Choose a 2D process-grid shape for ``ndivs`` ranks over a domain.

    FMS's own ``mpp_define_layout2D`` algorithm
    (``mpp/include/mpp_domains_define.inc``), adapted verbatim rather
    than reimplemented: first guess the divisor along axis 0 that
    matches the domain's aspect ratio (``round(sqrt(ndivs * extent0 /
    extent1))``), then walk it down until it evenly divides ``ndivs``
    (guaranteed to terminate at 1). This closed-form guess-and-adjust is
    FMS's actual algorithm for the 2D case -- which is the only case
    climtools' own Cartesian partitioning ever uses (every
    ``mpi_open_dataset``/``mpi_create_dataset`` call in this codebase
    partitions exactly two dimensions, e.g. ``("lat", "lon")``) -- in
    place of a general N-dimensional prime-factorization heuristic that
    was solving a more general problem than the one ever actually posed
    to it.

    Parameters
    ----------
    extent0, extent1 : int
        Global length of each of the two dimensions being partitioned.
    ndivs : int
        Number of process-grid cells (MPI ranks) to lay out.
    Returns
    -------
    tuple of (int, int)
        Process-grid divisions along axis 0 and axis 1;
        ``divisions[0] * divisions[1] == ndivs``.
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
) -> Domain:
    """Partition one or more dimensions across ``mpi_context``'s ranks.

    FMS's ``mpp_define_domains``: called once to build the domain
    descriptor every subsequent :func:`mpp_sum`/:func:`mpp_update_domains`
    call is handed, rather than each op recomputing its own bounds. A
    single partitioned dimension is split with balanced contiguous
    ``[start, stop)`` slabs (:func:`~.xarray.chunks.get_balanced_bounds`);
    two or more are split as a Cartesian process grid
    (:func:`~.xarray.cartesian.get_cartesian_topology`), reusing the same
    layout/neighbor logic climtools' xarray-facing layer already relies
    on rather than duplicating it.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context; ``mpi_context.comm`` is the communicator every rank
        in the returned :class:`Domain` belongs to.
    global_sizes : mapping of str to int
        Global length of every dimension named in ``dims`` (and, for the
        Cartesian case, of any other dimension the topology needs -- see
        :func:`~.xarray.cartesian.get_cartesian_topology`).
    dims : str or sequence of str
        Dimension name(s) to partition.
    min_partition_size : int, mapping, or None, optional
        Per :func:`~.xarray.chunks.get_balanced_bounds`'s ``min_chunk``:
        guaranteed minimum local length per partitioned dimension for
        every rank that receives any data. Single-dimension partitions
        only; ignored for a Cartesian partition.
    Returns
    -------
    Domain
        This rank's partition of ``global_sizes`` along ``dims``.
    """
    from .chunks import get_balanced_bounds

    comm = mpi_context.comm
    dim_tuple = (dims,) if isinstance(dims, str) else tuple(dims)

    if len(dim_tuple) == 1:
        dim = dim_tuple[0]
        length = int(global_sizes[dim])
        min_chunk = (
            min_partition_size
            if not isinstance(min_partition_size, Mapping)
            else min_partition_size.get(dim)
        )
        start, stop = get_balanced_bounds(length, comm.rank, comm.size, min_chunk)
        return Domain(
            dims=dim_tuple,
            global_sizes={dim: length},
            starts={dim: start},
            stops={dim: stop},
            comm=comm,
        )

    from .cartesian import get_cartesian_topology

    sizes = {d: int(global_sizes[d]) for d in dim_tuple}
    topology = get_cartesian_topology(comm, dim_tuple, sizes)
    starts = {d: topology.bounds[d][0] for d in dim_tuple}
    stops = {d: topology.bounds[d][1] for d in dim_tuple}
    return Domain(
        dims=dim_tuple,
        global_sizes=sizes,
        starts=starts,
        stops=stops,
        comm=comm,
        cart={
            "grid_shape": topology.grid_shape,
            "coords": topology.coords,
            "periods": topology.periods,
        },
    )


def mpp_reduce(
    local: np.ndarray, op: MPI.Op, comm: MPI.Comm | None, domain: Domain | None = None
) -> np.ndarray:
    """Global elementwise reduction of ``local`` under an arbitrary MPI op.

    The generic kernel behind :func:`mpp_sum`/:func:`mpp_max`/
    :func:`mpp_min` (each fixes ``op`` to ``MPI.SUM``/``MAX``/``MIN``);
    call this one directly for any other reducible op (``MPI.PROD``,
    ``MPI.LAND``, ``MPI.LOR``, ...). A single direct buffer-based
    FMS's own ``MPP_REDUCE_`` (``mpp/include/mpp_reduce_mpi.fh``) uses
    for ``mpp_max``/``mpp_min``, and the pattern ``mpp_sum`` follows too.
    No verification/agreement handshake runs here (unlike the xarray
    layer's own ``comm_reduce``, which additionally checks every rank
    posted the same operation before committing to a collective) --
    FMS trusts compiled SPMD Fortran by construction, and this is the
    equivalent trust boundary: the caller is responsible for every rank
    reaching this call with a compatible ``local`` shape/dtype and the
    same ``op``, exactly as an FMS caller is responsible for every PE
    reaching the matching ``call mpp_sum(...)``.

    Deliberately always calls ``Allreduce``, even over a size-1
    communicator, rather than short-circuiting: ``comm_reduce`` (this
    kernel's xarray-facing caller) can resolve *different* ranks onto
    *different* communicators of different sizes for a replicated-axis
    reduction group (see ``replica_count`` there), and Allreduce over a
    size-1 communicator is already a well-defined, cheap no-op in every
    MPI implementation -- a same-communicator early return here would be
    consistent, but was confirmed by direct testing to interact badly
    with that replicated-group path (the reduction result's shape only
    matched what the caller expected on the branch that actually called
    Allreduce). Not worth the risk for a case MPI already handles for
    free.

    Parameters
    ----------
    local : numpy.ndarray
        This rank's local buffer.
    op : MPI.Op
        Reduction operation.
    comm : mpi4py.MPI.Comm or None
        Communicator to reduce over. Takes priority over ``domain`` when
        both are given.
    domain : Domain or None, optional
        Used for its ``.comm`` when ``comm`` is not given directly.
    Returns
    -------
    numpy.ndarray
        The globally reduced buffer, same shape and dtype as ``local``.
    """
    active_comm = comm if comm is not None else (domain.comm if domain else MPI.COMM_WORLD)
    recv = np.empty_like(local)
    active_comm.Allreduce(local, recv, op=op)
    return recv


def mpp_sum(
    local: np.ndarray,
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray:
    """Global sum of ``local`` across every rank in ``domain``/``comm``.

    FMS's ``mpp_sum``. Pass either ``domain`` (its ``.comm`` is used) or
    ``comm`` directly; at least one is required.
    """
    return mpp_reduce(local, MPI.SUM, comm, domain)


def mpp_max(
    local: np.ndarray,
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray:
    """Global elementwise max of ``local`` across every rank in ``domain``/``comm``.

    FMS's ``mpp_max``.
    """
    return mpp_reduce(local, MPI.MAX, comm, domain)


def mpp_min(
    local: np.ndarray,
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray:
    """Global elementwise min of ``local`` across every rank in ``domain``/``comm``.

    FMS's ``mpp_min``.
    """
    return mpp_reduce(local, MPI.MIN, comm, domain)


def mpp_get_neighbors(
    domain: Domain, dim: str, *, periodic: bool = False
) -> tuple[int | None, int | None]:
    """Return the (lower, upper) neighbor rank along ``dim``.

    Single lookup point for "who is this rank's neighbor along a
    partitioned dimension" -- FMS builds this once into ``domain2D`` at
    ``mpp_define_domains`` time (its ``pe``/layout tables); here it is
    computed on demand from :class:`Domain`, since climtools' domains are
    cheap, frozen dataclasses rather than a persistent Fortran module
    state. Handles both a single partitioned dimension (linear
    ``rank -+ 1`` in ``domain.comm``) and a multi-dimensional Cartesian
    partition (a face neighbor along ``dim`` in the process grid, which
    does *not* coincide with ``rank -+ 1`` in general).

    Parameters
    ----------
    domain : Domain
        This rank's partition.
    dim : str
        Partitioned dimension to look up neighbors along.
    periodic : bool, optional
        Wrap at the global edge instead of returning ``None`` there.
    Returns
    -------
    tuple of (int or None, int or None)
        Lower and upper neighbor rank; ``None`` on a side with no
        neighbor (a true global edge, non-periodic).
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
    """Fill ``before``/``after`` halo cells of ``fields`` from neighbor ranks.

    FMS's ``mpp_update_domains``: exchanges boundary slabs with the ranks
    adjacent along ``dim`` and returns each field padded with them, using
    nonblocking point-to-point buffer sends (``Isend``/``Irecv`` + one
    shared ``Waitall``) -- one pair of messages per neighbor per field,
    all posted together and waited on once, not a collective and not one
    round trip per field. This is FMS's *group* update: passing a
    ``Mapping`` of several same-shaped-along-``axis`` fields batches all
    of their messages into that single ``Waitall``, exactly as updating
    several fields together is cheaper than one ``mpp_update_domains``
    call per field (each of which would otherwise pay its own
    synchronization latency). Passing a single ``numpy.ndarray`` instead
    is the plain single-field case; the return shape mirrors whichever
    was passed in.

    At a non-periodic global edge, the corresponding side is left
    unpadded (``left_pad``/``right_pad`` report 0 for that side) rather
    than raising -- callers that need to know they are at an edge
    trim/branch on those return values.

    Parameters
    ----------
    fields : numpy.ndarray or mapping of str to numpy.ndarray
        This rank's local buffer(s), contiguous or not. Every array must
        share the same length along ``axis`` (this rank's own local
        extent) but may otherwise differ in shape and dtype.
    domain : Domain
        This rank's partition; used only for edge detection when
        ``left_rank``/``right_rank`` are not given explicitly.
    dim : str
        Name of the partitioned dimension being exchanged (for edge
        lookup on ``domain``; purely a label here, not used for any
        indexing decision beyond that).
    axis : int
        Position of ``dim`` in each field's own axes.
    before, after : int
        Halo width requested from the lower/upper neighbor.
    periodic : bool, optional
        Wrap at the global edge (rank 0's lower neighbor is the last
        rank, and symmetrically for the last rank's upper neighbor)
        instead of leaving that side unpadded.
    left_rank, right_rank : int or None, optional
        Explicit lower/upper neighbor rank. When omitted, derived from
        ``domain.comm``'s linear rank order (``rank - 1``/``rank + 1``,
        wrapped under ``periodic``) -- correct for a single partitioned
        dimension; a caller managing a Cartesian partition must pass
        these explicitly (its neighbors are Cartesian-topology face
        neighbors along ``dim``, not linear rank +-1 in the flattened
        communicator; see ``xarray/cartesian.py``'s
        ``CartesianTopology.neighbors``).
    Returns
    -------
    tuple of (result, int, int)
        ``(padded, left_pad, right_pad)``: each field padded with up to
        ``before`` elements prepended and up to ``after`` appended along
        ``axis`` (``padded`` is a single array if ``fields`` was one, a
        dict of arrays if ``fields`` was a mapping), and how many
        elements were actually added on each side (equal to
        ``before``/``after`` except at an unpadded global edge, where it
        is 0).
    """
    single = isinstance(fields, np.ndarray)
    items: dict[str, np.ndarray] = {"": fields} if single else dict(fields)

    comm = domain.comm
    if left_rank is None or right_rank is None:
        default_left, default_right = mpp_get_neighbors(domain, dim, periodic=periodic)
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

    # Post every message for every field up front, then wait once: the
    # whole point of a group update, and why this isn't just a loop
    # calling a single-field version once per field.
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
