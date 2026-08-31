"""Cartesian process-grid topology for multi-dimensional MPI partitioning.

Generalizes the existing single-dimension partitioning (one rank owns one
contiguous, linearly-indexed slab, with neighbors at ``rank - 1``/
``rank + 1``) to two or more partition dimensions, using an
``mpi4py.MPI`` Cartesian communicator so that neighbor discovery and
multi-axis halo exchange reduce to ``Cartcomm.Shift`` instead of manual
rank arithmetic.

Design is deliberately narrow and mirrors NOAA-GFDL's FMS ``mpp_domains``
domain decomposition (see ``mpp_define_layout2D`` and
``mpp_compute_extent`` in ``FMS/mpp/include/mpp_domains_define.inc``)
without importing its full generality:

- **Process-grid shape** (:func:`compute_layout`) follows the same
  aspect-ratio idea as ``mpp_define_layout2D`` -- keep work-per-rank
  balanced across axes -- generalized from GFDL's closed-form 2D-only
  formula to a greedy prime-factor assignment that also works for more
  than two partition dimensions (GFDL's own domain decomposition never
  needs more than two, since it only ever splits horizontal, not
  vertical, extent).
- **Per-axis bounds** reuse :func:`~.chunks.get_balanced_bounds` exactly
  -- the same near-equal contiguous split already used by the existing
  one-dimensional path -- applied independently along each Cartesian
  axis, rather than adopting FMS's mirror-symmetric ``mpp_compute_extent``
  scheme. climtools's existing balanced split is simpler and already
  relied upon everywhere else in this codebase; reusing it per-axis
  maximizes code and test reuse instead of introducing a second,
  differently-balanced splitting rule.
- **Neighbor discovery** uses one ``Cartcomm.Shift`` per axis (face
  neighbors only). Corner/edge (diagonal) ghost values are not fetched by
  a dedicated diagonal exchange; :meth:`~.arithmetic.Arithmetic.halo_exchange`
  instead performs the per-axis exchanges in sequence, so a later axis's
  exchange naturally carries an earlier axis's already-received halo into
  the corner -- the same "update overlap" ordering trick FMS itself
  documents relying on for single-width halos, without needing FMS's
  explicit ``NORTH_EAST``/``SOUTH_WEST``/... diagonal ``pearray`` lookups
  (see ``mpp_get_neighbor_pe_2d``).
- Global domains are never periodic here (``periods=(False, ...)``);
  climtools has no cyclic/folded-domain concept to preserve, unlike FMS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from mpi4py import MPI

from .chunks import get_balanced_bounds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mpi4py.MPI import Cartcomm, Comm, Intracomm

__all__ = [
    "CartesianTopology",
    "build_cartesian_topology",
    "compute_layout",
    "get_cartesian_topology",
]


def compute_layout(extents: Sequence[int], nranks: int) -> tuple[int, ...]:
    """Choose a process-grid shape balancing work per rank across axes.

    Greedily assigns each prime factor of ``nranks`` (largest first) to
    whichever axis currently has the most global extent per already
    assigned division -- the same goal as GFDL's
    ``mpp_define_layout2D`` aspect-ratio factorization
    (``idiv = nint(sqrt(ndivs*isz/jsz))``, reduced until it divides
    ``ndivs``), generalized here to an arbitrary number of axes via
    greedy factor assignment instead of a closed-form 2D-only formula.

    Parameters
    ----------
    extents : sequence of int
        Global length of each partition dimension, in axis order.
    nranks : int
        Number of MPI ranks (process-grid cells) to lay out.

    Returns
    -------
    tuple of int
        Process-grid shape, one entry per axis, with
        ``math.prod(shape) == nranks``.

    Raises
    ------
    ValueError
        If ``extents`` is empty, any extent is not positive, or
        ``nranks`` is not positive.
    """
    if not extents:
        raise ValueError("compute_layout() requires at least one extent.")
    if any(extent <= 0 for extent in extents):
        raise ValueError(f"All extents must be positive; got {tuple(extents)!r}.")
    if nranks <= 0:
        raise ValueError(f"nranks must be positive; got {nranks}.")

    ndims = len(extents)
    shape = [1] * ndims

    factors: list[int] = []
    remaining = nranks
    factor = 2
    while factor * factor <= remaining:
        while remaining % factor == 0:
            factors.append(factor)
            remaining //= factor
        factor += 1
    if remaining > 1:
        factors.append(remaining)
    factors.sort(reverse=True)

    for f in factors:
        axis = max(range(ndims), key=lambda i: extents[i] / shape[i])
        shape[axis] *= f

    return tuple(shape)


@dataclass(frozen=True)
class CartesianTopology:
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
        """Return the ``meta["cart"]`` descriptor for this topology."""
        return {
            "grid_shape": self.grid_shape,
            "coords": self.coords,
            "periods": (False,) * len(self.dims),
        }

    def sub_comm(self, merge_axes: Sequence[str]) -> Comm:
        """Return the communicator grouping ranks for a partial collective.

        The returned communicator spans every coordinate along each axis
        in ``merge_axes`` (ranks that differ only there are grouped
        together for the collective) and is restricted to this rank's own
        coordinate along every other partition axis (ranks that differ
        there are kept in separate groups, since they own physically
        different data along that axis). Built with ``Cartcomm.Sub`` and
        cached per distinct ``merge_axes`` set for the life of this
        topology object, since ``Sub`` is itself a communicator-creation
        collective and should not be repeated on every reduction call --
        see :func:`get_cartesian_topology` for how the topology itself is
        cached across calls.

        Parameters
        ----------
        merge_axes : sequence of str
            Subset of :attr:`dims` to group ranks across. When it equals
            the full set of :attr:`dims`, the result spans every rank in
            :attr:`cart_comm` (equivalent to a full-communicator
            collective).

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


def build_cartesian_topology(
    comm: Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianTopology:
    """Build a rank's Cartesian topology for a multi-dimensional partition.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
        Communicator to lay out. Every rank must call this together.
    dims : sequence of str
        Partition dimension names, in the order they should be laid out
        across Cartesian axes. Must have at least two entries -- a single
        partition dimension should use the existing one-dimensional fast
        path instead, which needs no Cartesian communicator at all.
    sizes : mapping of str to int
        Global length of each dimension in ``dims``.

    Returns
    -------
    CartesianTopology
        This rank's view of the process grid.

    Raises
    ------
    ValueError
        If fewer than two dimensions are given.
    """
    if len(dims) < 2:
        raise ValueError(
            "build_cartesian_topology() requires at least two partition "
            + f"dimensions; got {tuple(dims)!r}. A single partition dimension "
            + "uses the existing one-dimensional path directly."
        )

    extents = [int(sizes[dim]) for dim in dims]
    grid_shape = compute_layout(extents, comm.size)

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

    return CartesianTopology(
        dims=tuple(dims),
        grid_shape=grid_shape,
        coords=coords,
        cart_comm=cart_comm,
        bounds=bounds,
        neighbors=neighbors,
    )


# One MPI communicator "attribute" keyval, created once at import time, used
# to cache each communicator's CartesianTopology objects on the communicator
# itself via Comm.Set_attr/Get_attr -- MPI's own mechanism for attaching
# application state to a communicator's lifetime. This is deliberately not a
# plain module-level dict keyed by id(comm): a communicator can be freed and
# a new, unrelated one later allocated at the same id(), which would silently
# return a stale, wrong topology from an id-keyed cache. Comm.Set_attr avoids
# that: the attribute lives and dies with the specific communicator object
# (mpi4py calls the delete callback -- none is registered here, so this is a
# no-op cleanup -- when the communicator is freed), and a Dup()'d or
# otherwise distinct communicator starts with no attribute at all rather than
# inheriting one, which was confirmed directly against this mpi4py build.
_TOPOLOGY_KEYVAL = MPI.Comm.Create_keyval()


def get_cartesian_topology(
    comm: Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianTopology:
    """Return (building and caching once) a rank's Cartesian topology.

    Every call with the same ``comm`` and ``dims`` returns the identical
    :class:`CartesianTopology` object (including its already-built
    ``cart_comm`` and any ``sub_comm`` communicators built so far) instead
    of repeating the ``Create_cart`` collective -- important since this is
    called from every distributed reduction, halo exchange, and
    (re)partition, not just once at startup.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
        Communicator to lay out. Every rank must call this together with
        the same ``dims``.
    dims : sequence of str
        Partition dimension names, in Cartesian-axis order. Must have at
        least two entries; see :func:`build_cartesian_topology`.
    sizes : mapping of str to int
        Global length of each dimension in ``dims``. Only consulted the
        first time a given ``(comm, dims)`` pair is requested -- a
        mismatched ``sizes`` on a later call with the same ``dims`` is not
        detected, since the topology (a function of ``dims`` and rank
        count only, not of ``sizes``' particular values beyond the first
        build) is already cached.

    Returns
    -------
    CartesianTopology
        This rank's (cached) view of the process grid.
    """
    dims = tuple(dims)
    cache = comm.Get_attr(_TOPOLOGY_KEYVAL)
    if cache is None:
        cache = {}
        comm.Set_attr(_TOPOLOGY_KEYVAL, cache)
    cached = cache.get(dims)
    if cached is not None:
        return cached
    topology = build_cartesian_topology(comm, dims, sizes)
    cache[dims] = topology
    return topology


def dim_comm(runtime: Any, meta: Mapping[str, Any], dim: str) -> Comm:
    """Return the communicator whose ranks vary along ``dim`` alone.

    Shared by every caller that needs to reason about rank order or rank
    ownership strictly along one partition axis -- "which rank owns
    global index/label i along dim", "gather every rank's total along dim
    in order" -- rather than a value-order-independent collective like a
    sum or a min/max Allreduce (which any full- or sub-communicator
    computes identically regardless of member order). Mixing in ranks
    that vary along a different, unrelated axis would attribute another
    axis's slice to the wrong position; see
    :meth:`~.reductions.Reduction._first_last_combine` for the first
    place this distinction mattered.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator this resolves against.
    meta : mapping
        Distribution metadata (as returned by
        :func:`~.meta.get_mpi_meta`) of the object being operated on.
    dim : str
        The single partition dimension to resolve a communicator for.

    Returns
    -------
    mpi4py.MPI.Comm
        The full runtime communicator for the one-dimensional case
        (unchanged behavior), or the cached Cartesian sub-communicator
        fixed on every other partition axis otherwise -- see
        :meth:`CartesianTopology.sub_comm`.
    """
    dims = meta["dims"]
    if len(dims) <= 1 or "cart" not in meta:
        return cast("Comm", runtime.comm)
    topology = get_cartesian_topology(runtime.comm, dims, meta["global_sizes"])
    return topology.sub_comm((dim,))
