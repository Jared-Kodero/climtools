"""Manage Cartesian MPI process-grid topology for multidimensional xarray partitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from mpi4py import MPI

from .chunks import get_balanced_bounds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mpi4py.MPI import Cartcomm, Comm

    from ..mpi.runtime import MPIContext

__all__ = [
    "CartesianTopology",
    "build_cartesian_topology",
    "compute_layout",
    "get_cartesian_topology",
]


def compute_layout(extents: Sequence[int], nranks: int) -> tuple[int, ...]:
    """Choose a process-grid shape balancing work per rank across axes.

    Parameters
    ----------
    extents : sequence of int
        Global length of each partition dimension, in axis order.
    nranks : int
        Number of MPI ranks (process-grid cells) to lay out.
    Returns
    -------
    tuple of int
        Process-grid shape, one entry per axis, with ``math.prod(shape) == nranks``.

    Raises
    ------
    ValueError
        If ``extents`` is empty, any extent is not positive, or ``nranks`` is not positive.
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


def build_cartesian_topology(
    comm: MPI.Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianTopology:
    """Build a rank's Cartesian topology for a multi-dimensional partition.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
        Communicator to lay out.
    dims : sequence of str
        Partition dimension names, in the order they should be laid out across Cartesian axes.
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
            f"build_cartesian_topology() requires at least two partition "
            + f"dimensions; got {tuple(dims)!r}"
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
    comm: MPI.Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianTopology:
    """Return (building and caching once) a rank's Cartesian topology.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
        Communicator to lay out.
    dims : sequence of str
        Partition dimension names, in Cartesian-axis order.
    sizes : mapping of str to int
        Global length of each dimension in ``dims``.
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


def dim_comm(runtime: MPIContext, meta: Mapping[str, Any], dim: str) -> Comm:
    """Return the communicator whose ranks vary along ``dim`` alone.

    Parameters
    ----------
    runtime : MPIContext
        Runtime whose communicator this resolves against.
    meta : mapping
        Distribution metadata (as returned by :func:`~.meta.get_mpi_meta`) of the object being operated on.
    dim : str
        The single partition dimension to resolve a communicator for.
    Returns
    -------
    mpi4py.MPI.Comm
        The full runtime communicator for the one-dimensional case (unchanged behavior),or
        the cached Cartesian sub-communicator fixed on every other partition axis otherwise
        -- see :meth:`CartesianTopology.sub_comm`.
    """
    dims = meta["dims"]
    if len(dims) <= 1 or "cart" not in meta:
        return cast("Comm", runtime.comm)
    topology = get_cartesian_topology(runtime.comm, dims, meta["global_sizes"])
    return topology.sub_comm((dim,))
