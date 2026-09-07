"""Manage Cartesian MPI process-grid topology for multidimensional xarray partitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from ..mpi.mpi_init import MPI
from .chunks import get_balanced_bounds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mpi4py.MPI import Cartcomm, Comm

    from ..mpi.context import MPIContext

__all__ = [
    "CartesianTopology",
    "build_cartesian_topology",
    "compute_layout",
    "get_cartesian_topology",
]


def compute_layout(extents: Sequence[int], nranks: int) -> tuple[int, ...]:
    """Choose a process-grid shape balancing work per rank across axes.

    Raises
    ------
    ValueError
        If ``extents`` is empty, any extent is not positive, or ``nranks`` is not positive.

    """
    if not extents:
        raise ValueError("requires at least one extent")
    if any(extent <= 0 for extent in extents):
        raise ValueError(f"All extents must be positive; got {tuple(extents)!r}.")
    if nranks <= 0:
        raise ValueError(f"nranks must be positive; got {nranks}.")

    # Use FMS's 2-D layout algorithm for the supported Cartesian case; keep the N-D
    # heuristic as fallback.
    if len(extents) == 2:
        from .mpp import mpp_define_layout

        return mpp_define_layout(extents[0], extents[1], nranks)

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


# Cache topologies on the communicator so cache lifetime follows the MPI communicator
# lifetime.
_TOPOLOGY_KEYVAL = MPI.Comm.Create_keyval()


def get_cartesian_topology(
    comm: MPI.Intracomm,
    dims: Sequence[str],
    sizes: Mapping[str, int],
) -> CartesianTopology:
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
    topology = build_cartesian_topology(comm, dims, sizes)
    cache[cache_key] = topology
    return topology


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
        Full communicator for 1-D partitions or the corresponding Cartesian subcommunicator.
    """
    dims = meta["dims"]
    if len(dims) <= 1 or "cart" not in meta:
        return cast("Comm", mpi_context.comm)
    topology = get_cartesian_topology(mpi_context.comm, dims, meta["global_sizes"])
    return topology.sub_comm((dim,))
