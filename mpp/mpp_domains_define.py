"""Define how an array is divided across ranks.

Mirrors FMS ``mpp/include/mpp_domains_define.inc``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..mpi.mpi_init import MPI
from ..xarray.chunks import get_balanced_bounds
from .mpp_domains import CartesianDomain, Domain, _no_proc_null

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mpi4py.MPI import Comm

    from ..mpi.context import MPIContext


_TOPOLOGY_KEYVAL = MPI.Comm.Create_keyval()


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
