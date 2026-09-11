"""Domain types shared by the ``mpp_domains`` modules.

Mirrors FMS ``mpp/mpp_domains.F90``, which declares the domain types and
leaves the routines that act on them to the files it includes:
:mod:`~climtools.mpp.mpp_domains_define`,
:mod:`~climtools.mpp.mpp_domains_util`,
:mod:`~climtools.mpp.mpp_do_update`,
:mod:`~climtools.mpp.mpp_group_update`,
:mod:`~climtools.mpp.mpp_global_field` and
:mod:`~climtools.mpp.mpp_global_reduce`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mpi4py.MPI import Cartcomm, Comm


from ..mpi.diagnostics import MPIError


class DomainMismatchError(MPIError):
    """Raised when rank-local domains do not tile the global domain."""


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
    cyclic : dict[str, bool]
        Whether each dimension wraps at the global edges, so the first and
        last ranks are neighbours. FMS carries this on the domain as
        ``CYCLIC_GLOBAL_DOMAIN`` rather than passing it per call.
    fold : int
        Folded edges, from :mod:`~climtools.mpp.mpp_parameter`. Zero means
        none. A fold joins an axis to itself in reverse, closing a tripolar
        grid across the pole.

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
    cyclic: dict[str, bool] = field(default_factory=dict)
    fold: int = 0

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
