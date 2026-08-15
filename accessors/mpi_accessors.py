"""MPI collective operations exposed on xarray objects.

Reached two ways, which are the same object::

    ds.xgeo.mpi.sum()      # through the geospatial accessor
    ds.mpi.sum()           # directly

Every method here is collective over ``MPI_COMM_WORLD``: each rank contributes
the object the accessor is bound to, and every rank must call the method, in
the same order. Reductions combine the ranks' contributions in rank order, so
the result is identical on every rank down to the last bit.

Nothing in this module initializes MPI at import. Process coordination is
delegated to ``MPI.world``, whose shared coordinator is created lazily when an
operation first needs it. ``MPI_Init_thread`` therefore runs only on first use.

Without the compiled extension, and only when the launcher reports a single
task, every operation degrades to the identity: a reduction returns the local
object, a gather returns a one-item list. The same script therefore runs under
``python`` and under ``mpirun -n N python``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from ..lib_mpi import mpi

if TYPE_CHECKING:
    from typing import Literal


class MPIAccessor:
    """
    Collective operations bound to a DataArray or Dataset.

    Parameters
    ----------
    xarray_obj : xarray.DataArray or xarray.Dataset
        Local contribution of this rank.
    """

    __slots__ = ("_obj",)

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset) -> None:
        self._obj = xarray_obj

    def __repr__(self) -> str:
        kind = type(self._obj).__name__
        return f"<xgeo mpi accessor on {kind}>"

    # -- admin & environment ----------------------------------------------
    def available(self) -> bool:
        """
        Return whether the native MPI runtime can be loaded and initialized.

        Returns
        -------
        bool
            True if MPI runtime is available, False otherwise.
        """
        return mpi.world.available()

    def launcher_size(self) -> int:
        """
        Return the world size advertised by the process launcher.

        Returns
        -------
        int
            World size retrieved from launcher metadata.
        """
        return mpi.world.launcher_size()

    def rank(self) -> int:
        """
        Return this process's rank in ``MPI_COMM_WORLD``.

        Returns
        -------
        int
            Rank of the current process.
        """
        return mpi.world.rank()

    def size(self) -> int:
        """
        Return the number of ranks in ``MPI_COMM_WORLD``.

        Returns
        -------
        int
            Total number of processes in the world.
        """
        return mpi.world.size()

    def is_root(self, root: int = 0) -> bool:
        """
        Return whether this process has rank ``root``.

        Parameters
        ----------
        root : int, default 0
            The rank to check against.

        Returns
        -------
        bool
            True if the process matches the root rank, False otherwise.
        """
        return mpi.world.is_root(root)

    def abort(self, code: int = 1) -> None:
        """
        Abort all ranks in ``MPI_COMM_WORLD`` with a process exit code.

        Parameters
        ----------
        code : int, default 1
            Exit code to return to the process launcher.
        """
        mpi.world.abort(code)

    def finalize(self) -> None:
        """
        Finalize MPI when initialized by the shared world coordinator.
        """
        mpi.world.finalize()

    # -- synchronization --------------------------------------------------
    def barrier(self) -> xr.DataArray | xr.Dataset:
        """
        Wait for every rank, then return the bound object unchanged.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The local object, allowing the call to sit inside a method chain.
        """
        mpi.world.barrier()
        return self._obj

    def consensus(self, ok: bool | None = None) -> bool:
        """
        Return True only when every rank contributes a true value.

        Parameters
        ----------
        ok : bool, optional
            An explicit boolean value to contribute. If not provided,
            the bound xarray object is implicitly reduced to a boolean.

        Returns
        -------
        bool
            True if all ranks evaluate to True, False otherwise.
        """
        if ok is None:
            if isinstance(self._obj, xr.Dataset):
                ok = all(bool(da.all().item()) for da in self._obj.data_vars.values())
            else:
                ok = bool(self._obj.all().item())
        return mpi.world.consensus(ok)

    # -- data movement ----------------------------------------------------
    def bcast(self, root: int = 0) -> xr.DataArray | xr.Dataset:
        """
        Replace this object with the one held by ``root``.

        Parameters
        ----------
        root : int, default 0
            Source rank.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The object held by ``root``, distributed to every rank.
        """
        return mpi.world.bcast(self._obj, root=root)

    def gather(self, root: int = 0) -> list[xr.DataArray | xr.Dataset] | None:
        """
        Collect every rank's object onto ``root``.

        Parameters
        ----------
        root : int, default 0
            Destination rank.

        Returns
        -------
        list of xarray.DataArray or xarray.Dataset, or None
            Objects from all ranks in rank order on ``root``, ``None``
            elsewhere.

        Notes
        -----
        The whole distributed dataset lands in one process, so this is a
        memory cliff on anything large. Use :meth:`to_netcdf` to write a
        distributed dataset without gathering it.
        """
        return mpi.world.gather(self._obj, root=root)

    def allgather(self) -> list[xr.DataArray | xr.Dataset]:
        """
        Collect every rank's object onto every rank, in rank order.

        Returns
        -------
        list of xarray.DataArray or xarray.Dataset
            Objects from all ranks in rank order, identical everywhere.
        """
        return mpi.world.allgather(self._obj)

    def scatter(self, dim: str, root: int = 0) -> xr.DataArray | xr.Dataset:
        """
        Split ``root``'s object along ``dim`` and keep this rank's block.

        Parameters
        ----------
        dim : str
            Dimension to partition.
        root : int, default 0
            Rank holding the global object.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            This rank's contiguous block.

        Notes
        -----
        Blocks are contiguous and the remainder falls on the leading ranks,
        which is the layout :meth:`to_netcdf` requires: it recovers each
        rank's file offset from an all-gather of the local lengths, so a
        strided split would scatter a rank's records across the whole file.
        """
        if not isinstance(dim, str) or not dim:
            raise TypeError("dim must be a non-empty string.")

        obj = mpi.world.bcast(self._obj, root=root)
        if dim not in obj.sizes:
            raise KeyError(f"Dimension {dim!r} is not present in the dataset.")
        start, stop = mpi.world.partition(int(obj.sizes[dim]))
        return obj.isel({dim: slice(start, stop)})

    def concat(self, dim: str, root: int | None = None) -> xr.DataArray | xr.Dataset:
        """
        Join every rank's object along ``dim`` in rank order.

        Parameters
        ----------
        dim : str
            Dimension to concatenate along, normally the partitioned one.
        root : int or None, optional
            Rank the result is assembled on. If None, assemble on every rank.
            Default is None.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The reassembled global object, or the local object unchanged on
            non-root ranks when ``root`` is given.

        Notes
        -----
        This materializes the global object in one process's memory per rank
        that receives it, which is what the parallel writer exists to avoid.
        It is meant for a final reduced result, not for the partitioned data.
        """
        if not isinstance(dim, str) or not dim:
            raise TypeError("dim must be a non-empty string.")

        if root is None:
            parts = mpi.world.allgather(self._obj)
        else:
            gathered = mpi.world.gather(self._obj, root=root)
            if gathered is None:
                return self._obj
            parts = gathered
        return xr.concat(parts, dim=dim)

    def partition(self, dim: str) -> xr.DataArray | xr.Dataset:
        """
        Keep only this rank's contiguous block along ``dim``.

        Parameters
        ----------
        dim : str
            Dimension to partition.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            This rank's block.

        Notes
        -----
        Unlike :meth:`scatter`, nothing is communicated. Every rank is assumed
        to already hold, or be able to open lazily, the same global object,
        and simply selects its own slice. That is the cheaper pattern when the
        data comes from a shared filesystem.
        """
        if not isinstance(dim, str) or not dim:
            raise TypeError("dim must be a non-empty string.")
        if dim not in self._obj.sizes:
            raise KeyError(f"Dimension {dim!r} is not present in the dataset.")
        start, stop = mpi.world.partition(int(self._obj.sizes[dim]))
        return self._obj.isel({dim: slice(start, stop)})

    def split(self, dim: str) -> xr.DataArray | xr.Dataset:
        """
        Keep only this rank's contiguous block along ``dim``.

        Alias for :meth:`partition`.

        Parameters
        ----------
        dim : str
            Dimension to partition.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            This rank's block.
        """
        return self.partition(dim)

    # -- reductions -------------------------------------------------------
    def reduce(
        self, op: Literal["sum", "prod", "min", "max", "any", "all"] = "sum"
    ) -> xr.DataArray | xr.Dataset:
        """
        Combine this rank's object with every other rank's.

        Parameters
        ----------
        op : {"sum", "prod", "min", "max", "any", "all"}, default "sum"
            Associative operator applied across ranks.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Reduction over all ranks, identical on every rank.

        Notes
        -----
        The operator is applied to whole xarray objects, so xarray's alignment
        rules apply: ranks holding different coordinate values along a shared
        dimension produce the intersection, not an error. Reduce objects that
        are already aligned, which for a partitioned workload means the
        per-rank partial results rather than the partitioned data itself.
        """
        return mpi.world.reduce(self._obj, op)

    def sum(self) -> xr.DataArray | xr.Dataset:
        """
        Sum this object across ranks.

        See :meth:`reduce` for further details.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The sum across all ranks.
        """
        return mpi.world.sum(self._obj)

    def prod(self) -> xr.DataArray | xr.Dataset:
        """
        Multiply this object across ranks.

        See :meth:`reduce` for further details.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The product across all ranks.
        """
        return mpi.world.prod(self._obj)

    def min(self) -> xr.DataArray | xr.Dataset:
        """
        Elementwise minimum across ranks.

        See :meth:`reduce` for further details.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The minimum across all ranks.
        """
        return mpi.world.min(self._obj)

    def max(self) -> xr.DataArray | xr.Dataset:
        """
        Elementwise maximum across ranks.

        See :meth:`reduce` for further details.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The maximum across all ranks.
        """
        return mpi.world.max(self._obj)

    def any(self) -> xr.DataArray | xr.Dataset:
        """
        Elementwise logical OR across ranks.

        See :meth:`reduce` for further details.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The logical OR across all ranks.
        """
        return mpi.world.any(self._obj)

    def all(self) -> xr.DataArray | xr.Dataset:
        """
        Elementwise logical AND across ranks.

        See :meth:`reduce` for further details.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The logical AND across all ranks.
        """
        return mpi.world.all(self._obj)

    def mean(self) -> xr.DataArray | xr.Dataset:
        """
        Arithmetic mean over ranks of the bound objects.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            ``sum over ranks / size``, identical on every rank.

        Notes
        -----
        Each rank carries equal weight. This is not the mean over a
        partitioned dimension: a rank holding three records would count as
        much as a rank holding three thousand. A partitioned mean is a ratio
        of two sums, so form the local weighted numerator and the local
        weight, reduce each with :meth:`sum`, and divide afterwards.
        """
        return mpi.world.mean(self._obj)


@xr.register_dataarray_accessor("mpi")
class MPIDataArray(MPIAccessor):
    """``.mpi`` accessor on a ``DataArray``."""

    __slots__ = ()


@xr.register_dataset_accessor("mpi")
class MPIDataset(MPIAccessor):
    """``.mpi`` accessor on a ``Dataset``."""

    __slots__ = ()


__all__ = ["MPIAccessor", "MPIDataArray", "MPIDataset"]
