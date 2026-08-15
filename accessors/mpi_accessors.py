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

from ..lib_mpi import MPI

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike
    from typing import Any, Literal


class MPIAccessor:
    """Collective operations bound to a DataArray or Dataset.

    Parameters
    ----------
    xarray_obj : xarray.DataArray or xarray.Dataset
        Local contribution of this rank.
    """

    __slots__ = ("_obj",)

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset) -> None:
        self._obj = xarray_obj

    @property
    def rank(self) -> int:
        """Rank of this process in ``MPI_COMM_WORLD``."""
        return MPI.world.rank()

    @property
    def size(self) -> int:
        """Number of ranks in ``MPI_COMM_WORLD``."""
        return MPI.world.size()

    @property
    def is_root(self) -> bool:
        """Whether this process is rank zero."""
        return MPI.world.is_root()

    def __repr__(self) -> str:
        kind = type(self._obj).__name__
        return f"<xgeo mpi accessor on {kind}>"

    # -- reductions -------------------------------------------------------
    def reduce(
        self,
        op: Literal["sum", "prod", "min", "max", "any", "all"] = "sum",
    ) -> xr.DataArray | xr.Dataset:
        """Combine this rank's object with every other rank's.

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
        return MPI.world.reduce(self._obj, op)

    def sum(self) -> xr.DataArray | xr.Dataset:
        """Sum this object across ranks. See :meth:`reduce`."""
        return MPI.world.sum(self._obj)

    def prod(self) -> xr.DataArray | xr.Dataset:
        """Multiply this object across ranks. See :meth:`reduce`."""
        return MPI.world.prod(self._obj)

    def min(self) -> xr.DataArray | xr.Dataset:
        """Elementwise minimum across ranks. See :meth:`reduce`."""
        return MPI.world.min(self._obj)

    def max(self) -> xr.DataArray | xr.Dataset:
        """Elementwise maximum across ranks. See :meth:`reduce`."""
        return MPI.world.max(self._obj)

    def any(self) -> xr.DataArray | xr.Dataset:
        """Elementwise logical OR across ranks. See :meth:`reduce`."""
        return MPI.world.any(self._obj)

    def all(self) -> xr.DataArray | xr.Dataset:
        """Elementwise logical AND across ranks. See :meth:`reduce`."""
        return MPI.world.all(self._obj)

    def mean(self) -> xr.DataArray | xr.Dataset:
        """Arithmetic mean over ranks of the bound objects.

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
        return MPI.world.mean(self._obj)

    # -- movement ---------------------------------------------------------
    def bcast(self, root: int = 0) -> xr.DataArray | xr.Dataset:
        """Replace this object with the one held by ``root``.

        Parameters
        ----------
        root : int, default 0
            Source rank.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The object held by ``root``, on every rank.
        """
        return MPI.world.bcast(self._obj, root=root)

    def gather(self, root: int = 0) -> list[Any] | None:
        """Collect every rank's object onto ``root``.

        Parameters
        ----------
        root : int, default 0
            Destination rank.

        Returns
        -------
        list or None
            Objects from all ranks in rank order on ``root``, ``None``
            elsewhere.

        Notes
        -----
        The whole distributed dataset lands in one process, so this is a
        memory cliff on anything large. Use :meth:`to_netcdf` to write a
        distributed dataset without gathering it.
        """
        return MPI.world.gather(self._obj, root=root)

    def allgather(self) -> list[Any]:
        """Collect every rank's object onto every rank, in rank order."""
        return MPI.world.allgather(self._obj)

    def concat(self, dim: str, root: int | None = None) -> xr.DataArray | xr.Dataset:
        """Join every rank's object along ``dim`` in rank order.

        Parameters
        ----------
        dim : str
            Dimension to concatenate along, normally the partitioned one.
        root : int or None, optional
            Rank the result is assembled on. If ``None``, assemble on every
            rank.

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
            parts = MPI.world.allgather(self._obj)
        else:
            gathered = MPI.world.gather(self._obj, root=root)
            if gathered is None:
                return self._obj
            parts = gathered
        return xr.concat(parts, dim=dim)

    def scatter(self, dim: str, root: int = 0) -> xr.DataArray | xr.Dataset:
        """Split ``root``'s object along ``dim`` and keep this rank's block.

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

        obj = MPI.world.bcast(self._obj, root=root)
        if dim not in obj.sizes:
            raise KeyError(f"Dimension {dim!r} is not present in the dataset.")
        start, stop = MPI.world.partition(int(obj.sizes[dim]))
        return obj.isel({dim: slice(start, stop)})

    def partition(self, dim: str) -> xr.DataArray | xr.Dataset:
        """Keep only this rank's contiguous block along ``dim``.

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
        start, stop = MPI.world.partition(int(self._obj.sizes[dim]))
        return self._obj.isel({dim: slice(start, stop)})

    # -- synchronization --------------------------------------------------
    def barrier(self) -> xr.DataArray | xr.Dataset:
        """Wait for every rank, then return the bound object unchanged.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The local object, so the call can sit inside a method chain.
        """
        MPI.world.barrier()
        return self._obj

    # -- output -----------------------------------------------------------
    def to_netcdf(
        self,
        file: str | PathLike[str],
        partition_dim: str | None = None,
        *,
        unlimited_dim: str | Iterable[str] = (),
        zlib: bool = False,
        complevel: int = 4,
        shuffle: bool = True,
        chunks: Any = None,
        hints: str | None = None,
        nofill: bool = True,
        allow_serial: bool = True,
        strict_compression: bool = False,
    ) -> str:
        """Write this rank's slab into one shared NetCDF-4 file.

        Every rank contributes its contiguous block of ``partition_dim`` and
        the file is written once, collectively. Nothing is gathered to rank
        zero and no per-rank files are produced.

        Parameters
        ----------
        file : str or os.PathLike
            Output path, visible to every rank. An existing file is replaced.
        partition_dim : str or None, optional
            Dimension partitioned across ranks. Inferred when omitted, from
            the one dimension whose length or coordinate values differ
            between ranks.
        unlimited_dim : str or iterable of str, default ()
            Dimensions defined as unlimited record dimensions.
        zlib : bool, default False
            Request deflate compression. Compression during a collective
            write needs a NetCDF-C and HDF5 built with parallel filters,
            which many stacks lack; the default is therefore off.
        complevel : int, default 4
            Deflate level from 1 to 9, used when ``zlib`` is true.
        shuffle : bool, default True
            Apply the HDF5 shuffle filter alongside compression.
        chunks : mapping of str to iterable of int, optional
            Explicit chunk shape for selected variables.
        hints : str or None, optional
            Semicolon-separated MPI-IO hints in ``key=value`` form.
        nofill : bool, default True
            Disable NetCDF pre-filling, which is a large speed-up.
        allow_serial : bool, default True
            Permit a one-rank world, so the same call works unlaunched.
        strict_compression : bool, default False
            Fail rather than warn when compression was requested but the
            linked libraries cannot apply it in parallel.

        Returns
        -------
        str
            The output path, after the collective write completes.

        Notes
        -----
        All ranks must supply matching variable names, dtypes, dimension
        names and attributes. Arrays that do not carry ``partition_dim`` are
        treated as replicated and are checked for bit-identity across ranks
        before the write, so a per-rank difference is reported rather than
        silently resolved in rank zero's favour.
        """
        from ..lib_netcdf.parallel import to_netcdf_parallel

        return to_netcdf_parallel(
            self._obj,
            file,
            partition_dim=partition_dim,
            deflate=complevel if zlib else None,
            shuffle=shuffle,
            chunks=chunks,
            unlimited_dim=unlimited_dim,
            hints=hints,
            nofill=nofill,
            allow_serial=allow_serial,
            strict_compression=strict_compression,
        )


@xr.register_dataarray_accessor("mpi")
class MPIDataArray(MPIAccessor):
    """``.mpi`` accessor on a ``DataArray``."""

    __slots__ = ()


@xr.register_dataset_accessor("mpi")
class MPIDataset(MPIAccessor):
    """``.mpi`` accessor on a ``Dataset``."""

    __slots__ = ()


__all__ = ["MPIAccessor", "MPIDataArray", "MPIDataset"]
