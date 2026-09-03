"""Provide the MPI-distributed xarray wrapper."""

from __future__ import annotations

import operator as _operator
from functools import wraps
from typing import TYPE_CHECKING, Any

from mpi4py import MPI

import xarray as xr

from .arithmetic import (
    align,
    apply,
    coarsen_reduce,
    evaluate,
    halo_exchange,
    matmul,
    reindex,
    rolling_reduce,
    sortby,
)
from .elementwise import (
    bfill,
    cumsum,
    diff,
    differentiate,
    ffill,
    interp,
    median,
    roll,
    shift,
    where,
)
from .handles import MPIGroupBy, MPIResample, MPIRolling
from .indexing import isel, sel
from .io import attach_save_chunks, repartition
from .meta import PARTITIONED_ATTR as _PARTITIONED_ATTR
from .meta import assign_mpi_meta, get_mpi_meta, strip_mpi_meta
from .reductions import (
    all_reduce,
    any_reduce,
    first_reduce,
    last_reduce,
    max_reduce,
    mean_reduce,
    min_reduce,
    prod_reduce,
    sum_reduce,
)
from .statistics import std, var

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping
    from os import PathLike
    from types import EllipsisType
    from typing import Literal

    import numpy as np

    from ..mpi.context import MPIContext


#: Attrs key for the lightweight boolean flag :func:`mark_partitioned`
#: stamps onto ``MPIXarray.data`` (and, for a Dataset, every distributed
#: variable) after the full ``mpi_meta`` dict is popped into ``.meta``.
#: ``.meta`` (or, inside the engine, the ``mpi_meta`` dict reattached
#: transiently by :meth:`MPIXarray._prepare`) remains the source of truth
#: for distribution state; this exists so code that sees ``.data`` on its
#: own (e.g. after it is handed to a plain xarray function) has a cheap,
#: human-inspectable hint that it was part of an MPI partition. Defined in
#: :mod:`.meta` as :data:`~.meta.PARTITIONED_ATTR` (imported here under its
#: original name) so :func:`~.meta.strip_mpi_meta` and every NetCDF
#: attribute writer in :mod:`.netcdf` filter the identical key -- see
#: :func:`~.meta.strip_export_attrs`.

#: Sentinel distinguishing "no fill value given" from a genuine ``other=None``
#: in :meth:`MPIXarray.where`.
_WHERE_UNSET = object()

#: Sentinel distinguishing "fill_value not given" (defer to the engine's own
#: ``NaN`` default) from a genuine ``fill_value=None`` in :meth:`MPIXarray.reindex`.
_FILL_VALUE_UNSET = object()

#: Explicit allowlist of read-only xarray properties ``MPIXarray.__getattr__``
#: forwards to ``self.data``. Deliberately a fixed list, not a
#: ``callable(...)`` check: a heuristic based on callability would pass
#: through non-callable-but-still-wrong things (e.g. accessor namespaces)
#: just as easily as it would block a legitimate future property, and is
#: no easier to audit than this list is.
_SAFE_PASSTHROUGH_ATTRS: frozenset[str] = frozenset(
    {
        "attrs",
        "chunks",
        "chunksizes",
        "coords",
        "data_vars",
        "dims",
        "dtype",
        "dtypes",
        "encoding",
        "indexes",
        "name",
        "nbytes",
        "ndim",
        "shape",
        "size",
        "sizes",
        "T",
        "values",
        "variables",
    }
)

#: Explicit allowlist of xarray methods ``MPIXarray.__getattr__`` forwards
#: to ``self.apply(...)`` -- i.e. runs rank-locally, through the same
#: partition-preservation check every other ``apply()`` call gets, and
#: with zero MPI communication when the partition dimension isn't touched
#: at all (which for every method below is *always*, regardless of what
#: dimension name is passed).
#:
#: Deliberately an allowlist, not a blocklist of "known-dangerous" names:
#: several xarray methods are *structurally* partition-preserving (same
#: length, same coordinate labels -- passes ``apply()``'s existing
#: post-call check) while being *value*-wrong without a neighboring
#: rank's data when applied along the partition dimension --
#: ``shift``, ``rolling(...).reduce()``, ``differentiate``, ``pad``,
#: ``interpolate_na``/``ffill``/``bfill``, roughly the same family
#: ``rolling_reduce``/``halo_exchange`` exist to handle correctly. Those
#: would silently produce wrong values through a blind passthrough, not
#: raise -- the structural check has no way to catch them. Only names
#: individually verified never to depend on dimension content this way,
#: for any arguments, belong here. ``squeeze`` is deliberately excluded
#: for a related but distinct reason: whether it removes the partition
#: dimension can depend on this rank's own local length in an uneven
#: partition, which risks the exact asymmetric-raise hazard (one rank's
#: post-call check fails, others' don't, and only the failing rank ever
#: raises) fixed for ``halo_exchange`` earlier -- not addressed here.
_SAFE_PASSTHROUGH_METHODS: frozenset[str] = frozenset(
    {
        "astype",
        "assign_attrs",
        "assign_coords",
        "chunk",
        "clip",
        "compute",
        "drop_vars",
        "expand_dims",
        "fillna",
        "load",
        "persist",
        "rename",
        "rename_dims",
        "rename_vars",
        "reset_coords",
        "round",
        "set_coords",
        "transpose",
    }
)


# store this we will monkey patch to all ds/da binary op on MPIXarray instance
dataarray_binary_op = xr.DataArray._binary_op
dataset_binary_op = xr.Dataset._binary_op


class MPIXarray:
    """MPI aware xarray Dataset/DataArray object

    Parameters
    ----------
    data : MPIXarray or xarray.Dataset or xarray.DataArray
        Rank-local data. Existing ``MPIXarray`` instances are adopted unchanged.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime or communicator used by distributed operations.
    meta : dict, optional
        Explicit distribution metadata. If omitted, metadata is read from ``data``.
    auto_partition : bool, default True
        Partition replicated input that has no distribution metadata.
    dim : Hashable or {"auto"}, default "auto"
        Dimension used when auto-partitioning.
    chunk_info : mapping of str to int, optional
        Chunk-size hints used when auto-partitioning.
    log_partitions : bool, default False
        Log the partition layout when auto-partitioning.

    Notes
    -----
    Distribution metadata is stored on ``.meta`` and attached to xarray attributes
    only while an engine operation is executing. Replicated input is assumed to be
    present on every rank; use :func:`~.constructors.mpi_partition_data` for
    root-owned input.
    """

    #: NumPy ufunc dispatch (`np.log(mpixarray)`, `np.add(a, b)`, ...): for
    #: an elementwise call (`method == "__call__"`, the overwhelming
    #: majority of ufuncs -- `log`, `sqrt`, `exp`, `sin`, `add`,
    #: `multiply`, `isnan`, ...), every input is elementwise-independent
    #: by definition, so it is exactly the kind of partition-preserving,
    #: rank-local callable `apply()` exists for: no communication, and
    #: the result stays a distributed MPIXarray rather than being
    #: silently gathered onto every rank (`apply()`/`check_operands_distribution`
    #: already validate any MPIXarray operands share a compatible
    #: partition -- the same check `__add__`/etc. below rely on -- and a
    #: plain scalar or numpy array operand is left untouched, so ordinary
    #: broadcasting applies exactly as it would for `self.data`). Any
    #: other method (`reduce`, `accumulate`, `outer`, `at`, ...) is a
    #: non-elementwise, potentially cross-rank operation (`np.add.reduce`
    #: is a sum across the array, for instance) that must not be routed
    #: through this rank-local path -- returning `NotImplemented` lets
    #: NumPy raise its own clear error rather than silently mishandling
    #: it; use the dedicated distributed method (`.sum()`, `.cumsum()`,
    #: ...) instead. `out=` is similarly refused: MPIXarray is immutable
    #: by construction (see the class docstring), so there is no rank-local
    #: buffer to write into in place.
    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Dispatch NumPy ufuncs through the distributed wrapper."""
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        return self.apply(ufunc, *inputs, **kwargs)

    def __init__(
        self,
        data: MPIXarray | xr.Dataset | xr.DataArray,
        mpi_context: MPIContext | MPI.Intracomm,
        meta: dict[str, Any] | None = None,
        *,
        auto_partition: bool = True,
        dim: Hashable | Literal["auto"] = "auto",
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> None:
        """Initialize a distributed xarray wrapper."""
        from ..mpi.context import MPIContext

        if not isinstance(mpi_context, MPIContext):
            mpi_context = MPIContext(mpi_context)

        if isinstance(data, MPIXarray):
            self.data = data.data
            self.meta = data.meta
            self._runtime = data._runtime
            return

        if meta is None:
            meta = get_mpi_meta(data)
            if meta is not None:
                data = strip_mpi_meta(data)
        self.data = data
        self.meta = meta
        self._runtime = mpi_context

        if self.meta is None and auto_partition:
            partitioned = repartition(
                mpi_context,
                self.data,
                dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            )
            new_meta = get_mpi_meta(partitioned)
            if new_meta is not None:
                partitioned = strip_mpi_meta(partitioned)
            self.data = partitioned
            self.meta = new_meta

        self.data = mark_partitioned(self.data, self.meta)

    def __repr__(self) -> str:
        """Return the distributed wrapper representation."""
        return repr(self.data)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped xarray object."""
        if name == "data":
            raise AttributeError(name)
        if name in _SAFE_PASSTHROUGH_ATTRS:
            return getattr(self.data, name)
        if name in _SAFE_PASSTHROUGH_METHODS:
            return self._safe_method_wrapper(name)
        raise AttributeError(
            f"{name!r} is not an MPI-aware method or an allowlisted "
            + f"passthrough. Use `.data.{name}` for xarray's plain, "
            + f"rank-local version, or add `{name}` to MPIXarray if it "
            + "needs to be distribution-safe."
        )

    def _safe_method_wrapper(self, name: str) -> Callable[..., Any]:
        """Return a bound, ``apply()``-based forward of ``self.data.<name>``."""

        def call(*args: Any, **kwargs: Any) -> Any:
            """Call the wrapped xarray method through MPI-aware apply."""
            return self.apply(
                lambda d, *a, **kw: getattr(d, name)(*a, **kw), self, *args, **kwargs
            )

        call.__name__ = name
        call.__doc__ = (
            f"MPI-aware forward of xarray's `.{name}(...)`, via `apply()`. "
            f"See MPIXarray's `_SAFE_PASSTHROUGH_METHODS`."
        )
        return call

    #
    # Each redirects to `apply()`, so operand handling is exactly `apply()`'s
    # (and `where()`'s) existing, already-tested contract -- no new rules:
    #
    # - MPIXarray + MPIXarray: must share the same partition (same dim,
    #   same start/stop on this rank) or raises, pointing at `.align()`.
    # - MPIXarray + a raw xarray.Dataset/DataArray: fine if it doesn't
    #   carry the partition dimension at all (broadcasts normally); if it
    #   does, it must already be sized -- and, where both sides have an
    #   index, exactly labeled -- to match this rank's own local slice, or
    #   raises pointing at `.align()`. A *replicated* full-size object is
    #   not auto-repartitioned to fit -- that would be a silent guess
    #   about intent; wrap it first (`MPIXarray(full_array, mpi_context)`,
    #   which auto-partitions replicated input) if that is what you want.
    # - MPIXarray + a plain scalar or numpy array: never carries
    #   distribution info, so it is never partition-checked at all --
    #   xarray's ordinary shape-based broadcasting applies, exactly as it
    #   would for `self.data + other` directly. A numpy array meant to
    #   line up with the partition dimension must already be sized to
    #   this rank's own local length, the same requirement as above.
    #
    # Reversed operand order (`other + self`) works reliably when `other`
    # is a plain scalar or numpy array (`5 + ds`, `np.array([1, 2]) * ds`
    # -- verified: `__array_ufunc__ = None` above makes Python's own
    # numeric protocol and numpy's ufunc machinery correctly defer to
    # `__radd__`/etc.). It does NOT work when `other` is a raw
    # `xarray.Dataset`/`DataArray` (`raw_dataset + ds`): xarray's own
    # `Dataset._binary_op`/`DataArray._binary_op` only return
    # `NotImplemented` for `DataTree`/`GroupBy`, not for an unrecognized
    # type generally, and do not consult `__array_ufunc__`/
    # `__array_priority__` at that level -- so it fails inside xarray's
    # own internals rather than reaching `MPIXarray.__radd__` at all. Not
    # fixable from this side without monkeypatching xarray itself. Put
    # the `MPIXarray` operand on the left instead (`ds + raw_dataset`,
    # which does work, verified above).
    #
    # No in-place operators (`__iadd__` etc.): MPIXarray is immutable by
    # construction (`.data`/`.meta` are only ever assigned in `__init__`;
    # see its class docstring), so `x += y` falls back to Python's default
    # `x = x.__add__(y)` and rebinds the name, consistent with every other
    # method here returning a new MPIXarray rather than mutating `self`.
    #
    # `@`/`__matmul__` is the one exception: matrix multiplication can
    # reduce across the partition dimension, so it is not just an
    # elementwise `apply()` call -- it redirects to the dedicated
    # MPI-aware `matmul()` instead.
    #
    # `__len__`/`__iter__` are deliberately not defined. `self.data` (a
    # rank-local slice) already has both, but exposing them directly on
    # MPIXarray would silently return this rank's own local length/items
    # for a distributed object where a caller is far more likely to mean
    # the global ones -- an AttributeError/TypeError on `len(mpixarray)`
    # is safer than a number that is quietly rank-dependent. Use
    # `.meta["global_size"]` (distributed) or `len(.data)` (local, or
    # replicated) explicitly instead.

    def __add__(self, other: Any) -> Any:
        """Elementwise addition (``self + other``); see the note above."""
        return self.apply(_operator.add, self, other)

    def __radd__(self, other: Any) -> Any:
        """Elementwise addition (``other + self``); see the note above."""
        return self.apply(_operator.add, other, self)

    def __sub__(self, other: Any) -> Any:
        """Elementwise subtraction (``self - other``); see the note above."""
        return self.apply(_operator.sub, self, other)

    def __rsub__(self, other: Any) -> Any:
        """Elementwise subtraction (``other - self``); see the note above."""
        return self.apply(_operator.sub, other, self)

    def __mul__(self, other: Any) -> Any:
        """Elementwise multiplication (``self * other``); see the note above."""
        return self.apply(_operator.mul, self, other)

    def __rmul__(self, other: Any) -> Any:
        """Elementwise multiplication (``other * self``); see the note above."""
        return self.apply(_operator.mul, other, self)

    def __truediv__(self, other: Any) -> Any:
        """Elementwise division (``self / other``); see the note above."""
        return self.apply(_operator.truediv, self, other)

    def __rtruediv__(self, other: Any) -> Any:
        """Elementwise division (``other / self``); see the note above."""
        return self.apply(_operator.truediv, other, self)

    def __floordiv__(self, other: Any) -> Any:
        """Elementwise floor division (``self // other``); see the note above."""
        return self.apply(_operator.floordiv, self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        """Elementwise floor division (``other // self``); see the note above."""
        return self.apply(_operator.floordiv, other, self)

    def __mod__(self, other: Any) -> Any:
        """Elementwise modulo (``self % other``); see the note above."""
        return self.apply(_operator.mod, self, other)

    def __rmod__(self, other: Any) -> Any:
        """Elementwise modulo (``other % self``); see the note above."""
        return self.apply(_operator.mod, other, self)

    def __pow__(self, other: Any) -> Any:
        """Elementwise exponentiation (``self ** other``); see the note above."""
        return self.apply(_operator.pow, self, other)

    def __rpow__(self, other: Any) -> Any:
        """Elementwise exponentiation (``other ** self``); see the note above."""
        return self.apply(_operator.pow, other, self)

    def __neg__(self) -> Any:
        """Elementwise negation (``-self``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.neg, self)

    def __pos__(self) -> Any:
        """Elementwise unary plus (``+self``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.pos, self)

    def __abs__(self) -> Any:
        """Elementwise absolute value (``abs(self)``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.abs, self)

    def __and__(self, other: Any) -> Any:
        """Elementwise logical/bitwise AND (``self & other``); see the note above."""
        return self.apply(_operator.and_, self, other)

    def __rand__(self, other: Any) -> Any:
        """Elementwise logical/bitwise AND (``other & self``); see the note above."""
        return self.apply(_operator.and_, other, self)

    def __or__(self, other: Any) -> Any:
        """Elementwise logical/bitwise OR (``self | other``); see the note above."""
        return self.apply(_operator.or_, self, other)

    def __ror__(self, other: Any) -> Any:
        """Elementwise logical/bitwise OR (``other | self``); see the note above."""
        return self.apply(_operator.or_, other, self)

    def __xor__(self, other: Any) -> Any:
        """Elementwise logical/bitwise XOR (``self ^ other``); see the note above."""
        return self.apply(_operator.xor, self, other)

    def __rxor__(self, other: Any) -> Any:
        """Elementwise logical/bitwise XOR (``other ^ self``); see the note above."""
        return self.apply(_operator.xor, other, self)

    def __invert__(self) -> Any:
        """Elementwise logical/bitwise NOT (``~self``); MPI-aware via :meth:`apply`."""
        return self.apply(_operator.invert, self)

    def __lt__(self, other: Any) -> Any:
        """Elementwise ``self < other``; see the note above."""
        return self.apply(_operator.lt, self, other)

    def __le__(self, other: Any) -> Any:
        """Elementwise ``self <= other``; see the note above."""
        return self.apply(_operator.le, self, other)

    def __gt__(self, other: Any) -> Any:
        """Elementwise ``self > other``; see the note above."""
        return self.apply(_operator.gt, self, other)

    def __ge__(self, other: Any) -> Any:
        """Elementwise ``self >= other``; see the note above."""
        return self.apply(_operator.ge, self, other)

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        """Elementwise ``self == other`` -- returns an array, not a bool."""
        return self.apply(_operator.eq, self, other)

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        """Elementwise ``self != other`` -- returns an array, not a bool."""
        return self.apply(_operator.ne, self, other)

    # Elementwise `__eq__` (returning an array, not a bool -- matching
    # xarray's own DataArray/Dataset) makes MPIXarray unhashable, exactly
    # as xarray's own Dataset/DataArray already are. Not overridden back
    # to identity hashing: nothing here needs MPIXarray instances to be
    # usable as dict keys or set members.
    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        """Truth value (``if mpixarray:``, ``bool(mpixarray)``)."""
        if self.meta is not None:
            raise ValueError(
                "the truth value of a distributed MPIXarray is ambiguous: "
                + "different ranks could take different branches and "
                + "deadlock on a later collective"
            )
        return bool(self.data)

    def __getitem__(self, key: Any) -> Any:
        """Select a Dataset variable by name; mirrors ``xarray.Dataset[key]``."""
        if isinstance(key, str):
            return self.apply(lambda d, k: d[k], self, key)
        raise TypeError(
            "only a str key is supported to select a Dataset variable "
            + f"by name; got {key!r} ({type(key).__name__})"
        )

    def __matmul__(self, other: Any) -> MPIXarray:
        """Matrix multiplication (``self @ other``); redirects to :meth:`matmul`."""
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> Any:
        """Matrix multiplication (``other @ self``); MPI-aware like :meth:`matmul`."""
        result = matmul(self._runtime, unwrap(other), self._prepare())
        return finalize(result, self._runtime)

    def _prepare(self) -> xr.Dataset | xr.DataArray:
        """Return ``self.data`` with ``self.meta`` reattached to ``.attrs``."""
        if self.meta is None:
            return self.data
        prepared = self.data.copy(deep=False)
        assign_mpi_meta(prepared, self.meta)
        return prepared

    def to_netcdf(
        self,
        file: str | PathLike[str],
        unlimited_dim: str | Iterable[str] | None = None,
        partition_dim: str | None = None,
        *,
        parallel: bool = False,
        batch_size: int = 24,
        format: str = "NETCDF4",
        shuffle: bool = True,
        zlib: bool = True,
        complevel: int = 4,
        show_progress: bool = True,
        stdout: Any = None,
        chunks: Mapping[str, Iterable[int]] | None = None,
        hints: str | None = None,
        nofill: bool = True,
        allow_serial: bool = False,
    ) -> None:
        """Write this xarray object to NetCDF.

        Parameters
        ----------
        file : str or os.PathLike
            Output path.
        unlimited_dim : str or iterable of str, optional
            Unlimited dimension names.
        partition_dim : str, optional
            Dimension used for MPI partitioning.
        parallel : bool, default False
            Use MPI-parallel NetCDF output.
        batch_size : int, default 24
            Number of slices per serial append.
        format : str, default "NETCDF4"
            NetCDF format used for serial output.
        shuffle : bool, default True
            Enable the HDF5 shuffle filter.
        zlib : bool, default True
            Enable zlib compression.
        complevel : int, default 4
            zlib compression level.
        show_progress : bool, default True
            Show serial write progress.
        stdout : Any, optional
            Stream used for progress output.
        chunks : mapping, optional
            Explicit NetCDF variable chunk shapes.
        hints : str, optional
            Semicolon-separated MPI-IO hints.
        nofill : bool, default True
            Disable NetCDF pre-filling for parallel output.
        allow_serial : bool, default False
            Allow the parallel writer to run with one MPI rank.
        Raises
        ------
        ValueError
            If distributed data are passed to the serial writer.
        """
        from .io import to_netcdf

        if not parallel and self.meta is not None:
            raise ValueError(
                "data is distributed but parallel=False (the default) "
                + "would silently write only this rank's local slice as "
                + "the whole file. Pass parallel=True, or gather/"
                + "replicate to one rank first for serial output."
            )

        prepared = self._prepare()
        if parallel:
            prepared = attach_save_chunks(self._runtime, prepared)
        if self.meta is not None and get_mpi_meta(prepared) is None:
            prepared = prepared.copy(deep=False)
            assign_mpi_meta(prepared, self.meta)

        to_netcdf(
            prepared,
            file,
            self._runtime,
            unlimited_dim,
            partition_dim,
            parallel=parallel,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
            chunks=chunks,
            hints=hints,
            nofill=nofill,
            allow_serial=allow_serial,
        )

    def repartition(
        self,
        dim: Hashable | Literal["auto"] = "auto",
        *,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> MPIXarray:
        """Partition a replicated object across MPI ranks.

        Parameters
        ----------
        dim : Hashable or {"auto"}, optional
            New partition dimension.
        chunk_info : mapping of str to int, optional
            Effective chunk-size hints.
        log_partitions : bool, optional
            Log the resulting rank layout.
        Returns
        -------
        MPIXarray
            Rank-local slice with ``.meta`` set.
        """
        return finalize(
            repartition(
                self._runtime,
                self._prepare(),
                dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            self._runtime,
        )

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        *,
        partition_dim: Hashable | Literal["auto"] | None = None,
        **indexers_kwargs: Any,
    ) -> MPIXarray:
        """Index with global integer coordinates on the partition dimension.

        Parameters
        ----------
        indexers : mapping, optional
            Integer indexers using global coordinates on the partition dimension.
        partition_dim : Hashable or {"auto"} or None, optional
            Dimension to scatter a resulting single-element partition dimension across, when a slice indexer collapses it to length one.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.
        Returns
        -------
        MPIXarray
            Indexed object with ``.meta`` updated.
        """
        return finalize(
            isel(
                self._runtime,
                self._prepare(),
                indexers,
                partition_dim=partition_dim,
                **indexers_kwargs,
            ),
            self._runtime,
        )

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: Any = None,
        drop: bool = False,
        *,
        partition_dim: Hashable | Literal["auto"] | None = None,
        **indexers_kwargs: Any,
    ) -> MPIXarray:
        """Index with global coordinate labels on the partition dimension.

        Parameters
        ----------
        indexers : mapping, optional
            Label indexers using global semantics on the partition dimension.
        method : str, optional
            Inexact matching method passed to xarray.
        tolerance : Any, optional
            Maximum distance for inexact matches.
        drop : bool, optional
            Drop selected coordinate variables.
        partition_dim : Hashable or {"auto"} or None, optional
            Dimension to scatter a resulting single-element partition dimension across, when a label slice collapses it to length one.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.
        Returns
        -------
        MPIXarray
            Indexed object with ``.meta`` updated.
        """
        return finalize(
            sel(
                self._runtime,
                self._prepare(),
                indexers,
                method,
                tolerance,
                drop,
                partition_dim=partition_dim,
                **indexers_kwargs,
            ),
            self._runtime,
        )

    def sum(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Sum over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return finalize(
            sum_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                min_count=min_count,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def prod(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Multiply over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return finalize(
            prod_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                min_count=min_count,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def mean(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the mean over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return finalize(
            mean_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def min(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the minimum over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return finalize(
            min_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def max(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the maximum over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return finalize(
            max_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def any(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Return whether any value is true over the requested dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Logical OR over the requested dimensions, with ``.meta`` updated.
        """
        return finalize(
            any_reduce(
                self._runtime,
                self._prepare(),
                dim,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def all(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Return whether every value is true over the requested dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Logical AND over the requested dimensions, with ``.meta`` updated.
        """
        return finalize(
            all_reduce(
                self._runtime,
                self._prepare(),
                dim,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def first(
        self,
        dim: str,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Select the first valid value along one dimension.

        Parameters
        ----------
        dim : str
            Dimension to pick a position along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Selected object with ``.meta`` updated.
        """
        return finalize(
            first_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def last(
        self,
        dim: str,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Select the last valid value along one dimension.

        Parameters
        ----------
        dim : str
            Dimension to pick a position along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Selected object with ``.meta`` updated.
        """
        return finalize(
            last_reduce(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def var(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the variance over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        ddof : int, optional
            Delta degrees of freedom; the divisor is ``N - ddof``.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return finalize(
            var(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                ddof=ddof,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    def std(
        self,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Compute the standard deviation over one or more dimensions.

        Parameters
        ----------
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        ddof : int, optional
            Delta degrees of freedom; the divisor is ``N - ddof``.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)partition the result once the active partition dimension is reduced away.
        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return finalize(
            std(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                ddof=ddof,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._runtime,
        )

    # -- Groupby (xarray-styled entry points; groupby_reduce/resample_reduce
    #    are internal engine dispatch names, not part of this public surface) -

    def groupby(self, dim: Hashable, labels: xr.DataArray | np.ndarray) -> MPIGroupBy:
        """Group by ``labels`` along ``dim``, mirroring ``xarray.Dataset.groupby``.

        Parameters
        ----------
        dim : Hashable
            Dimension being grouped and reduced.
        labels : array-like
            Group key for every position along this rank's local ``dim`` axis.
        Returns
        -------
        MPIGroupBy
            Chainable handle; call ``.sum()``, ``.mean()``, ``.count()``,
            ``.min()``, or ``.max()`` on it to get the reduced :class:`MPIXarray`.
        """
        return MPIGroupBy(self, dim, labels)

    def resample(self, dim: Hashable, freq: str) -> MPIResample:
        """Resample a datetime dimension using xarray semantics.

        Parameters
        ----------
        dim : Hashable
            Datetime dimension to resample.
        freq : str
            Pandas offset alias (e.g.
        Returns
        -------
        MPIResample
            Chainable handle; call ``.sum()``, ``.mean()``, ``.count()``,
            ``.min()``, or ``.max()`` on it to get the reduced :class:`MPIXarray`.
        """
        return MPIResample(self, dim, freq)

    def align(
        self,
        other: MPIXarray | xr.Dataset | xr.DataArray,
        dim: Hashable | Literal["auto"] | None = None,
        *,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> tuple[MPIXarray, MPIXarray]:
        """Return ``(self, other)`` partitioned identically across ranks.

        Parameters
        ----------
        other : MPIXarray, xarray.Dataset, or xarray.DataArray
            Operand to align against ``self``.
        dim : Hashable or {"auto"} or None, optional
            Dimension to partition both operands along when neither is yet distributed.
        chunk_info : mapping, optional
            Forwarded to ``repartition`` when neither operand is yet distributed.
        log_partitions : bool, optional
            Forwarded to ``repartition`` when neither operand is yet distributed.
        Returns
        -------
        tuple of MPIXarray
            ``(left, right)``, each with matching distribution metadata.
        """
        left, right = align(
            self._runtime,
            self._prepare(),
            unwrap(other),
            dim,
            chunk_info=chunk_info,
            log_partitions=log_partitions,
        )
        return finalize(left, self._runtime), finalize(right, self._runtime)

    def reindex(
        self,
        indexers: Mapping[Hashable, Any] | None = None,
        *,
        method: str | None = None,
        tolerance: float | Iterable[float] | None = None,
        fill_value: Any = _FILL_VALUE_UNSET,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
        **indexers_kwargs: Any,
    ) -> MPIXarray:
        """Reindex onto new coordinate labels, redistributing if needed.

        Parameters
        ----------
        indexers : mapping, optional
            New coordinate labels per dimension.
        method : str, optional
            Forwarded to xarray's ``reindex``.
        tolerance : float or iterable of float, optional
            Forwarded to xarray's ``reindex``.
        fill_value : Any, optional
            Value used for labels with no match.
        chunk_info : mapping, optional
            Forwarded to ``repartition`` when a partition dimension is reindexed.
        log_partitions : bool, optional
            Forwarded to ``repartition`` when a partition dimension is reindexed.
        **indexers_kwargs : Any
            Additional indexers given as keywords.
        Returns
        -------
        MPIXarray
            The reindexed object.
        """
        kwargs: dict[str, Any] = {
            "method": method,
            "tolerance": tolerance,
            "chunk_info": chunk_info,
            "log_partitions": log_partitions,
        }
        if fill_value is not _FILL_VALUE_UNSET:
            kwargs["fill_value"] = fill_value
        return finalize(
            reindex(
                self._runtime, self._prepare(), indexers, **kwargs, **indexers_kwargs
            ),
            self._runtime,
        )

    def sortby(
        self,
        by: Hashable | Any | Iterable[Hashable | Any],
        *,
        ascending: bool = True,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> MPIXarray:
        """Sort by one or more keys, redistributing if needed.

        Parameters
        ----------
        by : Hashable, DataArray, or sequence of these
            Sort key(s): variable/coordinate name(s) or explicit DataArray(s).
        ascending : bool, optional
            Sort order.
        chunk_info : mapping, optional
            Forwarded to ``repartition`` when a partition dimension is sorted.
        log_partitions : bool, optional
            Forwarded to ``repartition`` when a partition dimension is sorted.
        Returns
        -------
        MPIXarray
            The sorted object.
        """
        return finalize(
            sortby(
                self._runtime,
                self._prepare(),
                by,
                ascending=ascending,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            self._runtime,
        )

    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

        Parameters
        ----------
        func : callable
            Partition-preserving, rank-local function of ``args``/``kwargs``.
        *args : Any
            Positional arguments to ``func``.
        **kwargs : Any
            Keyword arguments to ``func``.
        Returns
        -------
        MPIXarray or Any
            ``func``'s result, wrapped if it is an xarray Dataset/DataArray, otherwise returned as-is.
        """
        unwrapped_args = tuple(unwrap(arg) for arg in args)
        unwrapped_kwargs = {name: unwrap(value) for name, value in kwargs.items()}
        result = apply(self._runtime, func, *unwrapped_args, **unwrapped_kwargs)
        return finalize(result, self._runtime)

    def matmul(self, right: MPIXarray | Any) -> MPIXarray:
        """Matrix multiplication (``self @ right``), correct under MPI.

        Parameters
        ----------
        right : MPIXarray or Any
            Right operand: an xarray DataArray (distributed or not, wrapped or not) or a plain array/scalar.
        Returns
        -------
        MPIXarray
            The matrix product.
        """
        result = matmul(self._runtime, self._prepare(), unwrap(right))
        return finalize(result, self._runtime)

    def _halo_exchange(
        self, dim: Hashable | None = None, *, before: int, after: int
    ) -> tuple[MPIXarray, int, int]:
        """Pad with boundary slices fetched from the adjacent ranks."""
        padded, left_pad, right_pad = halo_exchange(
            self._runtime, self._prepare(), dim, before=before, after=after
        )
        return finalize(padded, self._runtime), left_pad, right_pad

    def rolling_reduce(
        self,
        dim: Hashable,
        window: int,
        reduce: str = "mean",
        *,
        center: bool = True,
        min_periods: int | None = None,
    ) -> MPIXarray:
        """Windowed reduction along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to roll over.
        window : int
            Window size, as in ``xarray.DataArray.rolling``.
        reduce : str, optional
            Name of the reduction to call on the rolling object (e.g.
        center : bool, optional
            As in ``xarray.DataArray.rolling``.
        min_periods : int or None, optional
            As in ``xarray.DataArray.rolling``.
        Returns
        -------
        MPIXarray
            Rolled-and-reduced object with ``.meta`` preserved.
        """
        result = rolling_reduce(
            self._runtime,
            self._prepare(),
            dim,
            window,
            reduce,
            center=center,
            min_periods=min_periods,
        )
        return finalize(result, self._runtime)

    def coarsen_reduce(
        self,
        dim: Hashable,
        window: int,
        reduce: str = "mean",
        *,
        boundary: str = "exact",
        side: str = "left",
        coord_func: str = "mean",
    ) -> MPIXarray:
        """Block reduction along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to coarsen along.
        window : int
            Block size, as in ``xarray.DataArray.coarsen``.
        reduce : str, optional
            Name of the reduction to call on the coarsen object (e.g.
        boundary : {"exact", "trim", "pad"}, optional
            As in ``xarray.DataArray.coarsen``.
        side : {"left"}, optional
            Only "left" (the ``xarray`` default) is supported for a distributed dimension so far.
        coord_func : str, optional
            As in ``xarray.DataArray.coarsen``.
        Returns
        -------
        MPIXarray
            Coarsened-and-reduced object with ``.meta`` updated to match the new, block-reduced length along ``dim``.
        """
        result = coarsen_reduce(
            self._runtime,
            self._prepare(),
            dim,
            window,
            reduce,
            boundary=boundary,
            side=side,
            coord_func=coord_func,
        )
        return finalize(result, self._runtime)

    def rolling(
        self,
        dim: Hashable,
        window: int,
        *,
        center: bool = True,
        min_periods: int | None = None,
    ) -> MPIRolling:
        """Windowed rolling handle, mirroring ``xarray.DataArray.rolling``.

        Parameters
        ----------
        dim : Hashable
            Dimension to roll over.
        window : int
            Window size, as in ``xarray.DataArray.rolling``.
        center : bool, optional
            As in ``xarray.DataArray.rolling``.
        min_periods : int or None, optional
            As in ``xarray.DataArray.rolling``.
        Returns
        -------
        MPIRolling
            Chainable handle; call ``.mean()``, ``.sum()``, ``.min()``, ``.max()``, ``.std()``, or ``.count()`` on it to get the rolled-and-reduced :class:`MPIXarray`.
        """
        return MPIRolling(self, dim, window, center=center, min_periods=min_periods)

    def evaluate(self, expression: str, /, **variables: Any) -> Any:
        """Evaluate a string expression, respecting normal operator precedence.

        Parameters
        ----------
        expression : str
            A Python expression referencing ``variables`` by name, e.g.
        **variables : Any
            Values bound to the names used in ``expression``.
        Returns
        -------
        MPIXarray or Any
            The expression's value, wrapped if it is an xarray Dataset/DataArray, otherwise returned as-is.
        """
        unwrapped = {name: unwrap(value) for name, value in variables.items()}
        result = evaluate(self._runtime, expression, **unwrapped)
        return finalize(result, self._runtime)

    def where(
        self,
        cond: MPIXarray | xr.Dataset | xr.DataArray | Any,
        other: MPIXarray | xr.Dataset | xr.DataArray | Any = _WHERE_UNSET,
        *,
        drop: bool = False,
    ) -> MPIXarray:
        """Elementwise selection, mirroring ``xarray.DataArray.where``.

        Parameters
        ----------
        cond : MPIXarray, xarray.Dataset, xarray.DataArray, or Any
            Boolean condition, following ``xarray.DataArray.where``.
        other : MPIXarray, xarray.Dataset, xarray.DataArray, or Any, optional
            Fill value where ``cond`` is False.
        drop : bool, optional
            Must be False when ``self`` is distributed; ``drop=True`` can remove a different number of positions on different ranks.
        Returns
        -------
        MPIXarray
            The selected object, with ``.meta`` unchanged.
        """
        args = (cond,) if other is _WHERE_UNSET else (cond, other)
        return finalize(
            where(
                self._runtime,
                self._prepare(),
                *(unwrap(arg) for arg in args),
                drop=drop,
            ),
            self._runtime,
        )

    def cumsum(
        self,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Cumulative sum along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to accumulate along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        Returns
        -------
        MPIXarray
            Cumulative sum with ``.meta`` unchanged.
        """
        return finalize(
            cumsum(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
            ),
            self._runtime,
        )

    def median(
        self,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Median over ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        Returns
        -------
        MPIXarray
            Reduced object.
        """
        return finalize(
            median(
                self._runtime,
                self._prepare(),
                dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
            ),
            self._runtime,
        )

    def diff(
        self,
        dim: Hashable,
        n: int = 1,
        *,
        label: Literal["upper", "lower"] = "upper",
    ) -> MPIXarray:
        """``n``-th order difference along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to difference along.
        n : int, optional
            Order of the difference.
        label : {"upper", "lower"}, optional
            As in ``xarray.DataArray.diff``.
        Returns
        -------
        MPIXarray
            The differenced object, ``n`` elements shorter along ``dim`` globally, with ``.meta`` updated to match.
        """
        return finalize(
            diff(self._runtime, self._prepare(), dim, n, label=label), self._runtime
        )

    def shift(
        self,
        dim: Hashable,
        periods: int = 1,
        *,
        fill_value: Any = _FILL_VALUE_UNSET,
    ) -> MPIXarray:
        """Shift by ``periods`` along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to shift along.
        periods : int, optional
            Number of positions to shift by.
        fill_value : Any, optional
            As in ``xarray.DataArray.shift``; defaults to xarray's own dtype-aware NA fill when omitted.
        Returns
        -------
        MPIXarray
            The shifted object, same shape and distribution as ``self``.
        """
        kwargs: dict[str, Any] = {}
        if fill_value is not _FILL_VALUE_UNSET:
            kwargs["fill_value"] = fill_value
        return finalize(
            shift(self._runtime, self._prepare(), dim, periods, **kwargs), self._runtime
        )

    def roll(self, dim: Hashable, shift_by: int) -> MPIXarray:
        """Circularly shift by ``shift_by`` along ``dim``, wrapping at the edge.

        Parameters
        ----------
        dim : Hashable
            Dimension to roll along.
        shift_by : int
            Number of positions to roll by; positive rolls toward higher indices, matching ``xarray.DataArray.roll``.
        Returns
        -------
        MPIXarray
            The rolled object, same shape and distribution as ``self``.
        """
        return finalize(
            roll(self._runtime, self._prepare(), dim, shift_by), self._runtime
        )

    def ffill(self, dim: Hashable, limit: int | None = None) -> MPIXarray:
        """Forward-fill along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to fill along.
        limit : int or None, optional
            As in ``xarray.DataArray.ffill``.
        Returns
        -------
        MPIXarray
            The forward-filled object, same shape and distribution as ``self``.
        """
        return finalize(
            ffill(self._runtime, self._prepare(), dim, limit), self._runtime
        )

    def bfill(self, dim: Hashable, limit: int | None = None) -> MPIXarray:
        """Backward-fill along ``dim``, correct when ``dim`` is distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to fill along.
        limit : int or None, optional
            As in ``xarray.DataArray.bfill``.
        Returns
        -------
        MPIXarray
            The backward-filled object, same shape and distribution as ``self``.
        """
        return finalize(
            bfill(self._runtime, self._prepare(), dim, limit), self._runtime
        )

    def interp(
        self, dim: Hashable, new_coord: Any, method: str = "linear", **kwargs: Any
    ) -> MPIXarray:
        """Interpolate onto ``new_coord`` along ``dim``, correct when distributed.

        Parameters
        ----------
        dim : Hashable
            Dimension to interpolate along.
        new_coord : array-like
            This rank's own local slice of the new target coordinate.
        method : str, optional
            As in ``xarray.DataArray.interp``.
        **kwargs : Any
            Forwarded to ``xarray.DataArray.interp``.
        Returns
        -------
        MPIXarray
            Interpolated result, with ``.meta`` recomputed for the new length along ``dim``.
        """
        return finalize(
            interp(self._runtime, self._prepare(), dim, new_coord, method, **kwargs),
            self._runtime,
        )

    def differentiate(
        self,
        coord: Hashable,
        edge_order: Literal[1, 2] = 1,
        datetime_unit: Any = None,
    ) -> MPIXarray:
        """Differentiate along ``coord``, correct when ``coord`` is distributed.

        Parameters
        ----------
        coord : Hashable
            Coordinate to differentiate along.
        edge_order : {1, 2}, optional
            As in ``xarray.DataArray.differentiate``.
        datetime_unit : Any, optional
            As in ``xarray.DataArray.differentiate``.
        Returns
        -------
        MPIXarray
            The derivative, same shape and distribution as ``self``.
        """
        return finalize(
            differentiate(
                self._runtime,
                self._prepare(),
                coord,
                edge_order=edge_order,
                datetime_unit=datetime_unit,
            ),
            self._runtime,
        )


def mark_partitioned(
    data: xr.Dataset | xr.DataArray, meta: dict[str, Any] | None
) -> xr.Dataset | xr.DataArray:
    """Set the lightweight partition marker from MPI metadata.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to copy and mark.
    meta : dict or None
        Distribution metadata, or None for replicated data.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Shallow copy whose partition marker matches ``meta``.
    """
    marked = data.copy(deep=False)
    if meta is None:
        marked.attrs.pop(_PARTITIONED_ATTR, None)
        if isinstance(marked, xr.Dataset):
            for variable in marked.variables.values():
                variable.attrs.pop(_PARTITIONED_ATTR, None)
        return marked

    dims = meta["dims"]
    marked.attrs[_PARTITIONED_ATTR] = True
    if isinstance(marked, xr.Dataset):
        for variable in marked.variables.values():
            variable.attrs.pop(_PARTITIONED_ATTR, None)
            if any(dim in variable.dims for dim in dims):
                variable.attrs[_PARTITIONED_ATTR] = True
    return marked


def finalize(result: Any, mpi_context: MPIContext) -> Any:
    """Wrap an engine result when it is an xarray object.

    Parameters
    ----------
    result : Any
        Result returned by ``_MPIXarrayOps``.
    mpi_context : MPIContext
        Runtime bound to the wrapped result.
    Returns
    -------
    MPIXarray or Any
        Wrapped xarray result, or the original non-xarray value.
    """
    if isinstance(result, (xr.Dataset, xr.DataArray)):
        return MPIXarray(result, mpi_context, auto_partition=False)
    return result


def unwrap(value: Any) -> Any:
    """Prepare an ``MPIXarray`` operand for engine use.

    Parameters
    ----------
    value : Any
        Candidate operand.
    Returns
    -------
    Any
        Prepared xarray data for ``MPIXarray`` input, otherwise ``value``.
    """
    return value._prepare() if isinstance(value, MPIXarray) else value


@wraps(dataarray_binary_op)
def _da_binary_op(
    self: xr.DataArray, other: Any, f: Any, reflexive: bool = False
) -> Any:
    """Defer mixed binary operations to ``MPIXarray``."""
    if isinstance(other, MPIXarray):
        return NotImplemented
    return dataarray_binary_op(self, other, f, reflexive)


@wraps(dataset_binary_op)
def _ds_binary_op(
    self: xr.Dataset,
    other: Any,
    f: Any,
    reflexive: bool = False,
    join: Any = None,
) -> Any:
    """Defer mixed binary operations to ``MPIXarray``."""
    if isinstance(other, MPIXarray):
        return NotImplemented
    return dataset_binary_op(self, other, f, reflexive, join)


xr.DataArray._binary_op = _da_binary_op
xr.Dataset._binary_op = _ds_binary_op
