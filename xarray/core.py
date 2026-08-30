"""Provide the MPI-distributed xarray wrapper.

``MPIXarray`` stores rank-local xarray data and delegates distributed
operations to :class:`~.ops._MPIXarrayOps`.
"""

from __future__ import annotations

import operator as _operator
from functools import wraps
from typing import TYPE_CHECKING, Any

import xarray as xr

from .handles import MPIGroupBy, MPIResample, MPIRolling
from .meta import _assign_meta, get_mpi_meta, strip_mpi_meta
from .ops import MPIXarrayOps

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping
    from os import PathLike
    from types import EllipsisType
    from typing import Literal

    import numpy as np

    from ..mpi.runtime import MPIRuntime


#: Attrs key for the lightweight boolean flag :func:`mark_partitioned`
#: stamps onto ``MPIXarray.data`` (and, for a Dataset, every distributed
#: variable) after the full ``mpi_meta`` dict is popped into ``.meta``. Not
#: read by anything in this package -- ``.meta`` (or, inside the engine,
#: the ``mpi_meta`` dict reattached transiently by :meth:`MPIXarray._prepare`)
#: is always the source of truth for distribution state. This exists only
#: so code that sees ``.data`` on its own (e.g. after it is handed to a
#: plain xarray function, or written to disk) has a cheap, human-inspectable
#: hint that it was part of an MPI partition, without exposing the full
#: metadata dict.
_PARTITIONED_ATTR = "mpi_partitioned"

#: Sentinel distinguishing "no fill value given" from a genuine ``other=None``
#: in :meth:`MPIXarray.where`.
_WHERE_UNSET = object()

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
    """Wrap rank-local xarray data with MPI distribution state.

    Parameters
    ----------
    data : MPIXarray or xarray.Dataset or xarray.DataArray
        Rank-local data. Existing ``MPIXarray`` instances are adopted unchanged.
    runtime : MPIRuntime
        Runtime used by distributed operations.
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

    #: Tells NumPy (and anything that follows its convention, including
    #: xarray's own binary-op dispatch) not to try to handle a mixed
    #: operation itself when the *other* operand is an ``MPIXarray`` --
    #: defer to this class's own ``__radd__``/etc. instead. Without this,
    #: ``xr.DataArray(...) + mpixarray_instance`` could be handled by
    #: xarray's ``__add__`` treating the ``MPIXarray`` as some generic
    #: array-like (undefined, unpredictable) instead of reaching
    #: :meth:`__radd__` at all.
    __array_ufunc__ = None

    def __init__(
        self,
        data: MPIXarray | xr.Dataset | xr.DataArray,
        runtime: MPIRuntime,
        meta: dict[str, Any] | None = None,
        *,
        auto_partition: bool = True,
        dim: Hashable | Literal["auto"] = "auto",
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> None:
        if isinstance(data, MPIXarray):
            self.data = data.data
            self.meta = data.meta
            self._runtime = data._runtime
            self._ops = data._ops
            return

        if meta is None:
            meta = get_mpi_meta(data)
            if meta is not None:
                data = strip_mpi_meta(data)
        self.data = data
        self.meta = meta
        self._runtime = runtime
        self._ops = MPIXarrayOps(runtime)

        if self.meta is None and auto_partition:
            partitioned = self._ops.repartition(
                self.data, dim, chunk_info=chunk_info, log_partitions=log_partitions
            )
            new_meta = get_mpi_meta(partitioned)
            if new_meta is not None:
                partitioned = strip_mpi_meta(partitioned)
            self.data = partitioned
            self.meta = new_meta

        self.data = mark_partitioned(self.data, self.meta)

    def __repr__(self) -> str:
        return repr(self.data)

    def __getattr__(self, name: str) -> Any:
        if name == "data":
            raise AttributeError(name)
        if name in _SAFE_PASSTHROUGH_ATTRS:
            return getattr(self.data, name)
        if name in _SAFE_PASSTHROUGH_METHODS:
            return self._safe_method_wrapper(name)
        raise AttributeError(
            f"{name!r} is not an MPI-aware MPIXarray method, a recognized "
            + "passthrough property, or an allowlisted safe method. Use "
            + f"`.data.{name}` (or `.data.{name}(...)`) for xarray's plain, "
            + f"rank-local implementation, or add an MPI-aware `{name}` to "
            + "MPIXarray if it needs to be distribution-safe."
        )

    def _safe_method_wrapper(self, name: str) -> Callable[..., Any]:
        """Return a bound, ``apply()``-based forward of ``self.data.<name>``.

        Backs the :data:`_SAFE_PASSTHROUGH_METHODS` branch of
        :meth:`__getattr__`: calling the returned callable runs
        ``self.data.<name>(*args, **kwargs)`` rank-locally through
        :meth:`apply`, which reattaches/pops ``.meta`` exactly like every
        other method here and re-validates that the partition dimension
        (if this object is distributed) survives the call unchanged.

        Parameters
        ----------
        name : str
            A name already confirmed to be in :data:`_SAFE_PASSTHROUGH_METHODS`.

        Returns
        -------
        callable
            ``lambda *args, **kwargs: self.apply(...)`` bound to ``name``.
        """

        def call(*args: Any, **kwargs: Any) -> Any:
            return self.apply(
                lambda d, *a, **kw: getattr(d, name)(*a, **kw), self, *args, **kwargs
            )

        call.__name__ = name
        call.__doc__ = (
            f"MPI-aware forward of xarray's `.{name}(...)`, via `apply()`. "
            f"See MPIXarray's `_SAFE_PASSTHROUGH_METHODS`."
        )
        return call

    # -- Arithmetic dunder methods ------------------------------------------
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
    #   about intent; wrap it first (`MPIXarray(full_array, runtime)`,
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
        """Elementwise ``self == other`` -- returns an array, not a bool.

        Matches ``xarray.DataArray``/``Dataset``'s own ``__eq__``, and
        carries the same consequence: this makes :class:`MPIXarray`
        unhashable (see ``__hash__`` below), exactly as xarray's own
        Dataset/DataArray already are.
        """
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
        """Truth value (``if mpixarray:``, ``bool(mpixarray)``).

        Raises for a distributed object rather than evaluating this
        rank's own local slice: different ranks could hold different
        local data and therefore different local truthiness, so branching
        on it directly is a way to silently desynchronize ranks -- one
        rank takes the ``if`` branch and later reaches a collective call
        another rank, having taken the ``else`` branch, never posts,
        deadlocking. Reduce to a replicated scalar first (e.g.
        ``.all()``/``.any()``), or use ``.data`` directly if this rank's
        own local truthiness is genuinely what is wanted.

        For a replicated (non-distributed) object this delegates to
        ``bool(self.data)``, which itself still raises for a
        multi-element array (xarray/numpy's own "truth value is
        ambiguous" behavior) -- unchanged, not something this class
        needs to handle specially.

        Returns
        -------
        bool
            The truth value of a replicated, single-element result.

        Raises
        ------
        ValueError
            If this object is distributed, or (via ``self.data``) if it
            has more than one element.
        """
        if self.meta is not None:
            raise ValueError(
                "The truth value of a distributed MPIXarray is ambiguous "
                + "across ranks and unsafe to branch on directly -- "
                + "different ranks could take different branches and later "
                + "deadlock on a collective call one of them never posts. "
                + "Reduce it to a replicated scalar first (e.g. "
                + ".all()/.any()), or use .data directly if this rank's own "
                + "local truthiness is genuinely what you want."
            )
        return bool(self.data)

    def __getitem__(self, key: Any) -> Any:
        """Select a Dataset variable by name; mirrors ``xarray.Dataset[key]``.

        Only a string key is supported, which selects a data variable and
        is always partition-preserving -- it never touches dimension
        lengths or ordering, so (like the entries in
        :data:`_SAFE_PASSTHROUGH_METHODS`) it is safe to route through
        :meth:`apply`. Any other key (integer/slice/boolean positional
        indexing, as ``xarray.DataArray.__getitem__`` supports) has no
        dedicated MPI-aware handling here -- it can touch the partition
        dimension the same way a raw, unvalidated ``isel`` could -- so it
        raises rather than silently doing something dimension-unsafe. Use
        :meth:`isel`/:meth:`sel` for that, or ``.data[key]`` directly for
        xarray's plain, rank-local indexing.

        Parameters
        ----------
        key : Any
            Variable name (str) to select.

        Returns
        -------
        MPIXarray
            The selected variable, with ``.meta`` unchanged.

        Raises
        ------
        TypeError
            If ``key`` is not a string.
        """
        if isinstance(key, str):
            return self.apply(lambda d, k: d[k], self, key)
        raise TypeError(
            "MPIXarray.__getitem__ only supports selecting a Dataset "
            + f"variable by name (a str key); got {key!r} "
            + f"({type(key).__name__}). Use .isel(...)/.sel(...) for "
            + "label/position-based indexing, or .data[key] for xarray's "
            + "plain, rank-local indexing."
        )

    def __matmul__(self, other: Any) -> MPIXarray:
        """Matrix multiplication (``self @ other``); redirects to :meth:`matmul`.

        Unlike the other dunders above, this does not go through
        :meth:`apply`: contracting the partition dimension needs an MPI
        reduction, not a plain elementwise call, which is exactly what
        :meth:`matmul` (not a generic ``apply``) provides.
        """
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> Any:
        """Matrix multiplication (``other @ self``); MPI-aware like :meth:`matmul`."""
        result = self._ops.matmul(unwrap(other), self._prepare())
        return finalize(result, self._runtime)

    def _prepare(self) -> xr.Dataset | xr.DataArray:
        """Return ``self.data`` with ``self.meta`` reattached to ``.attrs``.

        The underlying engine (:class:`~.ops._MPIXarrayOps`) reads
        distribution metadata from ``value.attrs`` (see
        :func:`~.meta.get_mpi_meta`), but ``self.data`` deliberately never
        carries it (see class docstring). This reattaches ``self.meta`` to a
        shallow copy -- ``self.data`` itself is left untouched -- for the
        duration of a single engine call; the copy is discarded once that
        call's result is re-wrapped.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            ``self.data``, unchanged if ``self.meta`` is None, otherwise a
            shallow copy carrying ``self.meta`` in ``.attrs["mpi_meta"]``.
        """
        if self.meta is None:
            return self.data
        prepared = self.data.copy(deep=False)
        _assign_meta(prepared, self.meta)
        return prepared

    def _dispatch(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call ``_MPIXarrayOps.<name>`` on ``self.data`` and wrap the result.

        Parameters
        ----------
        name : str
            Method name on :class:`~.ops._MPIXarrayOps` to invoke.
        *args : Any
            Positional arguments forwarded to that method.
        **kwargs : Any
            Keyword arguments forwarded to that method.

        Returns
        -------
        MPIXarray or Any
            The method's result, wrapped by :func:`finalize`.
        """
        method = getattr(self._ops, name)
        return finalize(method(self._prepare(), *args, **kwargs), self._runtime)

    # -- IO --------------------------------------------------------------------

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
        from ..netcdf import io as netcdf_io

        if not parallel and self.meta is not None:
            raise ValueError(
                "MPIXarray.to_netcdf(): data is distributed but parallel=False "
                + "(the default). Serial NetCDF output is not rank-aware and "
                + "expects the complete object already assembled on the calling "
                + "rank; writing a distributed object this way would silently "
                + "write only this rank's local slice as the whole file. Pass "
                + "parallel=True to write the distributed object correctly, or "
                + "gather/replicate it to a single rank first if serial output "
                + "is what you want."
            )

        prepared = self._prepare()
        if parallel:
            prepared = self._ops.attach_save_chunks(prepared)
        if self.meta is not None and get_mpi_meta(prepared) is None:
            prepared = prepared.copy(deep=False)
            _assign_meta(prepared, self.meta)

        netcdf_io.to_netcdf(
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

    # -- IO: (re)distribution of an existing object --------------------------

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
            New partition dimension. "auto" selects the largest dimension.
        chunk_info : mapping of str to int, optional
            Effective chunk-size hints.
        log_partitions : bool, optional
            Log the resulting rank layout.

        Returns
        -------
        MPIXarray
            Rank-local slice with ``.meta`` set.
        """
        return self._dispatch(
            "repartition", dim, chunk_info=chunk_info, log_partitions=log_partitions
        )

    # -- Indexing --------------------------------------------------------------

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        *,
        partition_dim: Hashable | Literal["auto"] | None = None,
        **indexers_kwargs: Any,
    ) -> MPIXarray:
        """Index with global integer coordinates on the partition dimension.

        A scalar integer indexer on the partition dimension (e.g.
        ``isel(time=5)``) is detected automatically and handled the same
        way a dedicated ``isel_scalar`` call would: the dimension is
        dropped entirely, the one rank that owns that global index sends
        its value, and it is broadcast to every rank -- a replicated
        result (``.meta`` becomes None). There is no separate method for
        this; it falls out of the indexer's type.

        Parameters
        ----------
        indexers : mapping, optional
            Integer indexers using global coordinates on the partition dimension.
        partition_dim : Hashable or {"auto"} or None, optional
            Dimension to scatter a resulting single-element partition dimension
            across, when a slice indexer collapses it to length one. Not
            consulted for a scalar indexer (see above), which always replicates.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        MPIXarray
            Indexed object with ``.meta`` updated. Replicated (``.meta`` is
            None) if the partition dimension was indexed with a scalar.
        """
        return self._dispatch(
            "isel", indexers, partition_dim=partition_dim, **indexers_kwargs
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

        A scalar label indexer on the partition dimension is detected
        automatically and handled the same way a dedicated ``sel_scalar``
        call would -- see :meth:`isel`'s equivalent note; the same applies
        here with labels in place of integer positions.

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
            Dimension to scatter a resulting single-element partition dimension
            across, when a label slice collapses it to length one. Not
            consulted for a scalar label (see above), which always replicates.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        MPIXarray
            Indexed object with ``.meta`` updated. A scalar selection on the
            partition dimension is replicated on every rank.
        """
        return self._dispatch(
            "sel",
            indexers,
            method,
            tolerance,
            drop,
            partition_dim=partition_dim,
            **indexers_kwargs,
        )

    # -- Reductions --------------------------------------------------------

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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return self._dispatch(
            "sum",
            dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return self._dispatch(
            "prod",
            dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return self._dispatch(
            "mean",
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return self._dispatch(
            "min",
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return self._dispatch(
            "max",
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Logical OR over the requested dimensions, with ``.meta`` updated.
        """
        return self._dispatch(
            "any", dim, keep_attrs=keep_attrs, partition_dim=partition_dim
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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Logical AND over the requested dimensions, with ``.meta`` updated.
        """
        return self._dispatch(
            "all", dim, keep_attrs=keep_attrs, partition_dim=partition_dim
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
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Selected object with ``.meta`` updated.
        """
        return self._dispatch(
            "first",
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Selected object with ``.meta`` updated.
        """
        return self._dispatch(
            "last",
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
        )

    # -- Statistics ----------------------------------------------------------

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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        ddof : int, optional
            Delta degrees of freedom; the divisor is ``N - ddof``.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return self._dispatch(
            "var",
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
            Dimensions to reduce. None or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        ddof : int, optional
            Delta degrees of freedom; the divisor is ``N - ddof``.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        partition_dim : Hashable or {"auto"} or None, optional
            Where to (re)distribute the result once the active partition
            dimension is reduced away. The default, "auto", picks the
            largest surviving dimension and redistributes onto it --
            the result stays distributed, not replicated, even though
            the original partition dimension is gone. Pass None
            instead for a plain replicated result (every rank holds
            the same value, ``.meta`` becomes None), or a specific
            dimension name to choose it.

        Returns
        -------
        MPIXarray
            Reduced object with ``.meta`` updated.
        """
        return self._dispatch(
            "std",
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
        )

    # -- Groupby (xarray-styled entry points; groupby_reduce/resample_reduce
    #    are internal engine dispatch names, not part of this public surface) -

    def groupby(self, dim: Hashable, labels: xr.DataArray | np.ndarray) -> MPIGroupBy:
        """Group by ``labels`` along ``dim``, mirroring ``xarray.Dataset.groupby``.

        Unlike plain ``xarray`` groupby, this does not return an iterable
        of ``(label, subset)`` pairs -- only a reduction over each group is
        supported. Call a reduction method on the returned handle, e.g.
        ``ds.groupby("time", year).mean()``.

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
            ``.min()``, or ``.max()`` on it to get the reduced
            :class:`MPIXarray`.
        """
        return MPIGroupBy(self, dim, labels)

    def resample(self, dim: Hashable, freq: str) -> MPIResample:
        """Resample a datetime dimension using xarray semantics.

        Parameters
        ----------
        dim : Hashable
            Datetime dimension to resample.
        freq : str
            Pandas offset alias (e.g. "D", "MS", "YS").

        Returns
        -------
        MPIResample
            Chainable handle; call ``.sum()``, ``.mean()``, ``.count()``,
            ``.min()``, or ``.max()`` on it to get the reduced
            :class:`MPIXarray`.
        """
        return MPIResample(self, dim, freq)

    # -- Arithmetic ------------------------------------------------------------

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
            Dimension to distribute both operands along when neither is yet
            distributed. Required in that case; ignored otherwise.
        chunk_info : mapping, optional
            Forwarded to ``repartition`` when neither operand is yet distributed.
        log_partitions : bool, optional
            Forwarded to ``repartition`` when neither operand is yet distributed.

        Returns
        -------
        tuple of MPIXarray
            ``(left, right)``, each with matching distribution metadata.
        """
        left, right = self._ops.align(
            self._prepare(),
            unwrap(other),
            dim,
            chunk_info=chunk_info,
            log_partitions=log_partitions,
        )
        return finalize(left, self._runtime), finalize(right, self._runtime)

    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

        Any :class:`MPIXarray` found in ``args``/``kwargs`` is unwrapped to
        its underlying data (with ``.meta`` reattached) before the call.
        ``func`` must be partition-preserving; see
        :meth:`.arithmetic.Arithmetic.apply` for the exact contract.

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
            ``func``'s result, wrapped if it is an xarray Dataset/DataArray,
            otherwise returned as-is.
        """
        unwrapped_args = tuple(unwrap(arg) for arg in args)
        unwrapped_kwargs = {name: unwrap(value) for name, value in kwargs.items()}
        result = self._ops.apply(func, *unwrapped_args, **unwrapped_kwargs)
        return finalize(result, self._runtime)

    def matmul(self, right: MPIXarray | Any) -> MPIXarray:
        """Matrix multiplication (``self @ right``), correct under MPI.

        Parameters
        ----------
        right : MPIXarray or Any
            Right operand: an xarray DataArray (distributed or not, wrapped
            or not) or a plain array/scalar.

        Returns
        -------
        MPIXarray
            The matrix product. Replicated (``.meta`` is None) if the
            distributed dimension was contracted away.
        """
        result = self._ops.matmul(self._prepare(), unwrap(right))
        return finalize(result, self._runtime)

    def _halo_exchange(
        self, dim: Hashable | None = None, *, before: int, after: int
    ) -> tuple[MPIXarray, int, int]:
        """Pad with boundary slices fetched from the adjacent ranks.

        Private: an implementation primitive for :meth:`rolling_reduce`
        and (in :mod:`.elementwise`) :meth:`diff`, not something a caller
        building on :class:`MPIXarray` should need to reach for directly.
        Point-to-point communication with a fixed neighbor is a much
        narrower, easier-to-misuse tool than :meth:`apply`; every windowed
        operation this package supports already has its own MPI-aware
        method (``rolling_reduce``, ``diff``) built on top of it.

        Parameters
        ----------
        dim : Hashable, optional
            Must equal ``self``'s active partition dimension if given;
            defaults to it.
        before : int
            Number of elements requested from the left neighbor.
        after : int
            Number of elements requested from the right neighbor.

        Returns
        -------
        tuple[MPIXarray, int, int]
            ``(padded, left_pad, right_pad)``.
        """
        padded, left_pad, right_pad = self._ops.halo_exchange(
            self._prepare(), dim, before=before, after=after
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
            "mean", "sum", "min", "max", "std").
        center : bool, optional
            As in ``xarray.DataArray.rolling``.
        min_periods : int or None, optional
            As in ``xarray.DataArray.rolling``.

        Returns
        -------
        MPIXarray
            Rolled-and-reduced object with ``.meta`` preserved.
        """
        result = self._ops.rolling_reduce(
            self._prepare(), dim, window, reduce, center=center, min_periods=min_periods
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

        Correct even when ``dim`` is the active partition dimension (see
        :meth:`rolling_reduce`). Call a reduction method on the returned
        handle, e.g. ``ds.rolling("time", 5).mean()``.

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
            Chainable handle; call ``.mean()``, ``.sum()``, ``.min()``,
            ``.max()``, ``.std()``, or ``.count()`` on it to get the
            rolled-and-reduced :class:`MPIXarray`.
        """
        return MPIRolling(self, dim, window, center=center, min_periods=min_periods)

    def evaluate(self, expression: str, /, **variables: Any) -> Any:
        """Evaluate a string expression, respecting normal operator precedence.

        Any :class:`MPIXarray` found in ``variables`` is unwrapped to its
        underlying data (with ``.meta`` reattached) before evaluation.

        Parameters
        ----------
        expression : str
            A Python expression referencing ``variables`` by name, e.g.
            ``"(a + b) * c - d / e"``.
        **variables : Any
            Values bound to the names used in ``expression``.

        Returns
        -------
        MPIXarray or Any
            The expression's value, wrapped if it is an xarray
            Dataset/DataArray, otherwise returned as-is.
        """
        unwrapped = {name: unwrap(value) for name, value in variables.items()}
        result = self._ops.evaluate(expression, **unwrapped)
        return finalize(result, self._runtime)

    # -- Elementwise, scan, and gather-based operations -------------------

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
            Fill value where ``cond`` is False. Omit for xarray's default
            (NaN).
        drop : bool, optional
            Must be False when ``self`` is distributed; ``drop=True`` can
            remove a different number of positions on different ranks.

        Returns
        -------
        MPIXarray
            The selected object, with ``.meta`` unchanged.
        """
        args = (cond,) if other is _WHERE_UNSET else (cond, other)
        return self._dispatch("where", *(unwrap(arg) for arg in args), drop=drop)

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
        return self._dispatch("cumsum", dim, skipna=skipna, keep_attrs=keep_attrs)

    def median(
        self,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> MPIXarray:
        """Median over ``dim``, correct when ``dim`` is distributed.

        When ``dim`` is the active partition dimension, this gathers the
        full dimension onto every rank (``MPI_Allgather``) and reduces
        locally, since median has no MPI reduction operator the way
        sum/min/max do -- exact, but not memory-scalable for a large
        partition dimension. Unlike :meth:`mean`/:meth:`sum`/etc., only a
        single dimension is supported (not an iterable, ``None``, or ``...``).

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
            Reduced object. Replicated (``.meta`` is None) if ``dim`` was
            the active partition dimension.
        """
        return self._dispatch("median", dim, skipna=skipna, keep_attrs=keep_attrs)

    def diff(
        self,
        dim: Hashable,
        n: int = 1,
        *,
        label: Literal["upper", "lower"] = "upper",
    ) -> MPIXarray:
        """``n``-th order difference along ``dim``, correct when ``dim`` is distributed.

        Works along the active partition dimension too: borrows ``n``
        boundary values from the relevant neighbor
        (:meth:`halo_exchange`) and recomputes distribution metadata from
        each rank's new local length -- see
        :func:`~.elementwise.Elementwise.diff` for the exact mechanism.

        Parameters
        ----------
        dim : Hashable
            Dimension to difference along.
        n : int, optional
            Order of the difference. Must be less than every rank's local
            length along ``dim`` when ``dim`` is the partition dimension.
        label : {"upper", "lower"}, optional
            As in ``xarray.DataArray.diff``.

        Returns
        -------
        MPIXarray
            The differenced object, ``n`` elements shorter along ``dim``
            globally, with ``.meta`` updated to match.
        """
        return self._dispatch("diff", dim, n, label=label)


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

    dim = meta["dim"]
    marked.attrs[_PARTITIONED_ATTR] = True
    if isinstance(marked, xr.Dataset):
        for variable in marked.variables.values():
            variable.attrs.pop(_PARTITIONED_ATTR, None)
            if dim in variable.dims:
                variable.attrs[_PARTITIONED_ATTR] = True
    return marked


def finalize(result: Any, runtime: MPIRuntime) -> Any:
    """Wrap an engine result when it is an xarray object.

    Parameters
    ----------
    result : Any
        Result returned by ``_MPIXarrayOps``.
    runtime : MPIRuntime
        Runtime bound to the wrapped result.

    Returns
    -------
    MPIXarray or Any
        Wrapped xarray result, or the original non-xarray value.
    """
    if isinstance(result, (xr.Dataset, xr.DataArray)):
        return MPIXarray(result, runtime, auto_partition=False)
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
