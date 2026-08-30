"""MPI-aware elementwise, scan, and order-statistic operations.

New primitives that do not fit either existing shape in :mod:`.operator`:

- :meth:`Elementwise.where` is strictly partition-preserving and
  elementwise, so it is implemented the same way
  :meth:`~.operator.Arithmetic.apply` would run it, with one extra check
  ``apply`` cannot express: ``drop=True`` is refused on distributed data,
  since it can remove a different number of positions on different ranks.
- :meth:`Elementwise.cumsum` is partition-preserving in *length* (every
  rank keeps its own local length) but is not partition-preserving in
  *value*: each rank's running total must start from every earlier rank's
  total. That needs a rank-ordered prefix, which this builds the same way
  :func:`~.io.IO.attach_save_chunks` builds its save-chunk plan: rank 0
  gathers every rank's local total (:meth:`~..mpi.runtime.MPIRuntime.gather`),
  computes each rank's exclusive prefix locally, and scatters one prefix
  back to each rank (:meth:`~..mpi.runtime.MPIRuntime.scatter`) -- no new
  MPI collective, just the ``gather``/``scatter`` pair
  :class:`~..mpi.runtime.MPIRuntime` already provides.
- :meth:`Elementwise.median` genuinely reduces the partition dimension
  away, like :mod:`.reductions`, but has no associative MPI reduction
  operator the way sum/min/max do. It gathers every rank's slice onto
  rank 0 (:meth:`~..mpi.runtime.MPIRuntime.gather`), which alone
  reconstructs the full dimension and takes xarray's own median, then
  broadcasts the (already-reduced, small) result back
  (:meth:`~..mpi.runtime.MPIRuntime.broadcast`) -- correct, and only rank 0
  ever materializes the full dimension, unlike an ``Allgather`` that would
  replicate it onto every rank.
- :meth:`Elementwise.diff` along the active partition dimension borrows
  ``n`` boundary elements from the one neighbor that matters --
  :meth:`~.operator.Arithmetic.halo_exchange` (before-only for
  ``label="upper"``, which drops the global *first* ``n`` elements and so
  only leaves rank 0 short; after-only for ``label="lower"``, which drops
  the global *last* ``n`` and only leaves the last rank short) -- then
  recomputes every rank's ``start``/``stop``/``global_size`` the same way
  :meth:`~.indexing.Indexing.isel` already does for its own
  length-changing slice case: an ``allgather`` of each rank's new local
  length, then a running sum. Every rank keeps its original local length
  except the one at the affected edge, which is ``n`` shorter -- exactly
  matching where the global array actually lost those ``n`` elements.

Mixed into :class:`~.ops._MPIXarrayOps` alongside
:class:`~.operator.Arithmetic`; requires the ``self._runtime`` attribute
(and its ``gather``/``scatter``/``broadcast``/``is_root`` methods) and the
``self._check_operands_distribution``/``self._check_partition_preserved``/
``self._reattach_meta``/``self._agree`` helpers :class:`~.operator.Arithmetic`
and :class:`~.engine.ReductionPlanningMixin` define.

Collective error-safety note
-----------------------------
:mod:`.engine`'s ``Allreduce``-based reductions guard the risky local step
before every collective with ``_guarded``/``raise_if_error`` so one rank's
local exception cannot leave the others blocked in a collective they will
now never receive. :meth:`Elementwise.cumsum` and :meth:`Elementwise.median`
do the same around their own risky local step (the local
cumsum/total computation for the former; the rank-0-only ``xr.concat`` +
reduce for the latter, the more dangerous of the two since every other rank
is already waiting at the final ``broadcast`` when it runs).
:class:`~..mpi.runtime.MPIRuntime`'s ``gather``/``scatter``/``broadcast``
themselves remain thin, unguarded wrappers around ``mpi4py``'s pickle-based
collectives -- a failure *inside* mpi4py's own collective call (as opposed
to the local xarray computation surrounding it, which is what is guarded)
is not covered by this and is a known, narrower follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from .chunks import prune_chunk_info
from .meta import get_mpi_meta, set_mpi_meta, strip_mpi_meta

if TYPE_CHECKING:
    from collections.abc import Hashable

#: Sentinel distinguishing "no fill value given" from a genuine ``other=None``.
_UNSET = object()


class Elementwise:
    """Elementwise, scan, and gather-based operations for distributed xarray objects.

    Assumes the host class provides ``self._runtime`` (with its
    ``gather``/``scatter``/``broadcast``/``is_root``/``raise_if_error``
    methods) and the ``self._check_operands_distribution``,
    ``self._check_partition_preserved``, ``self._reattach_meta``,
    ``self._agree``, ``self._guarded`` helpers defined on
    :class:`~.operator.Arithmetic`/:class:`~.engine.ReductionPlanningMixin`;
    provided by :class:`~.ops._MPIXarrayOps`.
    """

    def where(
        self,
        value: xr.Dataset | xr.DataArray,
        cond: Any,
        other: Any = _UNSET,
        *,
        drop: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """Elementwise selection (``value.where(cond, other)``), MPI-safe.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to select from.
        cond : Any
            Boolean condition, following ``xarray.DataArray.where``.
        other : Any, optional
            Fill value where ``cond`` is False. Omit for xarray's default
            (NaN).
        drop : bool, optional
            Must be False for a distributed object.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The selected object, with ``.meta`` preserved unchanged.

        Raises
        ------
        ValueError
            If ``drop=True`` is requested on a distributed object, or the
            operands are distributed over incompatible partitions (see
            :meth:`~.operator.Arithmetic.apply`).
        """
        operands = (value, cond) if other is _UNSET else (value, cond, other)
        meta, reference = self._check_operands_distribution(operands)
        if meta is not None and drop:
            raise ValueError(
                "where(): drop=True is not supported on a distributed "
                "object; it can remove a different number of positions on "
                "different ranks and desynchronize the partition. Select "
                "with isel()/sel() first, or repartition afterwards."
            )

        self._agree(
            (
                "where",
                None if meta is None else (str(meta["dim"]), int(meta["global_size"])),
            )
        )
        result = value.where(cond) if other is _UNSET else value.where(cond, other)
        if meta is None:
            return result
        self._check_partition_preserved(result, meta, reference)
        return self._reattach_meta(result, meta)

    def cumsum(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """Cumulative sum along ``dim``, correct when ``dim`` is distributed.

        When ``dim`` is the active partition dimension, each rank's running
        total must include every earlier rank's total. This gathers every
        rank's local total onto rank 0, which computes each rank's
        *exclusive* prefix (the sum of every rank before it) and scatters
        one prefix back to each rank; every rank then adds its prefix onto
        its own local cumulative sum. No new MPI collective: just the
        ``gather``/``scatter`` pair already used elsewhere in this package
        (see :func:`~.io.IO.attach_save_chunks`).

        The rank-local cumulative-sum/total computation happens on every
        rank independently before the first collective; it is guarded the
        same way :meth:`~.engine.ReductionPlanningMixin._comm_reduce` guards
        its own local step, so a local failure on one rank (e.g. an
        unsupported dtype) is reported consistently on every rank via
        ``raise_if_error`` instead of leaving the other ranks blocked
        waiting at ``gather`` for a rank that already raised.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to accumulate.
        dim : Hashable
            Dimension to accumulate along.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics. Applied
            consistently to both the local cumulative sum and the local
            total that feeds the cross-rank prefix, so a rank's NaNs never
            change another rank's prefix.
        keep_attrs : bool or None, optional
            Whether to preserve attributes on the rank-local cumulative sum
            step; lost by the subsequent addition of the cross-rank prefix.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Cumulative sum with the same local length and ``.meta`` as
            ``value``.
        """
        meta = get_mpi_meta(value)
        if meta is None or meta["dim"] != dim:
            return value.cumsum(dim, skipna=skipna, keep_attrs=keep_attrs)

        self._agree(("cumsum", str(dim), int(meta["global_size"])))

        def _locals() -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
            local_cumsum = value.cumsum(dim, skipna=skipna, keep_attrs=keep_attrs)
            local_total = value.sum(dim, skipna=skipna)
            return local_cumsum, local_total

        locals_or_none, error = self._guarded(_locals)
        self._runtime.raise_if_error(
            error, "MPI xarray cumsum", signature=("cumsum", str(dim))
        )
        local_cumsum, local_total = locals_or_none

        totals = self._runtime.gather(local_total, root=0)
        prefixes = None
        if self._runtime.is_root():
            prefixes = []
            running = totals[0] * 0
            for total in totals:
                prefixes.append(running)
                running = running + total
        exclusive_prefix = self._runtime.scatter(prefixes, root=0)

        result = local_cumsum + exclusive_prefix
        return self._reattach_meta(result, meta)

    def median(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """Median over ``dim``, correct when ``dim`` is distributed.

        Median has no MPI reduction operator (unlike sum/min/max), so when
        ``dim`` is the active partition dimension this gathers every
        rank's slice onto rank 0, which reconstructs the full ``dim`` and
        takes xarray's own median locally, then broadcasts the (already
        reduced, small) result back to every rank. Only rank 0 ever
        materializes the full ``dim`` -- unlike an ``Allgather``, which
        would replicate it onto every rank.

        The reconstruct-and-reduce step runs on rank 0 only, immediately
        before every other rank is already waiting at the final
        ``broadcast`` -- the case :meth:`cumsum`'s equivalent guarding
        note describes as most dangerous to leave unguarded, since a
        rank-0-only failure there (e.g. an ``xr.concat`` dtype mismatch)
        would otherwise leave every other rank blocked forever. Guarded
        the same way: the root's attempt is wrapped and any exception
        deferred, then every rank (root or not) calls ``raise_if_error``
        together so the failure -- if any -- is reported consistently
        everywhere instead of only on rank 0.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce. Unlike :meth:`~.core.MPIXarray.mean`,
            :meth:`~.core.MPIXarray.sum`, etc., only a single dimension is
            supported (not an iterable, ``None``, or ``...``).
        dim : Hashable
            Dimension to reduce.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object. Replicated (``.meta`` is None) if ``dim`` was
            the active partition dimension; otherwise ``.meta`` is
            preserved unchanged.
        """
        meta = get_mpi_meta(value)
        if meta is None or meta["dim"] != dim:
            return value.median(dim, skipna=skipna, keep_attrs=keep_attrs)

        self._agree(("median", str(dim), int(meta["global_size"])))
        pieces = self._runtime.gather(value, root=0)

        def _reduce_on_root() -> xr.Dataset | xr.DataArray:
            full = (
                xr.concat(pieces, dim=dim, data_vars="minimal")
                if isinstance(value, xr.Dataset)
                else xr.concat(pieces, dim=dim)
            )
            return full.median(dim, skipna=skipna, keep_attrs=keep_attrs)

        result, error = (
            self._guarded(_reduce_on_root) if self._runtime.is_root() else (None, None)
        )
        self._runtime.raise_if_error(
            error, "MPI xarray median", signature=("median", str(dim))
        )
        result = self._runtime.broadcast(result, root=0)
        return strip_mpi_meta(result)

    def diff(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable,
        n: int = 1,
        *,
        label: Literal["upper", "lower"] = "upper",
    ) -> xr.Dataset | xr.DataArray:
        """``n``-th order difference along ``dim``, correct when ``dim`` is distributed.

        When ``dim`` is the active partition dimension: ``label="upper"``
        drops the global *first* ``n`` elements (xarray labels each
        difference with the later/"upper" of the two positions it came
        from), so every rank except rank 0 can compute its output at full
        local length by borrowing ``n`` elements from its left neighbor
        (:meth:`~.operator.Arithmetic.halo_exchange`); rank 0 has no left
        neighbor and is genuinely ``n`` shorter, which is exactly where the
        global array actually lost those ``n`` elements. ``label="lower"``
        is the mirror image: drops the global *last* ``n``, borrows from
        the right neighbor instead, and only the last rank comes up short.
        Either way, every rank's new ``start``/``stop``/``global_size`` is
        then recomputed from an ``allgather`` of each rank's new local
        length -- the same mechanism :meth:`~.indexing.Indexing.isel`
        already uses for its own length-changing slice case.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to difference.
        dim : Hashable
            Dimension to difference along.
        n : int, optional
            Order of the difference. Must be less than every rank's local
            length along ``dim`` when ``dim`` is the partition dimension
            (see ``halo_exchange``'s own limit: a rank can only forward
            data it owns, so a wider request would need a multi-hop relay
            this does not perform).
        label : {"upper", "lower"}, optional
            As in ``xarray.DataArray.diff``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The differenced object, ``n`` elements shorter along ``dim``
            globally -- and, when ``dim`` is the partition dimension, at
            exactly one rank (0 for "upper", the last rank for "lower")
            locally; every other rank's local length is unchanged.

        Raises
        ------
        ValueError
            If ``n`` is negative, ``label`` is not "upper"/"lower", or any
            rank's local length along ``dim`` is shorter than ``n`` (this
            last case is caught by :meth:`~.operator.Arithmetic.halo_exchange`
            itself, which checks every rank's local length together via a
            synchronized ``allgather`` before raising, so the error is
            consistent and every rank raises together rather than some
            hanging).
        """
        meta = get_mpi_meta(value)
        if meta is None or meta["dim"] != dim:
            return value.diff(dim, n=n, label=label)
        if n < 0:
            raise ValueError(f"diff(): n must be >= 0, got {n!r}.")
        if label not in ("upper", "lower"):
            raise ValueError(
                f"diff(): label must be 'upper' or 'lower', got {label!r}."
            )
        if n == 0:
            return self._reattach_meta(value.diff(dim, n=0, label=label), meta)

        before, after = (n, 0) if label == "upper" else (0, n)
        padded, _left_pad, _right_pad = self.halo_exchange(
            value, dim, before=before, after=after
        )
        diffed = padded.diff(dim, n=n, label=label)

        comm = self._runtime.comm
        counts = comm.allgather(int(diffed.sizes[dim]))
        new_global_size = sum(counts)
        new_start = sum(counts[: comm.rank])
        new_stop = new_start + counts[comm.rank]
        chunk_info = prune_chunk_info(meta["chunk_info"], diffed)
        set_mpi_meta(
            diffed,
            dim=dim,
            global_size=new_global_size,
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
        return diffed
