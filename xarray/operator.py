"""MPI-aware alignment and arithmetic for distributed xarray objects."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from mpi4py import MPI

import xarray as xr

from .meta import _partitions_match, get_mpi_meta, set_mpi_meta, strip_mpi_meta

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping

from collections.abc import Callable

# Callables apply() recognizes and transparently redirects to their
# dedicated implementation, so apply() is MPI-aware for them the same way
# evaluate() is: apply(operator.matmul, a, b) computes the same correct,
# MPI-reduced result as evaluate("a @ b", a=a, b=b) and mpi.xarray.matmul(a, b),
# instead of running the plain rank-local matmul and failing the post-call
# partition check whenever the distributed dimension gets contracted away.
_MATMUL_CALLABLES: frozenset[Callable[..., Any]] = frozenset(
    {operator.matmul, np.matmul}
)
# ast.MatMult ('@') is deliberately absent: whether matrix multiplication is
# rank-local depends on which dimension gets contracted, so it is routed to
# the dedicated Arithmetic.matmul() implementation in _eval_ast_node()
# instead of the generic apply(operator.matmul, ...) table below.
_AST_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,  # Bitwise left shift (<<)
    ast.RShift: operator.rshift,  # Bitwise right shift (>>)
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

# Complete Comparison Operations (==, !=, <, <=, >, >=, is, is not, in, not in)
_AST_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,  # Identity (is)
    ast.IsNot: operator.is_not,  # Negated identity (is not)
    ast.In: lambda a, b: a in b,  # Membership (in)
    ast.NotIn: lambda a, b: a not in b,  # Negated membership (not in)
}

# Complete Unary Operations (-, +, ~, not)
_AST_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,  # Logical negation (not)
}


# Boolean Operations (and, or)
_AST_BOOL_OPS: dict[type[ast.boolop], Callable[[list[Any]], Any]] = {
    ast.And: lambda values: all(values),
    ast.Or: lambda values: any(values),
}


class Arithmetic:
    """Alignment and arithmetic methods mixed into ``XarrayMPI``.

    Assumes the host class provides ``self.repartition`` (used by
    :meth:`align`), ``self._agree`` (used by :meth:`apply`, :meth:`matmul`,
    and :meth:`halo_exchange`), and ``self._comm_reduce`` (used by
    :meth:`matmul`, an ``Allreduce`` helper defined on
    :class:`~.engine.ReductionPlanningMixin`); all are provided by
    :class:`~climtools.core.xarray_mpi.XarrayMPI`.
    """

    # -- alignment ----------------------------------------------------------
    #
    # A convenience counterpart to xarray's own align(): instead of
    # reconciling coordinate labels/indexes, this reconciles which rank
    # owns which slice, so left and right end up combinable by apply()/
    # evaluate() with zero further MPI traffic. Two cases resolve without
    # any communication: a replicated operand sliced down onto an
    # already-distributed partner's exact bounds, and two replicated
    # operands independently repartitioned along the same dimension (which
    # is deterministic given (length, chunk size, rank, size), so both
    # land on identical bounds without needing to compare notes). Two
    # operands already distributed on genuinely different partitions do
    # need data movement to reconcile -- handled by gathering each back to
    # its full extent on every rank and repartitioning both onto a shared
    # scheme (see :meth:`align`/:meth:`_gather_full`); simple and correct,
    # though not memory-scalable, rather than a true personalized
    # Alltoallv that would avoid ever fully materializing either operand.

    def _gather_full(
        self, value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]
    ) -> xr.Dataset | xr.DataArray:
        """Reconstruct ``value``'s full, replicated extent on every rank.

        Used by :meth:`align` to reconcile two operands distributed on
        genuinely different partitions: :meth:`repartition` (which
        :meth:`align` then calls on the result) requires a replicated
        input present on every rank, not just rank 0, so this uses
        ``MPI_Allgather`` rather than the gather-to-root-then-broadcast
        pattern :meth:`~.elementwise.Elementwise.median` uses for its
        (much smaller) reduced result -- every rank ends up holding the
        complete array, which is correct but not memory-scalable for a
        large distributed dimension.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            This rank's own local slice.
        meta : mapping
            ``value``'s distribution metadata.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The full, replicated object (no ``mpi_meta`` attached).
        """
        dim = meta["dim"]
        pieces = self._runtime.comm.allgather(value)
        full = (
            xr.concat(pieces, dim=dim, data_vars="minimal")
            if isinstance(value, xr.Dataset)
            else xr.concat(pieces, dim=dim)
        )
        return strip_mpi_meta(full)

    def _align_replicated(
        self,
        other: Any,
        meta: dict[str, Any],
        partner: xr.Dataset | xr.DataArray | None = None,
    ) -> Any:
        """Slice a replicated operand onto an already-distributed partner's bounds.

        Parameters
        ----------
        other : Any
            Replicated operand to slice. Returned unchanged if it is not an
            xarray object or does not carry ``meta["dim"]``.
        meta : dict[str, Any]
            Distribution metadata of the already-distributed partner.
        partner : xarray.Dataset or xarray.DataArray, optional
            The already-distributed partner itself. When given and both
            objects carry an index along ``dim``, :func:`xarray.align` with
            ``join="exact"`` cross-checks the coordinate *labels* of the
            slice against the partner's own labels. This is a local,
            communication-free check: ``partner`` already holds only this
            rank's slice, so it catches a replicated operand whose ``dim``
            coordinate does not correspond label-for-label to the
            partner's (reordered, offset, or otherwise not simply "the same
            index sliced the same way"), which same-length position-based
            slicing alone cannot detect.

        Returns
        -------
        Any
            The sliced xarray object with distribution metadata attached, or
            ``other`` unchanged if it is not an xarray object.

        Raises
        ------
        ValueError
            If the operand length does not match the global size, or if
            coordinate labels fail the ``join="exact"`` validation check.
        """
        dim = meta["dim"]
        if not isinstance(other, (xr.Dataset, xr.DataArray)) or dim not in other.dims:
            return other

        length = int(other.sizes[dim])
        global_size = int(meta["global_size"])
        if length != global_size:
            raise ValueError(
                f"Cannot align: operand carries dimension {dim!r} at length "
                + f"{length}, but the distributed partner's global size "
                + f"along {dim!r} is {global_size}. align() only slices a "
                + "replicated (full-length) operand onto an existing "
                + "partition; lengths must match the whole distributed "
                + "dimension."
            )
        sliced = other.isel({dim: slice(meta["start"], meta["stop"])})

        if (
            partner is not None
            and dim in getattr(partner, "indexes", {})
            and dim in getattr(sliced, "indexes", {})
        ):
            try:
                xr.align(partner, sliced, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Cannot align: the replicated operand's {dim!r} labels "
                    + "do not match the distributed partner's labels for "
                    + "this rank's slice, even though both have length "
                    + f"{meta['stop'] - meta['start']}. xarray.align(..., "
                    + f"join='exact') reports: {exc}"
                ) from exc

        return self._reattach_meta(sliced, meta)

    def align(
        self,
        left: xr.Dataset | xr.DataArray,
        right: xr.Dataset | xr.DataArray,
        dim: Hashable | Literal["auto"] | None = None,
        *,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
        """Return ``(left, right)`` partitioned identically across ranks.

        The counterpart to :func:`xarray.align` for rank ownership rather
        than coordinate labels: after this call, ``left`` and ``right`` are
        guaranteed combinable by :meth:`apply`/:meth:`evaluate` without
        raising and without any further MPI communication.

        Four cases are handled:

        - Neither operand is distributed: returned unchanged, unless
          ``dim`` is given, in which case both are distributed along
          ``dim`` via ``repartition``. Repartition is a deterministic
          function of (length, chunk size, rank, size), so two operands of
          the same length along ``dim`` land on identical per-rank bounds
          without any coordination between them. No data moves between
          ranks.
        - One operand is already distributed, the other is replicated at
          the full length of the distributed dimension (or does not carry
          that dimension at all): the replicated operand is sliced down to
          the distributed operand's exact bounds. No data moves between
          ranks -- the replicated side already has the full array present
          on every rank, so this is a purely local slice.
        - Both are already distributed identically: returned unchanged.
        - Both are already distributed, but *differently* (different
          dimension, global size, or per-rank bounds): unlike the other
          three cases, this genuinely requires moving data between ranks.
          Each operand is gathered back to its full extent on every rank
          (``MPI_Allgather`` plus ``xr.concat`` -- the same mechanism as
          :meth:`~.elementwise.Elementwise.median`'s gather, but onto
          every rank rather than just rank 0, since ``repartition``
          requires a replicated input present everywhere), then both are
          repartitioned onto a shared scheme (``dim`` if given, else
          ``left``'s own dimension). Correct, but -- like ``median`` --
          not memory-scalable: every rank briefly holds the complete data
          for both operands.

        Parameters
        ----------
        left : xarray.Dataset or xarray.DataArray
            Left operand to align.
        right : xarray.Dataset or xarray.DataArray
            Right operand to align.
        dim : hashable or {"auto"}, optional
            Dimension to distribute both operands along when neither is
            currently distributed, or the shared dimension to reconcile
            onto when both are already distributed differently. Required
            when neither operand is distributed; defaults to ``left``'s
            own dimension when both are already distributed differently.
        chunk_info : mapping, optional
            Forwarded to ``repartition``.
        log_partitions : bool, optional
            Forwarded to ``repartition``.

        Returns
        -------
        tuple of xarray.Dataset or xarray.DataArray
            ``(left, right)``, each carrying matching distribution
            metadata (or neither carrying any, if both remain replicated).

        Raises
        ------
        ValueError
            If neither operand is distributed and ``dim`` is omitted.

        Examples
        --------
        >>> left, right = mpi.xarray.align(local_field, full_climatology)
        >>> anomaly = mpi.xarray.apply(operator.sub, left, right)

        >>> left, right = mpi.xarray.align(a_full, b_full, dim="time")
        >>> combined = mpi.xarray.apply(operator.add, left, right)

        >>> # both already distributed, but on different dimensions
        >>> left, right = mpi.xarray.align(a_by_time, b_by_space)
        >>> combined = mpi.xarray.apply(operator.add, left, right)
        """
        left_meta = self._operand_meta(left)
        right_meta = self._operand_meta(right)

        if left_meta is not None and right_meta is not None:
            if _partitions_match(left_meta, right_meta):
                return left, right
            target_dim = dim if dim is not None else left_meta["dim"]
            full_left = self._gather_full(left, left_meta)
            full_right = self._gather_full(right, right_meta)
            return (
                self.repartition(
                    full_left,
                    target_dim,
                    chunk_info=chunk_info,
                    log_partitions=log_partitions,
                ),
                self.repartition(
                    full_right,
                    target_dim,
                    chunk_info=chunk_info,
                    log_partitions=log_partitions,
                ),
            )

        if left_meta is not None:
            return left, self._align_replicated(right, left_meta, partner=left)

        if right_meta is not None:
            return self._align_replicated(left, right_meta, partner=right), right

        if dim is None:
            return left, right

        if (
            isinstance(left, (xr.Dataset, xr.DataArray))
            and isinstance(right, (xr.Dataset, xr.DataArray))
            and dim in getattr(left, "indexes", {})
            and dim in getattr(right, "indexes", {})
        ):
            try:
                xr.align(left, right, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Cannot align: left and right disagree on {dim!r} "
                    + "coordinate labels, so distributing each "
                    + "independently would silently combine mismatched "
                    + f"slices. xarray.align(..., join='exact') reports: {exc}"
                ) from exc

        return (
            self.repartition(
                left, dim, chunk_info=chunk_info, log_partitions=log_partitions
            ),
            self.repartition(
                right, dim, chunk_info=chunk_info, log_partitions=log_partitions
            ),
        )

    # -- arithmetic -----------------------------------------------------------

    @staticmethod
    def _operand_meta(operand: Any) -> dict[str, Any] | None:
        """Return ``operand``'s MPI distribution metadata, if any.

        Parameters
        ----------
        operand : Any
            The object to inspect for MPI metadata.

        Returns
        -------
        dict[str, Any] or None
            The distribution metadata dictionary if present, otherwise None.
        """
        if isinstance(operand, (xr.Dataset, xr.DataArray)):
            return get_mpi_meta(operand)
        return None

    @staticmethod
    def _reattach_meta(result: Any, meta: dict[str, Any]) -> Any:
        """Tag ``result`` with ``meta`` if it is an xarray object.

        Parameters
        ----------
        result : Any
            The computation result to be tagged.
        meta : dict[str, Any]
            The distribution metadata dictionary to reattach.

        Returns
        -------
        Any
            The tagged result object if it is an xarray dataset or dataarray,
            otherwise returned unmodified.
        """
        if isinstance(result, (xr.Dataset, xr.DataArray)):
            set_mpi_meta(
                result,
                dim=meta["dim"],
                global_size=meta["global_size"],
                start=meta["start"],
                stop=meta["stop"],
                chunk_info=meta["chunk_info"],
            )
        return result

    def _check_operands_distribution(
        self, operands: Iterable[Any]
    ) -> tuple[dict[str, Any] | None, Any]:
        """Return the mpi_meta to attach to a multi-operand call's result.

        Parameters
        ----------
        operands : iterable of Any
            Every positional and keyword argument passed to :meth:`apply`.

        Returns
        -------
        tuple[dict[str, Any] | None, Any]
            ``(meta, reference)``: metadata to reattach to the result (or
            None when no operand is distributed) together with the first
            distributed operand itself, used by :meth:`apply` as the
            coordinate baseline for post-call validation.

        Raises
        ------
        ValueError
            If two operands are distributed over different partitions, if a
            replicated operand carries the distributed dimension at a
            different length than the partition owns, if a replicated
            operand's coordinate labels along the distributed dimension do
            not match the distributed partition's labels for this rank's
            slice (equal length alone does not imply equal coordinates), or
            (on more than one rank) if that coordinate check cannot even
            run because either side has no coordinate for the distributed
            dimension -- equal length alone is not enough evidence the
            operand is genuinely this rank's own data rather than another
            rank's same-length slice by coincidence.
        """
        operands = list(operands)
        metas = [self._operand_meta(item) for item in operands]

        ref_index = next((i for i, item in enumerate(metas) if item is not None), None)
        if ref_index is None:
            return None, None
        meta = metas[ref_index]
        reference = operands[ref_index]

        for other, other_meta in zip(operands, metas, strict=True):
            if other_meta is not None:
                if not _partitions_match(meta, other_meta):
                    raise ValueError(
                        "Cannot combine operands distributed over "
                        + f"different partitions: dim={meta['dim']!r} "
                        + f"range=[{meta['start']}:{meta['stop']}) vs "
                        + f"dim={other_meta['dim']!r} "
                        + f"range=[{other_meta['start']}:{other_meta['stop']}). "
                        + "Call mpi.xarray.align(...) first."
                    )
                continue

            dim = meta["dim"]
            if isinstance(other, (xr.Dataset, xr.DataArray)) and dim in other.dims:
                owned = meta["stop"] - meta["start"]
                local = int(other.sizes[dim])
                if local != owned:
                    raise ValueError(
                        f"Operand carries dimension {dim!r} at length "
                        + f"{local}, which does not match this rank's "
                        + f"owned partition length {owned}. Call "
                        + "mpi.xarray.align(...) first."
                    )
                reference_indexed = dim in getattr(reference, "indexes", {})
                other_indexed = dim in getattr(other, "indexes", {})
                if reference_indexed and other_indexed:
                    try:
                        xr.align(reference, other, join="exact")
                    except (ValueError, KeyError) as exc:
                        raise ValueError(
                            f"Operand carries dimension {dim!r} at the "
                            + f"expected local length ({owned}), but its "
                            + "coordinate labels do not match the "
                            + "distributed partition's labels for this "
                            + "rank's slice; equal length does not imply "
                            + "equal coordinates. Call mpi.xarray.align(...) "
                            + "first. xarray.align(..., join='exact') "
                            + f"reports: {exc}"
                        ) from exc
                elif self._runtime.comm.size > 1:
                    # Equal length is necessary but not sufficient: without a
                    # coordinate on dim to check exactly (the branch above),
                    # there is no way to tell this rank's own correctly
                    # aligned slice apart from, say, a different rank's
                    # slice of the same length -- a silently wrong answer
                    # that would otherwise pass unnoticed. Refuse rather
                    # than trust length alone once more than one rank makes
                    # that ambiguity possible.
                    missing = [
                        name
                        for name, indexed in (
                            ("the distributed side", reference_indexed),
                            ("the operand", other_indexed),
                        )
                        if not indexed
                    ]
                    raise ValueError(
                        f"Operand carries dimension {dim!r} at the "
                        + f"expected local length ({owned}), but its "
                        + "alignment with this rank's own owned slice "
                        + f"cannot be verified: {' and '.join(missing)} "
                        + f"has no coordinate for {dim!r}, so equal length "
                        + "alone is not enough evidence this is actually "
                        + "this rank's own data rather than, say, another "
                        + "rank's same-length slice by coincidence. Add a "
                        + f"coordinate for {dim!r} to both sides so it can "
                        + "be checked exactly, or build the operand from "
                        + "the distributed side directly (e.g. via "
                        + "isel()/apply() on it) instead of a separately "
                        + "constructed array."
                    )
        return meta, reference

    @staticmethod
    def _check_partition_preserved(
        result: Any, meta: Mapping[str, Any], reference: Any
    ) -> None:
        """Verify ``result`` still owns the same partition-dimension slice.

        Called after evaluating a callable passed to :meth:`apply`, before
        distribution metadata is reattached. Catches the case where the
        callable reduced, resized, reordered, renamed, or otherwise
        repartitioned the distributed dimension -- silently reattaching the
        pre-call metadata to such a result would misrepresent which global
        indices this rank's slice actually holds.

        Parameters
        ----------
        result : Any
            The value returned by the callable. Non-xarray results are not
            distributed data and pass unconditionally.
        meta : Mapping[str, Any]
            The distribution metadata captured before the call.
        reference : Any
            The distributed operand the metadata was taken from, used as
            the coordinate baseline for the label check below.

        Raises
        ------
        ValueError
            If the distributed dimension is missing from ``result``, its
            local length changed, or its coordinate labels no longer match
            this rank's owned interval.
        """
        if not isinstance(result, (xr.Dataset, xr.DataArray)):
            return

        dim = meta["dim"]
        owned = meta["stop"] - meta["start"]

        if dim not in result.dims:
            raise ValueError(
                "apply(): the callable removed or renamed the distributed "
                + f"dimension {dim!r} (result dims: {tuple(result.dims)!r}). "
                + "apply() only supports partition-preserving rank-local "
                + "callables; use the corresponding mpi.xarray reduction, "
                + "indexing, or groupby method for operations that change "
                + "the partition dimension."
            )

        local = int(result.sizes[dim])
        if local != owned:
            raise ValueError(
                "apply(): the callable changed the local length of the "
                + f"distributed dimension {dim!r} from {owned} to {local} "
                + "on this rank. apply() only supports partition-preserving "
                + "rank-local callables that leave every rank's owned "
                + "slice the same length; operations such as slicing, "
                + "dropping, or windowed reductions along the partition "
                + "dimension require values from neighboring ranks and "
                + "must not be done inside apply()."
            )

        if (
            isinstance(reference, (xr.Dataset, xr.DataArray))
            and dim in getattr(reference, "indexes", {})
            and dim in getattr(result, "indexes", {})
        ):
            try:
                xr.align(reference, result, join="exact")
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"apply(): the callable changed the {dim!r} coordinate "
                    + "labels on this rank, even though the local length "
                    + f"({local}) is unchanged. apply() only supports "
                    + "partition-preserving rank-local callables that leave "
                    + f"each rank's owned {dim!r} interval untouched. "
                    + f"xarray.align(..., join='exact') reports: {exc}"
                ) from exc

    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

        ``func`` must be a partition-preserving, rank-local callable: for
        the distributed dimension ``d``, ``output[d]`` on rank ``r`` must
        represent exactly the same global ``[start_r:stop_r)`` indices as
        the input. Concretely, the callable must not reduce, resize,
        reorder, rename, repartition, or require values owned by another
        rank along ``d`` (e.g. ``ds.mean("time")``, ``ds.isel(time=...)``,
        ``ds.dropna("time")``, ``ds.rename({"time": ...})``, or
        ``ds.rolling(time=...)`` when ``time`` is the partition dimension);
        use the corresponding ``mpi.xarray`` reduction, indexing, or
        groupby method for a dimension-changing operation, or
        :meth:`rolling_reduce`/:meth:`halo_exchange` for a windowed
        operation that genuinely needs a neighboring rank's boundary
        values. This is checked after the call
        (dimension present, local length unchanged, coordinate labels
        unchanged) rather than by inspecting ``func``, since arbitrary
        Python callables cannot be statically verified.

        One exception: ``apply`` recognizes ``operator.matmul``/
        ``numpy.matmul`` by identity and transparently redirects a
        two-positional-argument, no-keyword call to :meth:`matmul` instead
        of running it through the generic path above -- the same
        redirection :meth:`evaluate` performs for ``@`` -- since a plain
        rank-local matmul call would either silently drop MPI-owned data
        (bypassing the post-call check that would otherwise catch it) or,
        under that check, unconditionally fail whenever the distributed
        dimension happens to be one of the contracted dimensions, even
        though that case has a well-defined correct distributed answer
        (see :meth:`matmul`).

        Parameters
        ----------
        func : callable
            Any partition-preserving, rank-local function of the given
            ``args`` and ``kwargs``.
        *args : Any
            Positional arguments to ``func``: xarray Datasets or DataArrays
            (distributed or not) or plain scalars and arrays, in any mix.
        **kwargs : Any
            Keyword arguments to ``func``, checked for distribution
            metadata exactly like ``args``.

        Returns
        -------
        Any
            The result of ``func(*args, **kwargs)``. When any argument is
            distributed, the result is tagged with the same distribution metadata.

        Raises
        ------
        ValueError
            If the xarray arguments are distributed over incompatible
            partitions or their coordinates disagree, or if the callable's
            result no longer represents the same owned partition (missing
            dimension, changed local length, or changed coordinate labels).

        Examples
        --------
        >>> mpi.xarray.apply(operator.add, a, b)
        >>> mpi.xarray.apply(xr.where, cond, a, b)
        >>> mpi.xarray.apply(lambda x, *, factor: x * factor, a, factor=2.0)
        >>> mpi.xarray.apply(operator.matmul, a, b)  # redirected to matmul(), see below
        """
        if func in _MATMUL_CALLABLES and not kwargs and len(args) == 2:
            return self.matmul(*args)

        return self._apply_generic(func, args, kwargs)

    def _apply_generic(
        self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """The partition-preserving-callable path shared by :meth:`apply` and
        the "not actually contracted" fallback in :meth:`matmul`.

        Split out from :meth:`apply` so that fallback can invoke this
        validated generic path directly instead of calling
        ``self.apply(operator.matmul, ...)``, which would recurse straight
        back into :meth:`matmul` via the redirect above.
        """
        meta, reference = self._check_operands_distribution((*args, *kwargs.values()))

        self._agree(
            (
                "apply",
                getattr(func, "__name__", repr(func)),
                None if meta is None else (str(meta["dim"]), int(meta["global_size"])),
            )
        )

        result = func(*args, **kwargs)
        if meta is None:
            return result
        self._check_partition_preserved(result, meta, reference)
        return self._reattach_meta(result, meta)

    # -- dedicated non-partition-preserving operations -----------------------
    #
    # apply() only accepts callables that leave the partition dimension
    # untouched. The two methods below are the "dedicated implementations"
    # for the classes of operation that genuinely need to reduce or
    # communicate across it: matrix multiplication that contracts the
    # partition dimension (needs an MPI reduction), and windowed/rolling
    # reductions along the partition dimension (need boundary values owned
    # by a neighboring rank). Both compute the mathematically correct
    # distributed result instead of refusing outright.

    def matmul(self, left: xr.DataArray, right: Any) -> xr.DataArray:
        """Matrix multiplication (``left @ right``), correct under MPI.

        ``xarray.DataArray.__matmul__`` (and therefore ``@``) contracts
        over every dimension common to both operands:
        ``C = sum_{k in common dims} A_k * B_k``. When the distributed
        dimension ``d`` is not one of those common dimensions, the
        contraction never touches data owned by another rank and this is
        simply routed through :meth:`apply`. When ``d`` is contracted, the
        sum splits additively over ``d``:

        ``C_ij = sum_k A_ik * B_kj = sum_r (sum_{k in rank r} A_ik * B_kj) = sum_r C_ij^(r)``

        so each rank computes its local partial contraction over its own
        owned slice of ``d`` (an ordinary rank-local ``dot``), and one
        ``MPI_Allreduce`` (``MPI.SUM``) combines the partials into the
        correct, fully-replicated global result.

        Parameters
        ----------
        left : xarray.DataArray
            Left operand.
        right : Any
            Right operand: an ``xarray.DataArray`` (distributed or not) or
            a plain array/scalar ``left`` can be matrix-multiplied with.

        Returns
        -------
        xarray.DataArray
            The matrix product. Replicated (no ``mpi_meta``) if the
            distributed dimension was contracted away; otherwise tagged
            with the same distribution metadata as the input.

        Raises
        ------
        ValueError
            If ``left``/``right`` are distributed over incompatible
            partitions (see :meth:`apply`).
        TypeError
            If the dtype involved has no MPI reduction datatype, when the
            distributed dimension is contracted.

        Examples
        --------
        >>> mpi.xarray.matmul(a, b)  # same as mpi.xarray.evaluate("a @ b", a=a, b=b)
        """
        meta, _reference = self._check_operands_distribution((left, right))
        if meta is None:
            return self._apply_generic(operator.matmul, (left, right), {})

        dim = meta["dim"]
        if not (dim in getattr(left, "dims", ()) and dim in getattr(right, "dims", ())):
            # `dim` is not one of the dot product's common dimensions, so it
            # is never contracted: the operation only reads this rank's own
            # owned slice and apply()'s post-call check confirms it.
            return self._apply_generic(operator.matmul, (left, right), {})

        self._agree(("matmul", str(dim), int(meta["global_size"])))

        partial = operator.matmul(left, right)
        total = self._comm_reduce(
            partial, MPI.SUM, phase="MPI xarray distributed matrix multiplication"
        )
        return strip_mpi_meta(total)

    def halo_exchange(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable | None = None,
        *,
        before: int,
        after: int,
    ) -> tuple[xr.Dataset | xr.DataArray, int, int]:
        """Pad ``value`` with boundary slices from the adjacent ranks.

        The dedicated primitive for operations that need values owned by a
        neighboring rank along the partition dimension -- exactly what
        :meth:`apply` refuses to let a callable do internally, because it
        cannot verify the callable stayed rank-local. Fetches
        ``before``/``after`` elements from rank ``r - 1``/``r + 1`` with
        non-blocking point-to-point communication (no collective), and
        concatenates them onto this rank's own slice.

        At a global edge (rank 0 for ``before``, the last rank for
        ``after``) there is no neighbor to ask, so that side is padded
        with 0 elements rather than a value invented from nothing; a
        windowed computation built on the result naturally reports
        undefined (e.g. via ``min_periods``) there, which matches how
        ``xarray.DataArray.rolling`` already treats a global boundary.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Distributed object to pad. Must carry MPI metadata.
        dim : Hashable, optional
            Must equal ``value``'s active partition dimension if given;
            defaults to it.
        before, after : int
            Number of elements requested from the left/right neighbor.
            Must not exceed this rank's own local length, since a rank can
            only ever forward data it owns -- fetching a window wider than
            one rank's partition would need a multi-hop relay, which this
            primitive does not perform.

        Returns
        -------
        tuple[xarray.Dataset or xarray.DataArray, int, int]
            ``(padded, left_pad, right_pad)``: the padded object (replicated
            metadata stripped, since it is no longer a clean partition) and
            the number of elements actually prepended/appended (equal to
            ``before``/``after`` except at a global edge, where it is 0).

        Raises
        ------
        ValueError
            If ``value`` is not distributed, ``dim`` disagrees with its
            partition dimension, ``before``/``after`` are negative, or any
            rank's local partition is shorter than ``before``/``after``.
            That last check is a synchronized ``allgather`` of every
            rank's local length, so the error (if any) is raised
            consistently, together, on every rank -- not just the
            deficient one, which would otherwise leave the other ranks
            blocked in the point-to-point exchange below, waiting on a
            rank that already raised and will never reach it.
        """
        meta = self._operand_meta(value)
        if meta is None:
            raise ValueError(
                "halo_exchange() requires a distributed xarray object; "
                + "call mpi.xarray.repartition(...) first."
            )
        partition_dim = meta["dim"]
        if dim is not None and dim != partition_dim:
            raise ValueError(
                f"halo_exchange(): dim={dim!r} does not match the object's "
                + f"active partition dimension {partition_dim!r}."
            )
        if before < 0 or after < 0:
            raise ValueError("halo_exchange(): before and after must be >= 0.")

        comm = self._runtime.comm
        rank, size = comm.rank, comm.size
        left_rank = rank - 1 if rank > 0 else None
        right_rank = rank + 1 if rank < size - 1 else None

        self._agree(("halo_exchange", str(partition_dim), int(before), int(after)))

        local_len = int(value.sizes[partition_dim])
        lengths = comm.allgather(local_len)
        deficient = [
            (r, length)
            for r, length in enumerate(lengths)
            if length < before or length < after
        ]
        if deficient:
            raise ValueError(
                f"halo_exchange(): rank(s) {deficient} ([rank, local_length]) "
                + f"have a local partition along {partition_dim!r} shorter "
                + f"than the requested halo (before={before}, after={after}). "
                + "Each rank can only forward data it owns; repartition "
                + "with fewer, larger chunks (or a coarser process grid) "
                + "before requesting this wide a halo."
            )

        recv_before_req = (
            comm.irecv(source=left_rank) if left_rank is not None else None
        )
        recv_after_req = (
            comm.irecv(source=right_rank) if right_rank is not None else None
        )

        send_reqs = []
        if right_rank is not None:
            block = value.isel({partition_dim: slice(local_len - before, local_len)})
            send_reqs.append(comm.isend(block, dest=right_rank))
        if left_rank is not None:
            block = value.isel({partition_dim: slice(0, after)})
            send_reqs.append(comm.isend(block, dest=left_rank))

        before_block = recv_before_req.wait() if recv_before_req is not None else None
        after_block = recv_after_req.wait() if recv_after_req is not None else None
        MPI.Request.Waitall(send_reqs)

        pieces = [
            piece for piece in (before_block, value, after_block) if piece is not None
        ]
        if len(pieces) <= 1:
            padded = value
        elif isinstance(value, xr.Dataset):
            # data_vars="minimal": only concatenate variables that actually
            # vary along partition_dim. The default ("all") broadcasts every
            # *other* variable along it too, silently turning a static
            # (y, x) variable into a bogus (partition_dim, y, x) one
            # duplicated across before_block/value/after_block -- those three
            # pieces already agree exactly on any variable that lacks
            # partition_dim (each is this rank's or a neighbor's full,
            # untouched copy), so "minimal" is not just faster but the only
            # option that leaves such variables unchanged.
            padded = xr.concat(pieces, dim=partition_dim, data_vars="minimal")
        else:
            padded = xr.concat(pieces, dim=partition_dim)
        return (
            strip_mpi_meta(padded),
            before if before_block is not None else 0,
            after if after_block is not None else 0,
        )

    def rolling_reduce(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable,
        window: int,
        reduce: str = "mean",
        *,
        center: bool = True,
        min_periods: int | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """Windowed reduction along ``dim``, correct when ``dim`` is distributed.

        Equivalent to
        ``value.rolling({dim: window}, center=center, min_periods=min_periods).<reduce>()``,
        but safe to call when ``dim`` is the active MPI partition
        dimension: plain ``xarray`` rolling only ever sees this rank's own
        local slice, so a window that spans a partition edge silently
        computes over the wrong (or, near a rank boundary, insufficient)
        data -- exactly the ``rolling(...).mean()`` case :meth:`apply`'s
        docstring warns cannot be done inside a callable. This method
        fetches the missing boundary values with :meth:`halo_exchange`,
        rolls over the padded local array, then trims the halo padding
        back off so the result is partition-preserving and safe to hand
        back to :meth:`apply`/:meth:`evaluate`.

        When ``dim`` is not the active partition dimension (or ``value``
        is not distributed), this delegates directly to
        ``xarray``'s own ``rolling`` with no MPI involvement.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to roll over.
        dim : Hashable
            Dimension to roll over.
        window : int
            Window size, as in ``xarray.DataArray.rolling``.
        reduce : str, optional
            Name of the reduction to call on the rolling object (e.g.
            ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"std"``). Default
            ``"mean"``.
        center : bool, optional
            As in ``xarray.DataArray.rolling``. Default True.
        min_periods : int or None, optional
            As in ``xarray.DataArray.rolling``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The rolled-and-reduced result, with the same local length and
            distribution metadata as the input when ``dim`` is the
            partition dimension.
        """
        meta = self._operand_meta(value)
        if meta is None or meta["dim"] != dim:
            rolled = value.rolling(
                {dim: window}, center=center, min_periods=min_periods
            )
            return getattr(rolled, reduce)()

        before = (window - 1) // 2 if center else window - 1
        after = (window - 1) - before if center else 0

        padded, left_pad, _right_pad = self.halo_exchange(
            value, dim, before=before, after=after
        )
        rolled = padded.rolling({dim: window}, center=center, min_periods=min_periods)
        reduced = getattr(rolled, reduce)()

        local_len = int(value.sizes[dim])
        trimmed = reduced.isel({dim: slice(left_pad, left_pad + local_len)})
        return self._reattach_meta(trimmed, meta)

    def _eval_ast_node(self, node: ast.expr, variables: Mapping[str, Any]) -> Any:
        """Recursively evaluate one parsed expression node.

        Parameters
        ----------
        node : ast.expr
            The AST expression node to evaluate.
        variables : mapping of str to Any
            Variable bindings referenced within the expression.

        Returns
        -------
        Any
            The evaluated result of the node.

        Raises
        ------
        ValueError
            If an unsupported operator or unsupported expression node is encountered.
        NameError
            If a variable name cannot be found in ``variables``.
        """
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.MatMult):
                left = self._eval_ast_node(node.left, variables)
                right = self._eval_ast_node(node.right, variables)
                return self.matmul(left, right)

            function = _AST_BINARY_OPS.get(type(node.op))
            if function is None:
                raise ValueError(
                    f"Unsupported operator {type(node.op).__name__!r} in "
                    + "expression."
                )
            left = self._eval_ast_node(node.left, variables)
            right = self._eval_ast_node(node.right, variables)
            return self.apply(function, left, right)

        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            last_val = None
            for val_node in node.values:
                last_val = self._eval_ast_node(val_node, variables)
                if isinstance(last_val, (xr.Dataset, xr.DataArray)):
                    raise TypeError(
                        "evaluate(): 'and'/'or' use Python truth-value "
                        + "checks, which are not defined for xarray "
                        + "Datasets/DataArrays (no single element is "
                        + "'the' truth value of a multi-element array). "
                        + "Use the elementwise bitwise forms instead: '&' "
                        + "for 'and', '|' for 'or', e.g. "
                        + '"(a > 0) & (b < 1)".'
                    )
                if is_and and not last_val:
                    return last_val
                if not is_and and last_val:
                    return last_val
            return last_val

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError(
                    "Chained comparisons (e.g. 'a < b < c') are not "
                    + "supported; write them as separate comparisons."
                )
            function = _AST_COMPARE_OPS.get(type(node.ops[0]))
            if function is None:
                raise ValueError(
                    f"Unsupported comparison {type(node.ops[0]).__name__!r} "
                    + "in expression."
                )
            left = self._eval_ast_node(node.left, variables)
            right = self._eval_ast_node(node.comparators[0], variables)
            return self.apply(function, left, right)

        if isinstance(node, ast.UnaryOp):
            function = _AST_UNARY_OPS.get(type(node.op))
            if function is None:
                raise ValueError(
                    f"Unsupported unary operator {type(node.op).__name__!r} "
                    + "in expression."
                )
            operand = self._eval_ast_node(node.operand, variables)
            return self.apply(function, operand)

        if isinstance(node, ast.Name):
            try:
                return variables[node.id]
            except KeyError:
                raise NameError(
                    f"Name {node.id!r} is not defined; pass it as "
                    + f"evaluate(..., {node.id}=...)."
                ) from None

        if isinstance(node, ast.Constant):
            return node.value

        raise ValueError(
            f"Unsupported expression element {type(node).__name__!r}; "
            + "evaluate() only accepts variable names, numeric literals, "
            + "parentheses, and the arithmetic/comparison/bitwise/boolean "
            + "operators."
        )

    def evaluate(self, expression: str, /, **variables: Any) -> Any:
        """Evaluate a string expression, respecting normal operator precedence.

        Parameters
        ----------
        expression : str
            A Python expression referencing ``variables`` by name, for
            example ``"(a + b) * c - d / e"``.
        **variables : Any
            Values bound to the names used in ``expression``: xarray
            Datasets/DataArrays (distributed or not) or plain scalars.

        Returns
        -------
        Any
            The expression's value. Distribution metadata propagates
            through exactly as it would from the equivalent chain of
            :meth:`apply` calls.

        Raises
        ------
        ValueError
            If ``expression`` fails to parse, uses an unsupported operator
            or expression element, or chains comparisons.
        NameError
            If ``expression`` references a name not present in
            ``variables``.

        Examples
        --------
        >>> mpi.xarray.evaluate("a + b - c", a=ds1, b=ds2, c=ds3)
        >>> mpi.xarray.evaluate("(a + b) * c", a=ds1, b=ds2, c=ds3)
        >>> mpi.xarray.evaluate("anomaly / std", anomaly=a, std=s)
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"Could not parse expression {expression!r}: {exc}"
            ) from exc
        return self._eval_ast_node(tree.body, variables)
