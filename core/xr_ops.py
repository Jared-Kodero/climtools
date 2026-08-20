"""MPI-aware alignment and arithmetic for distributed xarray objects.

Split out of ``xarray_mpi.py``: :meth:`_ArithmeticMixin.align`,
:meth:`~_ArithmeticMixin.apply`, and :meth:`~_ArithmeticMixin.evaluate` form
a self-contained layer on top of the redistribution and reduction engine
defined there. The coupling back to that engine is exactly two methods —
``redistribute`` (used by :meth:`align`) and ``_agree`` (used by
:meth:`apply`) — both defined on
:class:`~climtools.core.xarray_mpi.XarrayMPI`, which mixes this class in.
"""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from .xr_meta import _partitions_match, get_mpi_meta, set_mpi_meta

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

# String tokens accepted by apply()/evaluate() as an alternative to passing
# a callable directly. Every value is a plain function of two positional
# arguments so user-supplied callables (operator.add, numpy ufuncs, lambdas,
# climtools.operator helpers) drop in identically.
_STRING_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
}

# Node-type-to-callable tables used by evaluate()'s ast walk. Kept alongside
# _STRING_OPERATORS because they cover the same operator set through a
# different key (an ast node type rather than a string token), not because
# the two are independent: every function referenced below also appears
# above.
_AST_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_AST_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_AST_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
}


class ArithmeticMixin:
    """Alignment and arithmetic methods mixed into ``XarrayMPI``.

    Assumes the host class provides ``self.redistribute`` (used by
    :meth:`align`) and ``self._agree`` (used by :meth:`apply`); both are
    defined on :class:`~climtools.core.xarray_mpi.XarrayMPI`.
    """

    # -- alignment ----------------------------------------------------------
    #
    # A convenience counterpart to xarray's own align(): instead of
    # reconciling coordinate labels/indexes, this reconciles which rank
    # owns which slice, so left and right end up combinable by apply()/
    # evaluate() with zero MPI traffic. Two cases resolve without any
    # communication and are handled here: a replicated operand sliced down
    # onto an already-distributed partner's exact bounds, and two
    # replicated operands independently redistributed along the same
    # dimension (which is deterministic given (length, chunk size, rank,
    # size), so both land on identical bounds without needing to compare
    # notes). Two operands already distributed on genuinely different
    # partitions are a data-movement problem (Alltoallv), which this
    # function does not attempt; it raises instead of guessing.

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
        meta : dict
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

        Three cases are handled, all without moving data between ranks:

        - Neither operand is distributed: returned unchanged, unless
          ``dim`` is given, in which case both are distributed along
          ``dim`` via ``redistribute``. Redistribution is a deterministic
          function of (length, chunk size, rank, size), so two operands of
          the same length along ``dim`` land on identical per-rank bounds
          without any coordination between them.
        - One operand is already distributed, the other is replicated at
          the full length of the distributed dimension (or does not carry
          that dimension at all): the replicated operand is sliced down to
          the distributed operand's exact bounds.
        - Both are already distributed identically: returned unchanged.

        Parameters
        ----------
        left, right : xarray.Dataset or xarray.DataArray
            Operands to align.
        dim : hashable or {"auto"}, optional
            Dimension to distribute both operands along when neither is
            currently distributed. Required in that case; ignored
            otherwise, since an already-distributed operand's dimension
            takes precedence.
        chunk_info : mapping, optional
            Forwarded to ``redistribute`` when neither operand is yet
            distributed.
        log_partitions : bool, optional
            Forwarded to ``redistribute`` when neither operand is yet
            distributed.

        Returns
        -------
        tuple of (xarray.Dataset or xarray.DataArray)
            ``(left, right)``, each carrying matching distribution
            metadata (or neither carrying any, if both remain replicated).

        Raises
        ------
        ValueError
            If both operands are already distributed but on different
            partitions (different dimension, global size, or per-rank
            bounds), or if neither is distributed and ``dim`` is omitted.

        Examples
        --------
        >>> left, right = mpi.xarray.align(local_field, full_climatology)
        >>> anomaly = mpi.xarray.apply(left, "-", right)

        >>> left, right = mpi.xarray.align(a_full, b_full, dim="time")
        >>> combined = mpi.xarray.apply(left, "+", right)
        """
        left_meta = self._operand_meta(left)
        right_meta = self._operand_meta(right)

        if left_meta is not None and right_meta is not None:
            if _partitions_match(left_meta, right_meta):
                return left, right
            raise ValueError(
                "Cannot align operands already distributed over different "
                + f"partitions: left dim={left_meta['dim']!r} "
                + f"range=[{left_meta['start']}:{left_meta['stop']}) "
                + f"vs right dim={right_meta['dim']!r} "
                + f"range=[{right_meta['start']}:{right_meta['stop']}). "
                + "Reconciling different existing partitions requires "
                + "moving data between ranks, which align() does not do; "
                + "rebuild one operand from a replicated source with "
                + "mpi.xarray.redistribute using the other's dimension and "
                + "chunk_info instead."
            )

        if left_meta is not None:
            return left, self._align_replicated(right, left_meta, partner=left)

        if right_meta is not None:
            return self._align_replicated(left, right_meta, partner=right), right

        if dim is None:
            return left, right

        # Neither operand is distributed yet: both are redistributed
        # independently below, which is a deterministic function of
        # (length, chunk size, rank, size) and therefore lands both on
        # identical per-rank *positions* without any coordination. That
        # guarantee says nothing about whether position i on the left
        # actually corresponds to the same physical coordinate as position
        # i on the right. xarray.align(..., join="exact") checks the
        # coordinate labels along dim while both operands are still fully
        # replicated (a local, communication-free check, since every rank
        # holds the same complete data), and raises before any rank commits
        # to a position-based split that would otherwise silently combine
        # mismatched slices later in apply()/evaluate().
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
            self.redistribute(
                left,
                dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
            self.redistribute(
                right,
                dim,
                chunk_info=chunk_info,
                log_partitions=log_partitions,
            ),
        )

    # -- arithmetic -----------------------------------------------------------
    #
    # Binary operators never communicate. Two operands are combinable
    # without any MPI traffic exactly when every rank already holds
    # matching, aligned local slices: either neither side is distributed,
    # or both are distributed identically (same dimension, same global
    # size, same per-rank start/stop). _check_binary_distribution verifies
    # that before the op runs, because xarray's default coordinate-based
    # alignment would otherwise intersect or silently reshape mismatched
    # partitions instead of raising, turning a partitioning bug into a
    # wrong answer instead of an error. align() is the fix-up step for
    # operands this check rejects.

    @staticmethod
    def _operand_meta(operand: Any) -> dict[str, Any] | None:
        """Return ``operand``'s MPI distribution metadata, if any."""
        if isinstance(operand, (xr.Dataset, xr.DataArray)):
            return get_mpi_meta(operand)
        return None

    @staticmethod
    def _reattach_meta(result: Any, meta: dict[str, Any]) -> Any:
        """Tag ``result`` with ``meta`` if it is an xarray object.

        Shared by :meth:`align`, :meth:`apply`, and the ``evaluate`` unary
        operator handler: each computes a plain local elementwise result
        and needs to relabel it with the distribution metadata that
        justified computing it locally in the first place.
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

    def _check_binary_distribution(
        self,
        left: Any,
        right: Any,
    ) -> dict[str, Any] | None:
        """Return the mpi_meta to attach to a binary op's result.

        Parameters
        ----------
        left, right : Any
            Operands of a binary operator. Only :class:`xarray.Dataset` and
            :class:`xarray.DataArray` operands are inspected for
            distribution metadata; anything else (a scalar, a plain NumPy
            array) is treated as replicated.

        Returns
        -------
        dict or None
            Metadata to reattach to the result, or None when neither
            operand is distributed.

        Raises
        ------
        ValueError
            If both operands are distributed over different partitions, or
            if one operand is distributed and the other carries the same
            dimension at a different length.
        """
        left_meta = self._operand_meta(left)
        right_meta = self._operand_meta(right)

        if left_meta is None and right_meta is None:
            return None

        if left_meta is not None and right_meta is not None:
            if not _partitions_match(left_meta, right_meta):
                raise ValueError(
                    "Cannot combine operands distributed over different "
                    + f"partitions: left dim={left_meta['dim']!r} "
                    + f"range=[{left_meta['start']}:{left_meta['stop']}) "
                    + f"vs right dim={right_meta['dim']!r} "
                    + f"range=[{right_meta['start']}:{right_meta['stop']}). "
                    + "Call mpi.xarray.align(left, right) first."
                )
            return left_meta

        meta = left_meta if left_meta is not None else right_meta
        other = right if left_meta is not None else left
        assert meta is not None
        dim = meta["dim"]
        if isinstance(other, (xr.Dataset, xr.DataArray)) and dim in other.dims:
            owned = meta["stop"] - meta["start"]
            local = int(other.sizes[dim])
            if local != owned:
                raise ValueError(
                    f"Operand carries dimension {dim!r} at length {local}, "
                    + "which does not match this rank's owned partition "
                    + f"length {owned}. Call mpi.xarray.align(left, right) "
                    + "first."
                )
        return meta

    def apply(
        self,
        left: Any,
        op: str | Callable[[Any, Any], Any],
        right: Any,
    ) -> Any:
        """Combine two operands with a rank-local, MPI-aware binary operator.

        No data moves between ranks. When both operands are xarray objects
        distributed identically, or when only one operand is distributed
        and the other is replicated (or lacks the distributed dimension),
        the operator runs directly on each rank's own local slice and the
        result carries the same distribution metadata forward, ready for
        further arithmetic or a collective reduction. Mismatched
        partitions raise instead of silently combining misaligned data;
        call :meth:`align` first to reconcile them.

        Parameters
        ----------
        left : Any
            Left operand: an xarray Dataset/DataArray (distributed or not)
            or a plain scalar/array.
        op : str or callable
            Either a string token (``"+"``, ``"-"``, ``"*"``, ``"/"``,
            ``"//"``, ``"%"``, ``"**"``, ``"=="``, ``"!="``, ``"<"``,
            ``"<="``, ``">"``, ``">="``, ``"&"``, ``"|"``, ``"^"``) or a
            two-argument callable such as :func:`operator.add`.
        right : Any
            Right operand, same accepted types as ``left``.

        Returns
        -------
        Any
            The elementwise result. When either operand was distributed,
            the result is tagged with the same distribution metadata.

        Raises
        ------
        ValueError
            If the operands are distributed over incompatible partitions.
        TypeError
            If ``op`` is neither a recognized string token nor callable.

        Examples
        --------
        >>> mpi.xarray.apply(a, "+", b)
        >>> mpi.xarray.apply(a, operator.sub, b)
        """
        if isinstance(op, str):
            try:
                function = _STRING_OPERATORS[op]
            except KeyError:
                raise ValueError(
                    f"Unknown operator {op!r}. Supported string operators: "
                    + ", ".join(sorted(_STRING_OPERATORS))
                ) from None
        elif callable(op):
            function = op
        else:
            raise TypeError(
                "Operator must be a string (e.g. '+') or a two-argument "
                + f"callable (e.g. operator.add), got {type(op).__name__}."
            )

        meta = self._check_binary_distribution(left, right)

        # A cheap allgather that only checks every rank reached the same
        # compatibility outcome, so a rank-dependent bug surfaces here as
        # an immediate, diagnosable exception instead of a later collective
        # deadlocking on operands that silently diverged.
        self._agree(
            (
                "apply",
                op if isinstance(op, str) else getattr(op, "__name__", repr(op)),
                None if meta is None else (str(meta["dim"]), int(meta["global_size"])),
            )
        )

        result = function(left, right)
        return result if meta is None else self._reattach_meta(result, meta)

    def _eval_ast_node(self, node: ast.expr, variables: Mapping[str, Any]) -> Any:
        """Recursively evaluate one parsed expression node.

        Every accepted construct (binary op, comparison, unary op, name,
        constant) is handled inline below; anything else — calls,
        attribute access, subscripts, comprehensions, lambdas, ... — falls
        through to the final ``raise`` unevaluated. This is a whitelist,
        not a blacklist, so it stays safe as Python's grammar grows rather
        than needing to track new dangerous constructs.
        """
        if isinstance(node, ast.BinOp):
            function = _AST_BINARY_OPS.get(type(node.op))
            if function is None:
                raise ValueError(
                    f"Unsupported operator {type(node.op).__name__!r} in "
                    + "expression."
                )
            left = self._eval_ast_node(node.left, variables)
            right = self._eval_ast_node(node.right, variables)
            return self.apply(left, function, right)

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
            return self.apply(left, function, right)

        if isinstance(node, ast.UnaryOp):
            function = _AST_UNARY_OPS.get(type(node.op))
            if function is None:
                raise ValueError(
                    f"Unsupported unary operator {type(node.op).__name__!r} "
                    + "in expression."
                )
            operand = self._eval_ast_node(node.operand, variables)
            meta = self._operand_meta(operand)
            result = function(operand)
            return result if meta is None else self._reattach_meta(result, meta)

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
            + "parentheses, and the arithmetic/comparison/bitwise "
            + "operators."
        )

    def evaluate(self, expression: str, /, **variables: Any) -> Any:
        """Evaluate a string expression, respecting normal operator precedence.

        Variables named in ``expression`` are looked up in ``variables``,
        the same binding style as :meth:`pandas.DataFrame.query`. Unlike
        the strictly left-to-right chaining an explicit sequence of
        :meth:`apply` calls would give, standard Python operator precedence
        applies (``*``/`/`` before ``+``/``-``, right-associative ``**``,
        and so on), and parentheses group explicitly, exactly as they would
        in an ordinary Python expression.

        The expression is parsed with :mod:`ast` rather than through
        pandas' query engine: pandas' engine is built to resolve column
        names against a DataFrame's own namespace and dispatches through
        numexpr or its own evaluator, neither of which is designed to
        accept xarray Datasets/DataArrays as operands or to carry
        climtools's own distribution metadata through the computation.
        Parsing with the standard library instead needs no extra
        dependency, gets Python's exact, well-documented precedence rules
        for free from the grammar, and lets every operator still route
        through :meth:`apply`, so the same distribution-compatibility
        checks and metadata propagation apply here as to a single
        :meth:`apply` call. Only a small, explicit whitelist of node types
        is evaluated (names, literals, unary/binary/comparison operators);
        anything else, including attribute access, subscripts, and calls,
        raises rather than executing.

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
