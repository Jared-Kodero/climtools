"""MPI-aware alignment and arithmetic for distributed xarray objects."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from .meta import _partitions_match, get_mpi_meta, set_mpi_meta

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping

from collections.abc import Callable

# Complete Binary Operations (+, -, *, /, //, %, **, @, <<, >>, &, |, ^)
_AST_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.MatMult: operator.matmul,  # Matrix multiplication (@)
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
        left : xarray.Dataset or xarray.DataArray
            Left operand to align.
        right : xarray.Dataset or xarray.DataArray
            Right operand to align.
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
        tuple of xarray.Dataset or xarray.DataArray
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
        >>> anomaly = mpi.xarray.apply(operator.sub, left, right)

        >>> left, right = mpi.xarray.align(a_full, b_full, dim="time")
        >>> combined = mpi.xarray.apply(operator.add, left, right)
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
                left, dim, chunk_info=chunk_info, log_partitions=log_partitions
            ),
            self.redistribute(
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
    ) -> dict[str, Any] | None:
        """Return the mpi_meta to attach to a multi-operand call's result.

        Parameters
        ----------
        operands : iterable of Any
            Every positional and keyword argument passed to :meth:`apply`.

        Returns
        -------
        dict[str, Any] or None
            Metadata to reattach to the result, or None when no operand is
            distributed.

        Raises
        ------
        ValueError
            If two operands are distributed over different partitions, or
            if a replicated operand carries the distributed dimension at a
            different length than the partition owns.
        """
        operands = list(operands)
        metas = [self._operand_meta(item) for item in operands]

        meta = next((item for item in metas if item is not None), None)
        if meta is None:
            return None

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
        return meta

    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``func(*args, **kwargs)`` rank-locally, propagating MPI metadata.

        Parameters
        ----------
        func : callable
            Any function of the given ``args``/``kwargs``.
        *args : Any
            Positional arguments to ``func``: xarray Datasets/DataArrays
            (distributed or not) or plain scalars/arrays, in any mix.
        **kwargs : Any
            Keyword arguments to ``func``, checked for distribution
            metadata exactly like ``args``.

        Returns
        -------
        Any
            ``func(*args, **kwargs)``. When any argument was distributed,
            the result is tagged with the same distribution metadata.

        Raises
        ------
        ValueError
            If the xarray arguments are distributed over incompatible
            partitions.

        Examples
        --------
        >>> mpi.xarray.apply(operator.add, a, b)
        >>> mpi.xarray.apply(xr.where, cond, a, b)
        >>> mpi.xarray.apply(lambda x, *, factor: x * factor, a, factor=2.0)
        """
        meta = self._check_operands_distribution((*args, *kwargs.values()))

        self._agree(
            (
                "apply",
                getattr(func, "__name__", repr(func)),
                None if meta is None else (str(meta["dim"]), int(meta["global_size"])),
            )
        )

        result = func(*args, **kwargs)
        return result if meta is None else self._reattach_meta(result, meta)

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
            if isinstance(node.op, ast.And):
                last_val = None
                for val_node in node.values:
                    last_val = self._eval_ast_node(val_node, variables)
                    # Check falsiness (handling xarray objects safely if needed, or normal python truthiness)
                    if not last_val:
                        return last_val
                return last_val
            elif isinstance(node.op, ast.Or):
                last_val = None
                for val_node in node.values:
                    last_val = self._eval_ast_node(val_node, variables)
                    if last_val:
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
