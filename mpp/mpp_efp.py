"""Reproducible reductions in extended fixed point.

Mirrors FMS ``mpp/mpp_efp.F90``. A floating sum depends on the order the
terms arrive in, so a distributed sum would change with the rank count.
Values are accumulated as exact integer digits instead, making the result
independent of how the data was decomposed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from collections.abc import Sequence

_NUMBIT = 46


_NUMINT = 6


_PREC = float(2**_NUMBIT)


MAX_EFP_RANKS = 2 ** (63 - _NUMBIT) - 1


_SCALES = np.array([_PREC ** (2 - n) for n in range(_NUMINT)], dtype=np.float64)


_PREC_INT = 1 << _NUMBIT


_EFP_BLOCK = 1 << 16


MAX_PROD_RANKS = 1000


_PROD_BLOCK = 512


def _carry_overflow(digits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Renormalise digits so each holds less than one unit of the next scale.

    FMS ``carry_overflow``. Without this the running accumulator overflows
    int64 silently once enough terms have been added, which corrupts the sum
    rather than reporting it.
    """
    for n in range(_NUMINT - 1, 0, -1):
        # Truncate toward zero, exactly, without going through float64.
        carry = np.sign(digits[n]) * (np.abs(digits[n]) >> _NUMBIT)
        digits[n] -= carry * _PREC_INT
        digits[n - 1] += carry
    return digits


def _to_digits(array: np.ndarray[Any, Any], axis: int) -> np.ndarray[Any, Any]:
    """Sum values into signed integer digits along ``axis``.

    Accumulates in blocks of :data:`_EFP_BLOCK` terms, renormalising after
    each, so an arbitrarily long local axis cannot overflow the accumulator.
    """
    values = np.moveaxis(np.asarray(array, dtype=np.float64), axis, 0)
    digits = np.zeros((_NUMINT, *values.shape[1:]), dtype=np.int64)
    for start in range(0, values.shape[0], _EFP_BLOCK):
        block = values[start : start + _EFP_BLOCK]
        sign = np.where(block < 0.0, -1.0, 1.0)
        residual = np.abs(block)
        for n, scale in enumerate(_SCALES):
            digit = np.floor(residual / scale)
            digits[n] += (sign * digit).astype(np.int64).sum(axis=0)
            residual -= digit * scale
        _carry_overflow(digits)
    return digits


def _from_digits(digits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Inverse of :func:`_to_digits`, summing smallest scale first."""
    total = np.zeros(digits.shape[1:], dtype=np.float64)
    for n in range(_NUMINT - 1, -1, -1):
        total += digits[n].astype(np.float64) * _SCALES[n]
    return total


def mpp_reproducing_sum(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm,
    *,
    axis: int | None = None,
) -> np.ndarray[Any, Any]:
    """Compute a rank-count-invariant distributed sum.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    axis : int or None, optional
        Local reduction axis before the global sum.

    Returns
    -------
    numpy.ndarray
        Reproducible global sum.

    Raises
    ------
    ValueError
        If the rank count is unsupported, the input contains non-finite
        values, or the values are too large to sum without overflow.
    """
    if comm.size > MAX_EFP_RANKS:
        # Every rank sees the same communicator size, so this raises on all of
        # them or none: safe to do before a collective.
        raise ValueError(f"mpp_reproducing_sum supports at most {MAX_EFP_RANKS} ranks.")

    flat = np.asarray(local).reshape(-1) if axis is None else local
    reduce_axis = 0 if axis is None else axis

    # FMS `prec_error`: the top digit must stay small enough that summing it
    # across every rank still fits in int64. Checking the inputs rather than
    # the converted digits keeps the conversion itself from overflowing.
    prec_error = (2**63 - 1) // comm.size
    finite = bool(np.all(np.isfinite(flat)))
    representable = finite and bool(
        np.all(np.abs(flat) < prec_error * _SCALES[0] / _EFP_BLOCK)
    )

    if representable:
        digits = _to_digits(flat, reduce_axis)
    else:
        # Contribute zeros so the collective keeps its shape and every rank
        # still reaches the Allreduce that carries the flags.
        shape = np.moveaxis(np.asarray(flat), reduce_axis, 0).shape[1:]
        digits = np.zeros((_NUMINT, *shape), dtype=np.int64)

    # Whether a rank holds non-finite or over-large input is a rank-local
    # fact, so raising on it directly would let one rank leave while the
    # others waited in the Allreduce below, deadlocking the job over a data
    # error. Both flags ride along in the reduction instead.
    payload = np.empty(digits.size + 2, dtype=np.int64)
    payload[:-2] = digits.reshape(-1)
    payload[-2] = 0 if finite else 1
    payload[-1] = (
        0 if representable and not np.any(np.abs(digits[0]) > prec_error) else 1
    )
    total = np.empty_like(payload)
    comm.Allreduce(payload, total, op=MPI.SUM)
    if total[-2]:
        raise ValueError(
            f"mpp_reproducing_sum requires finite input; {int(total[-2])} of "
            f"{comm.size} ranks hold NaN or infinity."
        )
    if total[-1]:
        raise ValueError(
            f"mpp_reproducing_sum overflowed on {int(total[-1])} of "
            f"{comm.size} ranks; the values are too large to sum reproducibly."
        )
    return _from_digits(total[:-2].reshape(digits.shape))


#: Index of each companion field carried alongside a product mantissa.
PROD_EXPONENT, PROD_NAN, PROD_INF, PROD_ZERO, PROD_NEGATIVE = range(5)
_PROD_FIELDS = 5


def _moved_to_front(
    values: np.ndarray[Any, Any], axes: Sequence[int]
) -> np.ndarray[Any, Any]:
    """Collapse ``axes`` into a single leading axis, preserving the rest."""
    ordered = tuple(a % values.ndim for a in axes)
    moved = np.moveaxis(values, ordered, range(len(ordered)))
    kept = moved.shape[len(ordered) :]
    return moved.reshape((-1, *kept))


def mpp_prod_decompose(
    local: np.ndarray[Any, Any],
    axes: int | Sequence[int],
) -> np.ndarray[Any, Any]:
    """Decompose a rank-local product into exactly summable integer fields.

    A floating product depends on the order and grouping of its factors, so
    multiplying rank-local partials would make the result vary with the rank
    count. Every factor is instead split by ``frexp`` into a power of two and
    a mantissa in ``[0.5, 1)``; the exponents sum exactly as integers, and the
    mantissas are carried as ``log2`` values summed in extended fixed point.
    Both are order-free, so the product is reproducible.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    axes : int or sequence of int
        Local axes to reduce.

    Returns
    -------
    numpy.ndarray
        Integer fields: exponent, the NaN/infinity/zero/negative tallies, and
        the extended-fixed-point digits of the summed mantissa logarithms.
        Every field reduces with ``SUM``.
    """
    axis_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
    work = _moved_to_front(np.asarray(local).astype(np.float64, copy=False), axis_tuple)

    is_nan = np.isnan(work)
    is_inf = np.isinf(work)
    is_zero = work == 0.0
    ordinary = ~(is_nan | is_inf | is_zero)
    # Non-ordinary factors contribute 1.0, whose frexp is (0.5, 1): the
    # log2 of -1 and the exponent of +1 cancel, leaving the product untouched.
    magnitude = np.where(ordinary, np.abs(work), 1.0)

    mantissa, exponent = np.frexp(magnitude)
    fields = np.empty((_PROD_FIELDS + _NUMINT, *work.shape[1:]), dtype=np.int64)
    fields[PROD_EXPONENT] = exponent.sum(axis=0, dtype=np.int64)
    fields[PROD_NAN] = np.count_nonzero(is_nan, axis=0)
    fields[PROD_INF] = np.count_nonzero(is_inf, axis=0)
    fields[PROD_ZERO] = np.count_nonzero(is_zero, axis=0)
    fields[PROD_NEGATIVE] = np.count_nonzero(np.signbit(work) & ~is_nan, axis=0)
    fields[_PROD_FIELDS:] = _to_digits(np.log2(mantissa), 0)
    return fields


def mpp_prod_recombine(
    fields: np.ndarray[Any, Any],
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Rebuild a product from the reduced fields of :func:`mpp_prod_decompose`.

    Parameters
    ----------
    fields : numpy.ndarray
        Globally summed exponent, exception tallies and mantissa-log digits.
    dtype : numpy.dtype, optional
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Reconstructed product with signed zero/infinity and NaN handling.
    """
    n_nan = fields[PROD_NAN]
    n_inf = fields[PROD_INF]
    n_zero = fields[PROD_ZERO]
    sign = np.where(fields[PROD_NEGATIVE] % 2 == 1, -1.0, 1.0)

    # Split the summed logarithm into a whole power of two and a remainder in
    # [0, 1), so the whole part joins the exact integer exponent and only the
    # remainder goes through exp2.
    log_mantissa = _from_digits(fields[_PROD_FIELDS:])
    whole = np.floor(log_mantissa)
    mantissa = np.exp2(log_mantissa - whole)

    # ldexp takes a C int exponent; clipping is safe because anything beyond
    # this range has already saturated the float64 result either way.
    exponent = np.clip(
        fields[PROD_EXPONENT] + whole.astype(np.int64), -32768, 32768
    ).astype(np.int32)
    with np.errstate(over="ignore"):
        result = sign * np.ldexp(mantissa, exponent)

    result = np.where(n_zero > 0, sign * 0.0, result)
    result = np.where(n_inf > 0, sign * np.inf, result)
    result = np.where((n_zero > 0) & (n_inf > 0), np.nan, result)
    result = np.where(n_nan > 0, np.nan, result)
    return result if dtype is None else result.astype(dtype, copy=False)


def mpp_reproducing_prod(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm,
    *,
    axis: int | Sequence[int] = 0,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Compute a rank-count-invariant distributed product.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local values.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    axis : int or sequence of int, default 0
        Local reduction axes.
    dtype : numpy.dtype, optional
        Output dtype.

    Returns
    -------
    numpy.ndarray
        Reproducible global product.

    Raises
    ------
    ValueError
        If the communicator exceeds the supported rank limit.
    """
    if comm.size > MAX_PROD_RANKS:
        raise ValueError(
            f"mpp_reproducing_prod supports at most {MAX_PROD_RANKS} ranks."
        )
    values = np.asarray(local)
    out_dtype = np.dtype(dtype) if dtype is not None else values.dtype
    fields = mpp_prod_decompose(values, axis)
    reduced = np.empty_like(fields)
    comm.Allreduce(np.ascontiguousarray(fields), reduced, op=MPI.SUM)
    return mpp_prod_recombine(reduced, out_dtype)
