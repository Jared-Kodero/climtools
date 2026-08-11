"""Extensions to Python's :mod:`operator` module.

This module re-exports the standard operators from :mod:`operator` and adds
general mathematical transformations commonly used with scientific data.

Import this module instead of :mod:`operator` when the extended operator
interface is required.
"""

from operator import *
from typing import Any


def identity(x: Any, *_, **__) -> Any:
    r"""Return the input unchanged.

    Implements the identity transformation

        F(x) = x.

    Parameters
    ----------
    x
        Input value or array-like object.
    *_
        Additional positional arguments, accepted and ignored.
    **__
        Additional keyword arguments, accepted and ignored.

    Returns
    -------
    Any
        The input object unchanged.
    """
    return x


def affine(x: Any, coefficients: tuple[float, float]) -> Any:
    r"""Apply an affine transformation.

    Implements

        F(x) = mx + b,

    where ``m`` is the scale factor and ``b`` is the offset.

    Parameters
    ----------
    x
        Input value or array-like object.
    coefficients
        Two-element tuple ``(m, b)`` containing the scale factor and offset.

    Returns
    -------
    Any
        Affine-transformed input.
    """
    m, b = coefficients
    return m * x + b


def quadratic(
    x: Any,
    coefficients: tuple[float, float, float],
) -> Any:
    r"""Evaluate a quadratic polynomial.

    Implements

        F(x) = ax^2 + bx + c.

    Parameters
    ----------
    x
        Input value or array-like object.
    coefficients
        Three-element tuple ``(a, b, c)`` containing the quadratic, linear,
        and constant coefficients.

    Returns
    -------
    Any
        Quadratic polynomial evaluated at ``x``.
    """
    a, b, c = coefficients
    return a * x**2 + b * x + c


def power(x: Any, exponent: float) -> Any:
    r"""Raise the input to a specified power.

    Implements

        F(x) = x^p,

    where ``p`` is the exponent.

    Parameters
    ----------
    x
        Input value or array-like object.
    exponent
        Exponent ``p``.

    Returns
    -------
    Any
        Input raised elementwise to ``exponent``.
    """
    return x**exponent


def reciprocal(x: Any, *_, **__) -> Any:
    r"""Return the multiplicative inverse.

    Implements

        F(x) = 1 / x.

    Parameters
    ----------
    x
        Input value or array-like object.
    *_
        Additional positional arguments, accepted and ignored.
    **__
        Additional keyword arguments, accepted and ignored.

    Returns
    -------
    Any
        Elementwise reciprocal of the input.
    """
    return 1 / x
