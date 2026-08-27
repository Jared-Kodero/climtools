"""Correctness tests for :mod:`climtools.xarray.reductions` (``ReductionMixin``).

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi_reductions.py
"""

from __future__ import annotations

import numpy as np
from climtools import mpi
from mpi_fixtures import check, finish, make_dataset, make_field


def test_numeric_reductions() -> None:
    """sum/prod/mean/min/max against a serial xarray reference, with NaNs
    including an all-missing column."""
    full = make_field(n=19, ny=2, nx=4, nan_at=((0, 0, 0), (1, 0, 0), (18, 1, 1)))
    full = full.copy()
    full.values[:, :, 2] = np.nan  # one fully missing column
    distributed = mpi.xarray.redistribute(full, "t")

    for op in ("sum", "prod", "mean", "min", "max"):
        got = getattr(mpi.xarray, op)(
            distributed, dim="t", skipna=True, redistribute_on=None
        )
        ref = getattr(full, op)(dim="t", skipna=True)
        check(
            f"{op}: matches serial reference (NaNs, empty column)",
            np.allclose(got.values, ref.values, equal_nan=True),
        )


def test_any_all() -> None:
    full = make_field(n=15, ny=2, nx=2) > 0
    distributed = mpi.xarray.redistribute(full, "t")
    got_any = mpi.xarray.any(distributed, dim="t", redistribute_on=None)
    got_all = mpi.xarray.all(distributed, dim="t", redistribute_on=None)
    check(
        "any: matches serial reference",
        bool((got_any.values == full.any(dim="t").values).all()),
    )
    check(
        "all: matches serial reference",
        bool((got_all.values == full.all(dim="t").values).all()),
    )


def test_first_last() -> None:
    full = make_field(n=21, ny=2, nx=3, nan_at=((0, 0, 0), (1, 0, 0), (-1, 1, 2)))
    full = full.copy()
    full.values[:, 0, 1] = np.nan  # one fully missing column
    distributed = mpi.xarray.redistribute(full, "t")

    got_first = mpi.xarray.first(distributed, "t", skipna=True, redistribute_on=None)
    got_last = mpi.xarray.last(distributed, "t", skipna=True, redistribute_on=None)
    ref_first = full.bfill("t").isel(t=0)
    ref_last = full.ffill("t").isel(t=-1)
    check(
        "first: matches bfill-based serial reference",
        np.allclose(got_first.values, ref_first.values, equal_nan=True),
    )
    check(
        "last: matches ffill-based serial reference",
        np.allclose(got_last.values, ref_last.values, equal_nan=True),
    )


def test_reduction_on_non_partition_dim() -> None:
    """Reducing a dimension other than the active partition dimension stays
    a local, per-rank computation and still returns a correct result."""
    full = make_field(n=16, ny=2, nx=5)
    distributed = mpi.xarray.redistribute(full, "t")
    got = mpi.xarray.mean(distributed, dim="x", skipna=True, redistribute_on=None)
    ref = distributed.mean(dim="x", skipna=True)
    check(
        "mean over non-partition dim: matches plain xarray on the local shard",
        bool((got.values == ref.values).all()),
    )


def test_dataset_reduction_with_static_variable() -> None:
    ds = make_dataset(n=18, ny=2, nx=3)
    distributed = mpi.xarray.redistribute(ds, "t")
    got = mpi.xarray.mean(distributed, dim="t", skipna=True, redistribute_on=None)
    ref = ds["v"].mean(dim="t", skipna=True)
    check(
        "dataset mean: time-varying variable matches serial reference",
        np.allclose(got["v"].values, ref.values),
    )
    check(
        "dataset mean: static variable passes through unreduced",
        bool((got["s"].values == ds["s"].values).all()),
    )


if __name__ == "__main__":
    test_numeric_reductions()
    test_any_all()
    test_first_last()
    test_reduction_on_non_partition_dim()
    test_dataset_reduction_with_static_variable()
    finish()
