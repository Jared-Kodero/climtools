"""Correctness tests for :mod:`climtools.xarray.indexing` (``IndexingMixin``).

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi_indexing.py
"""

from __future__ import annotations

import numpy as np
from climtools import mpi
from mpi_fixtures import check, finish, make_dataset, make_series


def test_isel_scalar() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.redistribute(full, "t")
    got = mpi.xarray.isel(distributed, t=7)
    check(
        "isel: scalar global index matches source",
        np.isclose(float(got), float(full.isel(t=7))),
    )


def test_isel_slice() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.redistribute(full, "t")
    got = mpi.xarray.isel(distributed, t=slice(5, 20))
    ref = full.isel(t=slice(5, 20)).sel(t=got["t"])
    check(
        "isel: slice over global bounds matches source at the same coordinates",
        bool((got.values == ref.values).all()),
    )


def test_sel_scalar() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.redistribute(full, "t")
    label = full["t"].values[10]
    got = mpi.xarray.sel(distributed, t=label)
    check(
        "sel: scalar global label matches source",
        np.isclose(float(got), float(full.sel(t=label))),
    )


def test_sel_slice() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.redistribute(full, "t")
    lo, hi = full["t"].values[3], full["t"].values[18]
    got = mpi.xarray.sel(distributed, t=slice(lo, hi))
    ref = full.sel(t=slice(lo, hi)).sel(t=got["t"])
    check(
        "sel: label slice over global bounds matches source at the same coordinates",
        bool((got.values == ref.values).all()),
    )


def test_isel_scalar_dataset() -> None:
    ds = make_dataset(n=20)
    distributed = mpi.xarray.redistribute(ds, "t")
    got = mpi.xarray.isel(distributed, t=4)
    check(
        "isel dataset: scalar index matches source (time-varying var)",
        np.isclose(float(got["v"].isel(y=0, x=0)), float(ds["v"].isel(t=4, y=0, x=0))),
    )
    check(
        "isel dataset: static variable untouched by a t-scalar select",
        bool((got["s"].values == ds["s"].values).all()),
    )


if __name__ == "__main__":
    test_isel_scalar()
    test_isel_slice()
    test_sel_scalar()
    test_sel_slice()
    test_isel_scalar_dataset()
    finish()
