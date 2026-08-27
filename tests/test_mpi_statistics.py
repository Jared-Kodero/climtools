"""Correctness tests for :mod:`climtools.xarray.statistics` (``StatisticsMixin``).

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi_statistics.py
"""

from __future__ import annotations

import numpy as np
from mpi_fixtures import check, finish, make_dataset, make_field

from climtools import mpi


def test_var_std_ddof0() -> None:
    full = make_field(n=23, ny=2, nx=3, nan_at=((2, 0, 0),))
    distributed = mpi.xarray.redistribute(full, "t")
    got_var = mpi.xarray.var(
        distributed, dim="t", skipna=True, ddof=0, redistribute_on=None
    )
    got_std = mpi.xarray.std(
        distributed, dim="t", skipna=True, ddof=0, redistribute_on=None
    )
    ref_var = full.var(dim="t", skipna=True, ddof=0)
    ref_std = full.std(dim="t", skipna=True, ddof=0)
    check(
        "var (ddof=0): matches serial reference",
        np.allclose(got_var.values, ref_var.values),
    )
    check(
        "std (ddof=0): matches serial reference",
        np.allclose(got_std.values, ref_std.values),
    )
    check("std == sqrt(var)", np.allclose(got_std.values, np.sqrt(got_var.values)))


def test_var_std_ddof1() -> None:
    full = make_field(n=17, ny=2, nx=2, seed=5)
    distributed = mpi.xarray.redistribute(full, "t")
    got_var = mpi.xarray.var(
        distributed, dim="t", skipna=True, ddof=1, redistribute_on=None
    )
    ref_var = full.var(dim="t", skipna=True, ddof=1)
    check(
        "var (ddof=1): matches serial reference",
        np.allclose(got_var.values, ref_var.values),
    )


def test_var_on_non_partition_dim() -> None:
    full = make_field(n=14, ny=3, nx=2, seed=9)
    distributed = mpi.xarray.redistribute(full, "t")
    got = mpi.xarray.var(distributed, dim="y", skipna=True, redistribute_on=None)
    ref = distributed.var(dim="y", skipna=True)
    check(
        "var over non-partition dim: matches plain xarray on the local shard",
        np.allclose(got.values, ref.values),
    )


def test_dataset_var() -> None:
    ds = make_dataset(n=20, ny=2, nx=3, seed=2)
    distributed = mpi.xarray.redistribute(ds, "t")
    got = mpi.xarray.var(
        distributed, dim="t", skipna=True, ddof=1, redistribute_on=None
    )
    ref = ds["v"].var(dim="t", skipna=True, ddof=1)
    check(
        "dataset var: time-varying variable matches serial reference",
        np.allclose(got["v"].values, ref.values),
    )
    check(
        "dataset var: static variable passes through unreduced",
        bool((got["s"].values == ds["s"].values).all()),
    )


if __name__ == "__main__":
    test_var_std_ddof0()
    test_var_std_ddof1()
    test_var_on_non_partition_dim()
    test_dataset_var()
    finish()
