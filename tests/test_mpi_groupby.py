"""Correctness tests for :mod:`climtools.xarray.groupby` (``GroupbyMixin``).

These exercise the case the rest of the ``xarray.mpi`` collectives don't:
group boundaries that cross MPI rank boundaries (e.g. resampling a time
dimension that is itself the active partition dimension).

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi_groupby.py
"""

from __future__ import annotations

import numpy as np
from climtools import mpi
from mpi_fixtures import check, finish, make_dataset, make_field


def test_resample_reduce_dataarray() -> None:
    full = make_field(n=400, ny=2, nx=3, nan_at=((5, 0, 0), (399, 1, 2)))
    distributed = mpi.xarray.redistribute(full, "t")

    for op in ("sum", "mean", "count", "min", "max"):
        got = mpi.xarray.resample_reduce(
            distributed, "t", "D", op=op, skipna=True, redistribute_on=None
        )
        if op == "count":
            ref = full.resample(t="D").count()
        else:
            ref = getattr(full.resample(t="D"), op)(skipna=True)
        got_sorted = got.sortby("_mpi_group")
        ref_sorted = ref.sortby("t")
        check(
            f"resample_reduce({op}): matches serial xarray.resample",
            np.allclose(got_sorted.values, ref_sorted.values, equal_nan=True),
        )


def test_resample_reduce_dataset() -> None:
    ds = make_dataset(n=300, ny=2, nx=2, seed=11)
    distributed = mpi.xarray.redistribute(ds, "t")
    got = mpi.xarray.resample_reduce(
        distributed, "t", "D", op="mean", skipna=True, redistribute_on=None
    )
    ref = ds["v"].resample(t="D").mean(skipna=True)
    got_sorted = got["v"].sortby("_mpi_group")
    ref_sorted = ref.sortby("t")
    check(
        "resample_reduce dataset: time-varying variable matches serial reference",
        np.allclose(got_sorted.values, ref_sorted.values, equal_nan=True),
    )
    check(
        "resample_reduce dataset: static variable passes through unreduced",
        bool((got["s"].values == ds["s"].values).all()),
    )


def test_groupby_reduce_categorical() -> None:
    """Grouping is not required to be time-based: any label array works,
    including one where group membership does not respect rank boundaries."""
    full = make_field(n=200, ny=2, nx=2, seed=3)
    distributed = mpi.xarray.redistribute(full, "t")

    global_labels = np.array(["a", "b"] * (full.sizes["t"] // 2))
    local_index = np.searchsorted(full["t"].values, distributed["t"].values)
    local_labels = global_labels[local_index]

    got = mpi.xarray.groupby_reduce(
        distributed, "t", local_labels, op="mean", skipna=True, redistribute_on=None
    )
    group_da = full["t"].copy(data=global_labels).rename("g")
    ref = full.groupby(group_da).mean("t", skipna=True)
    got_sorted = got.sortby("_mpi_group")
    ref_sorted = ref.sortby("g")
    check(
        "groupby_reduce: categorical grouping matches serial xarray.groupby",
        np.allclose(got_sorted.values, ref_sorted.values, equal_nan=True),
    )


def test_groupby_reduce_on_non_partition_dim() -> None:
    """Grouping a dimension other than the active partition dimension stays
    local -- and a variable lacking that dimension passes through."""
    ds = make_dataset(n=12, ny=4, nx=2, seed=6)
    distributed = mpi.xarray.redistribute(ds, "t")
    labels = np.array(["even", "odd"] * (ds.sizes["y"] // 2))

    got = mpi.xarray.groupby_reduce(
        distributed, "y", labels, op="mean", skipna=True, redistribute_on=None
    )
    group_da = ds["y"].copy(data=labels).rename("g")
    ref = distributed["v"].groupby(group_da).mean("y", skipna=True)
    check(
        "groupby_reduce over non-partition dim: matches plain xarray on the local shard",
        np.allclose(got["v"].sortby("_mpi_group").values, ref.sortby("g").values),
    )
    check(
        "groupby_reduce over non-partition dim: static var untouched",
        "s" in got.data_vars,
    )


if __name__ == "__main__":
    test_resample_reduce_dataarray()
    test_resample_reduce_dataset()
    test_groupby_reduce_categorical()
    test_groupby_reduce_on_non_partition_dim()
    finish()
