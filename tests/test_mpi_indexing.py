"""Correctness tests for :mod:`climtools.xarray.indexing` (``IndexingMixin``).

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi_indexing.py
"""

from __future__ import annotations

import numpy as np
from climtools import mpi
from climtools.xarray.meta import get_mpi_meta
from mpi_fixtures import check, finish, make_dataset, make_field, make_series


def test_isel_scalar() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.repartition(full, "t")
    got = mpi.xarray.isel(distributed, t=7)
    check(
        "isel: scalar global index matches source",
        np.isclose(float(got), float(full.isel(t=7))),
    )


def test_isel_slice() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.repartition(full, "t")
    got = mpi.xarray.isel(distributed, t=slice(5, 20))
    ref = full.isel(t=slice(5, 20)).sel(t=got["t"])
    check(
        "isel: slice over global bounds matches source at the same coordinates",
        bool((got.values == ref.values).all()),
    )


def test_sel_scalar() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.repartition(full, "t")
    label = full["t"].values[10]
    got = mpi.xarray.sel(distributed, t=label)
    check(
        "sel: scalar global label matches source",
        np.isclose(float(got), float(full.sel(t=label))),
    )


def test_sel_slice() -> None:
    full = make_series(n=30)
    distributed = mpi.xarray.repartition(full, "t")
    lo, hi = full["t"].values[3], full["t"].values[18]
    got = mpi.xarray.sel(distributed, t=slice(lo, hi))
    ref = full.sel(t=slice(lo, hi)).sel(t=got["t"])
    check(
        "sel: label slice over global bounds matches source at the same coordinates",
        bool((got.values == ref.values).all()),
    )


def test_isel_scalar_dataset() -> None:
    ds = make_dataset(n=20)
    distributed = mpi.xarray.repartition(ds, "t")
    got = mpi.xarray.isel(distributed, t=4)
    check(
        "isel dataset: scalar index matches source (time-varying var)",
        np.isclose(float(got["v"].isel(y=0, x=0)), float(ds["v"].isel(t=4, y=0, x=0))),
    )
    check(
        "isel dataset: static variable untouched by a t-scalar select",
        bool((got["s"].values == ds["s"].values).all()),
    )


def test_isel_slice_singleton_default_no_repartition() -> None:
    """Without ``partition_dim``, a singleton slice still stays put on t."""
    full = make_series(n=12)
    distributed = mpi.xarray.repartition(full, "t")
    got = mpi.xarray.isel(distributed, t=slice(4, 5))
    meta = get_mpi_meta(got)
    check("isel singleton default: stays partitioned on t", meta["dim"] == "t")
    check("isel singleton default: global size is 1", meta["global_size"] == 1)
    total = mpi.comm.allreduce(got.sizes["t"])
    check("isel singleton default: exactly one element across all ranks", total == 1)
    check(
        "isel singleton default: owning rank's value matches source",
        got.sizes["t"] == 0 or np.isclose(float(got.isel(t=0)), float(full.isel(t=4))),
    )


def test_isel_slice_singleton_repartition_auto() -> None:
    """``partition_dim='auto'`` scatters the singleton onto the largest dim."""
    full = make_field(n=17, ny=3, nx=5, seed=1)
    distributed = mpi.xarray.repartition(full, "t")
    got = mpi.xarray.isel(distributed, t=slice(6, 7), partition_dim="auto")
    meta = get_mpi_meta(got)
    check(
        "isel singleton auto: repartitions onto x, the larger surviving dim",
        meta["dim"] == "x",
    )
    check(
        "isel singleton auto: global size on x matches source",
        meta["global_size"] == full.sizes["x"],
    )
    check(
        "isel singleton auto: t keeps its single global element locally",
        got.sizes["t"] == 1,
    )

    total_x = mpi.comm.allreduce(got.sizes["x"])
    check(
        "isel singleton auto: partition sizes sum to global x size",
        total_x == full.sizes["x"],
    )

    ref = full.isel(t=6).sel(x=got["x"])
    check(
        "isel singleton auto: local x-slice matches source at the same coordinates",
        bool((got.isel(t=0).values == ref.values).all()),
    )


def test_isel_slice_singleton_repartition_named() -> None:
    """A named ``partition_dim`` dimension is honored over the largest one."""
    full = make_field(n=17, ny=3, nx=5, seed=2)
    distributed = mpi.xarray.repartition(full, "t")
    got = mpi.xarray.isel(distributed, t=slice(0, 1), partition_dim="y")
    meta = get_mpi_meta(got)
    check(
        "isel singleton named: repartitions onto the requested dim", meta["dim"] == "y"
    )
    total_y = mpi.comm.allreduce(got.sizes["y"])
    check(
        "isel singleton named: partition sizes sum to global y size",
        total_y == full.sizes["y"],
    )
    ref = full.isel(t=0).sel(y=got["y"])
    check(
        "isel singleton named: local y-slice matches source at the same coordinates",
        bool((got.isel(t=0).values == ref.values).all()),
    )


def test_sel_slice_singleton_repartition_auto() -> None:
    full = make_field(n=17, ny=3, nx=5, seed=3)
    distributed = mpi.xarray.repartition(full, "t")
    label = full["t"].values[9]
    got = mpi.xarray.sel(distributed, t=slice(label, label), partition_dim="auto")
    meta = get_mpi_meta(got)
    check("sel singleton auto: repartitions onto x", meta["dim"] == "x")
    total_x = mpi.comm.allreduce(got.sizes["x"])
    check(
        "sel singleton auto: partition sizes sum to global x size",
        total_x == full.sizes["x"],
    )
    ref = full.sel(t=label).sel(x=got["x"])
    check(
        "sel singleton auto: local x-slice matches source at the same coordinates",
        bool((got.isel(t=0).values == ref.values).all()),
    )


def test_isel_slice_singleton_repartition_auto_noop() -> None:
    """``'auto'`` is a no-op when no other dimension is worth spreading."""
    full = make_series(n=12)
    distributed = mpi.xarray.repartition(full, "t")
    got = mpi.xarray.isel(distributed, t=slice(3, 4), partition_dim="auto")
    meta = get_mpi_meta(got)
    check(
        "isel singleton auto no-op: falls back to old_dim with nothing to spread onto",
        meta["dim"] == "t" and meta["global_size"] == 1,
    )


def test_isel_slice_singleton_repartition_dataset() -> None:
    ds = make_dataset(n=17, ny=3, nx=5, seed=4)
    distributed = mpi.xarray.repartition(ds, "t")
    got = mpi.xarray.isel(distributed, t=slice(2, 3), partition_dim="auto")
    meta = get_mpi_meta(got)
    check("isel dataset singleton auto: repartitions onto x", meta["dim"] == "x")
    total_x = mpi.comm.allreduce(got.sizes["x"])
    check(
        "isel dataset singleton auto: partition sizes sum to global x size",
        total_x == ds.sizes["x"],
    )
    ref_v = ds["v"].isel(t=2).sel(x=got["x"])
    check(
        "isel dataset singleton auto: time-varying var matches source",
        bool((got["v"].isel(t=0).values == ref_v.values).all()),
    )
    ref_s = ds["s"].sel(x=got["x"])
    check(
        "isel dataset singleton auto: static var repartitioned onto x too",
        bool((got["s"].values == ref_s.values).all()),
    )


def test_isel_slice_singleton_repartition_invalid_dim() -> None:
    full = make_field(n=17, ny=3, nx=5, seed=5)
    distributed = mpi.xarray.repartition(full, "t")
    raised = False
    try:
        mpi.xarray.isel(distributed, t=slice(0, 1), partition_dim="not_a_dim")
    except ValueError:
        raised = True
    check("isel singleton: invalid partition_dim raises ValueError", raised)


if __name__ == "__main__":
    test_isel_scalar()
    test_isel_slice()
    test_sel_scalar()
    test_sel_slice()
    test_isel_scalar_dataset()
    test_isel_slice_singleton_default_no_repartition()
    test_isel_slice_singleton_repartition_auto()
    test_isel_slice_singleton_repartition_named()
    test_sel_slice_singleton_repartition_auto()
    test_isel_slice_singleton_repartition_auto_noop()
    test_isel_slice_singleton_repartition_dataset()
    test_isel_slice_singleton_repartition_invalid_dim()
    finish()
