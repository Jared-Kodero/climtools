"""Shared fixtures and assertions for the ``xarray.mpi`` test modules.

Not a test module itself (no ``test_`` prefix). Each ``test_mpi_*.py`` module
run under ``mpirun -n N python -m mpi4py tests/test_mpi_*.py`` imports from
here: deterministic data builders (same seed on every rank, so a fresh
process on every rank builds bit-identical global arrays without any
communication) and a small ``check``/``finish`` pair that aggregates
pass/fail across ranks and exits nonzero on any failure, for use with
:mod:`tests/test.sh`-style CI loops.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr
from mpi4py import MPI

from climtools import mpi

RANK: int = mpi.comm.rank
SIZE: int = mpi.comm.size

_failures: list[str] = []


def check(name: str, condition: bool) -> None:
    """Record a named pass/fail. Failures print immediately, from every rank
    that hit them; passes print once, from rank 0."""
    if not condition:
        _failures.append(name)
        print(f"[rank {RANK}] FAIL: {name}", flush=True)
    elif RANK == 0:
        print(f"PASS: {name}", flush=True)


def finish() -> None:
    """Aggregate failures across every rank and exit(1) if any occurred.

    Call once, after every ``check()`` in a test module's ``__main__`` block.
    Every rank must reach this call (it does a collective), matching the
    error-propagation convention used throughout ``xarray.mpi`` itself."""
    total = mpi.comm.allreduce(len(_failures), op=MPI.SUM)
    mpi.comm.barrier()
    if RANK == 0:
        label = "FAILED" if total else "passed"
        print(f"=== {len(_failures)} local / {total} total check(s) {label} ===")
    if total:
        sys.exit(1)


def make_series(
    n: int = 24, *, seed: int = 0, nan_at: tuple[int, ...] = ()
) -> xr.DataArray:
    """Deterministic hourly time series, bit-identical on every rank."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    data = rng.normal(size=n)
    for i in nan_at:
        data[i] = np.nan
    return xr.DataArray(data, dims="t", coords={"t": times}, name="v")


def make_field(
    n: int = 24,
    ny: int = 3,
    nx: int = 4,
    *,
    seed: int = 0,
    nan_at: tuple[tuple[int, int, int], ...] = (),
) -> xr.DataArray:
    """Deterministic (t, y, x) field, bit-identical on every rank."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    data = rng.normal(size=(n, ny, nx))
    for index in nan_at:
        data[index] = np.nan
    return xr.DataArray(
        data,
        dims=("t", "y", "x"),
        coords={"t": times, "y": np.arange(ny), "x": np.arange(nx)},
        name="v",
    )


def make_dataset(n: int = 24, ny: int = 3, nx: int = 4, *, seed: int = 0) -> xr.Dataset:
    """Deterministic Dataset: one time-varying field, one static field."""
    field = make_field(n, ny, nx, seed=seed)
    rng = np.random.default_rng(seed + 1)
    static = xr.DataArray(
        rng.normal(size=(ny, nx)),
        dims=("y", "x"),
        coords={"y": field["y"], "x": field["x"]},
        name="s",
    )
    return xr.Dataset({"v": field, "s": static})
