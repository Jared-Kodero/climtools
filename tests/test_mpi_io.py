"""Correctness tests for :mod:`climtools.xarray.io` (``IOMixin``).

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi_io.py
"""

from __future__ import annotations

import numpy as np
from climtools import mpi
from climtools.xarray.meta import get_mpi_meta
from mpi_fixtures import RANK, check, finish, make_dataset, make_field


def test_repartition_dataarray() -> None:
    """Each rank's local slice matches the source at the same coordinates,
    and partition sizes sum back to the global size."""
    full = make_field(n=37)  # deliberately awkward w.r.t. common rank counts
    distributed = mpi.xarray.repartition(full, "t")
    meta = get_mpi_meta(distributed)
    check("repartition: attaches mpi_meta", meta is not None)
    check("repartition: partition dim is t", meta["dim"] == "t")

    local_ref = full.sel(t=distributed["t"])
    check(
        "repartition: local shard matches source at the same coordinates",
        bool((distributed == local_ref).all()),
    )
    total = mpi.comm.allreduce(distributed.sizes["t"])
    check("repartition: partition sizes sum to global size", total == full.sizes["t"])


def test_repartition_rejects_already_distributed() -> None:
    full = make_field(n=20)
    distributed = mpi.xarray.repartition(full, "t")
    try:
        mpi.xarray.repartition(distributed, "t")
        raised = False
    except ValueError:
        raised = True
    check("repartition: rejects an already-distributed object", raised)


def test_repartition_dataset() -> None:
    """A static (non-partitioned) variable survives repartition unchanged."""
    ds = make_dataset(n=25)
    distributed = mpi.xarray.repartition(ds, "t")
    local_ref = ds["v"].sel(t=distributed["t"])
    check(
        "repartition dataset: time-varying variable matches source",
        bool((distributed["v"] == local_ref).all()),
    )
    check(
        "repartition dataset: static variable is untouched",
        bool((distributed["s"] == ds["s"]).all()),
    )


def test_create_dataarray_fill_receives_global_bounds() -> None:
    """The fill callback's (start, stop) are this rank's own global-index
    bounds along the partitioned dimension."""

    def fill(start: int, stop: int) -> np.ndarray:
        return np.arange(start, stop, dtype=np.float64)

    arr = mpi.xarray.create_dataarray(fill, ["t"], shape={"t": 41}, dim="t")
    meta = get_mpi_meta(arr)
    expected = np.arange(meta["start"], meta["stop"], dtype=np.float64)
    check(
        "create_dataarray: fill values match (start, stop)",
        bool((arr.values == expected).all()),
    )

    total = mpi.comm.allreduce(arr.sizes["t"])
    check("create_dataarray: partition sizes sum to requested global size", total == 41)


def test_create_dataset_multiple_variables() -> None:
    def fill_v(start: int, stop: int) -> np.ndarray:
        return np.full(stop - start, float(RANK))

    ds = mpi.xarray.create_dataset(
        {"v": (["t"], fill_v), "s": (["y"], lambda: np.zeros(3))},
        sizes={"t": 30, "y": 3},
        dim="t",
    )
    check(
        "create_dataset: partitioned variable filled with this rank's id",
        bool((ds["v"].values == float(RANK)).all()),
    )
    check(
        "create_dataset: unpartitioned variable has full local length",
        ds.sizes["y"] == 3,
    )


if __name__ == "__main__":
    test_repartition_dataarray()
    test_repartition_rejects_already_distributed()
    test_repartition_dataset()
    test_create_dataarray_fill_receives_global_bounds()
    test_create_dataset_multiple_variables()
    finish()
