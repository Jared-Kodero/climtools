"""Correctness: prod(), resample(), and a 2D Cartesian (multi-dim) partition.

    mpirun -np <N> --oversubscribe python3 tests/test_mpi_prod_resample_cartesian.py

Uses three of the shared mock builders in ``mock_dataset.py``:
``create_multitype_dataset`` (prod), ``create_timeseries_dataset``
(resample -- a real ``datetime64`` coordinate, needed to exercise the
resample bin-alignment fix), and ``create_grid_dataset`` (the 2D
Cartesian partition). Each builds and writes its own small NetCDF file.
The prod/resample cases open theirs directly via ``mpi_open_dataset``;
the 2D case opens on rank 0 and distributes via ``mpi_partition_data``
instead, since ``mpi_open_dataset``'s parallel-open path only supports a
single ``partition_dim`` -- multi-dim Cartesian partitioning is only
available on an already-loaded object.
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mock_dataset import (
    OUTPUT_DIR,
    create_grid_dataset,
    create_multitype_dataset,
    create_timeseries_dataset,
)

from climtools import mpi
from climtools.xarray.core import unwrap
from climtools.xarray.io import mpi_open_dataset, mpi_partition_data

comm = mpi.comm
rank = comm.rank
size = comm.size
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        failures.append(f"{name}: {detail}")


def close(a, b, **kw):
    return bool(np.allclose(np.asarray(a), np.asarray(b), equal_nan=True, **kw))


def gather_1d(mpi_obj):
    """Reconstruct the full array along a still-active 1D partition dim.

    Robust to Cartesian replication: when only one of several partition
    axes remains active (e.g. after reducing over another axis), more
    than one rank can hold an identical replica of the same slice, so a
    naive ``allgather`` + ``concat`` would double-count it. Deduplicating
    by each rank's own ``(start, stop)`` bounds before concatenating
    fixes that (replicas of the same slice are identical, so keeping
    just one is correct).
    """
    local = unwrap(mpi_obj)
    meta = mpi_obj.meta
    if meta is None:
        return local
    dim = meta["dims"] if isinstance(meta["dims"], str) else meta["dims"][0]
    start, stop = meta["starts"][dim], meta["stops"][dim]
    pieces = comm.allgather((start, stop, local))
    by_bounds = {(s0, s1): piece for s0, s1, piece in pieces}
    ordered = [by_bounds[key] for key in sorted(by_bounds)]
    return xr.concat(ordered, dim=dim)


# --------------------------------------------------------------------------
# prod -- values transformed to stay near 1 (identically on both the
# distributed and native copies) so the product doesn't over/underflow
# float32; "var32" itself (a signed, smooth-but-zero-crossing signal) is
# unsuitable for a product directly.
# --------------------------------------------------------------------------
N = 11
path = OUTPUT_DIR / "prod.nc"
create_multitype_dataset(path, n=N, seed=3)

mda = mpi_open_dataset(str(path), mpi, partition_dim="x", log_partitions=False)["var32"]
native_da = xr.open_dataset(path)["var32"]


def to_near_one(da):
    return (0.9 + 0.2 / (1.0 + np.exp(-da))).astype(np.float32)


mda = mda.apply(to_near_one, mda)
native_da = to_near_one(native_da)

# prod isn't wrapped on MPIXarray directly; call the engine the same way
# core.py's other reduction wrappers do (reducing the sole dim here
# collapses to a replicated scalar, so no gather is needed).
prod_result = mda._ops.prod(unwrap(mda), "x")
native_prod = native_da.prod("x")
check("prod values", close(prod_result.values, native_prod.values, rtol=1e-3))
check("prod dtype", prod_result.dtype == native_prod.dtype, f"{prod_result.dtype} vs {native_prod.dtype}")

# --------------------------------------------------------------------------
# resample -- a real datetime64 coordinate at an intentionally awkward,
# non-bin-aligned start time, the exact shape of case that exposed the
# resample() bin-alignment bug (see xarray/groupby.py::_resample_bin_labels).
# --------------------------------------------------------------------------
ts_path = OUTPUT_DIR / "resample.nc"
create_timeseries_dataset(ts_path, n=40, freq="D", start="2020-01-01")

mda_t = mpi_open_dataset(str(ts_path), mpi, partition_dim="time", log_partitions=False)["v"]
native_da_t = xr.open_dataset(ts_path)["v"]

for freq in ("7D", "3D", "MS"):
    r = gather_1d(mda_t.resample("time", freq).mean())
    n = native_da_t.resample(time=freq).mean()
    check(
        f"resample {freq} mean values",
        close(r.values, n.values, rtol=1e-3),
        f"shapes {r.shape} vs {n.shape}",
    )
    check(
        f"resample {freq} bin count matches native "
        "(regression: per-rank pandas.resample(origin=...) silently "
        "ignores `origin` for Day-or-coarser frequencies and re-anchors "
        "on each rank's own local first timestamp, fragmenting bins at "
        "partition boundaries -- see _resample_bin_labels)",
        r.shape == n.shape,
        f"{r.shape} vs {n.shape}",
    )

# --------------------------------------------------------------------------
# 2D Cartesian multi-dim partition
# --------------------------------------------------------------------------
grid_path = OUTPUT_DIR / "grid.nc"
N_A, N_B = 13, 9  # both uneven against small rank counts
create_grid_dataset(grid_path, n_a=N_A, n_b=N_B)

# mpi_open_dataset's parallel-open path only supports a single partition_dim
# ("Hashable or 'auto'", not a sequence); multi-dim Cartesian partitioning
# is only available via mpi_partition_data on an already-loaded object, so
# rank 0 opens the (small) mock file directly and distributes it from there.
native_ds2 = xr.open_dataset(grid_path)
grid_ds = native_ds2 if rank == 0 else None
mds2 = mpi_partition_data(grid_ds, mpi, dim=["a", "b"], root=0)

local2 = unwrap(mds2)
meta2 = mds2.meta
starts = meta2["starts"]
stops = meta2["stops"]
check(
    "2D partition: local slab matches native slice at this rank's bounds",
    close(
        local2["v"].values,
        native_ds2["v"].isel(a=slice(starts["a"], stops["a"]), b=slice(starts["b"], stops["b"])).values,
    ),
    f"a=[{starts['a']}:{stops['a']}) b=[{starts['b']}:{stops['b']})",
)

# every element covered exactly once across ranks
counts = np.zeros((N_A, N_B), dtype=np.int64)
all_bounds = comm.allgather((starts["a"], stops["a"], starts["b"], stops["b"]))
for a0, a1, b0, b1 in all_bounds:
    counts[a0:a1, b0:b1] += 1
check(
    "2D partition: every global element covered exactly once",
    bool((counts == 1).all()),
    f"min={counts.min()} max={counts.max()}",
)

# a 2D reduction over one Cartesian axis
r2 = gather_1d(mds2.sum("a"))  # reduces the 'a' axis; 'b' remains partitioned
n2 = native_ds2.sum("a")
check(
    "2D partition: sum over one Cartesian axis matches native",
    close(r2["v"].values, n2["v"].values, rtol=1e-3),
    f"shapes {r2['v'].shape} vs {n2['v'].shape}",
)

comm.Barrier()
all_failures = comm.allgather(failures)
flat = [f"[rank {r}] {msg}" for r, fs in enumerate(all_failures) for msg in fs]
if rank == 0:
    if flat:
        print(f"FAILED ({len(flat)}):")
        for msg in flat:
            print(" -", msg)
    else:
        print(f"PASSED on {size} ranks")
comm.Barrier()
sys.exit(1 if flat else 0)
