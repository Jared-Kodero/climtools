"""Shared helpers and fixtures for the mpi_test_* correctness suite.

Every mpi_test_*.py module in this directory imports from here rather
than duplicating fixture setup or the RESULTS/record/report machinery.
`mpi_test.py` is the single entry point: it builds the fixtures once via
`build_fixtures()`, imports each mpi_test_*.py module and calls its
`run(fixtures)`, then calls `report()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from climtools import mpi, xgeo
from climtools.xarray.core import MPIXarray
from mock_dataset import _path, create_dataset

RESULTS: list[tuple[str, str, bool | None, str]] = []


def record(op: str, case: str, ok: bool | None, msg: str = "") -> None:
    """Record one check's outcome. `ok=None` means a declared
    NotImplementedError under an unsupported partition shape -- reported
    as SKIP, not FAIL."""
    RESULTS.append((op, case, ok, msg))


def local_of(value):
    """Materialize this rank's local slice, whether MPIXarray or plain xarray."""
    return value._prepare().load() if isinstance(value, MPIXarray) else value


@dataclass(frozen=True)
class Fixtures:
    native: xr.Dataset
    dist: MPIXarray
    dist2d: MPIXarray

    def gsize(self, dim: str) -> int:
        return self.native.sizes[dim]


def build_fixtures() -> Fixtures:
    """Generate the shared mock dataset once and open it two ways: a 1D
    partition (dim='time') and a 2D Cartesian partition (dims=('lat',
    'lon')) -- every test module compares against the same underlying
    data, just partitioned differently."""
    create_dataset(n_time=12, resolution_deg=10, plev_step=-250)
    mpi.comm.barrier()
    native = xr.open_dataset(_path).load()
    dist = xgeo.mpi_open_dataset(_path, mpi, partition_dim="time", log_partitions=False)
    dist2d = xgeo.mpi_open_dataset(
        _path, mpi, partition_dim=("lat", "lon"), log_partitions=False
    )
    return Fixtures(native=native, dist=dist, dist2d=dist2d)


def report() -> None:
    gathered = mpi.comm.gather(RESULTS, root=0)
    if mpi.comm.rank != 0:
        return
    combined: dict[tuple[str, str], list[bool | None]] = {}
    msgs: dict[tuple[str, str], str] = {}
    for rank_results in gathered:
        for op, case, ok, msg in rank_results:
            key = (op, case)
            combined.setdefault(key, []).append(ok)
            if msg:
                msgs[key] = msg
    print(f"\n=== mpi_test results ({mpi.comm.size} ranks) ===")
    n_pass = n_skip = n_fail = 0
    for (op, case), oks in combined.items():
        if all(o is None for o in oks):
            status = "SKIP"
            n_skip += 1
        elif all(o for o in oks if o is not None) and all(o is not None for o in oks):
            status = "PASS"
            n_pass += 1
        else:
            status = "FAIL"
            n_fail += 1
        print(f"[{status}] {op:<18} {case:<20} {msgs.get((op, case), '')}")
    print(f"--- {n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"(declared NotImplementedError under an unsupported partition shape) "
          f"of {len(combined)} checks ---")
    if n_fail:
        raise SystemExit(1)
