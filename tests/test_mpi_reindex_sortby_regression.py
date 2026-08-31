"""Regression tests for the reviewed patch.

Run under MPI, e.g.::

    mpirun -np 4 --oversubscribe python3 tests/test_mpi_reindex_sortby_regression.py

Uses ``mock_dataset.create_multitype_dataset`` (rank 0 writes one small
NetCDF file spanning float32/float64/int32 plus a shuffled sort key,
every rank barriers, then opens it) rather than building the array in
memory and broadcasting it.

Covers:

* the circular-import fix in ``mpi/runtime.py``, ``mpi/diagnostics.py``,
  ``xarray/io.py`` and ``xarray/netcdf.py`` (this file's own successful
  import is the test);
* the float32-preservation fix in ``xarray/arithmetic.py::_fill_chunk``,
  exercised through ``reindex`` (new, unmatched positions -> filled) and
  ``sortby`` (existing positions only, no fill) under uneven partitions;
* correctness of ``reindex``/``sortby`` against native xarray for both
  ``DataArray`` and ``Dataset``, with a dimension not evenly divisible by
  the rank count.
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mock_dataset import OUTPUT_DIR, create_multitype_dataset

from climtools import mpi
from climtools.xarray.core import unwrap
from climtools.xarray.io import mpi_open_dataset

comm = mpi.comm
rank = comm.rank
size = comm.size

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        failures.append(f"{name}: {detail}")


def gather_full(mpi_obj) -> xr.Dataset | xr.DataArray:
    """Gather a distributed object's dim back to a single-rank native object."""
    local = unwrap(mpi_obj)
    meta = mpi_obj.meta
    dim = meta["dims"] if isinstance(meta["dims"], str) else meta["dims"][0]
    pieces = comm.allgather(local)
    return xr.concat(pieces, dim=dim)


# --------------------------------------------------------------------------
# 1. float32 preservation through reindex() with new (fill-requiring)
#    positions -- this is exactly the case _fill_chunk handles, and the
#    case the uploaded patch regressed (np.asarray(nan).dtype is float64,
#    silently doubling memory for every float32 variable touched).
# --------------------------------------------------------------------------

N = 17  # deliberately not evenly divisible by common rank counts (2, 4, 8)

path = OUTPUT_DIR / "reindex_sortby_regression.nc"
create_multitype_dataset(path, n=N, seed=0)

mds = mpi_open_dataset(str(path), mpi, partition_dim="x", log_partitions=False)
mda = mds["var32"]
native_ds = xr.open_dataset(path)
da32 = native_ds["var32"]

new_labels = np.arange(-3, N + 5)  # extends both ends -> guarantees fill positions

reind_da = mda.reindex(x=new_labels)
reind_ds = mds.reindex(x=new_labels)

full_reind_da = gather_full(reind_da)
full_reind_ds = gather_full(reind_ds)

native_reind_da = da32.reindex(x=new_labels)
native_reind_ds = native_ds.reindex(x=new_labels)

check(
    "reindex DataArray float32 dtype preserved",
    full_reind_da.dtype == np.float32 == native_reind_da.dtype,
    f"got {full_reind_da.dtype}, native {native_reind_da.dtype}",
)
check(
    "reindex Dataset var32 dtype preserved",
    full_reind_ds["var32"].dtype == np.float32 == native_reind_ds["var32"].dtype,
    f"got {full_reind_ds['var32'].dtype}",
)
check(
    "reindex Dataset var64 dtype preserved",
    full_reind_ds["var64"].dtype == np.float64 == native_reind_ds["var64"].dtype,
    f"got {full_reind_ds['var64'].dtype}",
)
check(
    "reindex Dataset varint promotes to float64 under default NaN fill "
    "(matches native: an int dtype cannot hold NaN)",
    full_reind_ds["varint"].dtype == np.float64 == native_reind_ds["varint"].dtype,
    f"got {full_reind_ds['varint'].dtype}",
)
check(
    "reindex DataArray values match native (reordered to common coord order)",
    bool(
        np.allclose(
            full_reind_da.sortby("x").values,
            native_reind_da.sortby("x").values,
            equal_nan=True,
        )
    ),
    "",
)

# Non-default, integer fill_value: must also stay float32, not widen via
# np.asarray(0).dtype (int64) either.
reind_da_fill0 = gather_full(mda.reindex(x=new_labels, fill_value=0))
native_reind_da_fill0 = da32.reindex(x=new_labels, fill_value=0)
check(
    "reindex with integer fill_value=0 keeps float32",
    reind_da_fill0.dtype == np.float32 == native_reind_da_fill0.dtype,
    f"got {reind_da_fill0.dtype}",
)

# --------------------------------------------------------------------------
# 2. float32 preservation through sortby() -- a pure permutation, so it
#    should never touch _fill_chunk at all, but is covered here as the
#    sibling redistribution path added by the same patch. "key" is the
#    mock file's own deterministically shuffled sort column.
# --------------------------------------------------------------------------

sorted_mda = mda.sortby("key")
full_sorted = gather_full(sorted_mda)
native_sorted = da32.sortby("key")

check(
    "sortby preserves float32",
    full_sorted.dtype == np.float32 == native_sorted.dtype,
    f"got {full_sorted.dtype}",
)
check(
    "sortby values match native",
    bool(
        full_sorted.reset_coords(drop=True).equals(
            native_sorted.reset_coords(drop=True)
        )
    ),
    "",
)

# --------------------------------------------------------------------------
# 3. Report
# --------------------------------------------------------------------------

comm.Barrier()
all_failures = comm.allgather(failures)
flat = [f"[rank {r}] {msg}" for r, fs in enumerate(all_failures) for msg in fs]

if rank == 0:
    if flat:
        print(f"FAILED ({len(flat)}):")
        for msg in flat:
            print(" -", msg)
    else:
        print(f"PASSED on {size} ranks, N={N}")

comm.Barrier()
sys.exit(1 if flat else 0)
