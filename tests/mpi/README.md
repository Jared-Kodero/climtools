# Multi-dimensional MPI partition tests

These test the `partition_dim: str | Sequence[str]` generalization directly
against `mpi4py`, exercising the actual `Create_cart`/`Cart_sub` machinery
rather than mocking it. Each file is a standalone script (not yet wired into
a pytest+mpi harness — the project's existing test conventions weren't
included in the reference material this patch was built from) that asserts
against known-correct values computed independently in NumPy, and exits
non-zero on any rank that fails.

## Running

Each file's docstring gives its exact invocation, e.g.:

```sh
mpirun -n 6 python3 tests/mpi/test_reductions_multidim.py
```

All of them were run and verified against this patch, in both directions:
real 2x3/1x5/2x2 process grids (including deliberately uneven per-axis
splits, and a 5-rank/prime-number grid where `compute_layout` cannot split
evenly along both axes) and, separately, the untouched single-dimension
path (`test_backward_compat_1d.py`), to confirm zero behavior change there.

## What's covered, and how

Each script builds a small `Ops` class combining only the mixins it needs
(e.g. `class Ops(Reduction, Groupby, Statistics)`), attaches `mpi_meta` by
hand via `set_mpi_meta(..., dim=(...), cart=topology.as_meta_cart())`
exactly as `IO.partition()` would, and checks results against a value
computed directly from the known global array — not just internal
self-consistency.

| File | Covers |
|---|---|
| `test_cartesian_topology_smoke.py` | Raw `mpi4py` `Cartcomm.Shift()` semantics this whole design depends on |
| `test_partition_multidim.py` | `IO.partition()`'s N-D scatter; full-array tiling reconstruction |
| `test_reductions_multidim.py` | `sum`/`mean`/`min`/`var`/`groupby_reduce`, including the replicated-axis double-counting hazard |
| `test_finish_partial_reattachment.py` | Reducing away one of several partition axes (new code path, unreachable in 1-D) |
| `test_indexing_multidim.py` | `isel`/`sel`/`isel_scalar` on a *second* partition axis (regression test for a real bug found during development) |
| `test_halo_and_diff_multidim.py` | `halo_exchange()` per axis, and `diff()` end to end |
| `test_backward_compat_1d.py` | The original single-dimension path is behaviorally unchanged |

## Setup note

These were developed and run against a *reference subset* of the `xarray/`
package (the files provided for this task), not the full `climtools` repo,
which also needs `climtools.core.utils` and `climtools.core.progress`
(neither included in the reference material). Minimal stand-ins for both
were used during development and are **not** part of this patch — the real
repository already has them. `mpi4py` 4.1.2 and OpenMPI were used
throughout; per the task's constraints, `mpi4py 4.1.x` was requested and
matched.

One test (`test_partition_multidim.py`) also works around a circular
import that is already present in the untouched reference code
(`xarray/netcdf.py` &rarr; `constructors.py` &rarr; `core.py` &rarr;
`ops.py` &rarr; `io.py` &rarr; back to `netcdf.py`), confirmed to fail
identically against the pristine baseline before this patch. It is not
introduced or worsened by this patch; the real repository's `__init__.py`
(not included in the reference material) presumably resolves it via import
order. The workaround stubs `constructors.py` for that one test only,
since `IO.partition()` never touches it.
