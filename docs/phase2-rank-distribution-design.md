# Phase 2 design: Dask-based rank distribution and save_chunks wiring

**Status update (this revision):** the `mpi_meta["save_chunks"]`
mechanism described in section 4 below as a proposal has since been
implemented: `compute_save_chunks` in `core/xr_chunks.py`,
`set_save_chunks` in `core/xr_meta.py`, `XarrayMPI.attach_save_chunks` in
`core/xr_mpi.py`, wired into `lib_netcdf/parallel.py`'s already-distributed
collective write branch. That also fixes the specific bug flagged in
section 4's last paragraph: rank 0's `get_chunks(ds=local_ds, ...)` call
was chunking against its own local, 1/mpi_size-sized slice instead of the
array's true global shape, and never applied the HDF5 4 GiB cap on that
code path at all. None of this has been executed under real MPI -- see
the chat response accompanying this revision for exactly what
"implemented" does and does not mean here.

Section 1-3 (Dask `normalize_chunks`-based irregular-chunk rank
distribution) is unchanged from the previous revision and still not
implemented. Concretely: `get_effective_chunk_size` returns a single
`int`, `chunk_info` is a `dict[str, int]` (one scalar length per
dimension), and `get_chunk_bounds` takes one scalar `chunk_size` -- the
entire distribution_chunks representation, everywhere in
`core/xr_chunks.py` and `core/xr_mpi.py`, is a uniform per-dimension
chunk length. Nothing in the current code accepts or produces an
irregular per-dimension chunk tuple such as dask's own `(1000, 1000,
240)` for a single axis; the `tuple[int, ...]` return types that do exist
(`get_chunks`, `compute_save_chunks`) hold one scalar length *per
dimension of the array* (matching NetCDF's own `chunksizes` convention,
which is likewise always one fixed length per dimension), not multiple
block sizes within one dimension. Support for that kind of irregular,
dask-native chunk tuple -- and the corresponding
`get_chunk_bounds_from_chunks`-style grouping algorithm needed to assign
such chunks to ranks without splitting one -- remains exactly what
section 1 below describes: a proposal.

Status: design only for sections 1-3, implemented-but-untested for
section 4. This phase changes collective-I/O and rank-boundary logic in
`core/xr_mpi.py` and
`lib_netcdf/parallel.py`, which the repository's own test suite
(`tests/test_mpi.py`) exercises only under `mpirun` with a parallel
HDF5/netCDF4/mpi4py build. Neither is available in this environment, so
this section is a specification for implementation and testing on a real
MPI cluster, not a patch.

## 1. Current rank-distribution algorithm

`get_chunk_bounds(length, chunk_size, rank, size)` in `core/xr_chunks.py`
already does the core of what is asked for:

- If the dimension has fewer usable chunks than ranks
  (`chunk_count < min(length, size)`), fall back to
  `get_balanced_bounds`, a plain `divmod` split with no chunk structure to
  respect.
- Otherwise, chunks are distributed round-robin across ranks
  (`quotient, remainder = divmod(chunk_count, size)`) and a rank's bounds
  are always `[first_chunk * chunk_size, (first_chunk + local_chunks) *
  chunk_size)`, i.e. always a whole number of chunks. This already
  satisfies "rank boundaries align with chunk boundaries so sub-chunks do
  not straddle ranks."

`chunk_size` itself comes from `get_effective_chunk_size`, which prefers
the *native on-disk chunk* (`get_native_chunk_sizes`, read from
`encoding["chunksizes"]`/`encoding["preferred_chunks"]`) when it is
"useful" (`get_usable_native_chunk`: more than one chunk fits), and falls
back to `ceil(length / mpi_size)` otherwise.

What this does not yet do, which the brief asks for:

- It never consults `dask.array.core.normalize_chunks`, so it cannot
  reconcile a dataset already opened as **dask-chunked** (not just
  netCDF-chunked) with the rank split, and it has no way to combine a
  *dask* chunking scheme with a *native on-disk* chunking scheme when they
  disagree.
- It treats "distribution constraints" as a single scalar chunk size per
  dimension. Real dask chunk tuples are irregular (`(1000, 1000, 240)`,
  not `(1000, 1000, 1000)`), and forcing irregular chunks onto a uniform
  `chunk_size` either wastes the last, smaller chunk's boundary or, worse,
  silently ignores it.

## 2. Proposed algorithm

Add to `core/xr_chunks.py` (keeps the module's existing scope: pure
functions, no MPI calls, `mpi_size`/`rank` passed explicitly):

```python
def normalize_distribution_chunks(
    length: int,
    native_chunks: tuple[int, ...] | int | None,
    mpi_size: int,
) -> tuple[int, ...]:
    """Return an irregular distribution_chunks tuple summing to length.

    Wraps dask.array.core.normalize_chunks so the same auto-chunking
    dask itself uses for da.chunk("auto") is the basis for the split,
    rather than a hand-rolled ceil(length / mpi_size):

    - native_chunks=None: ask dask for its own "auto" 1-D chunking of
      an array of this length (dask picks a byte-size-driven chunk,
      identical in spirit to today's ceil(length / mpi_size) but aware
      of dask's own default target block size).
    - native_chunks=int: a uniform on-disk chunk size, as today.
    - native_chunks=tuple: an existing irregular dask/on-disk chunking
      (e.g. read from a zarr array's .chunks or an already-dask-backed
      DataArray's .chunksizes) is normalized as-is via
      dask.array.core.normalize_chunks(native_chunks, shape=(length,)),
      which validates it sums to length and fills in "auto"/-1 markers.

    The result is *not* yet re-grouped for mpi_size ranks; that is
    get_chunk_bounds_from_chunks's job below. Keeping normalization and
    rank-grouping as two functions mirrors the existing
    get_effective_chunk_size / get_chunk_bounds split and keeps this
    function testable against dask's behavior alone, with no rank
    concept involved.
    """
```

```python
def get_chunk_bounds_from_chunks(
    chunks: tuple[int, ...],
    rank: int,
    size: int,
) -> tuple[int, int]:
    """Group an irregular chunks tuple into size contiguous rank spans.

    Generalizes get_chunk_bounds (which assumes one uniform chunk_size)
    to an arbitrary dask-style chunks tuple. Chunk *boundaries* (the
    cumulative sums) are the only valid rank boundaries -- a rank's span
    is always a contiguous run of whole chunks, so this still guarantees
    "rank boundaries align with chunk boundaries" for irregular chunks,
    not only uniform ones.

    Algorithm: greedily assign chunks to ranks by cumulative byte-count-
    free length (chunk count, not chunk size, matching the existing
    get_chunk_bounds precedent of balancing by chunk count when
    chunk_count >= size), i.e. divmod(len(chunks), size) the same way
    get_chunk_bounds does today, but summing actual (possibly unequal)
    chunk lengths for each rank's span instead of multiplying by a
    constant chunk_size.
    """
```

`get_chunk_info`/`get_chunk_bounds` keep their existing signatures and
behavior for the common case (uniform native or auto chunk) so nothing
that already calls them breaks; the irregular path is additive, reached
only when a caller has an actual irregular `chunks` tuple to pass (e.g.
`XarrayMPI.open_dataset` when `use_mfdataset=True` opens a dask-backed,
non-uniformly-chunked multi-file dataset -- currently `get_chunk_info`
only ever sees `chunks=None`-opened metadata, so this path does not
change today's `open_dataset` behavior until a caller opts in).

## 3. I/O-performance and rank-0-avoidance requirements

The two-phase plan in `XarrayMPI.open_dataset` (rank 0 opens metadata
only with `chunks=None` and no data touched, computes the plan, broadcasts
it, then every rank opens/slices independently) already avoids
concentrating I/O on rank 0 for the read path -- confirmed by reading
`core/xr_mpi.py:540-620` in this repository. This is worth stating
explicitly because it means Phase 2 is a refinement of the *chunking
math* feeding that plan, not a redesign of the read path's rank-0
avoidance, which is already correct.

## 4. save_chunks wiring (implemented this revision)

`core/xr_chunks.py`'s `compute_save_chunks(value, meta)` derives
save_chunks directly from `mpi_meta`, without gathering any other rank's
data: each variable's true global shape is reconstructed from
`meta["global_size"]` (partition axis) plus the local object's own
already-full shape (every other axis), a lazy `dask.array.zeros` of that
shape is chunked with dask's `"auto"` heuristic, and the partition axis
is forced to `meta["chunk_info"][dim]` capped to the HDF5 4 GiB limit via
`_cap_partition_chunk_to_hdf5_limit`. `core/xr_meta.py::set_save_chunks`
attaches the result under `mpi_meta["save_chunks"]`.
`XarrayMPI.attach_save_chunks` wraps that in the plan-on-rank-0/bcast
pattern the user asked for and every other planning step in this module
already uses.

`lib_netcdf/parallel.py::to_netcdf_parallel`'s already-distributed branch
now calls `mpi.xarray.attach_save_chunks(local_ds)` collectively (only
when the caller did not pass explicit `chunks`) before its rank-0-only
schema-construction step, and that step now reads `chunk_map =
local_meta["save_chunks"]` instead of calling `get_chunks(ds=local_ds,
...)` directly. This is a real correctness fix, not only a refactor:
`get_chunks`'s `da.chunk("auto")` call was previously running against
rank 0's own local slice -- 1/mpi_size of the true data volume along the
partition dimension -- so its byte-size-driven auto-chunk decision for
every *other* dimension was made from the wrong total size, and the
partition-dimension chunk length it substituted
(`local_meta["chunk_info"].get(partition_dim, ...)`) was never checked
against the HDF5 4 GiB limit at all on this code path (unlike the
rank-0-owned write path, which already went through
`get_partition_chunk_size`'s cap).

The rank-0-owned (non-distributed) write branch is intentionally
untouched: it already holds the real, complete dataset on rank 0, so
`get_chunks`'s `da.chunk("auto")` there already sees the true global
shape and needs no mock array.

`lib_netcdf/serial.py` has not been checked in this pass for an
equivalent chunk-shape decision to align with the same helper.

This wiring has not been executed: `dask.array.zeros(..., chunks="auto")`'s
chunk-proposal heuristic has been exercised directly in this sandbox (both
at realistic sizes and deliberately extreme ones, to confirm it stays
lazy -- see the chat response accompanying this revision), but the
resulting save_chunks have never been compared against a real written
file's actual on-disk chunk shapes, and the MPI collective sequence
(`attach_save_chunks`'s internal `raise_if_error` + `bcast`, called from
inside an already-collective write path) has never executed under
`mpirun`.

## 5. Suggested implementation order (for a real MPI environment)

1. Add `normalize_distribution_chunks`/`get_chunk_bounds_from_chunks` to
   `core/xr_chunks.py` with unit tests against plain `dask.array` chunk
   tuples (no MPI needed for this step -- pure function, testable with
   `pytest` alone).
2. `compute_save_chunks` is already wired into
   `lib_netcdf/parallel.py`'s already-distributed write branch (section 4
   above). What remains for this step, under real `mpirun`: run
   `tests/test_mpi.py` and diff written file chunk shapes with `ncdump
   -h` or `h5dump -H` against the pre-`compute_save_chunks` behavior, to
   confirm the collective write sequence and the resulting on-disk
   chunking both behave as intended. The rank-0-owned (non-distributed)
   write branch intentionally still uses `get_partition_chunk_size`/
   `get_chunks` directly rather than `compute_save_chunks`, since it
   already holds the real, complete dataset and has no local-slice
   accuracy problem to solve.
3. Only after (1) and (2) pass under real `mpirun`, consider changing
   `XarrayMPI.open_dataset`/`redistribute` to call
   `get_chunk_bounds_from_chunks` for datasets with irregular on-disk
   chunking, since that is the change with the largest blast radius
   (every existing reduction and redistribution call site depends on
   `get_chunk_bounds`'s current contract).
