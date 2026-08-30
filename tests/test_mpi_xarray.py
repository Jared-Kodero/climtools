"""End-to-end correctness and performance test suite for ``climtools.xarray``.

Exercises every public :class:`~climtools.xarray.core.MPIXarray` pathway --
constructors, indexing, reductions, statistics, groupby/resample, rolling,
elementwise, arithmetic, and structural redistribution -- against a serial
(single-process) ``xarray`` reference computed independently on every rank
from the same in-memory/on-disk data, then prints a pass/fail and timing
summary on rank 0.

Run with (any rank count; deliberately includes counts that do not divide
the time dimension evenly, e.g. 5 or 7, to exercise uneven-partition
boundaries)::

    mpirun -n <N> python tests/test_mpi_xarray.py
    mpirun -n <N> --oversubscribe python tests/test_mpi_xarray.py  # single-node dev boxes

A single run only tests one rank count; invoke it multiple times with
different ``-n`` to cover the uneven-partition code paths. ``run_mpi_xarray_tests.sh``
does this automatically across a matrix of rank counts.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_PARENT = _TESTS_DIR.parents[1]  # directory containing climtools/
for _p in (str(_TESTS_DIR), str(_REPO_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from mpi4py import MPI

from climtools import mpi
from climtools.xarray import constructors as xm

import xarray as xr

from mock_dataset import OUTPUT_DIR, create_dataset, path

rank = mpi.comm.rank
size = mpi.comm.size
RUN_PERF = os.environ.get("CLIMTOOLS_TEST_PERF", "1") != "0"


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    name: str
    ok: bool
    detail: str = ""
    mpi_time: float | None = None
    native_time: float | None = None


@dataclass
class Suite:
    cases: list[CaseResult] = field(default_factory=list)

    def add(self, result: CaseResult) -> None:
        self.cases.append(result)

    @property
    def failed(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.ok]


suite = Suite()


def log(*values: object) -> None:
    if rank == 0:
        print(*values, flush=True)


# ---------------------------------------------------------------------------
# Gather / compare helpers
# ---------------------------------------------------------------------------


def gather_full(obj):
    """Reassemble a (possibly distributed) MPIXarray result onto rank 0.

    Returns the full, rank-0-only object for a distributed result (gathered
    and concatenated along its partition dimension, in rank order, which is
    correct because the wrapper always partitions each dimension into
    contiguous, monotonically increasing rank-ordered slices) or the plain
    replicated value (verified identical across all ranks) otherwise. Every
    rank must call this together (it is collective); only rank 0's return
    value is meaningful.
    """
    from climtools.xarray.core import MPIXarray

    if isinstance(obj, MPIXarray):
        meta = obj.meta
        if meta is None:
            pieces = mpi.comm.gather(obj.data, root=0)
            if rank == 0:
                first = pieces[0]
                for other in pieces[1:]:
                    xr.testing.assert_allclose(first, other)
                return first
            return None
        dim = meta["dim"]
        pieces = mpi.comm.gather(obj.data, root=0)
        if rank == 0:
            kwargs = {}
            if isinstance(pieces[0], xr.Dataset):
                # "minimal": only concatenate data_vars/coords that already
                # carry `dim` (e.g. "pr", "t2m"); variables that never had
                # it (e.g. the mock dataset's static (plev, lat, lon) "t"
                # profile) are identically replicated on every rank already
                # and must be taken once, not stacked into a new `dim`-sized
                # axis -- which is what xr.concat's default ("all") does,
                # silently fabricating a wrong extra dimension on exactly
                # the variables that were never partitioned.
                kwargs = {"data_vars": "minimal", "coords": "minimal"}
            return xr.concat(pieces, dim=dim, **kwargs)
        return None

    # Plain Python/numpy scalar (e.g. from .apply()/.evaluate()/matmul on a
    # fully-contracted dimension): still rank-replicated, verify agreement.
    pieces = mpi.comm.gather(obj, root=0)
    if rank == 0:
        for other in pieces[1:]:
            np.testing.assert_allclose(np.asarray(pieces[0]), np.asarray(other))
        return pieces[0]
    return None


def check(
    name: str,
    mpi_fn,
    native_fn,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Run ``mpi_fn`` (distributed) and ``native_fn`` (serial, rank-0-only
    reference), gather, compare, and record one :class:`CaseResult`.

    ``mpi_fn``/``native_fn`` take no arguments. Every rank must reach this
    call together -- it contains collective operations.

    Whether ``mpi_fn`` raised is reduced (``Allreduce``/``LOR``) across
    every rank *before* deciding whether to call the collective
    ``gather_full`` below: if it raised on only a subset of ranks (a
    genuine bug should raise identically everywhere via climtools' own
    ``raise_if_error``, but a test-harness-level mistake -- or a real bug
    that isn't rank-symmetric -- would not), letting the non-raising ranks
    proceed into ``gather_full``'s collective ``comm.gather`` while the
    raising ones return early is an immediate deadlock: the ranks that
    return never post the gather, and the ranks that reach it wait
    forever for a message that never comes.
    """
    mpi.comm.barrier()
    if os.environ.get("CLIMTOOLS_TEST_DEBUG"):
        log(f"  -> {name}")
    t0 = time.perf_counter()
    error: BaseException | None = None
    mpi_result = None
    try:
        mpi_result = mpi_fn()
    except BaseException as exc:  # noqa: BLE001 - want to record and continue
        error = exc
    mpi_time = time.perf_counter() - t0

    any_error = mpi.comm.allreduce(error is not None, op=MPI.LOR)
    if any_error:
        local_detail = f"{type(error).__name__}: {error}" if error is not None else None
        if error is not None:
            print(f"(rank {rank}) {name}:", flush=True)
            traceback.print_exception(type(error), error, error.__traceback__)
        details = [d for d in mpi.comm.gather(local_detail, root=0) if d is not None] if rank == 0 else None
        detail = "; ".join(details) if rank == 0 and details else ""
        detail = mpi.comm.bcast(detail if rank == 0 else None, root=0)
        suite.add(CaseResult(name, False, detail, mpi_time=mpi_time))
        return

    try:
        gathered = gather_full(mpi_result)
    except BaseException as exc:  # noqa: BLE001
        suite.add(CaseResult(name, False, f"gather failed: {exc!r}", mpi_time=mpi_time))
        return

    ok = True
    detail = ""
    native_time = None
    if rank == 0:
        t1 = time.perf_counter()
        try:
            expected = native_fn()
            native_time = time.perf_counter() - t1
            if isinstance(expected, (xr.Dataset, xr.DataArray)):
                xr.testing.assert_allclose(gathered, expected, rtol=rtol, atol=atol)
            else:
                np.testing.assert_allclose(
                    np.asarray(gathered), np.asarray(expected), rtol=rtol, atol=atol
                )
        except BaseException as exc:  # noqa: BLE001
            ok = False
            detail = f"{type(exc).__name__}: {exc}"

    ok = mpi.comm.bcast(ok if rank == 0 else None, root=0)
    detail = mpi.comm.bcast(detail if rank == 0 else None, root=0)
    suite.add(CaseResult(name, ok, detail, mpi_time=mpi_time, native_time=native_time))


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

# Deliberately not evenly divisible by common rank counts (5, 7, ...): 41 is
# prime, so every rank count > 1 up to 41 exercises an uneven partition.
N_TIME = 41
RESOLUTION_DEG = 15.0
PLEV_STEP = -250.0

create_dataset(path, n_time=N_TIME, resolution_deg=RESOLUTION_DEG, plev_step=PLEV_STEP)

# Every rank opens the reference independently (small file, cheap, and
# avoids a broadcast of the whole dataset just to get a serial baseline).
ref = xr.open_dataset(path).load()

mds = xm.mpi_open_dataset(path, mpi, partition_dim="time", log_partitions=(rank == 0))

log(f"\n=== climtools.xarray MPI test suite: size={size}, n_time={N_TIME} ===\n")


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

check(
    "constructors.mpi_open_dataset",
    lambda: mds,
    lambda: ref,
)

_root_value = ref if rank == 0 else None
check(
    "constructors.mpi_partition_data (distribute)",
    lambda: xm.mpi_partition_data(_root_value, mpi, "time"),
    lambda: ref,
)


def _fill_dataarray(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.float64)[:, None] * np.ones((1, 5))


check(
    "constructors.mpi_create_dataarray",
    lambda: xm.mpi_create_dataarray(
        mpi,
        _fill_dataarray,
        ("time", "x"),
        shape={"time": N_TIME, "x": 5},
        dim="time",
    ),
    lambda: xr.DataArray(
        np.arange(N_TIME, dtype=np.float64)[:, None] * np.ones((1, 5)),
        dims=("time", "x"),
    ),
)


def _fill_dataset_var(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.float64)


check(
    "constructors.mpi_create_dataset",
    lambda: xm.mpi_create_dataset(
        mpi,
        {"v": (("time",), _fill_dataset_var)},
        {"time": N_TIME},
        dim="time",
        log_partitions=False,
    ),
    lambda: xr.Dataset({"v": (("time",), np.arange(N_TIME, dtype=np.float64))}),
)

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

check(
    "isel: slice on partition dim",
    lambda: mds.isel(time=slice(2, N_TIME - 2)),
    lambda: ref.isel(time=slice(2, N_TIME - 2)),
)
check(
    "isel: slice on non-partition dim",
    lambda: mds.isel(lat=slice(1, 4)),
    lambda: ref.isel(lat=slice(1, 4)),
)
check(
    "isel: scalar on partition dim (replicate)",
    lambda: mds.isel(time=7),
    lambda: ref.isel(time=7),
)
check(
    "sel: scalar label on partition dim (replicate)",
    lambda: mds.sel(time=ref.time.values[5]),
    lambda: ref.sel(time=ref.time.values[5]),
)
check(
    "sel: slice on partition dim",
    lambda: mds.sel(time=slice(ref.time.values[1], ref.time.values[-2])),
    lambda: ref.sel(time=slice(ref.time.values[1], ref.time.values[-2])),
)

# ---------------------------------------------------------------------------
# Reductions (partition-dim + non-partition-dim, default partition_dim="auto")
# ---------------------------------------------------------------------------

def _dim_only(full_ds, mpi_ds, dim):
    """Like ``_time_only`` (defined below, for groupby/resample) but for an
    arbitrary ``dim``: needed here because ``var``/``std`` (unlike
    ``sum``/``mean``/``min``/``max``/``prod``, which all leave a variable
    lacking ``dim`` untouched) follow plain ``xarray``'s own
    ``Dataset.var(dim)``/``Dataset.std(dim)`` convention of returning 0
    (``ddof=0``) or NaN (``ddof>0``) for such a variable instead -- a
    quirk of xarray itself (a "reduction" over an axis of implicit length
    one), reproduced faithfully by climtools' passthrough to
    ``value.var(dim)``/``value.std(dim)`` for the non-distributed local
    case, so it is not a climtools bug, but it does mean a variable
    lacking ``dim`` must be excluded here for a meaningful comparison of
    the *distributed* code path instead of re-testing this xarray quirk.
    """
    dim_vars = [name for name, var in full_ds.data_vars.items() if dim in var.dims]
    other_vars = [v for v in full_ds.data_vars if v not in dim_vars]
    used_dims = {d for v in dim_vars for d in full_ds[v].dims}
    orphan_coords = [c for c in full_ds.coords if c in full_ds.dims and c not in used_dims]
    return mpi_ds.drop_vars(other_vars + orphan_coords), full_ds[dim_vars]


for op in ("sum", "mean", "min", "max", "prod"):
    check(
        f"reduce.{op}(time) [partition dim]",
        lambda op=op: getattr(mds, op)("time"),
        lambda op=op: getattr(ref, op)("time"),
    )
    check(
        f"reduce.{op}(lat) [non-partition dim]",
        lambda op=op: getattr(mds, op)("lat"),
        lambda op=op: getattr(ref, op)("lat"),
    )
for op in ("var", "std"):
    for dim in ("time", "lat"):
        _mds_d, _ref_d = _dim_only(ref, mds, dim)
        label = "partition dim" if dim == "time" else "non-partition dim"
        check(
            f"reduce.{op}({dim}) [{label}]",
            lambda op=op, dim=dim, d=_mds_d: getattr(d, op)(dim),
            lambda op=op, dim=dim, d=_ref_d: getattr(d, op)(dim),
        )

check(
    "reduce.sum(time, partition_dim=None) [replicated result]",
    lambda: mds.sum("time", partition_dim=None),
    lambda: ref.sum("time"),
)
check(
    "reduce.mean(...) [reduce all dims, fully replicated scalar]",
    lambda: mds.mean(),
    lambda: ref.mean(),
)

_bool_mds = mds["pr"] > mds["pr"].mean()
_bool_ref = ref["pr"] > ref["pr"].mean()
check(
    "reduce.any(time) on boolean",
    lambda: _bool_mds.any("time"),
    lambda: _bool_ref.any("time"),
)
check(
    "reduce.all(time) on boolean",
    lambda: _bool_mds.all("time"),
    lambda: _bool_ref.all("time"),
)
def _first_ref():
    # xarray has no .first(); replicate its "first valid value along dim"
    # semantics (skipna=True default) with plain numpy so the reference
    # doesn't depend on the optional `bottleneck` package that
    # DataArray.ffill/.bfill otherwise require.
    def first_valid(v: xr.DataArray) -> xr.DataArray:
        if "time" not in v.dims:
            return v
        axis = v.dims.index("time")
        arr = v.values
        valid = ~np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones_like(arr, dtype=bool)
        idx = np.argmax(valid, axis=axis)
        return v.isel(time=xr.DataArray(idx, dims=[d for d in v.dims if d != "time"]))

    return ref.map(first_valid, keep_attrs=True).drop_vars("time", errors="ignore")


def _last_ref():
    def last_valid(v: xr.DataArray) -> xr.DataArray:
        if "time" not in v.dims:
            return v
        axis = v.dims.index("time")
        arr = v.values
        valid = ~np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones_like(arr, dtype=bool)
        rev_idx = np.argmax(valid[::-1], axis=axis)
        idx = arr.shape[axis] - 1 - rev_idx
        return v.isel(time=xr.DataArray(idx, dims=[d for d in v.dims if d != "time"]))

    return ref.map(last_valid, keep_attrs=True).drop_vars("time", errors="ignore")


check(
    "reduce.first(time) [skipna semantics]",
    # first()/last() keep a per-position "time" coordinate recording which
    # global time index was picked at each point -- useful information,
    # but not something a plain-numpy reference conveniently reproduces,
    # so it is dropped from both sides for the comparison.
    lambda: mds.first("time").drop_vars("time", errors="ignore"),
    _first_ref,
)
check(
    "reduce.last(time) [skipna semantics]",
    lambda: mds.last("time").drop_vars("time", errors="ignore"),
    _last_ref,
)

# skipna correctness: inject NaNs into a copy and verify mean/sum survive
# an uneven split with missing data concentrated on a subset of ranks.
_nan_ref = ref.copy(deep=True)
_nan_mask = xr.DataArray(np.arange(N_TIME) % 5 != 0, dims="time")
_nan_ref["pr"] = _nan_ref["pr"].where(_nan_mask)
_nan_root = _nan_ref if rank == 0 else None
_nan_mds = xm.mpi_partition_data(_nan_root, mpi, "time")
check(
    "reduce.mean(time) with NaNs, skipna=True",
    lambda: _nan_mds.mean("time", skipna=True),
    lambda: _nan_ref.mean("time", skipna=True),
)
check(
    "reduce.sum(time) with NaNs, skipna=True, min_count=1",
    lambda: _nan_mds.sum("time", skipna=True, min_count=1),
    lambda: _nan_ref.sum("time", skipna=True, min_count=1),
)

# ---------------------------------------------------------------------------
# Groupby / resample (cross-rank-boundary groups)
# ---------------------------------------------------------------------------

# "t"/"slmsk" carry no "time" dim at all. climtools' groupby/resample
# deliberately leave such variables completely untouched -- no group
# dimension added -- rather than broadcasting them across every group the
# way plain `xarray.Dataset.groupby(...).mean()` does; see
# `Groupby._group_reduce_local`'s docstring. That is an intentional,
# documented difference in Dataset-level behavior, not something a
# variable-for-variable numerical comparison against raw xarray should be
# faulted on, so the grouped/resampled checks below compare only the
# variables that actually vary with "time".
mds_t, ref_t = _dim_only(ref, mds, "time")

_group_labels_local = (mds.data.time.values.astype("datetime64[h]").astype(np.int64)) % 4
for gop in ("sum", "mean", "count", "min", "max"):
    check(
        f"groupby.{gop}() across rank boundaries",
        lambda gop=gop: getattr(
            mds_t.groupby("time", _group_labels_local), gop
        )(),
        lambda gop=gop: getattr(
            ref_t.groupby(
                xr.DataArray(
                    (ref.time.values.astype("datetime64[h]").astype(np.int64)) % 4,
                    dims="time",
                    name="_mpi_group",
                )
            ),
            gop,
        )(),
    )

for rop in ("sum", "mean", "count", "min", "max"):
    check(
        f"resample.{rop}('12h') across rank boundaries",
        lambda rop=rop: getattr(mds_t.resample("time", "12h"), rop)(),
        lambda rop=rop: getattr(ref_t.resample(time="12h"), rop)(),
    )

# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------

check(
    "where",
    lambda: mds.where(mds > 0),
    lambda: ref.where(ref > 0),
)
check(
    "where with fill",
    lambda: mds.where(mds["pr"] > mds["pr"].mean(), -1.0),
    lambda: ref.where(ref["pr"] > ref["pr"].mean(), -1.0),
)
check(
    "cumsum(time) [partition dim, cross-rank prefix]",
    lambda: mds.cumsum("time"),
    lambda: ref.cumsum("time"),
)
check(
    "cumsum(lat) [non-partition dim]",
    lambda: mds.cumsum("lat"),
    lambda: ref.cumsum("lat"),
)
check(
    "median(time) [partition dim, allgather path]",
    lambda: mds.median("time"),
    lambda: ref.median("time"),
)
check(
    "median(lat) [non-partition dim]",
    lambda: mds.median("lat"),
    lambda: ref.median("lat"),
)
check(
    "diff(time, label=upper) [partition dim, halo exchange]",
    lambda: mds.diff("time"),
    lambda: ref.diff("time"),
)
check(
    "diff(time, label=lower) [partition dim, halo exchange]",
    lambda: mds.diff("time", label="lower"),
    lambda: ref.diff("time", label="lower"),
)
check(
    "diff(time, n=2) [partition dim, wider halo]",
    lambda: mds.diff("time", n=2),
    lambda: ref.diff("time", n=2),
)
check(
    "diff(lat) [non-partition dim]",
    lambda: mds.diff("lat"),
    lambda: ref.diff("lat"),
)

# ---------------------------------------------------------------------------
# Rolling (halo exchange)
# ---------------------------------------------------------------------------

check(
    "rolling_reduce(time, window=5, mean) [partition dim]",
    lambda: mds.rolling_reduce("time", 5, "mean"),
    lambda: ref.rolling(time=5, center=True).mean(),
)
check(
    "rolling(time, 5).sum() [partition dim, handle-style]",
    lambda: mds.rolling("time", 5).sum(),
    lambda: ref.rolling(time=5, center=True).sum(),
)
check(
    "rolling(time, 4, center=False).mean() [partition dim, trailing window]",
    lambda: mds.rolling("time", 4, center=False).mean(),
    lambda: ref.rolling(time=4, center=False).mean(),
)
check(
    "rolling_reduce(lat, window=3) [non-partition dim]",
    lambda: mds.rolling_reduce("lat", 3, "mean"),
    lambda: ref.rolling(lat=3, center=True).mean(),
)

# ---------------------------------------------------------------------------
# Arithmetic / apply / matmul / align
# ---------------------------------------------------------------------------

check(
    "apply: elementwise func",
    lambda: mds.apply(lambda d: d * 2.0 + 1.0, mds),
    lambda: ref * 2.0 + 1.0,
)
check(
    "__add__: MPIXarray + MPIXarray",
    lambda: mds + mds,
    lambda: ref + ref,
)
check(
    "__add__: MPIXarray + scalar",
    lambda: mds + 5.0,
    lambda: ref + 5.0,
)
check(
    "__radd__: scalar + MPIXarray",
    lambda: 5.0 + mds,
    lambda: 5.0 + ref,
)
check(
    "__mul__ / __sub__ / __truediv__ chain",
    # partition_dim=None: mean()/std() must be replicated (not
    # redistributed onto some other surviving dim) to broadcast back
    # against `mds`, which is still distributed along "time" -- combining
    # operands distributed over two different dims is correctly rejected
    # (see operator.Arithmetic._check_operands_distribution) and is not
    # what this case is testing. Restricted to "t2m" specifically (not
    # "pr", whose per-cell std can be within float32 rounding of zero in
    # this synthetic dataset, which amplifies negligible mean/std rounding
    # noise into large *relative* z-score error through the division --
    # an ill-conditioned test case, not a distribution bug) and to the
    # "time"-varying variables generally, for the same reason as the
    # var/std cases above.
    lambda: (mds_t["t2m"] - mds_t["t2m"].mean("time", partition_dim=None))
    / mds_t["t2m"].std("time", partition_dim=None),
    lambda: (ref_t["t2m"] - ref_t["t2m"].mean("time")) / ref_t["t2m"].std("time"),
    # float32 data: "t2m"'s real time-variability (~0.05 K amplitude) rides
    # on a ~288 K base, so ordinary float32 rounding in the mean (relative
    # to 288, ~3e-5 absolute) gets amplified by the division through a
    # small std into a ~1e-4 z-score error -- classic condition-number
    # amplification from computing (x - mean) / std when std is small
    # relative to x's own magnitude, not specific to the MPI code path
    # (the same computation in plain single-process float32 numpy would
    # show comparable sensitivity to summation order). Looser than the
    # module default rtol/atol to absorb that, tight enough to still catch
    # a real bug (which would show errors many orders of magnitude larger).
    rtol=5e-3,
    atol=1e-3,
)

_ts_mpi = mds["pr"].mean(("lat", "lon"))  # distributed 1-D along time
_ts_ref = ref["pr"].mean(("lat", "lon"))
check(
    "matmul: contracts the partition dimension",
    lambda: _ts_mpi.matmul(_ts_mpi),
    lambda: _ts_ref.dot(_ts_ref),
)

_other_root = ref if rank == 0 else None
check(
    "align: two independently distributed operands",
    lambda: (lambda pair: pair[0] + pair[1])(
        mds.align(xm.mpi_partition_data(_other_root, mpi, "time"))
    ),
    lambda: ref + ref,
)

# ---------------------------------------------------------------------------
# Safe passthrough methods (structurally partition-preserving)
# ---------------------------------------------------------------------------

check(
    "astype",
    lambda: mds["pr"].astype("float32"),
    lambda: ref["pr"].astype("float32"),
)
check(
    "rename",
    lambda: mds.rename({"pr": "precip"}),
    lambda: ref.rename({"pr": "precip"}),
)
check(
    "transpose",
    lambda: mds["pr"].transpose("lon", "lat", "time"),
    lambda: ref["pr"].transpose("lon", "lat", "time"),
)

# ---------------------------------------------------------------------------
# repartition
# ---------------------------------------------------------------------------

check(
    "repartition: gather via sum then repartition onto lat",
    lambda: mds.sum("time", partition_dim=None).repartition("lat"),
    lambda: ref.sum("time"),
)

mpi.comm.barrier()


# ---------------------------------------------------------------------------
# Timing comparison on a larger synthetic array (illustrative only -- see
# summary caveat: a single oversubscribed development core does not show
# genuine parallel speedup, only that the distributed code path runs and
# its overhead relative to the serial baseline is bounded).
# ---------------------------------------------------------------------------

from mock_dataset import build_dataset  # noqa: E402

PERF_N_TIME = 400
PERF_RESOLUTION_DEG = 5.0
if RUN_PERF:
    # A dedicated sibling directory: create_dataset()/build_dataset() rmtree's
    # and rebuilds its *whole parent directory* on every call (see
    # mock_dataset.create_dataset), so reusing OUTPUT_DIR here would delete
    # the main fixture's .nc file out from under the still-open `mds`/`ref`.
    PERF_DIR = OUTPUT_DIR.parent / "climtools_mock_dataset_perf"
    perf_path = PERF_DIR / "perf.nc"
    if rank == 0:
        PERF_DIR.mkdir(parents=True, exist_ok=True)
        build_dataset(perf_path, PERF_N_TIME, PERF_RESOLUTION_DEG, -200.0)
    mpi.comm.barrier()

    perf_ref = xr.open_dataset(perf_path).load() if rank == 0 else None
    perf_mds = xm.mpi_open_dataset(
        perf_path, mpi, partition_dim="time", log_partitions=False
    )

    check(
        "PERF sum(time)",
        lambda: perf_mds.sum("time"),
        lambda: perf_ref.sum("time"),
        # float32, summed over 400 elements: distributed partial-sum order
        # differs from a single-process accumulation, so results agree to
        # float32 rounding accumulated across the reduction, not exactly.
        rtol=1e-4,
        atol=1e-3,
    )
    check(
        "PERF mean(time)",
        lambda: perf_mds.mean("time"),
        lambda: perf_ref.mean("time"),
        rtol=1e-4,
        atol=1e-3,
    )
    check(
        "PERF rolling_reduce(time, 7, mean)",
        lambda: perf_mds.rolling_reduce("time", 7, "mean"),
        lambda: perf_ref.rolling(time=7, center=True).mean(),
    )
    check(
        "PERF diff(time)",
        lambda: perf_mds.diff("time"),
        lambda: perf_ref.diff("time"),
    )
    perf_mds_t, perf_ref_t = (
        _dim_only(perf_ref, perf_mds, "time")
        if rank == 0
        else (_dim_only(ref, perf_mds, "time")[0], None)
    )
    _perf_group = (
        perf_mds.data.time.values.astype("datetime64[h]").astype(np.int64)
    ) % 24
    check(
        "PERF groupby.mean()",
        lambda: perf_mds_t.groupby("time", _perf_group).mean(),
        lambda: perf_ref_t.groupby(
            xr.DataArray(
                (perf_ref.time.values.astype("datetime64[h]").astype(np.int64)) % 24,
                dims="time",
                name="_mpi_group",
            )
        ).mean(),
    )

mpi.comm.barrier()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

if rank == 0:
    n_ok = sum(c.ok for c in suite.cases)
    n_total = len(suite.cases)
    width = max(len(c.name) for c in suite.cases)

    print(f"\n{'=' * (width + 40)}")
    print(f"climtools.xarray MPI test suite -- {size} rank(s), n_time={N_TIME}")
    print(f"{'=' * (width + 40)}")
    for c in suite.cases:
        status = "PASS" if c.ok else "FAIL"
        timing = ""
        if c.mpi_time is not None and c.native_time is not None and c.native_time > 0:
            timing = f"  mpi={c.mpi_time * 1e3:8.3f} ms  native={c.native_time * 1e3:8.3f} ms  x{c.native_time / c.mpi_time:5.2f}"
        elif c.mpi_time is not None:
            timing = f"  mpi={c.mpi_time * 1e3:8.3f} ms"
        print(f"{status:4s}  {c.name:<{width}s}{timing}")
        if not c.ok and c.detail:
            print(f"      -> {c.detail}")
    print(f"{'-' * (width + 40)}")
    print(f"{n_ok}/{n_total} passed" + (f"  ({n_total - n_ok} FAILED)" if n_ok < n_total else ""))
    print(
        "\nNote on timing: this machine oversubscribes MPI ranks onto a "
        "single core, so the mpi/native comparison above measures "
        "correctness-path overhead, not genuine parallel speedup. On a "
        "real multi-core/multi-node allocation, the mpi column reflects "
        "wall-clock time actually distributed across ranks."
    )

mpi.comm.barrier()
failed = mpi.comm.bcast(len(suite.failed) if rank == 0 else None, root=0)
if failed:
    raise SystemExit(1)
