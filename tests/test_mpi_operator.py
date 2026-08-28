"""Correctness tests for :mod:`climtools.xarray.operator` (``Arithmetic``).

Covers ``align``/``apply``/``evaluate``/``matmul``/``halo_exchange``/
``rolling_reduce`` -- previously untested (see ``test_mpi.py``'s own
docstring, which explicitly called this gap out).

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi_operator.py
"""

from __future__ import annotations

import operator

import numpy as np
from climtools import mpi
from climtools.xarray.meta import get_mpi_meta
from mpi4py import MPI
from mpi_fixtures import (
    RANK,
    SIZE,
    check,
    finish,
    make_dataset,
    make_field,
    make_series,
)

import xarray as xr


def _local_ref(full: xr.DataArray, dim: str, got: xr.DataArray) -> xr.DataArray:
    """Slice a serial reference to this rank's own owned bounds along ``dim``."""
    meta = get_mpi_meta(got)
    return full.isel({dim: slice(meta["start"], meta["stop"])})


def _uniform_halo_width(
    distributed: xr.DataArray | xr.Dataset, dim: str, want: int
) -> int:
    """Return a before/after halo width every rank can satisfy.

    ``halo_exchange`` requires every rank to request the *same* before/after
    (its internal ``_agree()`` call rejects otherwise, and a per-rank value
    derived from each rank's own -- differently sized -- local partition
    would violate that even though it looks reasonable locally). Uses the
    global minimum local length across ranks so the returned width is both
    identical everywhere and valid everywhere, degrading gracefully to a
    smaller (even zero) halo on a run with many more ranks than data rather
    than raising.
    """
    meta = get_mpi_meta(distributed)
    local_len = meta["stop"] - meta["start"]
    min_local_len = mpi.comm.allreduce(local_len, op=MPI.MIN)
    return min(want, min_local_len)


# -- apply() ------------------------------------------------------------------


def test_apply_add_two_distributed() -> None:
    a_full = make_series(n=23, seed=1)
    b_full = make_series(n=23, seed=2)
    a = mpi.xarray.repartition(a_full, "t")
    b = mpi.xarray.repartition(b_full, "t")
    got = mpi.xarray.apply(operator.add, a, b)
    ref = _local_ref(a_full + b_full, "t", got)
    check(
        "apply add: matches serial reference on this rank's slice",
        bool(np.allclose(got.values, ref.values)),
    )
    meta = get_mpi_meta(got)
    check(
        "apply add: result still carries mpi_meta on t",
        meta is not None and meta["dim"] == "t",
    )


def test_apply_scalar_operand() -> None:
    a_full = make_series(n=19, seed=3)
    a = mpi.xarray.repartition(a_full, "t")
    got = mpi.xarray.apply(operator.mul, a, 2.5)
    ref = _local_ref(a_full * 2.5, "t", got)
    check(
        "apply scalar: distributed * plain scalar matches reference",
        bool(np.allclose(got.values, ref.values)),
    )


def test_apply_rejects_unaligned_replicated_operand() -> None:
    a_full = make_series(n=19, seed=4)
    b_full = make_series(n=19, seed=5)
    a = mpi.xarray.repartition(a_full, "t")
    raised = False
    try:
        mpi.xarray.apply(operator.add, a, b_full)  # b_full still full-length
    except ValueError:
        raised = True
    check(
        "apply: full-length replicated operand without align() raises ValueError",
        raised or SIZE == 1,
    )


def test_apply_rejects_partition_breaking_callable() -> None:
    a_full = make_series(n=19, seed=6)
    a = mpi.xarray.repartition(a_full, "t")
    meta = get_mpi_meta(a)
    owned = meta["stop"] - meta["start"]
    # Every rank must still call apply() (it runs an internal _agree()
    # collective), regardless of whether this rank's own owned length lets
    # the callable actually be caught -- only the interpretation below
    # varies per rank, not participation in the call itself.
    raised = False
    try:
        mpi.xarray.apply(lambda x: x.isel(t=slice(0, 1)), a)
    except ValueError:
        raised = True
    if owned <= 1:
        # x.isel(t=slice(0, 1)) only provably shrinks a partition that owns
        # more than one element; at owned <= 1 it is a no-op in length terms
        # (0 -> 0 or 1 -> 1), so the post-call check has nothing to catch on
        # this particular rank. Legitimate at high rank counts relative to
        # data size, not a library bug -- don't assert either way here.
        check(
            "apply: a callable that shrinks the partition dim raises ValueError "
            + "(skipped on this rank: owns <=1 element, nothing to shrink)",
            True,
        )
        return
    check("apply: a callable that shrinks the partition dim raises ValueError", raised)


def test_apply_matmul_redirect_matches_direct_matmul() -> None:
    rng = np.random.default_rng(7)
    a_full = xr.DataArray(rng.normal(size=(17, 5)), dims=("t", "k"), name="a")
    b_full = xr.DataArray(rng.normal(size=(5, 3)), dims=("k", "n"), name="b")
    a = mpi.xarray.repartition(a_full, "t")
    via_apply = mpi.xarray.apply(operator.matmul, a, b_full)
    via_matmul = mpi.xarray.matmul(a, b_full)
    check(
        "apply(operator.matmul, ...) matches mpi.xarray.matmul directly",
        bool(np.allclose(via_apply.values, via_matmul.values)),
    )


# -- align() --------------------------------------------------------------


def test_align_both_undistributed_with_dim() -> None:
    a_full = make_series(n=21, seed=8)
    b_full = make_series(n=21, seed=9)
    left, right = mpi.xarray.align(a_full, b_full, dim="t")
    left_meta = get_mpi_meta(left)
    right_meta = get_mpi_meta(right)
    check(
        "align both undistributed: both now carry mpi_meta on t",
        left_meta is not None and right_meta is not None,
    )
    check(
        "align both undistributed: identical bounds without communication",
        left_meta is not None
        and right_meta is not None
        and left_meta["start"] == right_meta["start"]
        and left_meta["stop"] == right_meta["stop"],
    )
    combined = mpi.xarray.apply(operator.add, left, right)
    ref = _local_ref(a_full + b_full, "t", combined)
    check(
        "align both undistributed: combined result matches serial reference",
        bool(np.allclose(combined.values, ref.values)),
    )


def test_align_one_replicated() -> None:
    a_full = make_series(n=21, seed=10)
    climatology_full = make_series(n=21, seed=11)
    a = mpi.xarray.repartition(a_full, "t")
    left, right = mpi.xarray.align(a, climatology_full)
    right_meta = get_mpi_meta(right)
    check(
        "align one replicated: replicated operand now sliced to match",
        right_meta is not None,
    )
    anomaly = mpi.xarray.apply(operator.sub, left, right)
    ref = _local_ref(a_full - climatology_full, "t", anomaly)
    check(
        "align one replicated: anomaly matches serial reference",
        bool(np.allclose(anomaly.values, ref.values)),
    )


def test_align_already_matching_partitions_is_noop() -> None:
    a_full = make_series(n=19, seed=12)
    a = mpi.xarray.repartition(a_full, "t")
    b = mpi.xarray.repartition(a_full.copy(), "t")
    left, right = mpi.xarray.align(a, b)
    check("align already-matching: left returned unchanged", left is a)
    check("align already-matching: right returned unchanged", right is b)


def test_align_incompatible_partitions_raises() -> None:
    if SIZE < 2:
        check("align incompatible partitions: skipped (needs >=2 ranks)", True)
        return
    a_full = make_series(n=20, seed=13)
    a = mpi.xarray.repartition(a_full, "t")
    # Deliberately mis-slice "b" so its bounds disagree with a's on every
    # rank but the one where they'd coincidentally match by construction.
    shifted = a_full.isel(t=slice(1, 20))
    b = mpi.xarray.repartition(shifted, "t")
    raised = False
    try:
        mpi.xarray.align(a, b)
    except ValueError:
        raised = True
    check("align: incompatible existing partitions raise ValueError", raised)


# -- matmul() ---------------------------------------------------------------


def test_matmul_contracts_partition_dimension() -> None:
    rng = np.random.default_rng(20)
    t = 17
    a_full = xr.DataArray(rng.normal(size=(3, t)), dims=("m", "t"), name="a")
    b_full = xr.DataArray(rng.normal(size=(t, 4)), dims=("t", "n"), name="b")
    a = mpi.xarray.repartition(a_full, "t")
    b = mpi.xarray.repartition(b_full, "t")
    got = mpi.xarray.matmul(a, b)
    ref = a_full @ b_full
    check(
        "matmul contracting t: Allreduce-combined result matches serial a_full @ b_full",
        bool(np.allclose(got.values, ref.transpose(*got.dims).values)),
    )
    check(
        "matmul contracting t: result is replicated (no mpi_meta)",
        get_mpi_meta(got) is None,
    )


def test_matmul_does_not_contract_partition_dimension() -> None:
    rng = np.random.default_rng(21)
    t = 17
    a_full = xr.DataArray(rng.normal(size=(t, 5)), dims=("t", "k"), name="a")
    b_full = xr.DataArray(rng.normal(size=(5, 4)), dims=("k", "n"), name="b")
    a = mpi.xarray.repartition(a_full, "t")
    got = mpi.xarray.matmul(a, b_full)
    meta = get_mpi_meta(got)
    check(
        "matmul not contracting t: result stays distributed on t",
        meta is not None and meta["dim"] == "t",
    )
    ref = a_full @ b_full
    local_ref = (
        ref.isel(t=slice(meta["start"], meta["stop"])) if meta is not None else ref
    )
    check(
        "matmul not contracting t: local slice matches serial reference",
        bool(np.allclose(got.transpose(*local_ref.dims).values, local_ref.values)),
    )


# -- evaluate() ---------------------------------------------------------------


def test_evaluate_arithmetic_expression() -> None:
    a_full = make_series(n=18, seed=30)
    b_full = make_series(n=18, seed=31)
    c_full = make_series(n=18, seed=32)
    a = mpi.xarray.repartition(a_full, "t")
    _, b = mpi.xarray.align(a, b_full)
    _, c = mpi.xarray.align(a, c_full)
    got = mpi.xarray.evaluate("(a + b) * c", a=a, b=b, c=c)
    ref_full = (a_full + b_full) * c_full
    ref = _local_ref(ref_full, "t", got)
    check(
        "evaluate arithmetic: matches serial reference",
        bool(np.allclose(got.values, ref.values)),
    )


def test_evaluate_matmul_operator_matches_direct() -> None:
    rng = np.random.default_rng(40)
    t = 17
    a_full = xr.DataArray(rng.normal(size=(3, t)), dims=("m", "t"), name="a")
    b_full = xr.DataArray(rng.normal(size=(t, 4)), dims=("t", "n"), name="b")
    a = mpi.xarray.repartition(a_full, "t")
    b = mpi.xarray.repartition(b_full, "t")
    via_evaluate = mpi.xarray.evaluate("a @ b", a=a, b=b)
    via_matmul = mpi.xarray.matmul(a, b)
    check(
        "evaluate('a @ b'): matches mpi.xarray.matmul directly",
        bool(np.allclose(via_evaluate.values, via_matmul.values)),
    )


def test_evaluate_rejects_chained_comparison() -> None:
    a_full = make_series(n=10, seed=41)
    a = mpi.xarray.repartition(a_full, "t")
    raised = False
    try:
        mpi.xarray.evaluate("0 < a < 1", a=a)
    except ValueError:
        raised = True
    check("evaluate: chained comparison raises ValueError", raised)


def test_evaluate_rejects_and_or_on_xarray_operand() -> None:
    a_full = make_series(n=10, seed=42)
    a = mpi.xarray.repartition(a_full, "t")
    raised = False
    try:
        mpi.xarray.evaluate("a and a", a=a)
    except TypeError:
        raised = True
    check("evaluate: 'and' on an xarray operand raises TypeError", raised)


def test_evaluate_undefined_name_raises() -> None:
    raised = False
    try:
        mpi.xarray.evaluate("a + 1")
    except NameError:
        raised = True
    check("evaluate: undefined name raises NameError", raised)


# -- halo_exchange() ----------------------------------------------------------


def test_halo_exchange_matches_serial_neighbors() -> None:
    full = make_series(n=23, seed=50)
    distributed = mpi.xarray.repartition(full, "t")
    meta = get_mpi_meta(distributed)
    before = after = _uniform_halo_width(distributed, "t", 2)
    padded, left_pad, right_pad = mpi.xarray.halo_exchange(
        distributed, "t", before=before, after=after
    )

    is_first_rank = RANK == 0
    is_last_rank = RANK == SIZE - 1
    if before > 0:
        check(
            "halo_exchange: left_pad is 0 only at the global left edge",
            (left_pad == 0) == is_first_rank,
        )
        check(
            "halo_exchange: right_pad is 0 only at the global right edge",
            (right_pad == 0) == is_last_rank,
        )
    else:
        check(
            "halo_exchange: left/right pad edge check skipped "
            + "(more ranks than data leaves no room for any halo)",
            True,
        )

    window_start = meta["start"] - left_pad
    window_stop = meta["stop"] + right_pad
    ref = full.isel(t=slice(window_start, window_stop))
    check(
        "halo_exchange: padded values match the serial neighbor window",
        bool(np.allclose(padded.values, ref.values)),
    )
    check(
        "halo_exchange: result carries no mpi_meta (no longer a clean partition)",
        get_mpi_meta(padded) is None,
    )


def test_halo_exchange_on_dataset_leaves_static_var_alone() -> None:
    ds = make_dataset(n=20, ny=2, nx=3, seed=51)
    distributed = mpi.xarray.repartition(ds, "t")
    width = _uniform_halo_width(distributed, "t", 1)
    padded, left_pad, right_pad = mpi.xarray.halo_exchange(
        distributed, "t", before=width, after=width
    )
    check(
        "halo_exchange dataset: static var untouched (shape unchanged)",
        tuple(padded["s"].sizes.items()) == tuple(ds["s"].sizes.items()),
    )
    check(
        "halo_exchange dataset: time-varying var grew by left_pad + right_pad",
        int(padded["v"].sizes["t"]) - int(distributed["v"].sizes["t"])
        == left_pad + right_pad,
    )


def test_halo_exchange_rejects_halo_wider_than_local_partition() -> None:
    full = make_series(n=SIZE * 2 + 4, seed=52)
    distributed = mpi.xarray.repartition(full, "t")
    # Every rank must request the *same* before/after (halo_exchange's
    # _agree() check requires it), so this has to be a size large enough to
    # exceed every rank's local partition rather than derived from this
    # rank's own (possibly differently-sized) local_len.
    huge = int(full.sizes["t"]) + 1
    raised = False
    try:
        mpi.xarray.halo_exchange(distributed, "t", before=huge, after=0)
    except ValueError:
        raised = True
    check("halo_exchange: halo wider than local partition raises ValueError", raised)


# -- rolling_reduce() ---------------------------------------------------------


def test_rolling_reduce_matches_serial_rolling_mean() -> None:
    full = make_series(n=25, seed=60)
    distributed = mpi.xarray.repartition(full, "t")
    meta = get_mpi_meta(distributed)
    local_len = meta["stop"] - meta["start"]
    if mpi.comm.allreduce(local_len, op=MPI.MIN) < 1:
        check(
            "rolling_reduce mean window=3: skipped (a rank owns no data at this size)",
            True,
        )
        return
    got = mpi.xarray.rolling_reduce(
        distributed, "t", window=3, reduce="mean", center=True
    )
    meta = get_mpi_meta(got)
    full_rolled = full.rolling(t=3, center=True, min_periods=None).mean()
    ref = full_rolled.isel(t=slice(meta["start"], meta["stop"]))
    check(
        "rolling_reduce mean window=3: matches serial full.rolling(...).mean() on this rank's slice",
        bool(
            np.allclose(got.values, ref.values, equal_nan=True)
            if np.any(np.isnan(ref.values)) or np.any(np.isnan(got.values))
            else np.allclose(got.values, ref.values)
        ),
    )
    check(
        "rolling_reduce: result still carries mpi_meta on t",
        meta is not None and meta["dim"] == "t",
    )
    check(
        "rolling_reduce: local length unchanged by the roll",
        int(got.sizes["t"]) == meta["stop"] - meta["start"],
    )


def test_rolling_reduce_matches_serial_with_sum_and_no_center() -> None:
    full = make_series(n=25, seed=61)
    distributed = mpi.xarray.repartition(full, "t")
    meta = get_mpi_meta(distributed)
    local_len = meta["stop"] - meta["start"]
    if mpi.comm.allreduce(local_len, op=MPI.MIN) < 3:
        check(
            "rolling_reduce sum window=4 center=False: skipped "
            + "(a rank's slice is too small for this many ranks)",
            True,
        )
        return
    got = mpi.xarray.rolling_reduce(
        distributed, "t", window=4, reduce="sum", center=False
    )
    meta = get_mpi_meta(got)
    full_rolled = full.rolling(t=4, center=False).sum()
    ref = full_rolled.isel(t=slice(meta["start"], meta["stop"]))
    check(
        "rolling_reduce sum window=4 center=False: matches serial reference",
        bool(np.allclose(got.values, ref.values, equal_nan=True)),
    )


def test_rolling_reduce_non_partition_dim_delegates_to_xarray() -> None:
    field = make_field(n=20, ny=6, nx=3, seed=62)
    distributed = mpi.xarray.repartition(field, "t")
    got = mpi.xarray.rolling_reduce(
        distributed, "y", window=3, reduce="mean", center=True
    )
    ref = distributed.rolling(y=3, center=True).mean()
    check(
        "rolling_reduce over non-partition dim: matches plain xarray on the local shard",
        bool(np.allclose(got.values, ref.values, equal_nan=True)),
    )


def test_rolling_reduce_dataset_static_var_untouched() -> None:
    ds = make_dataset(n=22, ny=2, nx=3, seed=63)
    distributed = mpi.xarray.repartition(ds, "t")
    meta = get_mpi_meta(distributed)
    local_len = meta["stop"] - meta["start"]
    if mpi.comm.allreduce(local_len, op=MPI.MIN) < 1:
        check(
            "rolling_reduce dataset: skipped (a rank owns no data at this size)",
            True,
        )
        return
    got = mpi.xarray.rolling_reduce(
        distributed, "t", window=3, reduce="mean", center=True
    )
    check(
        "rolling_reduce dataset: static var shape unchanged (halo_exchange data_vars fix)",
        tuple(got["s"].sizes.items()) == tuple(ds["s"].sizes.items()),
    )
    check(
        "rolling_reduce dataset: static var values unchanged",
        bool(np.allclose(got["s"].values, distributed["s"].values)),
    )


if __name__ == "__main__":
    test_apply_add_two_distributed()
    test_apply_scalar_operand()
    test_apply_rejects_unaligned_replicated_operand()
    test_apply_rejects_partition_breaking_callable()
    test_apply_matmul_redirect_matches_direct_matmul()
    test_align_both_undistributed_with_dim()
    test_align_one_replicated()
    test_align_already_matching_partitions_is_noop()
    test_align_incompatible_partitions_raises()
    test_matmul_contracts_partition_dimension()
    test_matmul_does_not_contract_partition_dimension()
    test_evaluate_arithmetic_expression()
    test_evaluate_matmul_operator_matches_direct()
    test_evaluate_rejects_chained_comparison()
    test_evaluate_rejects_and_or_on_xarray_operand()
    test_evaluate_undefined_name_raises()
    test_halo_exchange_matches_serial_neighbors()
    test_halo_exchange_on_dataset_leaves_static_var_alone()
    test_halo_exchange_rejects_halo_wider_than_local_partition()
    test_rolling_reduce_matches_serial_rolling_mean()
    test_rolling_reduce_matches_serial_with_sum_and_no_center()
    test_rolling_reduce_non_partition_dim_delegates_to_xarray()
    test_rolling_reduce_dataset_static_var_untouched()
    finish()
