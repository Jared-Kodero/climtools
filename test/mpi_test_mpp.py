"""Direct checks of the FMS-adapted primitives in ``climtools.xarray.mpp``.

Every other module in this suite exercises these through the xarray layer,
which is the right way to test behaviour but a poor way to test the property
these routines exist for: that their answer does not depend on how the global
array was divided among ranks. A reduction that silently changes value at 8
ranks still passes a comparison against a native computation run at 8 ranks.

So the checks here are of two kinds. Some compare against a serial reference
computed on the same rank from the whole global array. The rest are
*invariance* checks: the same quantity is computed at this rank count and
compared against a value the module derives independently of it, so a result
that drifts with rank count fails even when it is self-consistent.
"""

from __future__ import annotations

import numpy as np
from climtools.xarray.chunks import get_balanced_bounds
from climtools.xarray.mpp import (
    Domain,
    mpp_chksum,
    mpp_complete_update_domains,
    mpp_define_layout,
    mpp_get_compute_domains,
    mpp_partition_offsets,
    mpp_prod_decompose,
    mpp_prod_recombine,
    mpp_reproducing_prod,
    mpp_reproducing_sum,
    mpp_slice_compute_domain,
    mpp_start_update_domains,
)
from mpi4py import MPI
from mpi_test_common import Fixtures, mpi, record


def _local_slice(global_length: int) -> tuple[int, int]:
    """This rank's half-open share of ``global_length``, balanced."""
    return get_balanced_bounds(global_length, mpi.comm.rank, mpi.comm.size)


def _check(op: str, case: str, fn) -> None:
    """Run one check, recording a failure rather than aborting the suite."""
    try:
        ok, detail = fn()
        # `ok is None` means the check does not apply at this rank count and
        # is reported as SKIP, following the suite's convention.
        record(op, case, None if ok is None else bool(ok), "" if ok else detail)
    except Exception as exc:  # noqa: BLE001 - matches the suite's convention
        record(op, case, False, f"{type(exc).__name__}: {str(exc)[:200]}")


def run(fx: Fixtures) -> None:
    """Run every mpp primitive check."""
    comm = mpi.comm
    nranks = comm.size

    # ------------------------------------------------------------------
    # Domain bookkeeping: answered locally, so the test is that it agrees
    # with what the ranks would have reported had they been asked.
    # ------------------------------------------------------------------
    def compute_domains() -> tuple[bool, str]:
        global_length = 1000
        table = mpp_get_compute_domains(global_length, nranks)
        mine = table[comm.rank]
        actual = _local_slice(global_length)
        gathered = comm.allgather(actual)
        if mine != actual:
            return False, f"own entry {mine} != actual {actual}"
        if table != gathered:
            return False, "table disagrees with the ranks' own bounds"
        covered = sum(stop - start for start, stop in table)
        return covered == global_length, f"table covers {covered}/{global_length}"

    _check(
        "mpp_get_compute_domains", "matches every rank's own bounds", compute_domains
    )

    def slice_domain() -> tuple[bool, str]:
        """The offsets isel derives locally must match an allgather's answer."""
        global_length = 997
        start, stop = _local_slice(global_length)
        for lo, hi in ((0, global_length), (10, 500), (496, 997), (300, 301)):
            local_start, local_stop, new_start = mpp_slice_compute_domain(
                start, stop, lo, hi
            )
            kept = local_stop - local_start
            counts = comm.allgather(kept)
            expected_start = sum(counts[: comm.rank])
            if new_start != expected_start:
                return False, f"slice({lo},{hi}): {new_start} != {expected_start}"
            if sum(counts) != hi - lo:
                return False, f"slice({lo},{hi}): kept {sum(counts)} of {hi - lo}"
        return True, ""

    _check("mpp_slice_compute_domain", "offsets match an allgather", slice_domain)

    def partition_offsets() -> tuple[bool, str]:
        """Exscan-derived offsets must match the allgather they replaced."""
        for length in (0, 1, 7 + comm.rank, 100):
            total, start, stop = mpp_partition_offsets(comm, length)
            counts = comm.allgather(length)
            if (total, start, stop) != (
                sum(counts),
                sum(counts[: comm.rank]),
                sum(counts[: comm.rank]) + length,
            ):
                return False, f"length={length}: got {(total, start, stop)}"
        return True, ""

    _check(
        "mpp_partition_offsets", "matches an allgather, incl. empty", partition_offsets
    )

    def define_layout() -> tuple[bool, str]:
        for extent0, extent1, npes in ((100, 100, 4), (10, 1000, 8), (721, 1440, 12)):
            rows, cols = mpp_define_layout(extent0, extent1, npes)
            if rows * cols != npes:
                return False, f"({extent0},{extent1}) on {npes}: {rows}x{cols}"
            if rows < 1 or cols < 1:
                return False, f"degenerate layout {rows}x{cols}"
        # A wide domain should be split more along its long axis than a
        # square one, which is the whole point of matching the aspect ratio.
        square_rows, _ = mpp_define_layout(100, 100, 8)
        wide_rows, _ = mpp_define_layout(10, 1000, 8)
        return wide_rows <= square_rows, "aspect ratio ignored"

    _check("mpp_define_layout", "factorises the rank count exactly", define_layout)

    # ------------------------------------------------------------------
    # Reproducing reductions: compared against a serial reference over the
    # whole global array, which is the reference the distributed answer is
    # supposed to reproduce at any rank count.
    # ------------------------------------------------------------------
    def reproducing_sum() -> tuple[bool, str]:
        length = 4096
        rng = np.random.default_rng(20260906)
        field = rng.standard_normal((length, 3)) * 1e7
        start, stop = _local_slice(length)
        distributed = mpp_reproducing_sum(field[start:stop], comm, axis=0)
        serial = mpp_reproducing_sum(field, MPI.COMM_SELF, axis=0)
        if not np.array_equal(distributed, serial):
            return False, f"{distributed} != serial {serial}"
        # The EFP digits truncate below 2**-138, so agreement with a plain
        # float64 sum is close rather than exact; that it is *bitwise* stable
        # across rank counts is the property checked above.
        plain = field.sum(axis=0)
        if not np.allclose(distributed, plain, rtol=1e-12):
            return False, f"{distributed} vs plain sum {plain}"
        return True, ""

    _check("mpp_reproducing_sum", "bitwise equal to a serial sum", reproducing_sum)

    def reproducing_prod() -> tuple[bool, str]:
        """The case a plain distributed product gets wrong.

        A float32 product long enough to overflow, containing an exact zero:
        once a partial saturates to inf, whether a zero is met before or
        after it decides between 0.0 and inf * 0 = NaN, so a naive
        implementation answers differently depending on which rank held the
        zero.
        """
        length, ncols = 512, 6
        rng = np.random.default_rng(7)
        field = (
            rng.random((length, ncols), dtype=np.float32) * np.float32(40.0)
        ).astype(np.float32)
        field[3, 0] = np.float32(0.0)  # early zero: 0.0 serially, NaN naively
        field[400, 1] = np.float32(0.0)  # late zero
        field[9, 2] = np.float32(np.nan)
        field[11, 3] = np.float32(np.inf)
        field[:, 4] = np.float32(1.0)  # representable, no saturation
        start, stop = _local_slice(length)

        f32 = np.dtype(np.float32)
        distributed = mpp_reproducing_prod(field[start:stop], comm, axis=0, dtype=f32)
        serial = mpp_reproducing_prod(field, MPI.COMM_SELF, axis=0, dtype=f32)
        if not np.array_equal(distributed, serial, equal_nan=True):
            return False, f"{distributed} != serial {serial}"
        if distributed[0] != 0.0:
            return False, f"early zero gave {distributed[0]!r}, want 0.0"
        if not np.isnan(distributed[2]):
            return False, f"NaN column gave {distributed[2]!r}"
        if distributed[4] != np.float32(1.0):
            return False, f"all-ones column gave {distributed[4]!r}"
        return True, ""

    _check("mpp_reproducing_prod", "overflow with a zero, vs serial", reproducing_prod)

    def prod_decompose_roundtrip() -> tuple[bool, str]:
        """decompose + recombine on one rank must equal a plain product."""
        rng = np.random.default_rng(11)
        field = rng.standard_normal((40, 5))
        mantissa, companions = mpp_prod_decompose(field, 0)
        rebuilt = mpp_prod_recombine(mantissa, companions)
        plain = field.prod(axis=0)
        if not np.allclose(rebuilt, plain, rtol=1e-12):
            return False, f"{rebuilt} vs {plain}"
        signs_ok = np.array_equal(np.signbit(rebuilt), np.signbit(plain))
        return signs_ok, "sign disagreement"

    _check(
        "mpp_prod_decompose", "round-trips to a plain product", prod_decompose_roundtrip
    )

    # ------------------------------------------------------------------
    # Checksums: the point is invariance, so the same global field is
    # checksummed from this rank count and compared against the one-rank
    # value computed locally.
    # ------------------------------------------------------------------
    def chksum_invariant() -> tuple[bool, str]:
        length = 8192
        rng = np.random.default_rng(3)
        field = (rng.standard_normal(length) * 100.0).astype(np.float32)
        field[5] = np.nan
        start, stop = _local_slice(length)
        distributed = mpp_chksum(field[start:stop], comm)
        serial = mpp_chksum(field)
        if distributed != serial:
            return False, f"{distributed} != serial {serial}"
        masked = mpp_chksum(field[start:stop], comm, mask_val=float("nan"))
        if masked == distributed:
            return False, "mask_val=nan did not exclude the NaN"
        if masked != mpp_chksum(field, mask_val=float("nan")):
            return False, "masked checksum is not rank-count invariant"
        return True, ""

    _check("mpp_chksum", "invariant to rank count", chksum_invariant)

    def chksum_detects_change() -> tuple[bool, str]:
        """A checksum that misses a one-bit change is not worth computing."""
        length = 4096
        field = np.arange(length, dtype=np.float64)
        start, stop = _local_slice(length)
        before = mpp_chksum(field[start:stop], comm)
        perturbed = field.copy()
        perturbed[length // 2] = np.nextafter(perturbed[length // 2], np.inf)
        after = mpp_chksum(perturbed[start:stop], comm)
        return before != after, "one-ulp change left the checksum unchanged"

    _check("mpp_chksum", "detects a one-ulp change", chksum_detects_change)

    # ------------------------------------------------------------------
    # Halo exchange, at the primitive level.
    # ------------------------------------------------------------------
    def halo_update() -> tuple[bool, str]:
        """Received halos must equal the neighbours' edge rows."""
        if nranks == 1:
            return None, "no neighbours at one rank"
        local_length, ncols, halo = 12, 3, 2
        start, _ = _local_slice(local_length * nranks)
        base = np.arange(
            comm.rank * local_length * ncols,
            (comm.rank + 1) * local_length * ncols,
            dtype=np.float64,
        ).reshape(local_length, ncols)
        domain = Domain(
            dims=("x",),
            global_sizes={"x": local_length * nranks},
            starts={"x": comm.rank * local_length},
            stops={"x": (comm.rank + 1) * local_length},
            comm=comm,
        )
        left = comm.rank - 1 if comm.rank > 0 else None
        right = comm.rank + 1 if comm.rank < nranks - 1 else None
        update = mpp_start_update_domains(
            {"v": base},
            domain,
            "x",
            0,
            before=halo,
            after=halo,
            left_rank=left,
            right_rank=right,
        )
        got_before, got_after, left_pad, right_pad = mpp_complete_update_domains(update)

        if (left is None) != (left_pad == 0):
            return False, f"left_pad={left_pad} at rank {comm.rank}, left={left}"
        if (right is None) != (right_pad == 0):
            return False, f"right_pad={right_pad} at rank {comm.rank}, right={right}"

        if left is not None:
            # The lower neighbour's last `halo` rows, reconstructed from the
            # same closed-form numbering rather than from what it sent.
            expected = np.arange(
                left * local_length * ncols,
                (left + 1) * local_length * ncols,
                dtype=np.float64,
            ).reshape(local_length, ncols)[-halo:]
            if not np.array_equal(got_before["v"], expected):
                return False, f"before-halo mismatch on rank {comm.rank}"
        if right is not None:
            expected = np.arange(
                right * local_length * ncols,
                (right + 1) * local_length * ncols,
                dtype=np.float64,
            ).reshape(local_length, ncols)[:halo]
            if not np.array_equal(got_after["v"], expected):
                return False, f"after-halo mismatch on rank {comm.rank}"
        del start
        return True, ""

    _check("mpp_update_domains", "halos match the neighbours", halo_update)

    def halo_group_update() -> tuple[bool, str]:
        """A group update must not mix fields of different dtypes."""
        if nranks == 1:
            return None, "no neighbours at one rank"
        local_length, halo = 8, 1
        floats = np.full((local_length, 2), float(comm.rank), dtype=np.float64)
        ints = np.full((local_length, 2), comm.rank, dtype=np.int32)
        domain = Domain(
            dims=("x",),
            global_sizes={"x": local_length * nranks},
            starts={"x": comm.rank * local_length},
            stops={"x": (comm.rank + 1) * local_length},
            comm=comm,
        )
        left = comm.rank - 1 if comm.rank > 0 else None
        right = comm.rank + 1 if comm.rank < nranks - 1 else None
        update = mpp_start_update_domains(
            {"f": floats, "i": ints},
            domain,
            "x",
            0,
            before=halo,
            after=halo,
            left_rank=left,
            right_rank=right,
        )
        got_before, got_after, _, _ = mpp_complete_update_domains(update)
        if left is not None:
            if got_before["f"].dtype != np.float64 or got_before["i"].dtype != np.int32:
                return False, "dtypes not preserved through the group update"
            if not np.all(got_before["f"] == float(left)):
                return False, "float field carried the wrong neighbour's data"
            if not np.all(got_before["i"] == left):
                return False, "int field carried the wrong neighbour's data"
        if right is not None and not np.all(got_after["i"] == right):
            return False, "upper int halo carried the wrong neighbour's data"
        return True, ""

    _check("mpp_update_domains", "mixed-dtype group update", halo_group_update)

    mpi.comm.barrier()
