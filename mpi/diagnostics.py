import builtins
import datetime
import faulthandler
import json
import os
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from mpi4py import MPI

from ..core.utils import LockFile, tmp


class MPIError(Exception):
    """MPI runtime or synchronized distributed-execution error."""


class MPIDiagnostics:
    """Diagnostics and error-handling utilities for an MPI runtime."""

    def log(
        self,
        message: str,
        *args: Any,
        root: int = 0,
        timestamp: bool = False,
        prefix: bool = True,
        logger: Callable[..., None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Emit a message from a specific MPI rank.

        Parameters
        ----------
        message : str
            Message or format string.
        *args : Any
            Passed to ``logger``, or used for %-formatting when no logger.
        root : int, optional
            Rank allowed to log. -1 logs on every rank. Default 0.
        timestamp : bool, optional
            Prepend a timestamp when using ``print``. Default False.
        prefix : bool, optional
            Prepend an ``[MPI RANK n]`` tag. Default True.
        logger : callable, optional
            Callable used instead of ``print``. Default None.
        **kwargs : Any
            Forwarded to ``logger`` or ``print``.
        """
        if root != -1 and not self.is_root(root):
            return

        current_rank = root if root != -1 else self.comm.rank
        mpi_str = f"[MPI RANK {current_rank:{len(str(self.comm.size))}d}]"

        if logger is None:
            if args:
                message = message % args

            msg_prefix = ""

            if timestamp:
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg_prefix += f"{time_str} - "

            if prefix:
                msg_prefix += f"{mpi_str} "

            kwargs.setdefault("flush", True)

            with self._mpi_lock:
                print(f"{msg_prefix}{message}", **kwargs)

        else:
            if prefix:
                message = f"{mpi_str} {message}"

            with self._mpi_lock:
                logger(message, *args, **kwargs)

    @contextmanager
    def watchdog(
        self,
        phase: str = "",
        timeout: float = 3600.0,
        *,
        abort: bool = True,
    ) -> Generator[None, None, None]:
        """Dump every rank's stack if the enclosed block stalls.

        Parameters
        ----------
        phase : str, optional
            Label reported with the traceback dump.
        timeout : float, optional
            Seconds of no progress before dumping. <= 0 disables the watchdog.
        abort : bool, optional
            Call ``MPI_Abort`` after dumping. Default True.

        Yields
        ------
        None
        """
        if timeout <= 0.0:
            yield
            return

        finished = threading.Event()
        rank = self.comm.rank
        label = phase or "unnamed phase"

        def _fire() -> None:
            if finished.wait(timeout):
                return

            sys.stderr.write(
                f"\n[MPI RANK {rank}] WATCHDOG: no progress for {timeout:g} s "
                + f"at {label}. Stack for this rank follows.\n"
            )
            sys.stderr.flush()

            faulthandler.dump_traceback(
                file=sys.stderr,
                all_threads=True,
            )
            sys.stderr.flush()

            if abort:
                time.sleep(5.0 + 0.25 * (self.comm.size - 1))

                sys.stderr.write(
                    f"[MPI RANK {rank}] WATCHDOG: aborting MPI_COMM_WORLD.\n"
                )
                sys.stderr.flush()

                self.comm.Abort(1)

        thread = threading.Thread(
            target=_fire,
            name=f"climtools-mpi-watchdog-{rank}",
            daemon=True,
        )
        thread.start()

        try:
            yield
        finally:
            finished.set()

    def raise_if_error(
        self,
        error: BaseException | None,
        phase: str,
        signature: Any = None,
    ) -> None:
        """Raise consistently if any MPI rank reports an error."""
        detail = None if error is None else (type(error).__name__, str(error))

        states = self.comm.allgather((detail, signature))

        failures = [
            (rank, item) for rank, (item, _) in enumerate(states) if item is not None
        ]

        if not failures:
            signatures = [item for _, item in states]

            if builtins.any(item != signatures[0] for item in signatures[1:]):
                disagreeing = [
                    rank
                    for rank, item in enumerate(signatures)
                    if item != signatures[0]
                ]

                raise MPIError(
                    f"MPI ranks posted different collectives during {phase}. "
                    + f"Ranks {disagreeing} disagree with rank 0 "
                    + f"({signatures[0]!r} on rank 0, "
                    + f"{signatures[disagreeing[0]]!r} on rank "
                    + f"{disagreeing[0]})."
                )

            return

        if len(failures) == self.comm.size and error is not None:
            raise error

        rank, detail = failures[0]
        name, message = detail

        raise MPIError(f"Rank {rank} failed during {phase} with {name}: {message}")

    @staticmethod
    def _format_ranks(ranks: list[int]) -> str:
        """Compress a sorted rank list into comma-joined contiguous spans."""
        spans: list[tuple[int, int]] = []
        start = prev = ranks[0]

        for r in ranks[1:]:
            if r == prev + 1:
                prev = r
                continue

            spans.append((start, prev))
            start = prev = r

        spans.append((start, prev))

        parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in spans]
        noun = "Rank" if len(ranks) == 1 else "Ranks"

        return f"{noun} " + ", ".join(parts)

    def _install_abort_hook(self) -> bool:
        """Install deduplicated reporting for uncaught MPI exceptions."""
        if getattr(sys.excepthook, "_climtools_mpi_abort", False):
            return False

        if not self.alive(MPI.COMM_WORLD):
            return False

        error_name = f"{uuid.uuid4().hex}.error" if MPI.COMM_WORLD.rank == 0 else None
        error_name = MPI.COMM_WORLD.bcast(error_name, root=0)

        error_file = tmp / error_name
        error_lock = LockFile(tmp / f"{error_name}.lock")
        finished_file = tmp / f"{error_name}.done"

        def _abort_excepthook(
            exc_type: type[BaseException],
            exc_value: BaseException,
            tb: Any,
        ) -> None:
            record = {
                "rank": MPI.COMM_WORLD.rank,
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": "".join(
                    traceback.format_exception(
                        exc_type,
                        exc_value,
                        tb,
                    )
                ),
            }

            try:
                with error_lock:
                    reporter = not error_file.exists()

                    with error_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                if not reporter:
                    while not finished_file.exists():
                        time.sleep(5)

                    return

                time.sleep(5)

                records: list[dict[str, Any]] = []

                with (
                    error_lock,
                    error_file.open("r", encoding="utf-8") as f,
                ):
                    for line in f:
                        records.append(json.loads(line))

                groups: dict[
                    tuple[str, str],
                    tuple[list[int], str],
                ] = {}

                for item in records:
                    key = (
                        item["type"],
                        item["message"],
                    )

                    if key not in groups:
                        groups[key] = (
                            [],
                            item["traceback"],
                        )

                    groups[key][0].append(item["rank"])

                for (name, _), (ranks, traceback_text) in groups.items():
                    ranks.sort()

                    label = self._format_ranks(ranks)

                    sys.stderr.write(f"\n({label}) got {name}\n")
                    sys.stderr.write(traceback_text)

                sys.stderr.flush()
                finished_file.touch()

            except Exception:
                sys.__excepthook__(
                    exc_type,
                    exc_value,
                    tb,
                )
                sys.stderr.flush()

            finally:
                MPI.COMM_WORLD.Abort(1)

        _abort_excepthook._climtools_mpi_abort = True  # type: ignore[attr-defined]
        sys.excepthook = _abort_excepthook

        return True

    def missing_pnetcdf(self) -> None:
        """Print a one-time root-only hint if parallel NetCDF-4 is missing."""
        comm = self.comm

        if comm.Get_size() <= 1 or comm.Get_rank() != 0:
            return

        try:
            import netCDF4

            if netCDF4.__has_parallel4_support__:
                return
        except Exception:
            return

        sys.stderr.write(
            "[climtools] netCDF4 is not built with parallel NetCDF-4/HDF5 "
            + "support, so xgeo.to_netcdf(..., parallel=True) will raise on "
            + f"this {comm.Get_size()}-rank run. mpi.xarray and the rest of the "
            + "MPI runtime are unaffected. To build the parallel stack, "
            + "run `env/setup_env.sh` from the climtools repository (see the "
            + "README's Installation section); nothing else needs it.\n"
        )
        sys.stderr.flush()
