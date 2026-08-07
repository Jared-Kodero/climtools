from __future__ import annotations

import datetime
import sys
import threading
from collections.abc import Iterable, Iterator
from typing import Any

from dask.diagnostics import ProgressBar

from .tools import apply_widget_css

apply_widget_css()


class DaskProgressBar(ProgressBar):
    """
    Dask progress bar styled like rich.progress.track.

    Requires:
        pip install rich
    """

    def __init__(
        self,
        description: str | None = None,
        transient: bool = False,
        refresh_per_second: int = 10,
        step_pct: int = 10,
        stdout: Any = None,
    ) -> None:
        super().__init__()

        self.transient = transient
        self.refresh_per_second = refresh_per_second

        self._progress = None
        self._task_id = None
        self._total = 0
        self._completed = 0
        self.description = description + ":" if description else ""
        self._isatty = sys.stdout.isatty()
        self._interactive = "ipykernel" in sys.modules or self._isatty
        self.step_pct = step_pct  # new kwarg, e.g. step_pct: int = 10
        self._stream = stdout or sys.stdout  # new kwarg, e.g. file=None
        self._last_emitted = -1
        self._lock = threading.Lock()
        self._wrote_header = False
        self._start_time = datetime.datetime.now(tz=None)
        self._elapsed = None
        self.step_pct = 1 if self._isatty else step_pct

    def _interactive_start(self, dsk) -> bool:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._total = len(dsk)
        self._completed = 0

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=self.transient,
            refresh_per_second=self.refresh_per_second,
        )

        self._progress.start()
        self._task_id = self._progress.add_task(
            self.description,
            total=self._total,
        )

    def _fd_start(self, dsk) -> None:
        """Initialise milestone-based percent printing for a non-tty stream."""
        self._total = len(dsk)
        self._completed = 0
        self._last_emitted = -1
        if self._total == 0:
            return
        with self._lock:
            if not self._wrote_header:
                self.description = self.description or ""

                print(
                    f"{self.description}",
                    end=" ",
                    file=self._stream,
                    flush=True,
                )
                self._wrote_header = True

    def _elapsed_time(self):
        elapsed = datetime.datetime.now() - self._start_time
        total_seconds = elapsed.total_seconds()

        if total_seconds < 1:
            self._elapsed = f"{total_seconds * 1000:.0f} ms"
            return self._elapsed

        total_seconds = int(round(total_seconds))

        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        self._elapsed = f"{hours:02}:{minutes:02}:{seconds:02}"

    def _emit_pct(self) -> None:
        """Write the current milestone in place, no newline. Caller holds no lock."""
        if self._total == 0:
            return
        pct = (100 * self._completed) // self._total
        milestone = (pct // self.step_pct) * self.step_pct
        with self._lock:
            if milestone <= self._last_emitted:
                return

            self._last_emitted = milestone
            end_char = "\n" if milestone >= 100 else " "

            self._elapsed_time()
            if milestone >= 100 and self._elapsed is not None:
                msg = f"{milestone}% [Finished in: {self._elapsed}]"
            else:
                msg = f"{milestone}%"

            print(msg, end=end_char, file=self._stream, flush=True)

    def _start(self, dsk: Any) -> None:
        if self._interactive:
            self._interactive_start(dsk)
        elif not self._isatty:
            self._fd_start(dsk)

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        if self._interactive:
            self._completed += 1
            if self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, completed=self._completed)
        elif not self._isatty:
            self._completed += 1
            self._emit_pct()

    def _finish(self, dsk, state, errored) -> None:
        if not errored:
            if self._interactive:
                if self._progress is not None and self._task_id is not None:
                    self._completed = self._total
                    self._progress.update(
                        self._task_id, completed=self._total, refresh=True
                    )
                    self._progress.refresh()
            elif not self._isatty:
                with self._lock:
                    if self._last_emitted < 100:
                        self._elapsed_time()
                        print(
                            f"100% [Finished in: {self._elapsed}]",
                            file=self._stream,
                            flush=True,
                        )
        else:
            print(file=self._stream, flush=True)

        if self._progress is not None:
            self._progress.stop()


class SerialProgressBar:
    """
    Serial-loop progress reporter behavior.

    Interactive sessions (ipykernel or tty) render a rich progress bar.
    Non-tty streams receive milestone percentages at step_pct intervals.

    Requires:
        pip install rich
    """

    def __init__(
        self,
        iterable: Iterable | None = None,
        total: int | None = None,
        description: str = "",
        transient: bool = False,
        refresh_per_second: int = 10,
        step_pct: int = 10,
        stdout: Any = None,
    ) -> None:
        self.iterable = iterable
        self.transient = transient
        self.refresh_per_second = refresh_per_second

        if total is not None:
            self._total = total
        elif iterable is not None and hasattr(iterable, "__len__"):
            self._total = len(iterable)
        else:
            self._total = 0

        self._completed = 0
        self._progress = None
        self._task_id = None

        self.description = description + ":" if description else ""
        self._isatty = sys.stdout.isatty()
        self._interactive = "ipykernel" in sys.modules or self._isatty
        self._stream = stdout or sys.stdout
        self._last_emitted = -1
        self._lock = threading.Lock()
        self._wrote_header = False
        self._started = False
        self._start_time = datetime.datetime.now()
        self._elapsed = None
        self.step_pct = 1 if self._isatty else step_pct

    def _interactive_start(self) -> None:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=self.transient,
            refresh_per_second=self.refresh_per_second,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            self.description,
            total=self._total if self._total > 0 else None,
        )

    def _fd_start(self) -> None:
        self._last_emitted = -1
        if self._total == 0:
            return
        with self._lock:
            if not self._wrote_header:
                self.description = self.description or ""
                print(
                    f"{self.description}",
                    end=" ",
                    file=self._stream,
                    flush=True,
                )
                self._wrote_header = True

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._interactive:
            self._interactive_start()
        elif not self._isatty:
            self._fd_start()

    def _elapsed_time(self):
        elapsed = datetime.datetime.now() - self._start_time
        total_seconds = elapsed.total_seconds()

        if total_seconds < 1:
            self._elapsed = f"{total_seconds * 1000:.0f} ms"
            return self._elapsed

        total_seconds = int(round(total_seconds))

        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        self._elapsed = f"{hours:02}:{minutes:02}:{seconds:02}"

    def _emit_pct(self) -> None:
        if self._total == 0:
            return
        pct = (100 * self._completed) // self._total
        milestone = (pct // self.step_pct) * self.step_pct
        with self._lock:
            if milestone <= self._last_emitted:
                return

            self._last_emitted = milestone
            end_char = "\n" if milestone >= 100 else " "

            self._elapsed_time()
            if milestone >= 100 and self._elapsed is not None:
                msg = f"{milestone}% [Finished in: {self._elapsed}]"
            else:
                msg = f"{milestone}%"

            print(msg, end=end_char, file=self._stream, flush=True)

    def update(self, n: int = 1) -> None:
        """Advance the counter by n steps."""
        self._completed += n
        if self._interactive:
            if self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, completed=self._completed)
        elif not self._isatty:
            self._emit_pct()

    def _finish(self, errored: bool = False) -> None:
        if not errored:
            if self._interactive:
                if self._progress is not None and self._task_id is not None:
                    completed = self._total if self._total > 0 else self._completed
                    self._progress.update(
                        self._task_id, completed=completed, refresh=True
                    )
                    self._progress.refresh()
            elif not self._isatty:
                with self._lock:
                    if self._total > 0 and self._last_emitted < 100:
                        self._elapsed_time()
                        print(
                            f"100% [Finished in: {self._elapsed}]",
                            file=self._stream,
                            flush=True,
                        )
        else:
            print(file=self._stream, flush=True)

        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def __iter__(self) -> Iterator:
        if self.iterable is None:
            raise ValueError("No iterable provided to wrap.")
        self._start()
        errored = False
        try:
            for item in self.iterable:
                yield item
                self.update()
        except BaseException:
            errored = True
            raise
        finally:
            self._finish(errored=errored)

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._finish(errored=exc_type is not None)
        return False
