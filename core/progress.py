from __future__ import annotations

import datetime
import sys
import threading
import uuid
from typing import TYPE_CHECKING

from dask.diagnostics import ProgressBar

from .tools import LockFile, RedirectStreams, tmp

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any

    from _typeshed import SupportsWrite


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
        file: SupportsWrite[str] | None = None,
        lockfile: LockFile | Any | None = None,
    ) -> None:
        super().__init__()

        self.transient = transient
        self.refresh_per_second = refresh_per_second

        self._progress = None
        self._task_id = None
        self._total = 0
        self._completed = 0
        self._description = description + ":" if description else ""
        self._isatty = sys.stdout.isatty()
        self._interactive = "ipykernel" in sys.modules or self._isatty
        self._step_pct = 1 if self._isatty else step_pct
        self._last_emitted = -1
        self._thread_lock = threading.Lock()
        self._process_lock = lockfile or LockFile()
        self._wrote_header = False
        self._start_time = datetime.datetime.now(tz=None)
        self._elapsed = None

        self._temp_path = tmp / f".progress/{uuid.uuid4().hex}"
        self._temp_path.parent.mkdir(exist_ok=True, parents=True)

        self._redirect_ctx = RedirectStreams(
            stdout_target=self._temp_path,
            stderr_target=self._temp_path,
        )
        self._stream = file or self._redirect_ctx.original_streams[0]

        self._progress_start_pos: int | None = None
        self._progress_parts: list[str] = []

    def _interactive_start(self, dsk) -> None:
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
            console=None,
        )

        self._progress.start()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total,
        )

    def _safe_print(self, *args, **kwargs) -> None:
        """Print safely to the designated stream."""
        if "file" not in kwargs:
            kwargs["file"] = self._stream

        with self._thread_lock, self._process_lock:
            print(*args, **kwargs)

    def _fd_start(self, dsk) -> None:
        """Initialise milestone-based percent printing for a non-TTY stream."""
        self._total = len(dsk)
        self._completed = 0
        self._last_emitted = -1
        self._progress_parts = []
        self._progress_start_pos = None

        if self._total == 0:
            return

        try:
            if self._stream.seekable():
                self._progress_start_pos = self._stream.tell()
        except (AttributeError, OSError, ValueError):
            self._progress_start_pos = None

        if not self._wrote_header:
            self._description = self._description or ""

            header = f"{self._description} "

            self._safe_print(
                self._description,
                end=" ",
                flush=True,
            )
            self._progress_parts.append(header)
            self._wrote_header = True

    def _elapsed_time(self):
        elapsed = datetime.datetime.now(tz=None) - self._start_time
        total_seconds = elapsed.total_seconds()

        if total_seconds < 1:
            self._elapsed = f"{total_seconds * 1000:.0f} ms"
            return self._elapsed

        total_seconds = round(total_seconds)

        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        self._elapsed = f"{hours:02}:{minutes:02}:{seconds:02}"

        return self._elapsed

    def _emit_pct(self) -> None:
        """Write the current milestone."""
        if self._total == 0:
            return

        pct = (100 * self._completed) // self._total
        milestone = (pct // self._step_pct) * self._step_pct

        if milestone <= self._last_emitted:
            return

        self._last_emitted = milestone
        end_char = "\n" if milestone >= 100 else " "

        self._elapsed_time()

        if milestone >= 100 and self._elapsed is not None:
            msg = f"{milestone}% [Finished in: {self._elapsed}]"
        else:
            msg = f"{milestone}%"

        self._safe_print(
            msg,
            end=end_char,
            flush=True,
        )
        self._progress_parts.append(f"{msg}{end_char}")

    def _start(self, dsk: Any) -> None:
        self._start_time = datetime.datetime.now(tz=None)

        self._temp_path = tmp / f".progress/{uuid.uuid4().hex}"
        self._temp_path.parent.mkdir(exist_ok=True, parents=True)

        self._redirect_ctx = RedirectStreams(
            stdout_target=self._temp_path,
            stderr_target=self._temp_path,
        )
        self._redirect_ctx.redirect()

        if self._interactive:
            self._interactive_start(dsk)
        elif not self._isatty:
            self._fd_start(dsk)

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        if self._interactive:
            self._completed += 1

            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    completed=self._completed,
                )

        elif not self._isatty:
            self._completed += 1
            self._emit_pct()

    def _finish(self, dsk, state, errored) -> None:
        if not errored:
            if self._interactive:
                if self._progress is not None and self._task_id is not None:
                    self._completed = self._total

                    self._progress.update(
                        self._task_id,
                        completed=self._total,
                        refresh=True,
                    )
                    self._progress.refresh()

            elif not self._isatty:
                if self._last_emitted < 100:
                    self._elapsed_time()

                    msg = f"100% [Finished in: {self._elapsed}]"

                    self._safe_print(
                        msg,
                        flush=True,
                    )
                    self._progress_parts.append(f"{msg}\n")

        elif not self._interactive:
            self._safe_print(flush=True)
            self._progress_parts.append("\n")

        if self._progress is not None:
            self._progress.stop()
            self._progress = None

        if self._redirect_ctx is not None:
            self._redirect_ctx.restore()

        if self._temp_path is None or not self._temp_path.exists():
            return

        try:
            content = ""

            if self._temp_path.stat().st_size > 0:
                content = self._temp_path.read_text(encoding="utf-8")

            progress_text = "".join(self._progress_parts)

            can_reorder = (
                not self._interactive
                and self._progress_start_pos is not None
                and hasattr(self._stream, "seek")
                and hasattr(self._stream, "truncate")
                and hasattr(self._stream, "write")
            )

            if can_reorder:
                with self._thread_lock, self._process_lock:
                    self._stream.flush()
                    self._stream.seek(self._progress_start_pos)
                    self._stream.truncate()

                    if content:
                        self._stream.write(content)

                    if progress_text:
                        self._stream.write(progress_text)

                    self._stream.flush()

            elif content:
                with self._thread_lock, self._process_lock:
                    print(
                        content,
                        end="",
                        file=self._stream,
                        flush=True,
                    )

        finally:
            self._temp_path.unlink(missing_ok=True)


class SerialProgressBar:
    """
    Serial-loop progress reporter behavior.
    """

    def __init__(
        self,
        iterable: Iterable | None = None,
        total: int | None = None,
        description: str = "",
        transient: bool = False,
        refresh_per_second: int = 10,
        step_pct: int = 10,
        file: SupportsWrite[str] | None = None,
        lockfile: LockFile | Any | None = None,
    ) -> None:
        self._iterable = iterable
        self._transient = transient
        self._refresh_per_second = refresh_per_second

        if total is not None:
            self._total = total
        elif iterable is not None and hasattr(iterable, "__len__"):
            self._total = len(iterable)
        else:
            self._total = 0

        self._completed = 0
        self._progress = None
        self._task_id = None

        self._description = description + ":" if description else ""
        self._isatty = sys.stdout.isatty()
        self._interactive = "ipykernel" in sys.modules or self._isatty
        self._last_emitted = -1
        self._thread_lock = threading.Lock()
        self._process_lock = lockfile or LockFile()
        self._wrote_header = False
        self._started = False
        self._start_time = datetime.datetime.now(tz=None)
        self._elapsed = None
        self._step_pct = 1 if self._isatty else step_pct

        self._temp_path = tmp / f".progress/{uuid.uuid4().hex}"
        self._temp_path.parent.mkdir(exist_ok=True, parents=True)

        self._redirect_ctx = RedirectStreams(
            stdout_target=self._temp_path,
            stderr_target=self._temp_path,
        )
        self._stream = file or self._redirect_ctx.original_streams[0]

        self._progress_start_pos: int | None = None
        self._progress_parts: list[str] = []

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
            transient=self._transient,
            refresh_per_second=self._refresh_per_second,
        )

        self._progress.start()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total if self._total > 0 else None,
        )

    def _safe_print(self, *args, **kwargs) -> None:
        """Print safely to the designated stream."""
        if "file" not in kwargs:
            kwargs["file"] = self._stream

        with self._thread_lock, self._process_lock:
            print(*args, **kwargs)

    def _fd_start(self) -> None:
        self._last_emitted = -1
        self._progress_parts = []
        self._progress_start_pos = None

        if self._total == 0:
            return

        try:
            if self._stream.seekable():
                self._progress_start_pos = self._stream.tell()
        except (AttributeError, OSError, ValueError):
            self._progress_start_pos = None

        if not self._wrote_header:
            self._description = self._description or ""

            header = f"{self._description} "

            self._safe_print(
                self._description,
                end=" ",
                flush=True,
            )
            self._progress_parts.append(header)
            self._wrote_header = True

    def _start(self) -> None:
        if self._started:
            return

        self._started = True
        self._start_time = datetime.datetime.now(tz=None)

        self._temp_path = tmp / f".progress/{uuid.uuid4().hex}"
        self._temp_path.parent.mkdir(exist_ok=True, parents=True)

        self._redirect_ctx = RedirectStreams(
            stdout_target=self._temp_path,
            stderr_target=self._temp_path,
        )
        self._redirect_ctx.redirect()

        if self._interactive:
            self._interactive_start()
        elif not self._isatty:
            self._fd_start()

    def _elapsed_time(self):
        elapsed = datetime.datetime.now(tz=None) - self._start_time
        total_seconds = elapsed.total_seconds()

        if total_seconds < 1:
            self._elapsed = f"{total_seconds * 1000:.0f} ms"
            return self._elapsed

        total_seconds = round(total_seconds)

        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        self._elapsed = f"{hours:02}:{minutes:02}:{seconds:02}"

        return self._elapsed

    def _emit_pct(self) -> None:
        if self._total == 0:
            return

        pct = (100 * self._completed) // self._total
        milestone = (pct // self._step_pct) * self._step_pct

        if milestone <= self._last_emitted:
            return

        self._last_emitted = milestone
        end_char = "\n" if milestone >= 100 else " "

        self._elapsed_time()

        if milestone >= 100 and self._elapsed is not None:
            msg = f"{milestone}% [Finished in: {self._elapsed}]"
        else:
            msg = f"{milestone}%"

        self._safe_print(
            msg,
            end=end_char,
            flush=True,
        )
        self._progress_parts.append(f"{msg}{end_char}")

    def update(self, n: int = 1) -> None:
        """Advance the counter by n steps."""
        self._completed += n

        if self._interactive:
            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    completed=self._completed,
                )

        elif not self._isatty:
            self._emit_pct()

    def _finish(self, errored: bool = False) -> None:
        if not errored:
            if self._interactive:
                if self._progress is not None and self._task_id is not None:
                    completed = self._total if self._total > 0 else self._completed

                    self._progress.update(
                        self._task_id,
                        completed=completed,
                        refresh=True,
                    )
                    self._progress.refresh()

            elif not self._isatty:
                if self._total > 0 and self._last_emitted < 100:
                    self._elapsed_time()

                    msg = f"100% [Finished in: {self._elapsed}]"

                    self._safe_print(
                        msg,
                        flush=True,
                    )
                    self._progress_parts.append(f"{msg}\n")

        elif not self._interactive:
            self._safe_print(flush=True)
            self._progress_parts.append("\n")

        if self._progress is not None:
            self._progress.stop()
            self._progress = None

        if self._redirect_ctx is not None:
            self._redirect_ctx.restore()

        if self._temp_path is None or not self._temp_path.exists():
            return

        try:
            content = ""

            if self._temp_path.stat().st_size > 0:
                content = self._temp_path.read_text(encoding="utf-8")

            progress_text = "".join(self._progress_parts)

            can_reorder = (
                not self._interactive
                and self._progress_start_pos is not None
                and hasattr(self._stream, "seek")
                and hasattr(self._stream, "truncate")
                and hasattr(self._stream, "write")
            )

            if can_reorder:
                with self._thread_lock, self._process_lock:
                    self._stream.flush()
                    self._stream.seek(self._progress_start_pos)
                    self._stream.truncate()

                    if content:
                        self._stream.write(content)

                    if progress_text:
                        self._stream.write(progress_text)

                    self._stream.flush()

            elif content:
                with self._thread_lock, self._process_lock:
                    print(
                        content,
                        end="",
                        file=self._stream,
                        flush=True,
                    )

        finally:
            self._temp_path.unlink(missing_ok=True)

    def __iter__(self) -> Iterator:
        if self._iterable is None:
            raise ValueError("No iterable provided to wrap.")

        self._start()
        errored = False

        try:
            for item in self._iterable:
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
