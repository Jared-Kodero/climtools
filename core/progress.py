from __future__ import annotations

import datetime
import os
import shutil
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


class BaseProgress:
    """
    Provide shared progress state and output handling.

    Parameters
    ----------
    description : str, optional
        Text shown before the progress indicator.
    step_pct : int, optional
        Percentage interval for non-interactive milestone output.
    file : SupportsWrite[str] or None, optional
        Stream used for progress output. Defaults to ``sys.stdout``.
    lockfile : LockFile or Any or None, optional
        Cross-process lock used to serialize progress writes.
    """

    def __init__(
        self,
        description: str = "",
        step_pct: int = 10,
        file: SupportsWrite[str] | None = None,
        lockfile: LockFile | Any | None = None,
    ) -> None:
        self._progress = None
        self._task_id = None
        self._total = 0
        self._completed = 0
        self._description = f"{description}:" if description else ""
        self._stream = file or sys.stdout
        self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._interactive = "ipykernel" in sys.modules or self._isatty
        self._step_pct = 1 if self._isatty else step_pct
        self._last_emitted = -1
        self._thread_lock = threading.Lock()
        self._process_lock = lockfile or LockFile()
        self._wrote_header = False
        self._start_time = datetime.datetime.now(tz=None)
        self._elapsed = None
        self._progress_parts: list[str] = []
        self._progress_stream = None
        self._owns_progress_stream = False
        self._new_redirect()

    def _new_redirect(self) -> None:
        self._temp_path = tmp / f".progress/{uuid.uuid4().hex}"
        self._temp_path.parent.mkdir(exist_ok=True, parents=True)
        self._redirect_ctx = RedirectStreams(
            stdout_target=self._temp_path,
            stderr_target=self._temp_path,
            fd_level=True,
            truncate=False,
        )

    def _open_progress_stream(self) -> None:
        """Preserve the original output stream for live progress."""
        if self._progress_stream is not None:
            return

        try:
            progress_fd = os.dup(self._stream.fileno())
        except (AttributeError, OSError, ValueError):
            self._progress_stream = self._stream
            return

        encoding = getattr(self._stream, "encoding", None) or "utf-8"
        errors = getattr(self._stream, "errors", None) or "strict"
        self._progress_stream = os.fdopen(
            progress_fd,
            "w",
            encoding=encoding,
            errors=errors,
            buffering=1,
        )
        self._owns_progress_stream = True

    def _close_progress_stream(self) -> None:
        if self._progress_stream is None:
            return

        if self._owns_progress_stream:
            self._progress_stream.close()

        self._progress_stream = None
        self._owns_progress_stream = False

    def _safe_print(self, *args, **kwargs) -> None:
        """Print safely to the current progress stream."""
        if "file" not in kwargs:
            kwargs["file"] = self._progress_stream or self._stream

        with self._thread_lock, self._process_lock:
            print(*args, **kwargs)

    def _elapsed_time(self) -> str:
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
        """Write the current non-interactive progress milestone."""
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

        self._safe_print(msg, end=end_char, flush=True)
        self._progress_parts.append(f"{msg}{end_char}")

    def _start_file_progress(self, total: int) -> None:
        """Keep progress live in the output file and redirect runtime output to temp."""
        self._total = total
        self._completed = 0
        self._last_emitted = -1
        self._progress_parts = []

        self._stream.flush()
        self._open_progress_stream()
        progress_fd = self._progress_stream.fileno()
        progress_path = f"/proc/self/fd/{progress_fd}"

        with open(progress_path, "rb") as source, self._temp_path.open("wb") as target:
            shutil.copyfileobj(source, target)

        self._redirect_ctx.redirect()

        if self._total == 0 or self._wrote_header:
            return

        header = f"{self._description} "
        self._safe_print(self._description, end=" ", flush=True)
        self._progress_parts.append(header)
        self._wrote_header = True

    def _finish_file_progress(self, errored: bool) -> None:
        if not errored:
            if self._total > 0 and self._last_emitted < 100:
                self._elapsed_time()
                msg = f"100% [Finished in: {self._elapsed}]"
                self._safe_print(msg, flush=True)
                self._progress_parts.append(f"{msg}\n")
        else:
            self._safe_print(flush=True)
            self._progress_parts.append("\n")

        self._redirect_ctx.restore()

        if self._progress_stream is None:
            return

        try:
            self._progress_stream.flush()
            progress_fd = self._progress_stream.fileno()
            progress_path = f"/proc/self/fd/{progress_fd}"
            encoding = self._progress_stream.encoding or "utf-8"
            errors = self._progress_stream.errors or "strict"

            with self._temp_path.open("ab") as target:
                target.write("".join(self._progress_parts).encode(encoding, errors))

            with self._thread_lock, self._process_lock:
                with (
                    self._temp_path.open("rb") as source,
                    open(progress_path, "wb") as target,
                ):
                    shutil.copyfileobj(source, target)
                    target.flush()

                os.lseek(progress_fd, 0, os.SEEK_END)
        finally:
            self._close_progress_stream()
            self._temp_path.unlink(missing_ok=True)

    def _finish_interactive_output(self) -> None:
        self._redirect_ctx.restore()

        try:
            if self._temp_path.exists() and self._temp_path.stat().st_size > 0:
                content = self._temp_path.read_text(encoding="utf-8")
                self._safe_print(content, end="", flush=True)
        finally:
            self._close_progress_stream()
            self._temp_path.unlink(missing_ok=True)

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def _finish_output(self, errored: bool) -> None:
        self._stop_progress()

        if self._interactive:
            self._finish_interactive_output()
        elif not self._isatty:
            self._finish_file_progress(errored)


class DaskProgressBar(BaseProgress, ProgressBar):
    """
    Report Dask task progress with Rich-style interactive rendering.

    Parameters
    ----------
    description : str or None, optional
        Text shown before the progress indicator.
    transient : bool, optional
        Remove the interactive Rich display after completion when ``True``.
    refresh_per_second : int, optional
        Maximum Rich display refresh rate.
    step_pct : int, optional
        Percentage interval for non-interactive milestone output.
    file : SupportsWrite[str] or None, optional
        Stream used for progress output. Defaults to ``sys.stdout``.
    lockfile : LockFile or Any or None, optional
        Cross-process lock used to serialize progress writes.

    Notes
    -----
    Interactive rendering requires ``rich``.
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
        ProgressBar.__init__(self)
        BaseProgress.__init__(
            self,
            description=description or "",
            step_pct=step_pct,
            file=file,
            lockfile=lockfile,
        )
        self.transient = transient
        self.refresh_per_second = refresh_per_second

    def _interactive_start(self, dsk: Any) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._total = len(dsk)
        self._completed = 0

        console = Console(
            file=self._progress_stream,
            force_terminal=self._isatty,
        )
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=self.transient,
            refresh_per_second=self.refresh_per_second,
            console=console,
            redirect_stdout=False,
            redirect_stderr=False,
        )

        self._progress.start()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total,
        )

    def _start(self, dsk: Any) -> None:
        self._start_time = datetime.datetime.now(tz=None)
        self._new_redirect()

        if self._interactive:
            self._open_progress_stream()
            self._redirect_ctx.redirect()
            self._interactive_start(dsk)
        elif not self._isatty:
            self._start_file_progress(len(dsk))

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        self._completed += 1

        if self._interactive:
            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    completed=self._completed,
                )
        elif not self._isatty:
            self._emit_pct()

    def _finish(self, dsk, state, errored) -> None:
        if not errored and self._interactive:
            if self._progress is not None and self._task_id is not None:
                self._completed = self._total
                self._progress.update(
                    self._task_id,
                    completed=self._total,
                    refresh=True,
                )
                self._progress.refresh()

        self._finish_output(errored)


class SerialProgressBar(BaseProgress):
    """
    Report progress for serial iteration or manual updates.

    Parameters
    ----------
    iterable : Iterable or None, optional
        Iterable to wrap when using the object as an iterator.
    total : int or None, optional
        Total number of steps for manual updates. If omitted, the length of
        ``iterable`` is used when available.
    description : str, optional
        Text shown before the progress indicator.
    transient : bool, optional
        Remove the interactive Rich display after completion when ``True``.
    refresh_per_second : int, optional
        Maximum Rich display refresh rate.
    step_pct : int, optional
        Percentage interval for non-interactive milestone output.
    file : SupportsWrite[str] or None, optional
        Stream used for progress output. Defaults to ``sys.stdout``.
    lockfile : LockFile or Any or None, optional
        Cross-process lock used to serialize progress writes.
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
        BaseProgress.__init__(
            self,
            description=description,
            step_pct=step_pct,
            file=file,
            lockfile=lockfile,
        )

        self._iterable = iterable
        self._transient = transient
        self._refresh_per_second = refresh_per_second
        self._started = False

        if total is not None:
            self._total = total
        elif iterable is not None and hasattr(iterable, "__len__"):
            self._total = len(iterable)

    def _interactive_start(self) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        console = Console(
            file=self._progress_stream,
            force_terminal=self._isatty,
        )
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=self._transient,
            refresh_per_second=self._refresh_per_second,
            console=console,
            redirect_stdout=False,
            redirect_stderr=False,
        )

        self._progress.start()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total if self._total > 0 else None,
        )

    def _start(self) -> None:
        if self._started:
            return

        self._started = True
        self._start_time = datetime.datetime.now(tz=None)
        self._new_redirect()

        if self._interactive:
            self._open_progress_stream()
            self._redirect_ctx.redirect()
            self._interactive_start()
        elif not self._isatty:
            self._start_file_progress(self._total)

    def update(self, n: int = 1) -> None:
        """
        Advance the progress counter.

        Parameters
        ----------
        n : int, optional
            Number of completed steps to add. Defaults to 1.
        """
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
        if not errored and self._interactive:
            if self._progress is not None and self._task_id is not None:
                completed = self._total if self._total > 0 else self._completed
                self._progress.update(
                    self._task_id,
                    completed=completed,
                    refresh=True,
                )
                self._progress.refresh()

        self._finish_output(errored)

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
