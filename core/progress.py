from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING

from dask.diagnostics import ProgressBar

from .tools import LockFile, RedirectStreams, tmp

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any

    from _typeshed import SupportsWrite


class BaseProgress:
    """Provide shared progress state and output handling."""

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
        self._last_emitted = -1

        self._description = f"{description}:" if description else ""
        self._stream = file if file is not None else sys.stdout
        self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._interactive = "ipykernel" in sys.modules or self._isatty
        self._step_pct = 1 if self._isatty else step_pct

        self._thread_lock = threading.Lock()
        self._process_lock = lockfile or LockFile()

        self._start_time = time.monotonic()
        self._progress_parts: list[str] = []
        self._progress_stream = None
        self._wrote_header = False

        self._reset_redirect()

    def _reset_redirect(self) -> None:
        self._temp_path = tmp / ".progress" / uuid.uuid4().hex
        self._temp_path.parent.mkdir(
            exist_ok=True,
            parents=True,
        )
        self._redirect = RedirectStreams(
            stdout_target=self._temp_path,
            stderr_target=self._temp_path,
            fd_level=True,
            truncate=False,
        )

    def _preserve_progress_stream(self) -> None:
        if self._progress_stream is not None:
            return

        try:
            self._progress_stream = RedirectStreams.duplicate(self._stream)
        except (AttributeError, OSError, ValueError):
            self._progress_stream = self._stream

    def _close_progress_stream(self) -> None:
        if (
            self._progress_stream is not None
            and self._progress_stream is not self._stream
        ):
            self._progress_stream.close()

        self._progress_stream = None

    def _safe_print(self, *args, **kwargs) -> None:
        """Print safely to the live progress destination."""
        if "file" not in kwargs:
            kwargs["file"] = self._progress_stream or self._stream

        with self._thread_lock, self._process_lock:
            print(*args, **kwargs)

    def _format_elapsed(self) -> str:
        total_seconds = time.monotonic() - self._start_time

        if total_seconds < 1:
            return f"{total_seconds * 1000:.0f} ms"

        rounded_seconds = round(total_seconds)
        hours, rem = divmod(rounded_seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def _emit_milestone(self) -> None:
        if self._total <= 0:
            return

        pct = min(
            100,
            (100 * self._completed) // self._total,
        )
        milestone = (pct // self._step_pct) * self._step_pct

        if milestone <= self._last_emitted:
            return

        self._last_emitted = milestone
        end = "\n" if milestone >= 100 else " "

        msg = (
            f"{milestone}% [Finished in: {self._format_elapsed()}]"
            if milestone >= 100
            else f"{milestone}%"
        )

        self._safe_print(
            msg,
            end=end,
            flush=True,
        )
        self._progress_parts.append(f"{msg}{end}")

    def _start_redirected_output(self) -> None:
        """Keep milestones live while capturing incidental output."""
        self._preserve_progress_stream()

        if self._progress_stream is None:
            raise RuntimeError("Progress stream was not prepared.")

        self._stream.flush()

        progress_fd = self._progress_stream.fileno()
        progress_path = f"/proc/self/fd/{progress_fd}"

        with (
            open(progress_path, "rb") as source,
            self._temp_path.open("wb") as target,
        ):
            shutil.copyfileobj(
                source,
                target,
            )

        self._redirect.start()

        if self._total <= 0 or not self._description:
            return

        self._safe_print(
            self._description,
            end=" ",
            flush=True,
        )
        self._progress_parts.append(f"{self._description} ")
        self._wrote_header = True

    def _finish_redirected_output(
        self,
        errored: bool,
    ) -> None:
        if not errored:
            if self._total > 0 and self._last_emitted < 100:
                msg = f"100% [Finished in: {self._format_elapsed()}]"
                self._safe_print(
                    msg,
                    flush=True,
                )
                self._progress_parts.append(f"{msg}\n")

        elif self._wrote_header or self._last_emitted >= 0:
            self._safe_print(flush=True)
            self._progress_parts.append("\n")

        self._redirect.stop()

        if self._progress_stream is None:
            self._temp_path.unlink(missing_ok=True)
            return

        try:
            self._progress_stream.flush()

            progress_fd = self._progress_stream.fileno()
            progress_path = f"/proc/self/fd/{progress_fd}"

            encoding = (
                getattr(
                    self._progress_stream,
                    "encoding",
                    None,
                )
                or "utf-8"
            )
            errors = (
                getattr(
                    self._progress_stream,
                    "errors",
                    None,
                )
                or "strict"
            )

            progress_text = "".join(self._progress_parts)

            with self._temp_path.open("ab") as target:
                target.write(
                    progress_text.encode(
                        encoding,
                        errors,
                    )
                )

            with (
                self._thread_lock,
                self._process_lock,
            ):
                with (
                    self._temp_path.open("rb") as source,
                    open(
                        progress_path,
                        "wb",
                    ) as target,
                ):
                    shutil.copyfileobj(
                        source,
                        target,
                    )
                    target.flush()

                os.lseek(
                    progress_fd,
                    0,
                    os.SEEK_END,
                )

        finally:
            self._close_progress_stream()
            self._temp_path.unlink(missing_ok=True)

    def _start_interactive_renderer(
        self,
        *,
        transient: bool,
        refresh_per_second: int,
    ) -> None:
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
            transient=transient,
            refresh_per_second=refresh_per_second,
            console=console,
            redirect_stdout=False,
            redirect_stderr=False,
        )

        self._progress.start()

        self._task_id = self._progress.add_task(
            self._description,
            total=(self._total if self._total > 0 else None),
        )

    def _start_progress(
        self,
        *,
        total: int,
        transient: bool,
        refresh_per_second: int,
    ) -> None:
        self._start_time = time.monotonic()

        self._total = total
        self._completed = 0
        self._last_emitted = -1
        self._progress_parts = []
        self._wrote_header = False

        self._reset_redirect()

        if self._interactive:
            self._preserve_progress_stream()
            self._redirect.start()

            self._start_interactive_renderer(
                transient=transient,
                refresh_per_second=(refresh_per_second),
            )
            return

        self._start_redirected_output()

    def _advance(
        self,
        n: int = 1,
    ) -> None:
        self._completed += n

        if self._interactive:
            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    completed=self._completed,
                )
            return

        self._emit_milestone()

    def _stop_renderer(self) -> None:
        if self._progress is not None:
            self._progress.stop()

        self._progress = None
        self._task_id = None

    def _finish_interactive_output(
        self,
    ) -> None:
        self._redirect.stop()

        try:
            if self._temp_path.exists() and self._temp_path.stat().st_size > 0:
                content = self._temp_path.read_text(encoding="utf-8")

                self._safe_print(
                    content,
                    end="",
                    flush=True,
                )

        finally:
            self._close_progress_stream()
            self._temp_path.unlink(missing_ok=True)

    def _finish_progress(
        self,
        errored: bool,
    ) -> None:
        if (
            not errored
            and self._interactive
            and self._progress is not None
            and self._task_id is not None
        ):
            completed = self._total if self._total > 0 else self._completed
            self._completed = completed

            self._progress.update(
                self._task_id,
                completed=completed,
                refresh=True,
            )
            self._progress.refresh()

        self._stop_renderer()

        if self._interactive:
            self._finish_interactive_output()
        else:
            self._finish_redirected_output(errored)


class DaskProgressBar(BaseProgress, ProgressBar):
    """Report progress for Dask task execution."""

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

        self._transient = transient
        self._refresh_per_second = refresh_per_second

    def _start(
        self,
        dsk: Any,
    ) -> None:
        self._start_progress(
            total=len(dsk),
            transient=self._transient,
            refresh_per_second=(self._refresh_per_second),
        )

    def _posttask(
        self, key: Any, result: Any, dsk: Any, state: Any, worker_id: Any
    ) -> None:
        self._advance()

    def _finish(self, dsk: Any, state: Any, errored: bool) -> None:
        self._finish_progress(errored=errored)


class SerialProgressBar(BaseProgress):
    """Report progress for serial iteration or manual updates."""

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
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
        elif iterable is not None and hasattr(
            iterable,
            "__len__",
        ):
            self._total = len(iterable)

    def _start(self) -> None:
        if self._started:
            return

        self._started = True

        self._start_progress(
            total=self._total,
            transient=self._transient,
            refresh_per_second=(self._refresh_per_second),
        )

    def update(
        self,
        n: int = 1,
    ) -> None:
        """Advance progress by ``n`` completed steps."""
        if not self._started:
            self._start()

        self._advance(n)

    def _finish(
        self,
        errored: bool = False,
    ) -> None:
        if not self._started:
            return

        try:
            self._finish_progress(errored=errored)
        finally:
            self._started = False

    def __iter__(
        self,
    ) -> Iterator[Any]:
        if self._iterable is None:
            raise ValueError("No iterable provided to wrap.")

        self._start()
        errored = False

        try:
            for item in self._iterable:
                yield item
                self._advance()

        except BaseException:
            errored = True
            raise

        finally:
            self._finish(errored=errored)

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finish(errored=exc_type is not None)
        return False
