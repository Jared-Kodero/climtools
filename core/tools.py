from __future__ import annotations

import fcntl
import getpass
import inspect
import logging
import os
import socket
import sys
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable

script_dir = Path(__file__).resolve().parent
host = socket.gethostname()
user = getpass.getuser()
home = Path.home()

n_cpus = len(os.sched_getaffinity(0))
ipykernel = "ipykernel" in sys.modules
isatty = sys.stdout.isatty() or ipykernel


tmp = Path(f"/tmp/{user}/xgeo/{uuid.uuid4().hex}")
tmp.mkdir(parents=True, exist_ok=True)


current_dask_cluster = None
current_dask_client = None
widget_css_applied = False
fix_widget = True


class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class LockFile:
    """
    A context manager class for file locking with sleep and timeout mechanisms.

    This class utilizes `fcntl.flock` to acquire an exclusive, non-blocking
    lock on a specified file. If the lock is held by another process, it will
    wait and retry based on the provided delay until the timeout is reached.

    Parameters
    ----------
    filepath : Path | None, optional
        The path to the lock file. Defaults to a local ".lock" file if None.
    timeout : float | None, optional
        The maximum time in seconds to wait for the lock. If None, it will wait indefinitely.
    delay : float, optional
        The time in seconds to sleep between lock acquisition attempts. Defaults to 0.1.
    """

    def __init__(
        self,
        filepath: Path | None = None,
        timeout: float | None = None,
        delay: float = 0.1,
    ):
        self.filepath = filepath or Path(".lock")
        self.timeout = timeout
        self.delay = delay
        self.fd: int | None = None

    def acquire(self):
        """Acquire an exclusive lock on the file."""
        start_time = time.time()
        self.fd = os.open(self.filepath, os.O_RDWR | os.O_CREAT)

        try:
            while True:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except (OSError, BlockingIOError):
                    if (
                        self.timeout is not None
                        and (time.time() - start_time) >= self.timeout
                    ):
                        raise TimeoutError(
                            f"Could not acquire lock on {self.filepath} within {self.timeout}s"
                        )
                    time.sleep(self.delay)
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise

    def release(self) -> None:
        """Release the acquired file lock and close the underlying file descripto"""
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class LockedLogger:
    """
    A wrapper around `logging.Logger` to synchronize logging output across processes.

    Parameters
    ----------
    logger : logging.Logger
        The standard library logger instance to wrap.
    lock_file : LockFile
        An instance of the LockFile context manager to use for synchronization.
    """

    def __init__(self, logger: logging.Logger, lock_file: LockFile) -> None:
        self._logger = logger
        self._lock_file = lock_file

    @wraps(logging.Logger.info)
    def info(self, *args, **kwargs) -> None:
        with self._lock_file:
            self._logger.info(*args, **kwargs)

    @wraps(logging.Logger.warning)
    def warning(self, *args, **kwargs) -> None:
        with self._lock_file:
            self._logger.warning(*args, **kwargs)

    @wraps(logging.Logger.error)
    def error(self, *args, **kwargs) -> None:
        with self._lock_file:
            self._logger.error(*args, **kwargs)


def locked_print(
    *values: Any,
    lockfile: LockFile | Any | None = None,
    sep: str | None = " ",
    end: str | None = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """
    Wraps the standard print function with a lock object to prevent interleaved output.

    Matches standard print signatures for IDE autocomplete.

    Parameters
    ----------
    *values : Any
        The values to be printed.
    lockfile : LockFile | Any, required
        An instance of `LockFile`, or any lock object that supports the standard
        `with` context manager protocol (`__enter__` and `__exit__`).
    sep : str | None, optional
        String inserted between values. Defaults to a space.
    end : str | None, optional
        String appended after the last value. Defaults to a newline.
    file : TextIO | None, optional
        A file-like object (stream) to print to. Defaults to `sys.stdout`.
    flush : bool, optional
        Whether to forcefully flush the stream. Defaults to False.

    Raises
    ------
    ValueError
        If no valid lock object is provided via the `lockfile` argument.
    """
    if lockfile is None:
        raise ValueError(
            "We need a lockfile obj:\n(e.g., climtools.LockFile, threading.Lock...)"
        )

    with lockfile:
        print(*values, sep=sep, end=end, file=file, flush=flush)


class RedirectStreams:
    """
    Temporarily redirect standard output and standard error to files
    or in-memory buffers, storing original streams for external access.
    """

    def __init__(
        self,
        stdout_target: Path | str | TextIO | None = None,
        stderr_target: Path | str | TextIO | None = None,
    ):
        self.stdout_target = stdout_target
        self.stderr_target = stderr_target

        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr

        self.stdout = None
        self.stderr = None
        self._managed_stdout = False
        self._managed_stderr = False

    @property
    def original_streams(self) -> tuple[TextIO, TextIO]:
        """Returns the original (stdout, stderr) pair."""
        return self.orig_stdout, self.orig_stderr

    @property
    def active_streams(self) -> tuple[TextIO | None, TextIO | None]:
        """Returns the currently active redirected (stdout, stderr) pair."""
        return self.stdout, self.stderr

    def _prepare_target(self, target: Path | str | TextIO | None, is_stdout: bool):
        if target is None:
            return None, False

        # If it's already a stream/buffer (has a write method)
        if hasattr(target, "write"):
            return target, False

        # Otherwise, treat it as a path/string and open it
        file_obj = open(target, "a", encoding="utf-8")
        return file_obj, True

    def redirect(self):
        # Handle matching targets
        if self.stdout_target == self.stderr_target and self.stdout_target is not None:
            self.stdout, self._managed_stdout = self._prepare_target(
                self.stdout_target, True
            )
            self.stderr = self.stdout
            self._managed_stderr = False
        else:
            self.stdout, self._managed_stdout = self._prepare_target(
                self.stdout_target, True
            )
            self.stderr, self._managed_stderr = self._prepare_target(
                self.stderr_target, False
            )

        if self.stdout:
            sys.stdout = self.stdout
        if self.stderr:
            sys.stderr = self.stderr

        return self.stdout, self.stderr

    def restore(self):
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr

        # Only close streams that this class actually opened
        if self._managed_stdout and self.stdout:
            self.stdout.close()

        if self._managed_stderr and self.stderr and self.stderr is not self.stdout:
            self.stderr.close()

    def __enter__(self):
        return self.redirect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


def get_fsig(func: Callable) -> dict:
    """
    Map the named parameters of ``func`` to their default values.

    Variadic parameters (``*args`` and ``**kwargs``) are excluded: they are not
    keywords a caller can bind by name, so including them would let the literal
    names ``args`` and ``kwargs`` pass through keyword filters built from this
    mapping.

    Parameters
    ----------
    func : Callable
        Function, method or class whose signature is inspected.

    Returns
    -------
    dict
        Parameter name mapped to its default, or to None when the parameter
        has no default.
    """
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    params = {}

    for name, param in inspect.signature(func).parameters.items():
        if param.kind in variadic:
            continue
        params[name] = (
            None if param.default is inspect.Parameter.empty else param.default
        )

    return params


def apply_widget_css() -> None:
    """Inject theme-aware styling for Jupyter and Matplotlib widgets."""

    global widget_css_applied

    if "ipykernel" not in sys.modules or widget_css_applied:
        return

    from IPython.display import HTML, display

    css = """
    <style>
    /* 1. Force transparent backgrounds on all widget containers */
    .cell-output-ipywidget-background,
    .jupyter-widgets,
    .jupyter-matplotlib,
    .jupyter-matplotlib-figure,
    .jupyter-matplotlib-canvas-container,
    .jupyter-matplotlib-canvas-div {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* 2. Map standard Jupyter variables to VS Code editor settings */
    :root {
        --jp-widgets-color:
            var(--vscode-editor-foreground, CanvasText);
        --jp-widgets-font-size:
            var(--vscode-editor-font-size);
    }

    /* 3. Use the active environment foreground color */
    .jupyter-widgets,
    .jupyter-matplotlib {
        color: var(--vscode-editor-foreground, CanvasText) !important;
        --jp-widgets-color:
            var(--vscode-editor-foreground, CanvasText) !important;
    }

    /* 4. VS Code theme-class fallbacks */
    .vscode-dark .jupyter-widgets,
    .vscode-light .jupyter-widgets {
        color: var(--vscode-editor-foreground, CanvasText) !important;
        --jp-widgets-color:
            var(--vscode-editor-foreground, CanvasText) !important;
    }
    </style>
    """
    display(HTML(css))
    widget_css_applied = True
