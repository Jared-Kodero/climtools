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
    """Temporarily redirect standard output and standard error.

    Descriptor-level redirection ensures that code holding existing references
    to ``sys.stdout`` or ``sys.stderr`` follows the redirection, including
    existing ``logging.StreamHandler`` instances. If descriptor-level
    redirection is unavailable, the class falls back to replacing the Python
    stream objects.

    Parameters
    ----------
    stdout_target : pathlib.Path, str, TextIO, or None, optional
        Destination for standard output. ``None`` leaves stdout unchanged.
    stderr_target : pathlib.Path, str, TextIO, or None, optional
        Destination for standard error. ``None`` leaves stderr unchanged.
    fd_level : bool, optional
        Redirect underlying file descriptors when ``True``. Defaults to
        ``False``.
    truncate : bool, optional
        Open path targets in write mode when ``True`` and append mode when
        ``False``. Defaults to ``False``.
    """

    def __init__(
        self,
        stdout_target: Path | str | TextIO | None = None,
        stderr_target: Path | str | TextIO | None = None,
        *,
        fd_level: bool = False,
        truncate: bool = False,
    ) -> None:
        self.stdout_target = stdout_target
        self.stderr_target = stderr_target
        self.fd_level = fd_level
        self.truncate = truncate

        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None

        self._own_stdout = False
        self._own_stderr = False
        self._stdout_fd: int | None = None
        self._stderr_fd: int | None = None
        self._fd_active = False
        self._py_active = False

    @property
    def original_streams(self) -> tuple[TextIO, TextIO]:
        """Return the streams active when the instance was created.

        Returns
        -------
        tuple of TextIO
            Original ``(stdout, stderr)`` streams.
        """
        return self.orig_stdout, self.orig_stderr

    @property
    def active_streams(self) -> tuple[TextIO | None, TextIO | None]:
        """Return the prepared redirection streams.

        Returns
        -------
        tuple of TextIO or None
            Active ``(stdout, stderr)`` target streams.
        """
        return self.stdout, self.stderr

    def _open(self, target: Path | str | TextIO | None) -> tuple[TextIO | None, bool]:
        if target is None:
            return None, False

        if hasattr(target, "write"):
            return target, False

        mode = "w" if self.truncate else "a"
        return open(target, mode, encoding="utf-8"), True

    def _setup(self) -> None:
        if self.stdout_target == self.stderr_target and self.stdout_target is not None:
            self.stdout, self._own_stdout = self._open(self.stdout_target)
            self.stderr = self.stdout
            self._own_stderr = False
            return

        self.stdout, self._own_stdout = self._open(self.stdout_target)
        self.stderr, self._own_stderr = self._open(self.stderr_target)

    @staticmethod
    def redirect_fd(source: TextIO, target: TextIO) -> int:
        """Redirect one file descriptor and preserve its original destination.

        Parameters
        ----------
        source : TextIO
            Stream whose file descriptor will be redirected.
        target : TextIO
            Stream providing the new file-descriptor destination.

        Returns
        -------
        int
            Duplicated file descriptor for restoring ``source`` later.
        """
        source.flush()
        source_fd = source.fileno()
        saved_fd = os.dup(source_fd)

        try:
            os.dup2(target.fileno(), source_fd)
        except BaseException:
            os.close(saved_fd)
            raise

        return saved_fd

    def _py_redirect(self) -> tuple[TextIO | None, TextIO | None]:
        if self.stdout is not None:
            sys.stdout = self.stdout

        if self.stderr is not None:
            sys.stderr = self.stderr

        self._py_active = True
        return self.stdout, self.stderr

    def _fd_redirect(self) -> tuple[TextIO | None, TextIO | None]:
        stdout_fd: int | None = None
        stderr_fd: int | None = None

        try:
            if self.stdout is not None:
                stdout_fd = self.redirect_fd(self.orig_stdout, self.stdout)

            if self.stderr is not None:
                stderr_fd = self.redirect_fd(self.orig_stderr, self.stderr)
        except BaseException:
            self._fd_restore(stdout_fd, stderr_fd)
            raise

        self._stdout_fd = stdout_fd
        self._stderr_fd = stderr_fd
        self._fd_active = True
        return self.stdout, self.stderr

    def _fd_restore(
        self,
        stdout_fd: int | None,
        stderr_fd: int | None,
    ) -> None:
        self.orig_stdout.flush()
        self.orig_stderr.flush()

        try:
            if stdout_fd is not None:
                os.dup2(stdout_fd, self.orig_stdout.fileno())

            if stderr_fd is not None:
                os.dup2(stderr_fd, self.orig_stderr.fileno())
        finally:
            if stdout_fd is not None:
                os.close(stdout_fd)

            if stderr_fd is not None:
                os.close(stderr_fd)

    def _close(self) -> None:
        if self._own_stdout and self.stdout is not None:
            self.stdout.close()

        if (
            self._own_stderr
            and self.stderr is not None
            and self.stderr is not self.stdout
        ):
            self.stderr.close()

        self.stdout = None
        self.stderr = None
        self._own_stdout = False
        self._own_stderr = False

    def redirect(self) -> tuple[TextIO | None, TextIO | None]:
        """Activate stdout and stderr redirection.

        Descriptor-level redirection is attempted when ``fd_level=True``. If
        the source or target streams do not expose usable file descriptors,
        redirection falls back to replacing ``sys.stdout`` and ``sys.stderr``.

        Returns
        -------
        tuple of TextIO or None
            Active ``(stdout, stderr)`` target streams.
        """
        self._setup()

        if not self.fd_level:
            return self._py_redirect()

        try:
            self.orig_stdout.fileno()
            self.orig_stderr.fileno()

            if self.stdout is not None:
                self.stdout.fileno()

            if self.stderr is not None:
                self.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            self._close()
            self._setup()
            return self._py_redirect()

        try:
            return self._fd_redirect()
        except (AttributeError, OSError, ValueError):
            self._close()
            self._setup()
            return self._py_redirect()

    def restore(self) -> None:
        """Restore the original stdout and stderr destinations.

        Notes
        -----
        Saved file descriptors are restored and closed after descriptor-level
        redirection. Streams opened internally for path targets are also
        closed.
        """
        if self._fd_active:
            self._fd_restore(self._stdout_fd, self._stderr_fd)
            self._stdout_fd = None
            self._stderr_fd = None
            self._fd_active = False
        elif self._py_active:
            sys.stdout = self.orig_stdout
            sys.stderr = self.orig_stderr
            self._py_active = False

        self._close()

    def __enter__(self) -> tuple[TextIO | None, TextIO | None]:
        return self.redirect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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
