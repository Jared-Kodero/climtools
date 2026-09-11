from __future__ import annotations

import atexit
import fcntl
import getpass
import inspect
import logging
import os
import random
import shutil
import signal
import socket
import sys
import time
import uuid
from functools import wraps
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable

N_CPUS: int = len(os.sched_getaffinity(0))
HOST: str = socket.gethostname()
USER: str = getpass.getuser()
HOME: str = Path.home()


TMP = Path(f"/tmp/{USER}/xgeo/{uuid.uuid4().hex}")
TMP.mkdir(parents=True, exist_ok=True)


script_dir = Path(__file__).resolve().parent
ipykernel = "ipykernel" in sys.modules
isatty = sys.stdout.isatty() or ipykernel


current_dask_cluster = None
current_dask_client = None
widget_css_applied = False
fix_widget = True


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


def exclude_key(name: str | list[str], data: dict) -> dict:
    """Exclude keys in-place and return the dictionary without copying."""
    keys = {name} if isinstance(name, str) else set(name)
    for k in keys:
        data.pop(k, None)
    return data


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
        time.sleep(random.uniform(0, self.delay))
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
                    time.sleep(self.delay + random.uniform(0, self.delay * 0.1))
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
    """Redirect standard output and standard error.

    Descriptor-level redirection ensures that code holding existing
    references to ``sys.stdout`` or ``sys.stderr`` follows the redirection,
    including existing ``logging.StreamHandler`` instances. If
    descriptor-level redirection is unavailable, the class falls back to
    replacing the Python stream objects.

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

        self.orig_stdout: TextIO | None = None
        self.orig_stderr: TextIO | None = None
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None

        self._own_stdout = False
        self._own_stderr = False
        self._stdout_fd: int | None = None
        self._stderr_fd: int | None = None
        self._mode: Literal["fd", "python"] | None = None

    @property
    def original_streams(self) -> tuple[TextIO | None, TextIO | None]:
        """Return the streams active immediately before redirection."""
        return self.orig_stdout, self.orig_stderr

    @property
    def active_streams(self) -> tuple[TextIO | None, TextIO | None]:
        """Return the prepared redirection targets."""
        return self.stdout, self.stderr

    @staticmethod
    def fd_path(fd: int) -> str:
        """Return the /proc/self/fd path for a file descriptor."""
        return f"/proc/self/fd/{fd}"

    @staticmethod
    def duplicate(stream: TextIO) -> TextIO:
        """Return a writable stream backed by a duplicate descriptor."""
        stream.flush()
        fd = os.dup(stream.fileno())
        encoding = getattr(stream, "encoding", None) or "utf-8"
        errors = getattr(stream, "errors", None) or "strict"

        try:
            return os.fdopen(fd, "w", encoding=encoding, errors=errors, buffering=1)
        except BaseException:
            os.close(fd)
            raise

    def start(self) -> tuple[TextIO | None, TextIO | None]:
        """Activate stdout and stderr redirection."""
        if self._mode is not None:
            raise RuntimeError("Stream redirection is already active.")

        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self._prepare_targets()

        if self.fd_level:
            try:
                return self._start_fd()
            except (AttributeError, OSError, ValueError):
                self._close_targets()
                self._prepare_targets()

        return self._start_python()

    def stop(self) -> None:
        """Restore stdout and stderr and close internally opened targets."""
        try:
            if self._mode == "fd":
                self._restore_fds(self._stdout_fd, self._stderr_fd)
            elif self._mode == "python":
                if self.orig_stdout is not None:
                    sys.stdout = self.orig_stdout
                if self.orig_stderr is not None:
                    sys.stderr = self.orig_stderr
        finally:
            self._stdout_fd = None
            self._stderr_fd = None
            self._mode = None
            self._close_targets()

    def __enter__(self) -> tuple[TextIO | None, TextIO | None]:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # -- internal helpers, used only by start()/stop() --------------------

    def _point(self, source: TextIO, target: TextIO) -> int:
        source.flush()
        source_fd = source.fileno()
        saved_fd = os.dup(source_fd)
        try:
            os.dup2(target.fileno(), source_fd)
        except BaseException:
            os.close(saved_fd)
            raise
        return saved_fd

    def _reset(self, saved_fd: int | None, stream: TextIO | None) -> None:
        if saved_fd is None or stream is None:
            return
        try:
            stream.flush()
            os.dup2(saved_fd, stream.fileno())
        finally:
            os.close(saved_fd)

    def _same_target(
        self, first: Path | str | TextIO | None, second: Path | str | TextIO | None
    ) -> bool:
        if first is second:
            return first is not None
        if isinstance(first, (str, Path)) and isinstance(second, (str, Path)):
            return os.fspath(first) == os.fspath(second)
        return False

    def _open_target(
        self, target: Path | str | TextIO | None
    ) -> tuple[TextIO | None, bool]:
        if target is None:
            return None, False
        if hasattr(target, "write"):
            return target, False
        mode = "w" if self.truncate else "a"
        return open(target, mode, encoding="utf-8"), True

    def _prepare_targets(self) -> None:
        try:
            if self._same_target(self.stdout_target, self.stderr_target):
                self.stdout, self._own_stdout = self._open_target(self.stdout_target)
                self.stderr = self.stdout
                return
            self.stdout, self._own_stdout = self._open_target(self.stdout_target)
            self.stderr, self._own_stderr = self._open_target(self.stderr_target)
        except BaseException:
            self._close_targets()
            raise

    def _close_targets(self) -> None:
        if self._own_stdout and self.stdout is not None:
            self.stdout.close()
        if self._own_stderr and self.stderr is not None:
            self.stderr.close()
        self.stdout = None
        self.stderr = None
        self._own_stdout = False
        self._own_stderr = False

    def _start_python(self) -> tuple[TextIO | None, TextIO | None]:
        if self.stdout is not None:
            sys.stdout = self.stdout
        if self.stderr is not None:
            sys.stderr = self.stderr
        self._mode = "python"
        return self.stdout, self.stderr

    def _start_fd(self) -> tuple[TextIO | None, TextIO | None]:
        if self.orig_stdout is None or self.orig_stderr is None:
            raise RuntimeError("Original streams have not been captured.")

        stdout_fd: int | None = None
        stderr_fd: int | None = None

        try:
            if self.stdout is not None:
                stdout_fd = self._point(self.orig_stdout, self.stdout)
            if self.stderr is not None:
                stderr_fd = self._point(self.orig_stderr, self.stderr)
        except BaseException:
            self._restore_fds(stdout_fd, stderr_fd)
            raise

        self._stdout_fd = stdout_fd
        self._stderr_fd = stderr_fd
        self._mode = "fd"
        return self.stdout, self.stderr

    def _restore_fds(self, stdout_fd: int | None, stderr_fd: int | None) -> None:
        pairs = ((stdout_fd, self.orig_stdout), (stderr_fd, self.orig_stderr))
        error: BaseException | None = None

        for saved_fd, stream in pairs:
            try:
                self._reset(saved_fd, stream)
            except BaseException as exc:
                if error is None:
                    error = exc

        if error is not None:
            raise error


class SharedMemoryObject:
    """Shared-memory transport for NumPy and xarray objects.

    Supported
    ---------
    - numpy.ndarray
    - xarray.DataArray
    - xarray.Dataset

    When this object is pickled, for example by multiprocessing.Queue,
    Pool, Process, or ProcessPoolExecutor, the receiving process gets the
    original ndarray/DataArray/Dataset rather than SharedMemoryObject.

    Pointer-free NumPy buffers are placed in shared memory. Object-dtype
    arrays fall back to ordinary pickle transport because their memory
    contains process-local Python pointers.

    Parameters
    ----------
    obj
        Object to place into shared memory.
    readonly
        If True, reconstructed NumPy buffers are marked non-writeable.

    Notes
    -----
    The creating process owns the shared-memory segments. It must keep this
    object alive until all workers have finished using the reconstructed
    objects.

    Use as a context manager whenever possible.
    """

    _attachments: ClassVar[dict[str, shared_memory.SharedMemory]] = {}

    _atexit_registered: ClassVar[bool] = False

    def __init__(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
        *,
        readonly: bool = True,
    ) -> None:
        self._owners: list[shared_memory.SharedMemory] = []
        self._closed = False
        self._readonly = readonly

        self._ensure_atexit()

        try:
            self._spec = self._encode_object(obj)
        except Exception:
            self.close()
            raise

    @classmethod
    def _ensure_atexit(cls) -> None:
        if cls._atexit_registered:
            return

        atexit.register(cls.close_attachments)
        cls._atexit_registered = True

    @staticmethod
    def _open_attachment(
        name: str,
    ) -> shared_memory.SharedMemory:
        # Python >= 3.13 supports track=False. This prevents receiver
        # processes from independently trying to unlink memory owned by
        # the creating process.
        try:
            return shared_memory.SharedMemory(
                name=name,
                track=False,
            )
        except TypeError:
            # Python <= 3.12
            return shared_memory.SharedMemory(name=name)

    @classmethod
    def _attach(
        cls,
        name: str,
    ) -> shared_memory.SharedMemory:
        cls._ensure_atexit()

        shm = cls._attachments.get(name)

        if shm is None:
            shm = cls._open_attachment(name)
            cls._attachments[name] = shm

        return shm

    def _encode_array(
        self,
        array: np.ndarray,
    ) -> dict[str, Any]:
        array = np.asarray(array)

        order: Literal["C", "F"]

        if array.flags.f_contiguous and not array.flags.c_contiguous:
            order = "F"
        else:
            order = "C"

        # Raw shared memory cannot safely represent object-containing
        # dtypes because the buffer contains CPython pointers.
        if array.dtype.hasobject:
            return {
                "shape": array.shape,
                "dtype": array.dtype,
                "order": order,
                "shm_name": None,
                "inline": np.array(
                    array,
                    copy=True,
                    order=order,
                ),
            }

        contiguous = np.array(
            array,
            copy=True,
            order=order,
        )

        shm = shared_memory.SharedMemory(
            create=True,
            size=max(contiguous.nbytes, 1),
        )

        self._owners.append(shm)

        target = np.ndarray(
            contiguous.shape,
            dtype=contiguous.dtype,
            buffer=shm.buf,
            order=order,
        )

        target[...] = contiguous

        return {
            "shape": contiguous.shape,
            "dtype": contiguous.dtype,
            "order": order,
            "shm_name": shm.name,
            "inline": None,
        }

    def _encode_variable(
        self,
        name: Any,
        role: Literal["data", "coord"],
        variable: xr.Variable,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "role": role,
            "dims": tuple(variable.dims),
            "attrs": dict(variable.attrs),
            "encoding": dict(variable.encoding),
            "array": self._encode_array(np.asarray(variable.data)),
        }

    def _encode_object(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
    ) -> dict[str, Any]:
        if isinstance(obj, xr.Dataset):
            variables = [
                self._encode_variable(name, "data", variable)
                for name, variable in obj.data_vars.items()
            ]

            variables.extend(
                self._encode_variable(
                    name,
                    "coord",
                    variable,
                )
                for name, variable in obj.coords.items()
            )

            return {
                "kind": "dataset",
                "attrs": dict(obj.attrs),
                "encoding": dict(obj.encoding),
                "variables": variables,
                "readonly": self._readonly,
            }

        if isinstance(obj, xr.DataArray):
            variables = [self._encode_variable(None, "data", obj.variable)]

            variables.extend(
                self._encode_variable(
                    name,
                    "coord",
                    variable,
                )
                for name, variable in obj.coords.items()
            )

            return {
                "kind": "dataarray",
                "name": obj.name,
                "variables": variables,
                "readonly": self._readonly,
            }

        if isinstance(obj, np.ndarray):
            return {
                "kind": "ndarray",
                "array": self._encode_array(obj),
                "readonly": self._readonly,
            }

        raise TypeError(
            f"Expected np.ndarray, xr.DataArray, or xr.Dataset; got {type(obj)!r}."
        )

    @classmethod
    def _decode_array(cls, spec: dict[str, Any], *, readonly: bool) -> np.ndarray:
        shm_name = spec["shm_name"]

        if shm_name is None:
            array = spec["inline"]
        else:
            shm = cls._attach(shm_name)

            array = np.ndarray(
                spec["shape"], dtype=spec["dtype"], buffer=shm.buf, order=spec["order"]
            )

        if readonly:
            array.flags.writeable = False

        return array

    @classmethod
    def _decode_variable(cls, spec: dict[str, Any], *, readonly: bool) -> xr.Variable:
        variable = xr.Variable(
            spec["dims"],
            cls._decode_array(
                spec["array"],
                readonly=readonly,
            ),
            attrs=dict(spec["attrs"]),
        )

        variable.encoding.update(spec["encoding"])

        return variable

    @classmethod
    def _rebuild(
        cls,
        spec: dict[str, Any],
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        readonly = spec["readonly"]
        kind = spec["kind"]

        if kind == "ndarray":
            return cls._decode_array(
                spec["array"],
                readonly=readonly,
            )

        decoded = [
            (
                variable_spec,
                cls._decode_variable(variable_spec, readonly=readonly),
            )
            for variable_spec in spec["variables"]
        ]

        if kind == "dataset":
            data_vars = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "data"
            }

            coords = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "coord"
            }

            dataset = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs=dict(spec["attrs"]),
            )

            dataset.encoding.update(spec["encoding"])

            return dataset

        if kind == "dataarray":
            data_variable = next(
                variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "data"
            )

            coords = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "coord"
            }

            return xr.DataArray(data_variable, coords=coords, name=spec["name"])

        raise RuntimeError(f"Unknown object kind: {kind!r}")

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Control multiprocessing/pickle reconstruction."""
        if self._closed:
            raise RuntimeError("Cannot serialize a closed SharedMemoryObject.")

        return self._rebuild, (self._spec,)

    def get(self) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Return a local view of the shared object."""
        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self._rebuild(self._spec)

    @property
    def nbytes(self) -> int:
        """Total allocated shared-memory bytes."""
        return sum(shm.size for shm in self._owners)

    @property
    def closed(self) -> bool:
        """Whether the owner has been closed."""
        return self._closed

    def close(self) -> None:
        """Unlink and close owned shared-memory segments."""
        if self._closed:
            return

        for shm in self._owners:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

            shm.close()

        self._owners.clear()
        self._closed = True

    @classmethod
    def close_attachments(cls) -> None:
        """Close receiver-side shared-memory handles."""
        for name, shm in list(cls._attachments.items()):
            try:
                shm.close()
            except (BufferError, OSError):
                # A live NumPy array may still export the
                # underlying mmap. The OS will clean it up
                # when this process terminates.
                continue

            del cls._attachments[name]

    def __enter__(self):
        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _cleanup(*_):
    shutil.rmtree(TMP, ignore_errors=True)


_previous_handlers: dict[int, Any] = {}


def _cleanup_then_chain(signum, frame):
    """Remove the scratch directory, then let the signal do its job.

    Installing ``_cleanup`` directly as the handler silently cancelled the
    signal, because a Python handler that returns normally suppresses it:
    SIGINT stopped raising KeyboardInterrupt and SIGTERM stopped terminating,
    for every program that imported climtools. A supervisor that asks politely
    and is ignored escalates to SIGKILL, and an interactive user whose first
    Ctrl-C appears to do nothing presses it again -- which under ``srun --pty``
    tears down the job step. Cleanup is a side errand, not grounds for
    swallowing the signal.
    """
    _cleanup()
    previous = _previous_handlers.get(signum, signal.SIG_DFL)
    if callable(previous):
        previous(signum, frame)
        return
    signal.signal(signum, previous)
    os.kill(os.getpid(), signum)


atexit.register(_cleanup)
for _sig in (signal.SIGTERM, signal.SIGINT):
    _previous_handlers[_sig] = signal.getsignal(_sig)
    signal.signal(_sig, _cleanup_then_chain)
