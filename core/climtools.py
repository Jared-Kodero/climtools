from __future__ import annotations

import atexit
import fcntl
import getpass
import inspect
import json
import logging
import os
import random
import shutil
import signal
import socket
import sys
import time
import uuid
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from logging import Handler, Logger

nproc: int = len(os.sched_getaffinity(0))
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
        self.filepath = Path(filepath or ".lock")
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
    def __init__(
        self,
        lock_file: LockFile,
        logger: Logger | None = None,
        *,
        name: str | None = None,
        format: str = "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt: str | None = "%Y-%m-%d %H:%M",
        style: str = "%",
        level: int | str = logging.INFO,
        handlers: Iterable[Handler] | None = None,
        force: bool = False,
    ) -> None:
        if logger is None:
            if name is None:
                name = sys._getframe(1).f_globals["__name__"]

            self.basicConfig(
                format=format,
                datefmt=datefmt,
                style=style,
                level=level,
                handlers=handlers,
                force=force,
            )

            logger = logging.getLogger(name)

        self._logger = logger
        self._lock_file = lock_file

    @property
    def name(self) -> str:
        return self._logger.name

    @wraps(logging.basicConfig)
    def basicConfig(self, *args, **kwargs) -> None:
        logging.basicConfig(*args, **kwargs)

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

    @wraps(logging.Logger.exception)
    def exception(self, *args, **kwargs) -> None:
        with self._lock_file:
            self._logger.exception(*args, **kwargs)


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
            "We need a lockfile obj:\n(e.g., xgeo.LockFile, threading.Lock...)"
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
"""Provide array I/O and shared-memory transport."""


def to_xnpy(
    obj: np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    mode: Literal["w", "w-"] = "w-",
    parallel: bool = False,
    max_workers: int | None = None,
) -> None:
    """Write a NumPy, pandas, or xarray object to a memory-mappable XNpy store.

    Large arrays are streamed in bounded slabs. Dask-backed variables are written
    with :func:`dask.array.store`, allowing all chunks to execute in one task
    graph without serial chunk scheduling.

    Metadata (``attrs``, labels, names) is encoded with type tags so that tuples,
    dicts with non-string keys, and nested pandas or xarray objects (stored as
    sub-stores under ``attrs/``) round-trip. Text and categorical columns keep a
    missing-value mask and their categories. MultiIndex rows and columns are
    supported.

    Parameters
    ----------
    obj : numpy.ndarray or pandas.Series or pandas.DataFrame or xarray.DataArray or xarray.Dataset
        Object to store. Labels and attrs must be built from str, int, float,
        bool, None, tuples, lists, dicts, NumPy scalars and arrays, or nested
        pandas and xarray objects.
    path : str or pathlib.Path
        Destination store directory.
    mode : {"w", "w-"}, default="w-"
        Write mode. ``"w"`` replaces an existing store. ``"w-"`` requires that
        the destination not exist.
    parallel : bool, default=False
        Write arrays concurrently using a thread pool. Dask chunk parallelism is
        independent of this option.
    max_workers : int, optional
        Maximum number of write workers when ``parallel=True``. If omitted, at
        most four workers are used.

    Returns
    -------
    None
        The store is written to ``path``.
    """
    if mode not in {"w", "w-"}:
        raise ValueError(f"Unsupported XNpy mode: {mode!r}.")

    root = Path(path)
    if root.exists():
        if mode == "w-":
            raise FileExistsError(root)
        shutil.rmtree(root)

    jobs: list[tuple[Any, str]] = []
    nested: list[Any] = []

    def add(source: Any, folder: str, name: Any) -> str:
        file = f"{folder}/{name}.npy"
        jobs.append((source, file))
        return file

    def add_variable(source: xr.DataArray, folder: str, name: Any) -> dict[str, Any]:
        return {
            "dims": list(source.dims),
            "attrs": dict(source.attrs),
            "file": add(source, folder, name),
        }

    def add_column(values: Any, folder: str, name: Any) -> dict[str, Any]:
        series = pd.Series(values)
        if isinstance(series.dtype, pd.CategoricalDtype):
            return {
                "encoding": "categorical",
                "file": add(series.cat.codes.to_numpy(), folder, name),
                "categories": series.cat.categories.tolist(),
                "ordered": bool(series.cat.ordered),
            }
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            return {
                "encoding": "string",
                "file": add(series.fillna("").to_numpy().astype(str), folder, name),
                "mask": add(series.isna().to_numpy(), folder, f"{name}.mask"),
                "extension": isinstance(series.dtype, pd.StringDtype),
            }
        return {"encoding": "array", "file": add(series.to_numpy(), folder, name)}

    def add_index(index: pd.Index) -> list[dict[str, Any]]:
        return [
            add_column(index.get_level_values(i), "coords", f"index.{i}")
            for i in range(index.nlevels)
        ]

    if isinstance(obj, xr.Dataset):
        metadata: dict[str, Any] = {
            "kind": "dataset",
            "attrs": dict(obj.attrs),
            "variables": {
                name: add_variable(var, "variables", name)
                for name, var in obj.data_vars.items()
            },
            "coords": {
                name: add_variable(coord, "coords", name)
                for name, coord in obj.coords.items()
            },
        }
    elif isinstance(obj, xr.DataArray):
        metadata = {
            "kind": "dataarray",
            "name": obj.name,
            "variable": add_variable(
                obj,
                "variables",
                "data" if obj.name is None else obj.name,
            ),
            "coords": {
                name: add_variable(coord, "coords", name)
                for name, coord in obj.coords.items()
            },
        }
    elif isinstance(obj, pd.DataFrame):
        metadata = {
            "kind": "dataframe",
            "attrs": dict(obj.attrs),
            "index_names": list(obj.index.names),
            "index": add_index(obj.index),
            "columns_names": list(obj.columns.names),
            "columns": [
                {"label": label, "column": add_column(column, "columns", position)}
                for position, (label, column) in enumerate(obj.items())
            ],
        }
    elif isinstance(obj, pd.Series):
        metadata = {
            "kind": "series",
            "name": obj.name,
            "attrs": dict(obj.attrs),
            "index_names": list(obj.index.names),
            "index": add_index(obj.index),
            "column": add_column(obj, "variables", "series"),
        }
    else:
        metadata = {"kind": "ndarray", "file": add(obj, "variables", "array")}

    def encode(value: Any) -> Any:
        """Convert metadata to JSON with tags for tuples, dicts, arrays, and stores."""
        if value is None or isinstance(value, str | bool | int | float):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist(), "dtype": value.dtype.str}
        if isinstance(value, pd.DataFrame | pd.Series | xr.DataArray | xr.Dataset):
            nested.append(value)
            return {"__xnpy__": f"attrs/{len(nested) - 1}"}
        if isinstance(value, tuple):
            return {"__tuple__": [encode(item) for item in value]}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {"__dict__": [[encode(k), encode(v)] for k, v in value.items()]}
        raise TypeError(f"Unsupported metadata type: {type(value).__name__}.")

    def write_array(source: Any, path: Path) -> None:
        slab_bytes = 256 * 1024**2

        # Object arrays (e.g. strings) cannot be memory-mapped or saved without pickle.
        if source.dtype == object:
            source = np.asarray(source).astype(str)

        if getattr(source, "chunks", None) is not None:
            import dask.array as da

            target = np.lib.format.open_memmap(
                path, "w+", dtype=source.dtype, shape=source.shape
            )
            data = source.data if isinstance(source, xr.DataArray) else source
            da.store(data, target, lock=False, scheduler="threads")
            target.flush()
        elif source.nbytes > slab_bytes and source.ndim:
            # Stream large or lazily indexed arrays in bounded first-axis slabs.
            target = np.lib.format.open_memmap(
                path, "w+", dtype=source.dtype, shape=source.shape
            )
            step = max(slab_bytes * source.shape[0] // source.nbytes, 1)
            for start in range(0, source.shape[0], step):
                target[start : start + step] = np.asarray(source[start : start + step])
            target.flush()
        else:
            np.save(path, np.asarray(source), allow_pickle=False)

    try:
        metadata = encode(metadata)
        root.mkdir(parents=True)
        for folder in {(root / file).parent for _, file in jobs}:
            folder.mkdir(exist_ok=True)

        workers = (max_workers or 4) if parallel else 1
        with ThreadPoolExecutor(workers) as executor:
            list(executor.map(lambda job: write_array(job[0], root / job[1]), jobs))

        for position, value in enumerate(nested):
            to_xnpy(
                value,
                root / "attrs" / str(position),
                parallel=parallel,
                max_workers=max_workers,
            )

        with (root / "metadata.json").open("w") as file:
            json.dump(metadata, file)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def open_xnpy(
    path: str | Path,
    variable: Hashable | None = None,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | xr.Dataset:
    """Open a NumPy, pandas, or xarray object from a memory-mappable XNpy store.

    The return type is determined at runtime from the store manifest. Use
    :func:`open_xnpy_dataframe` or :func:`open_xnpy_dataset` for static typing.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    variable : hashable, optional
        Dataset variable or DataFrame column label to return. Ignored for
        ndarray, Series, and DataArray stores. For a DataFrame, only the
        selected column payloads are opened and a one-column DataFrame is
        returned. If omitted, return the complete stored object.
    mmap_mode : {"r", "r+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`. Use ``"c"`` for
        copy-on-write arrays that accept in-place edits without modifying the
        files, or ``None`` to load payloads into ordinary in-memory arrays.
        Text columns are always loaded into memory.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame or xarray.DataArray or xarray.Dataset
        Reconstructed object. A Dataset store with ``variable`` returns a
        DataArray. Array payloads are memory-mapped unless ``mmap_mode=None``.

    Notes
    -----
    Memory mapping defers physical reads until array pages are accessed, so stores
    larger than available RAM can be opened without materializing their payloads.
    """
    root = Path(path)

    def decode(value: Any) -> Any:
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            if "__tuple__" in value:
                return tuple(decode(item) for item in value["__tuple__"])
            if "__dict__" in value:
                return {decode(k): decode(v) for k, v in value["__dict__"]}
            if "__ndarray__" in value:
                return np.array(value["__ndarray__"], dtype=value["dtype"])
            if "__xnpy__" in value:
                return open_xnpy(root / value["__xnpy__"], mmap_mode=mmap_mode)
            return {k: decode(v) for k, v in value.items()}
        return value

    metadata = decode(json.loads((root / "metadata.json").read_text()))
    kind = metadata["kind"]

    def load(file: str) -> np.ndarray:
        return np.load(root / file, mmap_mode=mmap_mode, allow_pickle=False)

    def load_variable(info: dict[str, Any]) -> xr.Variable:
        return xr.Variable(info["dims"], load(info["file"]), attrs=info["attrs"])

    def load_column(spec: dict[str, Any]) -> Any:
        values = load(spec["file"])
        if spec["encoding"] == "categorical":
            return pd.Categorical.from_codes(
                values, spec["categories"], ordered=spec["ordered"]
            )
        if spec["encoding"] == "string":
            out = values.astype(object)
            out[np.asarray(load(spec["mask"]))] = pd.NA if spec["extension"] else np.nan
            return pd.array(out, dtype="string") if spec["extension"] else out
        return values

    def load_index() -> pd.Index:
        levels = [load_column(spec) for spec in metadata["index"]]
        names = metadata["index_names"]
        if len(levels) == 1:
            return pd.Index(levels[0], name=names[0])
        return pd.MultiIndex.from_arrays(levels, names=names)

    if kind == "ndarray":
        return load(metadata["file"])
    elif kind == "series":
        series = pd.Series(
            load_column(metadata["column"]),
            index=load_index(),
            name=metadata["name"],
            copy=False,
        )
        series.attrs = metadata["attrs"]
        return series
    elif kind == "dataframe":
        columns = [(c["label"], c["column"]) for c in metadata["columns"]]
        if variable is not None:
            columns = [column for column in columns if column[0] == variable]
            if not columns:
                raise KeyError(variable)

        names = metadata["columns_names"]
        labels = [label for label, _ in columns]
        frame = pd.DataFrame(
            {position: load_column(spec) for position, (_, spec) in enumerate(columns)},
            index=load_index(),
            copy=False,
        )
        frame.columns = (
            pd.MultiIndex.from_tuples(labels, names=names)
            if len(names) > 1
            else pd.Index(labels, name=names[0], tupleize_cols=False)
        )
        frame.attrs = metadata["attrs"]
        return frame
    elif kind in {"dataarray", "dataset"}:
        coords = {
            name: load_variable(info) for name, info in metadata["coords"].items()
        }
        if kind == "dataarray":
            return xr.DataArray(
                load_variable(metadata["variable"]),
                coords=coords,
                name=metadata["name"],
            )

        dataset = xr.Dataset(
            {name: load_variable(info) for name, info in metadata["variables"].items()},
            coords=coords,
            attrs=metadata["attrs"],
        )
        return dataset if variable is None else dataset[variable]
    else:
        raise ValueError(f"Unknown XNpy kind: {kind!r}.")


def open_xnpy_ndarray(
    path: str | Path,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> np.ndarray:
    """Open an ndarray store.

    Raises
    ------
    TypeError
        If the store does not hold an ndarray.
    """
    obj = open_xnpy(path, mmap_mode=mmap_mode)
    if not isinstance(obj, np.ndarray):
        raise TypeError(f"Store {path} does not hold an ndarray.")
    return obj


def open_xnpy_dataframe(
    path: str | Path,
    variable: Hashable | None = None,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> pd.DataFrame:
    """Open a DataFrame or Series store, optionally restricted to one column.

    For a DataFrame store, only the payloads of the column labelled ``variable``
    are opened and the result is a one-column DataFrame carrying the stored
    index. ``variable`` is ignored for a Series store.

    Raises
    ------
    TypeError
        If the store holds neither a DataFrame nor a Series.
    """
    obj = open_xnpy(path, variable, mmap_mode=mmap_mode)
    if not isinstance(obj, pd.DataFrame | pd.Series):
        raise TypeError(f"Store {path} does not hold a DataFrame or Series.")
    return obj


def open_xnpy_dataset(
    path: str | Path,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> xr.Dataset:
    """Open a Dataset store.

    Raises
    ------
    TypeError
        If the store does not hold a Dataset.
    """
    obj = open_xnpy(path, mmap_mode=mmap_mode)
    if not isinstance(obj, xr.Dataset):
        raise TypeError(f"Store {path} does not hold a Dataset.")
    return obj


class SharedMemoryObject:
    """Zero-copy shared-memory transport wrapper for NumPy and xarray objects.

    Enables efficient inter-process communication by placing compatible arrays
    into shared memory blocks, bypassing the serialization overhead of large data buffers.

    Parameters
    ----------
    obj : numpy.ndarray, xarray.DataArray, or xarray.Dataset
        The target object to place into shared memory.
    readonly : bool, default=True
        If True, reconstructed NumPy buffers in receiving processes will be
        marked as read-only to prevent unintended mutation.

    Supported Types & Limitations
    -----------------------------
    * **NumPy:** Pointer-free data types only. Object-dtype arrays automatically
      fall back to standard pickling because their memory contains local pointers.
    * **Xarray:** ``DataArray`` and ``Dataset`` structures (backing data is shared).

    Behavior & Pickling
    -------------------
    When pickled (e.g., via ``multiprocessing``, ``Queue``, or ``ProcessPoolExecutor``),
    receiving processes transparently unpack the object back into a native container
    referencing the underlying shared memory block.

    Lifecycle Management & Best Practices
    -------------------------------------
    The creating process retains ownership of the shared-memory segments.
    **You must keep this instance alive** until all worker processes have finished
    using the reconstructed objects.

    * **Recommended:** Always use this class as a context manager to ensure
      proper resource cleanup:

        >>> with SharedMemoryObject(my_array) as shared_obj:
        ...     pool.apply_async(worker_func, args=(shared_obj,))
    """

    _attachments: ClassVar[dict[str, shared_memory.SharedMemory]] = {}

    _atexit_registered: ClassVar[bool] = False

    def __init__(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
        *,
        readonly: bool = True,
    ) -> None:
        """Allocate shared memory blocks and encode the object specification."""

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
        """Register process-exit cleanup handler for receiver-side attachments."""

        if cls._atexit_registered:
            return

        atexit.register(cls.close_attachments)

        cls._atexit_registered = True

    @staticmethod
    def _open_attachment(
        name: str,
    ) -> shared_memory.SharedMemory:
        """Open an existing shared memory block, disabling tracking on Python 3.13+."""

        try:
            return shared_memory.SharedMemory(
                name=name,
                track=False,
            )

        except TypeError:
            return shared_memory.SharedMemory(name=name)

    @classmethod
    def _attach(
        cls,
        name: str,
    ) -> shared_memory.SharedMemory:
        """Attach to a shared memory block, caching the handle locally."""

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
        """Copy a NumPy array into a new shared memory block or inline fallback."""

        array = np.asarray(array)

        order: Literal["C", "F"]

        if array.flags.f_contiguous and not array.flags.c_contiguous:
            order = "F"

        else:
            order = "C"

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
        """Encode an xarray Variable or coordinate with its array payload."""

        return {
            "name": name,
            "role": role,
            "dims": tuple(variable.dims),
            "attrs": dict(variable.attrs),
            "array": self._encode_array(np.asarray(variable.data)),
        }

    def _encode_object(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
    ) -> dict[str, Any]:
        """Encode a supported NumPy or xarray object into a serializable spec."""

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
        """Reconstruct a NumPy array from shared memory or inline buffer."""

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
        """Reconstruct an xarray Variable from its encoded specification."""

        variable = xr.Variable(
            spec["dims"],
            cls._decode_array(
                spec["array"],
                readonly=readonly,
            ),
            attrs=dict(spec["attrs"]),
        )

        return variable

    @classmethod
    def _rebuild(
        cls,
        spec: dict[str, Any],
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Reconstruct the original object in a receiving process."""

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
                continue

            del cls._attachments[name]

    def __enter__(self):
        """Enter context manager."""

        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager, ensuring shared memory is unlinked and closed."""

        self.close()
