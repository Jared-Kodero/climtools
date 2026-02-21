from __future__ import annotations

import functools
import getpass
import inspect
import io
import os
import pprint
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from tabulate import tabulate
from tqdm import tqdm as tqdm_terminal
from tqdm.notebook import tqdm as tqdm_notebook

host = socket.gethostname()
user = getpass.getuser()
home = Path.home()
tmp = Path("/tmp")
n_cpus = len(os.sched_getaffinity(0))


script_dir = Path(__file__).resolve().parent
current_dask_cluster = None
current_dask_client = None


class RicedDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BoundingBox:
    def __init__(
        self,
        lon_bounds: tuple[float, float],
        lat_bounds: tuple[float, float],
        height_bounds: tuple[float, float] = (None, None),
        wrap_lon: bool = False,
    ) -> BoundingBox:
        """
        Initialize a BoundingBox with user-defined longitude and latitude bounds.

        Parameters
        ----------
        lon_bounds : tuple[float, float]
            Upper and lower longitude values
        lat_bounds : tuple[float, float]
            Upper and lower latitude values
        height_bounds : tuple[float, float], optional
            Upper and lower height values. Default is (None, None).

        wrap_lon : bool, optional
            If True, the longitude bounds are used as provided, else they are sorted
            to ensure min < max. Default is False.
        """
        if len(lon_bounds) != 2 or len(lat_bounds) != 2:
            raise ValueError("Bounds must be tuples of length 2.")

        LatBounds = namedtuple("LatBounds", ["min", "max"])
        LonBounds = namedtuple("LonBounds", ["min", "max"])
        HeightBounds = namedtuple("HeightBounds", ["min", "max"])
        CenterPoint = namedtuple("CenterPoint", ["lat", "lon", "height"])

        if wrap_lon:
            self.lon = LonBounds(min=lon_bounds[0], max=lon_bounds[1])
        else:
            self.lon = LonBounds(min=min(lon_bounds), max=max(lon_bounds))

        self.lat = LatBounds(min=min(lat_bounds), max=max(lat_bounds))

        if height_bounds == (None, None):
            self.height = HeightBounds(min=None, max=None)
        else:
            self.height = HeightBounds(min=min(height_bounds), max=max(height_bounds))

        _lat_c = (lat_bounds[0] + lat_bounds[1]) / 2
        _lon_c = (lon_bounds[0] + lon_bounds[1]) / 2

        if height_bounds == (None, None):
            _height_c = None
        else:
            _height_c = (height_bounds[0] + height_bounds[1]) / 2

        self.center = CenterPoint(lat=_lat_c, lon=_lon_c, height=_height_c)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def cwd() -> Path:
    """
    Get the current working directory.
    """
    return Path.cwd().resolve()


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def to_numeric(x: Any, use_numpy: bool = False) -> Any:
    """
    Cast input x to int or float (optionally using numpy types).
    Returns the original input if casting fails.
    """
    if not isinstance(x, str):
        return x

    if x is None or str(x).strip() == "":
        return np.nan if use_numpy else None

    _int = np.int64 if use_numpy else int
    _float = np.float64 if use_numpy else float

    try:
        if "." in str(x):
            res = _float(x)
        else:
            res = _int(x)
    except (ValueError, TypeError):
        try:
            res = _float(x)
        except (ValueError, TypeError):
            res = None

    if res is None:
        return x

    return res


def timeit(func):
    """
    Decorator to time a function and print its runtime in appropriate units.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()

        result = func(*args, **kwargs)
        end = time.perf_counter()

        elapsed = end - start
        unit = "seconds"

        if elapsed > 86400:  # > 1 day
            elapsed /= 86400
            unit = "days"
        elif elapsed > 3600:  # > 1 hour
            elapsed /= 3600
            unit = "hours"
        elif elapsed > 60:  # > 1 minute
            elapsed /= 60
            unit = "minutes"

        print(f"[ {func.__name__} ] finished in {round(elapsed, 2):>4} {unit}")
        return result

    return wrapper


def mkdir(path: Path):
    """
    Create a directory using the mkdir command in unix-like systems.

    """
    path = Path(path).resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def du(path: Path, units: Literal["B", "kB", "MB", "GB", "TB"] = "B") -> int | float:
    """
    Return the size of a file or directory.

    Parameters
    ----------
    path : Path
        File or directory to measure.
    units : {"B", "kB", "MB", "GB", "TB"}, default "B"
        Output units using SI decimal scaling (1 kB = 10^3 B).

    """

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    scale = {
        "B": 1,
        "kB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "TB": 10**12,
    }
    return size / scale[units]


def file_kind(
    file_path: Path,
) -> str:
    """
    Tests each argument in an attempt to classify it. for more info, see `file` unix command.
    """

    if isinstance(file_path, Path):
        file_path = str(file_path)

    cmd = ["file", "-b", file_path]

    try:
        res = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("ERROR :", e.stderr)
        return None


def symlink(
    src: Path,
    dst: Path,
):
    """
    Create a symbolic link from src to dst.
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()

    # Create parent directories for the link
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing link or file if already exists
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    # Create symbolic link
    if src.is_dir():
        dst.symlink_to(src, target_is_directory=True)
    elif src.is_file():
        dst.symlink_to(src)
    else:
        raise FileNotFoundError(f"Source path does not exist: {src}")


def rm(arg: Path | list[Path]):
    """
    Remove files or directories

    """

    if not isinstance(arg, list):
        arg = [arg]

    for f in arg:
        f = Path(f).resolve()
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f, ignore_errors=True)


def cp(
    src: Path,
    dst: Path,
):
    """
    Copy files or directories

    """

    src = Path(src).resolve()
    dst = Path(dst).resolve()

    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    elif src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def mv(
    src: Path,
    dst: Path,
):
    """
    Move files or directories

    """

    src = Path(src).resolve()
    dst = Path(dst).resolve()

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)


def get_func_signature(func):
    """
    Get the signature of a function as a dictionary.
    """
    sig = inspect.signature(func)
    return {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in sig.parameters.items()
    }


class RedirectStreams:
    """
    A lightweight context manager to temporarily redirect `sys.stdout` and `sys.stderr`.

    This utility captures all printed output and error messages within its context,
    either to an in-memory buffer (`io.StringIO`) or to a file on disk. It is
    especially useful in parallel or batch workflows where subprocesses may print
    diagnostic information that needs to be captured for logging or post-analysis.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Path to a file where the redirected output will be written.
        If `None` (default), output is captured in memory using an `io.StringIO` buffer.
    mode : str, optional
        File mode for the output file if `path` is provided (default is `"w+"`).
        Common options:
        - `"w"` : overwrite file each time.
        - `"a"` : append to existing file.
        - `"w+"` : overwrite and allow reading.

    Attributes
    ----------
    target : io.StringIO or io.TextIOWrapper
        The active output stream (in-memory or file object).
    state : str
        Either `"buffer"` (for in-memory) or `"file"` (for file-based capture).
    _original_stdout, _original_stderr : io.TextIOWrapper
        Saved references to the original standard output and error streams.

    Methods
    -------
    start():
        Redirect `sys.stdout` and `sys.stderr` to the target.
    stop():
        Restore `sys.stdout` and `sys.stderr` to their original streams.
    retrieve() -> str:
        Return the captured output as a string. Only available for buffer mode.
        For file-based redirection, reads from the target file.
    __enter__(), __exit__():
        Context management protocol for use with `with` statements.

    Examples
    --------
    **Example 1: Capture output in memory**

    >>> from pathlib import Path
    >>> import sys
    >>> redirector = RedirectStreams()
    >>> with redirector:
    ...     print("This will be captured.")
    ...     sys.stderr.write("Error message.\\n")
    >>> print(redirector.retrieve())
    This will be captured.
    Error message.

    **Example 2: Redirect output to a file**

    >>> from pathlib import Path
    >>> log_path = Path("output.log")
    >>> with RedirectStreams(log_path, mode="w") as rs:
    ...     print("Writing to log file...")
    ...     sys.stderr.write("This also goes to the log.\\n")
    >>> # Retrieve contents for verification
    >>> print(rs.retrieve())
    Writing to log file...
    This also goes to the log.

    **Example 3: Combined with multiprocessing or logging**

    >>> import multiprocessing
    >>> def task(i):
    ...     with RedirectStreams(Path(f"task_{i}.log")):
    ...         print(f"Task {i} started")
    ...         print(f"Task {i} completed")
    >>> if __name__ == "__main__":
    ...     procs = [multiprocessing.Process(target=task, args=(i,)) for i in range(4)]
    ...     for p in procs: p.start()
    ...     for p in procs: p.join()

    Notes
    -----
    - Always call `retrieve()` *after* exiting the context if you want to
      inspect the captured output.
    - When using file mode, the file remains open until `stop()` is called.
      It is automatically closed at the end of the context.
    """

    def __init__(self, path: Path = None, mode: str = "w+"):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.state = None

        if path is None:
            self.target = io.StringIO()
            self.state = "buffer"
        else:
            self.target = open(path, mode)
            self.state = "file"

    def retrieve(self) -> str:
        """Retrieve the contents of the redirected streams."""
        if self.state == "buffer":
            return self.target.getvalue()
        else:
            self.target.seek(0)
            contents = self.target.read()
            self.target.seek(0, os.SEEK_END)
            return contents

    def start(self):
        """Redirect sys.stdout and sys.stderr to the target."""
        sys.stdout = self.target
        sys.stderr = self.target

    def stop(self):
        """Restore sys.stdout and sys.stderr to their original values."""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self.target.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class FileLock:
    """
    A lightweight file-based lock mechanism for inter-process synchronization.

    This class provides a simple and portable locking mechanism using a sentinel
    file to coordinate access between concurrent processes. The lock is acquired
    by atomically creating a lock file, and released by deleting it. If the file
    already exists, the process waits (with randomized backoff) until it becomes
    available.

    File-based locks are especially useful in multi-process or distributed
    environments where shared state is coordinated through the filesystem
    (e.g., cluster scratch directories or shared network mounts).

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Path to the lock file. If `None` (default), a lock file named
        `.lock` is created in the current working directory.

    Attributes
    ----------
    path : pathlib.Path
        Filesystem path of the lock file.
    _sysrand : random.SystemRandom
        Cryptographically secure random number generator for backoff delays.

    Methods
    -------
    acquire():
        Acquire the lock by atomically creating the lock file.
    release():
        Release the lock by deleting the lock file.
    sleep(low=0.1, high=5):
        Wait for a random delay between `low` and `high` seconds before retrying.
    __enter__(), __exit__():
        Context management support for use with `with` statements.

    Examples
    --------
    **Example 1: Basic use**

    >>> from pathlib import Path
    >>> lock = FileLock(Path("mytask.lock"))
    >>> with lock:
    ...     print("Lock acquired, performing critical section...")
    ...     # Perform safe file write or shared resource update
    ...     time.sleep(2)
    >>> print("Lock released.")

    **Example 2: Protect shared output in parallel tasks**

    >>> import multiprocessing, time
    >>> from pathlib import Path
    >>> def task(i):
    ...     with FileLock(Path("shared.lock")):
    ...         with open("shared.txt", "a") as f:
    ...             f.write(f"Task {i} started\\n")
    ...             time.sleep(0.5)
    ...             f.write(f"Task {i} finished\\n")
    >>> if __name__ == "__main__":
    ...     procs = [multiprocessing.Process(target=task, args=(i,)) for i in range(4)]
    ...     for p in procs: p.start()
    ...     for p in procs: p.join()
    >>> print(Path("shared.txt").read_text())

    **Example 3: Custom backoff interval**

    >>> lock = FileLock()
    >>> lock.sleep(low=0.5, high=2)  # Wait with controlled random delay
    >>> # Typically used internally when the lock file already exists.

    Notes
    -----
    - The lock is implemented via `os.open(..., os.O_CREAT | os.O_EXCL)` for
      atomic file creation across processes.
    - Safe to use across processes on shared filesystems (e.g., NFS, Lustre),
      provided atomic file creation is supported.
    - Always use the context manager form (`with FileLock(...):`) to ensure
      release even if exceptions occur.
    """

    def __init__(self, path: Path = None):
        self.path = path if path is not None else Path.cwd() / ".lock"
        self._sysrand = random.SystemRandom()

    def sleep(self, low=0.1, high=5):
        """Wait for a random duration between `low` and `high` seconds before retrying."""
        delay = self._sysrand.uniform(low, high)
        time.sleep(delay)

    def acquire(self):
        """
        Acquire the file lock by creating a lock file atomically.

        This method blocks until the lock file can be created. If another process
        already holds the lock, the method waits for a random delay and retries.

        Notes
        -----
        The atomic creation uses:
        - `os.O_CREAT` : create file if it does not exist.
        - `os.O_EXCL`  : fail if file already exists (ensures atomicity).
        - `os.O_WRONLY`: open for writing only.
        """
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break  # Lock acquired successfully
            except FileExistsError:
                self.sleep()

    def release(self):
        """Release the file lock by deleting the lock file."""
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class LogMsg:
    """
    Log one or more messages to standard output or a file, optionally including traceback and exception details.

    This class provides structured logging  and automatic direct to error exception handling without long tracebacks. It supports traceback formatting, exception information, and flexible output redirection to
    file paths or file-like objects.

    Parameters
    ----------
    *values : Any or None
        Objects to log.
    -------
    None
    """

    def __init__(self, *values: Any | None, exc_info=(None, None, None)) -> None:
        self.RED = "\033[31m"
        self.BOLD = "\033[1m"
        self.RESET = "\033[0m"
        self.ipykernel = "ipykernel" in sys.modules
        self.isatty = sys.stdout.isatty() or self.ipykernel
        self.values = values if len(values) > 0 else None
        self.fd = sys.stdout.fileno()
        self.exc_info = exc_info
        self.exc_type = self.exc_info[0]
        self.exc_value = self.exc_info[1]
        self.exc_tb = self.exc_info[2]
        self.has_exc = self.exc_type is not None

        if self.values:
            self.parse_values()
        if self.has_exc:
            self.check_exceptions()

        return None

    def parse_values(self) -> None:
        if all(isinstance(v, str) for v in self.values):
            values = " ".join(self.values)
            self.lprint(values, pretty=False)
        else:
            for v in self.values:
                if not isinstance(v, str):
                    self.lprint(v, pretty=True)
                else:
                    self.lprint(v, pretty=False)
        return None

    def lprint(self, obj: Any, pretty: bool = True) -> None:
        if isinstance(obj, pd.DataFrame):
            obj = tabulate(obj, headers="keys", tablefmt="psql", showindex=False)
            print("\n")
            print(obj)
            print("\n")
            return None

        if self.ipykernel:
            print(obj)

        elif pretty:
            pprint.pprint(obj, sort_dicts=False, compact=True)

        else:
            _ = os.write(self.fd, f"{obj}\n".encode("utf-8"))
        return None

    def check_exceptions(self) -> None:
        ft = traceback.extract_tb(self.exc_tb)
        ft_user = [
            x
            for x in ft
            if "site-packages" not in str(Path(x.filename).resolve())
            and str(x.filename).endswith(".py")
        ]

        ft = ft_user
        new_ft = []
        for frame in ft:
            file_path = Path(frame.filename).resolve()
            lineno = f"{frame.lineno:>5}"
            frame_line = frame.line.strip() if frame.line else ""
            pointer = " " * (len(str(lineno)) + 3) + "^" * len(frame_line)

            if self.isatty:
                frame_line = f"{self.RED}{frame_line}{self.RESET}"
                pointer = f"{self.RED}{pointer}{self.RESET}"

            frame_msg = (
                f"{lineno} | {frame_line}\n{pointer}\n\t  {frame.name} :  {file_path}\n"
            )
            new_ft.append(frame_msg)

        new_ft = "\n".join(new_ft)
        error_type = f"{self.exc_type.__qualname__} : {self.exc_value}"

        output = f"\n{error_type}\n {new_ft}\n"

        self.lprint(output, pretty=False)

        return None

    def __repr__(self):
        return ""


@dataclass
class AdaptiveIteratorWithProgress:
    """
    Adaptive iterator with Progress reporting.
    """

    iterable: Iterable[Any]
    log_id: str = "Job"
    message: str = "Progress"
    major_step: int = 10
    minor_step: int = 1

    _count: int = 0
    _total: int = 0
    _prev_pct: float = -1.0
    _threshold: float = 0.0
    _use_tqdm: bool = False

    def __enter__(self):
        self._total = len(self.iterable)

        if "ipykernel" in sys.modules:
            self._tqdm_cls = tqdm_notebook
            self._use_tqdm = True

        elif sys.stdout.isatty():
            self._tqdm_cls = tqdm_terminal
            self._use_tqdm = True

        else:
            self._use_tqdm = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def __iter__(self) -> Iterator[Any]:
        if self._use_tqdm:
            with self._tqdm_cls(self.iterable, desc=self.log_id) as pbar:
                for item in pbar:
                    yield item
        else:
            for item in self.iterable:
                yield item
                self._advance()

    def _advance(self):
        self._count += 1
        pct = round(100.0 * self._count / self._total)

        if (
            (pct >= self._threshold) or (self._count == self._total)
        ) and pct != self._prev_pct:
            logmsg(f"{self.message} : {self.log_id:>25} {pct:7.2f}% completed.")

            self._threshold = self._next_threshold(
                pct, self.major_step, self.minor_step
            )
            self._prev_pct = pct

    @staticmethod
    def _next_threshold(percent: float, major_step: float, minor_step: float) -> float:
        step = 1.0
        if percent < 80.0:
            step = major_step
        elif percent < 95.0:
            step = minor_step

        return min(100.0, ((percent // step) + 1.0) * step)


def logmsg(*values: Any | None) -> LogMsg:
    """
    Log one or more messages to standard output or a file, optionally including traceback and exception details.

    This function provides structured logging  and automatic direct to error exception handling without long tracebacks. It supports traceback formatting, exception information, and flexible output redirection to
    file paths or file-like objects.

    Parameters
    ----------
    *values : Any or None
        Objects to log.
    -------
    LogMsg
        An instance of the LogMsg class.
    """
    if len(values) == 0:
        return None
    return LogMsg(*values)


def logexc(*values: Any | None) -> LogMsg:
    """
    Log traceback and exception details.

    This function provides automatic direct to error exception handling without long tracebacks. It supports traceback formatting, exception information, and flexible output redirection to
    file paths or file-like objects.

    Parameters
    ----------
    *values : Any or None
        Objects to log.
    -------
    logexc
        An instance of the logexc class.
    """
    # if there is no exception, do nothing
    exc_info = sys.exc_info()
    exc_type = exc_info[0]
    if exc_type is None:
        return None
    return LogMsg(*values, exc_info=exc_info)


def logobj(*values: Any | None) -> LogMsg:
    """
    Log one or more non string objects to standard output or a file, optionally including traceback and exception details.

    This function provides structured logging. It supports traceback formatting, exception information, and flexible output redirection to
    file paths or file-like objects.

    Parameters
    ----------
    *values : Any or None
        Objects to log.
    -------
    logobj
        An instance of the logobj class.
    """
    # if all values are strings, do nothing
    if all(isinstance(v, str) for v in values):
        return None
    return LogMsg(*values)


def _aip(
    iterable: Iterable[Any],
    log_id: str = "Job",
    message: str = "Progress",
    major_step: int = 10,
    minor_step: int = 1,
):
    """
    Adaptive Iterator With Progress Reporting.
    """
    return AdaptiveIteratorWithProgress(
        iterable=iterable,
        log_id=log_id,
        message=message,
        major_step=major_step,
        minor_step=minor_step,
    )


aip: AdaptiveIteratorWithProgress = _aip


def set_vscode_widget_theme():
    css = """
    <style>
    /* overwrite hard-coded white background by VS Code for ipywidgets */
    .cell-output-ipywidget-background {
        background-color: transparent !important;
    }

    /* map VS Code theme variables to Jupyter widget variables */
    :root {
        --jp-widgets-color: var(--vscode-editor-foreground);
        --jp-widgets-font-size: var(--vscode-editor-font-size);
    }
    </style>
    """
    display(HTML(css))


if "ipykernel" in sys.modules:
    set_vscode_widget_theme()
