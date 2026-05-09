from __future__ import annotations

import functools
import getpass
import inspect
import io
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

from IPython import get_ipython
from IPython.display import HTML, display

host = socket.gethostname()
user = getpass.getuser()
home = Path.home()
tmp = Path("/tmp")
n_cpus = int(os.environ.get("NTASKS", len(os.sched_getaffinity(0))))
ipykernel = "ipykernel" in sys.modules
isatty = sys.stdout.isatty() or ipykernel

script_dir = Path(__file__).resolve().parent
current_dask_cluster = None
current_dask_client = None


class RicedDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


LatBounds = namedtuple("LatBounds", ["min", "max"])
LonBounds = namedtuple("LonBounds", ["min", "max"])
HeightBounds = namedtuple("HeightBounds", ["min", "max"])
CenterPoint = namedtuple("CenterPoint", ["lat", "lon", "height"])

ANSI_COLORS = {
    "RESET": "\033[0m",
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BRIGHT_BLACK": "\033[90m",
    "BRIGHT_RED": "\033[91m",
    "BRIGHT_GREEN": "\033[92m",
    "BRIGHT_YELLOW": "\033[93m",
    "BRIGHT_BLUE": "\033[94m",
    "BRIGHT_MAGENTA": "\033[95m",
    "BRIGHT_CYAN": "\033[96m",
    "BRIGHT_WHITE": "\033[97m",
}


class BoundingBox:
    def __init__(
        self,
        lon_bounds: Tuple[float, float],
        lat_bounds: Tuple[float, float],
        height_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        wrap_lon: bool = False,
    ) -> None:
        """
        Initialize a BoundingBox with user-defined longitude and latitude bounds.
        """

        if len(lon_bounds) != 2 or len(lat_bounds) != 2:
            raise ValueError("Bounds must be tuples of length 2.")

        # Longitude handling
        if wrap_lon:
            lon_min, lon_max = lon_bounds
        else:
            lon_min, lon_max = sorted(lon_bounds)

        self.lon = LonBounds(min=lon_min, max=lon_max)

        # Latitude always sorted
        lat_min, lat_max = sorted(lat_bounds)
        self.lat = LatBounds(min=lat_min, max=lat_max)

        # Height handling
        if height_bounds == (None, None):
            self.height = HeightBounds(min=None, max=None)
            height_center = None
        else:
            h_min, h_max = sorted(height_bounds)
            self.height = HeightBounds(min=h_min, max=h_max)
            height_center = 0.5 * (h_min + h_max)

        # Center calculation
        lat_center = 0.5 * (lat_min + lat_max)

        if wrap_lon and lon_max < lon_min:
            # Antimeridian crossing
            lon_center = ((lon_min + lon_max + 360.0) / 2.0) % 360.0
        else:
            lon_center = 0.5 * (lon_min + lon_max)

        self.center = CenterPoint(
            lat=lat_center,
            lon=lon_center,
            height=height_center,
        )

    def __getitem__(self, key: str) -> Any:
        if key not in {"lat", "lon", "height", "center"}:
            raise KeyError(f"{key} is not a valid BoundingBox attribute.")
        return getattr(self, key)

    def __repr__(self) -> str:
        return f"BoundingBox(lon={self.lon}, lat={self.lat}, height={self.height})"


def cwd() -> Path:
    """
    Get the current working directory.
    """
    return Path.cwd().resolve()


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def timeit(func: Callable) -> Callable:
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


def get_fsig(func: Callable) -> dict:
    """
    Get the signature of a function as a dictionary.
    """
    sig = inspect.signature(func)
    params = {}

    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            params[name] = None
        else:
            params[name] = param.default

    return params


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


def handle_errors(type, value, tb):
    frames = traceback.extract_tb(tb)

    frame = [
        f
        for f in frames
        if "site-packages" not in str(Path(f.filename).resolve())
        and f.filename.endswith(".py")
    ][-1]

    file_name = Path(frame.filename).name
    lineno = f"{frame.lineno}"
    code_line = frame.line.strip() if frame.line else ""

    if ipykernel:
        execution_count = get_ipython().execution_count - 1
        e_msg = f"An Exception occurred in cell: In [{execution_count}], line: {lineno}"
    else:
        e_msg = f"An Exception occurred in file: {file_name}, line: {lineno}"

    if isatty:
        RED = ANSI_COLORS["RED"]
        RESET = ANSI_COLORS["RESET"]

    else:
        RED = ""
        RESET = ""

    print(e_msg)
    print(f"  {code_line}  ")
    print(f"{RED}  {'^' * (len(code_line))}  {RESET}")
    print(f"{RED}{type.__qualname__}: {value}{RESET}")


def logexc(*values: Any | None) -> None:
    """
    Log messages and automatically format the active exception, if present.
    """

    if values:
        if all(isinstance(v, str) for v in values):
            print(" ".join(values))
        else:
            for v in values:
                print(v)

    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_type is None:
        return
    handle_errors(exc_type, exc_value, exc_tb)


def fix_vscode_widget():

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


# Set the custom exception handler for the current session
sys.excepthook = handle_errors
