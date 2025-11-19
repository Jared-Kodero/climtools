import atexit
import functools
import getpass
import inspect
import io
import os
import pprint
import random
import resource
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import psutil
from tabulate import tabulate

host = socket.gethostname()
user = getpass.getuser()
home = Path.home()
tmp = Path(os.environ.get("TMPDIR", "/tmp"))
n_cpus = len(os.sched_getaffinity(0))
tmp_files = []

script_dir = Path(__file__).resolve().parent
current_dask_cluster = None
current_dask_client = None


class ConfigMap(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def cwd():
    """
    Get the current working directory.
    """
    return Path.cwd().resolve()


def cleanup():
    rm(tmp_files)


atexit.register(cleanup)


def which(cmd: str) -> bool | None:
    try:
        path = (
            subprocess.check_output(["which", cmd], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        return True if path else False
    except subprocess.CalledProcessError:
        return None


def to_numeric(x, use_numpy: bool = False):
    """
    Cast input x to int or float (optionally using numpy types).
    Returns the original input if casting fails.
    """
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
    def wrapper(*args, **kwargs):
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

        print(f"[ {func.__name__} ] finished in {elapsed:.2f} {unit}")
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


def f_type(file_path: Path) -> str:
    """
    Get the file type using the `file` command in unix-like systems.
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


class MultiProcManager:
    """
    A lightweight, file-based process manager for throttling concurrent execution.

    This class coordinates multiple concurrently running processes by creating
    and removing small marker files in a shared directory (default: ".pids").
    Each active process registers itself by creating a file named after its PID
    and unregisters upon completion. Child processes are grouped under a subdirectory
    named after a specified parent PID (if provided). The manager monitors this directory to
    limit the number of simultaneously active processes based on CPU availability.

    It is designed for lightweight parallel workflows such as batch experiments,
    simulations, or Monte Carlo ensembles, where using a full multiprocessing
    pool introduces unnecessary overhead. It integrates seamlessly with
    ``subprocess.Popen``, ``subprocess.run``, or ``multiprocessing.Process``.

    Parameters
    ----------
    pid : int, optional
        Process ID of the current process. Defaults to ``os.getpid()``.
    main_process_pid : int, optional
        If provided, the process file will be created under a subdirectory
        named after this main process PID. This is useful for grouping
        child processes under a common parent.

    cpu_limit : float, optional
        Fraction of total CPUs allowed to be active simultaneously (default 0.8).


    Methods
    -------
    register():
        Registers the current process by throttling (if necessary) and creating a marker file.
    unregister():
        Unregisters the process by deleting its marker file and cleaning up stale entries.
    throttle():
        Blocks until the fraction of active processes falls below ``cpu_limit``.
    wait():
        Blocks until all process marker files are cleared (i.e., all processes complete).
    sleep(low=0.1, high=5):
        Sleeps for a random interval between ``low`` and ``high`` seconds.

    Examples
    --------
    Basic parallel execution with automatic registration:

    >>> import multiprocessing, random, time, os
    >>> from pathlib import Path
    >>>
    >>> def worker(i):
    ...     with MultiProcManager():
    ...         print(f"Process {i} started (PID {os.getpid()})")
    ...         time.sleep(random.uniform(1, 5))
    ...         print(f"Process {i} finished")
    ...
    >>> if __name__ == "__main__":
    ...     procs = []
    ...     for i in range(20):
    ...         p = multiprocessing.Process(target=worker, args=(i,))
    ...         p.start()
    ...         procs.append(p)
    ...
    ...     for p in procs:
    ...         p.join()
    ...
    ...     MultiProcManager().wait()
    ...     print("All processes completed.")
    """

    def __init__(self, pid: int = None, ppid: int = None, cpu_limit: float = 0.8):

        self.pid = pid or os.getpid()
        self.ppid = ppid
        self.rlimit = resource.getrlimit(resource.RLIMIT_NPROC)
        self.soft_limit = self.rlimit[0]
        self.hard_limit = self.rlimit[1]
        self._sysrand = random.SystemRandom()
        self.n_cpus = n_cpus
        self.cpu_limit = cpu_limit
        self.root = Path.cwd() / ".pids"
        self.storage = self.root
        if self.ppid:
            self.storage = self.storage / str(self.ppid)
            if self.storage.is_file():
                self.storage.unlink(missing_ok=True)
        self.storage.mkdir(exist_ok=True)

        self.proc_file = self.storage / str(self.pid)

        # cleanup stale process files
        self.cleanup()

    def cleanup(self):
        for p in self.storage.rglob("*"):
            try:
                pid = int(p.name)
                # os.kill(pid, 0) raises ProcessLookupError if PID doesn’t exist
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError):
                if p.is_file():
                    p.unlink(missing_ok=True)
                rm(p)
            except PermissionError:
                continue

    def sleep(self, low=0.1, high=10):
        delay = self._sysrand.uniform(low, high)
        time.sleep(delay)

    def ulimit_u(self):
        """
        Monitor the number of processes owned by the current user and block
        if it approaches the system-imposed soft limit.

        Note: This method requires the `psutil` library to be installed.
        """

        threshold = int(0.95 * self.soft_limit)  # e.g., start waiting at ~95% usage

        while True:
            # Count processes owned by the current user
            proc_count = sum(
                1
                for p in psutil.process_iter(["username"])
                if p.info["username"] == psutil.Process().username()
            )

            if proc_count >= threshold:
                self.sleep()

            else:
                break
        return True

    def current_usage(self, path: Path) -> float:
        n_procs = len(list(path.glob("*")))
        return n_procs / self.n_cpus

    def check_concurrency(self, path: Path) -> bool:
        return self.current_usage(path) < 0.95

    def throttle(self):
        """Block until CPU utilization (estimated by active processes) falls below 80%."""

        self.ulimit_u()  # ensure we are within system process limits
        path = self.storage.parent if self.ppid else self.storage

        if self.check_concurrency(path):
            return

        while True:
            n_procs = len(list(path.glob("*")))
            usage = n_procs / self.n_cpus

            if usage <= self.cpu_limit:
                break

            # Sleep briefly before checking again
            self.sleep()

    # a another method to make sure all processes are done
    def wait(self):
        """
        Block until all tracked processes finish.

        If `parent_pid` is set, waits only for its child processes;
        otherwise, waits for all processes under the storage path.
        """

        # check until there are no files in the proc dir
        while True:
            if self.proc_file.is_dir():
                procs = self.proc_file.glob("*")
            else:
                procs = self.root.rglob("*")

            n_procs = len(list(procs))
            if n_procs == 0:
                break
            self.sleep(1, 60)  # Wait for a random time before checking again

    def block(self):
        """
        Block until all tracked processes finish.
        """
        # check until there are no files in the proc dir
        while True:
            n_procs = len(list(self.root.rglob("*")))
            if n_procs == 0:
                break
            self.sleep(1, 120)  # Wait for a random time before checking again

    def register(self):
        """Register the process by creating a process file."""
        self.throttle()
        self.proc_file.touch()

    def unregister(self):
        """Unregister the process if alive"""
        self.proc_file.unlink(missing_ok=True)
        self.cleanup()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unregister()


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


class LogExc:
    """
    Log traceback and exception details.

    This class provides automatic direct to error exception handling without long tracebacks. It supports traceback formatting, exception information, and flexible output redirection to
    file paths or file-like objects.

    Parameters
    ----------
    *values : Any or None
        Objects to log.

    file : Path or Path, optional
        File path or file-like object to write the log messages to. If None, logs to standard output.
    -------
    None
    """

    def __init__(self, *values: Any | None) -> None:
        # Force evaluation of current exception info inside LogMsg
        LogMsg(*values)
        return None

    def __repr__(self):
        return ""


class LogMsg:
    """
    Log one or more messages to standard output or a file, optionally including traceback and exception details.

    This class provides structured logging  and automatic direct to error exception handling without long tracebacks. It supports traceback formatting, exception information, and flexible output redirection to
    file paths or file-like objects.

    Parameters
    ----------
    *values : Any or None
        Objects to log.

    file : Path or Path, optional
        File path or file-like object to write the log messages to. If None, logs to standard output.
    -------
    None
    """

    def __init__(self, *values: Any | None) -> None:
        self.RED = "\033[31m"
        self.BOLD = "\033[1m"
        self.RESET = "\033[0m"
        self.jupyter = "ipykernel" in sys.modules
        self.isatty = sys.stdout.isatty() or self.jupyter
        self.values = values if len(values) > 0 else None
        self.fd = sys.stdout.fileno()
        self.exc_info = sys.exc_info()
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

        if pretty:
            pprint.pprint(obj, sort_dicts=False, compact=True)
        elif self.jupyter:
            print(obj)
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
                f"{lineno} | {frame_line}\n{pointer}\n\t"
                f"  {frame.name} :  {file_path}\n"
            )
            new_ft.append(frame_msg)

        new_ft = "\n".join(new_ft)
        error_type = f"{self.exc_type.__qualname__} : {self.exc_value}"

        output = f"\n{error_type}\n {new_ft}\n"

        self.lprint(output, pretty=False)

        return None

    def __repr__(self):
        return ""
