import atexit
import functools
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Union

from .log import *

CPU_COUNT = len(os.sched_getaffinity(0))

CURRENT_DASK_CLUSTER = None
CURRENT_DASK_CLIENT = None


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


def mkdir(arg: Union[str, Path]) -> None:
    """
    Create a directory using the mkdir command in unix-like systems.

    """

    try:

        if isinstance(arg, Path):
            arg = str(arg)
        elif isinstance(arg, list):
            arg = [str(i) for i in arg]

        arg = str(arg).strip()

        if arg in {"", ".", ".."}:
            raise ValueError("Invalid or empty path passed to mkdir command")

        subprocess.run(
            [
                "mkdir",
                "-p",
                arg,
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return None


def file_type(file_path: Union[str, Path]) -> str:
    """
    Get the file type using the `file` command in unix-like systems.
    """

    if isinstance(file_path, Path):
        file_path = str(file_path)

    result = subprocess.run(
        ["file", "-b", file_path],
        check=True,
        text=True,
        capture_output=True,
    )

    return result.stdout.strip()


def rm(arg: Union[str, Path, list[Union[str, Path]]]) -> None:
    """
    Remove files or directories using the rm command in unix-like systems.
    By default, it uses the -rf flags to remove directories and their contents recursively and forcefully.

    """

    try:

        if isinstance(arg, str):
            arg = [arg]
        elif isinstance(arg, Path):
            arg = [str(arg)]
        elif isinstance(arg, list):
            arg = [str(i) for i in arg]

        arg = [str(i).strip() for i in arg]

        if any(i in {"", ".", ".."} for i in arg):
            raise ValueError(
                "Invalid, dangerous or empty path passed to rm -rf command"
            )

        subprocess.run(
            [
                "rm",
                "-rf",
                *arg,
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return None


def cp(
    src: Union[str, Path],
    dst: Union[str, Path],
) -> None:
    """
    Copy files or directories using the cp command.
    By default, it uses the -rf flags to copy directories and their contents recursively and forcefully.

    """

    try:

        if isinstance(src, Path):
            src = str(Path(src).resolve())
        if isinstance(dst, Path):
            dst = str(Path(dst).resolve())

        subprocess.run(
            [
                "cp",
                "-rf",
                src,
                dst,
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return None


def mv(
    src: Union[str, Path],
    dst: Union[str, Path],
) -> None:
    """
    Move files or directories using the mv command.

    """

    try:

        if isinstance(src, Path):
            src = str(Path(src).resolve())
        if isinstance(dst, Path):
            dst = str(Path(dst).resolve())

        subprocess.run(
            [
                "mv",
                "-f",
                src,
                dst,
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return None


def close_dask():
    """
    Close the active Dask client and cluster if they exist.
    This is useful for cleaning up resources when done with Dask computations.
    """
    global CURRENT_DASK_CLIENT, CURRENT_DASK_CLUSTER
    if CURRENT_DASK_CLIENT and CURRENT_DASK_CLUSTER:
        CURRENT_DASK_CLIENT.close()
        CURRENT_DASK_CLUSTER.close()
        CURRENT_DASK_CLIENT = None
        CURRENT_DASK_CLUSTER = None


def setup_dask(
    *,
    workers: int = CPU_COUNT,
    threads_per_worker: int = 1,
    processes=True,
    filter_warnings=True,
):
    """
    - Imports Dask and Dask distributed.
    - Creates a Dask client.
    - Sets up the Dask dashboard.

    Parameters:
    ___________
        workers (int, optional): Number of workers to create. Default is 8.
        threads_per_worker (int, optional): Number of threads per worker. Default is 4.
        processes (bool, optional): Whether to use processes instead of threads. Default is True.
        get_info (bool, optional): Whether to return the Dask dashboard URL. Default is False.
        dynamic_port (bool, optional): Whether to use a dynamic port. Default is False and uses port 8787.
        filter_warnings (bool, optional): Whether to filter warnings. Default is True.

    Example:
    ________
        >>> setup_dask(get_info=True, filter_warnings=False)
    """

    global CURRENT_DASK_CLIENT, CURRENT_DASK_CLUSTER

    if CURRENT_DASK_CLIENT and CURRENT_DASK_CLUSTER:
        return CURRENT_DASK_CLIENT

    from dask.distributed import Client, LocalCluster

    if filter_warnings:
        silence_level = logging.ERROR
    else:
        silence_level = logging.WARN

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=threads_per_worker,
        memory_limit=0,
        silence_logs=silence_level,
        processes=processes,
    )
    client = Client(cluster)

    CURRENT_DASK_CLIENT = client
    CURRENT_DASK_CLUSTER = cluster

    def _cleanup():
        CURRENT_DASK_CLIENT.close()
        CURRENT_DASK_CLUSTER.close()

    atexit.register(_cleanup)

    return client
