import pprint
import sys
import traceback
from os import PathLike
from pathlib import Path
from typing import Any

# Detect whether output is a terminal
use_color = sys.stdout.isatty()

# ANSI color codes
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _print_result(v, file, pretty=True):
    if file is None and pretty:
        pprint.pprint(v, sort_dicts=False, compact=True)
    elif file and pretty:
        with open(file, "a") as f:
            pprint.pprint(v, stream=f, sort_dicts=False, compact=True)

    elif file is None and not pretty:
        print(v)
    elif file and not pretty:
        with open(file, "a") as f:
            print(v, file=f)


def log(
    *values: Any | None,
    file: PathLike | Path = None,
) -> None:
    """
    Log one or more messages to standard output or a file, optionally including traceback and exception details.

    This utility function provides structured logging with support for log levels,
    traceback formatting, exception information, and flexible output redirection to
    file paths or file-like objects.

    Parameters
    ----------
    *values : Any or None
        Objects to log.

    level : {'INFO', 'ERROR', 'WARNING', 'DEBUG'}, optional
        Logging level tag to prepend to the message. If not specified, no level is shown.

    out : str or PathLike or Path or file-like object, optional
        Destination for the log output. May be a file path (str or Path), or an open
        file-like object. If None, logs to standard output (stdout).

    full_traceback : bool, default=True
        By default, includes the full traceback in the log output when an exception is present.
        If False, limits the traceback to a specified number of frames.

    frames : int, default=5
        Number of stack frames to include in the traceback when `full_traceback` is False.

    exception : bool, default=True
        Whether to include exception information (e.g., exception type and message) in the log output.

    Returns
    -------
    None
    """

    exc_info = sys.exc_info()

    for v in values:
        if not isinstance(v, str):
            return _print_result(v, file, pretty=True)
        elif not any(exc_info):
            _print_result(v, file, pretty=False)

    if any(exc_info):
        return _exceptions(values, file, exc_info)


def _exceptions(values, file, exc_info):

    msg = " ".join(map(str, values)) if values else ""
    exc_type, exc_value, exc_traceback = exc_info

    ft = traceback.extract_tb(exc_traceback)

    ft_user = [
        x
        for x in ft
        if "site-packages" not in str(Path(x.filename).resolve())
        and ".pyx" not in str(Path(x.filename).resolve())
    ]

    ft = ft_user
    new_ft = []
    for frame in ft:
        file_path = Path(frame.filename).resolve()
        lineno = f"{frame.lineno:>5}"
        frame_line = frame.line.strip() if frame.line else ""
        pointer = " " * (len(str(lineno)) + 3) + "^" * len(frame_line)

        if use_color:
            frame_line = f"{RED}{frame_line}{RESET}"
            pointer = f"{RED}{pointer}{RESET}"

        frame_msg = (
            f"{lineno} | {frame_line}\n{pointer}\n\t" f"  {frame.name} :  {file_path}\n"
        )
        new_ft.append(frame_msg)

    new_ft = "\n".join(new_ft)
    error_type = f"{exc_type.__qualname__} : {exc_value}"

    output = f"\n{error_type}\n {new_ft}\n\t{msg}\n"

    _print_result(output, file, pretty=False)

    return None
