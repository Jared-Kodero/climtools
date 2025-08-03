import pprint
import sys
import traceback
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Union

from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init()  # initialize colorama for colored output


def line_break(char: str = "═", n: int = 120):
    """
    Print a line break with a specified character and length.
    """
    print(char * n)


def log(
    *values: Any | None,
    level: Literal["INFO", "ERROR", "WARNING", "DEBUG"] = None,
    out: Union[str, PathLike, Path, None] = None,
    full_traceback: bool = False,
    frames: int = 5,
    exception: bool = True,
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

    full_traceback : bool, default=False
        If True, include the full exception traceback regardless of frame depth.
        Overrides the `frames` parameter.

    frames : int, default=5
        Number of stack frames to include in the traceback when `full_traceback` is False.

    exception : bool, default=True
        Whether to include exception information (e.g., exception type and message) in the log output.

    Returns
    -------
    None
    """

    if any(not isinstance(v, str) for v in values):
        return _obj_print(values, out)

    msg_str = " ".join(map(str, values)) if values else ""
    caller_frame = sys._getframe(1)
    file = caller_frame.f_code.co_filename
    module = Path(file).stem
    c_time = datetime.now().strftime("%H:%M:%S")
    is_term = sys.stdout.isatty()

    if is_term:
        result = _on_term(
            msg_str,
            level,
            module,
            c_time,
            frames=frames,
            full_traceback=full_traceback,
            exception=exception,
        )
    else:
        result = _on_any(
            msg_str,
            level,
            module,
            c_time,
            frames=frames,
            full_traceback=full_traceback,
            exception=exception,
        )

    if out is None:
        print(result, flush=True)
    else:
        with open(out, "a") as f:
            f.write(result + "\n")

    return None


def _on_any(
    msg,
    level,
    module,
    c_time,
    frames,
    full_traceback,
    exception,
) -> None:
    exc_type, exc_value, exc_traceback = sys.exc_info()

    if exc_type is None or not exception:
        level = "INFO" if level is None else level
        output = f"{c_time} - {level:<5} - {module} - {msg}"
    else:

        level = "ERROR" if level is None else level
        ft = traceback.extract_tb(exc_traceback)
        ft = [f for f in ft]
        # ft.reverse()

        ft_user = [
            frame
            for frame in ft
            if "site-packages" not in str(Path(frame.filename).resolve())
        ]

        ft_sys = [
            frame
            for frame in ft
            if "site-packages" in str(Path(frame.filename).resolve())
        ]

        ft = ft_user + ft_sys

        if not full_traceback:
            ft = ft[:frames]

        new_ft = []
        for frame in ft:
            file_path = Path(frame.filename).resolve()
            frame_msg = (
                f"{frame.lineno:>5} |{frame.line.strip()}\n\t"
                f" ⤷ {frame.name}  ->  {file_path}\n"
            )
            new_ft.append(frame_msg)
        new_ft = "\n".join(new_ft)
        error_type = f"{exc_type.__qualname__}:\n -> {exc_value}"
        output = (
            f"{c_time} - "
            f"{level:<5} - "
            f"{module} - "
            f"{error_type}\n"
            f"{new_ft}\n\t{msg}\n"
        )

    return output


def _on_term(
    msg,
    level,
    module,
    c_time,
    frames,
    full_traceback,
    exception,
) -> None:

    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
    COLOR_MAP = {
        None: Fore.GREEN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "DEBUG": Fore.CYAN,
        "MODULE": Fore.BLUE,
        "TIME": Fore.WHITE,
    }

    t_color = COLOR_MAP["TIME"]
    l_color = COLOR_MAP.get(level, Fore.GREEN)
    m_color = COLOR_MAP["MODULE"]

    exc_type, exc_value, exc_traceback = sys.exc_info()

    if exc_type is None or not exception:
        level = "INFO" if level is None else level
        output = (
            f"{t_color}{c_time}{RESET} - "
            f"{BOLD}{l_color}{level:<5}{RESET} - "
            f"{m_color}{module}{RESET} - {msg}"
        )
    else:
        level = "ERROR" if level is None else level
        l_color = COLOR_MAP[level]
        ft = traceback.extract_tb(exc_traceback)
        ft = [f for f in ft]
        # ft.reverse()

        ft_user = [
            frame
            for frame in ft
            if "site-packages" not in str(Path(frame.filename).resolve())
        ]

        ft_sys = [
            frame
            for frame in ft
            if "site-packages" in str(Path(frame.filename).resolve())
        ]

        ft = ft_user + ft_sys

        if not full_traceback:
            ft = ft[:frames]

        new_ft = []
        for frame in ft:
            file_path = Path(frame.filename).resolve()
            frame_msg = (
                f"{BOLD}{frame.lineno:>5}{RESET} | "
                f"{l_color}{frame.line.strip()}{RESET}\n\t ⤷ "
                f"{frame.name}{RESET}  ->  {m_color}{file_path}{RESET}"
            )
            new_ft.append(frame_msg)

        new_ft = "\n".join(new_ft)

        error_type = f"{exc_type.__qualname__}:\n -> {exc_value}"
        output = (
            f"{t_color}{c_time}{RESET} - "
            f"{BOLD}{l_color}{level:<5}{RESET} - "
            f"{m_color}{module}{RESET} - "
            f"{BOLD}{l_color}{error_type}{RESET}\n"
            f"{new_ft}\n\t{msg}\n"
        )

    return output


def _obj_print(values, out):
    values = [*values]

    if out:
        with open(out, "a") as f:
            for v in values:
                pprint.pprint(v, stream=f, sort_dicts=True, compact=True)
    else:
        for v in values:
            pprint.pprint(v, sort_dicts=True, compact=True)
    return
