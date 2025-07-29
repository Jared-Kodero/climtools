import pprint
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init()  # initialize colorama for colored output


def log(
    *msg: Any | None,
    level: Literal["INFO", "ERROR", "WARNING", "DEBUG"] = None,
    frames=5,
    full_traceback: bool = False,
    exception: bool = True,
) -> None:
    """

    This function logs messages to standard output.

    Parameters:

        msg (str, optional): Message to log. Default is None.
        level (Literal["INFO", "ERROR", "WARNING", "DEBUG"], optional): Type of message. Default is None.
        frames (int, optional): Number of frames to display in the traceback. Default is 2.
        full_traceback (bool, optional): Whether to display the full traceback. Default is False.
        exception (bool, optional): Whether to display the exception information. Default is True.


    Example:

        >>> log("This is an error message", level="ERROR")
        12:00 - ERROR - module_name - This is an error message

        >>> log("This is an error message", level="ERROR", full_traceback=True)
        12:00 - ERROR - module_name - ValueError
          -> This is an error message
          File "module_name", line 10, in <module>
            raise ValueError("This is an error message")
    """

    if len(msg) == 1 and not isinstance(msg[0], str):
        pprint.pprint(msg[0], sort_dicts=True, compact=True)
        return

    msg_text = " ".join(map(str, msg)) if msg else ""
    caller_frame = sys._getframe(1)
    file = caller_frame.f_code.co_filename
    module = Path(file).stem
    c_time = datetime.now().strftime("%H:%M:%S")
    is_term = sys.stdout.isatty()

    if is_term:
        log_on_term(
            msg_text,
            level,
            module,
            c_time,
            frames=frames,
            full_traceback=full_traceback,
            exception=exception,
        )
    else:
        log_on_any(
            msg_text,
            level,
            module,
            c_time,
            frames=frames,
            full_traceback=full_traceback,
            exception=exception,
        )

    return None


def log_on_any(
    msg: str,
    level: Literal["INFO", "ERROR", "WARNING", "DEBUG"] | None,
    module: str,
    c_time: str,
    frames: int,
    full_traceback: bool,
    exception: bool,
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
            f"{c_time} - {level:<5} - {module} - {error_type}\n" f"{new_ft}\n\t{msg}\n"
        )

    print(output, flush=True)


def log_on_term(
    msg: str,
    level: Literal["INFO", "ERROR", "WARNING", "DEBUG"] | None,
    module: str,
    c_time: str,
    frames: int,
    full_traceback: bool,
    exception: bool,
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

    print(output, flush=True)
