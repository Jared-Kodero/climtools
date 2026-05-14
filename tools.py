from __future__ import annotations

import getpass
import inspect
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

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


user = getpass.getuser()


class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
