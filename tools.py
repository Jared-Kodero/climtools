from __future__ import annotations

import getpass
import inspect
import os
import socket
import sys
from pathlib import Path
from typing import Callable

from IPython.display import HTML, display

host = socket.gethostname()
user = getpass.getuser()
home = Path.home()
tmp = Path("/tmp")
n_cpus = int(len(os.sched_getaffinity(0)))
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
