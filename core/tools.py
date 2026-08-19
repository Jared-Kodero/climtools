from __future__ import annotations

import getpass
import inspect
import os
import socket
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

script_dir = Path(__file__).resolve().parent
host = socket.gethostname()
user = getpass.getuser()
home = Path.home()

n_cpus = len(os.sched_getaffinity(0))
ipykernel = "ipykernel" in sys.modules
isatty = sys.stdout.isatty() or ipykernel


tmp = Path(f"/tmp/{user}/xgeo/{uuid.uuid4().hex}")
tmp.mkdir(parents=True, exist_ok=True)


current_dask_cluster = None
current_dask_client = None
widget_css_applied = False
fix_widget = True


class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


import fcntl
import os
import time


class LockFile:
    """A context manager class for file locking with sleep and timeout."""

    def __init__(
        self,
        filepath: Path | None = None,
        timeout: float | None = None,
        delay: float = 0.1,
    ):
        self.filepath = filepath or Path(".lock")
        self.timeout = timeout
        self.delay = delay
        self.fd = None

    def __enter__(self):
        start_time = time.time()
        # Open the file descriptor
        self.fd = os.open(self.filepath, os.O_RDWR | os.O_CREAT)

        while True:
            try:
                # Try to acquire an exclusive, non-blocking lock
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self  # Lock acquired, enter the 'with' block

            except (OSError, BlockingIOError):
                # Lock is held by another process; check timeout and sleep
                if (
                    self.timeout is not None
                    and (time.time() - start_time) >= self.timeout
                ):
                    os.close(self.fd)
                    raise TimeoutError(
                        f"Could not acquire lock on {self.filepath} within {self.timeout}s"
                    )

                time.sleep(self.delay)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            # Release the lock and close the file descriptor safely
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


@contextmanager
def redirect_streams(
    stdout: Path | None = None,
    stderr: Path | None = None,
):
    """
    Temporarily redirect standard output and standard error to files.

    Each specified file is opened in append mode. If ``stdout`` and
    ``stderr`` refer to the same path, both streams share a single file
    handle. Any stream whose path is ``None`` remains unchanged.

    The original streams are restored and all opened files are closed when
    the context exits, including when an exception is raised.

    Parameters
    ----------
    stdout : Path or None, optional
        File path to which ``sys.stdout`` is redirected.
    stderr : Path or None, optional
        File path to which ``sys.stderr`` is redirected.

    Yields
    ------
    tuple[TextIO | None, TextIO | None]
        The opened output and error file handles, respectively.
    """

    org_stdout = sys.stdout
    org_stderr = sys.stderr

    out_file = None
    err_file = None

    try:
        if stdout == stderr and stdout is not None:
            out_file = open(stdout, "a")
            err_file = out_file

        else:
            if stdout:
                out_file = open(stdout, "a")

            if stderr:
                err_file = open(stderr, "a")

        if out_file:
            sys.stdout = out_file

        if err_file:
            sys.stderr = err_file

        yield out_file, err_file

    finally:
        sys.stdout = org_stdout
        sys.stderr = org_stderr

        if out_file:
            out_file.close()

        if err_file and err_file is not out_file:
            err_file.close()


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
