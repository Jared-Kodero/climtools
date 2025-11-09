import hashlib
import json
import os
import pprint
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

import cmocean
import matplotlib
import matplotlib.pyplot as plt

# ANSI color codes
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


_file_dir = Path(__file__).resolve().parent
_src_dir = _file_dir / "data" / "cmaps"
_meta_file = _file_dir / ".cmap_meta.json"
_cmap_file = _file_dir / "cmaps_inventory.py"

ipcc_cmap_list = [f.stem for f in _src_dir.glob("*.txt")]
plt_cmap_list = plt.colormaps()
cmocean_cmap_list = list(cmocean.cm.cmapnames)
all_cmaps = ipcc_cmap_list + plt_cmap_list + cmocean_cmap_list


def _print_result(v, file=None, pretty=True):
    if pretty:
        if file is None:
            pprint.pprint(v, sort_dicts=False, compact=True)
        else:
            with open(file, "a", encoding="utf-8") as f:
                pprint.pprint(v, stream=f, sort_dicts=False, compact=True)
    else:
        if file is None:
            # direct write to stdout, unbuffered
            os.write(sys.stdout.fileno(), f"{v}\n".encode("utf-8"))
        else:
            with open(file, "a", encoding="utf-8") as f:
                f.write(f"{v}\n")
                f.flush()  # ensure immediate write


def logmsg(
    *values: Any | None,
    file: Path | Path = None,
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

    file : Path or Path, optional
        File path or file-like object to write the log messages to. If None, logs to standard output.
    -------
    None
    """

    exc_info = sys.exc_info()

    # unpack values adding  a a space between str values
    values = [v for v in values]

    if all(isinstance(v, str) for v in values):
        values = " ".join(values)
        return _print_result(values, file, pretty=False)

    else:
        for v in values:
            if not isinstance(v, str):
                _print_result(v, file, pretty=True)
            else:
                _print_result(v, file, pretty=False)

    _print_result("\n", file, pretty=False)

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
        and str(x.filename).endswith(".py")
    ]

    ft = ft_user
    new_ft = []
    for frame in ft:
        file_path = Path(frame.filename).resolve()
        lineno = f"{frame.lineno:>5}"
        frame_line = frame.line.strip() if frame.line else ""
        pointer = " " * (len(str(lineno)) + 3) + "^" * len(frame_line)

        if sys.stdout.isatty():
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


def _compute_hash():
    """Compute a hash from versions and src file metadata (names + modification times)."""
    src_files = sorted(_src_dir.glob("*.txt"))
    src_state = {f.name: os.path.getmtime(f) for f in src_files}

    data = {
        "matplotlib_version": matplotlib.__version__,
        "cmocean_version": cmocean.__version__,
        "src_state": src_state,
    }

    hash_str = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return hash_str, data


def _load_meta():
    if _meta_file.exists():
        try:
            with open(_meta_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _write_meta(data, hash_str):
    data["_hash"] = hash_str
    with open(_meta_file, "w") as f:
        json.dump(data, f, indent=2)


def _cmap_file_contents():

    imports = """
    from dataclasses import dataclass
    from matplotlib.colors import Colormap

    from .cmap_funcs import *
    \n
    """
    imports = textwrap.dedent(imports)

    body = """
    @dataclass
    class ColorMaps:

        @staticmethod
        def new(colors: list[str], N: int = 25, *, discrete: bool = True):
            return blend(colors, N=N, discrete=discrete)
        \n
            """

    for name in all_cmaps:
        if name.endswith("_r") or "cmo" in name.lower():
            continue
        body = f""" {body}
        
        @staticmethod
        def {name.lower()}(
            N: int = 25,
            reverse: bool = False,
            split: tuple[float, float] = (0, 1),
            add_colors: dict[int, str | list[str]] = None,
            discrete: bool = True,
        ) -> Colormap:
            return get_func("{name}", N, reverse, split, add_colors, discrete)
            \n
        """
    body = textwrap.dedent(body)

    init = """
    cmaps: ColorMaps = ColorMaps()
    cm : ColorMaps = cmaps
    """
    init = textwrap.dedent(init)
    return imports, body, init


def gen_cmap_file():
    """Generate plot_cmaps.py only if versions or src files changed (added, removed, or modified)."""
    new_hash, meta_data = _compute_hash()
    old_meta = _load_meta()

    if _cmap_file.exists() and old_meta.get("_hash") == new_hash:
        return  # Up to date
    imports, body, init = _cmap_file_contents()
    with open(_cmap_file, "w") as f:
        f.write(imports)
        f.write(body)
        f.write(init)

    _write_meta(meta_data, new_hash)
