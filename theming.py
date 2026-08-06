"""
A module for configuring the global matplotlib and seaborn theme for publication-quality plots.
"""

import os
import platform
import shutil
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns

from .plot_utils import interactive_backend

__all__ = ["apply", "reset", "spine_off"]


def format_crs_coordinates(ax: AxesType) -> str:

    def fmt_str(x: float, y: float) -> str:
        lon, lat = ccrs.PlateCarree().transform_point(
            x,
            y,
            src_crs=ax.projection,
        )

        if not np.isfinite(lon) or not np.isfinite(lat):
            return ""

        longitude = f"{abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"
        latitude = f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}"

        return f"lat={latitude} lon={longitude}"

    ax.format_coord = fmt_str


def interactive_backend(enable=True):
    """Enable or disable the interactive Matplotlib backend."""
    import matplotlib

    if enable:
        try:
            matplotlib.use("module://ipympl.backend_nbagg")
        except Exception:
            matplotlib.use("nbagg")
    else:
        matplotlib.use("module://matplotlib_inline.backend_inline")


def _install_latex():
    # Define paths
    _file_dir = Path(__file__).resolve().parent
    script = _file_dir / "data" / "script" / "latex.install"
    std_out = _file_dir / "data" / "script" / "latex.install.out"
    lock_file = _file_dir / "data" / "script" / "lock"

    lock_file.touch()

    if lock_file.exists():
        return

    if not script.exists():
        warnings.warn(
            "Skipping LaTeX installation: installation script not found. See https://www.tug.org/texlive/"
        )
        return False

    # check os type (only linux supported)
    os_name = platform.system()
    if os_name != "Linux":
        warnings.warn("Skipping LaTeX installation: only Linux is supported.")
        return False

    os.system(f"chmod +x {script}")
    cmd = f"nohup bash -c {script} > {std_out} 2>&1 &"
    os.system(cmd)

    return None


def reset():
    """Reset matplotlib and seaborn settings to their defaults."""
    sns.reset_defaults()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.switch_backend("agg")


def apply(
    *,
    interactive: bool = False,
    font_scale: float = 1.5,
    line_width: float = 1.5,
    font_size: int | None = None,
    column_width: Literal["single", "double"] | None = None,
    latex: bool = False,
    palette: Literal[
        "pastel", "deep", "muted", "bright", "dark", "colorblind"
    ] = "colorblind",
    context: Literal["paper", "notebook", "talk", "poster"] = "paper",
    style: str = "ticks",
    spine: bool = True,
    rc: dict | None = None,
):
    """
    Configure the global matplotlib and seaborn theme for publication-quality plots.

    This function sets figure dimensions, font scaling, line widths, and optionally enables latexrendering,
    ensuring that visualizations are formatted for either single- or double-column layouts typically required
    by scientific journals.

    Parameters
    ----------

    interactive : bool, optional
        If True, configures matplotlib for interactive use in Jupyter notebooks.

    font_scale : float, optional
        Scaling factor for fonts. This is passed to `seaborn.set_theme()`. Default is 1.5.

    column_width : "single" or "double". Default is "single".
        Target layout width:
        - "single" corresponds to 9 cm (≈ 3.54 inches),
        - "double" corresponds to 18 cm (≈ 7.09 inches).
        Used to determine appropriate font and layout scaling. Default is "single". Overidden when `figsize` is set.

    line_width : float, optional
        Default line width for plot elements. Applied via `matplotlib.rcParams`. Default is 1.5.

    latex: bool, optional
        If True, enables latextext rendering for all plot text via `matplotlib.rcParams["text.usetex"]`.
        Requires a working latexinstallation. Default is False.
    palette : str, optional
        Seaborn color palette to use. Default is "colorblind".
    context : str, optional
        Sets the context for the plot. Options are "paper", "notebook", "talk", or "poster". Default is "paper".
        This affects font sizes and other parameters to suit different presentation formats.
    style : str, optional
        Seaborn style to use. Options include "darkgrid", "whitegrid", "dark", "white", and "ticks". Default is "ticks".
    spine: bool, default False
        If True, top and left spines are removed.
    rc : dict, optional
        Additional rc parameters to pass to `seaborn.set_theme()`. These will override the defaults set by this function.

    """

    if interactive:
        interactive_backend()

    if column_width == "single":
        font_scale = 1
        line_width = 1
        fig_size = (3.5, 2.19)  # Single-column (9 cm × 5.56 cm)
    elif column_width == "double":
        fig_size = (7.09, 4.38)  # Double-column (18 cm × 11.12 cm)
    else:
        fig_size = None

    if latex and not shutil.which("latex") is not None:
        warnings.warn("Latex not found. Attempting to install LaTeX...")
        _install_latex()
        latex = False

    _rc = {
        "lines.linewidth": line_width,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.minor.visible": True,
        "xtick.minor.visible": True,
        "savefig.dpi": 1200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": latex,
        "svg.fonttype": "none",
    }

    if font_size:
        font_dict = {
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "axes.titlepad": font_size - 2,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "figure.titlesize": font_size + 2,
            "figure.labelsize": font_size,
        }

        _rc.update(font_dict)

    if fig_size:
        _rc["figure.figsize"] = fig_size

    if rc:
        _rc.update(rc)

    sns.set_theme(
        style=style,
        font="sans-serif",
        context=context,
        font_scale=font_scale,
        palette=palette,
        rc=_rc,
    )

    if not spine:
        spine_off()


def spine_off(ax=None):
    ax = plt.gca() if ax is None else ax
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
