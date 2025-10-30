import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import seaborn as sns

if "ipykernel" in sys.modules:
    import matplotlib_inline as plt_inline

    plt_inline.backend_inline.set_matplotlib_formats("retina")


def check_latex() -> int:
    latex = subprocess.run(["which", "latex"], capture_output=True, text=True)

    res = int(latex.returncode)

    if not res:
        return True
    return False


def install_latex():
    # Define paths
    _file_dir = Path(__file__).resolve().parent
    script = _file_dir / "script" / "latex.install"
    std_out = _file_dir / "script" / "latex.install.out"
    lock_file = _file_dir / "script" / "lock"

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

    return


class IPCCTheme:
    def __init__(self):
        self.latex: bool = check_latex()
        self.install_latex: Callable = install_latex

        # --- Context management methods ---

    def __enter__(self):
        # Return self so you can use it as "with IPCCTheme() as theme:"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Automatically reset when leaving context
        self.reset()

    def reset(self):
        """Reset matplotlib and seaborn settings to their defaults."""
        sns.reset_defaults()
        plt.rcParams.update(plt.rcParamsDefault)

    def apply(
        self,
        *,
        font_scale: float = 1.5,
        line_width: float = 1.5,
        column_width: Literal["single", "double"] = None,
        latex: bool = False,
        context: Literal["paper", "notebook", "talk", "poster"] = "paper",
        style: str = "ticks",
        rc: dict = None,
    ):
        """
        Configure the global matplotlib and seaborn theme for publication-quality plots.

        This function sets figure dimensions, font scaling, line widths, and optionally enables latexrendering,
        ensuring that visualizations are formatted for either single- or double-column layouts typically required
        by scientific journals.

        Parameters
        ----------

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
        context : str, optional
            Sets the context for the plot. Options are "paper", "notebook", "talk", or "poster". Default is "paper".
            This affects font sizes and other parameters to suit different presentation formats.
        style : str, optional
            Seaborn style to use. Options include "darkgrid", "whitegrid", "dark", "white", and "ticks". Default is "ticks".
        rc : dict, optional
            Additional rc parameters to pass to `seaborn.set_theme()`. These will override the defaults set by this function.

        """

        if column_width == "single":
            font_scale = 1
            line_width = 1
            fig_size = (3.5, 2.19)  # Single-column (9 cm × 5.56 cm)
        elif column_width == "double":
            fig_size = (7.09, 4.38)  # Double-column (18 cm × 11.12 cm)
        else:
            fig_size = None

        if latex and not self.latex:
            warnings.warn("Latex not found. Attempting to install LaTeX...")
            self.install_latex()
            latex = False

        default_rc = {
            "lines.linewidth": line_width,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.minor.visible": True,
            "xtick.minor.visible": True,
            "savefig.dpi": 1200,
            "text.usetex": latex,
        }

        if fig_size:
            default_rc["figure.figsize"] = fig_size
        if latex:
            default_rc["font.size"] = 12

        if not rc:
            rc = {}

        rc.update(default_rc)

        sns.set_theme(
            style=style,
            font="sans-serif",
            context=context,
            font_scale=font_scale,
            palette="colorblind",
            rc=rc,
        )

        return self

    def spine_off(self, ax=None):
        ax = plt.gca() if ax is None else ax
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


theme: IPCCTheme = IPCCTheme()
