import getpass
import socket
from pathlib import Path

HOST = socket.gethostname()
USER = getpass.getuser()
HOME = Path.home()

# PERSONAL DIRECTORIES
# ---------------------
TMP = (HOME / "tmp").resolve()

SCRATCH_DIR = HOME / "scratch"  #: Path: Scratch directory
DEEPS_SHARE_DIR = HOME / "deeps_data"
DATA_DIR = HOME / f"data/{USER}/data"
SHARED_DIR = HOME / "data" / "shared"  #: Path: Shared data directory
WORK_DIR = HOME / "research" / "projects"
FIG_DIR = Path.cwd() / "figures"
SCRIPTS_DIR = HOME / "research/scripts"  #: Path: Scripts directory
ERA5_DIR = DATA_DIR / "era5"  #: Path: ERA5 directory
ERA5_RAW_DIR = SHARED_DIR / "era5"  #: Path: Shared ERA5 raw data directory

if not TMP.exists():
    TMP = Path("/tmp")


def CWD():
    """
    Get the current working directory.
    """
    return Path.cwd().resolve()
