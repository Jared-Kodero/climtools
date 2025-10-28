from _typeshed import Incomplete
from os import PathLike
from pathlib import Path
from typing import Any

use_color: Incomplete
RED: str
BOLD: str
RESET: str

def log(*values: Any | None, file: PathLike | Path = None) -> None: ...
