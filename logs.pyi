from os import PathLike
from pathlib import Path
from typing import Any

USE_COLOR: bool
RED: str
RESET: str

def log(*values: Any | None, file: PathLike | Path = None) -> None: ...
