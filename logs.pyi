from _typeshed import Incomplete
from datetime import datetime as datetime
from os import PathLike
from pathlib import Path
from typing import Any

USE_COLOR: Incomplete
RED: str
RESET: str

def log(*values: Any | None, file: PathLike | Path = None) -> None: ...
