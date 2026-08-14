"""Load the module stack recorded by the MPI-NetCDF build."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class ModuleLoadError(RuntimeError):
    """Raised when the recorded module stack cannot be loaded."""


def module(*args: str) -> None:
    """Apply an Lmod operation to the current Python process environment."""
    try:
        lmod_cmd = os.environ["LMOD_CMD"]
    except KeyError as exc:
        raise ModuleLoadError(
            "LMOD_CMD is not set; cannot load required modules"
        ) from exc

    result = subprocess.run(
        [lmod_cmd, "python", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    exec(result.stdout, {"os": os})  # noqa: S102


def _manifest_modules(path: Path) -> list[str]:
    """Read the ``modules`` sequence from the generated build manifest."""
    modules: list[str] = []
    in_modules = False

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ModuleLoadError(f"cannot read build manifest {path}: {exc}") from exc

    for line in lines:
        if not in_modules:
            if line.strip() == "modules:":
                in_modules = True
            continue

        if line.startswith("  - "):
            modules.append(line[4:].strip())
            continue
        if line.strip() == "[]":
            return []
        if line and not line.startswith(" "):
            break

    return modules


def ensure_required_modules(manifest: Path | None = None) -> None:
    """Load build-time MPI/NetCDF modules that are not already loaded."""
    path = manifest or Path(__file__).resolve().parent / "build" / "build.yml"
    if not path.is_file():
        return
    required = _manifest_modules(path)
    if not required:
        return

    loaded = {item for item in os.environ.get("LOADEDMODULES", "").split(":") if item}
    missing = [name for name in required if name not in loaded]
    if missing:
        module("load", *missing)
