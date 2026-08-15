"""Load the module stack recorded by the MPI-NetCDF build."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

MANIFEST_NAME = "stack_env.yaml"


def load_env_stack() -> dict[str, object]:
    """Return the recorded build manifest, or an empty mapping if unavailable."""
    path = Path(__file__).resolve().parent / MANIFEST_NAME
    if not path.is_file():
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    try:
        with path.open(encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError):
        return {}

    return dict(manifest) if isinstance(manifest, dict) else {}


class ModuleLoadError(RuntimeError):
    """Raised when the recorded module stack cannot be loaded."""


def _load_hint(modules: Sequence[str]) -> str:
    """Return the command that loads ``modules`` by hand."""
    return f"Load them first with:\n    module load {' '.join(modules)}"


def module(*args: str) -> None:
    """Apply an Lmod operation to the current Python process environment."""
    try:
        lmod_cmd = os.environ["LMOD_CMD"]
    except KeyError as exc:
        raise ModuleLoadError(
            "LMOD_CMD is not set; cannot load required modules"
        ) from exc

    try:
        result = subprocess.run(
            [lmod_cmd, "python", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ModuleLoadError(f"cannot run {lmod_cmd}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit status {exc.returncode}"
        raise ModuleLoadError(f"`module {' '.join(args)}` failed: {detail}") from exc

    exec(result.stdout, {"os": os})  # noqa: S102


def _loaded_modules() -> set[str]:
    """Return the module names currently recorded in ``LOADEDMODULES``."""
    return {item for item in os.environ.get("LOADEDMODULES", "").split(":") if item}


def check_env_stack() -> None:
    """Load build-time MPI/NetCDF modules that are not already loaded."""

    env_stack = load_env_stack()
    modules = env_stack.get("modules", [])

    if not modules:
        return

    missing = [name for name in modules if name not in _loaded_modules()]
    if not missing:
        return

    try:
        module("load", *missing)
    except ModuleLoadError as exc:
        raise ModuleLoadError(f"{exc}\n{_load_hint(missing)}") from exc

    # Lmod does not always signal a failed load through its exit status. A
    # module that is still absent here would otherwise surface much later, as
    # an unresolved symbol when libmpi_netcdf.so is opened.
    unloaded = [name for name in missing if name not in _loaded_modules()]
    if unloaded:
        names = ", ".join(unloaded)
        raise ModuleLoadError(
            f"not loaded after `module load`: {names}\n{_load_hint(missing)}"
        )
