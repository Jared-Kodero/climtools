"""Structural tests for the package layout.

The reorganisation is only worth anything if it stays true. These tests fail
if a module reappears at the package root, if a subpackage stops importing, if
a bundled data file stops resolving, or if the public names that existing code
depends on are dropped. They need no MPI and no compiled extension.

Run with::

    python -m climtools.tests.test_layout
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import climtools
from climtools.core.paths import PACKAGE_ROOT, SCRIPTS_DIR

#: The only module permitted at the package root. Every unit of
#: functionality owns a directory, so a new backend or plot type extends one
#: directory instead of growing the root.
ROOT_MODULES = {"__init__.py"}

#: Every functional area, and the directory it owns.
SUBPACKAGES = (
    "accessors",
    "cdo",
    "core",
    "examples",
    "lib_mpi",
    "netcdf",
    "preprocess",
    "stats",
    "tests",
    "viz",
    "xgeo",
)

#: Names existing code imports from the package root. Dropping any of these
#: is a breaking change, whatever the internal layout does.
PUBLIC_ROOT_NAMES = (
    "DaskProgressBar",
    "SerialProgressBar",
    "calc",
    "cdo",
    "cmaps",
    "lib_mpi",
    "mpi",
    "n_cpus",
    "netcdf",
    "operator",
    "plot",
    "redirect_streams",
    "xgeo",
    "MPI_RANK",
    "MPI_SIZE",
)

#: Names existing analysis scripts import from :mod:`climtools.xgeo`.
PUBLIC_XGEO_NAMES = (
    "append_to_netcdf",
    "remap",
    "to_netcdf",
)


def test_root_holds_only_the_facade() -> None:
    """No analysis module may sit at the package root."""
    found = {p.name for p in PACKAGE_ROOT.glob("*.py")}
    extra = found - ROOT_MODULES
    assert not extra, f"unexpected modules at the package root: {sorted(extra)}"


def test_every_subpackage_imports() -> None:
    """Each functional directory is an importable package."""
    for name in SUBPACKAGES:
        directory = PACKAGE_ROOT / name
        assert directory.is_dir(), f"missing subpackage directory: {name}"
        assert (directory / "__init__.py").is_file(), f"{name} has no __init__.py"
        importlib.import_module(f"climtools.{name}")


def test_public_root_names_survive() -> None:
    """The names existing code imports from the root still resolve."""
    missing = [n for n in PUBLIC_ROOT_NAMES if not hasattr(climtools, n)]
    assert not missing, f"public names lost from climtools: {missing}"


def test_public_xgeo_names_survive() -> None:
    """``from climtools.xgeo import ...`` keeps working."""
    xgeo = importlib.import_module("climtools.xgeo")
    missing = [n for n in PUBLIC_XGEO_NAMES if not hasattr(xgeo, n)]
    assert not missing, f"public names lost from climtools.xgeo: {missing}"


def test_data_lives_with_its_owner() -> None:
    """Each resource sits inside the subpackage that reads it.

    Data that travels with its package makes a move safe: the path derivation
    stays correct because the files move too. Data left behind at the root
    fails silently instead, since the import still succeeds and only the first
    read breaks.
    """
    assert (PACKAGE_ROOT / "viz" / "data" / "cmaps").is_dir(), "viz lost its colormaps"
    assert list((PACKAGE_ROOT / "viz" / "data" / "cmaps").glob("*.txt")), "no tables"
    assert (PACKAGE_ROOT / "viz" / "scripts" / "latex.install").is_file(), (
        "no installer"
    )
    assert (PACKAGE_ROOT / "xgeo" / "data" / "mask" / "era5_0.25_mask").is_file(), (
        "xgeo lost its default mask"
    )
    assert SCRIPTS_DIR.is_dir(), f"missing global scripts directory: {SCRIPTS_DIR}"


def test_no_orphan_data_at_the_root() -> None:
    """Only genuinely global resources may sit at the package root."""
    assert not (PACKAGE_ROOT / "data").exists(), (
        "root data/ directory is back; move each resource into the subpackage "
        "that reads it"
    )


def test_bundled_colormap_loads() -> None:
    """A colormap read from the bundled tables builds correctly."""
    cmap = climtools.cmaps.temp_div()
    assert cmap.N > 0


def test_accessor_is_registered() -> None:
    """Importing the package registers the ``.xgeo`` accessor."""
    import numpy as np
    import xarray as xr

    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )
    assert hasattr(da, "xgeo")
    assert hasattr(da.xgeo, "plot")


def test_alias_matches_its_subpackage() -> None:
    """A root alias must not shadow a subpackage of the same name.

    ``climtools.cdo`` is both an alias and a directory. If the alias were
    bound to the module inside it, the attribute and ``sys.modules`` entry
    would disagree, so ``climtools.cdo.x`` and ``from climtools.cdo import x``
    could resolve differently.
    """
    for name in ("cdo", "xgeo"):
        importlib.import_module(f"climtools.{name}")
        assert climtools_attr(name) is sys.modules[f"climtools.{name}"], (
            f"climtools.{name} attribute and module entry disagree"
        )


def climtools_attr(name: str) -> object:
    """Return a package attribute by name."""
    return getattr(climtools, name)


def main() -> int:
    """Run the layout tests and report."""
    tests = [
        value for name, value in sorted(globals().items()) if name.startswith("test_")
    ]
    failures = 0
    print(f"climtools layout tests: {PACKAGE_ROOT}\n")
    for test in tests:
        try:
            test()
        except Exception as exc:
            failures += 1
            print(f"  FAIL {test.__name__}: {exc}")
        else:
            print(f"  pass {test.__name__}")
    print(f"\nlayout: {len(tests) - failures} passed, {failures} failed")
    print("RESULT:", "SUCCESS" if failures == 0 else "FAILURE")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
