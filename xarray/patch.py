from __future__ import annotations

from pathlib import Path


# Dev Used in Dev Mode to test xr integration ( DO NOT MODIFY )
def fix_xarray(*, force: bool = False) -> tuple[Path, ...]:
    """Patch xarray source so IDEs resolve registered accessors for completion."""

    from importlib.util import find_spec

    # Check the marker first, before importing xarray or any integrations.
    xarray_spec = find_spec("xarray")
    if xarray_spec is None or xarray_spec.origin is None:
        raise RuntimeError("Cannot locate the xarray package.")

    marker = Path(xarray_spec.origin).resolve().parent / "_xgeo_patch.py"

    if not force and marker.exists():
        return ()

    # Everything below this point is only needed to create a new patch.
    import inspect
    import os
    import sys

    import xarray as xr

    if not __package__:
        raise RuntimeError("The accessor module must be imported as part of a package.")

    begin = "XGEO_IDE_TYPING BEGIN"
    end = "XGEO_IDE_TYPING END"
    guard = "_XGEO_TYPE_CHECKING"

    type_module = "xarray._xgeo_patch"

    # (class, class name, accessors registered by climtools on that class).
    # Each accessor is (attribute name, exported type name in _xgeo_patch).
    targets: tuple[tuple[type, str, tuple[tuple[str, str], ...]], ...] = (
        (xr.DataArray, "DataArray", (("xgeo", "GeoDataArray"),)),
        (xr.Dataset, "Dataset", (("xgeo", "GeoDataset"),)),
    )
    optional: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("metpy.xarray", ("metpy",)),
        ("cf_xarray", ("cf",)),
        ("pint_xarray", ("pint",)),
        ("rioxarray", ("rio",)),
    )
    integration_names = tuple(module.partition(".")[0] for module, _ in optional)

    sources: dict[str, Path] = {}
    for cls, class_name, _ in targets:
        source_file = inspect.getsourcefile(cls)
        if source_file is None:
            raise RuntimeError(f"Cannot locate xarray.{class_name} source.")
        sources[class_name] = Path(source_file).resolve()

    # Restore backups before attempting to modify if force is True, then unlink
    if force:
        for path in sources.values():
            backup = path.with_suffix(path.suffix + ".xgeo.bak")
            if backup.exists():
                path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
                backup.unlink()  # Remove the backup file after restoring

    def stat_of(path) -> list[int] | None:
        if not path:
            return None

        try:
            st = os.stat(path)
        except OSError:
            return None

        return [st.st_size, st.st_mtime_ns]

    def signature() -> dict:
        integrations: dict[str, dict | None] = {}

        for name in integration_names:
            try:
                spec = find_spec(name)
            except (ImportError, ValueError):
                spec = None

            if spec is None:
                integrations[name] = None
            else:
                integrations[name] = {
                    "origin": spec.origin,
                    "stat": stat_of(spec.origin),
                }

        return {
            "schema": 3,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "files": {label: stat_of(path) for label, path in sources.items()},
            "integrations": integrations,
        }

    # Heavy path begins here.
    import ast
    import importlib
    import random
    import time

    def discover() -> dict[type, list]:
        found: dict[type, list] = {cls: [] for cls, _, _ in targets}

        for module_name, names in optional:
            top = module_name.partition(".")[0]

            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                missing = exc.name or ""

                if missing in {top, module_name} or module_name.startswith(
                    missing + "."
                ):
                    continue

                raise RuntimeError(
                    f"{module_name!r} is installed but dependency {missing!r} is missing."
                ) from exc

            for name in names:
                registered = False

                for cls, class_name, _ in targets:
                    accessor = getattr(cls, name, None)
                    if accessor is None:
                        continue

                    registered = True

                    if not inspect.isclass(accessor):
                        raise RuntimeError(
                            f"{class_name}.{name} is not an accessor class."
                        )

                    if accessor.__qualname__ != accessor.__name__:
                        raise RuntimeError(
                            f"{class_name}.{name} is a nested class and cannot be imported."
                        )

                    found[cls].append(
                        (
                            name,
                            accessor.__module__,
                            accessor.__name__,
                        )
                    )

                if not registered:
                    raise RuntimeError(f"{module_name!r} did not register {name!r}.")

        return found

    def strip(source: str) -> str:
        out: list[str] = []
        skipping = False

        for line in source.splitlines(keepends=True):
            if begin in line:
                skipping = True
                continue

            if end in line:
                skipping = False
                continue

            if not skipping:
                out.append(line)

        return "".join(out)

    def region(
        tag: str,
        indent: str,
        body: list[str],
    ) -> str:
        return f"{indent}# {begin} {tag}\n" + "".join(body) + f"{indent}# {end} {tag}\n"

    def build(
        class_name: str,
        stubs: list,
    ) -> tuple[str, str]:
        alias = {attr: f"_xgeo_{class_name}_{attr}" for attr, _, _ in stubs}

        imports = [
            f"from typing import TYPE_CHECKING as {guard}\n",
            f"if {guard}:\n",
        ]

        for attr, mod, name in stubs:
            imports.append(f"    from {mod} import {name} as {alias[attr]}\n")

        props = [f"    if {guard}:\n"]

        for attr, _, _ in stubs:
            props.append("        @property\n")
            props.append(f"        def {attr}(self) -> {alias[attr]}: ...\n")

        return (
            region(f"imports {class_name}", "", imports),
            region(f"properties {class_name}", "    ", props),
        )

    discovered = discover()
    changed: list[Path] = []

    for cls, class_name, own_accessors in targets:
        path = sources[class_name]

        backup = path.with_suffix(path.suffix + ".xgeo.bak")
        raw = (backup if backup.exists() else path).read_text(encoding="utf-8")
        pristine = strip(raw)

        if not backup.exists():
            backup.write_text(pristine, encoding="utf-8")

        stubs: list = [
            *((attr, type_module, type_name) for attr, type_name in own_accessors),
            *discovered[cls],
        ]

        for attr, _, _ in stubs:
            if not attr.isidentifier():
                raise RuntimeError(f"Accessor name {attr!r} is not a valid identifier.")

        import_region, property_region = build(
            class_name,
            stubs,
        )

        tree = ast.parse(pristine, filename=str(path))

        node = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == class_name
            ),
            None,
        )

        if node is None:
            continue

        head = node.body[0]
        is_doc = (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
        )

        property_at = head.end_lineno if is_doc else head.lineno - 1
        import_at = node.lineno - 1

        lines = pristine.splitlines(keepends=True)
        lines.insert(property_at, property_region)
        lines.insert(import_at, import_region)
        patched = "".join(lines)

        compile(patched, str(path), "exec")

        if patched != path.read_text(encoding="utf-8"):
            path.write_text(
                patched,
                encoding="utf-8",
            )
            changed.append(path)

    meta: dict = signature()

    bridge = [
        "from __future__ import annotations",
        "from climtools.xarray.accessors import GeoDataArray as GeoDataArray",
        "from climtools.xarray.accessors import GeoDataset as GeoDataset",
        "__all__ = [GeoDataArray, GeoDataset]",
        "\n",
        f"meta: dict = {meta!r}\n",
    ]

    bridge = "\n".join(bridge)

    compile(bridge, str(marker), "exec")

    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(
        bridge,
        encoding="utf-8",
    )

    time.sleep(random.uniform(0, 1))
    tmp.replace(marker)

    return tuple(changed)
