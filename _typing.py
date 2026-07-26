from __future__ import annotations

from pathlib import Path

import xarray as xr


def fix_xarray1() -> tuple[Path, ...]:  # xarray typing hack. Works !!!!
    """Patch xarray source files so IDEs type ``DataArray.xgeo`` and ``Dataset.xgeo``."""
    import ast
    import inspect
    import shutil

    if not __package__:
        raise RuntimeError("The accessor module must be imported as part of a package.")

    type_module = f"{__package__}.xgeo_types"
    targets = (
        (xr.DataArray, "DataArray", "GeoDataArray"),
        (xr.Dataset, "Dataset", "GeoDataset"),
    )
    changed: list[Path] = []

    for runtime_class, class_name, accessor_type in targets:
        source_file = inspect.getsourcefile(runtime_class)
        if source_file is None:
            raise RuntimeError(f"Cannot locate xarray.{class_name} source.")

        path = Path(source_file).resolve()
        source = path.read_text(encoding="utf-8")
        marker = f"# xgeo IDE typing: {class_name}"
        if marker in source:
            continue

        tree = ast.parse(source, filename=str(path))
        node = next(
            (
                item
                for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == class_name
            ),
            None,
        )
        if node is None:
            raise RuntimeError(f"Cannot find {class_name} in {path}.")

        alias = f"_XGeo{accessor_type}"
        import_block = (
            f"{marker}\n"
            "from typing import TYPE_CHECKING as _XGEO_TYPE_CHECKING\n"
            "if _XGEO_TYPE_CHECKING:\n"
            f"    from {type_module} import {accessor_type} as {alias}\n\n"
        )

        first = node.body[0]
        after_docstring = (
            first.end_lineno
            if isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            else first.lineno - 1
        )
        property_block = (
            "\n"
            "    if _XGEO_TYPE_CHECKING:\n"
            "        @property\n"
            f"        def xgeo(self) -> {alias}:\n"
            "            ...\n"
        )

        lines = source.splitlines(keepends=True)
        for index, block in sorted(
            ((node.lineno - 1, import_block), (after_docstring, property_block)),
            reverse=True,
        ):
            lines.insert(index, block)

        patched = "".join(lines)
        compile(patched, str(path), "exec")

        backup = path.with_suffix(path.suffix + ".xgeo.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

        path.write_text(patched, encoding="utf-8")
        changed.append(path)

    return tuple(changed)


def fix_xarray() -> tuple[Path, ...]:  # xarray typing hack. Works !!!!
    """Patch xarray source files so IDEs type registered accessors."""
    import ast
    import importlib
    import inspect
    import shutil

    if not __package__:
        raise RuntimeError("The accessor module must be imported as part of a package.")

    type_module = f"{__package__}.xgeo_types"
    targets = (
        (xr.DataArray, "DataArray", "GeoDataArray"),
        (xr.Dataset, "Dataset", "GeoDataset"),
    )

    optional_registrations = (
        ("metpy.xarray", ("metpy",)),
        ("hvplot.xarray", ("hvplot",)),
        ("cf_xarray", ("cf",)),
        ("pint_xarray", ("pint",)),
        ("rioxarray", ("rio",)),
    )

    loaded_registrations: list[tuple[str, tuple[str, ...]]] = []

    for registration_module, accessor_names in optional_registrations:
        try:
            importlib.import_module(registration_module)
        except ModuleNotFoundError:
            continue
        except Exception as exc:
            raise RuntimeError(
                "Failed to import optional xarray integration "
                + f"{registration_module!r}."
            ) from exc

        loaded_registrations.append((registration_module, accessor_names))

    registered_types: dict[
        type,
        list[tuple[str, str, str]],
    ] = {runtime_class: [] for runtime_class, _, _ in targets}

    for registration_module, accessor_names in loaded_registrations:
        package_name = registration_module.partition(".")[0]

        for accessor_name in accessor_names:
            found = False

            for runtime_class, class_name, _ in targets:
                accessor_class = getattr(
                    runtime_class,
                    accessor_name,
                    None,
                )
                if accessor_class is None:
                    continue

                found = True

                if not inspect.isclass(accessor_class):
                    raise RuntimeError(
                        f"{class_name}.{accessor_name} is registered, "
                        + f"but {accessor_class!r} is not an accessor class."
                    )

                accessor_module = accessor_class.__module__
                accessor_type = accessor_class.__name__

                if accessor_module.partition(".")[0] != package_name:
                    raise RuntimeError(
                        f"{class_name}.{accessor_name} resolves to "
                        + f"{accessor_module}.{accessor_type}, not to an "
                        + f"accessor owned by {package_name!r}."
                    )

                if accessor_class.__qualname__ != accessor_type:
                    raise RuntimeError(
                        f"{class_name}.{accessor_name} uses nested accessor "
                        + f"class {accessor_class.__qualname__!r}, which "
                        + "cannot be imported by the generated "
                        + "TYPE_CHECKING block."
                    )

                owner_module = importlib.import_module(accessor_module)
                imported_class = getattr(
                    owner_module,
                    accessor_type,
                    None,
                )

                if imported_class is not accessor_class:
                    raise RuntimeError(
                        "Cannot re-import the registered accessor class "
                        + f"{accessor_module}.{accessor_type}."
                    )

                registered_types[runtime_class].append(
                    (
                        accessor_name,
                        accessor_module,
                        accessor_type,
                    )
                )

            if not found:
                raise RuntimeError(
                    f"{registration_module!r} imported successfully but "
                    + f"did not register the {accessor_name!r} accessor "
                    + "on DataArray or Dataset."
                )

    changed: list[Path] = []

    for runtime_class, class_name, accessor_type in targets:
        source_file = inspect.getsourcefile(runtime_class)
        if source_file is None:
            raise RuntimeError(f"Cannot locate xarray.{class_name} source.")

        path = Path(source_file).resolve()
        source = path.read_text(encoding="utf-8")

        tree = ast.parse(source, filename=str(path))
        node = next(
            (
                item
                for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == class_name
            ),
            None,
        )
        if node is None:
            raise RuntimeError(f"Cannot find {class_name} in {path}.")

        import_blocks: list[str] = []
        property_blocks: list[str] = []

        # Preserve the existing legacy patch and add it only if absent.
        marker = f"# xgeo IDE typing: {class_name}"

        if marker not in source:
            alias = f"_XGeo{accessor_type}"

            import_blocks.append(
                "\n".join(
                    (
                        marker,
                        "from typing import " + "TYPE_CHECKING as _XGEO_TYPE_CHECKING",
                        "if _XGEO_TYPE_CHECKING:",
                        f"    from {type_module} import "
                        + f"{accessor_type} as {alias}",
                        "",
                        "",
                    )
                )
            )

            property_blocks.append(
                "\n".join(
                    (
                        "",
                        "    if _XGEO_TYPE_CHECKING:",
                        "        @property",
                        f"        def xgeo(self) -> {alias}:",
                        "            ...",
                        "",
                    )
                )
            )

        for (
            accessor_name,
            accessor_module,
            registered_type,
        ) in registered_types[runtime_class]:
            if not accessor_name.isidentifier():
                raise RuntimeError(
                    f"Accessor name {accessor_name!r} is not a valid "
                    + "Python identifier."
                )

            import_marker = (
                "# xgeo IDE typing import: " + f"{class_name}.{accessor_name}"
            )
            property_marker = (
                "# xgeo IDE typing property: " + f"{class_name}.{accessor_name}"
            )

            has_import = import_marker in source
            has_property = property_marker in source

            if has_import and has_property:
                continue

            if has_import != has_property:
                raise RuntimeError(
                    "Partial IDE typing patch found for "
                    + f"{class_name}.{accessor_name} in {path}."
                )

            alias = f"_XGeo{class_name}_{accessor_name}"

            import_blocks.append(
                "\n".join(
                    (
                        import_marker,
                        "if _XGEO_TYPE_CHECKING:",
                        f"    from {accessor_module} import "
                        + f"{registered_type} as {alias}",
                        "",
                        "",
                    )
                )
            )

            property_blocks.append(
                "\n".join(
                    (
                        "",
                        f"    {property_marker}",
                        "    if _XGEO_TYPE_CHECKING:",
                        "        @property",
                        f"        def {accessor_name}(self) -> {alias}:",
                        "            ...",
                        "",
                    )
                )
            )

        if not import_blocks and not property_blocks:
            continue

        first = node.body[0]
        after_docstring = (
            first.end_lineno
            if isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            else first.lineno - 1
        )

        lines = source.splitlines(keepends=True)
        insertions = (
            (
                node.lineno - 1,
                "".join(import_blocks),
            ),
            (
                after_docstring,
                "".join(property_blocks),
            ),
        )

        for index, block in sorted(
            insertions,
            reverse=True,
        ):
            if block:
                lines.insert(index, block)

        patched = "".join(lines)
        compile(patched, str(path), "exec")

        backup = path.with_suffix(path.suffix + ".xgeo.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

        path.write_text(
            patched,
            encoding="utf-8",
        )
        changed.append(path)

    return tuple(changed)
