from __future__ import annotations

import json
import shutil
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal

import dask
import numpy as np
import pandas as pd

import xarray as xr


def to_xnpy(
    obj: np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    mode: Literal["w", "w-"] = "w-",
    scheduler: Literal["threads", "synchronous"] = "threads",
    num_workers: int | None = None,
) -> None:
    """Write a NumPy, pandas, or xarray object to a memory-mappable XNpy store.

    In-memory arrays are written with :func:`numpy.save`. Large lazily indexed
    arrays are streamed in bounded, chunk-aligned slabs. All array writes (variables,
    coordinates, columns, masks) are built as :func:`dask.delayed` tasks and
    Dask-backed variables as :func:`dask.array.store` tasks. They execute in a
    single :func:`dask.compute` call, so columns, variables, and chunks run
    concurrently under the selected scheduler.

    Metadata (``attrs``, labels, names) is encoded with type tags so that tuples,
    dicts with non-string keys, and nested pandas or xarray objects (stored as
    sub-stores under ``attrs/``) round-trip. Object and ``boolean`` bool columns
    are stored as int8 codes (1 true, 0 false, -1 missing) with their dtype.
    Text, nullable ``Int64``/``Float64``, and object numeric columns keep a
    missing-value mask; categorical columns keep their categories. Every column
    and index level stores its pandas dtype, which the reader restores.
    MultiIndex rows and columns are supported.

    Parameters
    ----------
    obj : numpy.ndarray or pandas.Series or pandas.DataFrame or xarray.DataArray or xarray.Dataset
        Object to store. Labels and attrs must be built from str, int, float,
        bool, None, tuples, lists, dicts, NumPy scalars and arrays, or nested
        pandas and xarray objects.
    path : str or pathlib.Path
        Destination store directory.
    mode : {"w", "w-"}, default="w-"
        Write mode. ``"w"`` replaces an existing store. ``"w-"`` requires that
        the destination not exist.
    scheduler : {"threads", "synchronous"}, default="threads"
        Dask scheduler used to execute all write tasks. ``"threads"`` writes
        concurrently and ``"synchronous"`` writes serially in the calling
        thread.
    num_workers : int, optional
        Number of workers passed to :func:`dask.compute`. If omitted, the Dask
        default is used.

    Returns
    -------
    None
        The store is written to ``path``.
    """
    if mode not in {"w", "w-"}:
        raise ValueError(f"Unsupported XNpy mode: {mode!r}.")

    if not isinstance(
        obj, (np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)
    ):
        raise TypeError(f"Unsupported XNpy object type: {type(obj)!r}.")
    # Process workers would write Dask chunks into pickled copies of the memmap
    # target, silently leaving zeros in the store.
    if scheduler not in {"threads", "synchronous"}:
        raise ValueError(f"Unsupported XNpy scheduler: {scheduler!r}.")

    root = Path(path)
    if root.exists():
        if mode == "w-":
            raise FileExistsError(root)
        shutil.rmtree(root)

    jobs: list[tuple[Any, str]] = []
    nested: list[Any] = []

    def add(source: Any, folder: str, name: Any) -> str:
        file = f"{folder}/{name}.npy"
        jobs.append((source, file))
        return file

    def add_variable(source: xr.DataArray, folder: str, name: Any) -> dict[str, Any]:
        return {
            "dims": list(source.dims),
            "attrs": dict(source.attrs),
            "file": add(source, folder, name),
        }

    def add_column(values: Any, folder: str, name: Any) -> dict[str, Any]:
        # Every column records its pandas dtype so the reader restores it exactly.
        series = pd.Series(values)
        dtype = str(series.dtype)
        if isinstance(series.dtype, pd.CategoricalDtype):
            return {
                "encoding": "categorical",
                "dtype": dtype,
                "file": add(series.cat.codes.to_numpy(), folder, name),
                "categories": series.cat.categories.tolist(),
                "ordered": bool(series.cat.ordered),
            }
        inferred = pd.api.types.infer_dtype(series) if series.dtype == object else None
        if isinstance(series.dtype, pd.BooleanDtype) or inferred == "boolean":
            # Bools with missing values are stored as int8 codes 1/0 (-1 missing)
            # with the column dtype, so "False" text can never read back as True.
            mask = series.isna().to_numpy()
            values = series.fillna(False).to_numpy(dtype=bool)
            return {
                "encoding": "bool",
                "dtype": dtype,
                "file": add(np.where(mask, -1, values).astype(np.int8), folder, name),
            }
        if isinstance(series.array, pd.arrays.IntegerArray | pd.arrays.FloatingArray):
            # Nullable int and float keep their dtype instead of float or text.
            numpy_dtype = series.dtype.numpy_dtype
            values = series.to_numpy(numpy_dtype, na_value=numpy_dtype.type(0))
            return {
                "encoding": "masked",
                "dtype": dtype,
                "file": add(values, folder, name),
                "mask": add(series.isna().to_numpy(), folder, f"{name}.mask"),
            }
        if inferred in {"integer", "floating", "mixed-integer-float"}:
            # Object numeric columns with missing values: store typed values and a mask.
            mask = series.isna().to_numpy()
            valid = np.asarray(series[~mask].tolist())
            if valid.dtype != object:
                values = np.zeros(len(series), dtype=valid.dtype)
                values[~mask] = valid
                return {
                    "encoding": "masked",
                    "dtype": dtype,
                    "file": add(values, folder, name),
                    "mask": add(mask, folder, f"{name}.mask"),
                }
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            return {
                "encoding": "string",
                "dtype": dtype,
                "file": add(series.fillna("").to_numpy().astype(str), folder, name),
                "mask": add(series.isna().to_numpy(), folder, f"{name}.mask"),
            }
        return {
            "encoding": "array",
            "dtype": dtype,
            "file": add(series.to_numpy(), folder, name),
        }

    def add_index(index: pd.Index) -> list[dict[str, Any]]:
        return [
            add_column(index.get_level_values(i), "coords", f"index.{i}")
            for i in range(index.nlevels)
        ]

    if isinstance(obj, xr.Dataset):
        metadata: dict[str, Any] = {
            "kind": "dataset",
            "attrs": dict(obj.attrs),
            "variables": {
                name: add_variable(var, "variables", name)
                for name, var in obj.data_vars.items()
            },
            "coords": {
                name: add_variable(coord, "coords", name)
                for name, coord in obj.coords.items()
            },
        }
    elif isinstance(obj, xr.DataArray):
        metadata = {
            "kind": "dataarray",
            "name": obj.name,
            "variable": add_variable(
                obj,
                "variables",
                "data" if obj.name is None else obj.name,
            ),
            "coords": {
                name: add_variable(coord, "coords", name)
                for name, coord in obj.coords.items()
            },
        }
    elif isinstance(obj, pd.DataFrame):
        metadata = {
            "kind": "dataframe",
            "attrs": dict(obj.attrs),
            "index_names": list(obj.index.names),
            "index": add_index(obj.index),
            "columns_names": list(obj.columns.names),
            "columns": [
                {"label": label, "column": add_column(column, "columns", position)}
                for position, (label, column) in enumerate(obj.items())
            ],
        }
    elif isinstance(obj, pd.Series):
        metadata = {
            "kind": "series",
            "name": obj.name,
            "attrs": dict(obj.attrs),
            "index_names": list(obj.index.names),
            "index": add_index(obj.index),
            "column": add_column(obj, "variables", "series"),
        }
    else:
        metadata = {"kind": "ndarray", "file": add(obj, "variables", "array")}

    def encode(value: Any) -> Any:
        """Convert metadata to JSON with tags for tuples, dicts, arrays, and stores."""
        if value is None or isinstance(value, str | bool | int | float):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist(), "dtype": value.dtype.str}
        if isinstance(value, pd.DataFrame | pd.Series | xr.DataArray | xr.Dataset):
            nested.append(value)
            return {"__xnpy__": f"attrs/{len(nested) - 1}"}
        if isinstance(value, tuple):
            return {"__tuple__": [encode(item) for item in value]}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {"__dict__": [[encode(k), encode(v)] for k, v in value.items()]}
        raise TypeError(f"Unsupported metadata type: {type(value).__name__}.")

    slab_bytes = 64 * 1024**2

    def write_array(source: Any, path: Path) -> None:
        np.save(path, np.asarray(source), allow_pickle=False)

    def write_slab(source: xr.Variable, path: Path, offset: int) -> None:
        slab = np.ascontiguousarray(source)
        with path.open("r+b") as file:
            file.seek(offset)
            file.write(slab)

    def flush(target: np.ndarray, *_: Any) -> None:
        target.flush()

    def make_task(source: Any, path: Path) -> list[Any]:
        # Object arrays (e.g. strings) cannot be memory-mapped or saved without pickle.
        if source.dtype == object:
            source = np.asarray(source).astype(str)

        if getattr(source, "chunks", None) is not None:
            import dask.array as da

            target = np.lib.format.open_memmap(
                path, "w+", dtype=source.dtype, shape=source.shape
            )
            data = source.data if isinstance(source, xr.DataArray) else source
            stored = da.store(data, target, lock=False, compute=False)
            return [dask.delayed(flush, pure=False)(target, stored)]

        if (
            isinstance(source, np.ndarray)
            or source.nbytes <= slab_bytes
            or not source.ndim
        ):
            # np.save writes in-memory arrays sequentially without a full copy.
            return [dask.delayed(write_array, pure=False)(source, path)]

        # Lazily indexed arrays (e.g. netCDF, HDF5, or larger than RAM) are written
        # in contiguous C-order blocks of about slab_bytes. Blocks split the first
        # axis whose single index fits in slab_bytes, iterating any leading axes,
        # so no task loads more than one block regardless of the array shape.
        # Blocks are aligned to on-disk chunks so no chunk is decompressed twice.
        offset = np.lib.format.open_memmap(
            path, "w+", dtype=source.dtype, shape=source.shape
        ).offset
        strides = [
            source.dtype.itemsize * int(np.prod(source.shape[axis + 1 :]))
            for axis in range(source.ndim)
        ]
        axis = next(axis for axis, size in enumerate(strides) if size <= slab_bytes)
        chunk = source.encoding.get("preferred_chunks", {}).get(source.dims[axis], 1)
        step = max(slab_bytes // strides[axis] // chunk, 1) * chunk
        return [
            # Each task gets only its lazy slice; nothing is read until it runs.
            dask.delayed(write_slab, pure=False)(
                source.variable[(*lead, slice(start, start + step))],
                path,
                offset
                + sum(i * size for i, size in zip(lead, strides, strict=False))
                + start * strides[axis],
            )
            for lead in np.ndindex(*source.shape[:axis])
            for start in range(0, source.shape[axis], step)
        ]

    try:
        metadata = encode(metadata)
        root.mkdir(parents=True)
        for folder in {(root / file).parent for _, file in jobs}:
            folder.mkdir(exist_ok=True)

        tasks = [
            task for source, file in jobs for task in make_task(source, root / file)
        ]
        dask.compute(*tasks, scheduler=scheduler, num_workers=num_workers)

        for position, value in enumerate(nested):
            to_xnpy(
                value,
                root / "attrs" / str(position),
                scheduler=scheduler,
                num_workers=num_workers,
            )

        with (root / "metadata.json").open("w") as file:
            json.dump(metadata, file, indent=2)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def open_xnpy(
    path: str | Path,
    variable: Hashable | None = None,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | xr.Dataset:
    """Open a NumPy, pandas, or xarray object from a memory-mappable XNpy store.

    The return type is determined at runtime from the store manifest. Use
    :func:`open_xnpy_dataframe` or :func:`open_xnpy_dataset` for static typing.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    variable : hashable, optional
        Dataset variable or DataFrame column label to return. Ignored for
        ndarray, Series, and DataArray stores. For a DataFrame, only the
        selected column payloads are opened and a one-column DataFrame is
        returned. If omitted, return the complete stored object.
    mmap_mode : {"r", "r+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`. Use ``"c"`` for
        copy-on-write arrays that accept in-place edits without modifying the
        files, or ``None`` to load payloads into ordinary in-memory arrays.
        Text columns are always loaded into memory.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame or xarray.DataArray or xarray.Dataset
        Reconstructed object. A Dataset store with ``variable`` returns a
        DataArray. Array payloads are memory-mapped unless ``mmap_mode=None``.

    Notes
    -----
    Memory mapping defers physical reads until array pages are accessed, so stores
    larger than available RAM can be opened without materializing their payloads.
    """
    root = Path(path)

    def decode(value: Any) -> Any:
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            if "__tuple__" in value:
                return tuple(decode(item) for item in value["__tuple__"])
            if "__dict__" in value:
                return {decode(k): decode(v) for k, v in value["__dict__"]}
            if "__ndarray__" in value:
                return np.array(value["__ndarray__"], dtype=value["dtype"])
            if "__xnpy__" in value:
                return open_xnpy(root / value["__xnpy__"], mmap_mode=mmap_mode)
            return {k: decode(v) for k, v in value.items()}
        return value

    metadata = decode(json.loads((root / "metadata.json").read_text()))
    kind = metadata["kind"]

    def load(file: str) -> np.ndarray:
        return np.load(root / file, mmap_mode=mmap_mode, allow_pickle=False)

    def load_variable(info: dict[str, Any]) -> xr.Variable:
        return xr.Variable(info["dims"], load(info["file"]), attrs=info["attrs"])

    def load_column(spec: dict[str, Any]) -> Any:
        values = load(spec["file"])
        if spec["encoding"] == "categorical":
            return pd.Categorical.from_codes(
                values, spec["categories"], ordered=spec["ordered"]
            )
        if spec["encoding"] == "bool":
            codes = np.asarray(values)
            if spec["dtype"] == "boolean":
                return pd.arrays.BooleanArray(codes == 1, codes < 0)
            out = (codes == 1).astype(object)
            out[codes < 0] = np.nan
            # An object Index keeps object dtype in Series, DataFrame, and indexes; a
            # bare object ndarray of strings would be inferred as str by pandas >= 3.
            return pd.Index(out, dtype=object, copy=False)
        if spec["encoding"] == "masked":
            mask = np.asarray(load(spec["mask"]))
            if spec["dtype"] == "object":
                out = values.astype(object)
                out[mask] = np.nan
                return pd.Index(out, dtype=object, copy=False)
            array_type = pd.api.types.pandas_dtype(spec["dtype"]).construct_array_type()
            return array_type(np.asarray(values), mask)
        if spec["encoding"] == "string":
            out = values.astype(object)
            # Stores written before dtypes were saved carry an "extension" flag.
            dtype = spec.get("dtype", "string" if spec.get("extension") else "object")
            if dtype == "object":
                out[np.asarray(load(spec["mask"]))] = np.nan
                return pd.Index(out, dtype=object, copy=False)
            dtype = pd.api.types.pandas_dtype(dtype)
            out[np.asarray(load(spec["mask"]))] = dtype.na_value
            return pd.array(out, dtype=dtype)
        return values

    def load_index() -> pd.Index:
        levels = [load_column(spec) for spec in metadata["index"]]
        names = metadata["index_names"]
        if len(levels) == 1:
            return pd.Index(levels[0], name=names[0])
        return pd.MultiIndex.from_arrays(levels, names=names)

    if kind == "ndarray":
        return load(metadata["file"])
    elif kind == "series":
        series = pd.Series(
            load_column(metadata["column"]),
            index=load_index(),
            name=metadata["name"],
            copy=False,
        )
        series.attrs = metadata["attrs"]
        return series
    elif kind == "dataframe":
        columns = [(c["label"], c["column"]) for c in metadata["columns"]]
        if variable is not None:
            columns = [column for column in columns if column[0] == variable]
            if not columns:
                raise KeyError(variable)

        names = metadata["columns_names"]
        labels = [label for label, _ in columns]
        frame = pd.DataFrame(
            {position: load_column(spec) for position, (_, spec) in enumerate(columns)},
            index=load_index(),
            copy=False,
        )
        frame.columns = (
            pd.MultiIndex.from_tuples(labels, names=names)
            if len(names) > 1
            else pd.Index(labels, name=names[0], tupleize_cols=False)
        )
        frame.attrs = metadata["attrs"]
        return frame
    elif kind in {"dataarray", "dataset"}:
        coords = {
            name: load_variable(info) for name, info in metadata["coords"].items()
        }
        if kind == "dataarray":
            return xr.DataArray(
                load_variable(metadata["variable"]),
                coords=coords,
                name=metadata["name"],
            )

        dataset = xr.Dataset(
            {name: load_variable(info) for name, info in metadata["variables"].items()},
            coords=coords,
            attrs=metadata["attrs"],
        )
        return dataset if variable is None else dataset[variable]
    else:
        raise ValueError(f"Unknown XNpy kind: {kind!r}.")


def open_xnpy_ndarray(
    path: str | Path,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> np.ndarray:
    """Open an ndarray store.

    Raises
    ------
    TypeError
        If the store does not hold an ndarray.
    """
    obj = open_xnpy(path, mmap_mode=mmap_mode)
    if not isinstance(obj, np.ndarray):
        raise TypeError(f"Store {path} does not hold an ndarray.")
    return obj


def open_xnpy_dataframe(
    path: str | Path,
    variable: Hashable | None = None,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> pd.DataFrame:
    """Open a DataFrame or Series store, optionally restricted to one column.

    For a DataFrame store, only the payloads of the column labelled ``variable``
    are opened and the result is a one-column DataFrame carrying the stored
    index. ``variable`` is ignored for a Series store.

    Raises
    ------
    TypeError
        If the store holds neither a DataFrame nor a Series.
    """
    obj = open_xnpy(path, variable, mmap_mode=mmap_mode)
    if not isinstance(obj, pd.DataFrame | pd.Series):
        raise TypeError(f"Store {path} does not hold a DataFrame or Series.")
    return obj


def open_xnpy_dataset(
    path: str | Path,
    *,
    mmap_mode: Literal["r", "r+", "c"] | None = "r",
) -> xr.Dataset:
    """Open a Dataset store.

    Raises
    ------
    TypeError
        If the store does not hold a Dataset.
    """
    obj = open_xnpy(path, mmap_mode=mmap_mode)
    if not isinstance(obj, xr.Dataset):
        raise TypeError(f"Store {path} does not hold a Dataset.")
    return obj
