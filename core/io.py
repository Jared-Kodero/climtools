"""Provide array I/O and shared-memory transport."""

from __future__ import annotations

import atexit
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

import xarray as xr


def to_xnpy(
    obj: np.ndarray | pd.DataFrame | xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    mode: Literal["w", "w-"] = "w-",
    parallel: bool = False,
    max_workers: int | None = None,
) -> None:
    """Write a NumPy, pandas, or xarray object to a memory-mappable XNpy store.

    Large arrays are streamed in bounded slabs. Dask-backed variables are written
    with :func:`dask.array.store`, allowing all chunks to execute in one task
    graph without serial chunk scheduling.

    Parameters
    ----------
    obj : numpy.ndarray or pandas.DataFrame or xarray.DataArray or xarray.Dataset
        Object to store.
    path : str or pathlib.Path
        Destination store directory.
    mode : {"w", "w-"}, default="w-"
        Write mode. ``"w"`` replaces an existing store. ``"w-"`` requires that
        the destination not exist.
    parallel : bool, default=False
        Write arrays concurrently using a thread pool. Dask chunk parallelism is
        independent of this option.
    max_workers : int, optional
        Maximum number of write workers when ``parallel=True``. If omitted, at
        most four workers are used.

    Returns
    -------
    None
        The store is written to ``path``.
    """
    XNpyStore(path).to_disk(
        obj,
        mode=mode,
        parallel=parallel,
        max_workers=max_workers,
    )


def open_xnpy(
    path: str | Path,
    variable: str | None = None,
    *,
    mmap_mode: str | None = "r",
) -> np.ndarray | pd.DataFrame | pd.Series | xr.DataArray | xr.Dataset:
    """Open a NumPy, pandas, or xarray object from a memory-mappable XNpy store.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    variable : str, optional
        Dataset variable or DataFrame column to return. If omitted, return the
        complete stored object.
    mmap_mode : {"r", "r+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`. Use ``None`` to load payloads
        into ordinary in-memory NumPy arrays.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame or pandas.Series or xarray.DataArray or
    xarray.Dataset
        Reconstructed object. Array payloads are memory-mapped unless
        ``mmap_mode=None``.

    Notes
    -----
    Memory mapping defers physical reads until array pages are accessed, so stores
    larger than available RAM can be opened without materializing their payloads.
    """
    return XNpyStore(path).open(variable=variable, mmap_mode=mmap_mode)


class XNpyStore:
    """Directory store for memory-mappable NumPy, pandas, and xarray objects."""

    META_FILE = "metadata.json"
    DEFAULT_MAX_WORKERS = 4
    SLAB_BYTES = 256 * 1024**2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta_path = self.path / self.META_FILE
        self._jobs: list[tuple[Any, Path]] = []

    def metadata(self) -> dict[str, Any]:
        """Return the JSON manifest without opening array payloads."""
        return json.loads(self.meta_path.read_text())

    def open(
        self,
        variable: str | None = None,
        *,
        mmap_mode: str | None = "r",
    ) -> np.ndarray | pd.DataFrame | pd.Series | xr.DataArray | xr.Dataset:
        """Open the stored object, one Dataset variable, or one DataFrame column."""
        metadata = self.metadata()
        kind = metadata["kind"]

        if kind == "ndarray":
            return np.load(
                self.path / metadata["file"],
                mmap_mode=mmap_mode,
                allow_pickle=False,
            )

        if kind == "dataframe":
            frame = pd.DataFrame(
                {
                    name: np.load(
                        self.path / file,
                        mmap_mode=mmap_mode,
                        allow_pickle=False,
                    )
                    for name, file in metadata["columns"].items()
                },
                index=pd.Index(
                    np.load(
                        self.path / metadata["index"],
                        mmap_mode=mmap_mode,
                        allow_pickle=False,
                    ),
                    name=metadata["index_name"],
                ),
                copy=False,
            )
            return frame if variable is None else frame[variable]

        coords = {
            name: self._load_variable(info, mmap_mode)
            for name, info in metadata["coords"].items()
        }

        if kind == "dataarray":
            return xr.DataArray(
                self._load_variable(metadata["variable"], mmap_mode),
                coords=coords,
                name=metadata["name"],
            )

        dataset = xr.Dataset(
            {
                name: self._load_variable(info, mmap_mode)
                for name, info in metadata["variables"].items()
            },
            coords=coords,
            attrs=metadata["attrs"],
        )
        return dataset if variable is None else dataset[variable]

    def to_disk(
        self,
        obj: np.ndarray | pd.DataFrame | xr.DataArray | xr.Dataset,
        *,
        mode: Literal["w", "w-"] = "w-",
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """Write an ndarray, DataFrame, DataArray, or Dataset to the store."""
        if mode not in {"w", "w-"}:
            raise ValueError(f"Unsupported XNpy mode: {mode!r}.")

        if self.path.exists():
            if mode == "w-":
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)

        self._jobs = []

        if isinstance(obj, xr.Dataset):
            metadata = {
                "kind": "dataset",
                "attrs": dict(obj.attrs),
                "variables": {
                    name: self._add_variable(var, "variables", name)
                    for name, var in obj.data_vars.items()
                },
                "coords": {
                    name: self._add_variable(coord, "coords", name)
                    for name, coord in obj.coords.items()
                },
            }
        elif isinstance(obj, xr.DataArray):
            metadata = {
                "kind": "dataarray",
                "name": obj.name,
                "variable": self._add_variable(
                    obj,
                    "variables",
                    "data" if obj.name is None else obj.name,
                ),
                "coords": {
                    name: self._add_variable(coord, "coords", name)
                    for name, coord in obj.coords.items()
                },
            }
        elif isinstance(obj, pd.DataFrame):
            metadata = {
                "kind": "dataframe",
                "index_name": obj.index.name,
                "index": self._add(obj.index.to_numpy(), "coords", "index"),
                "columns": {
                    str(name): self._add(column.to_numpy(), "variables", name)
                    for name, column in obj.items()
                },
            }
        else:
            metadata = {"kind": "ndarray", "file": self._add(obj, "variables", "array")}

        try:
            self.path.mkdir(parents=True)
            for folder in {path.parent for _, path in self._jobs}:
                folder.mkdir(exist_ok=True)

            workers = (max_workers or self.DEFAULT_MAX_WORKERS) if parallel else 1
            with ThreadPoolExecutor(workers) as executor:
                list(executor.map(self._save, *zip(*self._jobs)))

            with self.meta_path.open("w") as file:
                json.dump(metadata, file, default=self._json_default)
        except Exception:
            shutil.rmtree(self.path, ignore_errors=True)
            raise

    def _load_variable(
        self,
        info: dict[str, Any],
        mmap_mode: str | None,
    ) -> xr.Variable:
        return xr.Variable(
            info["dims"],
            np.load(self.path / info["file"], mmap_mode=mmap_mode, allow_pickle=False),
            attrs=info["attrs"],
        )

    def _add(self, source: Any, folder: str, name: Any) -> str:
        file = f"{folder}/{name}.npy"
        self._jobs.append((source, self.path / file))
        return file

    def _add_variable(
        self,
        source: xr.DataArray,
        folder: str,
        name: Any,
    ) -> dict[str, Any]:
        return {
            "dims": list(source.dims),
            "attrs": dict(source.attrs),
            "file": self._add(source, folder, name),
        }

    def _save(self, source: Any, path: Path) -> None:
        # Object arrays (e.g. strings) cannot be memory-mapped or saved without pickle.
        if source.dtype == object:
            source = np.asarray(source).astype(str)

        if getattr(source, "chunks", None) is not None:
            import dask.array as da

            target = np.lib.format.open_memmap(
                path, "w+", dtype=source.dtype, shape=source.shape
            )
            data = source.data if isinstance(source, xr.DataArray) else source
            da.store(data, target, lock=False, scheduler="threads")
            target.flush()
        elif source.nbytes > self.SLAB_BYTES and source.ndim:
            # Stream large or lazily indexed arrays in bounded first-axis slabs.
            target = np.lib.format.open_memmap(
                path, "w+", dtype=source.dtype, shape=source.shape
            )
            step = max(self.SLAB_BYTES * source.shape[0] // source.nbytes, 1)
            for start in range(0, source.shape[0], step):
                target[start : start + step] = np.asarray(source[start : start + step])
            target.flush()
        else:
            np.save(path, np.asarray(source), allow_pickle=False)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported metadata type: {type(value).__name__}.")


class SharedMemoryObject:
    """Zero-copy shared-memory transport wrapper for NumPy and xarray objects.

    Enables efficient inter-process communication by placing compatible arrays
    into shared memory blocks, bypassing the serialization overhead of large data buffers.

    Parameters
    ----------
    obj : numpy.ndarray, xarray.DataArray, or xarray.Dataset
        The target object to place into shared memory.
    readonly : bool, default=True
        If True, reconstructed NumPy buffers in receiving processes will be
        marked as read-only to prevent unintended mutation.

    Supported Types & Limitations
    -----------------------------
    * **NumPy:** Pointer-free data types only. Object-dtype arrays automatically
      fall back to standard pickling because their memory contains local pointers.
    * **Xarray:** ``DataArray`` and ``Dataset`` structures (backing data is shared).

    Behavior & Pickling
    -------------------
    When pickled (e.g., via ``multiprocessing``, ``Queue``, or ``ProcessPoolExecutor``),
    receiving processes transparently unpack the object back into a native container
    referencing the underlying shared memory block.

    Lifecycle Management & Best Practices
    -------------------------------------
    The creating process retains ownership of the shared-memory segments.
    **You must keep this instance alive** until all worker processes have finished
    using the reconstructed objects.

    * **Recommended:** Always use this class as a context manager to ensure
      proper resource cleanup:

        >>> with SharedMemoryObject(my_array) as shared_obj:
        ...     pool.apply_async(worker_func, args=(shared_obj,))
    """

    _attachments: ClassVar[dict[str, shared_memory.SharedMemory]] = {}

    _atexit_registered: ClassVar[bool] = False

    def __init__(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
        *,
        readonly: bool = True,
    ) -> None:
        """Allocate shared memory blocks and encode the object specification."""

        self._owners: list[shared_memory.SharedMemory] = []

        self._closed = False

        self._readonly = readonly

        self._ensure_atexit()

        try:
            self._spec = self._encode_object(obj)

        except Exception:
            self.close()

            raise

    @classmethod
    def _ensure_atexit(cls) -> None:
        """Register process-exit cleanup handler for receiver-side attachments."""

        if cls._atexit_registered:
            return

        atexit.register(cls.close_attachments)

        cls._atexit_registered = True

    @staticmethod
    def _open_attachment(
        name: str,
    ) -> shared_memory.SharedMemory:
        """Open an existing shared memory block, disabling tracking on Python 3.13+."""

        try:
            return shared_memory.SharedMemory(
                name=name,
                track=False,
            )

        except TypeError:
            return shared_memory.SharedMemory(name=name)

    @classmethod
    def _attach(
        cls,
        name: str,
    ) -> shared_memory.SharedMemory:
        """Attach to a shared memory block, caching the handle locally."""

        cls._ensure_atexit()

        shm = cls._attachments.get(name)

        if shm is None:
            shm = cls._open_attachment(name)

            cls._attachments[name] = shm

        return shm

    def _encode_array(
        self,
        array: np.ndarray,
    ) -> dict[str, Any]:
        """Copy a NumPy array into a new shared memory block or inline fallback."""

        array = np.asarray(array)

        order: Literal["C", "F"]

        if array.flags.f_contiguous and not array.flags.c_contiguous:
            order = "F"

        else:
            order = "C"

        if array.dtype.hasobject:
            return {
                "shape": array.shape,
                "dtype": array.dtype,
                "order": order,
                "shm_name": None,
                "inline": np.array(
                    array,
                    copy=True,
                    order=order,
                ),
            }

        contiguous = np.array(
            array,
            copy=True,
            order=order,
        )

        shm = shared_memory.SharedMemory(
            create=True,
            size=max(contiguous.nbytes, 1),
        )

        self._owners.append(shm)

        target = np.ndarray(
            contiguous.shape,
            dtype=contiguous.dtype,
            buffer=shm.buf,
            order=order,
        )

        target[...] = contiguous

        return {
            "shape": contiguous.shape,
            "dtype": contiguous.dtype,
            "order": order,
            "shm_name": shm.name,
            "inline": None,
        }

    def _encode_variable(
        self,
        name: Any,
        role: Literal["data", "coord"],
        variable: xr.Variable,
    ) -> dict[str, Any]:
        """Encode an xarray Variable or coordinate with its array payload."""

        return {
            "name": name,
            "role": role,
            "dims": tuple(variable.dims),
            "attrs": dict(variable.attrs),
            "array": self._encode_array(np.asarray(variable.data)),
        }

    def _encode_object(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
    ) -> dict[str, Any]:
        """Encode a supported NumPy or xarray object into a serializable spec."""

        if isinstance(obj, xr.Dataset):
            variables = [
                self._encode_variable(name, "data", variable)
                for name, variable in obj.data_vars.items()
            ]

            variables.extend(
                self._encode_variable(
                    name,
                    "coord",
                    variable,
                )
                for name, variable in obj.coords.items()
            )

            return {
                "kind": "dataset",
                "attrs": dict(obj.attrs),
                "variables": variables,
                "readonly": self._readonly,
            }

        if isinstance(obj, xr.DataArray):
            variables = [self._encode_variable(None, "data", obj.variable)]

            variables.extend(
                self._encode_variable(
                    name,
                    "coord",
                    variable,
                )
                for name, variable in obj.coords.items()
            )

            return {
                "kind": "dataarray",
                "name": obj.name,
                "variables": variables,
                "readonly": self._readonly,
            }

        if isinstance(obj, np.ndarray):
            return {
                "kind": "ndarray",
                "array": self._encode_array(obj),
                "readonly": self._readonly,
            }

        raise TypeError(
            f"Expected np.ndarray, xr.DataArray, or xr.Dataset; got {type(obj)!r}."
        )

    @classmethod
    def _decode_array(cls, spec: dict[str, Any], *, readonly: bool) -> np.ndarray:
        """Reconstruct a NumPy array from shared memory or inline buffer."""

        shm_name = spec["shm_name"]

        if shm_name is None:
            array = spec["inline"]

        else:
            shm = cls._attach(shm_name)

            array = np.ndarray(
                spec["shape"], dtype=spec["dtype"], buffer=shm.buf, order=spec["order"]
            )

        if readonly:
            array.flags.writeable = False

        return array

    @classmethod
    def _decode_variable(cls, spec: dict[str, Any], *, readonly: bool) -> xr.Variable:
        """Reconstruct an xarray Variable from its encoded specification."""

        variable = xr.Variable(
            spec["dims"],
            cls._decode_array(
                spec["array"],
                readonly=readonly,
            ),
            attrs=dict(spec["attrs"]),
        )

        return variable

    @classmethod
    def _rebuild(
        cls,
        spec: dict[str, Any],
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Reconstruct the original object in a receiving process."""

        readonly = spec["readonly"]

        kind = spec["kind"]

        if kind == "ndarray":
            return cls._decode_array(
                spec["array"],
                readonly=readonly,
            )

        decoded = [
            (
                variable_spec,
                cls._decode_variable(variable_spec, readonly=readonly),
            )
            for variable_spec in spec["variables"]
        ]

        if kind == "dataset":
            data_vars = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "data"
            }

            coords = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "coord"
            }

            dataset = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs=dict(spec["attrs"]),
            )

            return dataset

        if kind == "dataarray":
            data_variable = next(
                variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "data"
            )

            coords = {
                variable_spec["name"]: variable
                for variable_spec, variable in decoded
                if variable_spec["role"] == "coord"
            }

            return xr.DataArray(data_variable, coords=coords, name=spec["name"])

        raise RuntimeError(f"Unknown object kind: {kind!r}")

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Control multiprocessing/pickle reconstruction."""

        if self._closed:
            raise RuntimeError("Cannot serialize a closed SharedMemoryObject.")

        return self._rebuild, (self._spec,)

    def get(self) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Return a local view of the shared object."""

        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self._rebuild(self._spec)

    @property
    def nbytes(self) -> int:
        """Total allocated shared-memory bytes."""

        return sum(shm.size for shm in self._owners)

    @property
    def closed(self) -> bool:
        """Whether the owner has been closed."""

        return self._closed

    def close(self) -> None:
        """Unlink and close owned shared-memory segments."""

        if self._closed:
            return

        for shm in self._owners:
            try:
                shm.unlink()

            except FileNotFoundError:
                pass

            shm.close()

        self._owners.clear()

        self._closed = True

    @classmethod
    def close_attachments(cls) -> None:
        """Close receiver-side shared-memory handles."""

        for name, shm in list(cls._attachments.items()):
            try:
                shm.close()

            except (BufferError, OSError):
                continue

            del cls._attachments[name]

    def __enter__(self):
        """Enter context manager."""

        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager, ensuring shared memory is unlinked and closed."""

        self.close()
