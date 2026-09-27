"""Provide array I/O and shared-memory transport."""

from __future__ import annotations

import atexit
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import xarray as xr


def to_xnpy(
    obj: np.ndarray | xr.DataArray | xr.Dataset,
    path: str | Path,
    *,
    mode: Literal["w", "w-"] = "w-",
    parallel: bool = False,
    max_workers: int | None = None,
) -> None:
    """Write a NumPy or xarray object to a memory-mappable XNpy store.

    Large xarray variables are streamed in bounded slabs. Dask-backed variables
    are written with :func:`dask.array.store`, allowing all chunks to execute in
    one task graph without serial chunk scheduling.

    Parameters
    ----------
    obj : numpy.ndarray or xarray.DataArray or xarray.Dataset
        Object to store.
    path : str or pathlib.Path
        Destination store directory.
    mode : {"w", "w-"}, default="w-"
        Write mode. ``"w"`` transactionally replaces an existing store after the
        new store is complete. ``"w-"`` requires that the destination not exist.
    parallel : bool, default=False
        Write independent Dataset variables and coordinates concurrently using a
        thread pool. Dask chunk parallelism is independent of this option.
    max_workers : int, optional
        Maximum number of Dataset write workers when ``parallel=True``. If
        omitted, at most four workers are used.

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
) -> np.ndarray | xr.DataArray | xr.Dataset:
    """Open a NumPy or xarray object from a memory-mappable XNpy store.

    Parameters
    ----------
    path : str or pathlib.Path
        XNpy store directory.
    variable : str, optional
        Dataset variable to reconstruct. If omitted, reconstruct the complete
        stored object. For a stored DataArray, its variable name may be supplied.
    mmap_mode : {"r", "r+", "w+", "c"} or None, default="r"
        Memory-map mode passed to :func:`numpy.load`. Use ``None`` to load payloads
        into ordinary in-memory NumPy arrays.

    Returns
    -------
    numpy.ndarray or xarray.DataArray or xarray.Dataset
        Reconstructed object. Array payloads are memory-mapped unless
        ``mmap_mode=None``.

    Notes
    -----
    Memory mapping defers physical reads until array pages are accessed, so stores
    larger than available RAM can be opened without materializing their payloads.
    """
    return XNpyStore(path).open(variable=variable, mmap_mode=mmap_mode)


class XNpyStore:
    """Directory store for memory-mappable NumPy and xarray objects."""

    META_FILE = "metadata.json"
    VARIABLE_DIR = "variables"
    COORD_DIR = "coords"
    DEFAULT_MAX_WORKERS = 4
    SLAB_BYTES = 256 * 1024**2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta_path = self.path / self.META_FILE
        self.variable_path = self.path / self.VARIABLE_DIR
        self.coord_path = self.path / self.COORD_DIR

    def to_disk(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
        *,
        mode: Literal["w", "w-"] = "w-",
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """Write an ndarray, DataArray, or Dataset to the store."""
        if mode not in {"w", "w-"}:
            raise ValueError(f"Unsupported XNpy mode: {mode!r}.")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if not isinstance(obj, np.ndarray | xr.DataArray | xr.Dataset):
            raise TypeError(
                "Only numpy.ndarray, xarray.DataArray, and xarray.Dataset are supported."
            )

        if self.path.exists():
            if mode == "w-":
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)

        self.path.mkdir(parents=True)
        try:
            if isinstance(obj, xr.Dataset):
                metadata = self._save_dataset(obj, parallel, max_workers)
            elif isinstance(obj, xr.DataArray):
                metadata = self._save_dataarray(obj)
            else:
                self.variable_path.mkdir()
                metadata = {
                    "kind": "ndarray",
                    "array": self._save_array(self.variable_path / "array", obj),
                }

            with self.meta_path.open("w", encoding="utf-8") as file:
                json.dump(
                    metadata,
                    file,
                    default=self._json_default,
                    ensure_ascii=False,
                    allow_nan=True,
                    separators=(",", ":"),
                )
        except Exception:
            shutil.rmtree(self.path, ignore_errors=True)
            raise

    def metadata(self) -> dict[str, Any]:
        """Return the decoded JSON manifest without opening array payloads."""
        with self.meta_path.open("r", encoding="utf-8") as file:
            return json.load(file, object_hook=self._json_object_hook)

    def open(
        self,
        variable: str | None = None,
        *,
        mmap_mode: str | None = "r",
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Open the stored object or one Dataset variable."""
        metadata = self.metadata()

        def load_variable(info: dict[str, Any]) -> xr.Variable:
            return xr.Variable(
                info["dims"],
                self._load_array(info["array"], mmap_mode),
                attrs=info.get("attrs", {}),
            )

        def load_coords(names: list[str]) -> dict[str, xr.Variable]:
            return {name: load_variable(metadata["coords"][name]) for name in names}

        kind = metadata["kind"]
        if kind == "ndarray":
            if variable is not None:
                raise ValueError("variable= is valid only for xarray stores.")
            return self._load_array(metadata["array"], mmap_mode)

        if kind == "dataarray":
            if variable is not None and variable != metadata["variable_name"]:
                raise KeyError(variable)
            info = metadata["variable"]
            return xr.DataArray(
                load_variable(info),
                coords=load_coords(info["coords"]),
                name=metadata["name"],
            )

        if kind != "dataset":
            raise ValueError(f"Unknown store kind: {kind!r}.")

        variables = metadata["variables"]
        if variable is not None:
            if variable not in variables:
                raise KeyError(
                    f"{variable!r} not found; available variables: {tuple(variables)!r}"
                )
            info = variables[variable]
            return xr.DataArray(
                load_variable(info),
                coords=load_coords(info["coords"]),
                name=variable,
            )

        return xr.Dataset(
            data_vars={name: load_variable(info) for name, info in variables.items()},
            coords={
                name: load_variable(info)
                for name, info in metadata.get("coords", {}).items()
            },
            attrs=metadata.get("attrs", {}),
        )

    def _save_dataset(
        self,
        dataset: xr.Dataset,
        parallel: bool,
        max_workers: int | None,
    ) -> dict[str, Any]:
        self.variable_path.mkdir()
        self.coord_path.mkdir()

        jobs = [
            (
                "variables",
                name,
                variable,
                self._payload_path(self.variable_path, name),
                list(dataset[name].coords),
            )
            for name, variable in dataset.data_vars.items()
        ] + [
            (
                "coords",
                name,
                coord,
                self._payload_path(self.coord_path, name),
                None,
            )
            for name, coord in dataset.coords.items()
        ]

        def save(job: tuple[Any, ...]) -> tuple[str, str, dict[str, Any]]:
            role, name, source, path, coords = job
            info = self._write_array(source, path)
            if coords is not None:
                info["coords"] = coords
            return role, name, info

        if parallel and jobs:
            workers = min(max_workers or self.DEFAULT_MAX_WORKERS, len(jobs))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                saved = list(executor.map(save, jobs))
        else:
            saved = list(map(save, jobs))

        return {
            "kind": "dataset",
            "attrs": dict(dataset.attrs),
            "variables": {
                name: info for role, name, info in saved if role == "variables"
            },
            "coords": {name: info for role, name, info in saved if role == "coords"},
        }

    def _save_dataarray(self, array: xr.DataArray) -> dict[str, Any]:
        self.variable_path.mkdir()
        self.coord_path.mkdir(exist_ok=True)

        name = str(array.name) if array.name is not None else "data"
        variable = self._write_array(
            array,
            self._payload_path(self.variable_path, name),
        )
        variable["coords"] = list(array.coords)

        return {
            "kind": "dataarray",
            "name": array.name,
            "variable_name": name,
            "variable": variable,
            "coords": {
                coord_name: self._write_array(
                    coord,
                    self._payload_path(self.coord_path, coord_name),
                )
                for coord_name, coord in array.coords.items()
            },
        }

    def _write_array(self, array: xr.DataArray, path: Path) -> dict[str, Any]:
        return {
            "dims": list(array.dims),
            "attrs": dict(array.attrs),
            "array": self._save_array(path, array),
        }

    def _save_array(
        self,
        path: Path,
        source: np.ndarray | xr.DataArray,
    ) -> dict[str, Any]:
        dtype = np.dtype(source.dtype)
        shape = tuple(source.shape)

        if isinstance(source, xr.DataArray) and source.chunks is not None:
            import dask.array as da

            target = np.lib.format.open_memmap(path, "w+", dtype=dtype, shape=shape)
            try:
                da.store(source.data, target, lock=False, scheduler="threads")
                target.flush()
            finally:
                del target
        elif source.nbytes > self.SLAB_BYTES and source.ndim:
            target = np.lib.format.open_memmap(path, "w+", dtype=dtype, shape=shape)
            try:
                row_bytes = max(math.prod(shape[1:]) * dtype.itemsize, 1)
                step = max(self.SLAB_BYTES // row_bytes, 1)
                for start in range(0, shape[0], step):
                    stop = min(start + step, shape[0])
                    target[start:stop] = np.asarray(source[start:stop])
                target.flush()
            finally:
                del target
        else:
            with path.open("wb") as file:
                np.save(file, np.asarray(source), allow_pickle=False)

        return {
            "file": str(path.relative_to(self.path)),
            "dtype": dtype.str,
            "shape": list(shape),
        }

    def _load_array(
        self,
        info: dict[str, Any],
        mmap_mode: str | None,
    ) -> np.ndarray:
        path = self.path / info["file"]
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if array.dtype != np.dtype(info["dtype"]):
            raise TypeError(
                f"{path}: dtype {array.dtype} != {np.dtype(info['dtype'])}."
            )
        if array.shape != tuple(info["shape"]):
            raise ValueError(f"{path}: shape {array.shape} != {tuple(info['shape'])}.")
        return array

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.dtype):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, bytes):
            return {"__bytes__": value.hex()}
        if isinstance(value, set | frozenset):
            return {"__set__": list(value)}
        raise TypeError(f"Unsupported metadata type: {type(value).__name__}.")

    @staticmethod
    def _json_object_hook(value: dict[str, Any]) -> Any:
        if set(value) == {"__bytes__"}:
            return bytes.fromhex(value["__bytes__"])
        if set(value) == {"__set__"}:
            return set(value["__set__"])
        return value

    @staticmethod
    def _payload_path(parent: Path, name: Any) -> Path:
        value = str(name)
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Unsafe array name: {value!r}.")
        return parent / value


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
