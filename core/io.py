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
    mode: Literal["w", "w-", "a"] = "w-",
    parallel: bool = False,
):
    """Write a NumPy or xarray object to a memory-mappable XNpy store.

    Parameters
    ----------
    obj : numpy.ndarray or xarray.DataArray or xarray.Dataset
        Object to store.
    path : str or pathlib.Path
        Store path.
    mode : {"w", "w-", "a"}, default "w-"
        Write mode. ``"w"`` replaces an existing store, ``"w-"`` requires
        that the store does not already exist, and ``"a"`` adds or replaces
        complete Dataset variables and coordinates in an existing store.
    parallel : bool, default False
        Write Dataset variables and coordinates concurrently. The worker count
        is the number of data variables plus coordinates being written.

    Returns
    -------
    pathlib.Path
        Path to the completed store.
    """
    return XNpyStore(path).save(obj, mode=mode, parallel=parallel)


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
        Store path.
    variable : str, optional
        Dataset variable to reconstruct. None reconstructs the complete stored
        object. For a DataArray, the stored variable name may also be supplied.
    mmap_mode : {"r", "r+", "w+", "c"} or None, default "r"
        Memory-map mode passed to :func:`numpy.load`. None loads payloads into
        ordinary in-memory arrays.

    Returns
    -------
    numpy.ndarray or xarray.DataArray or xarray.Dataset
        Reconstructed object.
    """
    return XNpyStore(path).load(variable=variable, mmap_mode=mmap_mode)


class XNpyStore:
    """Directory store for memory-mappable NumPy and xarray objects."""

    META_FILE = "metadata.json"
    VARIABLE_DIR = "variables"
    COORD_DIR = "coords"
    FORMAT = "xnpy"
    VERSION = 2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta_path = self.path / self.META_FILE
        self.variable_path = self.path / self.VARIABLE_DIR
        self.coord_path = self.path / self.COORD_DIR

    def save(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
        *,
        mode: Literal["w", "w-", "a"] = "w-",
        parallel: bool = False,
    ):
        """Save an array, DataArray, or Dataset."""
        if mode not in {"w", "w-", "a"}:
            raise ValueError(f"Unsupported XNpy mode: {mode!r}.")

        if mode == "a" and self.path.exists():
            if not isinstance(obj, xr.Dataset):
                raise TypeError("mode='a' is supported only for xarray.Dataset stores.")
            metadata = self.metadata()
            if metadata["kind"] != "dataset":
                raise ValueError("mode='a' requires an existing Dataset store.")
            metadata = self._update_dataset(obj, metadata, parallel=parallel)
            self._write_metadata(metadata)
            return self.path

        if self.path.exists():
            if mode in {"w-", "a"}:
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)

        self.path.mkdir(parents=True)

        try:
            if isinstance(obj, xr.Dataset):
                metadata = self._save_dataset(obj, parallel=parallel)
            elif isinstance(obj, xr.DataArray):
                metadata = self._save_dataarray(obj)
            elif isinstance(obj, np.ndarray):
                metadata = self._save_ndarray(obj)
            else:
                raise TypeError(
                    "Only numpy.ndarray, xarray.DataArray, and xarray.Dataset are supported."
                )
            self._write_metadata(metadata)
        except Exception:
            shutil.rmtree(self.path, ignore_errors=True)
            raise

    def metadata(self) -> dict[str, Any]:
        """Read and validate the JSON manifest without opening payloads."""
        with self.meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("format") != self.FORMAT:
            raise ValueError(f"Not an {self.FORMAT!r} store: {self.path}.")
        if metadata.get("version") != self.VERSION:
            raise ValueError(
                f"Unsupported {self.FORMAT} version: {metadata.get('version')!r}."
            )
        return metadata

    def load(
        self,
        variable: str | None = None,
        *,
        mmap_mode: str | None = "r",
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Load a stored object or one Dataset variable."""
        metadata = self.metadata()
        kind = metadata["kind"]

        if kind == "dataset":
            if variable is None:
                return self._load_dataset(metadata, mmap_mode)
            return self._load_dataset_variable(metadata, variable, mmap_mode)

        if kind == "dataarray":
            if variable is not None and variable != metadata["variable_name"]:
                raise KeyError(variable)
            return self._load_dataarray(metadata, mmap_mode)

        if kind == "ndarray":
            if variable is not None:
                raise ValueError("variable= is valid only for xarray stores.")
            return self._load_array(metadata["array"], mmap_mode)

        raise ValueError(f"Unknown store kind: {kind!r}")

    def _save_dataset(
        self,
        dataset: xr.Dataset,
        *,
        parallel: bool = False,
    ) -> dict[str, Any]:
        """Save every data variable and coordinate of a Dataset."""
        self.variable_path.mkdir()
        self.coord_path.mkdir()

        variable_jobs = [
            (
                name,
                variable,
                self._payload_path(self.variable_path, name),
                list(dataset[name].coords),
            )
            for name, variable in dataset.data_vars.items()
        ]
        coord_jobs = [
            (name, coord, self._payload_path(self.coord_path, name))
            for name, coord in dataset.coords.items()
        ]

        variables, coords = self._write_dataset_jobs(
            variable_jobs,
            coord_jobs,
            parallel=parallel,
            atomic=False,
        )
        return {
            "kind": "dataset",
            "attrs": self._json_value(dict(dataset.attrs)),
            "variables": variables,
            "coords": coords,
        }

    def _update_dataset(
        self,
        dataset: xr.Dataset,
        metadata: dict[str, Any],
        *,
        parallel: bool,
    ) -> dict[str, Any]:
        """Add or replace complete Dataset variables and coordinates."""
        self.variable_path.mkdir(exist_ok=True)
        self.coord_path.mkdir(exist_ok=True)

        self._validate_replacements(dataset, metadata)
        variable_jobs = [
            (
                name,
                variable,
                self._payload_path(self.variable_path, name),
                list(dataset[name].coords),
            )
            for name, variable in dataset.data_vars.items()
        ]
        coord_jobs = [
            (name, coord, self._payload_path(self.coord_path, name))
            for name, coord in dataset.coords.items()
        ]
        variables, coords = self._write_dataset_jobs(
            variable_jobs,
            coord_jobs,
            parallel=parallel,
            atomic=True,
        )
        metadata["variables"].update(variables)
        metadata["coords"].update(coords)

        attrs = dict(metadata["attrs"])
        attrs.update(self._json_value(dict(dataset.attrs)))
        metadata["attrs"] = attrs
        return metadata

    def _write_dataset_jobs(
        self,
        variable_jobs: list[tuple[str, xr.DataArray, Path, list[str]]],
        coord_jobs: list[tuple[str, xr.DataArray, Path]],
        *,
        parallel: bool,
        atomic: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Write Dataset payload jobs, optionally one worker per payload."""

        def save_variable(
            job: tuple[str, xr.DataArray, Path, list[str]],
        ) -> tuple[str, dict[str, Any]]:
            name, variable, path, coords = job
            info = self._save_xarray_variable(variable, path, atomic=atomic)
            info["coords"] = coords
            return name, info

        def save_coord(
            job: tuple[str, xr.DataArray, Path],
        ) -> tuple[str, dict[str, Any]]:
            name, coord, path = job
            return name, self._save_xarray_variable(coord, path, atomic=atomic)

        jobs = len(variable_jobs) + len(coord_jobs)
        if parallel and jobs:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                variable_futures = [
                    executor.submit(save_variable, job) for job in variable_jobs
                ]
                coord_futures = [executor.submit(save_coord, job) for job in coord_jobs]
                variables = dict(future.result() for future in variable_futures)
                coords = dict(future.result() for future in coord_futures)
        else:
            variables = dict(save_variable(job) for job in variable_jobs)
            coords = dict(save_coord(job) for job in coord_jobs)
        return variables, coords

    def _save_dataarray(self, array: xr.DataArray) -> dict[str, Any]:
        """Save a DataArray and its coordinates."""
        self.variable_path.mkdir()
        if array.coords:
            self.coord_path.mkdir()
        name = str(array.name) if array.name is not None else "data"
        variable = self._save_xarray_variable(
            array,
            self._payload_path(self.variable_path, name),
        )
        variable["coords"] = list(array.coords)
        return {
            "kind": "dataarray",
            "name": array.name,
            "variable_name": name,
            "variable": variable,
            "coords": self._save_coords(array),
        }

    def _save_coords(self, obj: xr.Dataset | xr.DataArray) -> dict[str, Any]:
        """Save every coordinate directly under the coordinate directory."""
        if not obj.coords:
            return {}
        self.coord_path.mkdir(exist_ok=True)
        return {
            name: self._save_xarray_variable(
                coord,
                self._payload_path(self.coord_path, name),
            )
            for name, coord in obj.coords.items()
        }

    def _save_ndarray(self, array: np.ndarray) -> dict[str, Any]:
        """Save a bare NumPy array under the variable directory."""
        self.variable_path.mkdir()
        return {
            "kind": "ndarray",
            "array": self._save_array(self.variable_path / "array", array),
        }

    def _save_xarray_variable(
        self,
        variable: xr.Variable | xr.DataArray,
        path: Path,
        *,
        atomic: bool = False,
    ) -> dict[str, Any]:
        """Save one variable payload and reconstruction metadata."""
        return {
            "dims": list(variable.dims),
            "attrs": self._json_value(dict(variable.attrs)),
            "array": self._save_array(
                path,
                np.asarray(variable.data),
                atomic=atomic,
            ),
        }

    def _save_array(
        self,
        path: Path,
        array: np.ndarray,
        *,
        atomic: bool = False,
    ) -> dict[str, Any]:
        """Write one extensionless NPY payload."""
        if array.dtype.hasobject:
            raise TypeError("object-dtype arrays are not supported")
        target = path
        temporary = path.with_name(f".{path.name}.tmp") if atomic else path
        try:
            with temporary.open("wb") as file:
                np.save(file, array, allow_pickle=False)
            if atomic:
                temporary.replace(target)
        finally:
            if atomic:
                temporary.unlink(missing_ok=True)
        return {
            "file": str(target.relative_to(self.path)),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    def _validate_replacements(
        self,
        dataset: xr.Dataset,
        metadata: dict[str, Any],
    ) -> None:
        """Reject replacements that would make shared dimension sizes inconsistent."""
        sizes: dict[str, int] = {}
        incoming_vars = set(dataset.data_vars)
        incoming_coords = set(dataset.coords)

        for name, info in metadata["variables"].items():
            if name not in incoming_vars:
                self._merge_dimension_sizes(sizes, info)
        for name, info in metadata["coords"].items():
            if name not in incoming_coords:
                self._merge_dimension_sizes(sizes, info)
        for variable in dataset.data_vars.values():
            self._merge_variable_sizes(sizes, variable)
        for coord in dataset.coords.values():
            self._merge_variable_sizes(sizes, coord)

    @staticmethod
    def _merge_dimension_sizes(sizes: dict[str, int], info: dict[str, Any]) -> None:
        """Merge dimension sizes from stored variable metadata."""
        for dim, size in zip(info["dims"], info["array"]["shape"], strict=True):
            previous = sizes.setdefault(dim, size)
            if previous != size:
                raise ValueError(
                    f"Dimension {dim!r} has conflicting sizes {previous} and {size}."
                )

    @staticmethod
    def _merge_variable_sizes(
        sizes: dict[str, int],
        variable: xr.Variable | xr.DataArray,
    ) -> None:
        """Merge dimension sizes from an incoming xarray variable."""
        for dim, size in zip(variable.dims, variable.shape, strict=True):
            previous = sizes.setdefault(dim, size)
            if previous != size:
                raise ValueError(
                    f"Dimension {dim!r} has conflicting sizes {previous} and {size}."
                )

    def _load_dataset(
        self,
        metadata: dict[str, Any],
        mmap_mode: str | None,
    ) -> xr.Dataset:
        """Reconstruct a complete Dataset."""
        return xr.Dataset(
            data_vars={
                name: self._as_variable(info, mmap_mode)
                for name, info in metadata["variables"].items()
            },
            coords=self._load_coords(
                metadata["coords"],
                metadata["coords"],
                mmap_mode,
            ),
            attrs=metadata["attrs"],
        )

    def _load_dataset_variable(
        self,
        metadata: dict[str, Any],
        name: str,
        mmap_mode: str | None,
    ) -> xr.DataArray:
        """Reconstruct one Dataset variable and the coordinates it uses."""
        variables = metadata["variables"]
        if name not in variables:
            raise KeyError(
                f"{name!r} not found; available variables: {tuple(variables)!r}"
            )
        return self._build_dataarray(
            variables[name],
            metadata["coords"],
            name,
            mmap_mode,
        )

    def _load_dataarray(
        self,
        metadata: dict[str, Any],
        mmap_mode: str | None,
    ) -> xr.DataArray:
        """Reconstruct a stored DataArray."""
        return self._build_dataarray(
            metadata["variable"],
            metadata["coords"],
            metadata["name"],
            mmap_mode,
        )

    def _build_dataarray(
        self,
        info: dict[str, Any],
        coord_metadata: dict[str, Any],
        name: Any,
        mmap_mode: str | None,
    ) -> xr.DataArray:
        """Assemble a DataArray from its payload and recorded coordinates."""
        return xr.DataArray(
            self._load_array(info["array"], mmap_mode),
            dims=info["dims"],
            coords=self._load_coords(coord_metadata, info["coords"], mmap_mode),
            name=name,
            attrs=info["attrs"],
        )

    def _load_coords(
        self,
        metadata: dict[str, Any],
        names: Any,
        mmap_mode: str | None,
    ) -> dict[str, tuple[Any, ...]]:
        """Load the named coordinates."""
        return {name: self._as_variable(metadata[name], mmap_mode) for name in names}

    def _as_variable(
        self,
        info: dict[str, Any],
        mmap_mode: str | None,
    ) -> tuple[Any, ...]:
        """Return the ``(dims, data, attrs)`` triple xarray builds from."""
        return (
            info["dims"],
            self._load_array(info["array"], mmap_mode),
            info["attrs"],
        )

    def _load_array(
        self,
        info: dict[str, Any],
        mmap_mode: str | None,
    ) -> np.ndarray:
        """Load a payload and validate its recorded dtype and shape."""
        path = self.path / info["file"]
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        expected_dtype = np.dtype(info["dtype"])
        expected_shape = tuple(info["shape"])
        if array.dtype != expected_dtype:
            raise TypeError(f"{path}: dtype {array.dtype} != {expected_dtype}")
        if array.shape != expected_shape:
            raise ValueError(f"{path}: shape {array.shape} != {expected_shape}")
        return array

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        """Atomically replace the manifest after payload writes complete."""
        temporary = self.meta_path.with_name(f".{self.META_FILE}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(
                    {**metadata, "format": self.FORMAT, "version": self.VERSION},
                    file,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            temporary.replace(self.meta_path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        """Convert metadata to JSON-safe values without lossy coercion."""
        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TypeError("Non-finite float metadata are not supported.")
            return value
        if isinstance(value, np.generic):
            return cls._json_value(value.item())
        if isinstance(value, np.ndarray):
            return cls._json_value(value.tolist())
        if isinstance(value, list | tuple):
            return [cls._json_value(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("JSON metadata dictionaries require string keys.")
            return {key: cls._json_value(item) for key, item in value.items()}
        raise TypeError(f"Unsupported JSON metadata type: {type(value).__name__}.")

    @classmethod
    def _payload_path(cls, parent: Path, name: Any) -> Path:
        """Return a safe extensionless payload path."""
        value = str(name)
        if not value or value in {".", ".."}:
            raise ValueError(f"Empty or unsafe array name: {value!r}.")
        if "/" in value or "\\" in value:
            raise ValueError(f"Array names cannot contain path separators: {value!r}.")
        return parent / value


class SharedMemoryObject:
    """Zero-copy shared-memory transport wrapper for NumPy and xarray objects.

    This class enables efficient inter-process communication by placing
    compatible arrays into shared memory blocks, avoiding the overhead
    of pickling large data buffers.

    Supported Types
    --------------
    * ``numpy.ndarray`` (pointer-free data types only)
    * ``xarray.DataArray``
    * ``xarray.Dataset``

    Parameters
    ----------
    obj : numpy.ndarray, xarray.DataArray, or xarray.Dataset
        The array or dataset to place into shared memory.
    readonly : bool, default=False
        If True, the reconstructed NumPy buffers in receiving processes
        will be marked as read-only.

    Behavior & Pickling
    -------------------
    When pickled (e.g., via ``multiprocessing``, ``Queue``, or
    ``ProcessPoolExecutor``), receiving processes transparently unpack
    the object back into a native ndarray, DataArray, or Dataset referencing
    the shared memory.

    * **Pointer-free arrays** are placed in shared memory.
    * **Object-dtype arrays** automatically fall back to standard pickling
      because their memory contains process-local Python pointers.

    Lifecycle Management & Best Practices
    -------------------------------------
    The creating process retains ownership of the underlying shared-memory
    segments. **You must keep this ``SharedMemoryObject`` instance alive**
    until all worker processes have finished using the reconstructed objects.

    * **Recommended:** Always use this class as a context manager to
      ensure proper cleanup of shared resources:

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

        if cls._atexit_registered:
            return

        atexit.register(cls.close_attachments)

        cls._atexit_registered = True

    @staticmethod
    def _open_attachment(
        name: str,
    ) -> shared_memory.SharedMemory:

        # Python >= 3.13 supports track=False. This prevents receiver

        # processes from independently trying to unlink memory owned by

        # the creating process.

        try:
            return shared_memory.SharedMemory(
                name=name,
                track=False,
            )

        except TypeError:
            # Python <= 3.12

            return shared_memory.SharedMemory(name=name)

    @classmethod
    def _attach(
        cls,
        name: str,
    ) -> shared_memory.SharedMemory:

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

        array = np.asarray(array)

        order: Literal["C", "F"]

        if array.flags.f_contiguous and not array.flags.c_contiguous:
            order = "F"

        else:
            order = "C"

        # Raw shared memory cannot safely represent object-containing

        # dtypes because the buffer contains CPython pointers.

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

        return {
            "name": name,
            "role": role,
            "dims": tuple(variable.dims),
            "attrs": dict(variable.attrs),
            "encoding": dict(variable.encoding),
            "array": self._encode_array(np.asarray(variable.data)),
        }

    def _encode_object(
        self,
        obj: np.ndarray | xr.DataArray | xr.Dataset,
    ) -> dict[str, Any]:

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
                "encoding": dict(obj.encoding),
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

        variable = xr.Variable(
            spec["dims"],
            cls._decode_array(
                spec["array"],
                readonly=readonly,
            ),
            attrs=dict(spec["attrs"]),
        )

        variable.encoding.update(spec["encoding"])

        return variable

    @classmethod
    def _rebuild(
        cls,
        spec: dict[str, Any],
    ) -> np.ndarray | xr.DataArray | xr.Dataset:

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

            dataset.encoding.update(spec["encoding"])

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
                # A live NumPy array may still export the

                # underlying mmap. The OS will clean it up

                # when this process terminates.

                continue

            del cls._attachments[name]

    def __enter__(self):

        if self._closed:
            raise RuntimeError("SharedMemoryObject is closed.")

        return self

    def __exit__(self, *_: object) -> None:

        self.close()
