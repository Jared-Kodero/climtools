"""Provide array I/O and shared-memory transport."""

from __future__ import annotations

import atexit
import json
import shutil
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

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
    xNpy(path).to_disk(
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
    return xNpy(path).open(variable=variable, mmap_mode=mmap_mode)


class xNpy:
    META_FILE = "metadata.json"
    VARIABLE_DIR = "variables"
    COORD_DIR = "coords"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta_path = self.path / self.META_FILE
        self.variable_path = self.path / self.VARIABLE_DIR
        self.coord_path = self.path / self.COORD_DIR

    def metadata(self) -> dict[str, Any]:
        with self.meta_path.open() as file:
            return json.load(file)

    def open(
        self,
        variable: str | None = None,
        *,
        mmap_mode: str | None = "r",
    ):
        metadata = self.metadata()
        kind = metadata["kind"]

        if kind == "ndarray":
            return self._load_array(metadata["array"], mmap_mode)

        if kind == "dataframe":
            index = pd.Index(
                self._load_array(metadata["index"], mmap_mode),
                name=metadata["index_name"],
            )

            if variable is not None:
                return pd.Series(
                    self._load_array(metadata["columns"][variable], mmap_mode),
                    index=index,
                    name=variable,
                    copy=False,
                )

            return pd.DataFrame(
                {
                    name: self._load_array(info, mmap_mode)
                    for name, info in metadata["columns"].items()
                },
                index=index,
                copy=False,
            )

        coords = {
            name: xr.Variable(
                info["dims"],
                self._load_array(info["array"], mmap_mode),
                attrs=info.get("attrs", {}),
            )
            for name, info in metadata["coords"].items()
        }

        if kind == "dataarray":
            info = metadata["variable"]

            return xr.DataArray(
                xr.Variable(
                    info["dims"],
                    self._load_array(info["array"], mmap_mode),
                    attrs=info.get("attrs", {}),
                ),
                coords={name: coords[name] for name in info["coords"]},
                name=metadata["name"],
            )

        if variable is not None:
            info = metadata["variables"][variable]

            return xr.DataArray(
                xr.Variable(
                    info["dims"],
                    self._load_array(info["array"], mmap_mode),
                    attrs=info.get("attrs", {}),
                ),
                coords={name: coords[name] for name in info["coords"]},
                name=variable,
            )

        return xr.Dataset(
            data_vars={
                name: xr.Variable(
                    info["dims"],
                    self._load_array(info["array"], mmap_mode),
                    attrs=info.get("attrs", {}),
                )
                for name, info in metadata["variables"].items()
            },
            coords=coords,
            attrs=metadata.get("attrs", {}),
        )

    def to_disk(
        self,
        obj: np.ndarray | pd.DataFrame | xr.DataArray | xr.Dataset,
    ) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)

        self.path.mkdir(parents=True)

        if isinstance(obj, pd.DataFrame):
            metadata = self._save_dataframe(obj)
        elif isinstance(obj, xr.Dataset):
            metadata = self._save_dataset(obj)
        elif isinstance(obj, xr.DataArray):
            metadata = self._save_dataarray(obj)
        else:
            self.variable_path.mkdir()
            metadata = {
                "kind": "ndarray",
                "array": self._save_array(
                    self.variable_path / "array",
                    obj,
                ),
            }

        with self.meta_path.open("w") as file:
            json.dump(metadata, file, default=self._json_default)

    def _save_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        self.variable_path.mkdir()
        self.coord_path.mkdir()

        index_name = str(df.index.name or "index")

        return {
            "kind": "dataframe",
            "index_name": df.index.name,
            "index": self._save_array(
                self.coord_path / index_name,
                df.index.to_numpy(),
            ),
            "columns": {
                str(name): self._save_array(
                    self.variable_path / str(name),
                    values.to_numpy(),
                )
                for name, values in df.items()
            },
        }

    def _save_dataset(self, ds: xr.Dataset) -> dict[str, Any]:
        self.variable_path.mkdir()
        self.coord_path.mkdir()

        coords = {
            name: self._save_variable(
                coord,
                self.coord_path / str(name),
            )
            for name, coord in ds.coords.items()
        }

        variables = {}

        for name, variable in ds.data_vars.items():
            info = self._save_variable(
                variable,
                self.variable_path / str(name),
            )
            info["coords"] = list(ds[name].coords)
            variables[name] = info

        return {
            "kind": "dataset",
            "attrs": dict(ds.attrs),
            "variables": variables,
            "coords": coords,
        }

    def _save_dataarray(self, array: xr.DataArray) -> dict[str, Any]:
        self.variable_path.mkdir()
        self.coord_path.mkdir()

        name = str(array.name or "data")

        variable = self._save_variable(
            array,
            self.variable_path / name,
        )
        variable["coords"] = list(array.coords)

        return {
            "kind": "dataarray",
            "name": array.name,
            "variable": variable,
            "coords": {
                coord_name: self._save_variable(
                    coord,
                    self.coord_path / str(coord_name),
                )
                for coord_name, coord in array.coords.items()
            },
        }

    def _save_variable(
        self,
        variable: xr.DataArray,
        path: Path,
    ) -> dict[str, Any]:
        return {
            "dims": list(variable.dims),
            "attrs": dict(variable.attrs),
            "array": self._save_array(path, variable.values),
        }

    def _save_array(
        self,
        path: Path,
        array: np.ndarray,
    ) -> str:
        # Append .npy extension to the file path
        path = Path(str(path) + ".npy")

        with path.open("wb") as file:
            np.save(file, np.asarray(array), allow_pickle=False)

        return str(path.relative_to(self.path))

    def _load_array(
        self,
        path: str,
        mmap_mode: str | None,
    ) -> np.ndarray:
        return np.load(
            self.path / path,
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError


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
