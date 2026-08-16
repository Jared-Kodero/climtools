"""Transparent MPI reductions for xarray ``DataArray`` and ``Dataset`` objects.

Reached through the existing accessor surface::

    ds.xgeo.mpi.mean(dim="time")
    ds.mpi.mean(dim="time")

Reduction methods mirror xarray's public signatures. In a multi-rank job, the
largest requested reduction dimension that can use every rank is partitioned
contiguously and automatically. Each rank evaluates only its local block, the
reduced partial is materialized, and MPI combines the partials into the same
xarray structure on every rank. Users do not manage ranks, communicators,
partitioning, gather/scatter, or reconstruction.

Every rank must call a reduction in the same order and must begin with the same
logical xarray object, normally a lazily opened dataset on shared storage. MPI
initialization remains lazy through ``MPI.world``. Without a usable MPI runtime
in a serial process, methods delegate directly to xarray with unchanged
semantics.

Only mergeable reductions are parallelized: sum, product, minimum, maximum,
mean, count, variance, standard deviation, any, and all. Exact median, quantile,
and cumulative operations are intentionally absent because the available MPI
primitives would require transferring or reconstructing the unreduced axis.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

from .runtime import mpi

if TYPE_CHECKING:
    from typing import Literal


def norm_dims(
    obj: xr.DataArray | xr.Dataset, dim: Hashable | Iterable[Hashable] | None
) -> tuple[Hashable, ...]:
    """Return the effective reduction dimensions using xarray semantics."""
    if dim is None or dim is Ellipsis:
        return tuple(obj.dims)

    if isinstance(dim, Hashable) and dim in obj.dims:
        return (dim,)

    if isinstance(dim, Iterable) and not isinstance(dim, (str, bytes)):
        dims = tuple(dim)
    else:
        dims = (cast("Hashable", dim),)

    missing = [name for name in dims if name not in obj.dims]
    if missing:
        raise ValueError(
            f"Dimensions {missing!r} not found in xarray object dimensions "
            + f"{tuple(obj.dims)!r}."
        )
    return dims


def dim_has_data(obj: xr.DataArray | xr.Dataset, dim: Hashable) -> bool:
    if isinstance(obj, xr.DataArray):
        return dim in obj.dims
    return any(dim in variable.dims for variable in obj.data_vars.values())


def partition_for_reduction(
    obj: xr.DataArray | xr.Dataset, dim: Hashable | Iterable[Hashable] | None
) -> tuple[xr.DataArray | xr.Dataset, Hashable | None, tuple[Hashable, ...]]:
    """Return this rank's automatic block and the dimension used to split it."""
    dims = norm_dims(obj, dim)
    size = mpi.world.size()
    if size <= 1 or not dims:
        return obj, None, dims

    candidates = [
        name
        for name in dims
        if int(obj.sizes[name]) >= size and dim_has_data(obj, name)
    ]
    if not candidates:
        return obj, None, dims

    partition_dim = max(candidates, key=lambda name: int(obj.sizes[name]))
    start, stop = mpi.world.partition(int(obj.sizes[partition_dim]))
    return obj.isel({partition_dim: slice(start, stop)}), partition_dim, dims


def load_data(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Materialize a reduced partial before it crosses the MPI boundary."""
    return obj.compute()


def _rank_dim(parts: list[xr.DataArray]) -> str:
    dims = {name for part in parts for name in part.dims}
    name = "__xgeo_mpi_rank__"
    while name in dims:
        name += "_"
    return name


def combine_dataarrays(
    parts: list[xr.DataArray],
    op: Literal["sum", "prod", "min", "max", "any", "all"],
    *,
    skipna: bool | None = None,
) -> xr.DataArray:
    """Combine rank-local reduced arrays without changing xarray metadata."""
    template = parts[0]
    rank_dim = _rank_dim(parts)
    stacked = xr.concat(
        parts,
        dim=rank_dim,
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )

    if op in {"sum", "prod"}:
        combined = getattr(stacked, op)(dim=rank_dim, skipna=False, keep_attrs=True)
    elif op in {"min", "max"}:
        combined = getattr(stacked, op)(dim=rank_dim, skipna=skipna, keep_attrs=True)
    else:
        combined = getattr(stacked, op)(dim=rank_dim, keep_attrs=True)

    return template.copy(data=combined.data)


def distributed_names(
    original: xr.Dataset, template: xr.Dataset, partition_dim: Hashable
) -> tuple[Hashable, ...]:
    return tuple(
        name
        for name in template.data_vars
        if name in original.data_vars and partition_dim in original[name].dims
    )


def combine_objects(
    parts: list[xr.DataArray | xr.Dataset],
    original: xr.DataArray | xr.Dataset,
    partition_dim: Hashable,
    op: Literal["sum", "prod", "min", "max", "any", "all"],
    *,
    skipna: bool | None = None,
) -> xr.DataArray | xr.Dataset:
    template = parts[0]
    if isinstance(template, xr.DataArray):
        arrays = [cast("xr.DataArray", part) for part in parts]
        return combine_dataarrays(arrays, op, skipna=skipna)

    dataset = template.copy(deep=False)
    original_ds = cast("xr.Dataset", original)
    dataset_parts = [cast("xr.Dataset", part) for part in parts]
    for name in distributed_names(original_ds, dataset, partition_dim):
        combined = combine_dataarrays(
            [part[name] for part in dataset_parts], op, skipna=skipna
        )
        dataset[name] = dataset[name].copy(data=combined.data)
    return dataset


def combine_counts(
    parts: list[xr.DataArray | xr.Dataset],
    original: xr.DataArray | xr.Dataset,
    partition_dim: Hashable,
) -> xr.DataArray | xr.Dataset:
    return combine_objects(parts, original, partition_dim, "sum", skipna=False)


def min_count_active(dtype: np.dtype[Any], skipna: bool | None) -> bool:
    if skipna is not None:
        return skipna
    return np.issubdtype(dtype, np.inexact) or dtype.kind == "O"


def apply_min_count(
    result: xr.DataArray | xr.Dataset,
    counts: xr.DataArray | xr.Dataset,
    original: xr.DataArray | xr.Dataset,
    partition_dim: Hashable,
    min_count: int | None,
    skipna: bool | None,
) -> xr.DataArray | xr.Dataset:
    if min_count is None:
        return result

    if isinstance(result, xr.DataArray):
        original_da = cast("xr.DataArray", original)
        if not min_count_active(original_da.dtype, skipna):
            return result
        masked = result.where(cast("xr.DataArray", counts) >= min_count)
        return result.copy(data=masked.data)

    dataset = result.copy(deep=False)
    counts_ds = cast("xr.Dataset", counts)
    original_ds = cast("xr.Dataset", original)
    for name in distributed_names(original_ds, dataset, partition_dim):
        if not min_count_active(original_ds[name].dtype, skipna):
            continue
        masked = dataset[name].where(counts_ds[name] >= min_count)
        dataset[name] = dataset[name].copy(data=masked.data)
    return dataset


def restore_nondistributed(
    result: xr.DataArray | xr.Dataset,
    template: xr.DataArray | xr.Dataset | None,
    original: xr.DataArray | xr.Dataset,
    partition_dim: Hashable,
) -> xr.DataArray | xr.Dataset:
    if template is None or isinstance(result, xr.DataArray):
        return result

    dataset = cast("xr.Dataset", result).copy(deep=False)
    template_ds = cast("xr.Dataset", template)
    original_ds = cast("xr.Dataset", original)
    distributed = set(distributed_names(original_ds, dataset, partition_dim))
    for name in dataset.data_vars:
        if name not in distributed and name in template_ds.data_vars:
            dataset[name] = template_ds[name]
    return dataset


def weighted_mean(
    means: list[xr.DataArray],
    counts: list[xr.DataArray],
    *,
    skipna: bool | None,
) -> xr.DataArray:
    template = means[0]
    if np.issubdtype(template.dtype, np.datetime64):
        raise TypeError("datetime means require the serial xarray path")

    rank_dim = _rank_dim(means)
    weighted: list[xr.DataArray] = []
    for mean, count in zip(means, counts, strict=True):
        if np.issubdtype(mean.dtype, np.inexact):
            weight = count.astype(mean.dtype)
        else:
            weight = count
        value = mean
        if skipna is not False:
            value = value.where(count > 0, 0)
        weighted.append(value * weight)

    numerator = xr.concat(
        weighted, dim=rank_dim, coords="minimal", compat="override"
    ).sum(dim=rank_dim, skipna=False)
    denominator = xr.concat(
        counts, dim=rank_dim, coords="minimal", compat="override"
    ).sum(dim=rank_dim, skipna=False)
    if np.issubdtype(numerator.dtype, np.inexact):
        denominator = denominator.astype(numerator.dtype)
    combined = numerator / denominator
    if combined.dtype != template.dtype:
        combined = combined.astype(template.dtype)
    return template.copy(data=combined.data)


def global_mean(
    mean_parts: list[xr.DataArray | xr.Dataset],
    count_parts: list[xr.DataArray | xr.Dataset],
    original: xr.DataArray | xr.Dataset,
    partition_dim: Hashable,
    dim: Hashable | Iterable[Hashable] | None,
    *,
    skipna: bool | None,
    keep_attrs: bool | None,
    kwargs: dict[str, Any],
) -> xr.DataArray | xr.Dataset:
    template = mean_parts[0]
    if isinstance(template, xr.DataArray):
        if np.issubdtype(template.dtype, np.datetime64):
            return (
                cast(
                    "xr.DataArray",
                    original,
                )
                .mean(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
                .compute()
            )
        return weighted_mean(
            [cast("xr.DataArray", part) for part in mean_parts],
            [cast("xr.DataArray", part) for part in count_parts],
            skipna=skipna,
        )

    dataset = cast("xr.Dataset", template).copy(deep=False)
    original_ds = cast("xr.Dataset", original)
    means = [cast("xr.Dataset", part) for part in mean_parts]
    counts = [cast("xr.Dataset", part) for part in count_parts]
    effective_dims = norm_dims(original_ds, dim)
    for name in distributed_names(original_ds, dataset, partition_dim):
        if np.issubdtype(dataset[name].dtype, np.datetime64):
            variable_dims = tuple(
                item for item in effective_dims if item in original_ds[name].dims
            )
            serial = (
                original_ds[name]
                .mean(
                    dim=variable_dims,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                    **kwargs,
                )
                .compute()
            )
            dataset[name] = serial
            continue
        combined = weighted_mean(
            [part[name] for part in means],
            [part[name] for part in counts],
            skipna=skipna,
        )
        dataset[name] = dataset[name].copy(data=combined.data)
    return dataset


def global_variance(
    mean_parts: list[xr.DataArray | xr.Dataset],
    var_parts: list[xr.DataArray | xr.Dataset],
    count_parts: list[xr.DataArray | xr.Dataset],
    original: xr.DataArray | xr.Dataset,
    partition_dim: Hashable,
    ddof: int,
    *,
    skipna: bool | None,
    std: bool,
) -> xr.DataArray | xr.Dataset:
    def combine_one(
        means: list[xr.DataArray],
        variances: list[xr.DataArray],
        counts: list[xr.DataArray],
    ) -> xr.DataArray:
        template = variances[0]
        global_mean = weighted_mean(means, counts, skipna=skipna)
        rank_dim = _rank_dim(variances)
        terms: list[xr.DataArray] = []
        for mean, variance, count in zip(means, variances, counts, strict=True):
            count_for_var = count.astype(variance.dtype)
            delta = mean - global_mean
            correction = np.abs(delta) ** 2 * count_for_var
            term = variance * count_for_var + correction
            if skipna is not False:
                term = term.where(count > 0, 0)
            terms.append(term)

        m2 = xr.concat(terms, dim=rank_dim, coords="minimal", compat="override").sum(
            dim=rank_dim, skipna=False
        )
        total_count = xr.concat(
            counts, dim=rank_dim, coords="minimal", compat="override"
        ).sum(dim=rank_dim, skipna=False)
        denominator = total_count - ddof
        result = xr.where(denominator > 0, m2 / denominator, np.nan)
        if std:
            result = np.sqrt(result)
        if result.dtype != template.dtype:
            result = result.astype(template.dtype)
        return template.copy(data=result.data)

    template = var_parts[0]
    if isinstance(template, xr.DataArray):
        return combine_one(
            [cast("xr.DataArray", part) for part in mean_parts],
            [cast("xr.DataArray", part) for part in var_parts],
            [cast("xr.DataArray", part) for part in count_parts],
        )

    dataset = cast("xr.Dataset", template).copy(deep=False)
    original_ds = cast("xr.Dataset", original)
    means = [cast("xr.Dataset", part) for part in mean_parts]
    variances = [cast("xr.Dataset", part) for part in var_parts]
    counts = [cast("xr.Dataset", part) for part in count_parts]
    for name in distributed_names(original_ds, dataset, partition_dim):
        combined = combine_one(
            [part[name] for part in means],
            [part[name] for part in variances],
            [part[name] for part in counts],
        )
        dataset[name] = dataset[name].copy(data=combined.data)
    return dataset


def associative_reduction(
    obj: xr.DataArray | xr.Dataset,
    op: Literal["sum", "prod", "min", "max", "any", "all"],
    dim: Hashable | Iterable[Hashable] | None,
    *,
    skipna: bool | None = None,
    min_count: int | None = None,
    keep_attrs: bool | None = None,
    kwargs: dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Apply a mergeable xarray reduction across MPI ranks.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        Logical xarray object to reduce.
    op : {"sum", "prod", "min", "max", "any", "all"}
        Associative reduction to apply.
    dim : hashable, iterable of hashable, or None
        Dimension or dimensions to reduce using xarray semantics.
    skipna : bool or None, optional
        Whether to skip missing values for reductions that support it.
    min_count : int or None, optional
        Minimum number of valid values required for ``sum`` or ``prod``.
    keep_attrs : bool or None, optional
        Whether to preserve xarray attributes.
    kwargs : dict of str to Any or None, optional
        Additional keyword arguments forwarded to the xarray reduction.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Globally reduced xarray object, identical on every MPI rank.
    """
    call_kwargs = {} if kwargs is None else kwargs
    block, partition_dim, _ = partition_for_reduction(obj, dim)

    method = getattr(block, op)
    reduction_kwargs: dict[str, Any] = {"dim": dim, "keep_attrs": keep_attrs}
    if op in {"sum", "prod", "min", "max"}:
        reduction_kwargs["skipna"] = skipna
    if op in {"sum", "prod"}:
        reduction_kwargs["min_count"] = None if partition_dim is not None else min_count
    reduction_kwargs.update(call_kwargs)

    if partition_dim is None:
        return method(**reduction_kwargs)

    nondistributed_template: xr.DataArray | xr.Dataset | None = None
    if (
        isinstance(block, xr.Dataset)
        and min_count is not None
        and op in {"sum", "prod"}
    ):
        template_kwargs = dict(reduction_kwargs)
        template_kwargs["min_count"] = min_count
        nondistributed_template = load_data(method(**template_kwargs))

    local = load_data(method(**reduction_kwargs))
    local_count: xr.DataArray | xr.Dataset | None = None
    if min_count is not None and op in {"sum", "prod"}:
        local_count = load_data(block.count(dim=dim, keep_attrs=False))

    payload = mpi.world.allgather((local, local_count))
    parts = [cast("xr.DataArray | xr.Dataset", item[0]) for item in payload]
    result = combine_objects(parts, obj, partition_dim, op, skipna=skipna)
    result = restore_nondistributed(result, nondistributed_template, obj, partition_dim)
    if local_count is None:
        return result

    count_parts = [cast("xr.DataArray | xr.Dataset", item[1]) for item in payload]
    counts = combine_counts(count_parts, obj, partition_dim)
    return apply_min_count(result, counts, obj, partition_dim, min_count, skipna)


def variance_reduction(
    obj: xr.DataArray | xr.Dataset,
    dim: Hashable | Iterable[Hashable] | None,
    *,
    skipna: bool | None,
    ddof: int,
    keep_attrs: bool | None,
    std: bool,
    kwargs: dict[str, Any],
) -> xr.DataArray | xr.Dataset:
    """Compute a distributed variance or standard deviation.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        Logical xarray object to reduce.
    dim : hashable, iterable of hashable, or None
        Dimension or dimensions to reduce using xarray semantics.
    skipna : bool or None
        Whether to skip missing values.
    ddof : int
        Delta degrees of freedom used in the final denominator.
    keep_attrs : bool or None
        Whether to preserve xarray attributes.
    std : bool
        If True, return standard deviation; otherwise return variance.
    kwargs : dict of str to Any
        Additional keyword arguments forwarded to xarray reductions.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Global variance or standard deviation, identical on every MPI rank.
    """
    block, partition_dim, _ = partition_for_reduction(obj, dim)
    method = block.std if std else block.var
    if partition_dim is None:
        return method(
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    nondistributed_template: xr.DataArray | xr.Dataset | None = None
    if isinstance(block, xr.Dataset):
        nondistributed_template = load_data(
            method(
                dim=dim,
                skipna=skipna,
                ddof=ddof,
                keep_attrs=keep_attrs,
                **kwargs,
            )
        )

    local_mean = load_data(
        block.mean(dim=dim, skipna=skipna, keep_attrs=False, **kwargs)
    )
    local_var = load_data(
        block.var(
            dim=dim,
            skipna=skipna,
            ddof=0,
            keep_attrs=keep_attrs,
            **kwargs,
        )
    )
    local_count = load_data(block.count(dim=dim, keep_attrs=False))
    payload = mpi.world.allgather((local_mean, local_var, local_count))
    means = [cast("xr.DataArray | xr.Dataset", item[0]) for item in payload]
    variances = [cast("xr.DataArray | xr.Dataset", item[1]) for item in payload]
    counts = [cast("xr.DataArray | xr.Dataset", item[2]) for item in payload]
    result = global_variance(
        means,
        variances,
        counts,
        obj,
        partition_dim,
        ddof,
        skipna=skipna,
        std=std,
    )
    return restore_nondistributed(result, nondistributed_template, obj, partition_dim)


class MPIAccessor:
    """
    Collective operations bound to a DataArray or Dataset.

    Parameters
    ----------
    xarray_obj : xarray.DataArray or xarray.Dataset
        Logical xarray object reduced transparently across ``MPI_COMM_WORLD``.
    """

    __slots__ = ("_obj",)

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset) -> None:
        self._obj = xarray_obj

    def __repr__(self) -> str:
        kind = type(self._obj).__name__
        return f"<xgeo mpi accessor on {kind}>"

    # -- admin & environment ----------------------------------------------
    def available(self) -> bool:
        """
        Return whether the native MPI runtime can be loaded and initialized.

        Returns
        -------
        bool
            True if MPI runtime is available, False otherwise.
        """
        return mpi.world.available()

    def launcher_size(self) -> int:
        """
        Return the world size advertised by the process launcher.

        Returns
        -------
        int
            World size retrieved from launcher metadata.
        """
        return mpi.world.launcher_size()

    def rank(self) -> int:
        """
        Return this process's rank in ``MPI_COMM_WORLD``.

        Returns
        -------
        int
            Rank of the current process.
        """
        return mpi.world.rank()

    def size(self) -> int:
        """
        Return the number of ranks in ``MPI_COMM_WORLD``.

        Returns
        -------
        int
            Total number of processes in the world.
        """
        return mpi.world.size()

    def is_root(self, root: int = 0) -> bool:
        """
        Return whether this process has rank ``root``.

        Parameters
        ----------
        root : int, default 0
            The rank to check against.

        Returns
        -------
        bool
            True if the process matches the root rank, False otherwise.
        """
        return mpi.world.is_root(root)

    def abort(self, code: int = 1) -> None:
        """
        Abort all ranks in ``MPI_COMM_WORLD`` with a process exit code.

        Parameters
        ----------
        code : int, default 1
            Exit code to return to the process launcher.
        """
        mpi.world.abort(code)

    def finalize(self) -> None:
        """
        Finalize MPI when initialized by the shared world coordinator.
        """
        mpi.world.finalize()

    # -- synchronization --------------------------------------------------
    def barrier(self) -> xr.DataArray | xr.Dataset:
        """
        Wait for every rank, then return the bound object unchanged.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The local object, allowing the call to sit inside a method chain.
        """
        mpi.world.barrier()
        return self._obj

    def consensus(self, ok: bool | None = None) -> bool:
        """
        Return True only when every rank contributes a true value.

        Parameters
        ----------
        ok : bool, optional
            An explicit boolean value to contribute. If not provided,
            the bound xarray object is implicitly reduced to a boolean.

        Returns
        -------
        bool
            True if all ranks evaluate to True, False otherwise.
        """
        if ok is None:
            if isinstance(self._obj, xr.Dataset):
                ok = all(bool(da.all().item()) for da in self._obj.data_vars.values())
            else:
                ok = bool(self._obj.all().item())
        return mpi.world.consensus(ok)

    # -- data movement ----------------------------------------------------
    def bcast(self, root: int = 0) -> xr.DataArray | xr.Dataset:
        """
        Replace this object with the one held by ``root``.

        Parameters
        ----------
        root : int, default 0
            Source rank.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The object held by ``root``, distributed to every rank.
        """
        return mpi.world.bcast(self._obj, root=root)

    def gather(self, root: int = 0) -> list[xr.DataArray | xr.Dataset] | None:
        """
        Collect every rank's object onto ``root``.

        Parameters
        ----------
        root : int, default 0
            Destination rank.

        Returns
        -------
        list of xarray.DataArray or xarray.Dataset, or None
            Objects from all ranks in rank order on ``root``, ``None``
            elsewhere.

        Notes
        -----
        The whole distributed dataset lands in one process, so this is a
        memory cliff on anything large. Use :meth:`to_netcdf` to write a
        distributed dataset without gathering it.
        """
        return mpi.world.gather(self._obj, root=root)

    def allgather(self) -> list[xr.DataArray | xr.Dataset]:
        """
        Collect every rank's object onto every rank, in rank order.

        Returns
        -------
        list of xarray.DataArray or xarray.Dataset
            Objects from all ranks in rank order, identical everywhere.
        """
        return mpi.world.allgather(self._obj)

    def scatter(self, dim: str, root: int = 0) -> xr.DataArray | xr.Dataset:
        """
        Split ``root``'s object along ``dim`` and keep this rank's block.

        Parameters
        ----------
        dim : str
            Dimension to partition.
        root : int, default 0
            Rank holding the global object.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            This rank's contiguous block.

        Notes
        -----
        Blocks are contiguous and the remainder falls on the leading ranks,
        which is the layout :meth:`to_netcdf` requires: it recovers each
        rank's file offset from an all-gather of the local lengths, so a
        strided split would scatter a rank's records across the whole file.
        """
        if not isinstance(dim, str) or not dim:
            raise TypeError("dim must be a non-empty string.")

        obj = mpi.world.bcast(self._obj, root=root)
        if dim not in obj.sizes:
            raise KeyError(f"Dimension {dim!r} is not present in the dataset.")
        start, stop = mpi.world.partition(int(obj.sizes[dim]))
        return obj.isel({dim: slice(start, stop)})

    def concat(self, dim: str, root: int | None = None) -> xr.DataArray | xr.Dataset:
        """
        Join every rank's object along ``dim`` in rank order.

        Parameters
        ----------
        dim : str
            Dimension to concatenate along, normally the partitioned one.
        root : int or None, optional
            Rank the result is assembled on. If None, assemble on every rank.
            Default is None.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The reassembled global object, or the local object unchanged on
            non-root ranks when ``root`` is given.

        Notes
        -----
        This materializes the global object in one process's memory per rank
        that receives it, which is what the parallel writer exists to avoid.
        It is meant for a final reduced result, not for the partitioned data.
        """
        if not isinstance(dim, str) or not dim:
            raise TypeError("dim must be a non-empty string.")

        if root is None:
            parts = mpi.world.allgather(self._obj)
        else:
            gathered = mpi.world.gather(self._obj, root=root)
            if gathered is None:
                return self._obj
            parts = gathered
        return xr.concat(parts, dim=dim)

    def partition(self, dim: str) -> xr.DataArray | xr.Dataset:
        """
        Keep only this rank's contiguous block along ``dim``.

        Parameters
        ----------
        dim : str
            Dimension to partition.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            This rank's block.

        Notes
        -----
        Unlike :meth:`scatter`, nothing is communicated. Every rank is assumed
        to already hold, or be able to open lazily, the same global object,
        and simply selects its own slice. That is the cheaper pattern when the
        data comes from a shared filesystem.
        """
        if not isinstance(dim, str) or not dim:
            raise TypeError("dim must be a non-empty string.")
        if dim not in self._obj.sizes:
            raise KeyError(f"Dimension {dim!r} is not present in the dataset.")
        start, stop = mpi.world.partition(int(self._obj.sizes[dim]))
        return self._obj.isel({dim: slice(start, stop)})

    def split(self, dim: str) -> xr.DataArray | xr.Dataset:
        """
        Keep only this rank's contiguous block along ``dim``.

        Alias for :meth:`partition`.

        Parameters
        ----------
        dim : str
            Dimension to partition.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            This rank's block.
        """
        return self.partition(dim)

    # -- reductions -------------------------------------------------------
    def sum(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Reduce with ``xarray.sum`` while partitioning a reduced dimension."""
        return associative_reduction(
            self._obj,
            "sum",
            dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            kwargs=kwargs,
        )

    def prod(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Reduce with ``xarray.prod`` while partitioning a reduced dimension."""
        return associative_reduction(
            self._obj,
            "prod",
            dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            kwargs=kwargs,
        )

    def min(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Reduce with ``xarray.min`` while partitioning a reduced dimension."""
        return associative_reduction(
            self._obj, "min", dim, skipna=skipna, keep_attrs=keep_attrs, kwargs=kwargs
        )

    def max(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Reduce with ``xarray.max`` while partitioning a reduced dimension."""
        return associative_reduction(
            self._obj, "max", dim, skipna=skipna, keep_attrs=keep_attrs, kwargs=kwargs
        )

    def any(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Reduce with ``xarray.any`` while partitioning a reduced dimension."""
        return associative_reduction(
            self._obj, "any", dim, keep_attrs=keep_attrs, kwargs=kwargs
        )

    def all(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Reduce with ``xarray.all`` while partitioning a reduced dimension."""
        return associative_reduction(
            self._obj, "all", dim, keep_attrs=keep_attrs, kwargs=kwargs
        )

    def count(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Count non-missing values over a transparently partitioned dimension."""
        block, partition_dim, _ = partition_for_reduction(self._obj, dim)
        if partition_dim is None:
            return block.count(dim=dim, keep_attrs=keep_attrs, **kwargs)

        local = load_data(block.count(dim=dim, keep_attrs=keep_attrs, **kwargs))
        parts = [
            cast("xr.DataArray | xr.Dataset", part)
            for part in mpi.world.allgather(local)
        ]
        return combine_counts(parts, self._obj, partition_dim)

    def mean(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Compute a count-weighted MPI mean with xarray reduction semantics."""
        if isinstance(self._obj, xr.DataArray) and np.issubdtype(
            self._obj.dtype, np.datetime64
        ):
            return self._obj.mean(
                dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs
            )

        block, partition_dim, _ = partition_for_reduction(self._obj, dim)
        if partition_dim is None:
            return block.mean(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

        local_mean = load_data(
            block.mean(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
        )
        local_count = load_data(block.count(dim=dim, keep_attrs=False))
        payload = mpi.world.allgather((local_mean, local_count))
        means = [cast("xr.DataArray | xr.Dataset", item[0]) for item in payload]
        counts = [cast("xr.DataArray | xr.Dataset", item[1]) for item in payload]
        return global_mean(
            means,
            counts,
            self._obj,
            partition_dim,
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            kwargs=kwargs,
        )

    def var(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Compute variance from mergeable per-rank sufficient statistics."""
        return variance_reduction(
            self._obj,
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            std=False,
            kwargs=kwargs,
        )

    def std(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Compute standard deviation from mergeable per-rank sufficient statistics."""
        return variance_reduction(
            self._obj,
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            std=True,
            kwargs=kwargs,
        )


__all__ = ["MPIAccessor"]
