"""Plan distributed reductions and execute MPI collectives."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .cartesian import get_cartesian_topology
from .chunks import get_chunk_bounds, get_effective_chunk_size, prune_chunk_info
from .common import (
    CHECK_COLLECTIVE_AGREEMENT,
    MPI_REDUCIBLE_KINDS,
    PlanEntry,
    mpi_representable,
    op_name,
    partial_dtype,
)
from .meta import choose_partition_dim, mpp_update_meta, strip_mpi_meta
from .mpp import _mpp_reduce, mpp_reduce_scatter


def normalize_dim(
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
) -> tuple[Any, tuple[Hashable, ...]]:
    """Normalize a reduction dimension specification."""
    if not isinstance(value, (xr.DataArray, xr.Dataset)):
        raise TypeError("MPI xarray operations require an xarray DataArray or Dataset.")
    if dim is None or dim is ...:
        return dim, tuple(value.dims)
    if isinstance(dim, str):
        return dim, (dim,)
    dims = tuple(dim)
    return dims, dims


def skipna_enabled(dtype: np.dtype[Any], skipna: bool | None) -> bool:
    """Return the effective dtype-aware ``skipna`` setting."""
    if skipna is not None:
        return skipna
    return dtype.kind in "fc"


def _check_reducible(dtype: np.dtype[Any], operation: str) -> None:
    """Validate that a dtype supports the requested MPI reduction."""
    if operation in ("any", "all"):
        return
    if dtype.kind not in MPI_REDUCIBLE_KINDS:
        raise TypeError(f"No predefined MPI datatype for {dtype}.")
    if not mpi_representable(dtype.str):
        # Reject dtypes without predefined MPI types before entering a collective.
        raise TypeError(f"No predefined MPI datatype for {dtype}.")
    if operation in ("min", "max") and dtype.kind == "c":
        name = "minimum" if operation == "min" else "maximum"
        raise TypeError(f"MPI {name} is not defined for complex xarray data.")


def local_reduction_meta(
    meta: Mapping[str, Any] | None,
    dims: tuple[Hashable, ...],
    *,
    partition_dim: Hashable | Literal["auto"] | None,
) -> Mapping[str, Any] | None:
    """Return metadata when a reduction remains rank-local."""
    if meta is None or any(dim in dims for dim in meta["dims"]):
        return None
    if partition_dim not in (None, "auto"):
        raise ValueError("New partition_dim requires reducing the active partition dimension.")
    return meta


def finish_local_reduction(
    result: xr.Dataset | xr.DataArray, *, old_meta: Mapping[str, Any]
) -> xr.Dataset | xr.DataArray:
    """Restore metadata after a rank-local reduction."""
    dims = tuple(dim for dim in old_meta["dims"] if dim in result.dims)
    if not dims:
        return strip_mpi_meta(result)
    mpp_update_meta(
        result,
        dim=dims,
        global_size={dim: int(old_meta["global_sizes"][dim]) for dim in dims},
        start={dim: int(old_meta["starts"][dim]) for dim in dims},
        stop={dim: int(old_meta["stops"][dim]) for dim in dims},
        chunk_info=prune_chunk_info(old_meta["chunk_info"], result),
        cart=old_meta.get("cart"),
    )
    return result


def _agree(mpi_context: MPIContext, signature: tuple[Any, ...]) -> None:
    """Verify that all ranks entered the same reduction plan."""
    if not CHECK_COLLECTIVE_AGREEMENT or mpi_context.comm.size == 1:
        return
    digest = hashlib.blake2b(repr(signature).encode(), digest_size=16).digest()
    mpi_context.raise_if_error(
        None,
        "MPI xarray reduction planning",
        signature=("xarray_reduction_plan", digest),
    )


def mpp_reduction_plan(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dims: tuple[Hashable, ...],
    meta: Mapping[str, Any] | None,
    *,
    operation: str,
) -> tuple[PlanEntry, ...]:
    """Build and validate the rank-independent reduction plan."""
    if isinstance(value, xr.DataArray):
        items: tuple[tuple[Hashable, xr.DataArray], ...] = ((value.name, value),)
    else:
        items = tuple((name, value[name]) for name in value.data_vars)

    partition_dims: tuple[Hashable, ...] = () if meta is None else meta["dims"]
    grid_shape_by_dim: dict[Hashable, int] = {}
    if meta is not None and "cart" in meta:
        grid_shape_by_dim = dict(
            zip(meta["dims"], meta["cart"]["grid_shape"], strict=True)
        )

    entries = []
    for name, variable in items:
        variable_dims = tuple(dim for dim in dims if dim in variable.dims)
        if variable_dims:
            _check_reducible(variable.dtype, operation)

        owned = tuple(dim for dim in partition_dims if dim in variable.dims)
        replicated = tuple(dim for dim in partition_dims if dim not in variable.dims)
        reduced = tuple(dim for dim in variable_dims if dim in owned)
        comm_axes = (
            frozenset(reduced) | frozenset(replicated) if reduced else frozenset()
        )
        replica_count = (
            math.prod(grid_shape_by_dim.get(dim, 1) for dim in replicated)
            if reduced and replicated
            else 1
        )
        if replica_count != 1 and operation == "prod":
            # Replicated products cannot be deduplicated reliably because inversion
            # requires an n-th root.
            raise NotImplementedError(
                "Product reduction cannot remove replicated dimensions "
                + f"{tuple(str(d) for d in replicated)!r}."
            )

        entries.append(
            PlanEntry(
                name=name,
                dims=variable_dims,
                distributed=bool(comm_axes),
                dtype=variable.dtype,
                shape=tuple(
                    (
                        str(dim),
                        # Use global sizes for surviving partition dimensions because
                        # local extents may differ by rank.
                        int(
                            meta["global_sizes"][dim]
                            if meta is not None and dim in partition_dims
                            else value.sizes[dim]
                        ),
                    )
                    for dim in variable.dims
                    if dim not in variable_dims
                ),
                comm_axes=comm_axes,
                replica_count=int(replica_count),
            )
        )

    plan = tuple(entries)
    _agree(
        mpi_context,
        (
            operation,
            tuple(str(dim) for dim in dims),
            tuple(
                (
                    str(entry.name),
                    tuple(str(dim) for dim in entry.dims),
                    entry.distributed,
                    str(entry.dtype),
                    entry.shape,
                    tuple(sorted(str(d) for d in entry.comm_axes)),
                    entry.replica_count,
                )
                for entry in plan
            ),
        ),
    )
    return plan


def mpp_resolve_comm(
    mpi_context: MPIContext,
    meta: Mapping[str, Any] | None,
    comm_axes: Iterable[Hashable],
) -> MPI.Comm:
    """Return the communicator a plan entry's collective should use."""
    axes = frozenset(comm_axes)
    if meta is None or not axes or "cart" not in meta or len(meta["dims"]) <= 1:
        return mpi_context.comm
    topology = get_cartesian_topology(
        mpi_context.comm, meta["dims"], meta["global_sizes"]
    )
    return topology.sub_comm(axes)


def guarded(function: Any) -> tuple[Any, BaseException | None]:
    """Run a local operation and defer any exception for synchronization."""
    try:
        return function(), None
    except BaseException as exc:
        return None, exc


def mpp_comm_reduce(
    mpi_context: MPIContext,
    value: xr.DataArray | None,
    op: MPI.Op,
    *,
    expect_dtype: np.dtype[Any] | None = None,
    error: BaseException | None = None,
    phase: str = "MPI xarray reduction buffer preparation",
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, Sequence[int]] | None = None,
) -> xr.DataArray:
    """Reduce a validated DataArray buffer across ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    value : xarray.DataArray or None
        Collective buffer.
    op : mpi4py.MPI.Op
        Reduction operation.
    expect_dtype : numpy.dtype, optional
        Expected buffer dtype.
    error : BaseException, optional
        Deferred local error.
    phase : str
        Collective diagnostic label.
    comm : mpi4py.MPI.Comm, optional
        Reduction communicator.
    replica_count : int, default 1
        Number of duplicate replicas included in a SUM.
    scatter : tuple[Hashable, Sequence[int]], optional
        Target dimension and per-rank counts for ``Reduce_scatter``.

    Returns
    -------
    xarray.DataArray
        Globally reduced result or this rank's scattered slice.
    """
    send: np.ndarray[Any, Any] | None = None
    if error is None:
        try:
            if value is None:
                raise AssertionError("MPI xarray reduction buffer is missing.")
            send = np.asarray(value.values)
            if expect_dtype is not None and send.dtype != np.dtype(expect_dtype):
                send = send.astype(expect_dtype)
            if not send.flags.c_contiguous:
                send = np.ascontiguousarray(send)
            if send.dtype.kind not in MPI_REDUCIBLE_KINDS:
                raise TypeError(f"No predefined MPI datatype for {send.dtype}.")
            if not mpi_representable(send.dtype.str):
                raise TypeError(f"No predefined MPI datatype for {send.dtype}.")
        except BaseException as exc:
            error = exc
            send = None

    signature = (
        None
        if send is None
        else (
            op_name(op),
            send.dtype.str,
            tuple(int(length) for length in send.shape),
        )
    )
    mpi_context.raise_if_error(error, phase, signature, comm=comm)
    if send is None or value is None:
        raise AssertionError("MPI xarray reduction buffer is missing.")

    resolved_comm = comm if comm is not None else mpi_context.comm
    if scatter is not None:
        target, counts = scatter
        axis = value.dims.index(target)
        recv = mpp_reduce_scatter(send, op, resolved_comm, counts, axis=axis)
        start = sum(counts[: resolved_comm.rank])
        stop = start + counts[resolved_comm.rank]
        result = value.isel({target: slice(start, stop)}).copy(data=recv)
    else:
        recv = _mpp_reduce(send, op, resolved_comm)
        result = value.copy(data=recv)
    if replica_count != 1 and op == MPI.SUM:
        # Divide replicated SUM results by ``replica_count``; duplicate contributions
        # make the factor exact.
        if result.dtype.kind in "iu":
            result = result // replica_count
        else:
            result = result / replica_count
    return result


def mpp_count_valid_values(
    mpi_context: MPIContext,
    value: xr.DataArray,
    dims: tuple[Hashable, ...],
    *,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, Sequence[int]] | None = None,
) -> xr.DataArray:
    """Count valid values globally across the requested dimensions."""
    count: xr.DataArray | None = None
    error: BaseException | None = None
    try:
        count = value.count(dim=dims, keep_attrs=False)
    except BaseException as exc:
        error = exc
    return mpp_comm_reduce(
        mpi_context,
        count,
        MPI.SUM,
        expect_dtype=partial_dtype(value.dtype.str, "count", None),
        error=error,
        phase="MPI xarray count reduction",
        comm=comm,
        replica_count=replica_count,
        scatter=scatter,
    )


def dataset_result(
    value: xr.Dataset,
    dims: tuple[Hashable, ...],
    variables: Mapping[Hashable, xr.DataArray],
) -> xr.Dataset:
    """Rebuild a Dataset from reduced data variables."""
    reduced = set(dims)
    coords = {
        name: coord
        for name, coord in value.coords.items()
        if not reduced & set(coord.dims)
    }
    return xr.Dataset(dict(variables), coords=coords, attrs=dict(value.attrs))


def repartition_candidates(plan: tuple[PlanEntry, ...]) -> frozenset[Hashable]:
    """Return dimensions eligible for post-reduction repartition."""
    return frozenset(
        dim for entry in plan if entry.distributed for dim, _ in entry.shape
    )


def mpp_scatter_target(
    *,
    old_meta: Mapping[str, Any] | None,
    dims: tuple[Hashable, ...],
    partition_dim: Hashable | Literal["auto"] | None,
    auto_candidates: frozenset[Hashable],
    result_sizes: Mapping[Hashable, int],
    comm: MPI.Comm,
    replica_count: int,
) -> tuple[Hashable, list[int]] | None:
    """Choose a target for ``Reduce_scatter`` after a reduction.

    Parameters
    ----------
    old_meta : mapping or None
        Input distribution metadata.
    dims : tuple[Hashable, ...]
        Reduced dimensions.
    partition_dim : Hashable, {"auto"}, or None
        Requested output partition dimension.
    auto_candidates : frozenset[Hashable]
        Dimensions eligible for automatic partitioning.
    result_sizes : mapping[Hashable, int]
        Global result sizes.
    comm : mpi4py.MPI.Comm
        Candidate communicator.
    replica_count : int
        Number of replicated copies in the communicator.

    Returns
    -------
    tuple[Hashable, list[int]] or None
        Target dimension and per-rank counts, or None when scattering is unsuitable.
    """
    if (
        partition_dim != "auto"
        or old_meta is None
        or replica_count != 1
        or comm.size <= 1
        or any(dim not in dims for dim in old_meta["dims"])
    ):
        return None
    sizes = {
        dim: int(length)
        for dim, length in result_sizes.items()
        if dim in auto_candidates
    }
    if not any(length > 1 for length in sizes.values()):
        return None
    target = choose_partition_dim(sizes, comm.size, rank=comm.rank)
    length = sizes[target]
    chunk_size = get_effective_chunk_size(length, None, comm.size)
    counts = [
        stop - start
        for start, stop in (
            get_chunk_bounds(length, chunk_size, rank, comm.size)
            for rank in range(comm.size)
        )
    ]
    return target, counts


def mpp_finish_scatter(
    result: xr.Dataset | xr.DataArray,
    *,
    target: Hashable,
    counts: Sequence[int],
    comm: MPI.Comm,
) -> xr.Dataset | xr.DataArray:
    """Attach distribution metadata after ``Reduce_scatter``.

    Parameters
    ----------
    result : xarray.Dataset or xarray.DataArray
        Rank-local scattered result.
    target : Hashable
        Scattered dimension.
    counts : sequence of int
        Per-rank lengths along ``target``.
    comm : mpi4py.MPI.Comm
        Scatter communicator.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Result with updated MPI metadata.
    """
    rank = comm.rank
    start = sum(counts[:rank])
    stop = start + counts[rank]
    chunk_info = {
        str(dim): get_effective_chunk_size(int(length), None, comm.size)
        for dim, length in result.sizes.items()
    }
    mpp_update_meta(
        result,
        dim=target,
        global_size=sum(counts),
        start=start,
        stop=stop,
        chunk_info=chunk_info,
    )
    return result


def mpp_scatter_replicated_slice(
    variable: xr.DataArray, target: Hashable, start: int, stop: int
) -> xr.DataArray:
    """Slice a replicated variable to a scattered target range.

    Parameters
    ----------
    variable : xarray.DataArray
        Unreduced replicated variable.
    target : Hashable
        Scattered dimension.
    start, stop : int
        Rank-local half-open target bounds.

    Returns
    -------
    xarray.DataArray
        Matching local slice, or ``variable`` unchanged if ``target`` is absent.
    """
    return (
        variable.isel({target: slice(start, stop)})
        if target in variable.dims
        else variable
    )


def mpp_plan_scatter_target(
    mpi_context: MPIContext,
    old_meta: Mapping[str, Any] | None,
    dims: tuple[Hashable, ...],
    partition_dim: Hashable | Literal["auto"] | None,
    reduce_plan: tuple[PlanEntry, ...],
) -> tuple[Hashable, list[int], MPI.Comm] | None:
    """Choose one shared scatter target for a reduction plan.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    old_meta : mapping or None
        Input distribution metadata.
    dims : tuple[Hashable, ...]
        Reduced dimensions.
    partition_dim : Hashable, {"auto"}, or None
        Requested output partition dimension.
    reduce_plan : tuple[PlanEntry, ...]
        Per-variable reduction plan.

    Returns
    -------
    tuple[Hashable, list[int], mpi4py.MPI.Comm] or None
        Shared target, counts, and communicator when scatter is valid.
    """
    distributed_entries = [e for e in reduce_plan if e.dims and e.distributed]
    if not distributed_entries:
        return None
    combine_comms = {
        entry.name: mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        for entry in distributed_entries
    }
    if len({c.size for c in combine_comms.values()}) != 1:
        return None
    if any(entry.replica_count != 1 for entry in distributed_entries):
        return None
    comm = next(iter(combine_comms.values()))
    result_sizes: dict[Hashable, int] = {}
    for entry in reduce_plan:
        result_sizes.update(dict(entry.shape))
    target = mpp_scatter_target(
        old_meta=old_meta,
        dims=dims,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(reduce_plan),
        result_sizes=result_sizes,
        comm=comm,
        replica_count=1,
    )
    if target is None:
        return None
    return (*target, comm)


def mpp_finish(
    mpi_context: MPIContext,
    result: xr.Dataset | xr.DataArray,
    *,
    old_meta: Mapping[str, Any] | None,
    partition_dim: Hashable | Literal["auto"] | None,
    auto_candidates: frozenset[Hashable],
) -> xr.Dataset | xr.DataArray:
    """Finalize metadata and optional repartition after a reduction.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Finalized distributed result. Fully replicated (``.meta`` is
        None) if every previous partition dimension was reduced away.
        Otherwise, metadata is reattached for whichever dimension(s)
        survive, with no duplicated ownership: exactly one rank per
        distinct surviving range keeps the real result; every other
        rank that shared that range before the reduction is left with
        a genuinely empty (``start == stop``) slice instead of a
        redundant copy.

    """
    result = strip_mpi_meta(result)
    old_dims: tuple[Hashable, ...] = () if old_meta is None else old_meta["dims"]
    remaining_dims = tuple(dim for dim in old_dims if dim in result.dims)
    partition_removed = old_meta is not None and not remaining_dims

    if partition_dim is None:
        return result

    if partition_dim == "auto" and remaining_dims:
        # Reattach surviving partition axes directly and rebuild reduced Cartesian
        # topology lazily.
        assert old_meta is not None  # remaining_dims is empty otherwise
        cart = old_meta.get("cart") if len(remaining_dims) == len(old_dims) else None
        start = {dim: int(old_meta["starts"][dim]) for dim in remaining_dims}
        stop = {dim: int(old_meta["stops"][dim]) for dim in remaining_dims}

        reduced_dims = frozenset(old_dims) - frozenset(remaining_dims)
        if reduced_dims:
            # Keep one copy per surviving range and mark subgroup replicas empty to
            # preserve non-overlapping ownership.
            comm = mpp_resolve_comm(mpi_context, old_meta, reduced_dims)
            if comm.rank != 0:
                empty_dim = remaining_dims[0]
                result = result.isel({empty_dim: slice(0, 0)})
                stop[empty_dim] = start[empty_dim]

        mpp_update_meta(
            result,
            dim=remaining_dims,
            global_size={
                dim: int(old_meta["global_sizes"][dim]) for dim in remaining_dims
            },
            start=start,
            stop=stop,
            chunk_info=prune_chunk_info(old_meta["chunk_info"], result),
            cart=cart,
        )
        return result

    target = partition_dim
    if partition_dim == "auto":
        if not partition_removed:
            return result
        sizes = {
            dim: length
            for dim, length in result.sizes.items()
            if dim in auto_candidates
        }
        if not any(int(length) > 1 for length in sizes.values()):
            return result
        target = choose_partition_dim(
            sizes, mpi_context.comm.size, rank=mpi_context.comm.rank
        )
    elif partition_dim not in auto_candidates:
        raise ValueError(f"partition_dim={partition_dim!r} was not reduced collectively.")

    chunk_info = (
        prune_chunk_info(old_meta["chunk_info"], result) if old_meta is not None else {}
    )
    from .io import mpp_repartition

    return mpp_repartition(mpi_context, result, target, chunk_info=chunk_info)
