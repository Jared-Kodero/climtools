"""Plan distributed reductions and execute MPI collectives."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Iterable, Mapping
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from mpi4py import MPI

import xarray as xr

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .cartesian import get_cartesian_topology
from .chunks import prune_chunk_info
from .common import (
    CHECK_COLLECTIVE_AGREEMENT,
    MPI_REDUCIBLE_KINDS,
    PlanEntry,
    mpi_representable,
    op_name,
    partial_dtype,
)
from .meta import choose_partition_dim, mpp_update_meta, strip_mpi_meta
from .mpp import _mpp_reduce


def normalize_dim(
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
) -> tuple[Any, tuple[Hashable, ...]]:
    """Normalize a reduction dimension specification.

    Parameters
    ----------
    value : xr.Dataset | xr.DataArray
        Distributed xarray object.
    dim : str | Iterable[Hashable] | EllipsisType | None
        Dimension to operate on.
    Returns
    -------
    tuple[Any, tuple[Hashable, ...]]
        Normalized local dimension argument and dimension tuple.
    """
    if not isinstance(value, (xr.DataArray, xr.Dataset)):
        raise TypeError("MPI xarray operations require an xarray DataArray or Dataset.")
    if dim is None or dim is ...:
        return dim, tuple(value.dims)
    if isinstance(dim, str):
        return dim, (dim,)
    dims = tuple(dim)
    return dims, dims


def skipna_enabled(dtype: np.dtype[Any], skipna: bool | None) -> bool:
    """Return the effective dtype-aware ``skipna`` setting.

    Parameters
    ----------
    dtype : np.dtype[Any]
        NumPy dtype.
    skipna : bool | None
        Whether to ignore missing values.
    Returns
    -------
    bool
        Effective missing-value policy.
    """
    if skipna is not None:
        return skipna
    return dtype.kind in "fc"


def _check_reducible(dtype: np.dtype[Any], operation: str) -> None:
    """Validate that a dtype supports the requested MPI reduction."""
    if operation in ("any", "all"):
        return
    if dtype.kind not in MPI_REDUCIBLE_KINDS:
        raise TypeError(f"Unsupported MPI xarray dtype: {dtype}.")
    if not mpi_representable(dtype.str):
        # float16 and long double have a reducible NumPy kind but no
        # predefined MPI datatype. Rejecting them here raises on every
        # rank before any collective, instead of failing inside
        # Allreduce with MPI_ERR_TYPE once buffers are already posted.
        raise TypeError(
            f"Unsupported MPI xarray dtype: {dtype}. "
            + "No predefined MPI datatype represents it."
        )
    if operation in ("min", "max") and dtype.kind == "c":
        name = "minimum" if operation == "min" else "maximum"
        raise TypeError(f"MPI {name} is not defined for complex xarray data.")


def local_reduction_meta(
    meta: Mapping[str, Any] | None,
    dims: tuple[Hashable, ...],
    *,
    partition_dim: Hashable | Literal["auto"] | None,
) -> Mapping[str, Any] | None:
    """Return metadata when a reduction remains rank-local.

    Parameters
    ----------
    meta : Mapping[str, Any] | None
        MPI distribution metadata.
    dims : tuple[Hashable, ...]
        Dimensions to operate on.
    partition_dim : Hashable | Literal['auto'] | None
        Partition dimension to use for the result.
    Returns
    -------
    Mapping[str, Any] | None
        Metadata for a rank-local reduction, if applicable.
    """
    if meta is None or any(dim in dims for dim in meta["dims"]):
        return None
    if partition_dim not in (None, "auto"):
        raise ValueError(
            "partition_dim can name a new dimension only after the active "
            + "partition dimension has been reduced away."
        )
    return meta


def finish_local_reduction(
    result: xr.Dataset | xr.DataArray, *, old_meta: Mapping[str, Any]
) -> xr.Dataset | xr.DataArray:
    """Restore metadata after a rank-local reduction.

    Parameters
    ----------
    result : xr.Dataset | xr.DataArray
        Operation result.
    old_meta : Mapping[str, Any]
        Existing MPI distribution metadata.
    Returns
    -------
    xr.Dataset | xr.DataArray
        Reduction result with restored metadata.
    """
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
    """Build and validate the rank-independent reduction plan.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xr.Dataset | xr.DataArray
        Distributed xarray object.
    dims : tuple[Hashable, ...]
        Dimensions to operate on.
    meta : Mapping[str, Any] | None
        MPI distribution metadata.
    operation : str
        Operation name used for planning.
    Returns
    -------
    tuple[PlanEntry, ...]
        Validated reduction plan entries.
    """
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
            # A duplicated contribution inflates a sum by an exact,
            # exactly-undoable integer factor (divide it back out); it
            # inflates a product by raising it to the replica_count-th
            # power instead, which has no numerically reliable general
            # inverse (an n-th root is ill-defined for negative or
            # complex values and imprecise for float ones). Rather than
            # silently return a wrong answer, this combination is
            # explicitly unsupported for now.
            raise NotImplementedError(
                f"cannot yet reduce variable {name!r}: it is replicated "
                + f"along {tuple(str(d) for d in replicated)!r}, and "
                + "undoing a product's duplication has no exact general "
                + "inverse"
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
                        # A surviving dimension's *local* size is only
                        # rank-invariant (and therefore safe for the
                        # cross-rank agreement hash below) when it is
                        # not itself a partition dimension. Once more
                        # than one dimension can be partitioned, a
                        # surviving dimension can easily be a different
                        # partition axis than the one being reduced
                        # right now (e.g. reducing "lat" while "lon" is
                        # also partitioned) -- every rank then legally
                        # owns a different local "lon" extent, so its
                        # *global* size (identical everywhere) must be
                        # used instead.
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
    """Return the communicator a plan entry's collective should use.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    meta : Mapping[str, Any] | None
        MPI distribution metadata.
    comm_axes : Iterable[Hashable]
        Partition axes included in the communicator.
    Returns
    -------
    MPI.Comm
        Communicator for the requested partition axes.
    """
    axes = frozenset(comm_axes)
    if meta is None or not axes or "cart" not in meta or len(meta["dims"]) <= 1:
        return mpi_context.comm
    topology = get_cartesian_topology(
        mpi_context.comm, meta["dims"], meta["global_sizes"]
    )
    return topology.sub_comm(axes)


def guarded(function: Any) -> tuple[Any, BaseException | None]:
    """Run a local operation and defer any exception for synchronization.

    Parameters
    ----------
    function : Any
        Callable to execute.
    Returns
    -------
    tuple[Any, BaseException | None]
        Operation result and deferred exception.
    """
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
) -> xr.DataArray:
    """Combine a validated DataArray buffer across ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xr.DataArray | None
        Distributed DataArray buffer.
    op : MPI.Op
        MPI reduction operation.
    expect_dtype : np.dtype[Any] | None
        Expected dtype of the collective buffer.
    error : BaseException | None
        Deferred local exception, if any.
    phase : str
        Diagnostic label for the collective phase.
    comm : mpi4py.MPI.Comm, optional
        Communicator to reduce over.
    replica_count : int, optional
        Size of a replicated-axis subgroup folded into ``comm`` (see :attr:`~.common.PlanEntry.replica_count`).
    Returns
    -------
    xr.DataArray
        Globally reduced DataArray.
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
                raise TypeError(f"Unsupported MPI xarray dtype: {send.dtype}.")
            if not mpi_representable(send.dtype.str):
                raise TypeError(
                    f"Unsupported MPI xarray dtype: {send.dtype}. "
                    + "No predefined MPI datatype represents it."
                )
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

    recv = _mpp_reduce(send, op, comm if comm is not None else mpi_context.comm)
    result = value.copy(data=recv)
    if replica_count != 1 and op == MPI.SUM:
        # Every one of the replica_count duplicate copies contributed to
        # the raw sum above, so it is exactly replica_count times too
        # large. Integer dtypes divide exactly (each duplicate is a
        # bit-identical copy, so the raw sum is an exact multiple);
        # floating dtypes use true division for the same reason plain
        # division, not floor division, is correct for a value that
        # need not itself be an integer multiple of anything.
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
) -> xr.DataArray:
    """Count valid values globally across the requested dimensions.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xr.DataArray
        Distributed xarray object.
    dims : tuple[Hashable, ...]
        Dimensions to operate on.
    comm : MPI.Comm | None
        MPI communicator.
    replica_count : int
        Number of replicated contributions.
    Returns
    -------
    xr.DataArray
        Global valid-value counts.
    """
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
    )


def dataset_result(
    value: xr.Dataset,
    dims: tuple[Hashable, ...],
    variables: Mapping[Hashable, xr.DataArray],
) -> xr.Dataset:
    """Rebuild a Dataset from reduced data variables.

    Parameters
    ----------
    value : xr.Dataset
        Distributed xarray object.
    dims : tuple[Hashable, ...]
        Dimensions to operate on.
    variables : Mapping[Hashable, xr.DataArray]
        Reduced data variables.
    Returns
    -------
    xr.Dataset
        Dataset rebuilt from reduced variables.
    """
    reduced = set(dims)
    coords = {
        name: coord
        for name, coord in value.coords.items()
        if not reduced & set(coord.dims)
    }
    return xr.Dataset(dict(variables), coords=coords, attrs=dict(value.attrs))


def repartition_candidates(plan: tuple[PlanEntry, ...]) -> frozenset[Hashable]:
    """Return dimensions eligible for post-reduction repartition.

    Parameters
    ----------
    plan : tuple[PlanEntry, ...]
        Reduction plan entries.
    Returns
    -------
    frozenset[Hashable]
        Eligible post-reduction partition dimensions.
    """
    return frozenset(
        dim for entry in plan if entry.distributed for dim, _ in entry.shape
    )


def mpp_finish(
    mpi_context: MPIContext,
    result: xr.Dataset | xr.DataArray,
    *,
    old_meta: Mapping[str, Any] | None,
    partition_dim: Hashable | Literal["auto"] | None,
    auto_candidates: frozenset[Hashable],
) -> xr.Dataset | xr.DataArray:
    """Finalize metadata and optional repartition after a reduction.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    result : xr.Dataset | xr.DataArray
        Operation result.
    old_meta : Mapping[str, Any] | None
        Existing MPI distribution metadata.
    partition_dim : Hashable | Literal['auto'] | None
        Partition dimension to use for the result.
    auto_candidates : frozenset[Hashable]
        Dimensions eligible for automatic repartitioning.
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
        # At least one, but not every, previous partition dimension
        # survived the reduction (only possible with more than one
        # partition dimension to begin with -- with exactly one, this
        # is unreachable, since removing "the" partition dimension is
        # exactly what made this call require MPI communication in the
        # first place). Every rank already owns a valid, unmoved,
        # contiguous slice along whatever survived, so reattach
        # metadata for that surviving subset directly instead of
        # scattering through mpp_repartition() below. The Cartesian
        # topology descriptor is only carried forward when every one
        # of its axes survived unchanged; otherwise a fresh, smaller
        # topology (for just the surviving axes) is built lazily, on
        # demand, the next time a multi-axis collective needs one.
        assert old_meta is not None  # remaining_dims is empty otherwise
        cart = old_meta.get("cart") if len(remaining_dims) == len(old_dims) else None
        start = {dim: int(old_meta["starts"][dim]) for dim in remaining_dims}
        stop = {dim: int(old_meta["stops"][dim]) for dim in remaining_dims}

        reduced_dims = frozenset(old_dims) - frozenset(remaining_dims)
        if reduced_dims:
            # Every rank that shared this rank's position along every
            # surviving dimension but differed only along `reduced_dims`
            # computed its own local Allreduce over exactly that
            # sub-communicator, so every one of them already holds an
            # identical, individually-correct copy of this same
            # surviving-dimension range -- not a bug in the *values*,
            # but reattaching that same non-empty range to every one of
            # them here would claim duplicate, overlapping ownership,
            # violating the no-overlap invariant every other operation
            # in this package relies on (summing local sizes across
            # ranks to recover the global size, writing to disk without
            # double-counting, a later repartition/shuffle's own
            # position bookkeeping, ...). Keep the real data on exactly
            # one rank per distinct surviving-dimension range -- this
            # sub-communicator's own rank 0 -- and mark every other
            # member genuinely empty (start == stop) instead, exactly
            # like a rank :func:`~.chunks.get_balanced_bounds` already
            # leaves idle when a dimension is shorter than the rank
            # count. This is a metadata/ownership correction only: the
            # already-computed values are correct on every rank, so no
            # extra communication is needed to arrive at this, only to
            # discard the now-redundant copies' claim to ownership.
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
        raise ValueError(
            f"partition_dim={partition_dim!r} is not a dimension of any "
            + "variable that required an MPI collective in this reduction; "
            + "an untouched, replicated variable's own dimension cannot be "
            + "used as the new partition dimension."
        )

    chunk_info = (
        prune_chunk_info(old_meta["chunk_info"], result) if old_meta is not None else {}
    )
    from .io import mpp_repartition

    return mpp_repartition(mpi_context, result, target, chunk_info=chunk_info)
