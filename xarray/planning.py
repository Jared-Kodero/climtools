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
    """Return metadata when a reduction remains rank-local."""
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
    scatter : tuple[Hashable, Sequence[int]] | None, optional
        ``(dim, counts)`` from :func:`mpp_scatter_target`. When given, use
        ``Reduce_scatter`` (:func:`~.mpp.mpp_reduce_scatter`) along ``dim``
        instead of ``Allreduce``, so every rank keeps only its own
        ``counts[rank]``-length slice instead of materializing the full
        combined result -- worthwhile precisely when that slice is what
        the caller was going to keep anyway (see :func:`mpp_scatter_target`
        and :func:`mpp_finish_scatter`). ``sum(counts)`` must equal
        ``value.sizes[dim]``.

    Returns
    -------
    xr.DataArray
        Globally reduced DataArray, or -- when ``scatter`` is given --
        this rank's own slice of it along ``scatter[0]``.

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
    """Decide whether a reduction should scatter instead of Allreduce-then-slice.

    A full-reduction call site (:func:`mpp_finish` with
    ``partition_dim="auto"``) that removes every previous partition
    dimension normally ``Allreduce``s a full copy onto every rank, then
    slices it back down via :func:`~.io.mpp_repartition`. When the eventual
    placement is knowable in advance -- exactly this case -- that full
    materialization is avoidable: ``Reduce_scatter`` (see
    :func:`~.mpp.mpp_reduce_scatter`) gives each rank only the slice it was
    going to keep. This mirrors the same trade already made for groupby and
    resample reductions in ``groupby.py``'s ``can_scatter`` gate.

    Parameters
    ----------
    old_meta : mapping or None
        Distribution metadata of the value being reduced.
    dims : tuple of Hashable
        Dimensions being reduced over.
    partition_dim : Hashable or {"auto"} or None
        Caller's requested partition placement, as passed to :func:`mpp_finish`.
    auto_candidates : frozenset of Hashable
        Dimensions eligible for automatic post-reduction partitioning (see
        :func:`repartition_candidates`).
    result_sizes : mapping of Hashable to int
        Sizes of the reduced result, identical on every rank (only
        ``old_meta``'s partition dimension, not yet removed by this
        reduction, was ever distributed, so every other dimension is
        already fully present on every rank pre-reduction).
    comm : mpi4py.MPI.Comm
        Communicator the reduction's collective will run over.
    replica_count : int
        Replica-subgroup size folded into ``comm`` (see
        :attr:`~.common.PlanEntry.replica_count`). A replica subgroup keeps
        the Allreduce path: every member is supposed to end up with an
        identical answer, so scattering it apart would defeat that.

    Returns
    -------
    tuple[Hashable, list[int]] | None
        ``(target_dim, counts)`` -- ``counts[r]`` is rank ``r``'s share of
        ``target_dim``, computed the same way
        :func:`~.io.mpp_repartition` would for the same result -- or
        ``None`` when the Allreduce-then-:func:`mpp_finish` path should be
        used instead (an explicit ``partition_dim``, no previous partition
        dimension being removed, a replica subgroup, a single rank, or no
        eligible dimension left to split across ranks).

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
    """Attach distribution metadata after a ``Reduce_scatter``-based reduction.

    Companion to :func:`mpp_finish`'s ``partition_dim="auto"`` branch, for
    results produced via :func:`mpp_scatter_target` plus a scattering
    :func:`mpp_comm_reduce`/:func:`mpp_count_valid_values` call rather than
    a full ``Allreduce``. ``result`` already carries only this rank's slice
    along ``target``; routing it back through :func:`mpp_finish` would
    incorrectly auto-choose and re-slice a *second* time, this time from
    each rank's already-local size rather than the true global one.
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
    """Slice an untouched Dataset variable to match a scattered target's range.

    A ``Reduce_scatter``-based reduction (see :func:`mpp_scatter_target`)
    only shrinks the buffers of variables actually being reduced; a
    variable absent from the reduction's own ``dims`` (replicated,
    passed through unchanged) stays at its full, pre-reduction size along
    every dimension, including ``target``. Left unsliced, assembling the
    final Dataset from a mix of already-scattered and still-full-sized
    variables sharing ``target`` would make ``xr.Dataset``'s own
    coordinate-alignment silently reindex the smaller ones back up to
    the full extent, padding the gap with NaN, rather than raising --
    this keeps every variable a consistent size first.

    Returns ``variable`` sliced to ``[start, stop)`` along ``target`` when it
    has that dimension, otherwise unchanged.
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
    """Decide a single shared ``Reduce_scatter`` target for a reduction plan.

    Thin wrapper around :func:`mpp_scatter_target` for ``reductions.py``/
    ``statistics.py`` call sites that already hold a
    :func:`mpp_reduction_plan` result. Two things it adds over calling
    :func:`mpp_scatter_target` directly:

    - ``result_sizes`` is built from each entry's own ``shape`` field
      (the plan's own, already rank-agreed record of surviving global
      dimensions and lengths -- see :class:`~.common.PlanEntry`), never
      from a rank-local partial that might be ``None`` under a deferred
      local error. This keeps the scatter/no-scatter decision itself pure
      and rank-invariant; :func:`mpp_comm_reduce`'s own signature-based
      ``raise_if_error`` collective agreement, called downstream with
      ``scatter`` folded into the signature, is what actually catches any
      real cross-rank disagreement before a collective runs.
    - For a Dataset with more than one distributed variable, mirrors
      ``groupby.py``'s ``can_scatter`` gate: every variable needing
      cross-rank communication must resolve to a communicator of the same
      size, and none may be a replica subgroup, since a Dataset's
      variables are all sliced to the same length along the same target
      dimension together in one :func:`mpp_finish_scatter` call.

    Returns
    -------
    tuple[Hashable, list[int], MPI.Comm] | None
        ``(target_dim, counts, comm)`` ready for :func:`mpp_comm_reduce`'s
        ``scatter=`` argument and :func:`mpp_finish_scatter`, or ``None``
        when the Allreduce-then-:func:`mpp_finish` path should be used.

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
