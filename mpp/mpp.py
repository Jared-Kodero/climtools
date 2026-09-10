"""Rank-level communication primitives.

Mirrors FMS ``mpp/mpp.F90``: collectives and communicator handling that
know nothing about how an array is decomposed. Anything that reasons about
a decomposition lives in :mod:`~climtools.mpp.mpp_domains` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .mpp_domains import Domain


def mpp_reduce_scatter(
    local: np.ndarray[Any, Any],
    op: MPI.Op,
    comm: MPI.Comm,
    recvcounts: Sequence[int],
    *,
    axis: int = 0,
) -> np.ndarray[Any, Any]:
    """Reduce an array and retain each rank's contiguous slice.

    Parameters
    ----------
    local : numpy.ndarray
        Equal-shaped local reduction buffer on every rank.
    op : mpi4py.MPI.Op
        Reduction operator.
    comm : mpi4py.MPI.Comm
        Reduction communicator.
    recvcounts : sequence of int
        Elements retained by each rank along ``axis``.
    axis : int, default 0
        Axis split among ranks.

    Returns
    -------
    numpy.ndarray
        This rank's reduced slice.
    """
    moved = np.ascontiguousarray(np.moveaxis(local, axis, 0))
    per_slice = moved[0].size if moved.ndim > 1 else 1
    flat_counts = [c * per_slice for c in recvcounts]
    recvbuf = np.empty(flat_counts[comm.rank], dtype=moved.dtype)
    comm.Reduce_scatter(moved.reshape(-1), recvbuf, recvcounts=flat_counts, op=op)
    my_len = recvcounts[comm.rank]
    shape = (my_len, *moved.shape[1:]) if moved.ndim > 1 else (my_len,)
    return np.moveaxis(recvbuf.reshape(shape), 0, axis)


def _mpp_reduce(
    local: np.ndarray[Any, Any],
    op: MPI.Op,
    comm: MPI.Comm | None,
    domain: Domain | None = None,
) -> np.ndarray[Any, Any]:
    """Reduce rank-local arrays with an MPI reduction operator."""
    active_comm = (
        comm if comm is not None else (domain.comm if domain else MPI.COMM_WORLD)
    )
    recv = np.empty_like(local)
    active_comm.Allreduce(local, recv, op=op)
    return recv


def mpp_sum(
    local: np.ndarray[Any, Any],
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_sum``. Pass ``domain`` or ``comm``."""
    return _mpp_reduce(local, MPI.SUM, comm, domain)


def mpp_max(
    local: np.ndarray[Any, Any],
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_max``."""
    return _mpp_reduce(local, MPI.MAX, comm, domain)


def mpp_min(
    local: np.ndarray[Any, Any],
    domain: Domain | None = None,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray[Any, Any]:
    """FMS's ``mpp_min``."""
    return _mpp_reduce(local, MPI.MIN, comm, domain)


def mpp_chksum(
    local: np.ndarray[Any, Any],
    comm: MPI.Comm | None = None,
    *,
    mask_val: float | None = None,
) -> int:
    """Compute a rank-order-independent bitwise checksum.

    Parameters
    ----------
    local : numpy.ndarray
        Rank-local field.
    comm : mpi4py.MPI.Comm, optional
        Reduction communicator.
    mask_val : float, optional
        Sentinel excluded from the checksum.

    Returns
    -------
    int
        Global checksum.

    Raises
    ------
    TypeError
        If the element width is unsupported.
    """
    values = np.asarray(local)
    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    if mask_val is not None:
        keep = (
            ~np.isnan(values)
            if isinstance(mask_val, float) and np.isnan(mask_val)
            else values != mask_val
        )
        values = values[keep]

    # Reinterpret the bits and widen to int64 so the sum cannot overflow the
    # element type: FMS's TRANSFER, without a copy where NumPy allows it.
    width = values.dtype.itemsize
    if width not in (1, 2, 4, 8):
        raise TypeError(f"mpp_chksum: unsupported dtype {values.dtype}.")
    as_int = np.ascontiguousarray(values).view(f"i{width}")
    local_sum = np.int64(as_int.sum(dtype=np.int64))

    if comm is None or comm.size == 1:
        return int(local_sum)
    total = np.empty(1, dtype=np.int64)
    comm.Allreduce(np.array([local_sum], dtype=np.int64), total, op=MPI.SUM)
    return int(total[0])


def mpp_partition_offsets(comm: MPI.Comm, local_length: int) -> tuple[int, int, int]:
    """Recompute distributed offsets after a local length change.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Partition communicator.
    local_length : int
        This rank's new local length.

    Returns
    -------
    tuple[int, int, int]
        Global size and this rank's half-open ownership bounds.
    """
    length = np.array([int(local_length)], dtype=np.int64)
    total = np.empty_like(length)
    comm.Allreduce(length, total, op=MPI.SUM)
    prefix = np.zeros_like(length)
    comm.Exscan(length, prefix, op=MPI.SUM)
    if comm.rank == 0:
        prefix[0] = 0  # Exscan leaves rank 0's receive buffer undefined.
    start = int(prefix[0])
    return int(total[0]), start, start + int(length[0])


def mpp_sync(comm: MPI.Comm) -> None:
    """Block until every rank in ``comm`` has arrived.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator to synchronise.
    """
    comm.Barrier()


def mpp_broadcast(
    value: np.ndarray[Any, Any], comm: MPI.Comm, *, root: int = 0
) -> np.ndarray[Any, Any]:
    """Send ``root``'s array to every rank.

    Parameters
    ----------
    value : numpy.ndarray
        Array to send on ``root``; a correctly shaped and typed buffer
        elsewhere.
    comm : mpi4py.MPI.Comm
        Communicator to broadcast over.
    root : int, default 0
        Rank holding the source array.

    Returns
    -------
    numpy.ndarray
        The broadcast array, on every rank.
    """
    buffer = np.ascontiguousarray(value)
    comm.Bcast(buffer, root=root)
    return buffer


def mpp_gather(
    local: np.ndarray[Any, Any], comm: MPI.Comm, *, root: int | None = None
) -> np.ndarray[Any, Any] | None:
    """Collect equally sized rank-local arrays along a new leading axis.

    Parameters
    ----------
    local : numpy.ndarray
        This rank's contribution; the same shape on every rank.
    comm : mpi4py.MPI.Comm
        Communicator to gather over.
    root : int, optional
        Rank receiving the result. None gathers to every rank.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(comm.size, *local.shape)``, or None on non-root
        ranks when ``root`` is given.
    """
    send = np.ascontiguousarray(local)
    if root is None:
        recv = np.empty((comm.size, *send.shape), dtype=send.dtype)
        comm.Allgather(send, recv)
        return recv
    recv = (
        np.empty((comm.size, *send.shape), dtype=send.dtype)
        if comm.rank == root
        else None
    )
    comm.Gather(send, recv, root=root)
    return recv


def mpp_scatter(
    value: np.ndarray[Any, Any] | None,
    comm: MPI.Comm,
    *,
    root: int = 0,
) -> np.ndarray[Any, Any]:
    """Deal out ``root``'s leading axis, one slice per rank.

    Parameters
    ----------
    value : numpy.ndarray or None
        Array of shape ``(comm.size, *rest)`` on ``root``, ignored elsewhere.
    comm : mpi4py.MPI.Comm
        Communicator to scatter over.
    root : int, default 0
        Rank holding the source array.

    Returns
    -------
    numpy.ndarray
        This rank's slice, of shape ``rest``.
    """
    shape, dtype = comm.bcast(
        (value.shape[1:], value.dtype) if comm.rank == root else None, root=root
    )
    recv = np.empty(shape, dtype=dtype)
    comm.Scatter(
        np.ascontiguousarray(value) if comm.rank == root else None, recv, root=root
    )
    return recv


def mpp_alltoall(send: np.ndarray[Any, Any], comm: MPI.Comm) -> np.ndarray[Any, Any]:
    """Exchange one equally sized slice with every rank.

    Parameters
    ----------
    send : numpy.ndarray
        Array of shape ``(comm.size, *rest)``; entry ``i`` goes to rank ``i``.
    comm : mpi4py.MPI.Comm
        Communicator to exchange over.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(comm.size, *rest)``; entry ``i`` came from rank
        ``i``.
    """
    buffer = np.ascontiguousarray(send)
    recv = np.empty_like(buffer)
    comm.Alltoall(buffer, recv)
    return recv
