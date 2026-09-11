"""Fill halo points from neighbouring ranks.

Mirrors FMS ``mpp/include/mpp_do_update.fh`` and its nonblocking variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


from ..mpi.mpi_init import MPI
from .mpp_data import mpp_domains_get_stack, mpp_domains_put_stack
from .mpp_domains import Domain
from .mpp_domains_define import mpp_get_cartesian_domain
from .mpp_domains_util import mpp_get_neighbor_pe
from .mpp_parameter import (
    BOTH_UPDATE,
    CENTER,
    CORNER,
    FOLD_NORTH_EDGE,
    FOLD_SOUTH_EDGE,
    XUPDATE,
    YUPDATE,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class HaloWidthError(ValueError):
    """A rank's local partition is shorter than the halo an op asked for.

    Its own type rather than a bare ``ValueError`` because callers have to
    tell this architectural refusal apart from a genuine failure -- the test
    suite reports it as a skip, not a failure. That classification used to
    match a substring of the message, so shortening the message silently
    turned every one of those skips into a failure. Subclasses ``ValueError``
    so existing ``except ValueError`` handlers are unaffected.
    """


@dataclass
class DomainUpdate:
    """Store state for an in-flight halo exchange.

    Attributes
    ----------
    items : dict[str, numpy.ndarray]
        Fields being exchanged.
    groups : dict
        Fields grouped by wire dtype.
    recv_bufs : dict
        Receive buffers keyed by dtype and side.
    recv_reqs, send_reqs : list
        Outstanding MPI requests.
    axis : int
        Exchanged array axis.
    before, after : int
        Requested halo widths.
    single : bool
        Whether the input was a single array.
    unpack : Any
        Callable restoring wire representations.
    """

    items: dict[str, np.ndarray[Any, Any]]
    groups: dict[Any, list[str]]
    recv_bufs: dict[tuple[Any, str], np.ndarray[Any, Any]]
    recv_reqs: list[Any]
    send_reqs: list[Any]
    axis: int
    before: int
    after: int
    single: bool
    unpack: Any


def mpp_start_update_domains(
    fields: np.ndarray[Any, Any] | Mapping[str, np.ndarray[Any, Any]],
    domain: Domain,
    dim: str,
    axis: int,
    *,
    before: int,
    after: int,
    periodic: bool = False,
    left_rank: int | None = None,
    right_rank: int | None = None,
) -> DomainUpdate:
    """Start a nonblocking halo exchange.

    Parameters
    ----------
    fields : numpy.ndarray or mapping[str, numpy.ndarray]
        Field or fields sharing the exchanged axis.
    domain : Domain
        Rank-local domain descriptor.
    dim : str
        Partition dimension.
    axis : int
        Array axis corresponding to ``dim``.
    before, after : int
        Lower and upper halo widths.
    periodic : bool, default False
        Wrap across global edges.
    left_rank, right_rank : int or None, optional
        Explicit neighboring ranks.

    Returns
    -------
    DomainUpdate
        In-flight exchange state.
    """
    single = isinstance(fields, np.ndarray)
    items: dict[str, np.ndarray[Any, Any]] = {"": fields} if single else dict(fields)

    comm = domain.comm
    if left_rank is None or right_rank is None:
        default_left, default_right = mpp_get_neighbor_pe(
            domain, dim, periodic=periodic
        )
        left_rank = default_left if left_rank is None else left_rank
        right_rank = default_right if right_rank is None else right_rank

    def _view(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """View as a dtype the raw MPI buffer protocol accepts."""
        return arr.view(np.int64) if arr.dtype.kind in "mM" else arr

    def _slab(arr: np.ndarray[Any, Any], start: int, stop: int) -> np.ndarray[Any, Any]:
        """Return a contiguous copy of ``arr[start:stop]`` along the halo axis."""
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return np.ascontiguousarray(arr[tuple(idx)])

    def _halo_shape(name: str, width: int) -> tuple[int, ...]:
        """Return the shape of a ``width``-wide halo slab of one field."""
        arr = items[name]
        return (*arr.shape[:axis], width, *arr.shape[axis + 1 :])

    # Pack fields by wire dtype in deterministic order; no layout metadata is
    # transmitted.
    groups: dict[np.dtype[Any], list[str]] = {}
    for name, arr in items.items():
        groups.setdefault(_view(arr).dtype, []).append(name)
    for names in groups.values():
        names.sort()

    def _pack(names: list[str], side: str) -> np.ndarray[Any, Any]:
        """Flatten one edge of every named field into a single send buffer."""
        pieces = []
        for name in names:
            arr = items[name]
            slab = (
                _slab(arr, arr.shape[axis] - before, arr.shape[axis])
                if side == "right"
                else _slab(arr, 0, after)
            )
            pieces.append(_view(slab).reshape(-1))
        return np.concatenate(pieces)

    def _unpack(
        flat: np.ndarray[Any, Any], names: list[str], width: int
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Split a received buffer back into per-field halo slabs."""
        out: dict[str, np.ndarray[Any, Any]] = {}
        pos = 0
        for name in names:
            shape = _halo_shape(name, width)
            count = int(np.prod(shape)) if shape else 1
            # Restore each field's original dtype after unpacking the wire
            # representation.
            out[name] = flat[pos : pos + count].reshape(shape).view(items[name].dtype)
            pos += count
        return out

    can_send_right = right_rank is not None and before > 0
    can_send_left = left_rank is not None and after > 0
    can_recv_before = left_rank is not None and before > 0
    can_recv_after = right_rank is not None and after > 0

    recv_bufs: dict[tuple[np.dtype[Any], str], np.ndarray[Any, Any]] = {}
    recv_reqs = []
    for dtype, names in groups.items():
        if can_recv_before:
            count = sum(int(np.prod(_halo_shape(name, before))) for name in names)
            buf = mpp_domains_get_stack(count, dtype)
            recv_bufs[dtype, "before"] = buf
            recv_reqs.append(comm.Irecv(buf, source=left_rank))
        if can_recv_after:
            count = sum(int(np.prod(_halo_shape(name, after))) for name in names)
            buf = mpp_domains_get_stack(count, dtype)
            recv_bufs[dtype, "after"] = buf
            recv_reqs.append(comm.Irecv(buf, source=right_rank))

    send_reqs = []
    for dtype, names in groups.items():
        if can_send_right:
            send_reqs.append(comm.Isend(_pack(names, "right"), dest=right_rank))
        if can_send_left:
            send_reqs.append(comm.Isend(_pack(names, "left"), dest=left_rank))

    return DomainUpdate(
        items=items,
        groups=groups,
        recv_bufs=recv_bufs,
        recv_reqs=recv_reqs,
        send_reqs=send_reqs,
        axis=axis,
        before=before,
        after=after,
        single=single,
        unpack=_unpack,
    )


def mpp_complete_update_domains(
    update: DomainUpdate,
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]], int, int]:
    """Complete a halo exchange and return received slabs.

    Parameters
    ----------
    update : DomainUpdate
        In-flight exchange state.

    Returns
    -------
    tuple[dict, dict, int, int]
        Lower halos, upper halos, and realized lower/upper pad widths.
    """
    MPI.Request.Waitall(update.recv_reqs)
    MPI.Request.Waitall(update.send_reqs)

    recv_before: dict[str, np.ndarray[Any, Any]] = {}
    recv_after: dict[str, np.ndarray[Any, Any]] = {}
    for dtype, names in update.groups.items():
        if (dtype, "before") in update.recv_bufs:
            recv_before.update(
                update.unpack(update.recv_bufs[dtype, "before"], names, update.before)
            )
        if (dtype, "after") in update.recv_bufs:
            recv_after.update(
                update.unpack(update.recv_bufs[dtype, "after"], names, update.after)
            )

    # unpack() copies out of the wire buffers, so they can go back to the pool
    # for the next exchange to reuse.
    for buffer in update.recv_bufs.values():
        mpp_domains_put_stack(buffer)

    return (
        recv_before,
        recv_after,
        update.before if recv_before else 0,
        update.after if recv_after else 0,
    )


def mpp_update_domains(
    fields: np.ndarray[Any, Any] | Mapping[str, np.ndarray[Any, Any]],
    domain: Domain,
    dim: str,
    axis: int,
    *,
    before: int,
    after: int,
    periodic: bool = False,
    left_rank: int | None = None,
    right_rank: int | None = None,
) -> tuple[np.ndarray[Any, Any] | dict[str, np.ndarray[Any, Any]], int, int]:
    """Exchange halos and return padded local fields.

    Parameters
    ----------
    fields : numpy.ndarray or mapping[str, numpy.ndarray]
        Field or fields to exchange.
    domain : Domain
        Rank-local domain descriptor.
    dim : str
        Partition dimension.
    axis : int
        Array axis corresponding to ``dim``.
    before, after : int
        Lower and upper halo widths.
    periodic : bool, default False
        Wrap across global edges.
    left_rank, right_rank : int or None, optional
        Explicit neighboring ranks.

    Returns
    -------
    tuple[numpy.ndarray or dict, int, int]
        Padded field(s) and realized lower/upper pad widths.
    """
    update = mpp_start_update_domains(
        fields,
        domain,
        dim,
        axis,
        before=before,
        after=after,
        periodic=periodic,
        left_rank=left_rank,
        right_rank=right_rank,
    )
    recv_before, recv_after, left_pad, right_pad = mpp_complete_update_domains(update)

    padded = {
        name: np.concatenate(
            [
                piece
                for piece in (recv_before.get(name), arr, recv_after.get(name))
                if piece is not None
            ],
            axis=axis,
        )
        for name, arr in update.items.items()
    }
    return (padded[""] if update.single else padded), left_pad, right_pad


def mpp_update_domains_nd(
    field: np.ndarray[Any, Any],
    domain: Domain,
    dims: Sequence[str],
    halo: Mapping[str, tuple[int, int]],
    *,
    flags: int = BOTH_UPDATE,
    periodic: Mapping[str, bool] | None = None,
) -> tuple[np.ndarray[Any, Any], dict[str, tuple[int, int]]]:
    """Fill halos on several axes at once, corners included.

    A single-axis exchange cannot give a rank its diagonal neighbours, which a
    two-dimensional stencil needs. Updating one axis at a time and letting the
    second exchange carry the halo the first just received produces those
    corner points without a separate diagonal message, which is how FMS fills
    them for ``position=CENTER``.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's compute-domain values.
    domain : Domain
        Rank-local domain.
    dims : sequence of str
        Dimension name of each axis of ``field``.
    halo : mapping[str, tuple[int, int]]
        Halo width before and after, per dimension.
    flags : int, default BOTH_UPDATE
        Which edges to update, from :mod:`~climtools.mpp.mpp_parameter`.
        ``XUPDATE`` and ``YUPDATE`` select the first and second partitioned
        axis respectively.
    periodic : mapping[str, bool], optional
        Whether each dimension wraps at the global edges.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, tuple[int, int]]]
        The widened field and the halo width actually received per dimension,
        which is narrower than requested at a non-cyclic global edge.
    """
    wrap = dict(domain.cyclic)
    wrap.update(periodic or {})
    selected = [XUPDATE, YUPDATE]
    result = np.asarray(field)
    received: dict[str, tuple[int, int]] = {}

    for order, dim in enumerate(d for d in dims if d in domain.dims):
        if order < len(selected) and not flags & selected[order]:
            received[dim] = (0, 0)
            continue
        before, after = halo.get(dim, (0, 0))
        if not before and not after:
            received[dim] = (0, 0)
            continue
        axis = list(dims).index(dim)
        result, low, high = mpp_update_domains(
            result,
            domain,
            dim,
            axis,
            before=before,
            after=after,
            periodic=wrap.get(dim, False),
        )
        received[dim] = (low, high)

    return result, received


def mpp_get_boundary(
    field: np.ndarray[Any, Any],
    domain: Domain,
    dim: str,
    axis: int,
    *,
    position: str = CENTER,
) -> tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]:
    """Return the compute-domain edge values a neighbour needs.

    FMS ``mpp_get_boundary`` hands a staggered grid the single row or column
    sitting on the shared boundary, rather than a full halo.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's compute-domain values.
    domain : Domain
        Rank-local domain.
    dim : str
        Partitioned dimension.
    axis : int
        Array axis corresponding to ``dim``.
    position : {"center", "corner"}, default "center"
        Grid position. ``CORNER`` fields share their boundary row with the
        neighbour, so the upper edge is omitted to avoid duplicating it.

    Returns
    -------
    tuple[numpy.ndarray or None, numpy.ndarray or None]
        Lower and upper boundary slabs, or None where this rank sits at a
        global edge.
    """
    values = np.asarray(field)
    at_start = domain.starts[dim] == 0
    at_stop = domain.stops[dim] == domain.global_sizes[dim]
    index: list[Any] = [slice(None)] * values.ndim

    lower = None
    if not at_start:
        index[axis] = slice(0, 1)
        lower = values[tuple(index)].copy()

    upper = None
    if not at_stop and position != CORNER:
        index[axis] = slice(values.shape[axis] - 1, values.shape[axis])
        upper = values[tuple(index)].copy()

    return lower, upper


def mpp_do_update_fold(
    field: np.ndarray[Any, Any],
    domain: Domain,
    fold_dim: str,
    fold_axis: int,
    mirror_dim: str,
    mirror_axis: int,
    *,
    width: int = 1,
    parity: int = 1,
) -> np.ndarray[Any, Any]:
    """Fill the halo beyond a folded edge by mirroring along the other axis.

    A tripolar grid closes across the north pole by joining the top row to
    itself in reverse: the point one row beyond global row ``NY-1`` at column
    ``i`` is the value at row ``NY-1``, column ``NX-1-i``. That partner
    column usually lives on another rank, so the row is gathered along
    ``mirror_dim``, reversed, and each rank takes back its own columns.

    Parameters
    ----------
    field : numpy.ndarray
        This rank's compute-domain values.
    domain : Domain
        Domain describing ``field``; its ``fold`` selects the edge.
    fold_dim : str
        Dimension whose edge is folded.
    fold_axis : int
        Array axis of ``fold_dim``.
    mirror_dim : str
        Dimension the fold mirrors along.
    mirror_axis : int
        Array axis of ``mirror_dim``.
    width : int, default 1
        Rows to fill beyond the fold.
    parity : int, default 1
        ``+1`` for a scalar, ``-1`` for the component of a vector that
        reverses direction across the fold. FMS applies the same sign flip
        in ``mpp_do_updateV``.

    Returns
    -------
    numpy.ndarray
        ``field`` with ``width`` rows appended past the folded edge. Ranks
        not touching the edge get ``field`` back unchanged.

    Raises
    ------
    ValueError
        If ``width`` exceeds this rank's extent along ``fold_dim``.
    """
    north = bool(domain.fold & FOLD_NORTH_EDGE)
    south = bool(domain.fold & FOLD_SOUTH_EDGE)
    if not (north or south):
        return field

    values = np.asarray(field)
    if width > values.shape[fold_axis]:
        raise ValueError(
            f"Fold width {width} exceeds local {fold_dim!r} extent "
            + f"{values.shape[fold_axis]}."
        )

    at_edge = (
        domain.stops[fold_dim] == domain.global_sizes[fold_dim]
        if north
        else domain.starts[fold_dim] == 0
    )

    index: list[Any] = [slice(None)] * values.ndim
    index[fold_axis] = (
        slice(values.shape[fold_axis] - width, None) if north else slice(0, width)
    )
    edge_rows = values[tuple(index)] if at_edge else None

    # Every rank owning part of the edge contributes its columns, so the
    # mirrored partner can be found wherever it lives.
    comm = mpp_dim_comm_for(domain, mirror_dim)
    contributions = comm.allgather(
        None
        if edge_rows is None
        else (domain.starts[mirror_dim], np.moveaxis(edge_rows, mirror_axis, 0))
    )
    pieces = [c for c in contributions if c is not None]
    if not pieces:
        return values

    pieces.sort(key=lambda item: item[0])
    whole = np.concatenate([block for _, block in pieces], axis=0)
    mirrored = whole[::-1]
    start = domain.starts[mirror_dim]
    stop = domain.stops[mirror_dim]
    local = np.moveaxis(mirrored[start:stop], 0, mirror_axis)
    if parity != 1:
        local = local * parity

    # The fold reverses the order of the rows themselves as well.
    flip: list[Any] = [slice(None)] * values.ndim
    flip[fold_axis] = slice(None, None, -1)
    local = local[tuple(flip)]

    return np.concatenate((values, local) if north else (local, values), axis=fold_axis)


def mpp_dim_comm_for(domain: Domain, dim: str) -> MPI.Comm:
    """Return the communicator varying along ``dim`` for this domain.

    Parameters
    ----------
    domain : Domain
        Domain to take the communicator from.
    dim : str
        Dimension the communicator should vary along.

    Returns
    -------
    mpi4py.MPI.Comm
        The whole communicator for a one-axis domain, otherwise the
        Cartesian subcommunicator for ``dim``.
    """
    if domain.cart is None or len(domain.dims) == 1:
        return domain.comm
    return mpp_get_cartesian_domain(
        domain.comm, domain.dims, domain.global_sizes
    ).sub_comm((dim,))
