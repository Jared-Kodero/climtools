"""MPI-aware distributed xarray operations.

:class:`XarrayMPI` is the public interface that
:class:`~..core.lib_mpi.MPIRuntime` binds to ``runtime.xarray``. The
implementation is split by concern across sibling modules in this package:

- :mod:`.common`      Communicator-free constants and dtype helpers.
- :mod:`.engine`      Rank-independent reduction planning and MPI collective
  primitives shared by every reduction .
- :mod:`.io`          Dataset/DataArray open, distribute, repartition,
  create, and save-chunk attachment.
- :mod:`.indexing`    Global-coordinate ``isel``/``sel``.
- :mod:`.reductions`  ``sum``/``prod``/``mean``/``min``/``max``/``first``/
  ``last``/``any``/``all``.
- :mod:`.statistics`  ``std``/``var``.
- :mod:`.groupby`     ``groupby_reduce``/``resample_reduce``.
- :mod:`.operator`    ``align``/``apply``/``evaluate`` -- rank-local
  arithmetic restricted to partition-preserving operations, plus an
  ``ast``-based expression evaluator built on top of ``apply``; and the
  dedicated implementations for the operations that legitimately need to
  reduce or communicate across the partition dimension:
  ``matmul`` (MPI-reduced distributed matrix multiplication) and
  ``rolling_reduce``/``halo_exchange`` (windowed reductions via
  point-to-point boundary exchange with the adjacent ranks).

Chunk and metadata helpers below are communicator-free and re-exported here
only for backward compatibility; their implementations live in
:mod:`.chunks` and :mod:`.meta`.
"""

# xarray_mpi.py

from __future__ import annotations

from typing import TYPE_CHECKING

from .chunks import (
    get_balanced_bounds,
    get_chunk_bounds,
    get_chunk_info,
    get_chunk_overrides,
    get_effective_chunk_size,
    get_native_chunk_sizes,
    get_usable_native_chunk,
    prune_chunk_info,
)
from .groupby import Groupby
from .indexing import Indexing
from .io import IO
from .meta import choose_partition_dim, indexer_is_scalar
from .operator import Arithmetic
from .reductions import Reduction
from .statistics import Statistics

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime

__all__ = [
    "MPIXarray",
    "choose_partition_dim",
    "get_balanced_bounds",
    "get_chunk_bounds",
    "get_chunk_info",
    "get_chunk_overrides",
    "get_effective_chunk_size",
    "get_native_chunk_sizes",
    "get_usable_native_chunk",
    "indexer_is_scalar",
    "prune_chunk_info",
]


class MPIXarray(IO, Indexing, Reduction, Statistics, Groupby, Arithmetic):
    """MPI-aware xarray operations bound to an MPI runtime.

    Composes, by concern:

    - :class:`~.io.IO` -- open, distribute, repartition, create.
    - :class:`~.indexing.Indexing` -- global-coordinate ``isel``/``sel``.
    - :class:`~.reductions.Reduction` -- ``sum``/``prod``/``mean``/
      ``min``/``max``/``first``/``last``/``any``/``all``.
    - :class:`~.statistics.Statistics` -- ``std``/``var``.
    - :class:`~.groupby.Groupby` -- ``groupby_reduce``/``resample_reduce``.
    - :class:`~.operator.Arithmetic` -- ``align``/``apply``/``evaluate``:
      rank-local arithmetic restricted to partition-preserving, rank-local
      operations, validated after every call rather than by inspecting the
      callable beforehand; plus ``matmul`` and ``rolling_reduce``/
      ``halo_exchange``, the dedicated implementations for contraction and
      windowed operations that legitimately need an MPI reduction or a
      neighboring rank's boundary values.

    ``Reduction``, ``Statistics``, and ``Groupby`` all build on
    :class:`~.engine.ReductionPlanning` for collective planning;
    ``Arithmetic`` uses ``self.repartition`` (from ``IO``) and
    ``self._agree`` (from ``ReductionPlanning``).

    The names below are the exact same bound methods the base classes
    already provide through normal Python method resolution -- listing them
    here does not wrap, re-dispatch, or copy them (``MPIXarray.isel is
    Indexing.isel`` holds), so there is no extra call frame, no duplicated
    docstring/signature to drift out of sync, and no runtime cost versus
    plain inheritance. They are assigned explicitly only so the full public
    surface is visible directly on this class -- in ``help(MPIXarray)``,
    ``vars(MPIXarray)``, and a plain read of this file -- rather than only
    discoverable by walking the MRO through six base classes.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator is used for distributed operations."""

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime
