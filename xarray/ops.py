"""Compose MPI-aware xarray operation mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .arithmetic import Arithmetic
from .elementwise import Elementwise
from .groupby import Groupby
from .indexing import Indexing
from .io import IO
from .reductions import Reduction
from .statistics import Statistics

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime


class _MPIXarrayOps(
    IO, Indexing, Reduction, Statistics, Groupby, Arithmetic, Elementwise
):
    """Bind MPI-aware xarray operations to a runtime.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime used by distributed operations.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime
