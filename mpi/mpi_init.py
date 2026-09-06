"""Single place where mpi4py is imported, so ``MPI_Init`` can be held back.

Importing ``mpi4py.MPI`` calls ``MPI_Init`` as a side effect. Inside a Slurm
allocation that is not a private act: with PMI variables in the environment
the process registers as a PMIx client of the job step, and from then on its
death is the step's death. A Jupyter kernel that has done this takes the whole
allocation down every time it crashes or is restarted::

    mpi/pmix_v5: _errhandler: ... status = -61, source = [slurm.pmix.NNNN.0:0]
    srun: Job step aborted: Waiting up to 182 seconds for job step to finish.
    *** STEP NNNN.0 ON nodeXXXX CANCELLED ... DUE to SIGNAL Killed ***

``-61`` is ``PMIX_ERR_LOST_CONNECTION``: the client went away and the launcher
concluded a task of the step had died. Restarting a kernel is routine, so the
allocation is lost for an entirely ordinary action.

Every climtools module that needs mpi4py imports ``MPI`` from here rather than
from ``mpi4py`` directly, so the decision below is made once and cannot be
bypassed by import order.
"""

from __future__ import annotations

import os
import sys

import mpi4py

#: Launcher variables reporting the world *size*. Rank variables are useless
#: for this: Slurm exports ``SLURM_PROCID`` into every task of every step and
#: every child inherits it, so a kernel started inside an allocation carries
#: one without being an MPI rank.


LAUNCH_ENV = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "I_MPI_COMM_WORLD_SIZE",
)


def world_size() -> int:
    """World size the launcher advertises, or 1 when none does."""
    for key in LAUNCH_ENV:
        value = os.environ.get(key)
        if value and value.isdigit():
            return int(value)
    return 1


def _defer_mpi_init() -> bool:
    """Whether to hold back ``MPI_Init`` for this process.

    Only for an interactive kernel that is not part of a multi-rank launch.
    A batch job never has ipykernel loaded, so this cannot suppress MPI for a
    real run; and a single kernel cannot do useful MPI anyway, since its
    ``COMM_WORLD`` has one member. ``CLIMTOOLS_FORCE_MPI=1`` overrides, for
    anyone who wants the old behaviour back.
    """

    return "ipykernel" in sys.modules and world_size() <= 1


DEFERRED = _defer_mpi_init()

if DEFERRED:
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False


# Must follow the rc assignment above: importing MPI is what runs MPI_Init.
from mpi4py import MPI

__all__ = ["DEFERRED", "MPI", "mpi_is_available", "require_mpi"]


def mpi_is_available() -> bool:
    """Whether MPI calls are safe to make in this process."""
    return bool(MPI.Is_initialized()) and not MPI.Is_finalized()


def require_mpi() -> None:
    """Raise before any MPI call if ``MPI_Init`` was held back.

    Calling into MPI without initialising it does not raise -- the MPI
    implementation aborts the process -- so this has to run before the first
    communicator method, not around it.
    """
    if mpi_is_available():
        return
    raise RuntimeError("MPI not initialized")
