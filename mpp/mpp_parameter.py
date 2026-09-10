"""Flags shared by the ``mpp`` modules.

Mirrors FMS ``mpp/mpp_parameter.F90``. The update flags are bit masks so a
caller can request several edges in one call, exactly as in FMS, where
``XUPDATE`` is ``WUPDATE + EUPDATE``.
"""

from __future__ import annotations

from typing import Final

#: Update the halo on one edge of the process grid.
WUPDATE: Final = 1 << 0
EUPDATE: Final = 1 << 1
SUPDATE: Final = 1 << 2
NUPDATE: Final = 1 << 3

#: Update both edges of one axis.
XUPDATE: Final = WUPDATE | EUPDATE
YUPDATE: Final = SUPDATE | NUPDATE

#: Update every edge, corners included.
BOTH_UPDATE: Final = XUPDATE | YUPDATE

#: Where a field sits relative to its grid cell.
CENTER: Final = "center"
CORNER: Final = "corner"

#: Boundary treatment of a partitioned axis. ``CYCLIC_GLOBAL_DOMAIN`` wraps
#: the first and last ranks into neighbours; ``GLOBAL_DATA_DOMAIN`` leaves the
#: outermost halo untouched.
GLOBAL_DATA_DOMAIN: Final = "global"
CYCLIC_GLOBAL_DOMAIN: Final = "cyclic"

#: Whether a global sum is accumulated in extended fixed point, making it
#: independent of the rank count, or in plain floating point.
BITWISE_EXACT_SUM: Final = True
NON_BITWISE_EXACT_SUM: Final = False
