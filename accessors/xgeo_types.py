from __future__ import annotations

# Static-analysis bridge used by the development-only xarray source patch.
# Xarray imports this module only inside TYPE_CHECKING.
from .mpi_accessors import MPIAccessor as MPIAccessor
from .mpi_accessors import MPIDataArray as MPIDataArray
from .mpi_accessors import MPIDataset as MPIDataset
from .xarray_accessors import GeoDataArray as GeoDataArray
from .xarray_accessors import GeoDataset as GeoDataset

__all__ = [
    "GeoDataArray",
    "GeoDataset",
    "MPIAccessor",
    "MPIDataArray",
    "MPIDataset",
]
