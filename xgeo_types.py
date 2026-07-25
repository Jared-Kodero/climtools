from __future__ import annotations

# Static-analysis bridge used by the development-only xarray source patch.
# Xarray imports this module only inside TYPE_CHECKING.
from .accessors import GeoDataArray as GeoDataArray
from .accessors import GeoDataset as GeoDataset

__all__ = ["GeoDataArray", "GeoDataset"]
