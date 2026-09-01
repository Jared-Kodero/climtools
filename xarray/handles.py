from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Literal

    import numpy as np

    from .core import MPIXarray


class MPIRolling:
    """Chainable rolling-window handle, mirroring ``xarray``'s ``.rolling(...)``.

    Returned by :meth:`MPIXarray.rolling`; call one of the reduction
    methods below to compute the windowed reduction via
    :meth:`MPIXarray.rolling_reduce`.

    Parameters
    ----------
    parent : MPIXarray
        Object to roll over.
    dim : Hashable
        Dimension to roll over.
    window : int
        Window size, as in ``xarray.DataArray.rolling``.
    center : bool, optional
        As in ``xarray.DataArray.rolling``.
    min_periods : int or None, optional
        As in ``xarray.DataArray.rolling``.
    """

    def __init__(
        self,
        parent: MPIXarray,
        dim: Hashable,
        window: int,
        *,
        center: bool = True,
        min_periods: int | None = None,
    ) -> None:
        self._parent = parent
        self._dim = dim
        self._window = window
        self._center = center
        self._min_periods = min_periods

    def _reduce(self, reduce: str) -> MPIXarray:
        return self._parent.rolling_reduce(
            self._dim,
            self._window,
            reduce,
            center=self._center,
            min_periods=self._min_periods,
        )

    def mean(self) -> MPIXarray:
        """Rolling mean. See :meth:`MPIXarray.rolling_reduce`."""
        return self._reduce("mean")

    def sum(self) -> MPIXarray:
        """Rolling sum. See :meth:`MPIXarray.rolling_reduce`."""
        return self._reduce("sum")

    def min(self) -> MPIXarray:
        """Rolling minimum. See :meth:`MPIXarray.rolling_reduce`."""
        return self._reduce("min")

    def max(self) -> MPIXarray:
        """Rolling maximum. See :meth:`MPIXarray.rolling_reduce`."""
        return self._reduce("max")

    def std(self) -> MPIXarray:
        """Rolling standard deviation. See :meth:`MPIXarray.rolling_reduce`."""
        return self._reduce("std")

    def count(self) -> MPIXarray:
        """Rolling valid-value count. See :meth:`MPIXarray.rolling_reduce`."""
        return self._reduce("count")


class MPIGroupBy:
    """Chainable groupby handle, mirroring ``xarray``'s ``.groupby(...)``.

    Returned by :meth:`MPIXarray.groupby`; call one of the reduction
    methods below to compute the grouped reduction via the internal
    ``groupby_reduce`` engine dispatch.

    Parameters
    ----------
    parent : MPIXarray
        Object being grouped.
    dim : Hashable
        Dimension being grouped and reduced.
    labels : array-like
        Group key for every position along this rank's local ``dim`` axis.
    """

    def __init__(
        self, parent: MPIXarray, dim: Hashable, labels: xr.DataArray | np.ndarray
    ) -> None:
        self._parent = parent
        self._dim = dim
        self._labels = labels

    def _reduce(
        self,
        op: Literal["sum", "mean", "count", "min", "max"],
        *,
        skipna: bool | None,
        keep_attrs: bool | None,
        partition_dim: Hashable | Literal["auto"] | None,
    ) -> MPIXarray:
        from .core import finalize
        from .groupby import groupby_reduce

        return finalize(
            groupby_reduce(
                self._parent._runtime,
                self._parent._prepare(),
                self._dim,
                self._labels,
                op,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._parent._runtime,
        )

    def sum(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Sum within each group. See :meth:`MPIXarray.groupby`."""
        return self._reduce(
            "sum", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def mean(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Mean within each group. See :meth:`MPIXarray.groupby`."""
        return self._reduce(
            "mean", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def count(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Valid-value count within each group. See :meth:`MPIXarray.groupby`."""
        return self._reduce(
            "count", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def min(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Minimum within each group. See :meth:`MPIXarray.groupby`."""
        return self._reduce(
            "min", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def max(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Maximum within each group. See :meth:`MPIXarray.groupby`."""
        return self._reduce(
            "max", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )


class MPIResample:
    """Chainable resample handle, mirroring ``xarray``'s ``.resample(...)``.

    Returned by :meth:`MPIXarray.resample`; call one of the reduction
    methods below to compute the resampled reduction via the internal
    ``resample_reduce`` engine dispatch.

    Parameters
    ----------
    parent : MPIXarray
        Object being resampled.
    dim : Hashable
        Datetime dimension to resample.
    freq : str
        Pandas offset alias (e.g. "D", "MS", "YS").
    """

    def __init__(self, parent: MPIXarray, dim: Hashable, freq: str) -> None:
        self._parent = parent
        self._dim = dim
        self._freq = freq

    def _reduce(
        self,
        op: Literal["sum", "mean", "count", "min", "max"],
        *,
        skipna: bool | None,
        keep_attrs: bool | None,
        partition_dim: Hashable | Literal["auto"] | None,
    ) -> MPIXarray:
        from .core import finalize
        from .groupby import resample_reduce

        return finalize(
            resample_reduce(
                self._parent._runtime,
                self._parent._prepare(),
                self._dim,
                self._freq,
                op,
                skipna=skipna,
                keep_attrs=keep_attrs,
                partition_dim=partition_dim,
            ),
            self._parent._runtime,
        )

    def sum(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Sum within each resampled bin. See :meth:`MPIXarray.resample`."""
        return self._reduce(
            "sum", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def mean(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Mean within each resampled bin. See :meth:`MPIXarray.resample`."""
        return self._reduce(
            "mean", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def count(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Valid-value count within each resampled bin.

        See :meth:`MPIXarray.resample`.
        """
        return self._reduce(
            "count", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def min(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Minimum within each resampled bin. See :meth:`MPIXarray.resample`."""
        return self._reduce(
            "min", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )

    def max(
        self,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        partition_dim: Hashable | Literal["auto"] | None = "auto",
    ) -> MPIXarray:
        """Maximum within each resampled bin. See :meth:`MPIXarray.resample`."""
        return self._reduce(
            "max", skipna=skipna, keep_attrs=keep_attrs, partition_dim=partition_dim
        )
