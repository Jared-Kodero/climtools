from __future__ import annotations

import inspect

import numpy as np
import xarray as xr

from climtools import mpi
from climtools.core.xr_meta import get_mpi_meta


def _expected_partition(
    value: xr.DataArray,
    result: xr.DataArray,
) -> xr.DataArray:
    meta = get_mpi_meta(result)
    if meta is None:
        return value
    dim = str(meta["dim"])
    return value.isel({dim: slice(int(meta["start"]), int(meta["stop"]))})


def main() -> None:
    time_size = max(8, 2 * mpi.comm.size)
    full = xr.DataArray(
        np.arange(time_size * 3 * 5, dtype=np.float64).reshape(time_size, 3, 5),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(time_size),
            "lat": np.array([-30.0, 0.0, 30.0]),
            "lon": np.arange(5),
        },
        attrs={"units": "K"},
        name="field",
    )
    local = mpi.xarray.redistribute(full, "time")
    original_meta = get_mpi_meta(local)
    assert original_meta is not None

    local_sum = mpi.xarray.sum(local, dim="lat")
    local_prod = mpi.xarray.prod(local, dim="lat")
    local_mean = mpi.xarray.mean(local, dim="lat", keep_attrs=True)
    local_min = mpi.xarray.min(local, dim="lat")
    local_max = mpi.xarray.max(local, dim="lat")
    local_any = mpi.xarray.any(local > 0, dim="lat")
    local_all = mpi.xarray.all(local >= 0, dim="lat")
    for result in (
        local_sum,
        local_prod,
        local_mean,
        local_min,
        local_max,
        local_any,
        local_all,
    ):
        meta = get_mpi_meta(result)
        assert meta is not None
        assert meta["dim"] == "time"
        assert meta["start"] == original_meta["start"]
        assert meta["stop"] == original_meta["stop"]
    xr.testing.assert_allclose(local_sum, local.sum("lat"))
    xr.testing.assert_allclose(local_prod, local.prod("lat"))
    xr.testing.assert_allclose(local_mean, local.mean("lat", keep_attrs=True))
    xr.testing.assert_allclose(local_min, local.min("lat"))
    xr.testing.assert_allclose(local_max, local.max("lat"))
    xr.testing.assert_equal(local_any, (local > 0).any("lat"))
    xr.testing.assert_equal(local_all, (local >= 0).all("lat"))
    assert local_mean.attrs["units"] == "K"

    try:
        mpi.xarray.mean(local, dim="lat", redistribute_on="lon")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "explicit redistribution must not replace a surviving partition"
        )

    time_sum = mpi.xarray.sum(local, dim="time")
    prod_local = (local % 3.0) + 1.0
    prod_full = (full % 3.0) + 1.0
    time_prod = mpi.xarray.prod(prod_local, dim="time")
    time_mean = mpi.xarray.mean(local, dim="time", keep_attrs=True)
    time_min = mpi.xarray.min(local, dim="time")
    time_max = mpi.xarray.max(local, dim="time")
    time_any = mpi.xarray.any(local > float(full.mean()), dim="time")
    time_all = mpi.xarray.all(local >= 0, dim="time")
    for result in (
        time_sum,
        time_prod,
        time_mean,
        time_min,
        time_max,
        time_any,
        time_all,
    ):
        meta = get_mpi_meta(result)
        assert meta is not None
        assert meta["dim"] == "lon"
    xr.testing.assert_allclose(
        time_sum,
        _expected_partition(full.sum("time"), time_sum),
    )
    xr.testing.assert_allclose(
        time_prod,
        _expected_partition(prod_full.prod("time"), time_prod),
    )
    xr.testing.assert_allclose(
        time_mean,
        _expected_partition(full.mean("time", keep_attrs=True), time_mean),
    )
    xr.testing.assert_allclose(
        time_min,
        _expected_partition(full.min("time"), time_min),
    )
    xr.testing.assert_allclose(
        time_max,
        _expected_partition(full.max("time"), time_max),
    )
    xr.testing.assert_equal(
        time_any,
        _expected_partition((full > float(full.mean())).any("time"), time_any),
    )
    xr.testing.assert_equal(
        time_all,
        _expected_partition((full >= 0).all("time"), time_all),
    )
    assert time_mean.attrs["units"] == "K"

    mixed_mean = mpi.xarray.mean(local, dim=("time", "lat"))
    mixed_min = mpi.xarray.min(local, dim=("time", "lat"))
    mixed_max = mpi.xarray.max(local, dim=("time", "lat"))
    for result in (mixed_mean, mixed_min, mixed_max):
        meta = get_mpi_meta(result)
        assert meta is not None and meta["dim"] == "lon"
    xr.testing.assert_allclose(
        mixed_mean,
        _expected_partition(full.mean(("time", "lat")), mixed_mean),
    )
    xr.testing.assert_allclose(
        mixed_min,
        _expected_partition(full.min(("time", "lat")), mixed_min),
    )
    xr.testing.assert_allclose(
        mixed_max,
        _expected_partition(full.max(("time", "lat")), mixed_max),
    )

    global_max = mpi.xarray.max(local)
    global_min = mpi.xarray.min(local, dim=...)
    assert get_mpi_meta(global_max) is None
    assert get_mpi_meta(global_min) is None
    assert global_max.ndim == 0 and global_min.ndim == 0
    assert global_max.item() == full.max().item()
    assert global_min.item() == full.min().item()

    global_min_count = mpi.xarray.sum(
        local,
        dim="time",
        skipna=True,
        min_count=time_size,
        redistribute_on=None,
    )
    xr.testing.assert_allclose(
        global_min_count,
        full.sum("time", skipna=True, min_count=time_size),
    )

    sparse_time = max(1, mpi.comm.size // 2)
    sparse = xr.DataArray(
        np.column_stack(
            (
                np.full(sparse_time, np.nan),
                np.arange(1, sparse_time + 1, dtype=np.float64),
            )
        ),
        dims=("time", "lon"),
    )
    sparse_local = mpi.xarray.redistribute(sparse, "time")
    sparse_max = mpi.xarray.max(
        sparse_local,
        dim="time",
        skipna=True,
        redistribute_on=None,
    )
    sparse_min = mpi.xarray.min(
        sparse_local,
        dim="time",
        skipna=True,
        redistribute_on=None,
    )
    xr.testing.assert_allclose(sparse_max, sparse.max("time", skipna=True))
    xr.testing.assert_allclose(sparse_min, sparse.min("time", skipna=True))

    replicated = mpi.xarray.mean(local, dim="time", redistribute_on=None)
    assert get_mpi_meta(replicated) is None
    xr.testing.assert_allclose(replicated, full.mean("time"))

    singleton = xr.DataArray(
        np.arange(time_size, dtype=np.float64).reshape(time_size, 1),
        dims=("time", "member"),
    )
    singleton_local = mpi.xarray.redistribute(singleton, "time")
    singleton_sum = mpi.xarray.sum(singleton_local, dim="time")
    assert get_mpi_meta(singleton_sum) is None
    xr.testing.assert_allclose(singleton_sum, singleton.sum("time"))

    static = xr.DataArray(
        np.arange(15, dtype=np.float64).reshape(3, 5),
        dims=("lat", "lon"),
    )
    full_ds = xr.Dataset({"field": full, "static": static}, attrs={"title": "test"})
    local_ds = mpi.xarray.redistribute(full_ds, "time")
    reduced_ds = mpi.xarray.max(local_ds, dim=("time", "lat"), keep_attrs=True)
    reduced_ds_meta = get_mpi_meta(reduced_ds)
    assert reduced_ds_meta is not None and reduced_ds_meta["dim"] == "lon"
    expected_ds = xr.Dataset(
        {
            "field": full.max(("time", "lat"), keep_attrs=True),
            "static": static.max("lat", keep_attrs=True),
        },
        attrs={"title": "test"},
    ).isel(
        lon=slice(int(reduced_ds_meta["start"]), int(reduced_ds_meta["stop"]))
    )
    xr.testing.assert_allclose(reduced_ds, expected_ds)
    assert reduced_ds.attrs["title"] == "test"

    explicit = mpi.xarray.sum(local, dim="time", redistribute_on="lat")
    explicit_meta = get_mpi_meta(explicit)
    assert explicit_meta is not None and explicit_meta["dim"] == "lat"
    xr.testing.assert_allclose(
        explicit,
        _expected_partition(full.sum("time"), explicit),
    )

    for reduction in (
        mpi.xarray.sum,
        mpi.xarray.prod,
        mpi.xarray.mean,
        mpi.xarray.min,
        mpi.xarray.max,
        mpi.xarray.any,
        mpi.xarray.all,
    ):
        parameters = inspect.signature(reduction).parameters
        assert "mode" not in parameters
        assert "root" not in parameters
        assert parameters["redistribute_on"].default == "auto"

    if mpi.comm.rank == 0:
        print("mpi.xarray reduction placement checks passed")


if __name__ == "__main__":
    main()
