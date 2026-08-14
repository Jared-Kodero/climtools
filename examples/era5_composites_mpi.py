"""Worked example: a distributed composite workflow with a timed output stage.

The script computes warm-day composites of a gridded field and writes one
NetCDF-4 file, and is meant to be read as a template for converting an existing
serial analysis. It runs unchanged in both worlds::

    python -m climtools.examples.era5_composites_mpi
    mpirun -n 8 python -m climtools.examples.era5_composites_mpi
    srun --mpi=pmix --ntasks=8 python -m climtools.examples.era5_composites_mpi

Three kinds of function appear here on purpose, because the distinction is the
part that is easy to get wrong:

``@mpi(all_ranks=True)``
    Collective. Every rank runs the body on its own slab and every rank must
    call it, in the same order. Used for the stages that touch data.

``@mpi()`` and ``@mpi(broadcast=True)``
    The body runs on the root alone. Every rank still calls the wrapper,
    because the wrapper itself communicates. Used for reporting, and for
    decisions that must be identical everywhere, such as the output path.

undecorated
    Ordinary functions. Most of the analysis is of this kind: it is pure array
    work on data the rank already holds, needs no communication, and is
    therefore both simpler and faster left alone. Decorating these would add
    collectives for nothing.

The time axis is partitioned. Each rank holds a contiguous, non-overlapping
block of time steps and identical copies of the coordinates, which is exactly
the layout the parallel writer expects; it recovers the global offsets itself
from an all-gather of the local lengths.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr

from climtools import lib_mpi
from climtools.lib_mpi import mpi
from climtools.lib_netcdf import to_netcdf_serial
from climtools.lib_netcdf.parallel import to_netcdf_parallel

RANK, SIZE = lib_mpi.world()
ROOT = RANK == 0

#: Percentile above which a day counts as warm at a grid point.
WARM_PERCENTILE = 90.0


# --------------------------------------------------------------- decomposition


def slab_bounds(total: int, rank: int, size: int) -> tuple[int, int]:
    """Return the half-open slab this rank owns along the partitioned axis.

    Parameters
    ----------
    total : int
        Global length of the partitioned dimension.
    rank, size : int
        Position in and size of ``MPI_COMM_WORLD``.

    Returns
    -------
    tuple of int
        ``(start, stop)``. The remainder is spread over the leading ranks, so
        the slabs remain contiguous and their lengths differ by at most one.
        Contiguity is what lets the writer reconstruct the global offsets.
    """
    base, remainder = divmod(total, size)
    start = rank * base + min(rank, remainder)
    return start, start + base + (1 if rank < remainder else 0)


# ------------------------------------------------------- stages that use MPI


@mpi(broadcast=True)
def resolve_output_directory(requested: str | None) -> str:
    """Choose the output directory on the root and give it to every rank.

    Parameters
    ----------
    requested : str or None
        Directory supplied on the command line, or ``None``.

    Returns
    -------
    str
        Directory every rank will write into.

    Notes
    -----
    Broadcasting matters more than it looks. If each rank derived a path from
    its own clock or its own environment, the ranks would open different files
    and the collective write would fail, or worse, succeed against a path only
    some of them agree on.
    """
    directory = Path(requested) if requested else Path.cwd() / "composites_output"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def build_dataset(
    n_time: int,
    n_lat: int,
    n_lon: int,
    start: int,
    stop: int,
) -> xr.Dataset:
    """Build the time block ``[start, stop)`` of the global field.

    Parameters
    ----------
    n_time, n_lat, n_lon : int
        Shape of the global field.
    start, stop : int
        Half-open range of global time indices to materialise.

    Returns
    -------
    xarray.Dataset
        Block of the field with the full spatial grid.

    Notes
    -----
    Synthetic data stands in for a read so the example has no external
    dependency. In a real workflow the body becomes, for instance::

        files = sorted(Path(archive).glob("t2m_*.nc"))
        ds = xr.open_mfdataset(files, combine="by_coords")
        return ds.isel(time=slice(start, stop))

    The seed follows the global offset, so a block has the same values however
    many ranks the job uses. That is what makes the parallel and serial
    results comparable.
    """
    length = stop - start
    time_axis = np.datetime64("1990-01-01T00", "ns") + (
        np.arange(start, stop) * np.timedelta64(6, "h").astype("timedelta64[ns]")
    )
    lat = np.linspace(-89.5, 89.5, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)

    hours = (np.arange(start, stop) * 6) % 8766
    seasonal = 12.0 * np.cos(2.0 * np.pi * hours / 8766.0)
    meridional = 25.0 * np.cos(np.deg2rad(lat))

    rng = np.random.default_rng(1000)
    noise = rng.normal(0.0, 2.0, size=(n_time, n_lat, n_lon))[start:stop]

    t2m = (273.15 + meridional[None, :, None] + seasonal[:, None, None] + noise).astype(
        "float32"
    )
    assert t2m.shape == (length, n_lat, n_lon)

    ds = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), t2m)},
        coords={"time": time_axis, "lat": lat, "lon": lon},
        attrs={
            "title": "Synthetic 6-hourly near-surface temperature",
            "Conventions": "CF-1.8",
        },
    )
    ds["t2m"].attrs = {"units": "K", "long_name": "2 metre temperature"}
    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return ds


@mpi(all_ranks=True)
def load_slab(n_time: int, n_lat: int, n_lon: int) -> xr.Dataset:
    """Produce the time slab this rank owns.

    Parameters
    ----------
    n_time, n_lat, n_lon : int
        Shape of the global field.

    Returns
    -------
    xarray.Dataset
        Local slab with the full spatial grid and a contiguous block of time.

    Notes
    -----
    Only this rank's slab is materialised, so the read scales with the job in
    the same way the write does.
    """
    start, stop = slab_bounds(n_time, RANK, SIZE)
    return build_dataset(n_time, n_lat, n_lon, start, stop)


@mpi(all_ranks=True)
def global_warm_threshold(ds: xr.Dataset) -> xr.DataArray:
    """Compute the warm-day threshold from the whole record, not one slab.

    Parameters
    ----------
    ds : xarray.Dataset
        Local slab holding ``t2m``.

    Returns
    -------
    xarray.DataArray
        Threshold field on ``(lat, lon)``, identical on every rank.

    Notes
    -----
    This is the stage that genuinely needs communication. A percentile taken
    over a rank's own time block is a percentile of that block, so ranks would
    apply different thresholds to different parts of one record and the
    composite would be an artefact of the decomposition.

    An exact distributed percentile requires a global sort. The pooled moment
    estimate used here is cheap, deterministic and sufficient for a threshold;
    the point is that the statistic is formed from global sums, so every rank
    ends with the same field.

    .. math::

        \\bar{x} = \\frac{1}{N}\\sum_{r} n_r \\bar{x}_r, \\qquad
        s^2 = \\frac{1}{N}\\sum_{r} n_r
        \\left( s_r^2 + (\\bar{x}_r - \\bar{x})^2 \\right)

    where :math:`n_r`, :math:`\\bar{x}_r` and :math:`s_r^2` are the count, mean
    and variance on rank :math:`r`, :math:`N=\\sum_r n_r`, and the threshold is
    :math:`\\bar{x} + z\\,s` with :math:`z` the standard normal quantile at
    ``WARM_PERCENTILE``. Units are kelvin throughout.
    """
    from climtools.lib_mpi.native import allgather_i64, bcast_obj

    field = ds["t2m"]
    count = int(field.sizes["time"])
    # Accumulate in float64 so the moments do not depend on how many
    # time steps a rank happens to hold.
    wide = field.astype("float64")
    local_mean = wide.mean("time").values
    local_var = wide.var("time").values

    if SIZE == 1:
        pooled_mean, pooled_var = local_mean, local_var
    else:
        counts = allgather_i64(count, SIZE)
        total = float(sum(counts))

        # One broadcast per rank moves that rank's whole moment field. Doing
        # this point by point would issue one collective per grid cell and
        # dominate the run time.
        means = np.empty((SIZE, *local_mean.shape), dtype="float64")
        variances = np.empty((SIZE, *local_var.shape), dtype="float64")
        for source in range(SIZE):
            payload = (local_mean, local_var) if RANK == source else None
            gathered_mean, gathered_var = bcast_obj(payload, source)
            means[source] = gathered_mean
            variances[source] = gathered_var

        weights = np.asarray(counts, dtype="float64")[
            (slice(None), *([None] * local_mean.ndim))
        ]
        pooled_mean = (weights * means).sum(axis=0) / total
        pooled_var = (
            weights * (variances + (means - pooled_mean[None, ...]) ** 2)
        ).sum(axis=0) / total

    z = float(np.sqrt(2.0) * _erfinv(2.0 * WARM_PERCENTILE / 100.0 - 1.0))
    threshold = pooled_mean + z * np.sqrt(np.maximum(pooled_var, 0.0))

    return xr.DataArray(
        threshold.astype("float32"),
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        name="warm_threshold",
        attrs={
            "units": "K",
            "long_name": f"{WARM_PERCENTILE:g}th percentile temperature",
            "comment": "pooled global moment estimate over all ranks",
        },
    )


@mpi()
def report(summary: dict[str, float]) -> None:
    """Print the run summary from the root rank only.

    Parameters
    ----------
    summary : dict of str to float
        Values gathered by the caller.

    Notes
    -----
    Every rank calls this wrapper; only the root executes the body. Printing
    from every rank interleaves output lines unreadably, and writing a figure
    or a log file from every rank would have them fight over one path.
    """
    print("\n  composite summary")
    for key, value in summary.items():
        print(f"    {key:<28} {value:,.3f}")


# --------------------------------------------------- stages that do not use MPI


def to_anomalies(ds: xr.Dataset, threshold: xr.DataArray) -> xr.Dataset:
    """Convert temperatures to departures from the warm threshold.

    Parameters
    ----------
    ds : xarray.Dataset
        Local slab holding ``t2m`` in kelvin.
    threshold : xarray.DataArray
        Threshold field, identical on every rank.

    Returns
    -------
    xarray.Dataset
        Slab with an added anomaly field in kelvin.

    Notes
    -----
    Undecorated on purpose. The rank already holds everything this needs, so
    adding a collective here would cost synchronisation and buy nothing.
    """
    anomaly = ds["t2m"] - threshold
    anomaly.attrs = {
        "units": "K",
        "long_name": "temperature departure from the warm threshold",
    }
    return ds.assign(t2m_anomaly=anomaly)


def warm_composite(ds: xr.Dataset) -> xr.Dataset:
    """Mask each time step to the grid points exceeding the threshold.

    Parameters
    ----------
    ds : xarray.Dataset
        Slab carrying ``t2m_anomaly``.

    Returns
    -------
    xarray.Dataset
        Slab with the composite field and a per-step warm-point count.

    Notes
    -----
    Also undecorated, and also purely local. The number of warm points per
    time step is retained so the root can report a coverage figure without a
    second pass over the data.
    """
    warm = ds["t2m_anomaly"] > 0.0
    composite = ds["t2m_anomaly"].where(warm)
    composite.attrs = {
        "units": "K",
        "long_name": "warm-day temperature anomaly composite",
        "cell_methods": f"time: point where t2m > p{WARM_PERCENTILE:g}",
    }
    fraction = warm.mean(dim=("lat", "lon")).astype("float32")
    fraction.attrs = {"units": "1", "long_name": "fraction of warm grid points"}
    return ds.assign(warm_composite=composite, warm_fraction=fraction)


def _erfinv(value: float) -> float:
    """Return the inverse error function without requiring SciPy.

    Parameters
    ----------
    value : float
        Argument in the open interval ``(-1, 1)``.

    Returns
    -------
    float
        ``erf^{-1}(value)``, from the Giles (2010) rational approximation,
        accurate to about seven decimal digits over the range used here.
    """
    w = -np.log(np.maximum(1.0 - value * value, 1e-300))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for coefficient in (
            3.43273939e-07,
            -3.5233877e-06,
            -4.39150654e-06,
            0.00021858087,
            -0.00125372503,
            -0.00417768164,
            0.246640727,
            1.50140941,
        ):
            p = p * w + coefficient
    else:
        w = np.sqrt(w) - 3.0
        p = -0.000200214257
        for coefficient in (
            0.000100950558,
            0.00134934322,
            -0.00367342844,
            0.00573950773,
            -0.0076224613,
            0.00943887047,
            1.00167406,
            2.83297682,
        ):
            p = p * w + coefficient
    return float(p * value)


# ------------------------------------------------------------------ benchmark


def time_parallel_write(ds: xr.Dataset, path: Path, deflate: int | None) -> float:
    """Time one collective write, measured from the slowest rank.

    Parameters
    ----------
    ds : xarray.Dataset
        Local slab to write.
    path : pathlib.Path
        Output path shared by all ranks.
    deflate : int or None
        Compression level, or ``None``.

    Returns
    -------
    float
        Wall-clock seconds. Barriers on both sides make this the time of the
        slowest rank rather than of whichever rank happened to finish first,
        which is the only figure that means anything for a collective.
    """
    world = lib_mpi.MPI()
    world.barrier()
    start = time.perf_counter()
    to_netcdf_parallel(
        ds,
        path,
        deflate=deflate,
        allow_serial=SIZE == 1,
    )
    world.barrier()
    return time.perf_counter() - start


def time_serial_write(ds: xr.Dataset, path: Path, deflate: int | None) -> float:
    """Time the equivalent serial write of the whole dataset on one rank.

    Parameters
    ----------
    ds : xarray.Dataset
        Global dataset.
    path : pathlib.Path
        Output path.
    deflate : int or None
        Compression level, or ``None``.

    Returns
    -------
    float
        Wall-clock seconds.
    """
    start = time.perf_counter()
    to_netcdf_serial(
        ds,
        path,
        unlimited_dim="time",
        zlib=deflate is not None,
        complevel=deflate or 4,
        show_progress=False,
    )
    return time.perf_counter() - start


# ----------------------------------------------------------------------- main


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--time", type=int, default=480, help="global time steps")
    parser.add_argument("--lat", type=int, default=181, help="latitude points")
    parser.add_argument("--lon", type=int, default=360, help="longitude points")
    parser.add_argument("--output", default=None, help="output directory")
    parser.add_argument(
        "--deflate",
        type=int,
        default=None,
        help="compression level; needs parallel HDF5 filters",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="also write the same data serially and compare",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the workflow and return a process exit code."""
    arguments = parse_arguments(argv)

    if ROOT:
        info = lib_mpi.info()
        print(f"climtools composite workflow: {SIZE} rank(s)")
        print(f"  NetCDF-C            {info['netcdf']}")
        print(f"  parallel filters    {info['parallel_filters']}")
        print(
            f"  grid                {arguments.time} x {arguments.lat} x "
            f"{arguments.lon}"
        )

    directory = Path(str(resolve_output_directory(arguments.output)))

    # Stages that need MPI.
    slab = load_slab(arguments.time, arguments.lat, arguments.lon)
    threshold = global_warm_threshold(slab)

    # Stages that do not. Plain functions on data the rank already holds.
    slab = to_anomalies(slab, threshold)
    slab = warm_composite(slab)
    slab = slab.assign(warm_threshold=threshold)
    slab["t2m"].encoding.clear()

    output = directory / "warm_composites.nc"
    elapsed = time_parallel_write(slab, output, arguments.deflate)

    # Two float32 fields of the global shape reach the file.
    total_bytes = float(arguments.time * arguments.lat * arguments.lon * 4 * 2)

    summary = {
        "ranks": float(SIZE),
        "local time steps": float(slab.sizes["time"]),
        "payload (MiB)": total_bytes / 1024**2,
        "parallel write (s)": elapsed,
        "parallel throughput (MiB/s)": total_bytes / 1024**2 / max(elapsed, 1e-9),
        "mean warm fraction": float(slab["warm_fraction"].mean()),
    }

    if arguments.benchmark:
        # The serial reference writes the whole dataset from one rank, which
        # is the operation the parallel path replaces.
        gather = mpi(broadcast=True)

        @gather
        def serial_reference() -> float:
            whole = build_dataset(
                arguments.time, arguments.lat, arguments.lon, 0, arguments.time
            )
            whole = to_anomalies(whole, threshold)
            whole = warm_composite(whole)
            whole["t2m"].encoding.clear()
            return time_serial_write(
                whole,
                directory / "warm_composites_serial.nc",
                arguments.deflate,
            )

        serial_elapsed = float(serial_reference())
        summary["serial write (s)"] = serial_elapsed
        summary["serial throughput (MiB/s)"] = (
            total_bytes / 1024**2 / max(serial_elapsed, 1e-9)
        )
        summary["speed-up"] = serial_elapsed / max(elapsed, 1e-9)

    report(summary)

    if ROOT:
        print(f"\n  wrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
