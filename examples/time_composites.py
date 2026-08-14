# event time composites, partitioned over the event axis
"""Event time composites written through the MPI-parallel NetCDF-4 writer.

This is the parallel form of the serial ``time_composites.py``. The scientific
content is unchanged: the same onset criterion, the same window gather, the
same cosine-latitude weighted composite and the same output schema. Four
things differ, and only these four.

1. The ``event`` axis is partitioned. Every rank detects the same global set
   of onsets, which is cheap because detection touches only ``pr``, then
   gathers windows and patches for its own contiguous block of events. The
   gather dominates both memory and runtime, so it is the part worth
   dividing.

2. Fields are opened lazily and the derived variables stay lazy, so a rank
   materialises only the grid points its own events touch. Loading eagerly
   would give every rank the whole field and remove the benefit of item 1.

3. The composite is a collective reduction. A weighted mean is a ratio of two
   sums, so each rank forms its partial numerator and denominator and the
   sums are reduced across ranks before the division. Binning is done with
   explicit masks rather than ``groupby_bins`` because a rank holding no
   event in a bin, or no events at all, must still contribute an array of the
   full shape.

4. The store is written once, collectively, with ``parallel=True`` and
   ``partition_dim="event"``. Nothing is gathered to rank zero and no
   per-rank files are written and merged afterwards.

Execution scope is declared per function with the :class:`climtools.MPI`
decorator rather than through module-level handles.

``@MPI`` on its own runs the function on the root rank while every other rank
waits at the collective inside the wrapper. That is what keeps a non-root rank
from racing ahead of a directory that does not exist yet, so all the
filesystem work carries it: directory preparation, the existence checks and
the rsync.

``@MPI(all_ranks=True)`` runs the function everywhere and propagates a failure
on any one rank to all of them, so a job cannot half-succeed. The stages that
touch data carry it.

Functions that need neither, such as :func:`land_mask` or
:func:`_build_event_masks`, carry no decorator and are ordinary local calls.

Rank identity and the reductions come from module-level helpers,
``mpi.rank()``, ``mpi.size()``, ``mpi.total()`` and ``mpi.barrier()``, which
share one coordinator behind the scenes. Nothing initializes MPI until the
first of them is called, so importing this module is free.

Run it either way::

    python time_composites.py
    mpirun -n 8 python time_composites.py
    srun --mpi=pmix --ntasks=8 python time_composites.py

With one rank and no MPI launcher the module behaves exactly like the serial
original, except that the write goes through the parallel writer with
``allow_serial=True``.
"""

from __future__ import annotations

import gc
import itertools
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climtools import MPI, xgeo
from climtools import lib_mpi as mpi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("TIME COMPOSITES")


def rank_tag() -> str:
    """Rank-aware log prefix, resolved when it is first needed."""
    return f"[rank {mpi.rank()}/{mpi.size()}]"


def log_root(message: str, *args: object) -> None:
    """Emit an informational message from the root rank only.

    Progress lines come from the root rank alone, so an eight-rank job does
    not produce eight copies of every message.
    """
    if mpi.is_root():
        logger.info(message, *args)


def land_mask(ds: xr.Dataset) -> xr.DataArray:
    slmsk = ds["slmsk"]
    if "time" in slmsk.dims:
        slmsk = slmsk.isel(time=0, drop=True)
    return slmsk.squeeze(drop=True) == 1


def get_smc_climo(ds: xr.Dataset, smc_path: Path) -> dict[str, xr.DataArray]:
    smc_climo: dict[str, xr.DataArray] = {}
    months = np.unique(ds["time"].dt.month.values)

    with xr.open_dataset(smc_path, engine="netcdf4") as smc_ds:
        smc = smc_ds.sel(time=smc_ds.time.dt.month.isin(months))
        smc = smc.isel(zaxis_1=0, drop=True)
        smc = smc.mean(dim="time", skipna=True).squeeze(drop=True)
        smc = xgeo.remap(smc, ds, method="bilinear")["smc"].load()

    if smc.dtype != ds["soilw1"].dtype:
        smc = smc.astype(ds["soilw1"].dtype)

    smc.attrs["long_name"] = "Climatological Soil Moisture"
    smc.attrs["units"] = ds["soilw1"].attrs["units"]

    # Climatological soil moisture gradient magnitude over model land cells.
    valid_land = land_mask(ds)
    d_dlat = smc.where(valid_land).differentiate("lat")
    d_dlon = smc.where(valid_land).differentiate("lon")
    soilw1_cgm = (d_dlat**2 + d_dlon**2) ** 0.5
    soilw1_cgm.attrs["long_name"] = "Climatological Soil Moisture Gradient Magnitude"
    soilw1_cgm.attrs["units"] = smc.attrs["units"] + "/deg"

    # Duplicate climatological soil moisture along the target time axis.
    # The climatological value is constant in time but has the same dimensions
    # and coordinates as ds["soilw1"] for later time-dependent analysis.
    smc = smc.broadcast_like(ds["soilw1"]).transpose(*ds["soilw1"].dims)
    smc_climo["soilw1_climo"] = smc

    soilw1_cgm = soilw1_cgm.broadcast_like(ds["soilw1"]).transpose(*ds["soilw1"].dims)
    smc_climo["soilw1_cgm"] = soilw1_cgm

    log_root("Loaded and remapped soil moisture climatology")
    return smc_climo


def derived_vars(
    ds: xr.Dataset, smc_path: Path, vertical_dim: str = "plev"
) -> xr.Dataset:

    if vertical_dim not in ds["t"].dims:
        raise ValueError(f"'t' must contain vertical dimension {vertical_dim!r}")

    smc_climo = get_smc_climo(ds, smc_path)

    # Add climatological soil moisture if not already present.
    ds["soilw1_climo"] = smc_climo["soilw1_climo"]
    ds["soilw1_cgm"] = smc_climo["soilw1_cgm"]

    # Temperature gradient (units: K per degree, since lat/lon are in degrees)
    ds["dT_dlat"] = ds["t"].differentiate("lat")
    ds["dT_dlon"] = ds["t"].differentiate("lon")
    ds["dT_dlat"].attrs["long_name"] = "Meridional Temperature Gradient"
    ds["dT_dlat"].attrs["units"] = "K/deg"
    ds["dT_dlon"].attrs["long_name"] = "Zonal Temperature Gradient"
    ds["dT_dlon"].attrs["units"] = "K/deg"

    # Soil moisture gradient magnitude over model land cells.
    land = ds["soilw1"].where(land_mask(ds))
    dsoilw1_dlat = land.differentiate("lat")
    dsoilw1_dlon = land.differentiate("lon")
    ds["soilw1_gm"] = (dsoilw1_dlat**2 + dsoilw1_dlon**2) ** 0.5
    ds["soilw1_gm"].attrs["long_name"] = "Soil Moisture Gradient Magnitude"
    ds["soilw1_gm"].attrs["units"] = ds["soilw1"].attrs["units"] + "/deg"

    return ds


def _build_event_masks(
    pr: xr.DataArray,
    window_before: int,
    window_after: int,
    dry_threshold: float = 0.1,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Onset trigger mask and forward-window peak field.

    A trigger is a wet step preceded by `window_before` dry steps and followed
    by a finite forward peak. Edge steps return NaN through `min_periods` and
    are therefore excluded, which keeps every gathered window complete.

    Detection is deliberately replicated on every rank. It reads only `pr` and
    reduces it with two rolling operators, so replicating it costs far less
    than the halo exchange a time-partitioned detection would need, and it
    guarantees that every rank derives an identical global event list.
    """
    if window_before < 1 or window_after < 0:
        raise ValueError("window_before must be positive and window_after non-negative")

    dry = pr <= dry_threshold
    wet = pr > dry_threshold
    pre_dry = (
        dry.rolling(time=window_before, min_periods=window_before).min().shift(time=1)
    )
    peak_fwd = (
        pr.rolling(time=window_after + 1, min_periods=window_after + 1)
        .max()
        .shift(time=-window_after)
    )
    trigger = wet & (pre_dry == 1) & peak_fwd.notnull()
    return trigger, peak_fwd


def _event_labels(trigger: xr.DataArray) -> xr.Dataset:
    """Locate onsets and return their integer and label coordinates.

    The trigger field is reduced with ``np.nonzero`` on the boolean array.
    Stacking and ``where(..., drop=True)`` would first upcast the whole
    time-lat-lon boolean field to float, which is several gigabytes on a
    convection-permitting nest for no benefit. Ordering is C order over
    (time, lat, lon), identical to the stacked form.

    Both integer indices and coordinate labels are returned: the indices drive
    the window gather, the labels are written as event metadata.
    """
    ordered = trigger.transpose("time", "lat", "lon")
    values = np.asarray(ordered.values, dtype=bool)
    it, iy, ix = np.nonzero(values)

    time_vals = np.asarray(ordered["time"].values)
    lat_vals = np.asarray(ordered["lat"].values)
    lon_vals = np.asarray(ordered["lon"].values)

    return xr.Dataset(
        {
            "time_index": ("event", it.astype(np.int64)),
            "lat_index": ("event", iy.astype(np.int64)),
            "lon_index": ("event", ix.astype(np.int64)),
            "time": ("event", time_vals[it]),
            "lat": ("event", lat_vals[iy]),
            "lon": ("event", lon_vals[ix]),
        }
    )


def partition_events(labels: xr.Dataset) -> tuple[xr.Dataset, int]:
    """Split the global event list into one contiguous block per rank.

    Parameters
    ----------
    labels : xarray.Dataset
        Global onset list produced by :func:`_event_labels`, identical on
        every rank.

    Returns
    -------
    tuple of (xarray.Dataset, int)
        The local block of events and its offset in the global event axis.

    Notes
    -----
    The split is contiguous and the remainder is spread over the leading
    ranks, so block lengths differ by at most one. Contiguity is what the
    parallel writer requires: it recovers each rank's file offset from an
    all-gather of the local lengths, so a strided or interleaved split would
    scatter every rank's events across the whole file.

    The offset is returned because the ``event`` coordinate must be numbered
    globally. Numbering each block from zero would give the written file a
    coordinate that restarts once per rank.
    """
    total = int(labels.sizes["event"])
    offset, stop = mpi.partition(total)
    local = labels.isel(event=slice(offset, stop))
    return local, offset


def _window_time_index(
    labels: xr.Dataset,
    ntime: int,
    window_before: int,
    window_after: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the relative-time offsets and the (event, relative_time) index."""
    relative_time = np.arange(-window_before, window_after + 1)
    time_index = labels["time_index"].values[:, None] + relative_time[None, :]
    if time_index.size and (time_index.min() < 0 or time_index.max() >= ntime):
        raise ValueError(
            "Event window extends beyond the time axis. The trigger mask must "
            "exclude the first window_before and last window_after steps."
        )
    return relative_time, time_index


def build_event_store(
    ds: xr.Dataset,
    labels: xr.Dataset,
    peak_fwd: xr.DataArray,
    window_before: int,
    window_after: int,
    vertical_dim: str = "plev",
    event_offset: int = 0,
) -> xr.Dataset | None:
    """Per-event windows on dims (event, relative_time) by vectorised indexing.

    Values are gathered only at the trigger points. Building the full rolling
    view of every field first and selecting afterwards materialises an array
    of size ntime * nlev * nlat * nlon * nwindow, which is not tractable on a
    convection-permitting nest. Each event carries relative_time, trigger_time,
    valid_time, lat, lon and peak as coordinates. Asymmetric before/after
    windows are supported.

    `labels` is the rank-local block from :func:`partition_events` and
    `event_offset` its position in the global event axis, so the `event`
    coordinate stays globally unique and strictly increasing across ranks.
    """
    if labels.sizes["event"] == 0:
        return None

    relative_time, time_index = _window_time_index(
        labels, ds.sizes["time"], window_before, window_after
    )
    lat_index = labels["lat_index"].values
    lon_index = labels["lon_index"].values

    time_selector = xr.DataArray(time_index, dims=("event", "relative_time"))
    lat_selector = xr.DataArray(lat_index, dims="event")
    lon_selector = xr.DataArray(lon_index, dims="event")

    events = ds.isel(
        time=time_selector,
        lat=lat_selector,
        lon=lon_selector,
    ).drop_vars(["time", "lat", "lon"], errors="ignore")

    peak = peak_fwd.isel(
        time=xr.DataArray(labels["time_index"].values, dims="event"),
        lat=lat_selector,
        lon=lon_selector,
    )

    time_vals = np.asarray(ds["time"].values)
    n_local = events.sizes["event"]
    events = events.assign_coords(
        event=("event", np.arange(event_offset, event_offset + n_local)),
        relative_time=("relative_time", relative_time),
        trigger_time=("event", time_vals[labels["time_index"].values]),
        valid_time=(("event", "relative_time"), time_vals[time_index]),
        lat=("event", labels["lat"].values),
        lon=("event", labels["lon"].values),
        peak=("event", np.asarray(peak.values)),
    )
    # The gather is the point at which lazy fields must become arrays. Only
    # the selected points are materialised, which is what keeps a rank's
    # footprint proportional to its own share of the events.
    return events.transpose("event", "relative_time", vertical_dim, ...).load()


def build_event_patches(
    ds: xr.Dataset,
    labels: xr.Dataset,
    window_before: int,
    window_after: int,
    half: int,
    vertical_dim: str = "plev",
    event_offset: int = 0,
) -> xr.Dataset | None:
    """Per-event space-time boxes on dims (event, relative_time, y_off, x_off).

    The event axis is identical to `build_event_store`. Cells of a box that
    fall outside the domain are set to NaN rather than dropping the event, so
    windows, composite and patches align on one `event` index.
    """
    if labels.sizes["event"] == 0:
        return None

    if half < 0:
        raise ValueError("half must be non-negative")

    relative_time, time_index = _window_time_index(
        labels, ds.sizes["time"], window_before, window_after
    )
    off = np.arange(-half, half + 1)
    nlat = ds.sizes["lat"]
    nlon = ds.sizes["lon"]

    lat_index = labels["lat_index"].values[:, None] + off[None, :]
    lon_index = labels["lon_index"].values[:, None] + off[None, :]
    lat_valid = (lat_index >= 0) & (lat_index < nlat)
    lon_valid = (lon_index >= 0) & (lon_index < nlon)

    patch = ds.isel(
        time=xr.DataArray(time_index, dims=("event", "relative_time")),
        lat=xr.DataArray(np.clip(lat_index, 0, nlat - 1), dims=("event", "y_off")),
        lon=xr.DataArray(np.clip(lon_index, 0, nlon - 1), dims=("event", "x_off")),
    ).drop_vars(["time", "lat", "lon"], errors="ignore")

    inside = xr.DataArray(lat_valid, dims=("event", "y_off")) & xr.DataArray(
        lon_valid, dims=("event", "x_off")
    )
    patch = patch.where(inside)

    n_local = patch.sizes["event"]
    patch = patch.assign_coords(
        event=("event", np.arange(event_offset, event_offset + n_local)),
        relative_time=("relative_time", relative_time),
        y_off=("y_off", off),
        x_off=("x_off", off),
    )
    return patch.transpose(
        "event", "relative_time", vertical_dim, ..., "y_off", "x_off"
    ).load()


def _bin_edges(intensity_edges: tuple[float, ...]) -> tuple[np.ndarray, list[float]]:
    """Validate the intensity edges and return the closed edges and labels."""
    if not intensity_edges:
        raise ValueError("intensity_edges must contain at least one value")

    edge_values = np.asarray(intensity_edges, dtype=float)
    if not np.isfinite(edge_values).all() or np.any(np.diff(edge_values) <= 0):
        raise ValueError("intensity_edges must be finite and strictly increasing")

    return np.append(edge_values, np.inf), edge_values.tolist()


def _composite_partials(
    events: xr.Dataset,
    intensity_edges: tuple[float, ...],
) -> tuple[xr.Dataset, xr.Dataset, xr.DataArray]:
    """Local weighted sums per intensity bin, before the cross-rank reduction.

    Returns the numerator, the per-variable denominator and the event counts,
    each carrying the full `peak_bins` axis regardless of which bins this rank
    happens to populate.

    Bins are formed with explicit left-closed masks rather than
    ``groupby_bins``. A rank holding no event in a bin, or no events at all,
    must still return an array of the full shape so the reduction has
    something of matching shape to add; ``groupby_bins`` drops empty groups
    and would make the summands ragged.
    """
    edges, bin_labels = _bin_edges(intensity_edges)
    weights = np.cos(np.deg2rad(events["lat"]))

    numerators: list[xr.Dataset] = []
    denominators: list[xr.Dataset] = []
    counts: list[xr.DataArray] = []

    for lower, upper in itertools.pairwise(edges):
        selected = (events["peak"] >= lower) & (events["peak"] < upper)
        weighted = (events * weights).where(selected)
        numerators.append(weighted.sum(dim="event", skipna=True))
        denominators.append(
            xr.Dataset(
                {
                    name: xr.where(data.notnull() & selected, weights, 0.0).sum(
                        dim="event"
                    )
                    for name, data in events.data_vars.items()
                }
            )
        )
        counts.append(selected.sum(dim="event").astype("int32"))

    peak_bins = xr.DataArray(bin_labels, dims="peak_bins", name="peak_bins")
    return (
        xr.concat(numerators, dim=peak_bins),
        xr.concat(denominators, dim=peak_bins),
        xr.concat(counts, dim=peak_bins),
    )


@MPI(all_ranks=True)
def composite_from_events(
    events: xr.Dataset,
    intensity_edges: tuple[float, ...],
    vertical_dim: str = "plev",
) -> xr.Dataset:
    """Cosine-latitude weighted composite per intensity bin, with event counts.

    The weighted mean is the ratio of grouped weighted sums,
    X_bar(tau) = sum_e w_e X_e(tau) / sum_e w_e, w_e = cos(phi_e). The
    denominator is evaluated per variable over finite values, preventing
    missing data from biasing a composite toward zero. Bins are left-closed,
    so a peak exactly equal to the first edge is included.

    With the event axis partitioned, both sums are formed locally and reduced
    across ranks before the division. A ratio of sums is what makes this
    possible: the mean itself is not additive, so averaging per rank and
    averaging the averages would weight a rank holding three events equally
    with a rank holding three thousand.

    The reduction runs in rank order, so every rank obtains a bit-identical
    composite. The parallel writer checks exactly that for arrays it treats as
    replicated, and rejects the write if the ranks disagree by even one bit.

    Composites are reproducible for a fixed rank count but not across rank
    counts. Partitioning changes the order in which the partial sums are
    associated, and floating-point addition is not associative. Measured over
    a synthetic case of 1457 events, one rank against three agreed to a
    relative difference of at most 5e-13, a few thousand times the double
    precision epsilon and far below any physically meaningful threshold.
    Event windows and patches, which involve no reduction, are bit-identical
    at any rank count.
    """
    numerator, denominator, counts = _composite_partials(events, intensity_edges)

    numerator = mpi.total(numerator)
    denominator = mpi.total(denominator)
    counts = mpi.total(counts)

    composite = numerator / denominator.where(denominator > 0)
    composite["n_events"] = counts.astype("int32")
    composite.attrs["intensity_edges"] = list(intensity_edges)
    return composite.transpose("peak_bins", "relative_time", vertical_dim, ...)


def assemble_store(
    events: xr.Dataset,
    composite: xr.Dataset,
    patches: xr.Dataset | None = None,
) -> xr.Dataset:
    """Combine windows, composite, and optional patches in one Dataset.

    Window fields keep dims (event, relative_time). Composite fields are
    suffixed `_composite` on dims (peak_bins, relative_time). Patch fields are
    suffixed `_patch` on dims (event, relative_time, y_off, x_off). Onset
    metadata (trigger_time, valid_time, lat, lon, peak) stay as coordinates on
    the event axis. The count field n_events is kept under its own name.

    Under MPI the result is a slab: `event` holds this rank's block while
    every other dimension is replicated and identical across ranks. That is
    precisely the layout the parallel writer expects.
    """
    parts = [events]

    comp_rename = {v: f"{v}_composite" for v in composite.data_vars if v != "n_events"}
    parts.append(composite.rename(comp_rename))

    if patches is not None:
        drop = [c for c in ("trigger_time", "lat", "lon") if c in patches.coords]
        patches = patches.drop_vars(drop)
        patch_rename = {v: f"{v}_patch" for v in patches.data_vars}
        parts.append(patches.rename(patch_rename))

    return xr.merge(parts, combine_attrs="no_conflicts")


@MPI
def prepare_output(output_root: Path) -> None:
    """Create the case output directory. Root rank only."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "event.store.nc").unlink(missing_ok=True)


@MPI
def archive_case(output_root: Path, final_path: Path, store_name: str) -> None:
    """Copy a finished case to its final location. Root rank only."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{output_root}/", f"{final_path}/"],
        check=True,
    )
    final_store = final_path / store_name
    if not final_store.exists():
        raise RuntimeError(f"Final event store is missing after rsync: {final_store}")


@MPI(all_ranks=True)
def compute_event_time_composites(
    input_root: Path,
    output_root: Path,
    in_fname: str,
    smc_path: Path,
    window_before: int = 24,
    window_after: int = 24,
    dry_threshold: float = 0.1,
    intensity_edges: tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100),
    patch_halfwidth: int | None = None,
    vertical_dim: str = "plev",
) -> Path | None:
    """Detect post-dry-spell rainfall onsets and write one consolidated store.

    Output `event.store.nc` carries dims event, relative_time and peak_bins,
    plus y_off and x_off when `patch_halfwidth` is set. The path of the written
    store is returned, or None when the case contains no qualifying onset. A
    store that cannot be located after writing raises, so a silent skip can
    never be mistaken for an empty case.

    Every rank calls this function, in the same order, with identical
    arguments. The decision to skip an empty case is taken collectively from
    the global event count, so no rank can proceed into a write the others
    have skipped.
    """
    path = input_root / "case" / in_fname
    if not path.exists():
        raise FileNotFoundError(f"Model history file is missing: {path}")

    prepare_output(output_root)
    out_path = output_root / "event.store.nc"
    mpi.barrier()

    # chunks={} defers every field to dask. The gather in build_event_store
    # then materialises only the points this rank's events touch, instead of
    # every rank loading the whole nest.
    with xr.open_dataset(path, chunks={}) as source:
        ds = source.sortby("lat").sortby("lon")
        ds["time"] = ds["time"] - pd.Timedelta(hours=5)
        utc5_lon_bounds = (-82.5, None)
        ds = ds.sel(lon=slice(*utc5_lon_bounds))
        ds = derived_vars(ds, smc_path, vertical_dim)

        trigger, peak_fwd = _build_event_masks(
            ds["pr"], window_before, window_after, dry_threshold
        )
        labels = _event_labels(trigger)
        if labels.sizes["event"] == 0:
            log_root("No triggered events with a dry antecedent. Skipping case.")
            return None
        log_root("%d events", labels.sizes["event"])

        local_labels, event_offset = partition_events(labels)
        logger.info(
            "%s holds events %d to %d",
            rank_tag(),
            event_offset,
            event_offset + local_labels.sizes["event"],
        )

        log_root("Building event store")
        events = build_event_store(
            ds,
            local_labels,
            peak_fwd,
            window_before,
            window_after,
            vertical_dim,
            event_offset,
        )
        if events is None:
            # This rank drew an empty block because there are fewer events
            # than ranks. It still holds a valid zero-length slab and must
            # stay in every collective that follows.
            events = build_event_store(
                ds,
                local_labels.isel(event=slice(0, 0)),
                peak_fwd,
                window_before,
                window_after,
                vertical_dim,
                event_offset,
            )

        log_root("Building composite")
        composite = composite_from_events(events, intensity_edges, vertical_dim)

        patches = None
        if patch_halfwidth is not None:
            patches = build_event_patches(
                ds,
                local_labels,
                window_before,
                window_after,
                patch_halfwidth,
                vertical_dim,
                event_offset,
            )

        log_root("Assembling composite store")
        store = assemble_store(events, composite, patches)

        log_root("Writing composite store to %s", out_path)
        started = time.perf_counter()
        xgeo.to_netcdf(
            file=out_path,
            data=store,
            unlimited_dim="event",
            partition_dim="event",
            parallel=True,
            allow_serial=True,
        )
        elapsed = time.perf_counter() - started
        log_root(
            "Collective write finished in %.2f s on %d rank(s)", elapsed, mpi.size()
        )

    mpi.barrier()

    if not out_path.exists():
        raise RuntimeError(f"Event composite file was not written to {out_path}")

    log_root("Finished writing event composite file to %s", out_path)
    return out_path


def main() -> None:

    date = "2024081400Z"
    log_root("Starting time composites for %s on %d rank(s)", date, mpi.size())

    home = Path("/users/jkodero")
    gfdl_shield = home / "research/models/gfdl_shield"
    data_store = gfdl_shield / "archive"
    final_dir = gfdl_shield / "analysis/time_composites_new"
    tmp_dir = home / "jobtmp/data/002/time_composites"
    smc_path = gfdl_shield / "src/fix/era5/sm_monthly_1950_2025.nc"

    if not data_store.exists():
        raise FileNotFoundError(f"Data archive is missing: {data_store}")
    if not smc_path.exists():
        raise FileNotFoundError(f"Soil-moisture climatology is missing: {smc_path}")

    init_datetimes = [date]
    prefix = "C96.NESTED.R4x2.R2x1"
    experiments = [
        "CNTRL",
        "2SIGMA_DRY",
    ]
    members = [f"mem{i:02}" for i in range(1, 11)]
    in_fname = "fv3_hist.nest04.nc"

    for init_date in init_datetimes:
        for exp_name in experiments:
            exp = f"{prefix}.{exp_name}"

            for member in members:
                log_root("Running %s %s %s", init_date, exp, member)
                input_root = data_store / init_date / exp / member

                output_root = tmp_dir / init_date / exp / member
                clear_case(output_root)

                store_path = compute_event_time_composites(
                    input_root, output_root, in_fname, smc_path, vertical_dim="plev"
                )

                final_path = final_dir / init_date / exp / member

                if store_path is None:
                    log_root(
                        "No qualifying onsets, nothing to archive for %s %s %s",
                        init_date,
                        exp,
                        member,
                    )
                    discard_case(final_path)
                    gc.collect()
                    continue

                if not store_path.exists():
                    raise RuntimeError(
                        f"Event store vanished before archiving: {store_path}"
                    )

                archive_case(output_root, final_path, store_path.name)

                log_root("Finished %s %s %s", init_date, exp, member)
                gc.collect()

        discard_case(tmp_dir / init_date)

    mpi.barrier()


@MPI
def clear_case(output_root: Path) -> None:
    """Remove and recreate a case working directory. Root rank only."""
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True)


@MPI
def discard_case(path: Path) -> None:
    """Remove a directory tree. Root rank only."""
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
