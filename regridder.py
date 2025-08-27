import difflib
import subprocess
import uuid
from os import PathLike

import numpy as np
import xarray as xr
import xesmf as xe

from .logs import log
from .tools import _TMP_FILES, CPU_COUNT, CWD


def ESMF_RegridWeightGen(
    **kwargs,
):
    """
    Generate regridding weights using ESMF_RegridWeightGen.

    This function wraps the ESMF_RegridWeightGen command-line tool, allowing you to
    specify all options as keyword arguments. Both long and short forms of parameters
    are supported (e.g., `source` or `s`).

    For questions, comments, or feature requests please send email to:
    esmf_support@cgd.ucar.edu

    Visit http://www.earthsystemmodeling.org/ to find out more about the
    Earth System Modeling Framework.

    Parameters
    ----------
    source, s : str
        Path to the source grid file.
    destination, d : str
        Path to the destination grid file.
    weight, w : str
        Path to the output regridding weight file.
    method, m : str, default="bilinear"
        Interpolation method. Examples: "bilinear", "conserve".
    pole, p : str, default="all"
        Pole treatment method.
    line_type, l : str
        Path line type on a spherical surface.
        Defaults: "cartesian" (non-conservative) or "greatcircle" (conservative).
    norm_type : str, default="dstarea"
        Normalization type for conservative weights.
    extrap_method : str, default="none"
        Extrapolation method.
    extrap_num_src_pnts : int, default=8
        Number of source points for `nearestidavg` extrapolation.
    extrap_dist_exponent : float, default=2.0
        Distance exponent for `nearestidavg` extrapolation.
    extrap_num_levels : int
        Levels to fill for level-based extrapolation (e.g., creep).
    ignore_unmapped, i : bool, default=False
        Ignore unmapped destination points.
    ignore_degenerate : bool, default=False
        Ignore degenerate cells in input grids.
    src_type : str
        Source grid file type (SCRIP, ESMFMESH, UGRID, CFGRID, GRIDSPEC, MOSAIC, TILE).
    dst_type : str
        Destination grid file type (SCRIP, ESMFMESH, UGRID, CFGRID, GRIDSPEC, MOSAIC, TILE).
    t : str
        File type for both source and destination grids (cannot be used with src_type or dst_type).
    r : bool
        Mark source and destination grids as regional.
    src_regional : bool
        Mark source grid as regional.
    dst_regional : bool
        Mark destination grid as regional.
    64bit_offset : bool
        Output weight file in NetCDF 64-bit offset format.
    netcdf4 : bool
        Output weight file in NetCDF4 format.
    weight_only : bool
        Output contains only weights and grid indices.
    src_missingvalue : str
        Variable in GRIDSPEC/UGRID file for masking source grid.
    dst_missingvalue : str
        Variable in GRIDSPEC/UGRID file for masking destination grid.
    src_coordinates : str
        Comma-separated lon,lat variable names for source grid (GRIDSPEC).
    dst_coordinates : str
        Comma-separated lon,lat variable names for destination grid (GRIDSPEC).
    user_areas : bool
        Use user-provided areas for conservation.
    src_loc : str, {"center", "corner"}
        Source grid location for regridding.
    dst_loc : str, {"center", "corner"}
        Destination grid location for regridding.
    tilefile_path : str
        Alternative path for tile files (MOSAIC grids).
    no_log : bool
        Disable ESMF logging.
    check : bool
        Perform weight-checking using an analytic field.
    checkFlag : bool
        Enable extra error checking.
    help, h : bool
        Print help message and exit.
    version : bool
        Print ESMF version and license information and exit.
    V : bool
        Print ESMF version number and exit.

    Notes
    -----
    - Boolean flags should be passed as `True` to include them in the command.
    - Parameters must match valid ESMF_RegridWeightGen CLI flags.
    - This function does not enforce default values; all arguments must be provided explicitly if required.
    """

    try:

        PARAMS = [
            "source",
            "s",
            "destination",
            "d",
            "weight",
            "w",
            "method",
            "m",
            "pole",
            "p",
            "line_type",
            "l",
            "norm_type",
            "extrap_method",
            "extrap_num_src_pnts",
            "extrap_dist_exponent",
            "extrap_num_levels",
            "ignore_unmapped",
            "i",
            "ignore_degenerate",
            "src_type",
            "dst_type",
            "t",
            "r",
            "src_regional",
            "dst_regional",
            "64bit_offset",
            "netcdf4",
            "weight_only",
            "src_missingvalue",
            "dst_missingvalue",
            "src_coordinates",
            "dst_coordinates",
            "user_areas",
            "src_loc",
            "dst_loc",
            "tilefile_path",
            "no_log",
            "check",
            "checkFlag",
            "help",
            "h",
            "version",
            "V",
        ]

        bool_flags = [
            "ignore_unmapped",
            "i",
            "ignore_degenerate",
            "r",
            "src_regional",
            "dst_regional",
            "64bit_offset",
            "netcdf4",
            "weight_only",
            "user_areas",
            "no_log",
            "check",
            "checkFlag",
            "help",
            "h",
            "version",
            "V",
        ]

        weight_file = None

        def _hint(name):

            suggestions = difflib.get_close_matches(name, PARAMS, n=5)
            if suggestions != []:
                hint = f"Did you mean one of {suggestions}?"
            return hint

        subprocess_args = ["mpirun", "-np", str(CPU_COUNT), "ESMF_RegridWeightGen"]

        for k, v in kwargs.items():
            if k not in PARAMS:
                raise ValueError(f"Unknown parameter: {k}. {_hint(k)}")

            if k in bool_flags:
                if v:  # only append if True
                    subprocess_args.append(f"--{k}")
            else:
                if v:
                    subprocess_args.append(f"--{k}")
                    subprocess_args.append(str(v))  # cast to string to avoid TypeError

        if "--weight" not in subprocess_args or "-w" not in subprocess_args:
            weight_file = (CWD() / "weights" / f"{uuid.uuid4()}.nc").resolve()
            _TMP_FILES.append(weight_file)
            weight_file.parent.mkdir(parents=True, exist_ok=True)

            w = ["--weight", f"{weight_file}"]
            subprocess_args.extend(w)

        subprocess.run(subprocess_args, check=True, capture_output=True, text=True)

        if weight_file and weight_file.exists():
            return weight_file

    except subprocess.CalledProcessError as e:
        print(e.stderr)


def regrid_cam_se(dataset: xr.Dataset, weight_file: PathLike) -> xr.Dataset:
    """
    Regrid CAM-SE output using an existing ESMF weights file.

    Parameters
    ----------
    dataset: xarray.Dataset
        Input dataset to be regridded. Must have the `ncol` dimension.
    weight_file: str or Path
        Path to existing ESMF weights file

    Returns
    -------
    regridded
        xarray.Dataset after regridding.
    """

    assert isinstance(dataset, xr.Dataset)
    weights = xr.open_dataset(weight_file)

    # input variable shape
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shapew
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    log(f"Regridding from {in_shape} to {out_shape}")

    # Insert dummy dimension
    vars_with_ncol = [
        name for name in dataset.variables if "ncol" in dataset[name].dims
    ]
    updated = dataset.copy().update(
        dataset[vars_with_ncol].transpose(..., "ncol").expand_dims("dummy", axis=-2)
    )

    # construct a regridder
    # use empty variables to tell xesmf the right shape
    # https://github.com/pangeo-data/xESMF/issues/202
    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )

    regridder = xe.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        method="test",
        reuse_weights=True,
        periodic=True,
    )
    log(regridder)

    # Actually regrid, after renaming
    regridded = regridder(updated.rename({"dummy": "lat", "ncol": "lon"}))

    # merge back any variables that didn't have the ncol dimension
    # And so were not regridded
    return xr.merge([dataset.drop_vars(regridded.variables), regridded])
