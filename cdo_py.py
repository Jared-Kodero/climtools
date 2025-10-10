import os
import subprocess
import uuid
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import xarray as xr

from .pac_man import which
from .tools import (
    _TMP_FILES,
    CPU_COUNT,
    CWD,
    TMPDIR,
    execute_cmd,
    mv,
    rm,
    symlink,
    type_cast,
)


class CDONotFoundError(FileNotFoundError):
    pass


class CDO:
    """
    Python wrapper for Climate Data Operators (CDO) interpolation, merging, and transformation operations.

    This class provides convenient methods to call CDO operators from Python, including horizontal and vertical interpolation,
    file merging, format conversion, spectral transforms, and metadata inspection. Each method constructs and executes the
    appropriate CDO command-line call, handling temporary files and input validation.

    For a full list of available CDO operators and usage details, refer to:
        https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf

    You can also display CDO help by calling `cdo.help()` after creating an instance.

    Example usage:
        >>> cdo = CDO()
        >>> cdo.remapbil("input.nc", resolution=0.5)
        >>> cdo.mergetime(["file1.nc", "file2.nc"], "merged.nc")
    """

    def __init__(self):
        # check CDO availability on initialization
        self.cdo_path = which("cdo")
        if not self.cdo_path:
            raise CDONotFoundError(
                "CDO is not installed or not available in PATH.\
                See https://code.mpimet.mpg.de/projects/cdo/wiki"
            )

        self.tmp_dir = TMPDIR / "_tmp_cdo"
        self.tmp_dir.mkdir(exist_ok=True)
        self.cwd = CWD()
        self.cwd_tmp = self.cwd / ".tmp"

        self.cwd_tmp.mkdir(parents=True, exist_ok=True)
        _TMP_FILES.append(self.tmp_dir)

        os.environ["CDO_VERSION_INFO"] = "false"
        os.environ["CDO_HISTORY_INFO"] = "false"

    def _cdo(self, input_cmds: list[str]):
        try:
            cmd = ["cdo", "-s", "-w", "-P", str(CPU_COUNT)]
            cmd.extend(input_cmds)

            seen = set()
            cmd = [f"{x}" for x in cmd if not (x in seen or seen.add(x))]
            res = execute_cmd(cmd)
            return res

        except subprocess.CalledProcessError as e:
            print("ERROR:", e.stderr)

    def help(self) -> None:
        cmd = ["cdo", "-h"]
        res = self._cdo(cmd)
        print(res.stdout)

    def _bbox_from_griddes(self, infile) -> tuple[float, float, float, float]:

        griddes = self.griddes(infile, return_output=True)

        data = griddes.stdout.split("\n")
        data_dict = {}
        for line in data:
            if "=" in line:
                key, value = line.split("=")
                data_dict[key.strip()] = value.strip()

        for key in [
            "xsize",
            "ysize",
            "xfirst",
            "yfirst",
            "xinc",
            "yinc",
        ]:
            data_dict[key] = float(data_dict[key])

        lon_min = data_dict["xfirst"]
        lon_max = lon_min + (data_dict["xsize"] - 1) * data_dict["xinc"]
        lat_max = data_dict["yfirst"]
        lat_min = lat_max + (data_dict["ysize"] - 1) * data_dict["yinc"]

        print(f"Bounding box: {lon_min}, {lat_min}, {lon_max}, {lat_max}")

        return lon_min, lat_min, lon_max, lat_max

    def _make_grid_description(self, lon_min, lat_min, lon_max, lat_max, resolution):

        # Compute sizes
        xsize = abs(int(round((lon_max - lon_min) / resolution)) + 1)
        ysize = abs(int(round((lat_max - lat_min) / resolution)) + 1)

        grid_description = []
        grid_description.append(f"gridtype = lonlat")
        grid_description.append(f"xsize    = {xsize}")
        grid_description.append(f"ysize    = {ysize}")
        grid_description.append(f"xfirst   = {lon_min}")
        grid_description.append(f"xinc     = {resolution}")
        grid_description.append(f"yfirst   = {lat_min}")
        grid_description.append(f"yinc     = {resolution}")

        return "\n".join(grid_description)

    def _h_interp_data(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        resolution: float = 0.25,
        method: str = None,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:

        grdfile = f"{self.tmp_dir}/{uuid.uuid4()}.grid"
        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.nc"
            _TMP_FILES.extend([outfile, grdfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if Path(outfile).exists():
            rm(outfile)

        if bbox:
            lon_min, lat_min, lon_max, lat_max = bbox
        else:
            lon_min, lat_min, lon_max, lat_max = self._bbox_from_griddes(infile)

        grid_description = self._make_grid_description(
            lon_min, lat_min, lon_max, lat_max, resolution
        )

        with open(grdfile, "w") as f:
            f.write(grid_description.strip())

        cmd = [
            "-z",
            "zip",
            "-b",
            "F32",
            f"{method},{grdfile}",
            f"{infile}",
            f"{outfile}",
        ]

        res = self._cdo(cmd)
        if res.stdout:
            print(res.stdout)

        if as_xarray:
            ret = xr.open_dataset(outfile, chunks="auto")
        else:
            ret = outfile

        return ret

    def remapdis(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """Interpolate data using CDO's remapdis method."""

        return self._h_interp_data(
            infile=infile,
            outfile=outfile,
            resolution=resolution,
            method="remapdis",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapnn(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """Interpolate data using CDO's remapnn method."""
        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
            infile=infile,
            outfile=outfile,
            resolution=resolution,
            method="remapnn",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapcon(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """Interpolate data using CDO's remapcon method."""

        return self._h_interp_data(
            infile=infile,
            outfile=outfile,
            resolution=resolution,
            method="remapcon",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapbil(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """Interpolate data using CDO's remapbil method."""

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
            infile=infile,
            outfile=outfile,
            resolution=resolution,
            method="remapbil",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapbic(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """Interpolate data using CDO's remapbic method."""

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
            infile=infile,
            outfile=outfile,
            resolution=resolution,
            method="remapbic",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remaplaf(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """Interpolate data using CDO's remaplaf method."""

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
            infile=infile,
            outfile=outfile,
            resolution=resolution,
            method="remaplaf",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapeta(
        self,
        vct: Path | PathLike,
        oro: Path | PathLike,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        remapeta_ptop: float = None,
    ):
        """
        Interpolate between different vertical hybrid levels using CDO's `remapeta` operator.

        This operator prepares consistent vertical profiles for the free atmosphere by
        interpolating between hybrid sigma-pressure coordinate systems. The method is
        based on the HIRLAM scheme (adapted from [INTERA]) and uses vertical integration
        of the hydrostatic equation with corrections for surface pressure.

        The interpolation procedure involves:
            - Integration of the hydrostatic equation.
            - Extrapolation of surface pressure.
            - Planetary Boundary Layer (PBL) profile interpolation (using potential temperature).
            - Free atmosphere interpolation (linear, above n=0.8).
            - Merging of PBL and free atmosphere profiles.
            - Final surface pressure correction (mass conservation above ~400 hPa reference).

        Mass corrections near the surface may alter the vertical PBL structure, but the
        total atmospheric mass above 400 hPa is conserved to preserve geostrophic balance.

        Parameters
        ----------
        vct : Path or PathLike
            Path to the ASCII file containing the vertical coordinate table (VCT).
            Must describe a hybrid sigma-pressure system following ECHAM conventions.
        oro : Path or PathLike
            Path to the file with the orography (surface geopotential) of the target dataset.
            Optional, but recommended for consistent pressure field adjustments.
        infile : Path or PathLike
            Input file containing data on hybrid model levels (must share the same horizontal grid).
        outfile : Path or PathLike, optional
            Path to the output file. If not provided, a temporary NetCDF file is created.
        remapeta_ptop : float, optional
            Sets the minimum pressure level for condensation. Above this level the humidity is set to the constant 1.E-6. The default value is 0 Pa.
        """
        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.nc"
            _TMP_FILES.extend([outfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if not Path(vct).exists():
            raise FileNotFoundError(f"VCT file {vct} does not exist.")
        if not Path(oro).exists():
            raise FileNotFoundError(f"ORO file {oro} does not exist.")

        if remapeta_ptop is not None:
            os.environ["REMAPETA_PTOP"] = str(remapeta_ptop)

        cmd = [
            "-z",
            "zip",
            "-b",
            "F32",
            f"remapeta,{vct}",
            f"{infile}",
            f"{outfile}",
        ]

        res = self._cdo(cmd)
        if res.stdout:
            print(res.stdout)

    def vertintml(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        p_levels: list[float] = None,
        h_levels: list[float] = None,
        operator: Literal["ml2pl", "ml2hl"] = None,
        extrapolate: bool = False,
    ):
        """
        Interpolate 3D fields on hybrid sigma-pressure levels to pressure or height levels
        using CDO's `ml2pl` or `ml2hl` operators.

        The vertical interpolation requires the hybrid vertical coordinate definition
        (a and b coefficients), surface pressure, and optionally surface geopotential.
        Geopotential height must be available at the hybrid layer interfaces
        (model half-levels). All variables must be on the same horizontal grid.
        Missing values are not supported.

        Parameters
        ----------
        infile : Path or PathLike
            Input file containing 3D variables on hybrid sigma-pressure levels
            and the necessary auxiliary fields:
            - `a` and `b` coefficients (hybrid coordinate definition).
            - Surface pressure (`ps` or log surface pressure).
            - Surface geopotential (`z_surf`), required for temperature extrapolation.
            - Geopotential height at hybrid interfaces.
        outfile : Path or PathLike, optional
            Path to the output file. If not provided, a temporary NetCDF file is created.
        p_levels : list of float, optional
            Target pressure levels in pascals (Pa). Required when using operator="ml2pl".
        h_levels : list of float, optional
            Target height levels in meters (m). Required when using operator="ml2hl".
            Heights are internally converted to equivalent pressure levels via:
                p = 101325 * exp(-h / 7000).
        operator : {"ml2pl", "ml2hl"}
            Choice of vertical interpolation operator:
            - "ml2pl": Hybrid model → pressure levels.
            - "ml2hl": Hybrid model → height levels.

        Environment Variables
        ---------------------
        EXTRAPOLATE : int, default=0
            If set to 1, extrapolates missing values.
            Can also be triggered by using aliases `ml2plx` / `ml2hlx`.

        Notes
        -----
        - Required variables must be identifiable by GRIB1 code or NetCDF CF name:
            * log surface pressure (GRIB1 code 152, CF: `surface_air_pressure`)
            * surface pressure (134, CF: `surface_air_pressure`)
            * temperature (130, CF: `air_temperature`)
            * surface geopotential (129, CF: `surface_geopotential`)
            * geopotential height (156, CF: `geopotential_height`)
        - Supported parameter tables: WMO standard (table 2) and ECMWF local (table 128).
        - Input must use the same horizontal grid. Hybrid components must be defined
        at model half-levels, even if fields are provided at midpoints.
        """

        if outfile is None:
            outfile = f"{self.tmp_dir}/vertintml_{uuid.uuid4()}"
            # _TMP_FILES.extend([outfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if Path(outfile).exists():
            rm(outfile)
        if operator == "ml2pl" and p_levels is None:
            raise ValueError("p_levels must be provided for ml2pl operator.")
        if operator == "ml2hl" and h_levels is None:
            raise ValueError("h_levels must be provided for ml2hl operator.")

        if extrapolate:
            os.environ["EXTRAPOLATE"] = "1"

        cmd = [
            "-z",
            "zip",
            "-b",
            "F32",
        ]

        if operator == "ml2pl":
            levels = ",".join([str(p) for p in p_levels])
            cmd.append(f"{operator},{levels}")
        elif operator == "ml2hl":
            levels = ",".join([str(h) for h in h_levels])
            cmd.append(f"{operator},{levels}")

        cmd.extend([f"{infile}", f"{outfile}"])

        self._cdo(cmd)

        return outfile

    def vertintap(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        p_levels: list[float] = None,
        extrapolate: bool = False,
    ):
        """
        Interpolate 3D variables on hybrid sigma height coordinates to pressure levels. The input file must
        contain the 3D air pressure in pascal. The air pressure is identified by the NetCDF CF standard
        name air_pressure. Use the alias ap2plx or the environment variable EXTRAPOLATE to extrapolate
        missing values. This operator requires all variables on the same horizontal grid.

        Parameters
        ----------
        infile : Path or PathLike
            Input file containing 3D variables on pressure levels and the necessary
            auxiliary fields:
            - Surface geopotential (`z_surf`), required for temperature extrapolation.
            - Geopotential height at pressure level interfaces.
        outfile : Path or PathLike, optional
            Path to the output file. If not provided, a temporary NetCDF file is created.
        p_levels : list of float
            Target pressure levels in pascals (Pa). Required for the `ap2hl` operator.
        """
        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.nc"
            _TMP_FILES.extend([outfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if Path(outfile).exists():
            rm(outfile)
        if p_levels is None:
            raise ValueError("p_levels must be provided for ap2hl operator.")

        if extrapolate:
            os.environ["EXTRAPOLATE"] = "1"

        levels_str = ",".join([str(p) for p in p_levels])

        cmd = (
            [
                "-z",
                "zip",
                "-b",
                "F32",
                "ap2pl,",
                f"{levels_str}",
                f"{infile}",
                f"{outfile}",
            ],
        )

        self._cdo(cmd)

        return outfile

    def vertintgh(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        h_levels: list[float] = None,
        extrapolate: bool = False,
    ):
        """
        Interpolate 3D variables on hybrid sigma height coordinates to height levels. The input file must
        contain the 3D geometric height in meter. The geometric height is identified by the NetCDF CF
        standard name geometric_height_at_full_level_center. Use the alias gh2hlx or the environ
        ment variable EXTRAPOLATE to extrapolate missing values. This operator requires all variables on the
        same horizontal grid.

        Parameters
        ----------
        infile : Path or PathLike
            Input file containing 3D variables on height levels and the necessary
            auxiliary fields:
            - Surface geopotential (`z_surf`), required for temperature extrapolation.
            - Geopotential height at height level interfaces.
        outfile : Path or PathLike, optional
            Path to the output file. If not provided, a temporary NetCDF file is created.
        h_levels : list of float
            Target height levels in meters (m). Required for the `gh2hl` operator.
        """
        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.nc"
            _TMP_FILES.extend([outfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if Path(outfile).exists():
            rm(outfile)
        if h_levels is None:
            raise ValueError("h_levels must be provided for gh2hl operator.")

        if extrapolate:
            os.environ["EXTRAPOLATE"] = "1"

        levels_str = ",".join([str(h) for h in h_levels])
        cmd = [
            "-z",
            "zip",
            "-b",
            "F32",
            "gh2hl,",
            f"{levels_str}",
            f"{infile}",
            f"{outfile}",
        ]

        self._cdo(cmd)

        return outfile

    def mergetime(
        self,
        infiles: list[PathLike],
        outfile: Path | PathLike,
        *,
        as_xarray: bool = False,
        delete_input: bool = False,
    ):
        """
        Merge multiple netCDF files along the time dimension using CDO.
        This function uses the Climate Data Operators (CDO) to merge multiple netCDF files.

        Parameters
        ----------
        infiles : list[str]
            List of input netCDF files to be merged.
        outfile : str
        """

        infiles = [str(Path(f).resolve()) for f in infiles]
        infiles.sort()
        for f in infiles:
            if not Path(f).exists():
                raise FileNotFoundError(f"Input file {f} does not exist.")

        cmd = [
            "-z",
            "zip",
            "-b",
            "F32",
            "-mergetime",
            *infiles,
            outfile,
        ]
        self._cdo(cmd)
        if delete_input:
            rm(infiles)

        if as_xarray:
            ret = xr.open_dataset(outfile, chunks="auto")
        else:
            ret = outfile

        return ret

    def intlevel(
        self,
        infile: Path | PathLike,
        level: list[float] = None,
        zdescription: Path | PathLike = None,
        zvarname: str = None,
        extrapolate: bool = False,
        return_output=False,
    ) -> str:
        """
        This operator performs a linear vertical interpolation of 3D variables. The 1D target levels can be specified with the level parameter or read in via a Z-axis description file.

        Parameters
        ----------
        infile : str
            Input netCDF file.

        Returns
        -------
        str
            The vertical levels as a string.
        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if level is None and zdescription is None:
            raise ValueError("Either level or zdescription must be provided.")

        cmd = ["intlevel", infile]

        res = self._cdo(cmd)
        if return_output:
            return res.stdout
        else:
            if res.stdout:
                print(res.stdout)

    def intlevel3d(
        self,
        operator: Literal["intlevel3d", "intlevelx3d"] = None,
        infile: Path | PathLike = None,
        tgtcoordinates: Path | PathLike = None,
        return_output=False,
    ) -> str:
        """
        This operator performs a 3D vertical interpolation of 3D variables.
        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if not Path(tgtcoordinates).exists():
            raise FileNotFoundError(
                f"Target coordinates file {tgtcoordinates} does not exist."
            )

        cmd = [
            operator,
            infile,
            tgtcoordinates,
        ]

        res = self._cdo(cmd)
        if return_output:
            return res.stdout
        else:
            if res.stdout:
                print(res.stdout)

    def fourier_transform(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        epsilon: Literal[1, -1] = 1,
        as_xarray: bool = False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """
        The fourier operator performs the fourier transformation or the inverse fourier transformation of all input fields.
        If the number of timesteps is a power of 2 then the algorithm of the Fast Fourier Transformation (FFT) is used.
        """
        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.nc"
            _TMP_FILES.extend([outfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if Path(outfile).exists():
            rm(outfile)

        if epsilon not in [1, -1]:
            raise ValueError("epsilon must be either 1 (forward) or -1 (inverse).")

        cmd = [
            f"fourier,{epsilon}",
            infile,
            outfile,
        ]
        self._cdo(cmd)

        if as_xarray:
            ret = xr.open_dataset(outfile, chunks="auto")
        else:
            ret = outfile
        return ret

    def spectral_transform(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        operator: Literal["sp2gp", "gp2sp", "sp2sp", "dv2ps", "dv2uv", "uv2dv"] = None,
        p_type: Literal["linear", "quadratic", "cubic"] = None,
        trunc=False,
    ) -> str | PathLike:
        """
        Perform spectral transformations using CDO (Climate Data Operators).

        This function wraps CDO operators that transform fields between
        spectral and grid-point representations, or between different
        spectral resolutions. The transformations are based on
        Fast Fourier Transform (FFT) and (inverse) Legendre transformations.
        Missing values are not supported.

        Parameters
        ----------
        infile : str or PathLike
            Path to the input GRIB file.
        outfile : str or Path | PathLike, optional
            Path to the output file. If not provided, a temporary NetCDF file will be created.
        operator : {"sp2gp", "gp2sp", "sp2sp", "dv2ps", "dv2uv", "uv2dv"}
            The spectral transformation operator:
            - "sp2gp": Spectral → Gaussian gridpoint fields.
            - "gp2sp": Gaussian gridpoint → spectral coefficients.
            - "sp2sp": Spectral → spectral (change triangular truncation).
            - "dv2ps": Divergence/vorticity → velocity potential and stream function.
            - "dv2uv": Divergence/vorticity (spectral) → U and V wind (grid).
            - "uv2dv": U and V wind (grid) → divergence and vorticity (spectral).
        p_type : {"linear", "quadratic", "cubic"}, optional
            Grid type specification:
            - "linear": shortest wavelength represented by 2 grid points (ERA40 convention).
            - "quadratic": shortest wavelength represented by 3 grid points (ECHAM, ERA15).
            - "cubic": shortest wavelength represented by 4 grid points.
            Used with "sp2gp", "gp2sp", "dv2uv", and "uv2dv". Default depends on CDO (usually "quadratic").
        trunc : bool, default=False
            If True, applies triangular truncation (mandatory for "sp2sp").

        Notes
        -----
        Grid resolution and spectral resolution are related through the
        triangular truncation number T and the number of Gaussian grid points:
            - Linear grid:   4N ≈ 2(TL + 1)
            - Quadratic grid: 4N ≈ 3(TQ + 1)
            - Cubic grid:    4N ≈ 4(TC + 1)
        where N is the number of grid points between the poles.
        """

        if operator == "sp2sp":
            if not trunc:
                trunc = True

        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.nc"
            _TMP_FILES.extend([outfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if Path(outfile).exists():
            rm(outfile)

        cmd = [
            operator,
        ]

        if p_type and operator in ["sp2gp", "gp2sp"]:
            cmd.append(f"type={p_type}")

        elif p_type and operator in ["dv2uv", "uv2dv"]:
            cmd.append(f"gridtype={p_type}")

        elif trunc:
            cmd.append("trunc")

        cmd.extend([f"{infile}", f"{outfile}"])

        self._cdo(cmd)
        return outfile

    def griddes(self, infile: Path | PathLike, return_output=False) -> str:
        """
        Get the grid description of a netCDF file using CDO's griddes command.

        Parameters
        ----------
        infile : str
            Input netCDF file.

        Returns
        -------
        str
            The grid description as a string.
        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        cmd = ["griddes", infile]

        res = self._cdo(cmd)

        if return_output:
            return res.stdout
        else:
            if res.stdout:
                print(res.stdout)

    def info(
        self,
        infile: Path | PathLike,
        operator: Literal["info", "infon", "sinfo", "sinfon"] = None,
        return_output=False,
    ) -> str:
        """
        Print summary statistics or structural information for each field in the input file(s).
        Supported operators:

            - info: statistics by parameter identifier
            - infon: statistics by parameter name
            - sinfo: summary by parameter identifier
            - sinfon: summary by parameter name

        Parameters
        ----------
        infile : str
            Input netCDF file.
        operator : str
            The operator to use.
        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        cmd = [
            operator,
            infile,
        ]

        res = self._cdo(cmd)

        if return_output:
            return res.stdout
        else:
            if res.stdout:
                print(res.stdout)

    def showinfo(
        self,
        infile: Path | PathLike,
        return_output=False,
    ) -> str:
        """
        This module prints meta-data information of all input variables.

        Parameters
        ----------
        infile : str
            Input netCDF file.
        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        result = {}

        operators = [
            "showcode",
            "showname",
            "showstdname",
            "showlevel",
            "showyear",
            "showmon",
            "showtime",
            "showltype",
            "showfilter",
            "showformat",
        ]
        for operator in operators:

            cmd = [operator, infile]

            res = self._cdo(cmd)
            result[operator.removeprefix("show")] = res.stdout.strip()

        for k, v in result.items():
            v = v.split(" ")
            if k in ["code", "level", "ltype"]:
                v = [type_cast(i) for i in v if i]
            elif k in ["name", "stdname", "filter"]:
                v = [i for i in v if i]
            elif k in ["year", "mon", "time"]:
                v = sorted(set(filter(None, v)))
                v = v[0] if len(v) == 1 else v
            result[k] = v

        if return_output:
            return result
        else:
            import pprint

            pprint.pprint(result, compact=True, width=120, sort_dicts=False)

    def vlist(
        self,
        infile: Path | PathLike,
        return_output=False,
    ) -> str:
        """
        This module prints a list of all variables in the input dataset.

        Parameters
        ----------

        infile : str
            Input netCDF file.
        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        cmd = ["vlist", infile]
        res = self._cdo(cmd)

        if return_output:
            return res.stdout
        else:
            print(res.stdout)

    def ninfo(
        self,
        infile: Path | PathLike,
        return_output=False,
    ) -> str:
        """
        This module prints the number of variables, levels or times of the input dataset.

        Parameters
        ----------

        infile : str
            Input netCDF file.
        """
        import re

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        result = {}

        operators = [
            "npar",
            "nlevel",
            "nyear",
            "nmon",
            "ndate",
            "ntime",
            "ngridpoints",
            "ngrids",
        ]
        for operator in operators:

            cmd = [operator, infile]

            res = self._cdo(cmd)
            result[operator] = res.stdout.strip()
        for k, v in result.items():

            v = sorted(set(filter(None, re.split(r"[ \n]+", v))))
            v = [type_cast(i) for i in v]
            result[k] = v[0] if len(v) == 1 else v

        if return_output:
            return result
        else:
            import pprint

            pprint.pprint(result)

    def split(
        self,
        infile: Path | PathLike,
        outdir: Path | PathLike = None,
        obase: str = None,
        operator: Literal[
            "splitcode",
            "splitparam",
            "splitname",
            "splitlevel",
            "splitgrid",
            "splitzaxis",
            "splittabnum",
        ] = None,
    ) -> list[str]:
        """
        Splits the input file into multiple pieces using Climate Data Operators (CDO).
        The output files will be written to the specified directory, with filenames
        of the form <obase><xxx><suffix>, where:
          - <obase> is derived from the input filename,
          - <xxx> depends on the chosen operator,
          - <suffix> is determined by the file format.

        Parameters
        ----------
        infile : str
            Path to the input NetCDF or GRIB file.
        outdir : str
            Output directory where the split files will be written.
        obase : str, optional
            Base name for output files.
        operator : str, optional
            The splitting operator to use. Options include:

        Operators
        ---------
        splitcode : Split by GRIB code number
            Splits the dataset into one file per GRIB code number.
            ``xxx`` will be a three-digit code number.

        splitparam : Split by parameter identifier
            Splits the dataset into one file per parameter identifier.
            ``xxx`` will be the parameter identifier string.

        splitname : Split by variable name
            Splits the dataset into one file per variable name.
            ``xxx`` will be the variable name string.

        splitlevel : Split by vertical level
            Splits the dataset into one file per level.
            ``xxx`` will be a six-digit level index.

        splitgrid : Split by grid definition
            Splits the dataset into one file per grid.
            ``xxx`` will be a two-digit grid number.

        splitzaxis : Split by z-axis
            Splits the dataset into one file per z-axis.
            ``xxx`` will be a two-digit z-axis number.

        splittabnum : Split by GRIB1 parameter table number
            Splits the dataset into one file per GRIB1 parameter table.
            ``xxx`` will be a three-digit table number.
        """
        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        if operator is None:
            raise ValueError("operator must be specified.")

        op_dir = operator.removeprefix("split")
        if outdir is None:
            outdir = self.tmp_dir / Path(infile).stem / op_dir

            # Start clean: remove old axis directory if present
            if outdir.exists():
                rm(outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            # Ensure symlinked view of filestem in cwd_tmp
            filestem_dir = self.tmp_dir / Path(infile).stem
            if not (self.cwd_tmp / Path(infile).stem).exists():
                symlink(filestem_dir, self.cwd_tmp / Path(infile).stem)

            _TMP_FILES.extend([outdir, self.cwd_tmp])
        else:
            outdir = Path(outdir) / Path(infile).stem / op_dir
            outdir.mkdir(parents=True, exist_ok=True)

        if obase:
            prefix_dir = f"{outdir}/{obase}/"
        else:
            prefix_dir = f"{outdir}/"

        Path(prefix_dir).mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(prefix_dir)
            cmd = [operator, infile, ""]
            self._cdo(cmd)
        finally:
            os.chdir(f"{self.cwd}")

        outfiles = sorted([str(f) for f in Path(prefix_dir).glob("*")])

        return outfiles

    def to_nc4(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
        *,
        as_xarray: bool = False,
    ) -> None:
        """
        Convert a GRIB file to NetCDF format using CDO.

        Parameters
        ----------
        infile : str or PathLike
            Path to the input GRIB file.
        outfile : str or Path | PathLike, optional
            Path to the output NetCDF file. If not provided, a temporary file will be created.
        as_xarray : bool, optional
            If True, return the output as an xarray Dataset. Default is False.


        """

        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.nc"
            _TMP_FILES.extend([outfile])

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")
        if Path(outfile).exists():
            rm(outfile)

        cmd = [
            "-f",
            "nc4",
            "copy",
            f"{infile}",
            f"{outfile}",
        ]
        self._cdo(cmd)

        if as_xarray:
            ret = xr.open_dataset(outfile, chunks="auto")
        else:
            ret = outfile

        return ret

    def get_vert_coords(self, infile: Path | PathLike, return_output=False) -> str:
        """
        Extract vertical coordinates from a GRIB file
        """
        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        from io import StringIO

        sinfo_out = self.info(infile, "sinfo", return_output=True)

        start = sinfo_out.find("Vertical coordinates")
        end = sinfo_out.find("Time coordinate")
        if start == -1:
            return {}

        block = sinfo_out[start:end].strip()
        block = block.replace("Vertical coordinates :", "").strip()

        tmp_lines = []
        lines = []
        for line in StringIO(block):
            line = line.strip()
            digit = line.split(":")[0].strip()
            if not digit.isdigit():
                line = " " * 25 + f"{line}"  # pad to avoid issues
            tmp_lines.append(line)

        for line in tmp_lines:
            # Find the position of the last colon in each line
            split_lines = [line.rsplit(":", 1) for line in tmp_lines if ":" in line]

            # Find max width of the left-hand side
            max_left = max(len(left.rstrip()) for left, _ in split_lines)

            for line in tmp_lines:
                if ":" in line:
                    left, right = line.rsplit(":", 1)
                    lines.append(f"{left.rstrip().ljust(max_left)} :{right}")
                else:
                    lines.append(line)

        # remove duplicate lines
        lines = list(dict.fromkeys(lines))

        block = "\n".join(lines)

        if return_output:
            return block
        else:
            print(block)

    def netcdf_to_grib(
        self,
        infile: Path | PathLike,
        outfile: Path | PathLike = None,
    ) -> str | PathLike:
        """
        Convert a NetCDF file to GRIB format using CDO.

        Parameters
        ----------
        infile : str or PathLike
            Path to the input NetCDF file.
        outfile : str or Path | PathLike, optional
            Path to the output GRIB file. If not provided, a temporary file will be created.

        Returns
        -------
        str or PathLike
            Path to the output GRIB file.
        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4()}.grb"
            _TMP_FILES.extend([outfile])
        if Path(outfile).exists():
            rm(outfile)

        cmd = [
            "-f",
            "grb2",
            "copy",
            f"{infile}",
            f"{outfile}",
        ]
        self._cdo(cmd)

        return outfile

    def grib_to_netcdf(
        self,
        infile: Path | PathLike,
        outdir: Path | PathLike = None,
        merge: bool = True,
        *,
        p_levels: list[float] = None,
        h_levels: list[float] = None,
        extrapolate: bool = False,
        spectral_operator: Literal[
            "sp2gp", "gp2sp", "sp2sp", "dv2ps", "dv2uv", "uv2dv"
        ] = None,
        p_type: Literal["linear", "quadratic", "cubic"] = None,
        trunc=False,
    ) -> xr.DataArray | xr.Dataset | str | PathLike:
        """
        Convert a GRIB file to NetCDF with ECMWF-recommended pipeline:
        1. If spectral, convert to gridpoint (sp2gpl).
        2. Split GRIB into per-variable files (splitname).
        3. Optionally split further by z-axis (splitzaxis).
        4. Convert GRIBs → NetCDF.
        5. Interpolate hybrid levels → pressure levels .
        6. Merge into final NetCDF.

        Example usage
            >>> ds = cdo.grib_to_netcdf(
            >>>     path,
            >>>     outdir="data/gfs.0p25.2021082900.f000",
            >>>     spectral_operator="sp2gp",
            >>>     trunc=True,
            >>>     p_type="linear",
            >>>     extrapolate=True,
            >>> )

        """

        infile = Path(infile).resolve()
        if not infile.exists():
            raise FileNotFoundError(f"Input GRIB file {infile} does not exist.")

        if outdir is None:
            raise ValueError("Output directory must be specified.")

        if p_levels is None:
            p_levels = list(range(100000, 0, -5000))  # 0 to 1000 hPa, step 50hPa
        if h_levels is None:
            h_levels = list(range(0, 30000, 1000))  # 0 to 30km, step 1km

        Path(outdir).mkdir(parents=True, exist_ok=True)
        # Step 1: detect if spectral → convert to gridpoint
        gridinfo = self.griddes(infile, return_output=True)
        if "gridtype = spectral" in gridinfo:
            print("Input file is spectral, performing spectral transform...")
            spectral_transform_out = self.spectral_transform(
                infile=infile,
                outfile=None,
                operator=spectral_operator,
                p_type=p_type,
                trunc=trunc,
            )
            # rename workfile to infile but in the same directory
            infile = Path(spectral_transform_out).parent / infile.name
            mv(spectral_transform_out, infile)
            rm(spectral_transform_out)

        var_files = self.split(
            infile=infile, outdir=None, obase=None, operator="splitname"
        )

        var_files_dict = {}
        for vfile in var_files:

            name = Path(vfile).stem
            axis_files = self.split(
                infile=vfile, outdir=None, obase=name, operator="splitzaxis"
            )

            for axfile in axis_files:
                axis_name = Path(axfile).stem
                get_vert_coords_out = self.get_vert_coords(axfile, return_output=True)

                if "hybrid" in get_vert_coords_out:
                    v_interp_operator = "ml2pl"
                elif "sigma" in get_vert_coords_out:
                    v_interp_operator = "ml2hl"
                else:
                    v_interp_operator = None

                if v_interp_operator:

                    # vertintml_out = self.vertintml(
                    #     infile=axfile,
                    #     outfile=None,
                    #     p_levels=p_levels,
                    #     h_levels=h_levels,
                    #     operator=v_interp_operator,
                    #     extrapolate=extrapolate,
                    # )

                    # axfile = Path(vertintml_out).parent / Path(vfile).name
                    # mv(vertintml_out, axfile)
                    # rm(vertintml_out)

                    # # Step 4: convert to NetCDF
                    # ncfile = self.to_nc4(axfile)

                    # if str(name) not in var_files_dict:
                    #     var_files_dict[str(name)] = []
                    # var_files_dict[str(name)].append(ncfile)
                    ...

                else:

                    ncfile = self.to_nc4(axfile)

                    if str(name) not in var_files_dict:
                        var_files_dict[str(name)] = []
                    var_files_dict[str(name)].append(ncfile)

        from collections import defaultdict

        import xarray as xr

        primary_dict = {}
        secondary_dict = {}

        for k, v in var_files_dict.items():
            files = sorted(v)
            for f in files:
                ds = xr.open_dataset(f, chunks="auto")
                dims = list(ds.dims)
                for dim in dims:
                    if dim not in ["time", "lat", "lon"]:
                        if dim not in primary_dict:
                            primary_dict[dim] = []

                        primary_dict[dim].append(f)

                ds.close()

        for (
            lk,
            lv,
        ) in primary_dict.items():

            try:
                ds = xr.open_mfdataset(
                    lv,
                    combine="by_coords",
                    chunks="auto",
                    parallel=True,
                )
                primary_dict[lk] = ds
                print(f"{lk:<25} : processed with {len(lv)} levels")
            except Exception as e:
                for f in lv:
                    ds = xr.open_dataset(f, chunks="auto")
                    dims = list(ds.dims)
                    if dim not in ["time", "lat", "lon"]:
                        std_name = ds[dim].attrs["standard_name"]
                        std_name = std_name.replace(" ", "_")
                        if std_name not in secondary_dict:
                            secondary_dict[std_name] = []
                        secondary_dict[std_name].append(f)
                    ds.close()

        for lk, lv in secondary_dict.items():
            try:
                ds = xr.open_mfdataset(
                    lv,
                    combine="by_coords",
                    chunks="auto",
                    parallel=True,
                )
                primary_dict[lk] = ds
                print(f"{lk:<25} : processed with {len(lv)} levels")
            except Exception as e:
                print("Failed to combine level:", lk, e)

        print(f"\nSaving output to {outdir} ...\n")

        if not merge:
            for k, ds in primary_dict.items():
                if isinstance(ds, xr.Dataset):
                    ds.to_netcdf(Path(outdir) / f"{k}.nc", format="NETCDF4")
                    ds.close()
                    print(f"SAVED : {k}.nc")
        else:
            datasets = [d for d in primary_dict.values() if isinstance(d, (xr.Dataset))]
            ds = xr.merge(datasets, compat="override")
            ds.to_netcdf(Path(outdir) / f"{infile.stem}.nc", format="NETCDF4")
            print(f"SAVED : {infile.stem}.nc")
            return ds

    def run(self, cmd: list[str] = None, return_output=False) -> Any:
        """
        Run a Climate Data Operators (CDO) command.

        Parameters
        ----------
        cmd : list of str
            Sequence of CDO subcommand and arguments. For details on
            available operators, see:
            https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf

        Returns
        -------
        Any
            The output of the CDO command, if any.
            Executes the specified CDO command as a subprocess. Standard output
            and error streams are passed through to the system shell.

        Examples
        --------
        Remap input data (`infile.nc`) bilinearly onto a target grid (`gridfile`)
        and write the result to `outfile.nc`:

            >>> cmd = ["remapbil", "gridfile", "infile.nc", "outfile.nc"]
            >>> cdo.run(cmd)
        """

        if not isinstance(cmd, list) or cmd is None:
            example = 'e.g., cmd= ["xxx", "xxx", "xxx", "xxx"] \n cdo.run(cmd)'
            print("Invalid command format. usage:", example)
            raise TypeError(
                "Input must be a list of strings representing CDO command and arguments."
            )

        if not isinstance(return_output, bool):
            raise TypeError("return_output must be a boolean.")

        res = self._cdo(cmd)

        if return_output:
            return res
        else:
            print(res.stdout)

    def __dir__(self):
        return [
            "remapnn",
            "remapcon",
            "remapbil",
            "remapbic",
            "remaplaf",
            "remapeta",
            "vertintml",
            "vertintap",
            "vertintgh",
            "mergetime",
            "intlevel",
            "intlevel3d",
            "fourier_transform",
            "griddes",
            "info",
            "showinfo",
            "ninfo",
            "split",
            "grib_to_netcdf",
            "netcdf_to_grib",
            "to_nc4",
            "spectral_transform",
            "get_vert_coords",
            "run",
            "vlist",
        ]


cdo = CDO()
