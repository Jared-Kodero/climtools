import os
import subprocess
import uuid
from io import StringIO
from pathlib import Path
from typing import Any

import xarray as xr

from .tools import cwd, n_cpus, rm, tmp, which


class CDONotFoundError(RuntimeError):
    pass


class Cdo:
    """
    Limited Python wrapper for Climate Data Operators (CDO) interpolation, merging, and transformation operations.
    The original CDO python package wasn't doing what I wanted, so I made my own.

    This class provides convenient methods to call CDO operators from Python, including horizontal and vertical interpolation,
    file merging, format conversion, spectral transforms, and metadata inspection. Each method constructs and executes the
    appropriate CDO command-line call, handling temporary files and input validation.

    For a full list of available CDO operators and usage details, refer to:
        https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf

    You can also display CDO help by calling `cdo.help()` after creating an instance.

    Example usage:
        >>> cdo = Cdo()
        >>> cdo.remapbil("input.nc", resolution=0.5)
        >>> cdo.mergetime(["file1.nc", "file2.nc"], "merged.nc")
    """

    def __init__(self):
        # check CDO availability on initialization
        self.cdo_path = which("cdo")
        if not self.cdo_path:
            print(
                "CDO executable not found in system PATH. Please install CDO and ensure it is accessible."
            )

        self.tmp_dir = Path(tmp / f"{uuid.uuid4().hex}")
        self.tmp_dir.mkdir(exist_ok=True)
        self.cwd = cwd()

        os.environ["CDO_VERSION_INFO"] = "false"
        os.environ["CDO_HISTORY_INFO"] = "false"

    def run_cdo(self, input_cmds: list[str]):
        cmd = ["cdo", "-s", "-w", "-P", str(min(n_cpus, 32))]
        cmd.extend(input_cmds)

        seen = set()
        cmd = [f"{x}" for x in cmd if not (x in seen or seen.add(x))]

        try:
            res = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True,
            )
            return res
        except subprocess.CalledProcessError as e:
            print("ERROR :", e.stderr)
            return None

    def run(
        self,
        cmd: list[str] = None,
    ) -> Any:
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

        res = self.run_cdo(cmd)
        return res

    def _make_grid_description(self, lon_min, lat_min, lon_max, lat_max, resolution):
        # Compute sizes
        xsize = abs(int(round((lon_max - lon_min) / resolution)) + 1)
        ysize = abs(int(round((lat_max - lat_min) / resolution)) + 1)

        grid_description = []
        grid_description.append("gridtype = lonlat")
        grid_description.append(f"xsize    = {xsize}")
        grid_description.append(f"ysize    = {ysize}")
        grid_description.append(f"xfirst   = {lon_min}")
        grid_description.append(f"xinc     = {resolution}")
        grid_description.append(f"yfirst   = {lat_min}")
        grid_description.append(f"yinc     = {resolution}")

        return "\n".join(grid_description)

    def interpolate(
        self,
        obj: Path | xr.DataArray | xr.Dataset,
        outfile: Path = None,
        *,
        resolution: float = 0.25,
        method: str = None,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
    ) -> xr.DataArray | xr.Dataset | str:
        grdfile = f"{self.tmp_dir}/{uuid.uuid4().hex}.grid"
        if outfile is None:
            outfile = f"{self.tmp_dir}/{uuid.uuid4().hex}.nc"

        da_name = None

        if isinstance(obj, (xr.DataArray, xr.Dataset)):
            tmp_input = f"{self.tmp_dir}/{uuid.uuid4().hex}.nc"
            if isinstance(obj, xr.DataArray):
                da_name = obj.name or "var"
                obj = obj.to_dataset(name=da_name)

            obj.to_netcdf(tmp_input)
            obj = tmp_input

        if not Path(obj).exists():
            raise FileNotFoundError(f"Input file {obj} does not exist.")
        if Path(outfile).exists():
            rm(outfile)

        if bbox:
            lon_min, lat_min, lon_max, lat_max = bbox
        else:
            raise ValueError(
                "Bounding box [lon_min, lat_min, lon_max, lat_max] must be provided for interpolation."
            )

        grid_description = self._make_grid_description(
            lon_min, lat_min, lon_max, lat_max, resolution
        )

        with open(grdfile, "w") as f:
            f.write(grid_description.strip())

        cmd = [
            "-b",
            "F32",
            "-z",
            "zip",
            f"{method},{grdfile}",
            f"{obj}",
            f"{outfile}",
        ]

        res = self.run_cdo(cmd)
        if res.stdout:
            print(res.stdout)

        if as_xarray:
            ret = xr.open_dataset(outfile, chunks="auto")
            if da_name:
                ret = ret[da_name]
        else:
            ret = outfile

        return ret

    def griddes(
        self,
        infile: Path,
    ) -> dict:
        """
        Get the grid description of a netCDF file using CDO's griddes command.

        Parameters
        ----------
        infile : str
            Input netCDF file.

        """

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        cmd = ["griddes", infile]

        res = self.run_cdo(cmd)

        txt = StringIO(res.stdout)
        txt = txt.read().splitlines()

        result = {}
        grid_key = None
        for line in txt:
            if line.strip() == "#":
                continue
            if "# gridID" in line:
                grid_key = line.strip().replace("# ", "").replace(" ", "_")
                result[grid_key] = {}
                continue
            # split key and value by =
            key, value = line.split("=", 1)
            result[grid_key][key.strip()] = value.strip()

        if len(result) == 1:
            result = result[grid_key]

        return result

    def mergetime(
        self,
        infiles: list[Path],
        outfile: Path,
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
            "-b",
            "F32",
            "-z",
            "zip",
            "-mergetime",
            *infiles,
            outfile,
        ]
        self.run_cdo(cmd)
        if delete_input:
            rm(infiles)

        if as_xarray:
            ret = xr.open_dataset(outfile, chunks="auto")
        else:
            ret = outfile

        return ret

    def remapdis(
        self,
        obj: Path | xr.DataArray | xr.Dataset,
        outfile: Path = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
    ) -> xr.DataArray | xr.Dataset | str:
        """
        Interpolate data using CDO's remapdis method.

        bbox_format: (lon_min, lat_min, lon_max, lat_max)

        """

        return self.interpolate(
            obj=obj,
            outfile=outfile,
            resolution=resolution,
            method="remapdis",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapnn(
        self,
        obj: Path | xr.DataArray | xr.Dataset,
        outfile: Path = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str:
        """
        Interpolate data using CDO's remapnn method.

        bbox_format: (lon_min, lat_min, lon_max, lat_max)
        """
        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self.interpolate(
            obj=obj,
            outfile=outfile,
            resolution=resolution,
            method="remapnn",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapcon(
        self,
        obj: Path | xr.DataArray | xr.Dataset,
        outfile: Path = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
    ) -> xr.DataArray | xr.Dataset | str:
        """
        Interpolate data using CDO's remapcon method.

        bbox_format: (lon_min, lat_min, lon_max, lat_max)
        """

        return self.interpolate(
            obj=obj,
            outfile=outfile,
            resolution=resolution,
            method="remapcon",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapbil(
        self,
        obj: Path | xr.DataArray | xr.Dataset,
        outfile: Path = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str:
        """
        Interpolate data using CDO's remapbil method.

        bbox_format: (lon_min, lat_min, lon_max, lat_max)
        """

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self.interpolate(
            obj=obj,
            outfile=outfile,
            resolution=resolution,
            method="remapbil",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remapbic(
        self,
        obj: Path | xr.DataArray | xr.Dataset,
        outfile: Path = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str:
        """
        Interpolate data using CDO's remapbic method.

        bbox_format: (lon_min, lat_min, lon_max, lat_max)
        """

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self.interpolate(
            obj=obj,
            outfile=outfile,
            resolution=resolution,
            method="remapbic",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def remaplaf(
        self,
        obj: Path | xr.DataArray | xr.Dataset,
        outfile: Path = None,
        *,
        resolution: float = 0.25,
        bbox: tuple[float, float, float, float] = None,
        as_xarray: bool = False,
        extrapolate: bool = False,
    ) -> xr.DataArray | xr.Dataset | str:
        """
        Interpolate data using CDO's remaplaf method.

        bbox_format: (lon_min, lat_min, lon_max, lat_max)
        """

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self.interpolate(
            obj=obj,
            outfile=outfile,
            resolution=resolution,
            method="remaplaf",
            bbox=bbox,
            as_xarray=as_xarray,
        )


cdo: Cdo = Cdo()
