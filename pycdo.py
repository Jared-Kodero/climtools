import os
import subprocess
import uuid
import warnings
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import xarray as xr

from .tools import _tmp_files, cwd, n_cpus, rm, tmp, to_numeric, which


class CDONotFoundError(RuntimeError):
    pass


class CDO:
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
        >>> cdo = CDO()
        >>> cdo.remapbil("input.nc", resolution=0.5)
        >>> cdo.mergetime(["file1.nc", "file2.nc"], "merged.nc")
    """

    def __init__(self):
        # check CDO availability on initialization
        self.cdo_path = which("cdo")
        if not self.cdo_path:
            warnings.warn(
                "CDO is not installed or not available in PATH.\
                See https://code.mpimet.mpg.de/projects/cdo/wiki",
                UserWarning,
            )

        self.tmp_dir = tmp / "cdo_tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        self.cwd = cwd()

        os.environ["CDO_VERSION_INFO"] = "false"
        os.environ["CDO_HISTORY_INFO"] = "false"

    def _cdo(self, input_cmds: list[str]):

        cmd = ["cdo", "-s", "-w", "-P", str(int(n_cpus / 3))]
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

    def help(self, operator: str = None) -> None:
        """
        Display help for a specific CDO operator or general CDO help.

        Parameters
        ----------
        operator : str, optional
            The name of the CDO operator to get help for. If not provided, general CDO help is displayed.
        """
        cmd = ["cdo", "-h"]
        if operator:
            cmd.append(operator)
        res = self._cdo(cmd)
        print(res.stdout)

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

        res = self._cdo(cmd)

        txt = StringIO(res.stdout)
        txt = txt.read().splitlines()
        return txt

    def _bbox_from_griddes(self, infile) -> tuple[float, float, float, float]:

        data_dict = self.griddes(
            infile,
        )

        if len(data_dict) > 1:
            raise ValueError("Input file has multiple grids.")

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
            _tmp_files.extend([outfile, grdfile])

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
            lon_min, lat_min, lon_max, lat_max = self._bbox_from_griddes(obj)

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

        res = self._cdo(cmd)
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

        res = self._cdo(cmd)

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
        self._cdo(cmd)
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
        """Interpolate data using CDO's remapdis method."""

        return self._h_interp_data(
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
        """Interpolate data using CDO's remapnn method."""
        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
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
        """Interpolate data using CDO's remapcon method."""

        return self._h_interp_data(
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
        """Interpolate data using CDO's remapbil method."""

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
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
        """Interpolate data using CDO's remapbic method."""

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
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
        """Interpolate data using CDO's remaplaf method."""

        if extrapolate:
            os.environ["REMAP_EXTRAPOLATE"] = "on"

        return self._h_interp_data(
            obj=obj,
            outfile=outfile,
            resolution=resolution,
            method="remaplaf",
            bbox=bbox,
            as_xarray=as_xarray,
        )

    def showinfo(
        self,
        infile: Path,
    ) -> pd.DataFrame:
        """
        This module prints meta-data information of all input variables.

        Parameters
        ----------
        infile : str
            Input netCDF file.
        """
        operators = [
            "showcode",
            "showname",
            "showstdname",
            "showlevel",
            "showattribute",
        ]

        if not Path(infile).exists():
            raise FileNotFoundError(f"Input file {infile} does not exist.")

        results = {}

        attrs_df = None

        def _parse_attr(data):
            attrs_dict = {}
            for line in data:
                if line == "":
                    continue
                if "long_name" in line:
                    var = line.split("@")[0].strip()
                    attrs_dict[var] = {}
                    name = line.split("=")[-1].strip()
                    name = name.replace('"', "")
                    name = name.replace("'", "_")
                    attrs_dict[var]["long_name"] = name
                if "units" in line:
                    units = line.split("=")[-1].strip()
                    units = units.replace('"', "")
                    attrs_dict[var]["units"] = units

            attrs_df = pd.DataFrame(attrs_dict).T.reset_index()
            attrs_df.columns = ["short_name", "long_name", "units"]
            return attrs_df

        for op in operators:

            cmd = [op, infile]

            res = self._cdo(cmd)
            txt = StringIO(res.stdout)
            txt = txt.read().splitlines()

            if op == "showattribute":
                attrs_df = _parse_attr(txt)
                continue

            if op == "showlevel":
                res = [",".join([str(to_numeric(i)) for i in l.split()]) for l in txt]

            else:
                res = [to_numeric(i) for l in txt for i in l.split()]

            results[op.removeprefix("show")] = res

        df = pd.DataFrame(results)  # .explode("level")
        df = df.rename(columns={"name": "short_name"})
        df = df[["stdname", "short_name", "code", "level"]]
        df = df.merge(attrs_df, on="short_name", how="left")
        df = df[["long_name", "short_name", "stdname", "units", "code", "level"]]
        df = df.sort_values(by=["long_name", "short_name"]).reset_index(drop=True)

        return df

    def split(
        self,
        infile: Path,
        outdir: Path = None,
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
    ) -> list[Path]:
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
            outdir = Path(infile).parent / f"{op_dir}_files"
            # Start clean: remove old axis directory if present
            if outdir.exists():
                rm(outdir)
            outdir.mkdir(parents=True, exist_ok=True)

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

        outfiles = sorted([f for f in Path(prefix_dir).glob("*")])

        return outfiles


cdo: CDO = CDO()
