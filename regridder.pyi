import xarray as xr
from .logs import log as log
from .tools import cwd as cwd, execute_cmd as execute_cmd, n_cpu as n_cpu
from os import PathLike

def ESMF_RegridWeightGen(**kwargs): ...
def regrid_cam_se(dataset: xr.Dataset, weight_file: PathLike) -> xr.Dataset: ...
