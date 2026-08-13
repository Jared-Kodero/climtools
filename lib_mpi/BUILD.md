# Build and test the `mpi` package on Oscar

The package is self-contained under `mpi/`. Its native implementation is
`mpi_netcdf.c`, with the public C ABI declared in `mpi_netcdf.h`; `native.py`
provides the `ctypes` binding. Installation writes the platform-specific shared
library to `mpi/lib/libmpi_netcdf.so` and the verified runtime environment to
`mpi/modules.yml`.

The August 2026 Oscar module catalog includes `hpcx-mpi/2.25.1s`,
`netcdf-c-mpi/4.9.3`, and `python/3.13.5`. Module defaults can change. The
installer therefore discovers candidate versions of `hpcx-mpi`, `openmpi`,
`netcdf-c-mpi`, and `netcdf-mpi` unless exact MPI and NetCDF modules are
provided as overrides. Python is taken from an activated environment unless a
Python module or executable is specified explicitly.

## Python environment

The writer requires NumPy and xarray. Prepare them in the Python environment
that will run the package. For example:

```bash
module load python/3.13.5
python -m venv "$HOME/venvs/mpi-netcdf"
source "$HOME/venvs/mpi-netcdf/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy xarray
```

An activated virtual or Conda environment is used automatically. Alternatively,
set `MPI_PACKAGE_PYTHON_EXECUTABLE` to an absolute Python path. If that
interpreter depends on an Oscar module, also set `MPI_PACKAGE_PYTHON_MODULE` to
the exact module name. The installer loads that module after selecting the MPI
and NetCDF stack, then verifies that parallel NetCDF remains available.

## Installation with a real parallel-I/O test

Run the installer inside an allocation containing at least two tasks:

```bash
cd /path/to/the/directory/containing/mpi
salloc --nodes=1 --ntasks=2 --time=00:10:00 --mem-per-cpu=1G
./mpi/install.sh
exit
```

The installer performs all of the following before reporting success:

1. Purges the module environment and selects one compatible MPI and parallel
   NetCDF-C stack.
2. Optionally loads `MPI_PACKAGE_PYTHON_MODULE`, selects the requested or active
   Python executable, and requires NumPy and xarray.
3. Requires `nc-config --has-nc4=yes` and
   `nc-config --has-parallel4=yes` in the final environment.
4. Preprocesses `mpi.h`, `netcdf.h`, `netcdf_par.h`, and `netcdf_meta.h` with
   the active `mpicc` and NetCDF include flags.
5. Compiles `verify_parallel_netcdf.c` and launches it on two ranks. The probe
   calls `nc_create_par`, defines a NetCDF-4 variable, enables collective
   access, writes one value per rank, reopens the file, and verifies all values.
   Installation stops if any stage fails.
6. Builds `mpi/lib/libmpi_netcdf.so`, checks its dynamic dependencies and
   exported C ABI, then imports the Python package against that exact binary.
7. Atomically writes the successfully verified module selection and Python
   executable to `mpi/modules.yml`.

The two-rank write is the decisive check. A module name or configuration flag
alone cannot detect a mismatched MPI runtime, serial HDF5 appearing earlier in
the loader path, or a nonfunctional parallel NetCDF-C installation.

Exact module, interpreter, and launcher overrides are available for
nonstandard stacks:

```bash
MPI_PACKAGE_MPI_MODULE=hpcx-mpi/2.25.1s \
MPI_PACKAGE_NETCDF_MODULE=netcdf-c-mpi/4.9.3 \
MPI_PACKAGE_PYTHON_MODULE=python/3.13.5 \
MPI_PACKAGE_PYTHON_EXECUTABLE=/absolute/path/to/python \
MPI_PACKAGE_PROBE_LAUNCHER='srun --mpi=pmix --ntasks=2' \
./mpi/install.sh
```

Outside a Slurm allocation, the installer uses `mpirun -n 2` or
`mpiexec -n 2` when available. `MPI_PACKAGE_PROBE_LAUNCHER` takes precedence.

## Saved runtime environment

After every successful installation, `mpi/modules.yml` records the modules that
must be reloaded and the exact Python executable used for verification. A
typical file has this structure:

```yaml
schema_version: 1
load_order:
  - 'hpcx-mpi/2.25.1s-le4f'
  - 'netcdf-mpi/4.9.3-kuxq'
python:
  module: null
  executable: '/users/example/miniconda3/envs/mpi-netcdf/bin/python'
resolved_modules:
  - 'hpcx-mpi/2.25.1s-le4f'
  - 'hdf5-mpi/1.14.6-wybi'
  - 'netcdf-c-mpi/4.9.3-lezy'
  - 'netcdf-mpi/4.9.3-kuxq'
```

Load the entries under `load_order` in the listed order before a later run.
Entries under `resolved_modules` document the complete verified dependency
stack and normally load automatically. Use the absolute interpreter under
`python.executable`. When `python.module` is not `null`, it is already included
in `load_order` and must be loaded before that interpreter is run.

The installer executes as a child process, so its module changes do not modify
the calling shell. For the example above, a later shell would be prepared with:

```bash
module purge
module load hpcx-mpi/2.25.1s-le4f
module load netcdf-mpi/4.9.3-kuxq
/users/example/miniconda3/envs/mpi-netcdf/bin/python -m mpi.test_mpi
```

Use the module names and Python environment recorded by the local
`mpi/modules.yml`, not the illustrative values above.

## Complete Python test

From the directory containing `mpi/`, request four tasks, restore the runtime
environment recorded in `mpi/modules.yml`, and run:

```bash
salloc --nodes=1 --ntasks=4 --time=00:10:00 --mem-per-cpu=1G
module purge
module load <each-load_order-entry-in-order>
srun --mpi=pmix --ntasks=4 <python.executable> -m mpi.test_mpi
exit
```

Replace the placeholders with the values in the generated file. Issue one
`module load` command for each `load_order` entry.

`test_mpi.py` configures rank-aware logging, broadcasts root configuration,
performs deterministic work on every rank, gathers and validates the results,
removes a previous output collectively, constructs unequal xarray time slabs,
and invokes the parallel NetCDF writer as its final task. Rank zero then
requires the completed output to exist and be non-empty. The output defaults to
`mpi_parallel_test.nc`; set `MPI_TEST_OUTPUT` to change it.

## Use in another script

Run from the directory containing the package, or add that directory to
`PYTHONPATH`:

```python
from mpi import MPI_RANK, MPI_SIZE, info, to_netcdf
```

Every process in `MPI_COMM_WORLD` must call `to_netcdf` in the same order. Each
process supplies its contiguous slab of `partition_dim`; replicated variables,
coordinates, attributes, and writer options must be identical.

Keep the module stack and Python environment recorded in `mpi/modules.yml`
active at runtime. Oscar requires `srun --mpi=pmix` for MPI jobs. Do not load a
second MPI implementation or a serial HDF5/NetCDF stack after restoring the
verified environment.

## References

- [Brown CCV, RHEL 9.6 module migration guide](https://docs.ccv.brown.edu/oscar/sys-changes/rhel-9.6-module-migration-guide)
- [Brown CCV, MPI jobs](https://docs.ccv.brown.edu/oscar/submitting-jobs/mpi-jobs)
- [Unidata NetCDF-C, `nc_create_par` and parallel access](https://docs.unidata.ucar.edu/netcdf-c/4.9.2/group__datasets.html)
- [Open MPI wrapper compiler documentation](https://docs.open-mpi.org/en/v5.0.1/man-openmpi/man1/ompi-wrapper-compiler.1.html)