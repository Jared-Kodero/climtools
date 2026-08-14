# Build and test the MPI-NetCDF library

The native MPI-NetCDF library is built by `install.sh`. It first uses a working
MPI and parallel NetCDF-C stack already present on `PATH`; on systems with Lmod
or Environment Modules, it can discover a compatible stack automatically.

For writing Python against the built library, see [README.md](README.md).

The build requires an MPI C compiler, NetCDF-C with parallel NetCDF-4 support,
Python, NumPy, and xarray. The resulting shared library is written to
`lib/libmpi_netcdf.so`. The verified build configuration is recorded in
`build/build.yml`.

## Install

Activate the Python environment that will use the package and install its Python
dependencies:

```bash
python -m pip install numpy xarray
```

Then run the installer:

```bash
./install.sh
```

It accepts three options and is otherwise configured through the environment:

```text
-h, --help     show usage and exit
-f, --force    rebuild even when an up-to-date library is present
-c, --clean    remove build/ and lib/, then exit
```

Without `--force`, a library newer than the C sources is left alone.

For the strongest verification, run the installer inside an allocation that can
launch at least two MPI ranks. Under Slurm, for example:

```bash
salloc --nodes=1 --ntasks=2 --time=00:10:00 --mem-per-cpu=1G
./install.sh
exit
```

The installer performs the following checks before reporting success:

1. Reuses a working MPI and parallel NetCDF-C toolchain already on `PATH`, or
   searches the module system for a compatible stack.
2. Verifies that `mpi.h`, `netcdf.h`, `netcdf_par.h`, and `netcdf_meta.h` are
   usable with the selected `mpicc` and NetCDF compiler flags.
3. Requires `NC_HAS_PARALLEL4` at compile time.
4. Builds and, when a two-rank launcher is available, runs a parallel NetCDF-4
   write probe.
5. Builds `lib/libmpi_netcdf.so`, checks its dynamic dependencies and exported
   C ABI, and imports the Python binding against that exact library.
6. Confirms the ABI version the library reports matches the one the Python
   layer expects, catching a stale `libmpi_netcdf.so` left over from an
   earlier checkout.
7. Records the verified compiler, flags, modules, Python executable,
   parallel-filter capability, ABI version and probe outcome in
   `build/build.yml`.

If no two-rank launcher is available, the installer emits a warning and
continues after the compile-time capability checks. Set `MPI_NETCDF_SKIP_PROBE=1`
only when intentionally skipping the runtime probe.

The probe is retried oversubscribed when the host has too few cores to place
two ranks, and with `--allow-run-as-root` when running as root, since both are
scheduling limits rather than NetCDF capability limits. If every attempt
fails, the first failure is re-run with its output shown.

## Configuration

The installer is configured only through environment variables. It accepts no
positional or option arguments.

```text
MPI_NETCDF_MODULE         NetCDF-C module to load instead of searching
MPI_NETCDF_MPI_MODULE     MPI module to load instead of searching
MPI_NETCDF_PYTHON_MODULE  Python module to load before building
MPI_NETCDF_PYTHON         absolute path to the Python interpreter
MPI_NETCDF_CFLAGS         NetCDF compiler flags, overriding discovery
MPI_NETCDF_LIBS           NetCDF linker flags, overriding discovery
MPI_NETCDF_LAUNCHER       two-rank launcher, for example "mpirun -n 2"
MPI_NETCDF_SKIP_PROBE     set to 1 to skip the two-rank runtime probe
```

A typical explicit HPC configuration is:

```bash
MPI_NETCDF_MPI_MODULE=hpcx-mpi/2.25.1s \
MPI_NETCDF_MODULE=netcdf-c-mpi/4.9.3 \
MPI_NETCDF_LAUNCHER='srun --mpi=pmix --ntasks=2' \
./install.sh
```

If `mpicc` and a parallel NetCDF-C installation are already available, no module
overrides are needed. Compiler and linker flags are discovered from
`nc-config` or `pkg-config` unless `MPI_NETCDF_CFLAGS` and `MPI_NETCDF_LIBS` are
set explicitly.

## Runtime

Use the same MPI and NetCDF runtime stack that was verified during installation.
The exact build state is available in `build/build.yml`, which is also read at
import to reload the recorded module stack when one was used.

Do not mix a second MPI implementation or a serial HDF5/NetCDF stack into the
runtime environment. This is not merely a performance caution: the `netCDF4`
wheel on PyPI bundles its own serial HDF5, and loading it into a process that
already holds a parallel HDF5 will segfault when the two sets of symbols
collide. The package works around the common case by importing `netCDF4`
before it loads the native library, so the wheel's own symbols are pinned
first, but the safe configuration remains a single consistent stack, for
example a conda `netcdf4` built against the same MPI.

`MPI_NETCDF_LIBRARY` overrides the location of the compiled library, which is
useful when testing a build without reinstalling.

Under Slurm, launch MPI programs with the MPI mode required by the local
cluster. On Brown Oscar this is typically `srun --mpi=pmix`.

## References

- [Brown CCV, RHEL 9.6 module migration guide](https://docs.ccv.brown.edu/oscar/sys-changes/rhel-9.6-module-migration-guide)
- [Brown CCV, MPI jobs](https://docs.ccv.brown.edu/oscar/submitting-jobs/mpi-jobs)
- [Unidata NetCDF-C parallel I/O](https://docs.unidata.ucar.edu/netcdf-c/current/parallel_io.html)
- [Open MPI wrapper compiler documentation](https://docs.open-mpi.org/en/main/man-openmpi/man1/ompi-wrapper-compiler.1.html)
