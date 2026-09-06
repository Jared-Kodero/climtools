## 1. Environment / build requirements

The published `env/environment.yml`/`env/setup_env.py` target a real HPC module
system. This session's sandbox had none of that, so a parallel MPI/HDF5/NetCDF stack
was built from source — these are the exact steps, useful if you ever need to
reproduce a similar from-scratch environment (a CI container, a fresh cloud VM, etc.):

```bash
# System packages: OpenMPI + parallel HDF5 + build tooling
apt-get install -y libhdf5-openmpi-dev openmpi-bin libopenmpi-dev \
    m4 zlib1g-dev libcurl4-openssl-dev libjpeg-dev automake libtool \
    bison flex cmake python3-venv python3-dev

# NetCDF-C from source, linked against the *openmpi* HDF5 build specifically --
# CMake's find_package(HDF5) silently picked the serial HDF5 build in testing;
# autotools' ./configure correctly picked up the parallel one via these env vars.
git clone --branch v4.9.3 https://github.com/Unidata/netcdf-c.git
cd netcdf-c && mkdir build-autotools && cd build-autotools
CC=mpicc \
CPPFLAGS="-I/usr/include/hdf5/openmpi" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi" \
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/openmpi:$LD_LIBRARY_PATH" \
../configure --prefix=/usr/local --enable-parallel-tests --disable-dap --disable-byterange
make -j"$(nproc)" && make install && ldconfig
# Verify: nc-config --has-parallel  ->  must print "yes"
# Verify: ldd /usr/local/lib/libnetcdf.so | grep hdf5  ->  must show libhdf5_openmpi.so, NOT libhdf5_serial.so

# Python: venv with fresh setuptools (system pip's is too old for some sdists)
python3 -m venv venv && venv/bin/pip install --upgrade pip setuptools wheel cython numpy

# mpi4py + netCDF4 built against the same parallel stack
CC=mpicc HDF5_MPI=ON \
CPPFLAGS="-I/usr/include/hdf5/openmpi -I/usr/local/include" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/usr/local/lib" \
venv/bin/pip install mpi4py
CC=mpicc HDF5_MPI=ON \
CPPFLAGS="-I/usr/include/hdf5/openmpi -I/usr/local/include" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/usr/local/lib" \
venv/bin/pip install --no-binary netCDF4 --no-build-isolation netCDF4
# Verify: python -c "import netCDF4; print(netCDF4.__has_parallel4_support__)"  ->  must print 1

venv/bin/pip install pandas scipy xarray dask cf_xarray rich bottleneck
```

Always set `LD_LIBRARY_PATH` to include the openmpi HDF5 path (and `PATH` to include
`/usr/local/bin` for the just-built `nc-config`/`ncdump`) before running anything —
otherwise the serial HDF5/NetCDF the system package manager installs by default can
get picked up silently at runtime even if the build itself was correct.

**Running the test suite**: `test/mpi_test_common.py`'s `build_fixtures()` calls
`create_dataset(n_time=24*30, resolution_deg=0.25, plev_step=100)` by default — this
is real climate-model-scale (721x1440 grid, 720 timesteps) and will exhaust memory on
a small sandbox. For local iteration, temporarily shrink these
(`n_time=64, resolution_deg=5` is a reasonable smoke-test size), but **always run the
full production-scale config at least once before trusting a change** — several bugs
this session (the HDF5 chunk-size limit, the intermittent `prod` issue) only manifest
at production scale or under real parallel filesystem conditions a small local repro
won't exercise. Also worth knowing: `mock_dataset.py`'s RNG is unseeded by default;
seed it (`np.random.default_rng(0)`) for deterministic debugging, but test unseeded too
since at least one real, data-dependent edge case was only ever found that way.
## 2. Reproducing the environment without a source NetCDF build

The from-scratch NetCDF-C build above is only necessary when no packaged
parallel build exists. On Ubuntu 24.04 (noble) it does, and using it removes
the longest step:

```bash
apt-get install -y --no-install-recommends \
    openmpi-bin libopenmpi-dev libhdf5-openmpi-dev libnetcdf-mpi-dev
# Verify: grep "NC-4 Parallel Support" \
#   /usr/lib/x86_64-linux-gnu/netcdf/mpi/libnetcdf.settings  ->  must say "yes"

python3 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel "cython>=3,<4" numpy
CC=mpicc HDF5_MPI=ON \
NETCDF4_DIR=/usr/lib/x86_64-linux-gnu/netcdf/mpi \
HDF5_INCDIR=/usr/include/hdf5/openmpi \
HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
  venv/bin/pip install --no-binary netCDF4 --no-build-isolation "netCDF4==1.6.5"
# Verify: python -c "import netCDF4; print(netCDF4.__has_parallel4_support__)" -> 1
```

Three things this session had to discover the hard way:

* `HDF5_DIR`/`CPPFLAGS` are *not* enough for netCDF4's build script, which
  looks for the headers under `$HDF5_DIR/include` and gives up before reading
  `CPPFLAGS`. `HDF5_INCDIR`/`HDF5_LIBDIR` are the variables it honours for a
  Debian-style split HDF5 layout.
* netCDF4 must be pinned. Versions from 1.7 on ship a `netcdf-compat.h` that
  redeclares `nc_def_var_bzip2`/`nc_inq_var_bzip2`/`nc_def_var_blosc`/
  `nc_inq_var_blosc` as static, which collides with the non-static
  declarations in the packaged netcdf-c 4.9.0 and fails to compile. 1.6.5
  builds cleanly against it.
* `apt-get update` fails on the NodeSource repository (403 on its InRelease
  file) in a default container image. Remove or disable that source first,
  or apt refuses to proceed even though the Ubuntu archives are reachable.

Set `LD_LIBRARY_PATH` to include both
`/usr/lib/x86_64-linux-gnu/hdf5/openmpi` and
`/usr/lib/x86_64-linux-gnu/netcdf/mpi/lib` at runtime, for the same
serial-picked-up-silently reason as above.

**Running the test suite**: `build_fixtures()` defaults to the
production-scale config and now reads `CLIMTOOLS_TEST_NTIME`,
`CLIMTOOLS_TEST_RESOLUTION` and `CLIMTOOLS_TEST_PLEV_STEP` when they are set,
so a small local run no longer means editing the call and risking committing
the shrunk size. `CLIMTOOLS_TEST_NTIME=48 CLIMTOOLS_TEST_RESOLUTION=10` is a
usable smoke-test size. The warning above still stands: run the full-size
default at least once before trusting a change.
