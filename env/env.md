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