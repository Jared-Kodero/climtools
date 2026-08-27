"""Entry point for the ``xarray.mpi`` correctness test suite.

Imports every ``test_mpi_*.py`` module in this directory and runs its checks
in one process, then reports a single pass/fail summary. Each module is also
runnable standalone (see its own docstring) -- this file just saves running
five separate ``mpirun`` invocations.

Covers ``mpi.xarray``'s split mixins: ``IOMixin``, ``IndexingMixin``,
``ReductionMixin``, ``StatisticsMixin``, and ``GroupbyMixin`` (see
``xarray/mpi.py``'s module docstring for what each one contains). Does not
cover ``mpi.reduce``, ``ArithmeticMixin`` (``apply``/``align``/``evaluate``),
or the parallel NetCDF writer -- those need separate test coverage.

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi.py
"""

from __future__ import annotations

import test_mpi_groupby
import test_mpi_indexing
import test_mpi_io
import test_mpi_reductions
import test_mpi_statistics
from mpi_fixtures import finish

if __name__ == "__main__":
    test_mpi_io.test_redistribute_dataarray()
    test_mpi_io.test_redistribute_rejects_already_distributed()
    test_mpi_io.test_redistribute_dataset()
    test_mpi_io.test_create_dataarray_fill_receives_global_bounds()
    test_mpi_io.test_create_dataset_multiple_variables()

    test_mpi_indexing.test_isel_scalar()
    test_mpi_indexing.test_isel_slice()
    test_mpi_indexing.test_sel_scalar()
    test_mpi_indexing.test_sel_slice()
    test_mpi_indexing.test_isel_scalar_dataset()

    test_mpi_reductions.test_numeric_reductions()
    test_mpi_reductions.test_any_all()
    test_mpi_reductions.test_first_last()
    test_mpi_reductions.test_reduction_on_non_partition_dim()
    test_mpi_reductions.test_dataset_reduction_with_static_variable()

    test_mpi_statistics.test_var_std_ddof0()
    test_mpi_statistics.test_var_std_ddof1()
    test_mpi_statistics.test_var_on_non_partition_dim()
    test_mpi_statistics.test_dataset_var()

    test_mpi_groupby.test_resample_reduce_dataarray()
    test_mpi_groupby.test_resample_reduce_dataset()
    test_mpi_groupby.test_groupby_reduce_categorical()
    test_mpi_groupby.test_groupby_reduce_on_non_partition_dim()

    finish()
