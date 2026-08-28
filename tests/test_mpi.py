"""Entry point for the ``xarray.mpi`` correctness test suite.

Imports every ``test_mpi_*.py`` module in this directory and runs its checks
in one process, then reports a single pass/fail summary. Each module is also
runnable standalone (see its own docstring) -- this file just saves running
five separate ``mpirun`` invocations.

Covers ``mpi.xarray``'s split mixins: ``IOMixin``, ``IndexingMixin``,
``ReductionMixin``, ``StatisticsMixin``, ``GroupbyMixin``, and
``ArithmeticMixin`` (``align``/``apply``/``evaluate``/``matmul``/
``halo_exchange``/``rolling_reduce``) (see ``xarray/mpi.py``'s module
docstring for what each one contains). Does not cover ``mpi.reduce`` or the
parallel NetCDF writer -- those need separate test coverage.

Run with::

    mpirun -n N python -m mpi4py tests/test_mpi.py
"""

from __future__ import annotations

import test_mpi_groupby
import test_mpi_indexing
import test_mpi_io
import test_mpi_operator
import test_mpi_reductions
import test_mpi_statistics
from mpi_fixtures import finish

if __name__ == "__main__":
    test_mpi_io.test_repartition_dataarray()
    test_mpi_io.test_repartition_rejects_already_distributed()
    test_mpi_io.test_repartition_dataset()
    test_mpi_io.test_create_dataarray_fill_receives_global_bounds()
    test_mpi_io.test_create_dataset_multiple_variables()

    test_mpi_indexing.test_isel_scalar()
    test_mpi_indexing.test_isel_slice()
    test_mpi_indexing.test_sel_scalar()
    test_mpi_indexing.test_sel_slice()
    test_mpi_indexing.test_isel_scalar_dataset()
    test_mpi_indexing.test_isel_slice_singleton_default_no_repartition()
    test_mpi_indexing.test_isel_slice_singleton_repartition_auto()
    test_mpi_indexing.test_isel_slice_singleton_repartition_named()
    test_mpi_indexing.test_sel_slice_singleton_repartition_auto()
    test_mpi_indexing.test_isel_slice_singleton_repartition_auto_noop()
    test_mpi_indexing.test_isel_slice_singleton_repartition_dataset()
    test_mpi_indexing.test_isel_slice_singleton_repartition_invalid_dim()

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

    test_mpi_operator.test_apply_add_two_distributed()
    test_mpi_operator.test_apply_scalar_operand()
    test_mpi_operator.test_apply_rejects_unaligned_replicated_operand()
    test_mpi_operator.test_apply_rejects_partition_breaking_callable()
    test_mpi_operator.test_apply_matmul_redirect_matches_direct_matmul()
    test_mpi_operator.test_align_both_undistributed_with_dim()
    test_mpi_operator.test_align_one_replicated()
    test_mpi_operator.test_align_already_matching_partitions_is_noop()
    test_mpi_operator.test_align_incompatible_partitions_raises()
    test_mpi_operator.test_matmul_contracts_partition_dimension()
    test_mpi_operator.test_matmul_does_not_contract_partition_dimension()
    test_mpi_operator.test_evaluate_arithmetic_expression()
    test_mpi_operator.test_evaluate_matmul_operator_matches_direct()
    test_mpi_operator.test_evaluate_rejects_chained_comparison()
    test_mpi_operator.test_evaluate_rejects_and_or_on_xarray_operand()
    test_mpi_operator.test_evaluate_undefined_name_raises()
    test_mpi_operator.test_halo_exchange_matches_serial_neighbors()
    test_mpi_operator.test_halo_exchange_on_dataset_leaves_static_var_alone()
    test_mpi_operator.test_halo_exchange_rejects_halo_wider_than_local_partition()
    test_mpi_operator.test_rolling_reduce_matches_serial_rolling_mean()
    test_mpi_operator.test_rolling_reduce_matches_serial_with_sum_and_no_center()
    test_mpi_operator.test_rolling_reduce_non_partition_dim_delegates_to_xarray()
    test_mpi_operator.test_rolling_reduce_dataset_static_var_untouched()

    finish()
