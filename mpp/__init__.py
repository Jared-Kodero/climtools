"""FMS ``mpp`` communication layer, adapted to xarray objects.

The modules here mirror the FMS source files of the same names so the two can
be read side by side:

===========================  ==============================================
:mod:`~climtools.mpp.mpp`             rank-level collectives and communicators
:mod:`~climtools.mpp.mpp_domains`     compute domains, halos, redistribution
:mod:`~climtools.mpp.mpp_efp`         reproducible extended-fixed-point sums
===========================  ==============================================

Everything is re-exported here, so callers write ``from ..mpp import
mpp_update_domains`` without needing to know which file it lives in.
"""

from .mpp import (
    _mpp_reduce as _mpp_reduce,
)
from .mpp import (
    mpp_alltoall,
    mpp_broadcast,
    mpp_chksum,
    mpp_gather,
    mpp_max,
    mpp_min,
    mpp_partition_offsets,
    mpp_reduce_scatter,
    mpp_scatter,
    mpp_sum,
    mpp_sync,
)
from .mpp_domains import (
    CartesianDomain,
    Domain,
    DomainMismatchError,
    DomainUpdate,
    HaloWidthError,
    mpp_check_field,
    mpp_complete_update_domains,
    mpp_define_cartesian_domain,
    mpp_define_domains,
    mpp_define_layout,
    mpp_dim_comm,
    mpp_get_cartesian_domain,
    mpp_get_compute_domain,
    mpp_get_compute_domains,
    mpp_get_data_domain,
    mpp_get_domain_extents,
    mpp_get_global_domain,
    mpp_get_layout,
    mpp_get_neighbor_pe,
    mpp_get_pelist,
    mpp_global_field,
    mpp_global_max,
    mpp_global_min,
    mpp_global_sum,
    mpp_halo_exchange,
    mpp_redistribute,
    mpp_slice_compute_domain,
    mpp_start_update_domains,
    mpp_update_domains,
)
from .mpp_efp import (
    MAX_EFP_RANKS,
    MAX_PROD_RANKS,
    PROD_EXPONENT,
    PROD_INF,
    PROD_NAN,
    PROD_NEGATIVE,
    PROD_ZERO,
    mpp_prod_decompose,
    mpp_prod_recombine,
    mpp_reproducing_prod,
    mpp_reproducing_sum,
)

__all__ = [
    "MAX_EFP_RANKS",
    "MAX_PROD_RANKS",
    "PROD_EXPONENT",
    "PROD_INF",
    "PROD_NAN",
    "PROD_NEGATIVE",
    "PROD_ZERO",
    "CartesianDomain",
    "Domain",
    "DomainMismatchError",
    "DomainUpdate",
    "HaloWidthError",
    "mpp_alltoall",
    "mpp_broadcast",
    "mpp_check_field",
    "mpp_chksum",
    "mpp_complete_update_domains",
    "mpp_define_cartesian_domain",
    "mpp_define_domains",
    "mpp_define_layout",
    "mpp_dim_comm",
    "mpp_gather",
    "mpp_get_cartesian_domain",
    "mpp_get_compute_domain",
    "mpp_get_compute_domains",
    "mpp_get_data_domain",
    "mpp_get_domain_extents",
    "mpp_get_global_domain",
    "mpp_get_layout",
    "mpp_get_neighbor_pe",
    "mpp_get_pelist",
    "mpp_global_field",
    "mpp_global_max",
    "mpp_global_min",
    "mpp_global_sum",
    "mpp_halo_exchange",
    "mpp_max",
    "mpp_min",
    "mpp_partition_offsets",
    "mpp_prod_decompose",
    "mpp_prod_recombine",
    "mpp_redistribute",
    "mpp_reduce_scatter",
    "mpp_reproducing_prod",
    "mpp_reproducing_sum",
    "mpp_scatter",
    "mpp_slice_compute_domain",
    "mpp_start_update_domains",
    "mpp_sum",
    "mpp_sync",
    "mpp_update_domains",
]
