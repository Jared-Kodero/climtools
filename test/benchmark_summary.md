Combined from 4 rank-count run(s): [4, 8, 16, 32]. 44 (method, ranks) rows, 34 slower-than-native at their run's problem size.

| Method | Ranks | Native Xarray | MPI Xarray | Speedup | Accuracy | Dtype |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `coarsen_mean` | 4 | 0.9088 s | 2.0592 s | 0.44x SLOWER | PASS | PASS |
| `coarsen_mean` | 8 | 0.9050 s | 1.1166 s | 0.81x SLOWER | PASS | PASS |
| `coarsen_mean` | 16 | 0.9073 s | 1.1614 s | 0.78x SLOWER | PASS | PASS |
| `coarsen_mean` | 32 | 1.5513 s | 0.7794 s | 1.99x | PASS | PASS |
| `diff` | 4 | 0.0845 s | 1.2475 s | 0.07x SLOWER | PASS | PASS |
| `diff` | 8 | 0.0827 s | 0.7070 s | 0.12x SLOWER | PASS | PASS |
| `diff` | 16 | 0.0808 s | 0.7465 s | 0.11x SLOWER | PASS | PASS |
| `diff` | 32 | 0.1705 s | 0.6026 s | 0.28x SLOWER | PASS | PASS |
| `differentiate` | 4 | 0.5796 s | 2.2618 s | 0.26x SLOWER | PASS | PASS |
| `differentiate` | 8 | 0.5883 s | 1.2574 s | 0.47x SLOWER | PASS | PASS |
| `differentiate` | 16 | 0.5799 s | 1.2776 s | 0.45x SLOWER | PASS | PASS |
| `differentiate` | 32 | 1.1530 s | 0.8303 s | 1.39x | PASS | PASS |
| `isel` | 4 | 0.0004 s | 0.6128 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 8 | 0.0004 s | 0.3432 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 16 | 0.0004 s | 0.3055 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 32 | 0.0006 s | 0.1887 s | 0.00x SLOWER | PASS | PASS |
| `mean` | 4 | 0.1567 s | 1.3203 s | 0.12x SLOWER | PASS | PASS |
| `mean` | 8 | 0.1571 s | 0.8243 s | 0.19x SLOWER | PASS | PASS |
| `mean` | 16 | 0.1570 s | 0.8995 s | 0.17x SLOWER | PASS | PASS |
| `mean` | 32 | 0.2716 s | 0.5763 s | 0.47x SLOWER | PASS | PASS |
| `mpi_partition_data` | 4 | n/a | 0.2190 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 8 | n/a | 0.2931 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 16 | n/a | 0.5262 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 32 | n/a | 0.9994 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `np.log` | 4 | 0.3669 s | 0.8229 s | 0.45x SLOWER | PASS | PASS |
| `np.log` | 8 | 0.3700 s | 0.4707 s | 0.79x SLOWER | PASS | PASS |
| `np.log` | 16 | 0.3697 s | 0.4958 s | 0.75x SLOWER | PASS | PASS |
| `np.log` | 32 | 0.7959 s | 0.3144 s | 2.53x | PASS | PASS |
| `np.multiply` | 4 | 0.0818 s | 0.6698 s | 0.12x SLOWER | PASS | PASS |
| `np.multiply` | 8 | 0.0822 s | 0.4081 s | 0.20x SLOWER | PASS | PASS |
| `np.multiply` | 16 | 0.0844 s | 0.4123 s | 0.20x SLOWER | PASS | PASS |
| `np.multiply` | 32 | 0.1685 s | 0.2565 s | 0.66x SLOWER | PASS | PASS |
| `np.sqrt` | 4 | 0.3550 s | 0.7429 s | 0.48x SLOWER | PASS | PASS |
| `np.sqrt` | 8 | 0.3546 s | 0.4392 s | 0.81x SLOWER | PASS | PASS |
| `np.sqrt` | 16 | 0.3534 s | 0.4419 s | 0.80x SLOWER | PASS | PASS |
| `np.sqrt` | 32 | 0.4939 s | 0.2401 s | 2.06x | PASS | PASS |
| `rolling_mean` | 4 | 0.3294 s | 1.9321 s | 0.17x SLOWER | PASS | PASS |
| `rolling_mean` | 8 | 0.3286 s | 1.0789 s | 0.30x SLOWER | PASS | PASS |
| `rolling_mean` | 16 | 0.3306 s | 1.1595 s | 0.29x SLOWER | PASS | PASS |
| `rolling_mean` | 32 | 0.7927 s | 0.7430 s | 1.07x | PASS | PASS |
| `sum` | 4 | 0.3107 s | 0.6825 s | 0.46x SLOWER | PASS | PASS |
| `sum` | 8 | 0.3156 s | 0.4297 s | 0.73x SLOWER | PASS | PASS |
| `sum` | 16 | 0.3139 s | 0.4713 s | 0.67x SLOWER | PASS | PASS |
| `sum` | 32 | 0.7856 s | 0.2978 s | 2.64x | PASS | PASS |
