Combined from 4 rank-count run(s): [4, 8, 16, 32]. 44 (method, ranks) rows, 37 slower-than-native at their run's problem size.

| Method | Ranks | Native Xarray | MPI Xarray | Speedup | Accuracy | Dtype |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `coarsen_mean` | 4 | 0.1568 s | 0.3754 s | 0.42x SLOWER | PASS | PASS |
| `coarsen_mean` | 8 | 0.1541 s | 0.3783 s | 0.41x SLOWER | PASS | PASS |
| `coarsen_mean` | 16 | 0.2127 s | 0.2574 s | 0.83x SLOWER | PASS | PASS |
| `coarsen_mean` | 32 | 0.1555 s | 0.2117 s | 0.73x SLOWER | PASS | PASS |
| `diff` | 4 | 0.0216 s | 0.2561 s | 0.08x SLOWER | PASS | PASS |
| `diff` | 8 | 0.0216 s | 0.1427 s | 0.15x SLOWER | PASS | PASS |
| `diff` | 16 | 0.0218 s | 0.1949 s | 0.11x SLOWER | PASS | PASS |
| `diff` | 32 | 0.0215 s | 0.1529 s | 0.14x SLOWER | PASS | PASS |
| `differentiate` | 4 | 0.1347 s | 0.4184 s | 0.32x SLOWER | PASS | PASS |
| `differentiate` | 8 | 0.1333 s | 0.2909 s | 0.46x SLOWER | PASS | PASS |
| `differentiate` | 16 | 0.1338 s | 0.2054 s | 0.65x SLOWER | PASS | PASS |
| `differentiate` | 32 | 0.1329 s | 0.1475 s | 0.90x SLOWER | PASS | PASS |
| `isel` | 4 | 0.0001 s | 0.0981 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 8 | 0.0001 s | 0.0680 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 16 | 0.0001 s | 0.0755 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 32 | 0.0001 s | 0.0501 s | 0.00x SLOWER | PASS | PASS |
| `mean` | 4 | 0.0249 s | 0.2171 s | 0.11x SLOWER | PASS | PASS |
| `mean` | 8 | 0.0243 s | 0.2881 s | 0.08x SLOWER | PASS | PASS |
| `mean` | 16 | 0.0742 s | 0.2186 s | 0.34x SLOWER | PASS | PASS |
| `mean` | 32 | 0.0241 s | 0.1958 s | 0.12x SLOWER | PASS | PASS |
| `mpi_partition_data` | 4 | n/a | 0.1265 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 8 | n/a | 0.2210 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 16 | n/a | 0.2862 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 32 | n/a | 0.2627 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `np.log` | 4 | 0.0814 s | 0.1347 s | 0.60x SLOWER | PASS | PASS |
| `np.log` | 8 | 0.0815 s | 0.1917 s | 0.43x SLOWER | PASS | PASS |
| `np.log` | 16 | 0.2418 s | 0.1168 s | 2.07x | PASS | PASS |
| `np.log` | 32 | 0.0804 s | 0.1367 s | 0.59x SLOWER | PASS | PASS |
| `np.multiply` | 4 | 0.0212 s | 0.1095 s | 0.19x SLOWER | PASS | PASS |
| `np.multiply` | 8 | 0.0212 s | 0.0969 s | 0.22x SLOWER | PASS | PASS |
| `np.multiply` | 16 | 0.0559 s | 0.0607 s | 0.92x SLOWER | PASS | PASS |
| `np.multiply` | 32 | 0.0212 s | 0.0636 s | 0.33x SLOWER | PASS | PASS |
| `np.sqrt` | 4 | 0.0704 s | 0.1378 s | 0.51x SLOWER | PASS | PASS |
| `np.sqrt` | 8 | 0.0701 s | 0.2189 s | 0.32x SLOWER | PASS | PASS |
| `np.sqrt` | 16 | 0.1985 s | 0.1129 s | 1.76x | PASS | PASS |
| `np.sqrt` | 32 | 0.0702 s | 0.0975 s | 0.72x SLOWER | PASS | PASS |
| `rolling_mean` | 4 | 0.0669 s | 0.3325 s | 0.20x SLOWER | PASS | PASS |
| `rolling_mean` | 8 | 0.0669 s | 0.2192 s | 0.31x SLOWER | PASS | PASS |
| `rolling_mean` | 16 | 0.1958 s | 0.1548 s | 1.26x | PASS | PASS |
| `rolling_mean` | 32 | 0.0669 s | 0.1500 s | 0.45x SLOWER | PASS | PASS |
| `sum` | 4 | 0.0684 s | 0.2002 s | 0.34x SLOWER | PASS | PASS |
| `sum` | 8 | 0.0686 s | 0.1092 s | 0.63x SLOWER | PASS | PASS |
| `sum` | 16 | 0.0694 s | 0.1212 s | 0.57x SLOWER | PASS | PASS |
| `sum` | 32 | 0.0687 s | 0.1537 s | 0.45x SLOWER | PASS | PASS |
