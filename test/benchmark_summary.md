Combined from 4 rank-count run(s): [4, 8, 16, 32]. 44 (method, ranks) rows, 33 slower-than-native at their run's problem size.

| Method | Ranks | Native Xarray | MPI Xarray | Speedup | Accuracy | Dtype |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `coarsen_mean` | 4 | 0.1174 s | 0.2572 s | 0.46x SLOWER | PASS | PASS |
| `coarsen_mean` | 8 | 0.1172 s | 0.1686 s | 0.70x SLOWER | PASS | PASS |
| `coarsen_mean` | 16 | 0.1206 s | 0.1914 s | 0.63x SLOWER | PASS | PASS |
| `coarsen_mean` | 32 | 0.1248 s | 0.1839 s | 0.68x SLOWER | PASS | PASS |
| `diff` | 4 | 0.0196 s | 0.1576 s | 0.12x SLOWER | PASS | PASS |
| `diff` | 8 | 0.0195 s | 0.0915 s | 0.21x SLOWER | PASS | PASS |
| `diff` | 16 | 0.0211 s | 0.1453 s | 0.15x SLOWER | PASS | PASS |
| `diff` | 32 | 0.0410 s | 0.1445 s | 0.28x SLOWER | PASS | PASS |
| `differentiate` | 4 | 0.1143 s | 0.2913 s | 0.39x SLOWER | PASS | PASS |
| `differentiate` | 8 | 0.1152 s | 0.2137 s | 0.54x SLOWER | PASS | PASS |
| `differentiate` | 16 | 0.1245 s | 0.1628 s | 0.76x SLOWER | PASS | PASS |
| `differentiate` | 32 | 0.2062 s | 0.1263 s | 1.63x | PASS | PASS |
| `isel` | 4 | 0.0001 s | 0.0755 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 8 | 0.0001 s | 0.0432 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 16 | 0.0001 s | 0.0503 s | 0.00x SLOWER | PASS | PASS |
| `isel` | 32 | 0.0001 s | 0.0378 s | 0.00x SLOWER | PASS | PASS |
| `mean` | 4 | 0.0172 s | 0.1663 s | 0.10x SLOWER | PASS | PASS |
| `mean` | 8 | 0.0172 s | 0.1035 s | 0.17x SLOWER | PASS | PASS |
| `mean` | 16 | 0.0173 s | 0.1229 s | 0.14x SLOWER | PASS | PASS |
| `mean` | 32 | 0.0439 s | 0.0737 s | 0.60x SLOWER | PASS | PASS |
| `mpi_partition_data` | 4 | n/a | 0.0573 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 8 | n/a | 0.0920 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 16 | n/a | 0.1802 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `mpi_partition_data` | 32 | n/a | 0.4426 s | n/a (no native counterpart) | n/a (no native counterpart) | n/a |
| `np.log` | 4 | 0.0694 s | 0.1019 s | 0.68x SLOWER | PASS | PASS |
| `np.log` | 8 | 0.0697 s | 0.0615 s | 1.13x | PASS | PASS |
| `np.log` | 16 | 0.0740 s | 0.0804 s | 0.92x SLOWER | PASS | PASS |
| `np.log` | 32 | 0.1757 s | 0.0993 s | 1.77x | PASS | PASS |
| `np.multiply` | 4 | 0.0197 s | 0.1219 s | 0.16x SLOWER | PASS | PASS |
| `np.multiply` | 8 | 0.0197 s | 0.0538 s | 0.37x SLOWER | PASS | PASS |
| `np.multiply` | 16 | 0.0213 s | 0.0478 s | 0.45x SLOWER | PASS | PASS |
| `np.multiply` | 32 | 0.0484 s | 0.0416 s | 1.16x | PASS | PASS |
| `np.sqrt` | 4 | 0.0572 s | 0.0974 s | 0.59x SLOWER | PASS | PASS |
| `np.sqrt` | 8 | 0.0576 s | 0.0573 s | 1.00x | PASS | PASS |
| `np.sqrt` | 16 | 0.0597 s | 0.0729 s | 0.82x SLOWER | PASS | PASS |
| `np.sqrt` | 32 | 0.1622 s | 0.0502 s | 3.23x | PASS | PASS |
| `rolling_mean` | 4 | 0.0549 s | 0.2439 s | 0.23x SLOWER | PASS | PASS |
| `rolling_mean` | 8 | 0.0552 s | 0.1382 s | 0.40x SLOWER | PASS | PASS |
| `rolling_mean` | 16 | 0.0585 s | 0.1113 s | 0.53x SLOWER | PASS | PASS |
| `rolling_mean` | 32 | 0.1128 s | 0.1177 s | 0.96x SLOWER | PASS | PASS |
| `sum` | 4 | 0.0586 s | 0.0873 s | 0.67x SLOWER | PASS | PASS |
| `sum` | 8 | 0.0590 s | 0.0545 s | 1.08x | PASS | PASS |
| `sum` | 16 | 0.0624 s | 0.0869 s | 0.72x SLOWER | PASS | PASS |
| `sum` | 32 | 0.0642 s | 0.0905 s | 0.71x SLOWER | PASS | PASS |
