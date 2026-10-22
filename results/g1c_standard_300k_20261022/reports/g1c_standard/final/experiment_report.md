# G1C vs historical G1: three seeds, 300k updates

Evaluation completed with the user-requested direct mode. Additional recovery and seed0/seed2 noisy-input consistency checks were skipped. Seed0 also skipped training-asset and historical-checkpoint identity checks. Same-seed differences below are descriptive comparisons; identical historical noise is not established for the combined results. Actual new and baseline noise hashes are retained separately. Training was not restarted. Original failures remain archived.

Mean and sample SD across three training seeds. OOD uses the original hierarchical aggregation.

| Dataset | Noise | G1 PSNR | G1C PSNR | Difference dB |
|---|---|---:|---:|---:|
| CBSD68 | gaussian 50.0 | 26.1944 +/- 0.4051 | 26.5228 +/- 0.1561 | +0.3284 +/- 0.4349 |
| CBSD68 | poisson 3.5 | 27.6981 +/- 0.1007 | 27.7265 +/- 0.0445 | +0.0284 +/- 0.1257 |
| CBSD68 | spatial_gaussian 55.0 | 27.3865 +/- 0.1543 | 27.3713 +/- 0.2168 | -0.0152 +/- 0.1183 |
| Kodak24 | salt_pepper 0.02 | 23.7728 +/- 0.4120 | 23.8545 +/- 0.2592 | +0.0818 +/- 0.1673 |
| McMaster | spatial_gaussian 55.0 | 27.9125 +/- 0.2424 | 27.9405 +/- 0.2589 | +0.0280 +/- 0.2128 |
| Urban100 | gaussian 50.0 | 25.1613 +/- 0.3695 | 25.3466 +/- 0.1081 | +0.1853 +/- 0.3419 |
