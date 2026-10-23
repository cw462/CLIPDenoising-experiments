# G1C three-seed 300k results

All three training seeds completed 300,000 updates. Evaluation contains 9,450 new G1C measurements and 9,450 reused historical G1 measurements, compared by seed; identical noise is not established for the combined results.

At the user request, added recovery prechecks, seed0/seed2 noisy-input identity checks and the independent publication re-audit were skipped. The model inference and metric calculations were retained. Seed 1 results were reused; seeds 0 and 2 ran on separate GPUs with noisy-input consistency blocking disabled. A descriptive summary uses the original aggregation functions, without requiring noise identity. Actual new and baseline noise hashes are retained. Training was not restarted.

The first seed 0 evaluation stopped on a historical noisy-input mismatch. Two recovery precheck attempts also stopped (pretrained asset identity mismatch; seed 1 contract/identity comparison mismatch). Subsequent pretrained-file reads matched the recorded SHA256. The causes are unresolved; these checks are not reported as passed. Original failure artifacts and the direct-evaluation authorization record are retained under reproducibility/g1c_standard.

[Experiment report](reports/g1c_standard/final/experiment_report.md)

[OOD summary](reports/g1c_standard/final/ood_summary.csv)

Seed 0 additionally skipped training-asset and historical-checkpoint identity checks. The final CPU summary initially failed with an MKL/OpenMP threading conflict and completed with MKL_THREADING_LAYER=GNU; no model evaluations were repeated for that summary error.
