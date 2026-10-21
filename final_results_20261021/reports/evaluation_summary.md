# Fixed-checkpoint evaluation

Status: completed. Actual per-image measurements: 25200.
All included checkpoints are at update 300000. B0/G1 seed variation is the sample standard deviation over seed-level condition means. G2/G3 have one seed; no multi-seed stability claim is justified.

Every dataset/noise/strength condition is in condition_results.csv; paired_per_image_results.csv links each difference to matching image and noise hashes. Official three-family and extended five-family OOD aggregates are separate. Each OOD aggregate weights images, within-family strengths, noise families, and datasets equally at successive levels. Gaussian sigma15 is excluded.

runtime_results.csv uses the same 256x256 batch1 input size, 3 warmups, and 10 timed forwards. Per-image file I/O, noise generation, inference, and visual saving are recorded separately. CPU runs have no CUDA peak-memory measurement.

Negative image/condition/seed comparisons retained: 4819. failure_cases.csv lists every negative paired difference, worst first. Fixed-example comparison panels: 48. Display scales and crop coordinates are identical across B0/G1. Gate maps indicate modulation of a correction, not calibrated noise probability.

The flat/texture/edge masks are post-inference heuristic GT-gradient bins; GT is never an input to the network. A visual pattern is a mechanism hypothesis, not proof of denoising causality.
