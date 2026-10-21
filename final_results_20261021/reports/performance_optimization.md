# Measured execution optimization, 2026-10-20

The user requested using both GPUs more fully. The original one-run-per-GPU
execution was preserved under `archive_pre_cache/`, including its source
archive, completed smoke/pilot runs, interrupted early formal attempts, and
unaltered hashed metadata. `performance_revision.json` maps historical paths.
Those incomplete formal attempts do not count toward the new formal budget.
The old revision's checkpoint storage was subsequently relocated to
`D:\projects\CLIPDenoising_20261020\archives\archive_pre_cache.tar` after matching
the complete archive's local and remote SHA256
`1443b22f107f63826c133dfb37bec4f249ca8328cb8b1e34f1933c43b4822e21`.
This freed 1.265 GiB of server disk without affecting current revision
checkpoints. Historical source, logs and metadata remain on the server;
`archived_checkpoint_storage.json` and the local archive receipt record
the exact storage mapping and checkpoint hashes.

The revised runner caches decoded full RGB float32 images, with an immutable
canonical array and a separate working copy for every access. It does not cache
patches, noise, augmentation, or training batches. Random crop, augmentation,
noise, Mixup, PFA, BN behavior, optimizer, schedule and batch 16 remain unchanged.
The cache uses about 763 MiB per full-training process. A cold cache is rebuilt
from the same hash-verified data after restart and need not be checkpointed.

`tests/test_cache_contract.py` compares the production implementation against
the complete original dataset class bundled from upstream commit
`2647bf815df0299a03d16df34b896fa266ceefda`. The test checks all 400 development
training images and all 432 formal images over two permutations, default-off
behavior, mutation isolation, exact input/GT/RNG equality, and cold-cache
recovery. Actual device-specific results are in `cache_contract.json`.

CPU-only tests of 1, 2, 4 and 8 intra-op threads favored 4 threads for this
data path. Thread count alone did not remove the main repeated decoding cost.

Separate real-data GPU diagnostics used batch 16, the actual model, 30 warmup
updates per process, a common start barrier and 100 measured updates per
process. They never wrote formal checkpoints or accessed final test images.
Throughput is total completed updates divided by the shared wall-clock window,
including the slowest job; it is not the sum of unrelated per-job peak speeds.

| Configuration | Both-GPU aggregate updates/s |
|---|---:|
| Original uncached, 1 run/GPU | 20.50 |
| Cached, 1 run/GPU, two repetitions | 28.44 / 26.21 |
| Cached, 2 runs/GPU, two repetitions | 33.93 / 36.24 |
| Cached, 4 runs/GPU, two repetitions | 42.90 / 42.76 |
| Cached, 4 runs/GPU, 1000 updates/process | 41.97 |

`capacity_benchmark/cached_4_sustained/summary.json` records a longer 1000-update
per-process capacity check, and `performance_audit_sustained.json` records
temperature, clocks, memory and utilization during sustained operation.
These are capacity diagnostics, not accuracy results or completed formal runs.
During the sustained window, GPU utilization averaged 97% and 99%, with at
least 8237 MiB and 6718 MiB of GPU memory free, respectively. Available host RAM
remained above 68 GiB. The eight jobs completed 8000 diagnostic updates in
190.63 seconds. The slowest-job whole-window throughput is about 2.05 times the
original uncached two-job measurement.

The new configured formal capacity is four independent processes on each of
two RTX 3090 GPUs. The fixed eight formal runs fit one wave in their original
queue order with round-robin GPU assignment. Smoke and pilot still use one
process per GPU. Each formal run retains its own batch 16, paired initialization,
optimizer, seed/RNG, 300000 updates, output directory and full recovery state.
Neither DataParallel/DDP nor batch 64 is used. All formal runs use the same
revised source and begin from paired initialization; none inherits pilot or
archived attempt weights.

CUDA's original floating-point nondeterminism remains as documented in
`resume_numerics.md`; exact cached inputs do not imply bitwise equality of
independently executed CUDA training trajectories.

The capacity estimate is not a guaranteed finish time. Other workloads share
these GPUs, and full training includes checkpoint writes and later evaluation.
The persistent pipeline continues automatically through all formal runs,
fixed final tests and report generation.
