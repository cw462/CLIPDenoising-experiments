# CLIPDenoising execution progress

Updated: 2026-10-20T23:30:01Z
Pipeline status: **completed**

Source hash: `40a2b826983afbcac68e0b4419d759272bbd0703f6614f092301bc3148489eef`

The user authorized two GPUs to run independent experiments in parallel.
Formal capacity: 4 independent jobs per GPU, 8 jobs total. Smoke and pilot retain two jobs total.
Currently monitored training jobs: 0; peak concurrently monitored: 8.
Each child sees one GPU and retains effective batch 16; no DataParallel/DDP is used.
Only the pipeline controller writes this shared queue state.

| Stage | Experiment | Seed | Physical GPU | GPU slot | State | Optimizer updates |
|---|---|---:|---:|---:|---|---:|
| smoke | B0 | 0 | 0 | 0 | completed | 100/100 |
| smoke | G1 | 0 | 1 | 0 | completed | 100/100 |
| pilot | B0 | 0 | 0 | 0 | completed | 5000/5000 |
| pilot | G1 | 0 | 1 | 0 | completed | 5000/5000 |
| full | B0 | 0 | 0 | 0 | completed | 300000/300000 |
| full | G1 | 0 | 1 | 0 | completed | 300000/300000 |
| full | B0 | 1 | 0 | 1 | completed | 300000/300000 |
| full | G1 | 1 | 1 | 1 | completed | 300000/300000 |
| full | B0 | 2 | 0 | 2 | completed | 300000/300000 |
| full | G1 | 2 | 1 | 2 | completed | 300000/300000 |
| full | G2 | 0 | 0 | 3 | completed | 300000/300000 |
| full | G3 | 0 | 1 | 3 | completed | 300000/300000 |

Only a validated final checkpoint at the prescribed update count marks training completed.
Final test sets are accessed after all eight full runs are completed.
Short-run validation uses only the fixed 32-image CBSD432 development split.
