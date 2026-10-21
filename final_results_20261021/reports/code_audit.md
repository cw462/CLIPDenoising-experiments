# CLIPDenoising code and calibration audit

Source baseline: `alwaysuu/CLIPDenoising`, commit
`2647bf815df0299a03d16df34b896fa266ceefda`.

Execution performance revision: full decoded RGB images are cached without
changing sampled tensors or RNG, and the eight formal runs use independent
processes across two GPUs. See `performance_optimization.md` for measured
capacity, version archival and acceptance evidence. The local CPU model
acceptance measurements below describe the initial implementation; revised
server acceptance JSON files are rerun under the final frozen source.

This audit describes the implementation. Executed acceptance results, device,
timing, exact parameter counts, and pretrained matching counts are recorded in
`reports/calibration_contract.json`; missing weights cause an explicit skip and
`pretrained_acceptance_complete=false`, never a substitute random-weight run.
Synthetic inputs in these tests assess structure and gradients only. They are
not denoising benchmark or training results.

Executed on 2026-10-20 with PyTorch `2.13.0+cpu`, four CPU threads, and the
SHA256-verified official RN50 archive: all four tests passed without skips.
All 330 backbone state keys matched, covering 165 parameter tensors and
23,527,264 parameters; missing keys and shape mismatches were empty. The B0
versus original-source and B0 versus initial-G1 maximum absolute differences
were both exactly zero. All 165 BN buffers changed during training while every
encoder parameter remained identical. The three update test showed positive
residual/decoder gradients immediately and positive projection/trunk/gate
gradients from update two onward.

| Model | Total parameters | Trainable parameters | Added parameters |
|---|---:|---:|---:|
| B0 | 34,516,192 | 10,988,928 | 0 |
| G1 | 34,533,748 | 11,006,484 | 17,556 |
| G2 | 34,533,715 | 11,006,451 | 17,523 |
| G3 | 34,528,100 | 11,000,836 | 11,908 |

## Official call chain and preserved behavior

The official YAML constructs `ImageCleanModel` → `define_network` →
`CLIPDenoising` → `ModifiedResNet_RemoveFinalLayer` and five decoder blocks.
For `num_blocks=[3,4,6,3]`, `wf=64`, and a 128×128 RGB tensor, encoder outputs are:

| Index | Tensor | Channels | Spatial size |
|---|---|---:|---|
| 0 | Original noisy RGB, cast to encoder dtype | 3 | 128×128 |
| 1 | Three-convolution stem output | 64 | 64×64 |
| 2 | layer1 output | 256 | 32×32 |
| 3 | layer2 output | 512 | 16×16 |
| 4 | layer3 output | 1024 | 8×8 |

`encoder.layer4` still executes, produces 2048 channels at 4×4, and updates
training-mode BN statistics even though that feature is not returned. This
original computation is intentionally retained.

Decoder `up_path[0]` converts 1024→512 at 8×8 without upsampling or a skip.
The subsequent blocks produce 256 at 16×16, 128 at 32×32, 64 at 64×64, and
64 at 128×128. In the final block only, the upsampled decoder has 64 channels,
and `center_crop(out[0], up.shape[2:])` has 3 channels. The concatenation order
remains `[up, cropped_RGB]`. `up_path.4.conv_block` and `last` keep their original
names, shapes, initialization order, and reconstruction roles.

All original upsampling remains bilinear with `align_corners=True`. Evaluation
reflect-pads on the right and bottom to a multiple of 32, then crops to the
original height and width. Padding is an evaluation wrapper operation, as in
the source; it is not silently inserted into the training network.

Training images are RGB floating point divided by 255 with additive Gaussian
noise. Gaussian input is not clamped, resized to 224, or passed through CLIP
classification mean/std normalization. Output clamping belongs to the official
evaluation path only.

The original four PFA multipliers remain `1 + N(0, (gamma*i)^2)` for `i=1..4`.
The main configuration uses `gamma=0.025`. PFA is applied only in training and
never to the raw RGB skip. Encoder parameters have `requires_grad=False`; the
encoder is still switched to training mode with the containing model, preserving
the official BN buffer updates. No `encoder.eval()` or new `no_grad()` scope is
inserted into its forward path.

Official training retains L1, AdamW, gradient norm clipping at 0.01, and Mixup
after Gaussian sampling. Mixup changes the effective variance of mixed inputs;
the protocol is “base Gaussian sampling sigma=15 with official Mixup.”

## Final RGB calibration

The existing network accepts `adapter_mode`:

| Experiment | Mode | Correction |
|---|---|---|
| B0 | `none` | Original path |
| G1 | `gated` | `y + sigmoid(gate(t)) * residual(t)` |
| G2 | `ungated` | `y + residual(t)`; gate layer omitted |
| G3 | `no_decoder_guidance` | Gated correction using only `y` and `h`; projection omitted |

Only `up_path[4].rgb_calibration` receives a module. It is invoked after
`center_crop` and before the original concat. The number of decoder channels is
derived from the actual final convolution input width minus RGB channels.

`h = y - B(y)` uses a fixed 3×3 average filter with one-pixel reflection padding.
For G1/G2, a learned 1×1 projection maps decoder channels to 16; its output is
concatenated with RGB and h. Two 3×3 convolutions map 22→32→32 with ReLU.
G3 uses 6→32→32. The residual 3×3 convolution maps 32→3 and both its weight and
bias are zero-initialized. The gate is a 32→1 1×1 convolution plus sigmoid.
No extra outer zero scale or clamp is used.

New parameters are under `up_path.4.rgb_calibration.*`. The adapter is created
after every original layer, inside `torch.random.fork_rng(devices=[])`. This
preserves both paired original initialization and the subsequent CPU RNG stream.
The new convolutions are constructed on CPU before the standard device move.

`CLIPDenoising.load_baseline_state_dict` requires exact equality of shared keys
and shapes and allows only the complete set of new adapter keys to be absent.
Unexpected or missing original keys raise an exception before copying weights.
Ordinary full-model checkpoint recovery continues to use strict loading.

For selected visualizations, set
`model.up_path[4].rgb_calibration.capture_diagnostics=True`, run one forward,
and read detached `last_maps["g"]`, `["r"]`, and `["g_times_r"]`. Disable capture
after use. Normal training does not retain these maps or their computation
graphs. Gates express correction modulation, not calibrated confidence;
high-pass input contains both texture and noise.

## Strict pretrained checkpoint acceptance

`ModifiedResNet.load_pretrain_model` accepts a real OpenAI TorchScript RN50
archive or a tensor state dictionary. Non-JIT loading resets the file pointer
and uses `weights_only=True`. A complete CLIP state uses the `visual.` prefix;
an explicitly extracted backbone state can use unprefixed names.

Every target backbone parameter and buffer must match in name and shape,
including unused-output layer4 and BN buffers. Missing or incompatible tensors
are fatal. The complete key and element audit is exposed as
`model.encoder.pretrain_load_report`, and a compact `RN50_LOAD_AUDIT` is printed.
Unused source text/attention-pooling keys are listed explicitly. They do not
exist in the denoising backbone and are not mistaken for unloaded target layers.

## Acceptance command

```powershell
.\.venv\Scripts\python.exe tests/test_calibration_contract.py --weights pretrained/RN50.pt --report reports/calibration_contract.json --threads 4 --device cpu
```

The same command supports `--device cuda` on a CUDA PyTorch host. The test
compares B0 with the original architecture source locked at the official commit, compares
zero-initialized G1 with B0, tests the actual concat order, validates dimensions
128×128 and 137×151 via 160×160 reflection padding, checks exact paired initial
states and RNG, rejects incorrect baseline transfers and pretrained tensors,
and runs three full-network L1/AdamW updates. It checks decoder and residual
gradients on update one and projection/trunk/gate gradients after the zero
residual has learned. Encoder parameters must remain identical; train-mode
BN buffers must change consistently in B0/G1. Layer4 BN is explicitly checked.

Measured synthetic batch=1 timing is labeled as such and must not be presented
as official effective-batch-16 training throughput. Persistence, scheduler,
optimizer-count, dataset, and end-to-end experiment-runner tests are separate
gates owned by the main pipeline.

`tests/test_resume_contract.py` also passed on the same CPU environment using
the actual runner functions, actual 400-image development dataset, and real RN50.
Diagnostic micro-batch=1 was tested with accumulation=1 and accumulation=2:
save after update 2, reconstruct all objects, restore from disk, and compare
update 3 with uninterrupted training. In both cases all 415 model/BN/optimizer/
scheduler/RNG tensors were bitwise identical, maximum absolute difference zero.
Sampler state, Python/NumPy state, noisy input and Mixup target hashes also
matched. The scheduler resumed at epoch 1 and advanced to epoch 2. Incorrect
configuration hashes were rejected. These isolated diagnostic updates do not
count toward smoke, pilot, or formal training; the effective batch 16 in the
experiment configuration was not modified. Details are in
`reports/resume_contract.json`.
