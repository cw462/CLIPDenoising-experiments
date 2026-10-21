"""Exercise actual runner recovery using real development images and RN50.

The three diagnostic updates use micro-batch 1. They are isolated from smoke,
pilot and full runs, whose configured effective batch remains 16.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from basicsr.utils import get_root_logger
from tools.experiment_common import atomic_json, digest, load_paths, seed_all
from tools.run_experiment import (
    ExactSampler, build_dataset, build_model, checkpoint_payload, make_config,
    restore_checkpoint, train_update, validate_checkpoint,
)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


# FP32 numerical acceptance, matching the plan's 1e-6 output comparison scale.
# CUDA interpolate gradients are documented as nondeterministic in PyTorch 2.2.
# State immediately after restore, RNG, inputs and counters remain bitwise checks.
CUDA_ATOL = 1e-6
CUDA_RTOL = 1e-5


def compare_states(expected, actual, location="root", stats=None, device="cpu"):
    """Keep saved state exact; allow small CUDA *computed* float differences.

    The original model contains CUDA bilinear-upsample backward operations.
    A resumed next update can differ slightly even with identical inputs, RNG
    and optimizer state. This does not justify relaxing RNG, counters, sampler,
    scheduler/LR, or checkpoint serialization checks. CPU remains bitwise.
    """
    if stats is None:
        stats = {"tensor_count": 0, "max_abs_difference": 0.0,
                 "bitwise_equal": True, "allclose": True, "accepted": True,
                 "groups": {}, "failures": []}
    group_name = (location.split(".", 2)[1] if "." in location else "root").split("[", 1)[0]
    group = stats["groups"].setdefault(group_name, {
        "tensor_count": 0, "array_count": 0, "scalar_count": 0,
        "max_abs_difference": 0.0, "max_difference_path": None,
        "bitwise_equal": True, "allclose": True, "accepted": True,
        "tolerant_float_tensor_count": 0, "exact_tensor_count": 0,
        "non_bitwise_tensor_count": 0, "first_non_bitwise_tensor_paths": [],
    })

    def record(bitwise, close, accepted, difference=0.0):
        previous_max = group["max_abs_difference"]
        for target in (stats, group):
            target["bitwise_equal"] = target["bitwise_equal"] and bitwise
            target["allclose"] = target["allclose"] and close
            target["accepted"] = target["accepted"] and accepted
            target["max_abs_difference"] = max(target["max_abs_difference"], difference)
        if difference > previous_max or (
            difference and group["max_difference_path"] is None
        ):
            group["max_difference_path"] = location
        if not accepted:
            stats["failures"].append({
                "location": location, "max_abs_difference": difference,
                "bitwise_equal": bitwise, "allclose": close,
            })

    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or expected.shape != actual.shape
                or expected.dtype != actual.dtype):
            raise AssertionError("Tensor shape/dtype differs: " + location)
        first, second = expected.detach().cpu(), actual.detach().cpu()
        stats["tensor_count"] += 1
        group["tensor_count"] += 1
        floating = first.is_floating_point() or first.is_complex()
        if floating and (not torch.isfinite(first).all() or not torch.isfinite(second).all()):
            raise AssertionError("Non-finite checkpoint tensor at " + location)
        # Byte comparison also distinguishes signed zero; torch.equal alone does not.
        bitwise = torch.equal(first.contiguous().reshape(-1).view(torch.uint8),
                              second.contiguous().reshape(-1).view(torch.uint8))
        numeric_type = torch.complex128 if first.is_complex() else torch.float64
        difference = (float((first.to(numeric_type) - second.to(numeric_type)).abs().max())
                      if first.numel() else 0.0)
        close = bool(torch.allclose(first, second, atol=CUDA_ATOL, rtol=CUDA_RTOL)) if floating else bitwise
        # AdamW stores its step counter as a floating tensor on some versions.
        allow_tolerance = (device == "cuda" and floating
                           and group_name in {"params", "optimizer", "ema"}
                           and not location.endswith(".step"))
        group["tolerant_float_tensor_count" if allow_tolerance else "exact_tensor_count"] += 1
        if not bitwise:
            group["non_bitwise_tensor_count"] += 1
            if len(group["first_non_bitwise_tensor_paths"]) < 20:
                group["first_non_bitwise_tensor_paths"].append(location)
        record(bitwise, close, close if allow_tolerance else bitwise, difference)
    elif isinstance(expected, np.ndarray):
        if (not isinstance(actual, np.ndarray) or expected.dtype != actual.dtype
                or expected.shape != actual.shape):
            raise AssertionError("NumPy shape/dtype differs: " + location)
        group["array_count"] += 1
        exact = expected.tobytes() == actual.tobytes()
        record(exact, exact, exact)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError("Dictionary keys differ: " + location)
        for key in expected:
            compare_states(expected[key], actual[key], location + "." + str(key), stats, device)
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError("Sequence lengths differ: " + location)
        for index, (first, second) in enumerate(zip(expected, actual)):
            compare_states(first, second, location + "[{}]".format(index), stats, device)
    else:
        group["scalar_count"] += 1
        exact = type(expected) is type(actual) and expected == actual
        record(exact, exact, exact)
    return stats


def check_resume(paths, device, accumulation):
    paths = copy.deepcopy(paths)
    paths["device"] = device
    opt, unused_run = make_config("G1", "smoke", 0, paths, tag="resume_contract")
    before_hash = digest(opt)
    if opt["runtime"]["noisy_views_per_update"] != 16:
        raise AssertionError("Formal effective batch must remain 16")
    seed_all(0)
    model = build_model(opt)
    dataset = build_dataset(opt, paths)
    if digest(opt) != before_hash:
        raise AssertionError("Model/dataset construction mutated effective config")
    sampler = ExactSampler(len(dataset), opt["runtime"]["sampler_seed"])
    seed_all(opt["runtime"]["data_rng_seed"])
    observed_paths = [dataset.paths[index] for index in sampler.order[:3 * accumulation]]
    losses = []
    for update in (1, 2):
        losses.append(train_update(model, dataset, sampler, update, 1, accumulation))
    if model.schedulers[0].last_epoch != 1:
        raise AssertionError("Original scheduler must be at epoch 1 after update 2")

    with tempfile.TemporaryDirectory(prefix="clip_real_resume_contract_") as directory:
        path = Path(directory) / "update_000002.pth"
        payload = checkpoint_payload(model, sampler, opt, 2, 0.0)
        torch.save(payload, path)
        del payload
        validated = validate_checkpoint(path, opt, expected=2)
        del validated
        bad_config = dict(opt, config_hash="deliberately_incorrect_contract_hash")
        try:
            validate_checkpoint(path, bad_config, expected=2)
        except RuntimeError:
            wrong_config_rejected = True
        else:
            raise AssertionError("Checkpoint with mismatched config hash was accepted")

        expected_loss = train_update(model, dataset, sampler, 3, 1, accumulation)
        expected_input = tensor_digest(model.lq)
        expected_gt = tensor_digest(model.gt)
        expected = checkpoint_payload(model, sampler, opt, 3, 0.0)

        # Fresh objects are created before restoring RNG; no state is reused
        # from the first process except the disk checkpoint and configuration.
        restored_model = build_model(opt)
        restored_dataset = build_dataset(opt, paths)
        restored_sampler = ExactSampler(len(restored_dataset), 999999)
        restored = validate_checkpoint(path, opt, expected=2)
        restore_checkpoint(restored_model, restored_sampler, restored)
        # Serialization/restoration itself must be bitwise even on CUDA.
        # Tolerance applies only after executing the next floating-point update.
        immediate_state = checkpoint_payload(restored_model, restored_sampler, opt, 2, 0.0)
        restored_state_comparison = compare_states(restored, immediate_state, device="cpu")
        if not restored_state_comparison["accepted"]:
            raise AssertionError(
                "Checkpoint restoration itself is not bitwise: {!r}".format(
                    restored_state_comparison["failures"]
                )
            )
        del immediate_state
        del restored
        if restored_model.schedulers[0].last_epoch != 1:
            raise AssertionError("Resume reset the scheduler")
        actual_loss = train_update(
            restored_model, restored_dataset, restored_sampler, 3, 1, accumulation
        )
        actual = checkpoint_payload(restored_model, restored_sampler, opt, 3, 0.0)
        comparison = compare_states(expected, actual, device=device)
        loss_exact = expected_loss == actual_loss
        loss_close = math.isclose(expected_loss, actual_loss, abs_tol=CUDA_ATOL, rel_tol=CUDA_RTOL)
        loss_accepted = loss_close if device == "cuda" else loss_exact
        actual_input = tensor_digest(restored_model.lq)
        actual_gt = tensor_digest(restored_model.gt)
        inputs_exact = expected_input == actual_input
        gt_exact = expected_gt == actual_gt

    return {
        "passed": comparison["accepted"] and loss_accepted and inputs_exact and gt_exact,
        "experiment": "G1",
        "checkpoint_saved_update": 2,
        "compared_update": 3,
        "real_training_images": len(dataset),
        "example_input_files": observed_paths,
        "diagnostic_micro_batch": 1,
        "diagnostic_accumulation": accumulation,
        "configured_official_effective_batch_unchanged": opt["runtime"]["noisy_views_per_update"],
        "not_counted_as_smoke_pilot_or_formal": True,
        "losses_updates_1_2": losses,
        "uninterrupted_update3_loss": expected_loss,
        "restored_update3_loss": actual_loss,
        "all_model_bn_optimizer_scheduler_rng_sampler_states_bitwise_equal": comparison["bitwise_equal"],
        "bitwise_equal": comparison["bitwise_equal"],
        "allclose": comparison["allclose"],
        "comparison_groups": comparison["groups"],
        "comparison_failures": comparison["failures"],
        "immediate_checkpoint_restore_bitwise_equal": restored_state_comparison["bitwise_equal"],
        "immediate_checkpoint_restore_comparison": restored_state_comparison,
        "float_acceptance": (
            {"mode": "CUDA computed model/optimizer floats only", "atol": CUDA_ATOL, "rtol": CUDA_RTOL}
            if device == "cuda" else {"mode": "CPU bitwise; no tolerance allowed"}
        ),
        "loss_comparison": {
            "exact_equal": loss_exact, "allclose": loss_close,
            "accepted": loss_accepted,
            "max_abs_difference": abs(expected_loss - actual_loss),
        },
        "noisy_input_bitwise_equal": inputs_exact,
        "gt_bitwise_equal": gt_exact,
        "compared_tensors": comparison["tensor_count"],
        "max_abs_tensor_difference": comparison["max_abs_difference"],
        "noisy_input_sha256": expected_input,
        "gt_sha256": expected_gt,
        "restored_noisy_input_sha256": actual_input,
        "restored_gt_sha256": actual_gt,
        "wrong_config_rejected": wrong_config_rejected,
        "scheduler_last_epoch_after_update3": restored_model.schedulers[0].last_epoch,
        "learning_rate_after_update3": restored_model.get_current_learning_rate(),
        "remaining_numerical_limits": (
            "CUDA allclose is not bitwise replay. Original CUDA floating-point kernels, "
            "including bilinear-upsample backward, may differ across executions. "
            "This short resume check does not bound long-horizon divergence over 300000 "
            "updates or guarantee cross-device/version reproducibility. RNG, sampler, "
            "integer and optimizer step counters, scheduler/LR, noisy inputs and GT "
            "remain exact; training algorithms and global determinism flags are unchanged."
            if device == "cuda" else
            "Bitwise recovery was checked on this CPU environment for one resumed update, "
            "not guaranteed across devices, software versions or 300000 updates."
        ),
        "pretrained_load": model.net_g.encoder.pretrain_load_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/local_paths.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/resume_contract.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    start = time.perf_counter()
    report = {
        "scope": "Actual runner with real dev images, pretrained RN50, isolated diagnostic updates",
        "device": args.device,
        "torch_version": torch.__version__,
        "tests": [],
        "passed": False,
        "acceptance_policy": {
            "cpu": "All state, loss, inputs and GT must be bitwise/exact.",
            "cuda": "Only computed model/optimizer floating tensors and loss may use strict allclose.",
            "cuda_atol": CUDA_ATOL, "cuda_rtol": CUDA_RTOL,
            "always_exact": ["RNG", "sampler", "integer counters", "optimizer step counters",
                             "scheduler", "learning rates", "noisy input", "ground truth"],
            "training_algorithm_changed": False,
            "global_determinism_flags_changed": False,
        },
    }
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(args.threads)
        get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
        paths = load_paths(args.paths)
        for accumulation in (1, 2):
            result = check_resume(paths, args.device, accumulation)
            report["tests"].append(result)
            if not result["passed"]:
                raise AssertionError(
                    "Resume numerical/state contract failed for accumulation={}; "
                    "see recorded comparison_groups, loss and input checks".format(accumulation)
                )
            print("RESUME_CONTRACT passed: diagnostic micro_batch=1 accumulation={}".format(
                accumulation
            ), flush=True)
        report["passed"] = True
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
    report["wall_seconds"] = time.perf_counter() - start
    atomic_json(args.report, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
