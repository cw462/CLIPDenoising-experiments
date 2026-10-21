"""Full-budget C/G1C training with legacy global PFA and fail-closed inputs.

The original model, optimizer, sampler, Mixup and train_update are reused.
Replays verify inputs; only the first call's RNG transition is committed.
They are detection measures for an unresolved input anomaly, not a repair.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

_CUDA_INITIALIZED_BEFORE_IMPORT = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
_CUBLAS_CONFIG_BEFORE_IMPORT = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from basicsr.utils import get_root_logger
from tools import run_experiment as legacy
from tools.experiment_common import (
    RunLock, atomic_json, digest, load_paths, process_info, read_json, restore_rng,
    rng_state, seed_all, sha256_file, utc_now,
)
from tools.run_pfa_experiment import durable_log_rows, encoder_hash

VARIANTS = {"C": "B0", "G1C": "G1"}
SCHEMA = "c_only_input_protection_v1"
PROOF_FLAGS = ("dataset_replay_exact", "dataset_rng_exact", "feed_replay_exact", "raw_unchanged",
               "feed_rng_exact", "optimizer_inputs_unchanged", "snapshot_copies_exact")


def configure_execution(cuda):
    """Keep legacy TF32 settings, while requiring reproducible kernels."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before process startup")
    if cuda and _CUDA_INITIALIZED_BEFORE_IMPORT and _CUBLAS_CONFIG_BEFORE_IMPORT != ":4096:8":
        raise RuntimeError("CUDA was initialized before CUBLAS_WORKSPACE_CONFIG was configured")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


# Public compatibility name used by the independent actual-GPU contract.
configure_deterministic_execution = configure_execution


def configuration(variant, stage, seed, paths, tag=None):
    if (variant not in VARIANTS or stage != "full" or isinstance(seed, bool)
            or not isinstance(seed, int) or seed not in (0, 1, 2)
            or (variant == "C" and seed != 0)):
        raise ValueError("Full training supports G1C seeds 0/1/2; legacy C supports seed 0 only")
    if variant not in paths.get("formal_variants", VARIANTS) or seed not in paths.get("formal_seeds", (0, 1, 2)):
        raise ValueError("Variant/seed is outside the explicitly configured training scope")
    if paths["micro_batch"] != 16 or paths["accumulation"] != 1 or paths.get("amp", False):
        raise ValueError("Full C/G1C requires true batch 16, accumulation 1 and FP32")
    opt, _ = legacy.make_config(VARIANTS[variant], "full", seed, paths, tag)
    directory = Path(paths["output_root"]) / stage / variant / f"seed{seed}"
    if tag:
        directory /= tag  # make_config already validates the tag.
    opt.update(exp=variant, experiment=variant, name=f"full_{variant}_seed{seed}")
    opt["network_g"].update(pfa_rho=0.0, pfa_seed=None)
    opt["runtime"]["cpu_threads"] = int(paths.get("cpu_threads", 1))
    opt["path"].update(pretrain_network_g=None, experiments_root=str(directory),
                       models=str(directory / "checkpoints"), training_states=str(directory / "checkpoints"),
                       log=str(directory), visualization=str(directory / "visuals"))
    opt["train"]["pixel_opt"] = {
        "type": "L1StructureLoss", "loss_weight": 1.0, "reduction": "mean",
        "structure_weight": 0.05, "eps": 0.001, "deterministic_reflect_padding": True,
    }
    for field, default in (("checkpoint_every", 5000), ("checkpoint_keep", 2), ("status_every", 20)):
        value = paths.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
        opt["runtime"][field] = value
    opt["logger"]["save_checkpoint_freq"] = opt["runtime"]["checkpoint_every"]
    opt["full_training"] = {
        "schema": "c_only_full_v1", "variant": variant, "baseline_experiment": VARIANTS[variant],
        "baseline_training_seed": seed,
        "initialization": "RN50 pretrained encoder + seeded original decoder/adapter",
        "from_c0": False, "legacy_global_pfa": True, "budget": 300000,
        "diagnostic_tag": tag,
    }
    opt["execution_contract"] = {
        "schema": SCHEMA, "dataset": "two independent calls under identical RNG; exact bytes",
        "feed": "two calls on the same raw batch under identical RNG; exact bytes and RNG",
        "snapshot": "source before/after and independent clone/readback bytes must agree",
        "optimizer": "actual input/target unchanged; legacy global PFA RNG consumption allowed",
        "rng": "replay restores the first call's post-state; no retry on discrepancy",
        "deterministic_algorithms": True, "warn_only": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cudnn_allow_tf32": True, "cuda_matmul_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "historical_execution": {"cudnn_benchmark": False, "cudnn_deterministic": False,
                                 "cudnn_allow_tf32": True, "cuda_matmul_allow_tf32": False,
                                 "deterministic_algorithms": False},
        "historical_difference": "deterministic algorithms/cuDNN and CUBLAS workspace now enforced; "
                                 "FP32 parameter dtype and historical TF32 flags retained; "
                                 "historical trajectories are not claimed bitwise identical",
        "unresolved_input_anomaly": "guards detect discrepancies; root cause is not established",
    }
    opt.pop("config_hash", None)
    opt["config_hash"] = digest(opt)
    return opt, directory


class InputProtectionError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def _bytes_once(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_bytes(value):
    """Two actual reads, including separate CUDA-to-CPU transfers on CUDA."""
    first, second = _bytes_once(value), _bytes_once(value)
    if first != second:
        raise InputProtectionError("Repeated actual tensor readback differs",
                                   {"actual": value, "first_bytes": first, "second_bytes": second})
    return first


def tensor_hash(value):
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def exact_equal(first, second):
    if torch.is_tensor(first):
        if not torch.is_tensor(second) or first.dtype != second.dtype or first.shape != second.shape:
            return False
        # CUDA integer equality checks actual device tensors, not reference hashes.
        if first.device == second.device and first.is_cuda:
            integer = torch.int32 if first.dtype == torch.float32 else torch.uint8
            if not torch.equal(first.detach().contiguous().reshape(-1).view(integer),
                               second.detach().contiguous().reshape(-1).view(integer)):
                return False
        return tensor_bytes(first) == tensor_bytes(second)
    if isinstance(first, np.ndarray):
        return (isinstance(second, np.ndarray) and first.dtype == second.dtype
                and first.shape == second.shape and first.tobytes() == second.tobytes())
    if isinstance(first, dict):
        return (isinstance(second, dict) and first.keys() == second.keys()
                and all(exact_equal(first[key], second[key]) for key in first))
    if isinstance(first, (tuple, list)):
        return (type(first) is type(second) and len(first) == len(second)
                and all(exact_equal(a, b) for a, b in zip(first, second)))
    return type(first) is type(second) and first == second


def checked_clone(value):
    """Never assume a clone is faithful: independently verify source and copy."""
    if torch.is_tensor(value):
        before = tensor_bytes(value)
        result = value.detach().clone(memory_format=torch.preserve_format)
        after, cloned = tensor_bytes(value), tensor_bytes(result)
        if before != after or before != cloned or not exact_equal(value, result):
            raise InputProtectionError("Snapshot clone/source bytes differ", {
                "source": value, "clone": result, "source_before_bytes": before,
                "source_after_bytes": after, "clone_bytes": cloned})
        return result
    if isinstance(value, np.ndarray):
        result = value.copy()
        if not exact_equal(value, result):
            raise InputProtectionError("NumPy snapshot differs", {"source": value, "clone": result})
        return result
    if isinstance(value, dict):
        return {key: checked_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(checked_clone(item) for item in value)
    return copy.deepcopy(value)


class Protection:
    def __init__(self, directory, opt):
        self.directory, self.opt = Path(directory), opt
        self.update = 0
        self.items, self.feed, self.optimized = [], None, False
        self.sampler_state = None

    def fail(self, reason, evidence=None):
        failure = self.directory / "input_failure"
        failure.mkdir(parents=True, exist_ok=True)
        target = failure / f"update_{self.update:06d}_{time.time_ns()}.pth"
        payload = {"reason": str(reason), "update": self.update, "rng_at_failure": rng_state(),
                   "sampler_before_update": self.sampler_state, "evidence": evidence,
                   "config_hash": self.opt["config_hash"], "source_hash": self.opt["source_hash"]}
        with target.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(failure / "failure.json", {"reason": str(reason), "update": self.update,
                    "timestamp": utc_now(), "evidence": str(target), "sha256": sha256_file(target),
                    "config_hash": self.opt["config_hash"], "source_hash": self.opt["source_hash"],
                    "retry_allowed": False})
        raise InputProtectionError(str(reason), evidence)

    def begin_update(self, update, sampler=None):
        self.update = update
        self.items, self.feed, self.optimized = [], None, False
        self.sampler_state = copy.deepcopy(sampler.state_dict()) if sampler is not None else None

    def finish_update(self):
        if len(self.items) != 16 or self.feed is None or not self.optimized:
            self.fail("Incomplete input protection coverage", {"items": len(self.items),
                        "feed": self.feed, "optimized": self.optimized})
        return {"update": self.update, "dataset_replay_count": len(self.items),
                "dataset_sha256": digest(self.items), **{flag: True for flag in PROOF_FLAGS},
                **self.feed}


class ProtectedDataset:
    def __init__(self, dataset, protection):
        self.dataset, self.protection = dataset, protection

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        guard = self.protection
        evidence = {"index": index, "rng_before": rng_state()}
        try:
            first = self.dataset[index]
            evidence.update(first=first, rng_after_first=rng_state())
            snapshot = checked_clone(first)
            evidence["snapshot"] = snapshot
            restore_rng(evidence["rng_before"])
            second = self.dataset[index]
            evidence.update(second=second, rng_after_second=rng_state())
            if not exact_equal(snapshot, first) or not exact_equal(snapshot, second):
                raise InputProtectionError("Dataset replay differs")
            if not exact_equal(evidence["rng_after_first"], evidence["rng_after_second"]):
                raise InputProtectionError("Dataset replay RNG differs")
            restore_rng(evidence["rng_after_first"])
            guard.items.append({"index": index, "lq": tensor_hash(first["lq"]),
                                "gt": tensor_hash(first["gt"])})
            return first
        except InputProtectionError as error:
            evidence["comparison_failure"] = error.evidence
            guard.fail(str(error), evidence)


def protect_model(model, protection):
    """Wrap only this model instance; retain the original training implementations."""
    original_feed, original_optimize = model.feed_train_data, model.optimize_parameters

    def feed(data):
        evidence = {"raw": data, "rng_before": rng_state()}
        try:
            raw_snapshot = checked_clone(data)
            evidence["raw_snapshot"] = raw_snapshot
            original_feed(data)
            first = {"lq": model.lq, "gt": model.gt}
            evidence.update(first=first, rng_after_first=rng_state())
            first_snapshot = checked_clone(first)
            evidence["first_snapshot"] = first_snapshot
            if not exact_equal(data, raw_snapshot):
                raise InputProtectionError("Mixup modified raw batch")
            restore_rng(evidence["rng_before"])
            original_feed(data)
            second = {"lq": model.lq, "gt": model.gt}
            evidence.update(second=second, rng_after_second=rng_state())
            if not exact_equal(first, first_snapshot) or not exact_equal(second, first_snapshot):
                raise InputProtectionError("Mixup replay differs")
            if not exact_equal(data, raw_snapshot):
                raise InputProtectionError("Mixup modified raw batch")
            if not exact_equal(evidence["rng_after_first"], evidence["rng_after_second"]):
                raise InputProtectionError("Mixup replay RNG differs")
            # Use the actual first outputs, with their original layout and aliasing.
            model.lq, model.gt = first["lq"], first["gt"]
            restore_rng(evidence["rng_after_first"])
            protection.feed = {"input_sha256": tensor_hash(model.lq), "target_sha256": tensor_hash(model.gt)}
        except InputProtectionError as error:
            evidence["comparison_failure"] = error.evidence
            protection.fail(str(error), evidence)

    def optimize(update):
        evidence = {"actual_before": {"lq": model.lq, "gt": model.gt}, "rng_before": rng_state()}
        try:
            before = checked_clone(evidence["actual_before"])
            evidence["snapshot"] = before
            if protection.feed is None or (tensor_hash(model.lq) != protection.feed["input_sha256"]
                                           or tensor_hash(model.gt) != protection.feed["target_sha256"]):
                raise InputProtectionError("Inputs changed between feed and optimizer")
            original_optimize(update)
            after = {"lq": model.lq, "gt": model.gt}
            evidence.update(actual_after=after, rng_after=rng_state())
            if not exact_equal(before, after):
                raise InputProtectionError("Optimizer modified input/target")
            protection.feed.update(post_input_sha256=tensor_hash(model.lq),
                                   post_target_sha256=tensor_hash(model.gt))
            # Global RNG consumption by the original PFA is intentional.
            protection.optimized = True
        except InputProtectionError as error:
            evidence["comparison_failure"] = error.evidence
            protection.fail(str(error), evidence)

    model.feed_train_data, model.optimize_parameters = feed, optimize
    return model


def build_protected_model(opt, protection):
    return protect_model(legacy.build_model(opt), protection)


def build_protected_dataset(opt, paths, protection):
    return ProtectedDataset(legacy.build_dataset(opt, paths), protection)


def guarded_train_update(model, dataset, sampler, current, protection):
    protection.begin_update(current, sampler)
    loss = legacy.train_update(model, dataset, sampler, current, 16, 1)
    return loss, protection.finish_update()


def initialization_contract(model, opt):
    if opt["path"].get("pretrain_network_g") is not None:
        raise RuntimeError("Full training cannot load C0 or another restoration checkpoint")
    if not isinstance(model.optimizer_g, torch.optim.AdamW) or model.optimizer_g.state:
        raise RuntimeError("Full training requires fresh AdamW")
    if any(scheduler.last_epoch != 0 for scheduler in model.schedulers):
        raise RuntimeError("Full training requires fresh schedulers")
    if any(group["lr"] != 3e-4 for group in model.optimizer_g.param_groups):
        raise RuntimeError("Full training initial LR must be 3e-4")
    if any(value.requires_grad for value in model.net_g.encoder.parameters()) or not model.net_g.encoder.training:
        raise RuntimeError("Preserve frozen encoder parameters and training BN")
    if any(value.dtype != torch.float32 for value in model.net_g.parameters()):
        raise RuntimeError("Full training requires FP32 parameters")
    if model.net_g.pfa.rho != 0 or model.net_g.pfa.seed is not None or model.net_g.pfa.state_dict() is not None:
        raise RuntimeError("Full training must retain legacy global randn_like PFA")
    state_hash = digest({name: tensor_hash(value) for name, value in model.net_g.state_dict().items()})
    return {"fresh_adamw": True, "fresh_scheduler": True, "encoder_parameters_frozen": True,
            "encoder_bn_training": True, "fp32": True, "legacy_global_pfa": True,
            "initial_network_sha256": state_hash, "encoder_parameter_sha256": encoder_hash(model),
            "pretrained_sha256": opt["pretrained_sha256"], "seed": opt["seed"],
            "baseline_experiment": opt["full_training"]["baseline_experiment"]}


def validate_proof_row(row):
    return (row.get("dataset_replay_count") == 16 and all(row.get(flag) is True for flag in PROOF_FLAGS)
            and row.get("input_sha256") == row.get("post_input_sha256")
            and row.get("target_sha256") == row.get("post_target_sha256"))


def _restore_logs(directory, update):
    for name in ("train.jsonl", "input_protection.jsonl"):
        path = directory / name
        rows = durable_log_rows(path, update)
        if name == "input_protection.jsonl" and any(not validate_proof_row(row) for row in rows):
            raise RuntimeError("Committed input protection proof is invalid")
        # Preserve any interrupted tail, even though it was never checkpointed.
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        wanted = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        if original != wanted and original:
            archived = directory / "interrupted_tails"
            archived.mkdir(exist_ok=True)
            (archived / f"{time.time_ns()}_{name}").write_text(original, encoding="utf-8")
        path.write_text(wanted, encoding="utf-8")


def load_latest(directory, opt):
    record = read_json(directory / "checkpoints/latest.json")
    path = Path(record["path"])
    if path.resolve().parent != (directory / "checkpoints").resolve():
        raise RuntimeError("Checkpoint must belong to this run")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError("Checkpoint SHA256 mismatch")
    checkpoint = legacy.validate_checkpoint(path, opt, record["update"])
    if "pfa_rng" not in checkpoint or checkpoint["pfa_rng"] is not None:
        raise RuntimeError("Checkpoint must use legacy global PFA RNG")
    if not 1 <= checkpoint["update"] <= 300000:
        raise RuntimeError("Invalid full training checkpoint update")
    return checkpoint, record


def run(variant, stage, seed, paths, resume=False, stop_after=None, tag=None):
    cuda = paths["device"] == "cuda"
    if paths["device"] not in ("cpu", "cuda") or (cuda and not torch.cuda.is_available()):
        raise RuntimeError("Select CPU diagnostics or an available CUDA training device")
    configure_deterministic_execution(cuda)
    torch.set_num_threads(int(paths.get("cpu_threads", 1)))
    if cuda:
        torch.cuda.set_device(int(paths.get("gpu_index", 0)))
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
    opt, directory = configuration(variant, stage, seed, paths, tag)
    budget, end = 300000, 300000 if stop_after is None else stop_after
    if isinstance(end, bool) or not isinstance(end, int) or not 1 <= end <= budget:
        raise ValueError("--stop-after must be an update within the unchanged 300000-step budget")
    if directory.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {directory}; use --resume")
    if (directory / "input_failure").exists():
        raise RuntimeError("Input protection failed here; preserve evidence and use a new reviewed run")
    directory.mkdir(parents=True, exist_ok=True)
    with RunLock(directory / "process.lock"):
        config_file = directory / "effective_config.json"
        if config_file.exists() and read_json(config_file)["config_hash"] != opt["config_hash"]:
            raise RuntimeError("Source/config changed; use an independent output directory")
        atomic_json(config_file, opt)
        started_at, started = utc_now(), time.perf_counter()
        status = {"status": "initializing", "experiment": variant, "group": variant,
                  "stage": stage, "seed": seed, "config_hash": opt["config_hash"],
                  "source_hash": opt["source_hash"], "process": process_info(),
                  "started_at": started_at, "updated_at": started_at, "ended_at": None,
                  "update": 0, "budget": budget, "durable_update": 0, "checkpoint": None}
        atomic_json(directory / "status.json", status)
        current, elapsed, previous_elapsed = 0, 0.0, 0.0
        try:
            seed_all(seed)
            protection = Protection(directory, opt)
            model = build_protected_model(opt, protection)
            initialization = initialization_contract(model, opt)
            dataset = build_protected_dataset(opt, paths, protection)
            sampler = legacy.ExactSampler(len(dataset), seed + 20000)
            seed_all(seed + 10000)
            if (directory / "checkpoints/latest.json").exists():
                checkpoint, record = load_latest(directory, opt)
                legacy.restore_checkpoint(model, sampler, checkpoint)
                current, previous_elapsed = checkpoint["update"], checkpoint["elapsed_seconds"]
                status.update(durable_update=current, checkpoint=record["path"], checkpoint_sha256=record["sha256"])
                del checkpoint
            elif resume and (directory / "train.jsonl").exists():
                raise RuntimeError("No complete checkpoint exists for this prior run")
            if end < current:
                raise ValueError("--stop-after precedes the existing checkpoint")
            _restore_logs(directory, current)
            counts = {"total": sum(p.numel() for p in model.net_g.parameters()),
                      "trainable": sum(p.numel() for p in model.net_g.parameters() if p.requires_grad)}
            manifest_path = directory / "manifest.json"
            prior_manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if prior_manifest and prior_manifest.get("initialization_contract") != initialization:
                raise RuntimeError("Seeded initialization differs from the original invocation")
            manifest = {**opt["runtime"], "experiment": variant, "stage": stage, "seed": seed,
                        "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                        "git_commit": opt["git_commit"], "initialization_contract": initialization,
                        "pretrained_load": getattr(model.net_g.encoder, "pretrain_load_report", None),
                        "params": counts, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "gpu_name": torch.cuda.get_device_name() if cuda else None,
                        "command": sys.argv, "execution_contract": opt["execution_contract"],
                        "full_training": opt["full_training"]}
            atomic_json(manifest_path, manifest)
            status.update(status="running", update=current, updated_at=utc_now())
            atomic_json(directory / "status.json", status)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            training_started = time.perf_counter()
            elapsed = previous_elapsed
            with (directory / "train.jsonl").open("a", encoding="utf-8", buffering=1) as log, \
                    (directory / "input_protection.jsonl").open("a", encoding="utf-8", buffering=1) as proof:
                for current in range(current + 1, end + 1):
                    t0 = time.perf_counter()
                    loss, row = guarded_train_update(model, dataset, sampler, current, protection)
                    if cuda:
                        torch.cuda.synchronize()
                    elapsed = previous_elapsed + time.perf_counter() - training_started
                    item = {"update": current, "loss": loss, "lr": model.get_current_learning_rate()[0],
                            "seconds": time.perf_counter() - t0, "elapsed_seconds": elapsed,
                            "input_sha256": row["input_sha256"], "target_sha256": row["target_sha256"],
                            **{key: float(value) for key, value in model.get_current_log().items() if key != "l_pix"}}
                    proof.write(json.dumps(row, allow_nan=False) + "\n")
                    log.write(json.dumps(item, allow_nan=False) + "\n")
                    if current == 1 or current % opt["runtime"]["status_every"] == 0 or current == end:
                        status.update(**item, updated_at=utc_now(), invocation_wall_seconds=time.perf_counter() - started)
                        atomic_json(directory / "status.json", status)
                    if current == 1 or current % 100 == 0 or current == end:
                        print(f"{variant} full seed{seed} {current}/{budget} loss={loss:.6f} sec/update={item['seconds']:.3f}", flush=True)
                    if current % opt["runtime"]["checkpoint_every"] == 0 or current == end:
                        for stream in (log, proof):
                            stream.flush()
                            os.fsync(stream.fileno())
                        saved = legacy.save_checkpoint(model, sampler, opt, current, elapsed, directory)
                        status.update(checkpoint=str(saved), checkpoint_sha256=sha256_file(saved), durable_update=current)
                        atomic_json(directory / "status.json", status)
            # A resumed completed run is revalidated through the same proof/checkpoint path.
            if encoder_hash(model) != initialization["encoder_parameter_sha256"]:
                raise RuntimeError("Frozen encoder parameter changed")
            if any(parameter.grad is not None for parameter in model.net_g.encoder.parameters()):
                raise RuntimeError("Frozen encoder acquired gradients")
            _restore_logs(directory, current)
            proof_sha = sha256_file(directory / "input_protection.jsonl")
            state = "completed" if current == budget else "paused_at_checkpoint"
            checkpoint, record = load_latest(directory, opt)
            del checkpoint
            checks = {"initialization_contract": initialization, "encoder_parameters_unchanged": True,
                      "encoder_gradients_absent": True, "encoder_bn_training_after": model.net_g.encoder.training,
                      "input_protection_passed": True, "verified_updates": current,
                      "input_protection_sha256": proof_sha, "loss_finite": True, "update": current,
                      "complete": current == budget, "last_checkpoint_sha256_verified": True}
            atomic_json(directory / "training_checks.json", checks)
            atomic_json(directory / "input_protection.json", {"schema": SCHEMA,
                        "status": "passed" if current == budget else state,
                        "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                        "update": current, "verified_updates": current, "proof_sha256": proof_sha,
                        "all_checks_passed": True})
            if state == "completed":
                final = directory / "checkpoints/final.pth"
                if not final.exists():
                    os.link(record["path"], final)
                if sha256_file(final) != record["sha256"]:
                    raise RuntimeError("Final checkpoint differs from the validated latest checkpoint")
                manifest.update(final_checkpoint=str(final), final_checkpoint_sha256=record["sha256"])
            manifest.update(completed_optimizer_updates=current, total_training_seconds=elapsed,
                            input_protection_sha256=proof_sha)
            atomic_json(manifest_path, manifest)
            summary = {"status": state, "completed": current == budget, "updates": current,
                       "experiment": variant, "stage": stage, "seed": seed,
                       "update": current, "budget": budget, "config_hash": opt["config_hash"],
                       "source_hash": opt["source_hash"], "checkpoint_sha256": record["sha256"],
                       "input_protection_sha256": proof_sha, "elapsed_seconds": elapsed,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if cuda else None,
                       "training_checks": checks}
            atomic_json(directory / "summary.json", summary)
            status.update(status=state, update=current, durable_update=current, checkpoint=record["path"],
                          checkpoint_sha256=record["sha256"], ended_at=utc_now(), updated_at=utc_now(),
                          elapsed_seconds=elapsed, invocation_wall_seconds=time.perf_counter() - started)
            atomic_json(directory / "status.json", status)
            return directory
        except BaseException as error:
            status.update(status="failed", reason=f"{type(error).__name__}: {error}", update=current,
                          ended_at=utc_now(), updated_at=utc_now(), elapsed_seconds=elapsed)
            atomic_json(directory / "status.json", status)
            (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--stage", choices=["full"], default="full")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--paths", default="configs/plan_c_only_server.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--tag")
    args = parser.parse_args()
    run(args.variant, args.stage, args.seed, load_paths(args.paths), args.resume, args.stop_after, args.tag)


if __name__ == "__main__":
    main()
