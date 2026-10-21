"""Single-GPU paired B0/B1 continuation for proposal A (not RGB calibration G1)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.utils import get_root_logger
from tools.experiment_common import (
    RunLock, atomic_json, digest, load_paths, process_info, read_json, seed_all,
    sha256_file, torch_load, utc_now,
)
from tools.run_experiment import (
    ExactSampler, build_model, make_config, restore_checkpoint, save_checkpoint, train_update,
    validate_checkpoint,
)


def configuration(group, stage, seed, paths):
    if group not in ("B0", "B1"):
        raise ValueError("Proposal A supports B0 and B1 only")
    budget = int(paths["budgets"][stage])
    if budget < 2 or not 1 <= paths["micro_batch"] <= 432 or paths["accumulation"] != 1:
        raise ValueError("Use budget >= 2 and a real batch of 1..432; accumulation must be 1")
    if paths.get("amp", False):
        raise ValueError("Paired A/C continuation requires FP32 (amp=false)")
    if not (math.isfinite(paths["learning_rate"]) and math.isfinite(paths["minimum_learning_rate"])
            and 0 < paths["minimum_learning_rate"] <= paths["learning_rate"]):
        raise ValueError("Require finite 0 < minimum_learning_rate <= learning_rate")
    checkpoint_every, status_every = (int(paths.get("checkpoint_every", 100)),
                                       int(paths.get("status_every", 20)))
    if checkpoint_every < 1 or status_every < 1:
        raise ValueError("checkpoint_every and status_every must be positive")
    checkpoint_keep = paths.get("checkpoint_keep", 2)
    if isinstance(checkpoint_keep, bool) or not isinstance(checkpoint_keep, int) or checkpoint_keep < 1:
        raise ValueError("checkpoint_keep must be a positive integer")
    c0 = Path(paths["c0"])
    if not c0.is_absolute():
        c0 = ROOT / c0
    opt, _ = make_config("B0", "full", seed, paths)
    run = Path(paths["output_root"]) / stage / group / f"seed{seed}"
    opt.update(exp=group, experiment=group, stage=stage, name=f"plan_a_{stage}_{group}_seed{seed}")
    opt["network_g"].update(adapter_mode="none", pfa_rho=paths["rho"] if group == "B1" else 0.0,
                            pfa_seed=seed + 30000)
    opt["path"].update(pretrain_network_g=str(c0), param_key="params", strict_load_g=True,
                       experiments_root=str(run), models=str(run / "checkpoints"),
                       training_states=str(run / "checkpoints"), log=str(run), visualization=str(run / "visuals"))
    opt["datasets"]["train"].update(batch_size_per_gpu=paths["micro_batch"],
                                  mini_batch_sizes=[paths["micro_batch"]], iters=[budget])
    opt["train"].update(total_iter=budget,
                        scheduler={"type": "CosineAnnealingRestartCyclicLR", "periods": [budget - 1],
                                   "restart_weights": [1], "eta_mins": [paths["minimum_learning_rate"]]})
    opt["train"]["optim_g"].update(lr=paths["learning_rate"], weight_decay=paths["weight_decay"])
    opt["runtime"].update(noisy_views_per_update=paths["micro_batch"],
                          clean_patches_per_update=paths["micro_batch"], train_image_count=432,
                          independent_pfa_seed=seed + 30000, checkpoint_every=checkpoint_every,
                          checkpoint_keep=checkpoint_keep, status_every=status_every)
    opt["logger"]["save_checkpoint_freq"] = checkpoint_every
    opt["logger"]["print_freq"] = status_every
    opt["plan_a"] = {"c0_sha256": sha256_file(c0), "rho": opt["network_g"]["pfa_rho"],
                     "levels": [3, 4], "kernel": "3x3 mean / kernel L2, circular random-field padding",
                     "training_noise": "Gaussian sigma15 only", "new_optimizer": True}
    if paths.get("c0_expected_sha256") and opt["plan_a"]["c0_sha256"] != paths["c0_expected_sha256"]:
        raise RuntimeError("C0 SHA256 differs from the pinned common starting checkpoint")
    opt["continuation"] = {"c0_expected": dict(paths.get("c0_expected", {})),
                            "optimizer_state_from_c0": False,
                            "schedule": "one cosine cycle; update 1=start LR, budget=end LR"}
    opt["paired_protocol"] = paired_protocol(opt)
    opt["paired_protocol_hash"] = digest(opt["paired_protocol"])
    opt.pop("config_hash", None)
    opt["config_hash"] = digest(opt)
    return opt, run


def paired_protocol(opt):
    """Shared scientific controls, excluding the intended PFA/loss intervention."""
    train, network = copy.deepcopy(opt["train"]), copy.deepcopy(opt["network_g"])
    train.pop("pixel_opt", None)
    network.pop("pfa_rho", None)
    return {"schema": "ac_paired_continuation_v1", "stage": opt["stage"], "seed": opt["seed"],
            "c0_sha256": opt.get("plan_c", opt.get("plan_a"))["c0_sha256"],
            "pretrained_sha256": opt["pretrained_sha256"],
            "data_manifest_hash": opt["data_manifest_hash"], "source_hash": opt["source_hash"],
            "train": train, "network_except_pfa_rho": network,
            "dataset_train": copy.deepcopy(opt["datasets"]["train"]),
            "runtime": {key: opt["runtime"][key] for key in
                        ("micro_batch", "accumulation", "amp", "train_image_count", "freeze_bn",
                         "data_rng_seed", "sampler_seed", "independent_pfa_seed")}}


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def encoder_hash(model):
    return digest({name: tensor_hash(value) for name, value in model.net_g.encoder.named_parameters()})


def initialization_contract(model, opt):
    """Check the *loaded* C0 tensors, then require a completely fresh optimizer."""
    checkpoint = torch_load(opt["path"]["pretrain_network_g"])
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("params"), dict):
        raise RuntimeError("C0 must contain a complete params state dictionary")
    for key, expected in opt["continuation"]["c0_expected"].items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"C0 provenance mismatch for {key}: {checkpoint.get(key)!r} != {expected!r}")
    actual = model.net_g.state_dict()
    original = checkpoint["params"]
    if set(actual) != set(original) or any(
            actual[key].dtype != original[key].dtype
            or not torch.equal(actual[key].detach().cpu(), original[key].detach().cpu())
            for key in actual):
        raise RuntimeError("Initial network tensors do not exactly match C0 params")
    if not isinstance(model.optimizer_g, torch.optim.AdamW) or model.optimizer_g.state:
        raise RuntimeError("Continuation must start with a fresh AdamW optimizer")
    if any(scheduler.last_epoch != 0 for scheduler in model.schedulers):
        raise RuntimeError("Continuation scheduler must start at epoch zero")
    if any(group["lr"] != opt["train"]["optim_g"]["lr"] for group in model.optimizer_g.param_groups):
        raise RuntimeError("Initial optimizer LR differs from the continuation schedule")
    if any(p.requires_grad for p in model.net_g.encoder.parameters()):
        raise RuntimeError("Encoder parameters must remain frozen")
    if not model.net_g.encoder.training:
        raise RuntimeError("Preserve the original encoder BN training behavior")
    if any(p.dtype != torch.float32 for p in model.net_g.parameters()):
        raise RuntimeError("Paired continuation requires FP32 parameters")
    return {"strict_c0_params": True, "fresh_adamw": True, "fresh_scheduler": True,
            "encoder_parameters_frozen": True, "encoder_bn_training": True, "fp32": True,
            "encoder_parameter_sha256": encoder_hash(model),
            "c0_metadata": {key: checkpoint.get(key) for key in
                            ("experiment", "stage", "seed", "update", "source_hash", "config_hash")}}


def load_latest_checkpoint(latest, opt):
    reference = read_json(latest)
    checkpoint_path = Path(reference["path"])
    if checkpoint_path.resolve().parent != Path(latest).resolve().parent:
        raise RuntimeError("Latest checkpoint must belong to this run")
    if sha256_file(checkpoint_path) != reference["sha256"]:
        raise RuntimeError("Latest checkpoint SHA256 mismatch")
    checkpoint = validate_checkpoint(checkpoint_path, opt, expected=reference["update"])
    state = checkpoint.get("pfa_rng")
    if (not isinstance(state, dict) or state.get("rho") != opt["network_g"]["pfa_rho"]
            or state.get("seed") != opt["network_g"]["pfa_seed"] or not state.get("generators")):
        raise RuntimeError("Missing or mismatched independent PFA RNG checkpoint state")
    if not 1 <= checkpoint["update"] <= opt["train"]["total_iter"]:
        raise RuntimeError("Invalid checkpoint update")
    return checkpoint


def durable_log_rows(path, update):
    """Drop an interrupted tail, but never silently discard a committed update."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise RuntimeError("Malformed training log before its final line")
        if row["update"] <= update:
            rows.append(row)
    if [row["update"] for row in rows] != list(range(1, update + 1)):
        raise RuntimeError("Training log is missing or duplicating a durable update")
    return rows


def run(group, stage, seed, paths, resume=False, stop_after=None, configuration_builder=None):
    if paths["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    cuda = paths["device"] == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run on the authorized GPU server or select a CPU config")
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    if cuda:
        torch.cuda.set_device(int(paths.get("gpu_index", 0)))
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
    opt, directory = (configuration_builder or configuration)(group, stage, seed, paths)
    budget = opt["train"]["total_iter"]
    if stop_after is not None and not 1 <= stop_after <= budget:
        raise ValueError("--stop-after must be between 1 and the stage budget")
    if directory.exists() and not resume:
        raise FileExistsError(f"Run exists: {directory}; use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    with RunLock(directory / "process.lock"):
        config_file = directory / "effective_config.json"
        if config_file.exists() and read_json(config_file)["config_hash"] != opt["config_hash"]:
            raise RuntimeError("Source/config/C0 changed; use a new output directory")
        atomic_json(config_file, opt)
        started_at, wall_started = utc_now(), time.perf_counter()
        info = {"status": "initializing", "group": group, "experiment": group,
                "stage": stage, "seed": seed, "process": process_info(),
                "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                "started_at": started_at, "updated_at": started_at, "ended_at": None,
                "update": 0, "budget": budget, "checkpoint": None, "durable_update": 0,
                "elapsed_seconds": 0.0, "invocation_wall_seconds": 0.0}
        status_path = directory / "status.json"
        atomic_json(status_path, info)
        update, elapsed = 0, 0.0
        latest = directory / "checkpoints/latest.json"
        log_path = directory / "train.jsonl"
        try:
            seed_all(seed)
            model = build_model(opt)
            initialization = initialization_contract(model, opt)
            dataset = Dataset_GaussianDenoising(copy.deepcopy(opt["datasets"]["train"]))
            if len(dataset) != 432:
                raise RuntimeError(f"Expected all 432 training images, found {len(dataset)}")
            sampler = ExactSampler(len(dataset), seed + 20000)
            # Initialization/loading cannot perturb paired training noise or Mixup RNG.
            seed_all(seed + 10000)
            if latest.exists():
                checkpoint = load_latest_checkpoint(latest, opt)
                restore_checkpoint(model, sampler, checkpoint)
                update, elapsed = checkpoint["update"], checkpoint["elapsed_seconds"]
                del checkpoint
                reference = read_json(latest)
                info.update(checkpoint=reference["path"], checkpoint_sha256=reference["sha256"],
                            durable_update=update)
            elif log_path.exists():
                raise RuntimeError("Interrupted before first checkpoint; preserve this attempt and use a new output directory")
            info.update(update=update, elapsed_seconds=elapsed, status="running")
            atomic_json(status_path, info)
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                atomic_json(manifest_path, {
                    "group": group, "stage": stage, "seed": seed, "budget": budget,
                    "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                    "paired_protocol_hash": opt["paired_protocol_hash"],
                    "git_commit": opt.get("git_commit"), "started_at": started_at,
                    "c0_path": opt["path"]["pretrain_network_g"],
                    "c0_sha256": opt.get("plan_c", opt.get("plan_a"))["c0_sha256"],
                    "initialization_checks": initialization,
                    "paired_data_rng_seed": seed + 10000, "sampler_seed": seed + 20000,
                    "independent_pfa_seed": seed + 30000,
                    "checkpoint_every": opt["runtime"]["checkpoint_every"],
                })
            # A validated durable checkpoint is also required for a completed rerun.
            rows = durable_log_rows(log_path, update)
            if update < budget and stop_after is not None and stop_after <= update:
                raise ValueError("--stop-after must exceed the resumed update")
            # Remove only this run's uncommitted log tail beyond its durable checkpoint.
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            stop = budget if update >= budget or stop_after is None else stop_after
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            with log_path.open("a", encoding="utf-8", buffering=1) as stream:
                for current in range(update + 1, stop + 1):
                    if cuda:
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    loss = train_update(model, dataset, sampler, current, paths["micro_batch"], 1)
                    if cuda:
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    elapsed += seconds
                    row = {"update": current, "loss": loss, "lr": model.optimizer_g.param_groups[0]["lr"],
                           "seconds": seconds, "input_sha256": tensor_hash(model.lq),
                           "target_sha256": tensor_hash(model.gt)}
                    components = getattr(model.cri_pix, 'last_components', None)
                    row.update({key: float(value) for key, value in components.items()} if components else
                               {'pixel_l1': loss, 'structure': 0.0, 'weighted_structure': 0.0})
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    if current % opt["runtime"]["status_every"] == 0 or current == stop:
                        print(f"{group} {stage} {current}/{budget}: loss={loss:.6f}, {seconds:.3f}s/update", flush=True)
                        info.update(update=current, elapsed_seconds=elapsed, updated_at=utc_now(),
                                    invocation_wall_seconds=time.perf_counter() - wall_started,
                                    last_loss=loss, last_lr=row["lr"])
                        atomic_json(status_path, info)
                    if current % opt["runtime"]["checkpoint_every"] == 0 or current == stop:
                        saved = save_checkpoint(model, sampler, opt, current, elapsed, directory)
                        info.update(status="finalizing" if current == stop else "running",
                                    update=current, durable_update=current, checkpoint=str(saved),
                                    checkpoint_sha256=read_json(latest)["sha256"], elapsed_seconds=elapsed,
                                    updated_at=utc_now(), invocation_wall_seconds=time.perf_counter() - wall_started,
                                    ended_at=utc_now() if current == stop else None)
                        atomic_json(status_path, info)
            frozen = encoder_hash(model) == initialization["encoder_parameter_sha256"]
            if not frozen:
                raise RuntimeError("Frozen encoder parameters changed during continuation")
            atomic_json(directory / "training_checks.json", {
                **initialization, "encoder_parameters_unchanged": frozen,
                "encoder_bn_training_after": model.net_g.encoder.training,
                "encoder_gradients_absent": all(p.grad is None for p in model.net_g.encoder.parameters()),
                "last_checkpoint_sha256_verified": sha256_file(info["checkpoint"]) == info["checkpoint_sha256"],
                "update": stop, "complete": stop == budget, "checked_at": utc_now(),
            })
            timings = [row["seconds"] for row in rows if row["update"] > 10]
            atomic_json(directory / "summary.json", {
                "group": group, "stage": stage, "seed": seed, "updates": stop,
                "budget": budget, "completed": stop == budget, "batch_size": paths["micro_batch"],
                "device": torch.cuda.get_device_name() if cuda else "cpu", "rho": opt["network_g"]["pfa_rho"],
                "c0_sha256": opt.get("plan_c", opt.get("plan_a"))["c0_sha256"], "last_loss": rows[-1]["loss"],
                "last_pixel_l1": rows[-1]['pixel_l1'], "last_structure": rows[-1]['structure'],
                "last_weighted_structure": rows[-1]['weighted_structure'],
                "first_lr": rows[0]["lr"], "last_lr": rows[-1]["lr"],
                "elapsed_seconds": elapsed, "invocation_started_at": started_at,
                "ended_at": utc_now(), "invocation_wall_seconds": time.perf_counter() - wall_started,
                "checkpoint": info["checkpoint"], "checkpoint_sha256": info["checkpoint_sha256"],
                "mean_update_seconds_after_warmup": sum(timings) / len(timings) if timings else None,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
                "training_result_claim": "Runtime/stability verification only" if stage == "smoke" else "Requires paired evaluation",
            })
            info.update(status="completed" if stop == budget else "paused_at_checkpoint",
                        updated_at=utc_now(), ended_at=utc_now(),
                        invocation_wall_seconds=time.perf_counter() - wall_started)
            atomic_json(status_path, info)
            return directory
        except BaseException as error:
            info.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        reason=str(error), updated_at=utc_now(), ended_at=utc_now(),
                        invocation_wall_seconds=time.perf_counter() - wall_started,
                        last_durable_checkpoint=read_json(latest) if latest.exists() else None)
            atomic_json(status_path, info)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("B0", "B1"), required=True)
    parser.add_argument("--stage", choices=("smoke", "screen", "confirm", "confirm_short"), default="smoke")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--paths", default="configs/plan_a.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Stop at this update without shortening the LR schedule")
    args = parser.parse_args()
    run(args.group, args.stage, args.seed, load_paths(args.paths), args.resume, args.stop_after)


if __name__ == "__main__":
    main()
