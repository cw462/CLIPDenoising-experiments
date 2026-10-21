"""Single-GPU paired B0/B1 continuation for proposal A (not RGB calibration G1)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.utils import get_root_logger
from tools.experiment_common import (
    RunLock, atomic_json, digest, load_paths, read_json, seed_all, sha256_file, torch_load,
)
from tools.run_experiment import (
    ExactSampler, build_model, make_config, restore_checkpoint, save_checkpoint, train_update,
    validate_checkpoint,
)


def configuration(group, stage, seed, paths):
    budget = int(paths["budgets"][stage])
    if budget < 2 or not 1 <= paths["micro_batch"] <= 432 or paths["accumulation"] != 1:
        raise ValueError("Use budget >= 2 and a real batch of 1..432; accumulation must be 1")
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
                          independent_pfa_seed=seed + 30000)
    opt["plan_a"] = {"c0_sha256": sha256_file(c0), "rho": opt["network_g"]["pfa_rho"],
                     "levels": [3, 4], "kernel": "3x3 mean / kernel L2, circular random-field padding",
                     "training_noise": "Gaussian sigma15 only", "new_optimizer": True}
    opt.pop("config_hash", None)
    opt["config_hash"] = digest(opt)
    return opt, run


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(group, stage, seed, paths, resume=False, stop_after=None):
    if paths["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    cuda = paths["device"] == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: connect the RTX 3070 or select a CPU config")
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    if cuda:
        torch.cuda.set_device(int(paths.get("gpu_index", 0)))
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
    opt, directory = configuration(group, stage, seed, paths)
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
        model = build_model(opt)
        if any(parameter.requires_grad for parameter in model.net_g.encoder.parameters()):
            raise RuntimeError("Encoder must remain frozen")
        dataset = Dataset_GaussianDenoising(copy.deepcopy(opt["datasets"]["train"]))
        if len(dataset) != 432:
            raise RuntimeError(f"Expected all 432 training images, found {len(dataset)}")
        sampler = ExactSampler(len(dataset), seed + 20000)
        # Initialization/loading cannot perturb paired training noise or Mixup RNG.
        seed_all(seed + 10000)
        update, elapsed = 0, 0.0
        latest = directory / "checkpoints/latest.json"
        if latest.exists():
            checkpoint = validate_checkpoint(read_json(latest)["path"], opt)
            restore_checkpoint(model, sampler, checkpoint)
            update, elapsed = checkpoint["update"], checkpoint["elapsed_seconds"]
            del checkpoint
        elif (directory / "train.jsonl").exists():
            raise RuntimeError("Interrupted before first checkpoint; preserve this attempt and use a new output directory")
        if update >= budget:
            print(f"Already completed: {directory}")
            return
        if stop_after is not None and stop_after <= update:
            raise ValueError("--stop-after must exceed the resumed update")
        # Remove only this run's log tail beyond the last durable checkpoint.
        log_path = directory / "train.jsonl"
        rows = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
        rows = [row for row in rows if row["update"] <= update]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        stop = budget if stop_after is None else stop_after
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
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
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    if current % 20 == 0 or current == stop:
                        print(f"{group} {stage} {current}/{budget}: L1={loss:.6f}, {seconds:.3f}s/update", flush=True)
                    if current % 100 == 0 or current == stop:
                        saved = save_checkpoint(model, sampler, opt, current, elapsed, directory)
                        atomic_json(directory / "status.json", {
                            "status": "completed" if current == budget else "paused_at_checkpoint",
                            "update": current, "budget": budget, "checkpoint": str(saved),
                            "elapsed_seconds": elapsed,
                        })
            timings = [row["seconds"] for row in rows if row["update"] > 10]
            atomic_json(directory / "summary.json", {
                "group": group, "stage": stage, "seed": seed, "updates": stop,
                "budget": budget, "completed": stop == budget, "batch_size": paths["micro_batch"],
                "device": torch.cuda.get_device_name() if cuda else "cpu", "rho": opt["network_g"]["pfa_rho"],
                "c0_sha256": opt["plan_a"]["c0_sha256"], "last_loss": rows[-1]["loss"],
                "first_lr": rows[0]["lr"], "last_lr": rows[-1]["lr"],
                "mean_update_seconds_after_warmup": sum(timings) / len(timings) if timings else None,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
                "training_result_claim": "Runtime/stability verification only" if stage == "smoke" else "Requires paired evaluation",
            })
        except Exception as error:
            atomic_json(directory / "status.json", {"status": "failed", "reason": str(error),
                                                    "last_durable_checkpoint": read_json(latest) if latest.exists() else None})
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
