"""Train the original ImageCleanModel with controlled stages and full recovery."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from torch.utils.data._utils.collate import default_collate
from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.models.image_restoration_model import ImageCleanModel
from tools.experiment_common import (
    ROOT, BUDGETS, MODES, RunLock, atomic_json, digest, git_commit, load_paths,
    process_info, read_json, restore_rng, rng_state, seed_all, sha256_file,
    source_hash, torch_load, utc_now)


class ExactSampler:
    """Persist permutation, cursor and independent random state; no prefetch."""
    def __init__(self, n, seed):
        self.n = n
        self.random = np.random.RandomState(seed)
        self.order = self.random.permutation(n).tolist()
        self.cursor = 0
        self.epoch = 0

    def take(self, batch):
        # Match official drop_last=True epoch boundaries.
        if self.cursor + batch > self.n:
            self.order = self.random.permutation(self.n).tolist()
            self.cursor = 0
            self.epoch += 1
        indices = self.order[self.cursor:self.cursor + batch]
        self.cursor += batch
        return indices

    def state_dict(self):
        return {"n": self.n, "random": self.random.get_state(),
                "order": self.order, "cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        assert state["n"] == self.n
        self.random.set_state(state["random"])
        self.order, self.cursor, self.epoch = state["order"], state["cursor"], state["epoch"]


def make_config(exp, stage, seed, paths, tag=None):
    with (ROOT / "Denoising/Options/CLIPDenoising_SyntheticDenoising_GaussianSigma15.yml").open() as f:
        opt = yaml.safe_load(f)
    run = Path(paths["output_root"]) / stage / exp / f"seed{seed}"
    if tag:
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
            raise ValueError("Run tag must use alphanumeric, dash or underscore.")
        run = run / tag
    opt.update(exp=exp, experiment=exp, stage=stage, seed=seed, train_seed=seed,
               name=f"{stage}_{exp}_seed{seed}", is_train=True, dist=False, rank=0,
               world_size=1, num_gpu=0 if paths["device"] == "cpu" else 1,
               manual_seed=seed)
    opt["network_g"].update(model_path=paths["pretrained_rn50"], adapter_mode=MODES[exp])
    opt["train"]["total_iter"] = BUDGETS[stage]
    opt["path"].update(experiments_root=str(run), models=str(run / "checkpoints"),
                       training_states=str(run / "checkpoints"), log=str(run),
                       visualization=str(run / "visuals"), resume_state=None)
    train = opt["datasets"]["train"]
    train.update(dataroot_gt=paths["train_dir"], phase="train", scale=1,
                 batch_size_per_gpu=paths["micro_batch"], num_worker_per_gpu=0,
                 mini_batch_sizes=[paths["micro_batch"]], iters=[BUDGETS[stage]],
                 cache_decoded_images=bool(paths.get("cache_decoded_images", False)))
    opt["datasets"]["val"].update(dataroot_gt=paths["train_dir"], sigma_test=15,
                                  phase="val", scale=1)
    opt["runtime"] = {
        "device": paths["device"], "micro_batch": int(paths["micro_batch"]),
        "accumulation": int(paths["accumulation"]), "amp": bool(paths.get("amp", False)),
        "noisy_views_per_update": 16, "clean_patches_per_update": 16,
        "train_image_count": 432 if stage == "full" else 400,
        "data_workers": 0, "sampler": "persisted permutation/cursor/RNG; drop_last; no prefetch",
        "cache_decoded_images": bool(paths.get("cache_decoded_images", False)),
        "cpu_threads": int(paths.get("cpu_threads", 4)),
        "independent_runs_per_gpu": int(paths.get("jobs_per_gpu", 1)) if stage == "full" else 1,
        "parallelism": "independent models/optimizers/RNG; no DDP or combined batch",
        "freeze_bn": False, "data_rng_seed": seed + 10000,
        "sampler_seed": seed + 20000,
        "minimum_free_gb": paths.get("minimum_free_gb", 2)}
    opt["val"]["val_freq"] = 0 if stage == "full" else BUDGETS[stage]
    opt["logger"]["save_checkpoint_freq"] = 5000 if stage != "smoke" else 50
    opt["data_manifest_hash"] = sha256_file(paths["data_manifest"])
    opt["dev_split_hash"] = sha256_file(paths["dev_train"])
    opt["pretrained_sha256"] = sha256_file(paths["pretrained_rn50"])
    opt["source_hash"] = source_hash()
    opt["git_commit"] = git_commit()
    opt["config_hash"] = digest(opt)
    return opt, run


def build_dataset(opt, paths):
    ds = Dataset_GaussianDenoising(copy.deepcopy(opt["datasets"]["train"]))
    if opt["stage"] != "full":
        entries = Path(paths["dev_train"]).read_text().splitlines()
        ds.paths = [str(Path(paths["project_root"]) / item) for item in entries if item.strip()]
    assert len(ds) == opt["runtime"]["train_image_count"], (len(ds), opt["runtime"])
    return ds


def build_model(opt):
    # The official class retains its optimizer, scheduler, L1 and Mixup code.
    return ImageCleanModel(copy.deepcopy(opt))


def train_update(model, ds, sampler, current, micro_batch, accumulation):
    model.update_learning_rate(current, warmup_iter=-1)
    if accumulation == 1:
        model.feed_train_data(default_collate([ds[i] for i in sampler.take(micro_batch)]))
        model.optimize_parameters(current)
        loss = float(model.get_current_log()["l_pix"])
    else:
        model.optimizer_g.zero_grad(set_to_none=True)
        loss = 0.0
        for _ in range(accumulation):
            model.feed_train_data(default_collate([ds[i] for i in sampler.take(micro_batch)]))
            pred = model.net_g(model.lq)
            value = model.cri_pix(pred, model.gt)
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite micro-batch loss")
            (value / accumulation).backward()
            loss += float(value.detach()) / accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.net_g.parameters(), 0.01,
                                             error_if_nonfinite=True)
        model.optimizer_g.step()
        if model.ema_decay > 0:
            model.model_ema(decay=model.ema_decay)
    if not math.isfinite(loss):
        raise FloatingPointError(f"Nonfinite loss at update {current}")
    return loss


def checkpoint_payload(model, sampler, opt, update, elapsed):
    return {
        "params": model.net_g.state_dict(), "optimizer": model.optimizer_g.state_dict(),
        "scheduler": [s.state_dict() for s in model.schedulers], "scaler": None,
        "ema": model.net_g_ema.state_dict() if hasattr(model, "net_g_ema") else None,
        "update": update, "rng": rng_state(), "sampler": sampler.state_dict(),
        "pfa_rng": model.net_g.pfa.state_dict(),
        "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
        "experiment": opt["exp"], "stage": opt["stage"], "seed": opt["seed"],
        "elapsed_seconds": elapsed, "complete_update_boundary": True}


def validate_checkpoint(path, opt, expected=None):
    ckpt = torch_load(path)
    for key in ("params", "optimizer", "scheduler", "rng", "sampler", "scaler", "ema"):
        if key not in ckpt:
            raise RuntimeError(f"Incomplete recovery state: missing {key}")
    for key in ("config_hash", "source_hash", "stage", "seed"):
        if ckpt[key] != opt[key]:
            raise RuntimeError(f"Checkpoint {key} does not match config")
    if ckpt["experiment"] != opt["exp"] or not ckpt.get("complete_update_boundary"):
        raise RuntimeError("Wrong experiment or incomplete optimizer update")
    if expected is not None and ckpt["update"] != expected:
        raise RuntimeError(f"Expected update {expected}, got {ckpt['update']}")
    return ckpt


def save_checkpoint(model, sampler, opt, update, elapsed, run):
    keep = opt["runtime"].get("checkpoint_keep", 2)
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("checkpoint_keep must be a positive integer")
    out = run / "checkpoints"
    out.mkdir(exist_ok=True)
    free = shutil.disk_usage(out).free
    if free < opt["runtime"]["minimum_free_gb"] * 1024**3:
        raise RuntimeError(f"Insufficient disk space for checkpoint: {free / 1024**3:.2f} GiB free")
    target = out / f"update_{update:06d}.pth"
    temporary = target.with_suffix(".pth.tmp")
    with temporary.open("wb") as f:
        torch.save(checkpoint_payload(model, sampler, opt, update, elapsed), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, target)
    validate_checkpoint(target, opt, update)
    atomic_json(out / "latest.json", {"path": str(target), "update": update,
                                    "sha256": sha256_file(target)})
    # Only this run's generated checkpoint files can be retired.
    old = sorted(out.glob("update_*.pth"))[:-keep]
    for p in old:
        p.unlink()
    return target


def restore_checkpoint(model, sampler, ckpt):
    model.net_g.load_state_dict(ckpt["params"], strict=True)
    model.net_g.pfa.load_state_dict(ckpt.get("pfa_rng"))
    model.optimizer_g.load_state_dict(ckpt["optimizer"])
    if len(model.schedulers) != len(ckpt["scheduler"]):
        raise RuntimeError("Scheduler count mismatch")
    for scheduler, state in zip(model.schedulers, ckpt["scheduler"]):
        scheduler.load_state_dict(state)
    if ckpt["ema"] is not None:
        model.net_g_ema.load_state_dict(ckpt["ema"], strict=True)
    sampler.load_state_dict(ckpt["sampler"])
    restore_rng(ckpt["rng"])


def run_experiment(exp, stage, seed, paths, resume=False, tag=None):
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    if paths["device"] != "cpu" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(paths.get("gpu_index", 0))
    if paths["device"] != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; full queue must run on the authorized GPU server")
    if paths["micro_batch"] * paths["accumulation"] != 16:
        raise ValueError("Exactly 16 noisy views per optimizer update required")
    if paths.get("amp"):
        raise ValueError("This baseline implementation is FP32; AMP must be validated separately")
    opt, run = make_config(exp, stage, seed, paths, tag)
    if run.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {run}; use --resume")
    run.mkdir(parents=True, exist_ok=True)
    with RunLock(run / "process.lock"):
        config_file = run / "effective_config.json"
        if config_file.exists():
            prior = read_json(config_file)
            if prior["config_hash"] != opt["config_hash"]:
                raise RuntimeError("Existing run config/source differs; refusing to resume")
        else:
            atomic_json(config_file, opt)
        status_path = run / "status.json"
        info = {"status": "running", "experiment": exp, "stage": stage, "seed": seed,
                "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                "process": process_info(), "updated_at": utc_now(), "update": 0,
                "budget": BUDGETS[stage], "checkpoint": None,
                "log": str(run / "train.jsonl")}
        final = run / "checkpoints/final.pth"
        if final.exists():
            ckpt = validate_checkpoint(final, opt, BUDGETS[stage])
            info.update(status="completed", update=ckpt["update"], checkpoint=str(final))
            atomic_json(status_path, info)
            return run
        atomic_json(status_path, info)
        try:
            logging.getLogger("basicsr").setLevel(logging.ERROR)
            seed_all(seed)
            model = build_model(opt)
            ds = build_dataset(opt, paths)
            sampler = ExactSampler(len(ds), seed + 20000)
            seed_all(seed + 10000)
            initial_encoder = {k: p.detach().cpu().clone()
                               for k, p in model.net_g.encoder.named_parameters()}
            bn_before = model.net_g.encoder.bn1.running_mean.detach().cpu().clone()
            current, previous_elapsed = 0, 0.0
            latest = run / "checkpoints/latest.json"
            if resume and latest.exists():
                record = read_json(latest)
                if sha256_file(record["path"]) != record["sha256"]:
                    raise RuntimeError("Recovery checkpoint digest mismatch")
                ckpt = validate_checkpoint(record["path"], opt, record["update"])
                restore_checkpoint(model, sampler, ckpt)
                current, previous_elapsed = ckpt["update"], ckpt["elapsed_seconds"]
                saved = Path(record["path"])
                del ckpt
                info["recovery"] = "complete state restored"
            elif resume and (run / "train.jsonl").exists():
                raise RuntimeError("Incomplete prior run has no full checkpoint; explicit new tag required for restart")
            counts = {"total": sum(p.numel() for p in model.net_g.parameters()),
                      "trainable": sum(p.numel() for p in model.net_g.parameters() if p.requires_grad)}
            atomic_json(run / "manifest.json", {
                **opt["runtime"], "experiment": exp, "stage": stage, "seed": seed,
                "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                "git_commit": opt["git_commit"], "params": counts,
                "pretrained_load": model.net_g.encoder.pretrain_load_report,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(0) if paths["device"] != "cpu" else None,
                "python": sys.version, "command": sys.argv,
                "comparison_note": "Official batch 16" if paths["accumulation"] == 1
                    else "Micro-batch BN/Mixup differs from true batch16; common across all groups"})
            started = time.perf_counter()
            elapsed = previous_elapsed
            if torch.cuda.is_available() and paths["device"] != "cpu":
                torch.cuda.reset_peak_memory_stats()
            with (run / "train.jsonl").open("a", encoding="utf-8", buffering=1) as log:
                for update in range(current + 1, BUDGETS[stage] + 1):
                    if paths["device"] != "cpu":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    loss = train_update(model, ds, sampler, update,
                                        paths["micro_batch"], paths["accumulation"])
                    if paths["device"] != "cpu":
                        torch.cuda.synchronize()
                    elapsed = previous_elapsed + time.perf_counter() - started
                    item = {"update": update, "loss": loss, "lr": model.get_current_learning_rate()[0],
                            "seconds": time.perf_counter() - t0, "elapsed_seconds": elapsed,
                            "peak_memory_mb": torch.cuda.max_memory_allocated()/1024**2
                            if paths["device"] != "cpu" else None}
                    log.write(json.dumps(item, allow_nan=False) + "\n")
                    if update % 10 == 0 or update == 1:
                        info.update(updated_at=utc_now(), **item)
                        atomic_json(status_path, info)
                    if update % 100 == 0 or update == 1:
                        print(f"{exp} {stage} seed{seed} update {update}/{BUDGETS[stage]} "
                              f"loss={loss:.6f} sec/update={item['seconds']:.3f}", flush=True)
                    if update % opt["logger"]["save_checkpoint_freq"] == 0 or update == BUDGETS[stage]:
                        saved = save_checkpoint(model, sampler, opt, update, elapsed, run)
                        info["checkpoint"] = str(saved)
                        atomic_json(status_path, info)
            for k, p in model.net_g.encoder.named_parameters():
                if not torch.equal(initial_encoder[k], p.detach().cpu()):
                    raise AssertionError(f"Frozen encoder parameter changed: {k}")
            atomic_json(run / "training_checks.json", {
                "loss_finite": True, "frozen_encoder_unchanged": True,
                "encoder_bn_training": model.net_g.encoder.bn1.training,
                "bn_running_mean_changed": not torch.equal(bn_before, model.net_g.encoder.bn1.running_mean.cpu()),
                "update": BUDGETS[stage], "parameter_counts": counts})
            os.link(saved, final)
            validate_checkpoint(final, opt, BUDGETS[stage])
            manifest = read_json(run / "manifest.json")
            manifest.update(completed_optimizer_updates=BUDGETS[stage],
                            total_training_seconds=elapsed,
                            final_checkpoint=str(final),
                            final_checkpoint_sha256=sha256_file(final))
            atomic_json(run / "manifest.json", manifest)
            info.update(status="completed", update=BUDGETS[stage], checkpoint=str(final),
                        updated_at=utc_now(), elapsed_seconds=elapsed)
            atomic_json(status_path, info)
            return run
        except BaseException as e:
            info.update(status="blocked", reason=f"{type(e).__name__}: {e}", updated_at=utc_now())
            atomic_json(status_path, info)
            (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=list(MODES), required=True)
    p.add_argument("--stage", choices=list(BUDGETS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--paths", default="configs/local_paths.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tag")
    a = p.parse_args()
    run_experiment(a.exp, a.stage, a.seed, load_paths(a.paths), a.resume, a.tag)


if __name__ == "__main__":
    main()
