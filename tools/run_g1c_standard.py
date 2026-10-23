"""User-requested standard G1C training, without per-update input guards.

This uses the original single-pass data/Mixup/optimizer path. It does not
establish input consistency or resolve the previously observed host anomaly.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools import run_experiment as legacy
from tools.experiment_common import (
    RunLock, atomic_json, digest, load_paths, process_info, read_json,
    seed_all, sha256_file, utc_now,
)

INTEGRITY_MODE = "standard_user_requested_v1"
TOTAL_UPDATES = 300000


def configure_execution():
    """Retain the execution flags used by the original training entrypoint."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def configuration(seed, paths, tag=None):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (0, 1, 2):
        raise ValueError("G1C requires one of the original seeds 0, 1, 2")
    if paths.get("training_integrity_mode") != INTEGRITY_MODE:
        raise ValueError("Explicit training_integrity_mode=standard_user_requested_v1 is required")
    if paths.get("micro_batch") != 16 or paths.get("accumulation") != 1 or paths.get("amp", False):
        raise ValueError("Standard G1C requires true batch 16, accumulation 1 and FP32")
    if paths.get("formal_updates_per_run", TOTAL_UPDATES) != TOTAL_UPDATES:
        raise ValueError("The full training schedule is fixed at 300000 updates")
    if seed not in paths.get("formal_seeds", (0, 1, 2)):
        raise ValueError("Seed is outside the configured training scope")
    opt, _ = legacy.make_config("G1", "full", seed, paths, tag)
    directory = Path(paths["output_root"]) / "full" / "G1C" / f"seed{seed}"
    if tag:
        directory /= tag  # make_config has already validated the tag.
    opt.update(exp="G1C", experiment="G1C", name=f"full_G1C_seed{seed}",
               training_integrity_mode=INTEGRITY_MODE,
               per_update_input_consistency_checks=False)
    opt["network_g"].update(pfa_rho=0.0, pfa_seed=None)
    opt["path"].update(pretrain_network_g=None, experiments_root=str(directory),
                       models=str(directory / "checkpoints"),
                       training_states=str(directory / "checkpoints"),
                       log=str(directory), visualization=str(directory / "visuals"))
    opt["train"]["pixel_opt"] = {
        "type": "L1StructureLoss", "loss_weight": 1.0, "reduction": "mean",
        "structure_weight": 0.05, "eps": 0.001, "deterministic_reflect_padding": True,
    }
    opt["runtime"].update(integrity_mode=INTEGRITY_MODE,
                          per_update_input_consistency_checks=False)
    for field, default in (("checkpoint_every", 5000), ("checkpoint_keep", 2), ("status_every", 100)):
        value = paths.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
        opt["runtime"][field] = value
    opt["logger"]["save_checkpoint_freq"] = opt["runtime"]["checkpoint_every"]
    opt["full_training"] = {
        "schema": "g1c_standard_full_v1", "variant": "G1C",
        "baseline_experiment": "G1", "baseline_training_seed": seed,
        "seed": seed, "budget": TOTAL_UPDATES, "from_c0": False,
        "initialization": "original RN50 and seed-matched G1 decoder/adapter initialization",
        "optimizer": "fresh original AdamW", "legacy_global_pfa": True,
        "integrity_mode": INTEGRITY_MODE, "per_update_input_consistency_checks": False,
    }
    opt["execution_contract"] = {
        "schema": "g1c_standard_execution_v1", "integrity_mode": INTEGRITY_MODE,
        "per_update_input_consistency_checks": False,
        "deterministic_algorithms": False, "cudnn_deterministic": False,
        "cudnn_benchmark": False, "cudnn_allow_tf32": True,
        "matmul_allow_tf32": False, "parameter_dtype": "float32",
        "input_path": "original single-pass dataset, Mixup and optimizer; no replay or byte comparisons",
        "limitation": "Input consistency is not checked; the observed host anomaly is not resolved by this mode.",
    }
    opt.pop("config_hash", None)
    opt["config_hash"] = digest(opt)
    return opt, directory


def _restore_log(directory, committed_update):
    """Keep committed standard training rows; archive any uncommitted tail."""
    path = directory / "train.jsonl"
    if not path.exists():
        if committed_update:
            raise RuntimeError("Committed checkpoint has no training log")
        return
    original = path.read_bytes()
    rows = original.splitlines(keepends=True)
    if len(rows) < committed_update:
        raise RuntimeError("Training log is shorter than the committed checkpoint")
    for expected, line in enumerate(rows[:committed_update], 1):
        if json.loads(line)["update"] != expected:
            raise RuntimeError("Committed training log has missing or reordered updates")
    retained = b"".join(rows[:committed_update])
    if retained != original:
        archive = directory / "resume_archive"
        archive.mkdir(exist_ok=True)
        (archive / f"{time.time_ns()}_train.jsonl").write_bytes(original)
        temporary = path.with_suffix(".jsonl.resume.tmp")
        with temporary.open("wb") as handle:
            handle.write(retained)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def run(seed, paths, resume=False, stop_after=None, tag=None):
    opt, directory = configuration(seed, paths, tag)
    stop = TOTAL_UPDATES if stop_after is None else stop_after
    if isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= stop <= TOTAL_UPDATES:
        raise ValueError("stop_after must be an absolute update in [1, 300000]")
    configure_execution()
    torch.set_num_threads(opt["runtime"]["cpu_threads"])
    cuda = paths["device"] != "cpu"
    if cuda:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(paths.get("gpu_index", 0)))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")
    if directory.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {directory}; use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    with RunLock(directory / "process.lock"):
        config_file = directory / "effective_config.json"
        if config_file.exists():
            if read_json(config_file)["config_hash"] != opt["config_hash"]:
                raise RuntimeError("Existing run config/source differs; refusing to resume")
        else:
            atomic_json(config_file, opt)
        common = {"experiment": "G1C", "stage": "full", "seed": seed,
                  "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                  "training_integrity_mode": INTEGRITY_MODE, "integrity_mode": INTEGRITY_MODE,
                  "per_update_input_consistency_checks": False}
        info = dict(common, status="running", process=process_info(), started_at=utc_now(),
                    updated_at=utc_now(), update=0, budget=TOTAL_UPDATES,
                    checkpoint=None, checkpoint_update=0, log=str(directory / "train.jsonl"))
        status_path = directory / "status.json"
        atomic_json(status_path, info)
        current, elapsed, previous_elapsed, saved = 0, 0.0, 0.0, None
        try:
            logging.getLogger("basicsr").setLevel(logging.ERROR)
            seed_all(seed)
            model = legacy.build_model(opt)
            dataset = legacy.build_dataset(opt, paths)
            sampler = legacy.ExactSampler(len(dataset), seed + 20000)
            seed_all(seed + 10000)
            initial_encoder = {key: param.detach().cpu().clone()
                               for key, param in model.net_g.encoder.named_parameters()}
            bn_before = model.net_g.encoder.bn1.running_mean.detach().cpu().clone()
            latest = directory / "checkpoints/latest.json"
            if resume and latest.exists():
                record = read_json(latest)
                saved = Path(record["path"])
                if saved.resolve().parent != (directory / "checkpoints").resolve():
                    raise RuntimeError("Recovery checkpoint is outside this run")
                if sha256_file(saved) != record["sha256"]:
                    raise RuntimeError("Recovery checkpoint digest mismatch")
                checkpoint = legacy.validate_checkpoint(saved, opt, record["update"])
                current, previous_elapsed = checkpoint["update"], checkpoint["elapsed_seconds"]
                legacy.restore_checkpoint(model, sampler, checkpoint)
                del checkpoint
                _restore_log(directory, current)
                info.update(recovery="complete params/optimizer/scheduler/RNG/sampler state restored",
                            checkpoint=str(saved), checkpoint_update=current, update=current)
            elif resume and (directory / "train.jsonl").exists():
                raise RuntimeError("Prior training log has no complete recovery checkpoint")
            if current > stop:
                raise ValueError("stop_after precedes the restored checkpoint")
            counts = {"total": sum(p.numel() for p in model.net_g.parameters()),
                      "trainable": sum(p.numel() for p in model.net_g.parameters() if p.requires_grad)}
            manifest = dict(opt["runtime"], **common, git_commit=opt["git_commit"], params=counts,
                            pretrained_load=model.net_g.encoder.pretrain_load_report,
                            pretrained_sha256=opt["pretrained_sha256"],
                            full_training=opt["full_training"], execution_contract=opt["execution_contract"],
                            torch=torch.__version__, cuda=torch.version.cuda,
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            gpu_name=torch.cuda.get_device_name(0) if cuda else None,
                            python=sys.version, command=sys.argv,
                            comparison_note="Original batch16 FP32 and single-pass input path; only the configured structure loss changes the method.")
            atomic_json(directory / "manifest.json", manifest)
            info.update(update=current, updated_at=utc_now())
            atomic_json(status_path, info)
            started = time.perf_counter()
            elapsed = previous_elapsed
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            with (directory / "train.jsonl").open("a", encoding="utf-8", buffering=1) as log:
                for update in range(current + 1, stop + 1):
                    if cuda:
                        torch.cuda.synchronize()
                    step_started = time.perf_counter()
                    loss = legacy.train_update(model, dataset, sampler, update, 16, 1)
                    if cuda:
                        torch.cuda.synchronize()
                    current = update
                    elapsed = previous_elapsed + time.perf_counter() - started
                    row = {"update": current, "loss": loss, "lr": model.get_current_learning_rate()[0],
                           "seconds": time.perf_counter() - step_started, "elapsed_seconds": elapsed,
                           "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if cuda else None}
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    if current == 1 or current % opt["runtime"]["status_every"] == 0 or current == stop:
                        info.update(**row, updated_at=utc_now())
                        atomic_json(status_path, info)
                        print(f"G1C seed{seed} update {current}/{TOTAL_UPDATES} loss={loss:.6f} sec/update={row['seconds']:.3f}", flush=True)
                    if current % opt["runtime"]["checkpoint_every"] == 0 or current == stop:
                        log.flush()
                        os.fsync(log.fileno())
                        saved = legacy.save_checkpoint(model, sampler, opt, current, elapsed, directory)
                        info.update(checkpoint=str(saved), checkpoint_update=current, updated_at=utc_now())
                        atomic_json(status_path, info)
            for key, param in model.net_g.encoder.named_parameters():
                if not torch.equal(initial_encoder[key], param.detach().cpu()):
                    raise AssertionError(f"Frozen encoder parameter changed: {key}")
            completed = current == TOTAL_UPDATES
            checkpoint_sha = sha256_file(saved)
            if checkpoint_sha != read_json(latest)["sha256"]:
                raise RuntimeError("Saved checkpoint digest mismatch")
            checks = dict(common, complete=completed, update=current, loss_finite=True,
                          frozen_encoder_unchanged=True, encoder_parameters_unchanged=True,
                          encoder_gradients_absent=all(p.grad is None for p in model.net_g.encoder.parameters()),
                          encoder_bn_training=model.net_g.encoder.bn1.training,
                          bn_running_mean_changed=not torch.equal(bn_before, model.net_g.encoder.bn1.running_mean.cpu()),
                          parameter_counts=counts, last_checkpoint_sha256_verified=True,
                          checkpoint_sha256=checkpoint_sha)
            atomic_json(directory / "training_checks.json", checks)
            final = directory / "checkpoints/final.pth"
            if completed:
                if not final.exists():
                    os.link(saved, final)
                legacy.validate_checkpoint(final, opt, TOTAL_UPDATES)
                if sha256_file(final) != checkpoint_sha:
                    raise RuntimeError("Final checkpoint digest mismatch")
                manifest.update(final_checkpoint=str(final), final_checkpoint_sha256=checkpoint_sha)
            manifest.update(completed_optimizer_updates=current, total_training_seconds=elapsed)
            atomic_json(directory / "manifest.json", manifest)
            summary = dict(common, completed=completed, updates=current, budget=TOTAL_UPDATES,
                           checkpoint=str(final if completed else saved), checkpoint_sha256=checkpoint_sha,
                           elapsed_seconds=elapsed, ended_at=utc_now())
            atomic_json(directory / "summary.json", summary)
            info.update(status="completed" if completed else "paused_at_checkpoint", update=current,
                        checkpoint=str(final if completed else saved), checkpoint_update=current,
                        checkpoint_sha256=checkpoint_sha, elapsed_seconds=elapsed,
                        updated_at=utc_now(), ended_at=utc_now())
            atomic_json(status_path, info)
            return directory
        except BaseException as error:
            info.update(status="failed", update=current, elapsed_seconds=elapsed,
                        reason=f"{type(error).__name__}: {error}", updated_at=utc_now(), ended_at=utc_now())
            atomic_json(status_path, info)
            (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--tag")
    args = parser.parse_args()
    run(args.seed, load_paths(args.paths), args.resume, args.stop_after, args.tag)


if __name__ == "__main__":
    main()
