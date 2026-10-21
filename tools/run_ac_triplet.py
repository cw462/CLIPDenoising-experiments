"""Train matched B0/A/C models from one materialized input/target per update.

The three optimizers and PFA generators remain independent. The coordinator's
commit.json is the only recovery boundary; per-group files retain the existing
evaluation interface. No tolerance is used for the input/target contract.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time

# Must be set before importing torch or constructing any CUDA context.  Embedded
# callers which initialize CUDA first must supply the same setting themselves.
_CUDA_INITIALIZED_BEFORE_IMPORT = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
_CUBLAS_CONFIG_BEFORE_IMPORT = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.utils import get_root_logger
from tools.experiment_common import (
    RunLock, atomic_json, digest, load_paths, process_info, read_json, rng_state,
    seed_all, sha256_file, utc_now,
)
from tools.run_experiment import (
    ExactSampler, build_model, checkpoint_payload, restore_checkpoint, validate_checkpoint,
)
from tools.run_pfa_experiment import (
    configuration as pfa_configuration, durable_log_rows, encoder_hash,
    initialization_contract, tensor_hash,
)
from tools.run_structure_experiment import configuration as structure_configuration

GROUPS = ("B0", "B1", "B3")
EXECUTION_CONTRACT = {
    "schema": "ac_triplet_v2", "groups": list(GROUPS),
    "batch": "one sampler/data/Mixup realization per seed/update",
    "model_inputs": "independent contiguous clones; exact pre/post/reference checks",
    "optimizer_and_pfa": "independent per group; model steps cannot consume shared RNG",
    "recovery": "three validated checkpoints, then one atomic coordinator commit",
    "determinism": {
        "algorithms": True, "warn_only": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cudnn_allow_tf32": False, "cuda_matmul_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "structure_reflect_padding": "one-pixel slices/concatenation; exact reflection values",
    },
}


def configure_deterministic_execution(cuda):
    """Require deterministic kernels; unsupported operations must fail closed."""
    wanted = EXECUTION_CONTRACT["determinism"]["cublas_workspace_config"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != wanted:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must equal {wanted} before process startup")
    if cuda and _CUDA_INITIALIZED_BEFORE_IMPORT and _CUBLAS_CONFIG_BEFORE_IMPORT != wanted:
        raise RuntimeError("CUDA initialized before the required CUBLAS_WORKSPACE_CONFIG was set")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def configurations(stage, seed, paths, structure_paths):
    if Path(paths["output_root"]).resolve() != Path(structure_paths["output_root"]).resolve():
        raise ValueError("Triplet groups must share one output_root")
    result = {}
    for group in GROUPS:
        builder = structure_configuration if group == "B3" else pfa_configuration
        selected_paths = copy.deepcopy(structure_paths if group == "B3" else paths)
        if group == "B3":
            selected_paths["pixel_opt"]["deterministic_reflect_padding"] = True
        opt, directory = builder(group, stage, seed, selected_paths)
        opt["execution_contract"] = copy.deepcopy(EXECUTION_CONTRACT)
        opt["paired_protocol"]["execution_contract"] = copy.deepcopy(EXECUTION_CONTRACT)
        opt["paired_protocol_hash"] = digest(opt["paired_protocol"])
        opt.pop("config_hash", None)
        opt["config_hash"] = digest(opt)
        result[group] = (opt, directory)
    if len({opt["paired_protocol_hash"] for opt, _ in result.values()}) != 1:
        raise RuntimeError("Triplet scientific controls differ outside PFA/loss intervention")
    for field in ("device", "checkpoint_every", "checkpoint_keep", "status_every", "cpu_threads"):
        if len({opt["runtime"][field] for opt, _ in result.values()}) != 1:
            raise RuntimeError(f"Triplet runtime differs: {field}")
    if paths.get("gpu_index", 0) != structure_paths.get("gpu_index", 0):
        raise RuntimeError("Triplet groups must use the same CUDA device")
    return result


def exact_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_reference(reference, opt, directory, update):
    path = Path(reference["path"])
    if path.resolve().parent != (directory / "checkpoints").resolve():
        raise RuntimeError("Triplet checkpoint must belong to its group run")
    if reference["update"] != update or sha256_file(path) != reference["sha256"]:
        raise RuntimeError("Triplet checkpoint update or SHA256 mismatch")
    checkpoint = validate_checkpoint(path, opt, update)
    pfa = checkpoint.get("pfa_rng")
    if (not isinstance(pfa, dict) or pfa.get("rho") != opt["network_g"]["pfa_rho"]
            or pfa.get("seed") != opt["network_g"]["pfa_seed"] or not pfa.get("generators")):
        raise RuntimeError("Missing or mismatched independent PFA RNG checkpoint state")
    return checkpoint


def commit_checkpoints(models, sampler, configs, update, elapsed, coordinator, streams):
    """Keep the prior committed generation until every new group is durable."""
    for stream in streams.values():
        stream.flush()
        os.fsync(stream.fileno())
    references = {}
    controls = None
    for group in GROUPS:
        opt, directory = configs[group]
        out = directory / "checkpoints"
        out.mkdir(exist_ok=True)
        free = shutil.disk_usage(out).free
        if free < opt["runtime"]["minimum_free_gb"] * 1024**3:
            raise RuntimeError(f"Insufficient disk space: {free / 1024**3:.2f} GiB free")
        target = out / f"update_{update:06d}.pth"
        _atomic_torch(target, checkpoint_payload(models[group], sampler, opt, update, elapsed[group]))
        reference = {"path": str(target), "update": update, "sha256": sha256_file(target)}
        saved = _validate_reference(reference, opt, directory, update)
        current_controls = {key: saved[key] for key in ("rng", "sampler")}
        if controls is not None and not exact_state_equal(controls, current_controls):
            raise RuntimeError("Shared RNG/sampler changed during checkpoint transaction")
        controls = current_controls
        references[group] = reference
        del saved
    commit = {"schema": "ac_triplet_commit_v1", "update": update,
              "config_hashes": {group: configs[group][0]["config_hash"] for group in GROUPS},
              "paired_protocol_hash": configs["B0"][0]["paired_protocol_hash"],
              "source_hash": configs["B0"][0]["source_hash"], "groups": references,
              "committed_at": utc_now()}
    # If interrupted before this replacement, recovery still uses the prior files.
    atomic_json(coordinator / "commit.json", commit)
    for group in GROUPS:
        opt, directory = configs[group]
        atomic_json(directory / "checkpoints/latest.json", references[group])
    # latest.json is an evaluation view. Only the coordinator commits recovery.
    for group in GROUPS:
        opt, directory = configs[group]
        committed = Path(references[group]["path"])
        older = sorted(p for p in (directory / "checkpoints").glob("update_*.pth") if p < committed)
        keep_old = opt["runtime"]["checkpoint_keep"] - 1
        for path in older[:max(0, len(older) - keep_old)]:
            path.unlink()
    return commit


def restore_committed(models, sampler, configs, coordinator):
    commit = read_json(coordinator / "commit.json")
    if (commit.get("schema") != "ac_triplet_commit_v1"
            or commit.get("config_hashes") != {g: configs[g][0]["config_hash"] for g in GROUPS}
            or commit.get("paired_protocol_hash") != configs["B0"][0]["paired_protocol_hash"]
            or commit.get("source_hash") != configs["B0"][0]["source_hash"]
            or set(commit.get("groups", {})) != set(GROUPS)):
        raise RuntimeError("Triplet commit source/config/group mismatch")
    update = commit["update"]
    if not 1 <= update <= configs["B0"][0]["train"]["total_iter"]:
        raise RuntimeError("Invalid triplet committed update")
    elapsed, controls = {}, None
    for group in GROUPS:
        opt, directory = configs[group]
        saved = _validate_reference(commit["groups"][group], opt, directory, update)
        current_controls = {key: saved[key] for key in ("rng", "sampler")}
        if controls is not None and not exact_state_equal(controls, current_controls):
            raise RuntimeError("Committed group RNG/sampler differ")
        controls = current_controls
        restore_checkpoint(models[group], sampler, saved)
        elapsed[group] = saved["elapsed_seconds"]
        del saved
    # Repair incomplete publication only after all committed groups validate.
    for group in GROUPS:
        atomic_json(configs[group][1] / "checkpoints/latest.json", commit["groups"][group])
    return update, elapsed, commit


def _fail_pairing(coordinator, update, group, reason, raw, reference, model, before_rng, sampler, data_before):
    path = coordinator / f"failure_update_{update:06d}.pth"
    _atomic_torch(path, {"update": update, "group": group, "reason": reason,
                        "raw_batch": raw, "reference": {k: v.detach().cpu() for k, v in reference.items()},
                        "observed": {"lq": model.lq.detach().cpu(), "gt": model.gt.detach().cpu()},
                        "rng_before_model": before_rng, "rng_after_model": rng_state(),
                        "data_before": data_before, "pfa_rng_after_model": model.net_g.pfa.state_dict(),
                        "sampler": sampler.state_dict()})
    atomic_json(path.with_suffix(".json"), {"update": update, "group": group,
                                           "reason": reason, "tensor_file": str(path),
                                           "sha256": sha256_file(path), "at": utc_now()})
    raise RuntimeError(f"Exact triplet pairing failed at update {update} ({group}): {reason}")


def triplet_update(models, dataset, sampler, current, batch, coordinator):
    """One data/Mixup realization, then three independently owned exact copies."""
    data_before = {"rng": rng_state(), "sampler": copy.deepcopy(sampler.state_dict())}
    raw = default_collate([dataset[i] for i in sampler.take(batch)])
    models["B0"].feed_train_data(raw)
    reference = {"lq": models["B0"].lq.detach().clone(memory_format=torch.contiguous_format),
                 "gt": models["B0"].gt.detach().clone(memory_format=torch.contiguous_format)}
    hashes = {key: tensor_hash(value) for key, value in reference.items()}
    before_rng = rng_state()
    rows = {}
    for group in GROUPS:
        model = models[group]
        model.lq = reference["lq"].clone(memory_format=torch.contiguous_format)
        model.gt = reference["gt"].clone(memory_format=torch.contiguous_format)
        # Hash the actual model tensors before AND after the complete optimizer step.
        before = {"lq": tensor_hash(model.lq), "gt": tensor_hash(model.gt)}
        if before != hashes or model.lq.data_ptr() == reference["lq"].data_ptr() or model.gt.data_ptr() == reference["gt"].data_ptr():
            _fail_pairing(coordinator, current, group, "independent input copies differ from reference", raw, reference, model, before_rng, sampler, data_before)
        started = time.perf_counter()
        model.update_learning_rate(current, warmup_iter=-1)
        model.optimize_parameters(current)
        if model.lq.is_cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        after = {"lq": tensor_hash(model.lq), "gt": tensor_hash(model.gt)}
        if after != hashes:
            _fail_pairing(coordinator, current, group, "input or target changed during model step", raw, reference, model, before_rng, sampler, data_before)
        if not exact_state_equal(before_rng, rng_state()):
            _fail_pairing(coordinator, current, group, "model step consumed shared data RNG", raw, reference, model, before_rng, sampler, data_before)
        loss = float(model.get_current_log()["l_pix"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {group} update {current}")
        row = {"update": current, "loss": loss, "lr": model.optimizer_g.param_groups[0]["lr"],
               "seconds": seconds, "input_sha256": after["lq"], "target_sha256": after["gt"],
               "reference_input_sha256": hashes["lq"], "reference_target_sha256": hashes["gt"],
               "pre_input_sha256": before["lq"], "pre_target_sha256": before["gt"],
               "post_input_sha256": after["lq"], "post_target_sha256": after["gt"],
               "exact_reference_and_unchanged": True, "shared_rng_unchanged": True}
        components = getattr(model.cri_pix, "last_components", None)
        row.update({key: float(value) for key, value in components.items()} if components else
                   {"pixel_l1": loss, "structure": 0.0, "weighted_structure": 0.0})
        rows[group] = row
    return rows


def run(stage, seed, paths, structure_paths, resume=False, stop_after=None):
    if paths["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    cuda = paths["device"] == "cuda"
    configure_deterministic_execution(cuda)
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    if cuda:
        torch.cuda.set_device(int(paths.get("gpu_index", 0)))
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
    configs = configurations(stage, seed, paths, structure_paths)
    directories = {g: configs[g][1] for g in GROUPS}
    coordinator = Path(paths["output_root"]) / stage / "_triplet" / f"seed{seed}"
    budget = configs["B0"][0]["train"]["total_iter"]
    if stop_after is not None and not 1 <= stop_after <= budget:
        raise ValueError("stop-after must be within the unchanged stage budget")
    if not resume and any(path.exists() for path in [coordinator, *directories.values()]):
        raise FileExistsError("Triplet run exists; use --resume")
    coordinator.mkdir(parents=True, exist_ok=True)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(RunLock(coordinator / "process.lock"))
        for directory in directories.values():
            stack.enter_context(RunLock(directory / "process.lock"))
        for opt, directory in configs.values():
            config_file = directory / "effective_config.json"
            if config_file.exists() and read_json(config_file) != opt:
                raise RuntimeError("Source/config/C0 changed; use a new output root")
            atomic_json(config_file, opt)
        started_at, wall_started = utc_now(), time.perf_counter()
        info = {"status": "initializing", "stage": stage, "seed": seed, "groups": list(GROUPS),
                "process": process_info(), "started_at": started_at, "updated_at": started_at,
                "ended_at": None, "budget": budget, "update": 0, "durable_update": 0,
                "source_hash": configs["B0"][0]["source_hash"], "execution_contract": EXECUTION_CONTRACT}
        update, commit = 0, None
        elapsed = {g: 0.0 for g in GROUPS}

        def status(state, current):
            info.update(status=state, update=current, updated_at=utc_now(),
                        durable_update=commit["update"] if commit else 0,
                        invocation_wall_seconds=time.perf_counter() - wall_started,
                        ended_at=utc_now() if state in ("completed", "paused_at_checkpoint", "failed", "interrupted") else None)
            atomic_json(coordinator / "status.json", info)
            for group, directory in directories.items():
                reference = commit["groups"][group] if commit else {}
                atomic_json(directory / "status.json", {**info, "group": group, "experiment": group,
                    "config_hash": configs[group][0]["config_hash"], "checkpoint": reference.get("path"),
                    "checkpoint_sha256": reference.get("sha256"), "elapsed_seconds": elapsed[group],
                    "triplet_commit": str(coordinator / "commit.json")})
        status("initializing", 0)
        try:
            models, initializations = {}, {}
            for group in GROUPS:
                seed_all(seed)
                models[group] = build_model(configs[group][0])
                initializations[group] = initialization_contract(models[group], configs[group][0])
            if len({id(models[g].optimizer_g) for g in GROUPS}) != 3 or len({id(models[g].net_g.pfa) for g in GROUPS}) != 3:
                raise RuntimeError("Models must own independent optimizer/PFA objects")
            dataset = Dataset_GaussianDenoising(copy.deepcopy(configs["B0"][0]["datasets"]["train"]))
            if len(dataset) != 432:
                raise RuntimeError("Expected all 432 training images")
            sampler = ExactSampler(len(dataset), seed + 20000)
            seed_all(seed + 10000)
            if (coordinator / "commit.json").exists():
                update, elapsed, commit = restore_committed(models, sampler, configs, coordinator)
            elif any((directory / "train.jsonl").exists() for directory in directories.values()):
                raise RuntimeError("Interrupted before first triplet commit; preserve this attempt and use a new output root")
            if stop_after is not None and update < budget and stop_after <= update:
                raise ValueError("stop-after must exceed the committed update")
            rows = {}
            for group, directory in directories.items():
                opt = configs[group][0]
                manifest = directory / "manifest.json"
                if not manifest.exists():
                    atomic_json(manifest, {"group": group, "stage": stage, "seed": seed, "budget": budget,
                        "config_hash": opt["config_hash"], "source_hash": opt["source_hash"],
                        "paired_protocol_hash": opt["paired_protocol_hash"], "git_commit": opt.get("git_commit"),
                        "started_at": started_at, "c0_path": opt["path"]["pretrain_network_g"],
                        "c0_sha256": opt.get("plan_c", opt.get("plan_a"))["c0_sha256"],
                        "initialization_checks": initializations[group], "execution_contract": EXECUTION_CONTRACT,
                        "paired_data_rng_seed": seed + 10000, "sampler_seed": seed + 20000,
                        "independent_pfa_seed": seed + 30000, "checkpoint_every": opt["runtime"]["checkpoint_every"]})
                log_path = directory / "train.jsonl"
                rows[group] = durable_log_rows(log_path, update)
                for row in rows[group]:
                    if (not row.get("exact_reference_and_unchanged") or not row.get("shared_rng_unchanged")
                            or len({row[k] for k in ("input_sha256", "reference_input_sha256", "pre_input_sha256", "post_input_sha256")}) != 1
                            or len({row[k] for k in ("target_sha256", "reference_target_sha256", "pre_target_sha256", "post_target_sha256")}) != 1):
                        raise RuntimeError("Committed log lacks exact input/target checks")
                log_path.write_text("".join(json.dumps(row) + "\n" for row in rows[group]), encoding="utf-8")
            if any([(r["input_sha256"], r["target_sha256"]) for r in rows[g]] != [(r["input_sha256"], r["target_sha256"]) for r in rows["B0"]] for g in GROUPS):
                raise RuntimeError("Committed triplet input/target log sequences differ")
            stop = budget if update >= budget or stop_after is None else stop_after
            status("running", update)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            streams = {g: stack.enter_context((directories[g] / "train.jsonl").open("a", encoding="utf-8", buffering=1)) for g in GROUPS}
            for current in range(update + 1, stop + 1):
                began = time.perf_counter()
                step_rows = triplet_update(models, dataset, sampler, current, paths["micro_batch"], coordinator)
                for group in GROUPS:
                    row = step_rows[group]
                    row["triplet_wall_seconds"] = time.perf_counter() - began
                    rows[group].append(row)
                    elapsed[group] += row["seconds"]
                    streams[group].write(json.dumps(row, allow_nan=False) + "\n")
                update = current
                if current % configs["B0"][0]["runtime"]["status_every"] == 0 or current == stop:
                    print(f"triplet {stage} seed{seed} {current}/{budget}: " + ", ".join(f"{g}={step_rows[g]['loss']:.6f}" for g in GROUPS), flush=True)
                    status("running", current)
                if current % configs["B0"][0]["runtime"]["checkpoint_every"] == 0 or current == stop:
                    commit = commit_checkpoints(models, sampler, configs, current, elapsed, coordinator, streams)
                    status("finalizing" if current == stop else "running", current)
            for group in GROUPS:
                opt, directory = configs[group]
                model, initialization = models[group], initializations[group]
                frozen = encoder_hash(model) == initialization["encoder_parameter_sha256"]
                if not frozen:
                    raise RuntimeError(f"Frozen encoder changed in {group}")
                reference = commit["groups"][group]
                checkpoint_verified = sha256_file(reference["path"]) == reference["sha256"]
                encoder_gradients_absent = all(p.grad is None for p in model.net_g.encoder.parameters())
                if not checkpoint_verified or not encoder_gradients_absent or not model.net_g.encoder.training:
                    raise RuntimeError(f"Final checkpoint or frozen-encoder contract failed in {group}")
                atomic_json(directory / "training_checks.json", {**initialization,
                    "encoder_parameters_unchanged": frozen, "encoder_bn_training_after": model.net_g.encoder.training,
                    "encoder_gradients_absent": encoder_gradients_absent,
                    "last_checkpoint_sha256_verified": checkpoint_verified,
                    "exact_reference_pre_post_every_update": True, "shared_rng_unchanged_every_model_step": True,
                    "triplet_commit_update": commit["update"], "update": stop, "complete": stop == budget, "checked_at": utc_now()})
                timings = [r["seconds"] for r in rows[group] if r["update"] > 10]
                last = rows[group][-1]
                atomic_json(directory / "summary.json", {"group": group, "stage": stage, "seed": seed,
                    "updates": stop, "budget": budget, "completed": stop == budget, "batch_size": paths["micro_batch"],
                    "device": torch.cuda.get_device_name() if cuda else "cpu", "rho": opt["network_g"]["pfa_rho"],
                    "c0_sha256": opt.get("plan_c", opt.get("plan_a"))["c0_sha256"],
                    "last_loss": last["loss"], "last_pixel_l1": last["pixel_l1"], "last_structure": last["structure"],
                    "last_weighted_structure": last["weighted_structure"], "first_lr": rows[group][0]["lr"], "last_lr": last["lr"],
                    "elapsed_seconds": elapsed[group], "invocation_started_at": started_at, "ended_at": utc_now(),
                    "invocation_wall_seconds": time.perf_counter() - wall_started, "checkpoint": reference["path"],
                    "checkpoint_sha256": reference["sha256"], "execution_contract": EXECUTION_CONTRACT,
                    "mean_update_seconds_after_warmup": sum(timings) / len(timings) if timings else None,
                    "mean_triplet_wall_seconds_after_warmup": (sum(r["triplet_wall_seconds"] for r in rows[group] if r["update"] > 10)
                        / len(timings) if timings else None),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
                    "training_result_claim": "Runtime/stability verification only" if stage == "smoke" else "Requires paired evaluation"})
            status("completed" if stop == budget else "paused_at_checkpoint", stop)
            return directories
        except BaseException as error:
            info["reason"] = str(error)
            status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", update)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "screen", "confirm", "confirm_short"), default="smoke")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--paths", default="configs/plan_a.yaml")
    parser.add_argument("--structure-paths", default="configs/plan_c.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    run(args.stage, args.seed, load_paths(args.paths), load_paths(args.structure_paths), args.resume, args.stop_after)


if __name__ == "__main__":
    main()
