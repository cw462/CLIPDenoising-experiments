"""Evaluate fixed checkpoints without exposing benchmark results during development."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from tools.experiment_common import (
    load_paths, read_json, atomic_json, sha256_file, digest, source_hash,
)
from tools.evaluation_utils import (
    CONDITIONS, FULL_QUEUE, PROTOCOL, VISUAL_CONDITIONS, calculate_metrics,
    image_files, make_noise, read_rgb, region_errors, save_visuals, write_csv,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(path: Path | str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location="cpu")


def dependency_versions() -> dict:
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__}
    for name in ("Pillow", "scipy", "opencv-python", "opencv-python-headless",
                 "opencv-contrib-python", "torchvision"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def checkpoint_path(run_dir: Path, development: bool = False) -> Path:
    # Prefer a deliberately fixed final checkpoint. Do not choose by validation.
    names = ["checkpoints/final.pth", "final.pth", "checkpoints/update_300000.pth",
             "checkpoints/update_300000.pt", "checkpoints/300000.pth"]
    if development:
        names += ["checkpoints/latest.pth", "latest.pth"]
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    state_file = run_dir / "status.json"
    if development and state_file.is_file():
        state = read_json(state_file)
        path = Path(state.get("checkpoint", ""))
        if path.is_file():
            return path
    raise FileNotFoundError(f"No fixed {'development' if development else '300000-update'} checkpoint in {run_dir}")


def inspect_run(run_dir: Path, development: bool = False) -> dict:
    config_file = run_dir / "effective_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing effective config: {config_file}")
    config = read_json(config_file)
    if digest({key: value for key, value in config.items() if key != "config_hash"}) != config.get("config_hash"):
        raise ValueError(f"Effective config content hash mismatch: {config_file}")
    if config.get("source_hash") != source_hash():
        raise ValueError(f"Current evaluation/training source differs from frozen run source: {run_dir}")
    checkpoint = checkpoint_path(run_dir, development)
    # Only checkpoints created in the local experiment workspace are accepted.
    state = load_checkpoint(checkpoint)
    required = {"params", "optimizer", "scheduler", "update", "config_hash", "source_hash",
                "rng", "sampler", "scaler", "ema"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Not a complete training checkpoint: {checkpoint}; missing {sorted(missing)}")
    stage = config.get("stage")
    if development:
        if stage not in {"smoke", "pilot"}:
            raise ValueError("Development evaluation only accepts smoke/pilot runs")
    elif stage != "full" or int(state["update"]) != 300000:
        raise ValueError(f"Formal evaluation requires stage full and exactly 300000 updates: {checkpoint}")
    for key in ("config_hash", "source_hash"):
        if not config.get(key) or state[key] != config[key]:
            raise ValueError(f"Checkpoint/effective config {key} mismatch: {checkpoint}")
    experiment = config.get("exp", config.get("experiment"))
    seed = int(config.get("seed", config.get("train_seed", -1)))
    if experiment is None or seed < 0:
        raise ValueError(f"Missing experiment/seed in {config_file}")
    if (state.get("experiment"), state.get("seed"), state.get("stage")) != (experiment, seed, stage):
        raise ValueError(f"Checkpoint experiment/seed/stage mismatch: {checkpoint}")
    if not state.get("complete_update_boundary"):
        raise ValueError(f"Checkpoint was not saved at a complete optimizer update: {checkpoint}")
    result = {
        "run_dir": str(run_dir.resolve()), "experiment": experiment, "train_seed": seed,
        "stage": stage, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint), "update": int(state["update"]),
        "config_hash": state["config_hash"], "source_hash": state["source_hash"],
        "git_commit": config.get("git_commit", state.get("git_commit", "")),
        "config_file": str(config_file.resolve()), "config": config,
        "checkpoint_training_metadata": {
            key: state[key] for key in (
                "training_seconds", "elapsed_seconds", "total_training_seconds",
                "sampler_state", "sampling_protocol", "amp", "micro_batch",
                "accumulation", "noisy_views_per_update", "unique_clean_patches_per_update"
            ) if key in state and isinstance(state[key], (str, int, float, bool, type(None)))
        },
    }
    del state
    gc.collect()
    return result


def formal_gate(paths: dict) -> list[dict]:
    """Must succeed for all eight runs before any final test image is opened."""
    output_root = Path(paths["output_root"])
    runs, errors = [], []
    for experiment, seed in FULL_QUEUE:
        directory = output_root / "full" / experiment / f"seed{seed}"
        try:
            run = inspect_run(directory)
            if (run["experiment"], run["train_seed"]) != (experiment, seed):
                raise ValueError("Experiment identity does not match required queue")
            runs.append(run)
        except Exception as error:
            errors.append(f"{experiment}/seed{seed}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("Formal benchmark blocked; all eight fixed checkpoints are required:\n" + "\n".join(errors))
    if len({run["source_hash"] for run in runs}) != 1:
        raise ValueError("Formal runs were produced with different source hashes")
    return runs


def datasets_for(paths: dict, development: bool) -> dict[str, tuple[Path, list[Path]]]:
    if development:
        train_dir = Path(paths["train_dir"])
        split_file = Path(paths.get("dev_val", paths.get("dev_val_split", ROOT / "splits/dev_val.txt")))
        entries = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")]
        project_root = Path(paths.get("project_root", ROOT))
        files = []
        for entry in entries:
            candidate = Path(entry)
            if not candidate.is_absolute():
                root_candidate = project_root / candidate
                candidate = root_candidate if root_candidate.is_file() else train_dir / candidate
            files.append(candidate)
        if len(files) != 32 or len(set(files)) != 32 or not all(p.is_file() for p in files):
            raise ValueError("Development evaluation requires the audited, distinct 32-image split")
        return {"dev_val32": (train_dir, files)}
    datasets = {}
    for name, value in paths["test_datasets"].items():
        directory = Path(value)
        files = image_files(directory) if directory.is_dir() else []
        if not files:
            if name.upper() == "CBSD68":
                raise FileNotFoundError(f"Required final CBSD68 dataset is missing: {directory}")
            continue
        expected = {"CBSD68": 68, "KODAK": 24, "KODAK24": 24, "MCM": 18,
                    "MCMASTER": 18, "URBAN100": 100}.get(name.upper())
        if expected and len(files) != expected:
            raise ValueError(f"{name} expected {expected} images, got {len(files)}")
        datasets[name] = directory, files
    if not any(name.upper() == "CBSD68" for name in datasets):
        raise ValueError("No CBSD68 dataset configured")
    return datasets


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer(model, noisy: np.ndarray, device: torch.device) -> tuple[np.ndarray, float, float | None]:
    height, width = noisy.shape[:2]
    tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = F.pad(tensor, (0, (-width) % 32, 0, (-height) % 32), mode="reflect")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        restored = model(tensor)
    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
    restored = restored[:, :, :height, :width].clamp(0, 1).cpu().permute(0, 2, 3, 1)[0].numpy()
    return restored, elapsed, peak


def measure_runtime(model, device: torch.device) -> dict:
    # Synthetic input here measures timing only; never written as a denoising score.
    sample = torch.from_numpy(np.random.RandomState(2026).rand(1, 3, 256, 256).astype(np.float32)).to(device)
    timings = []
    with torch.inference_mode():
        for _ in range(3):
            model(sample)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(10):
            synchronize(device)
            start = time.perf_counter()
            model(sample)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "height": 256, "width": 256, "batch_size": 1, "warmup_forwards": 3,
        "timed_forwards": 10, "forward_ms_mean": float(np.mean(timings)),
        "forward_ms_std": float(np.std(timings, ddof=1)),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else "",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device), "torch_version": torch.__version__,
        "timing_input": "Synthetic fixed tensor; timing only, no benchmark metric",
    }


def append_access(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def evaluate_run(run: dict, paths: dict, development: bool, visuals: bool,
                 fixed_runs: list[dict]) -> Path:
    from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising

    run_dir = Path(run["run_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = run_dir / "evaluations" / (("development_" if development else "benchmark_") + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {key: value for key, value in run.items() if key != "config"}
    manifest.update({
        "status": "running", "started_at": utcnow(), "development": development,
        "protocol": PROTOCOL, "fixed_checkpoints": [
            {key: value for key, value in fixed.items() if key != "config"} for fixed in fixed_runs],
        "dependencies": dependency_versions(),
        "training_metadata": run["config"],
        "official_utils_sha256": sha256_file(ROOT / "Denoising/utils.py"),
    })
    access_path = Path(paths["reports_dir"]) / ("development_access.jsonl" if development else "test_access.jsonl")
    append_access(access_path, {"event": "begin", "at": utcnow(), "directory": str(directory),
                               "experiment": run["experiment"], "seed": run["train_seed"],
                               "checkpoint": run["checkpoint"], "checkpoint_sha256": run["checkpoint_sha256"],
                               "fixed_checkpoints": manifest["fixed_checkpoints"],
                               "purpose": "source_domain_development" if development else "fixed_final_benchmark"})
    atomic_json(directory / "manifest.json", manifest)
    try:
        device = torch.device(paths.get("device", "cuda"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for evaluation but no CUDA GPU is available")
        datasets = datasets_for(paths, development)
        manifest["unavailable_optional_datasets"] = (
            [name for name in paths["test_datasets"] if name not in datasets]
            if not development else []
        )
        dataset_manifest = []
        for name, (base, files) in datasets.items():
            for image in files:
                dataset_manifest.append({"dataset": name, "image_id": image.relative_to(base).as_posix(),
                                         "sha256": sha256_file(image)})
        manifest["dataset_manifest"] = dataset_manifest
        manifest["dataset_manifest_hash"] = hashlib.sha256(
            json.dumps(dataset_manifest, sort_keys=True).encode()).hexdigest()
        network = dict(run["config"]["network_g"])
        network.pop("type", None)
        network["model_path"] = str(paths["pretrained_rn50"])
        model = CLIPDenoising(**network)
        state = load_checkpoint(run["checkpoint"])
        model.load_state_dict(state["params"], strict=True)
        del state
        model.to(device).eval()
        adapter = getattr(model.up_path[4], "rgb_calibration", None)
        if adapter is not None:
            adapter.capture_diagnostics = False
        runtime = measure_runtime(model, device)
        runtime.update({key: run[key] for key in ("experiment", "train_seed", "stage",
                       "checkpoint", "checkpoint_sha256", "config_hash", "source_hash")})
        write_csv(directory / "runtime.csv", [runtime])
        rows = []
        conditions = (("gaussian", 15),) if development else CONDITIONS
        total = sum(len(files) for _, files in datasets.values()) * len(conditions)
        for dataset, (base, files) in datasets.items():
            fixed_examples = set(files[:2])
            for noise_type, level in conditions:
                for file in files:
                    image_id = file.relative_to(base).as_posix()
                    t0 = time.perf_counter()
                    clean = read_rgb(file)
                    read_ms = (time.perf_counter() - t0) * 1000
                    t0 = time.perf_counter()
                    noisy, noise_seed, noise_hash = make_noise(clean, noise_type, level, dataset, image_id)
                    noise_ms = (time.perf_counter() - t0) * 1000
                    selected = visuals and file in fixed_examples and (noise_type, float(level)) in VISUAL_CONDITIONS
                    if adapter is not None:
                        adapter.capture_diagnostics = selected
                    restored, forward_ms, peak = infer(model, noisy, device)
                    metrics = calculate_metrics(clean, restored)
                    save_ms = 0.0
                    visual_directory = ""
                    if selected:
                        t0 = time.perf_counter()
                        key = hashlib.sha256(image_id.encode()).hexdigest()[:10]
                        visual_directory = str(directory / "figures" / dataset /
                                               f"{file.stem}_{key}" / f"{noise_type}_{level:g}")
                        save_visuals(Path(visual_directory), clean, noisy, restored,
                                     adapter.last_maps if adapter is not None else None)
                        save_ms = (time.perf_counter() - t0) * 1000
                    row = {key: run[key] for key in (
                        "experiment", "train_seed", "stage", "checkpoint", "checkpoint_sha256",
                        "update", "git_commit", "config_hash", "source_hash")}
                    row.update({
                        "dataset": dataset, "image_id": image_id, "noise_type": noise_type,
                        "noise_level": level, "noise_seed": noise_seed, "noise_sha256": noise_hash,
                        "protocol_group": "extension" if noise_type in {"speckle", "salt_pepper"} else "official",
                        "domain": "ID" if noise_type == "gaussian" and level == 15 else "OOD",
                        "dataset_manifest_hash": manifest["dataset_manifest_hash"],
                        "height": clean.shape[0], "width": clean.shape[1], **metrics,
                        "forward_ms": forward_ms, "peak_memory_mb": peak if peak is not None else "",
                        "read_ms": read_ms, "noise_ms": noise_ms, "save_ms": save_ms,
                        "visual_directory": visual_directory, **region_errors(clean, restored),
                    })
                    rows.append(row)
                    if len(rows) % 20 == 0:
                        print(f"{run['experiment']} seed{run['train_seed']}: {len(rows)}/{total}", flush=True)
        write_csv(directory / "per_image_results.csv", rows)
        manifest.update({"status": "completed", "completed_at": utcnow(), "rows": len(rows),
                         "metrics_csv_sha256": sha256_file(directory / "per_image_results.csv"),
                         "runtime": runtime})
        atomic_json(directory / "manifest.json", manifest)
        append_access(access_path, {"event": "completed", "at": utcnow(), "directory": str(directory),
                                   "rows": len(rows), "checkpoint_sha256": run["checkpoint_sha256"]})
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return directory
    except Exception as error:
        manifest.update({"status": "blocked", "blocked_at": utcnow(),
                         "reason": f"{type(error).__name__}: {error}"})
        atomic_json(directory / "manifest.json", manifest)
        append_access(access_path, {"event": "blocked", "at": utcnow(), "directory": str(directory),
                                   "reason": manifest["reason"]})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default=str(ROOT / "configs/local_paths.yaml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--visuals", action="store_true")
    args = parser.parse_args(argv)
    paths = load_paths(args.paths)
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    try:
        if args.development:
            if args.run_dir is None:
                parser.error("--development requires --run-dir")
            runs = [inspect_run(args.run_dir.resolve(), development=True)]
            fixed = runs
        else:
            fixed = formal_gate(paths)
            runs = fixed
            if args.run_dir:
                runs = [run for run in fixed if Path(run["run_dir"]) == args.run_dir.resolve()]
                if not runs:
                    raise ValueError("--run-dir must be one of the eight registered formal runs")
        for run in runs:
            output = evaluate_run(run, paths, args.development,
                                  args.visuals or not args.development, fixed)
            print(f"Completed evaluation: {output}", flush=True)
    except Exception as error:
        status = {"status": "blocked", "at": utcnow(), "development": args.development,
                  "reason": f"{type(error).__name__}: {error}"}
        atomic_json(Path(paths["reports_dir"]) /
                    ("development_evaluation_status.json" if args.development else "evaluation_status.json"), status)
        print(status["reason"], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
