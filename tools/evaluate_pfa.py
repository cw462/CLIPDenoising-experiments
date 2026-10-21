"""Proposal A's fixed seven-condition evaluation, with an explicit held-out split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.ndimage import convolve
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.nn import functional as F

from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from Denoising.utils import add_poisson_noise
from tools.evaluation_utils import image_files, read_rgb, save_rgb, write_csv
from tools.experiment_common import (atomic_json, digest, load_paths, read_json,
                                     sha256_file, source_hash, torch_load, utc_now)

CONDITIONS = (("gaussian", 15), ("gaussian", 25), ("gaussian", 50),
              ("spatial_gaussian", 45), ("spatial_gaussian", 55),
              ("poisson", 2.5), ("poisson", 3.5))


def fixed_split(split, paths=None):
    paths = paths or {}
    base = Path(paths.get("project_root", ROOT))
    datasets = paths.get("test_datasets", {})
    files = image_files(Path(datasets.get("CBSD68", base / "data/CBSD68")))
    if len(files) != 68:
        raise ValueError("CBSD68 must contain exactly 68 images")
    for name, entries in (("dev", files[:20]), ("holdout", files[20:])):
        target = base / f"splits/plan_a_{name}.txt"
        root = Path(datasets.get("CBSD68", base / "data/CBSD68"))
        body = "".join("data/CBSD68/" + path.relative_to(root).as_posix() + "\n" for path in entries)
        if target.exists() and target.read_text() != body:
            raise RuntimeError(f"Fixed evaluation split changed: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
    if split == "kodak":
        files = image_files(Path(datasets.get("Kodak24", base / "data/Kodak24")))
        if len(files) != 24:
            raise ValueError("Kodak24 must contain exactly 24 images")
        return files
    return files[:20] if split == "dev" else files[20:]


def noisy_input(clean, family, level, seed):
    # The upstream Poisson helper uses NumPy's global RNG: scope and restore it.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        if family == "poisson":
            noisy = add_poisson_noise(clean, scale=level)
        else:
            noise = np.random.normal(0, level / 255, clean.shape)
            if family == "spatial_gaussian":
                for channel in range(3):
                    noise[..., channel] = convolve(noise[..., channel], np.ones((3, 3)) / 9, mode="reflect")
            elif family != "gaussian":
                raise ValueError(f"Unknown noise: {family}")
            noisy = clean + noise
        return np.ascontiguousarray(noisy, dtype=np.float32)
    finally:
        np.random.set_state(state)


def metrics(clean, restored):
    if not np.isfinite(restored).all():
        raise FloatingPointError("Nonfinite model output")
    result = {"psnr": float(peak_signal_noise_ratio(clean, restored, data_range=1)),
            "ssim": float(structural_similarity(clean, restored, data_range=1, channel_axis=2,
                                                 gaussian_weights=True, sigma=1.5,
                                                 use_sample_covariance=False))}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite metric")
    return result


def summarize(rows, candidate='B1'):
    # Noise repeats are averaged within each image first, never bootstrap units.
    per_condition = defaultdict(list)
    for row in rows:
        per_condition[(row["method"], row["image"], row["family"], row["level"])].append(row["psnr"])
    per_image = defaultdict(list)
    conditions = defaultdict(list)
    for (method, image, family, level), values in per_condition.items():
        value = float(np.mean(values))
        category = "id" if family == "gaussian" and level == 15 else "ood"
        per_image[(method, image, category)].append(value)
        if family == "spatial_gaussian":
            per_image[(method, image, "spatial")].append(value)
        conditions[(method, family, level)].append(value)
    image_means = {key: float(np.mean(values)) for key, values in per_image.items()}
    methods = sorted({row["method"] for row in rows})
    summary = {method: {category + "_psnr": float(np.mean([
        value for (m, _, c), value in image_means.items() if m == method and c == category
    ])) for category in ("id", "ood", "spatial")} for method in methods}
    paired = {}
    if "B0" in methods and candidate in methods:
        for category in ("id", "ood", "spatial"):
            differences = np.array([value - image_means[("B0", image, category)]
                                    for (method, image, c), value in image_means.items()
                                    if method == candidate and c == category])
            samples = np.random.RandomState(2026).choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
            paired[category] = {f"{candidate}_minus_B0_db": float(differences.mean()),
                                "paired_image_bootstrap_95_ci": np.percentile(samples, [2.5, 97.5]).tolist(),
                                "independent_images": len(differences),
                                "scope": "Image uncertainty only; not training-seed variability"}
    return {"methods": summary, "paired": paired,
            "conditions": [{"method": m, "family": f, "level": l, "psnr": float(np.mean(v))}
                           for (m, f, l), v in conditions.items()]}


STAGE_SEEDS = {"screen": (100,), "confirm": (101, 102)}
VISUAL_CONDITIONS = tuple(item for item in CONDITIONS if item != ("gaussian", 25))
SPLIT_DATASET = {"dev": "CBSD68_dev20", "holdout": "CBSD68_holdout48", "kodak": "Kodak24"}
PROTOCOL = {
    "version": "ac-fixed-seven-v1", "noise_seeds": [0, 1, 2],
    "conditions": CONDITIONS,
    "noise": "NumPy RNG reset per image/condition/seed; no input clipping except official Poisson helper",
    "spatial_gaussian": "3x3 mean convolution per RGB channel, reflect; nominal prefilter sigma/255",
    "metrics": "RGB float [0,1], border=0, output clamp, skimage Gaussian-window SSIM sigma1.5 population covariance",
    "aggregation": "noise seeds first within image/condition; six OOD conditions equal; images equal within dataset; datasets equal",
    "selection_disclosure": "CBSD68 first20 are this-round development; other48 and Kodak24 held out only within this round. Prior study evaluated all these images; not study-naive test sets.",
    "visual_selection": "first two lexical images per split; fixed six conditions; noise seed0; centered96pixel crop",
}


def resolve_c0(paths):
    path = Path(paths["c0"])
    return path if path.is_absolute() else Path(paths["project_root"]) / path


def common_training_contract(config):
    """Only the prescribed A perturbation and C loss are allowed to differ."""
    train = copy.deepcopy(config["train"])
    train.pop("pixel_opt")
    network = copy.deepcopy(config["network_g"])
    network.pop("pfa_rho", None)
    network.pop("pfa_seed", None)
    runtime = copy.deepcopy(config["runtime"])
    for key in ("data_rng_seed", "sampler_seed", "independent_pfa_seed"):
        runtime.pop(key, None)
    plan = config.get("plan_c", config.get("plan_a", {}))
    return {"c0_sha256": plan["c0_sha256"], "new_optimizer": plan["new_optimizer"],
            "source_hash": config["source_hash"], "data_manifest_hash": config["data_manifest_hash"],
            "pretrained_sha256": config["pretrained_sha256"], "network": network,
            "train": train, "train_dataset": config["datasets"]["train"], "runtime": runtime}


def validate_stage_gate(paths, stage, current_source=None):
    """Read metadata only. No held-out image is opened before every run completes."""
    if stage not in STAGE_SEEDS:
        raise ValueError("Evaluation requires screen or confirm stage")
    contracts, configs, checkpoints = [], {}, {}
    for seed in STAGE_SEEDS[stage]:
        for method in ("B0", "B1", "B3"):
            run = Path(paths["output_root"]) / stage / method / f"seed{seed}"
            config = read_json(run / "effective_config.json")
            raw = dict(config)
            stored_hash = raw.pop("config_hash")
            if digest(raw) != stored_hash:
                raise ValueError(f"Changed effective config: {run}")
            status = read_json(run / "status.json")
            latest = read_json(run / "checkpoints/latest.json")
            budget = int(config["train"]["total_iter"])
            if status.get("status") != "completed" or status.get("update") != budget:
                raise RuntimeError(f"Evaluation gate: unfinished {run}")
            if budget != paths["budgets"][stage] or config["seed"] != seed or config["exp"] != method:
                raise ValueError(f"Incorrect stage/seed/budget: {run}")
            if config["stage"] != stage or (current_source and config["source_hash"] != current_source):
                raise ValueError(f"Changed stage/source: {run}")
            checkpoint_path = Path(latest["path"])
            if not checkpoint_path.is_file() or int(latest["update"]) != budget:
                raise RuntimeError(f"Missing final checkpoint: {run}")
            if checkpoint_path.resolve().parent != (run / "checkpoints").resolve():
                raise ValueError(f"Checkpoint outside assigned run: {run}")
            if sha256_file(checkpoint_path) != latest["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {run}")
            contracts.append(common_training_contract(config))
            configs[(method, seed)] = config
            checkpoints[(method, seed)] = checkpoint_path
    if len({digest(contract) for contract in contracts}) != 1:
        raise ValueError("Unpaired C0/source/data/model/training configs")
    for seed in STAGE_SEEDS[stage]:
        paired = [configs[(method, seed)].get("paired_protocol_hash") for method in ("B0", "B1", "B3")]
        if len(set(paired)) != 1:
            raise ValueError(f"Unpaired within-seed stochastic protocol: seed{seed}")
    contract = contracts[0]
    if not contract["new_optimizer"] or sha256_file(resolve_c0(paths)) != contract["c0_sha256"]:
        raise ValueError("C0 provenance/new optimizer contract mismatch")
    if sha256_file(paths["data_manifest"]) != contract["data_manifest_hash"]:
        raise ValueError("Data manifest changed")
    if sha256_file(paths["pretrained_rn50"]) != contract["pretrained_sha256"]:
        raise ValueError("RN50 changed")
    return configs, checkpoints, contract


def visual_key(image_id, family, level):
    return hashlib.sha256(f"{image_id}|{family}|{level:g}".encode()).hexdigest()[:16]


def save_fixed_visual(directory, clean, noisy, restored):
    side = min(96, *clean.shape[:2])
    y, x = (clean.shape[0] - side) // 2, (clean.shape[1] - side) // 2
    for name, array in (("clean", clean), ("noisy", noisy), ("restored", restored),
                        ("error", np.abs(clean - restored).mean(axis=2) / 0.20)):
        save_rgb(directory / f"{name}.png", array)
        save_rgb(directory / f"{name}_crop.png", array[y:y + side, x:x + side])
    atomic_json(directory / "crop.json", {"x": x, "y": y, "size": side, "absolute_error_scale": [0, 0.20]})


def evaluate_method(method, training_seed, paths, stage, splits, evaluation_root, device, gate, limit=None):
    configs, checkpoint_paths, contract = gate
    output = evaluation_root / stage / method / ("seedC0" if method == "C0" else f"seed{training_seed}")
    path = resolve_c0(paths) if method == "C0" else checkpoint_paths[(method, training_seed)]
    config = None if method == "C0" else configs[(method, training_seed)]
    checkpoint = torch_load(path)
    if config is not None:
        for key in ("config_hash", "source_hash", "seed", "stage"):
            if checkpoint.get(key) != config[key]:
                raise ValueError(f"Checkpoint/config {key} mismatch: {path}")
        if checkpoint.get("experiment") != method or checkpoint.get("update") != config["train"]["total_iter"]:
            raise ValueError(f"Incorrect final checkpoint: {path}")
        if checkpoint.get("pfa_rng") is None or not checkpoint.get("complete_update_boundary"):
            raise ValueError("Incomplete training recovery checkpoint")
    identity = {"path": str(path.resolve()), "sha256": sha256_file(path),
                "update": checkpoint.get("update"), "method": method,
                "training_seed": "C0" if method == "C0" else training_seed,
                "config_hash": config["config_hash"] if config else None}
    completion = output / "summary.json"
    if completion.exists():
        old = read_json(completion)
        if old.get("completed") and old.get("identity") == identity and old.get("splits") == splits:
            if old.get("protocol_hash") == digest(PROTOCOL) and old.get("contract_hash") == digest(contract):
                if not old.get("diagnostic") and not limit:
                    print(f"Already evaluated: {output}", flush=True)
                    return
        raise FileExistsError(f"Incompatible evaluation output: {output}")
    model = CLIPDenoising(num_blocks=[3, 4, 6, 3], depth=5, wf=64, bias=False,
                          model_path=paths["pretrained_rn50"], aug_level=0.025)
    model.load_state_dict(checkpoint["params"], strict=True)
    del checkpoint
    model = model.to(device).eval()
    # Record the gate before opening any benchmark image.
    atomic_json(output / "access_gate.json", {"timestamp": utc_now(), "stage": stage,
                 "all_runs_complete": sorted(f"{m}/seed{s}" for m, s in configs),
                 "contract": contract, "source_hash": source_hash()})
    rows, visual_rows, image_manifest = [], [], []
    for split in splits:
        dataset = SPLIT_DATASET[split]
        files = fixed_split(split, paths)
        if limit:
            files = files[:limit]
        dataset_dir = Path(paths["test_datasets"]["Kodak24" if split == "kodak" else "CBSD68"])
        for image_index, file in enumerate(files):
            image_id = file.relative_to(dataset_dir).as_posix()
            clean = read_rgb(file)
            clean_hash = sha256_file(file)
            image_manifest.append({"dataset": dataset, "image": image_id, "sha256": clean_hash})
            for family, level in CONDITIONS:
                for noise_seed in (0, 1, 2):
                    noisy = noisy_input(clean, family, level, noise_seed)
                    tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(device)
                    h, w = tensor.shape[-2:]
                    tensor = F.pad(tensor, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
                    with torch.inference_mode():
                        restored = model(tensor)[..., :h, :w].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                    row = {"method": method, "training_seed": identity["training_seed"], "dataset": dataset,
                           "image": image_id, "family": family, "level": level, "noise_seed": noise_seed,
                           "input_sha256": hashlib.sha256(noisy.tobytes()).hexdigest(),
                           "clean_sha256": clean_hash, **metrics(clean, restored)}
                    rows.append(row)
                    if image_index < 2 and noise_seed == 0 and (family, level) in VISUAL_CONDITIONS:
                        relative = Path("visuals") / dataset / visual_key(image_id, family, level)
                        save_fixed_visual(output / relative, clean, noisy, restored)
                        visual_rows.append(dict(row, directory=relative.as_posix()))
            print(f"{stage} {method} seed{identity['training_seed']} {dataset}: {image_index + 1}/{len(files)}", flush=True)
    write_csv(output / "per_image.csv", rows)
    write_csv(output / "visual_manifest.csv", visual_rows)
    atomic_json(output / "summary.json", {"completed": True, "completed_at": utc_now(), "stage": stage,
                "identity": identity, "splits": splits, "rows": len(rows), "images": image_manifest,
                "data_hash": digest(image_manifest), "protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL),
                "contract_hash": digest(contract), "contract": contract, "diagnostic": bool(limit)})
    print(f"Evaluation saved to {output}", flush=True)


def main(proposal='A'):
    parser = argparse.ArgumentParser(description="Shared A/C evaluation; C0 and paired B0 evaluated once")
    parser.add_argument("--paths", default="configs/plan_a.yaml")
    parser.add_argument("--stage", choices=tuple(STAGE_SEEDS), required=True)
    parser.add_argument("--training-seed", type=int, default=100)
    parser.add_argument("--method", nargs="+", choices=("C0", "B0", "B1", "B3"),
                        default=["C0", "B0", "B3" if proposal == "C" else "B1"])
    parser.add_argument("--splits", nargs="+", choices=("dev", "holdout", "kodak"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, help="Diagnostic subset only, excluded from formal summary")
    parser.add_argument("--output", type=Path, help="Evaluation root; stage/method/seed subdirectories are appended")
    args = parser.parse_args()
    expected_splits = ["dev"] if args.stage == "screen" else ["holdout", "kodak"]
    splits = args.splits or expected_splits
    if splits != expected_splits:
        parser.error(f"Stage {args.stage} requires exactly --splits {' '.join(expected_splits)}")
    if set(args.method) != {"C0"} and args.training_seed not in STAGE_SEEDS[args.stage]:
        parser.error("Training seed does not belong to stage; noise seeds are separately fixed to 0/1/2")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if len(args.method) != len(set(args.method)):
        parser.error("Duplicate methods are not allowed")
    paths = load_paths(args.paths)
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    gate = validate_stage_gate(paths, args.stage, source_hash())
    output = args.output or ROOT / "reports/ac/evaluations"
    for method in args.method:
        evaluate_method(method, args.training_seed, paths, args.stage, splits, output, args.device, gate, args.limit)


if __name__ == "__main__":
    main()
