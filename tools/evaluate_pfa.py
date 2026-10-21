"""Proposal A's fixed seven-condition evaluation, with an explicit held-out split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
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
from tools.evaluation_utils import image_files, read_rgb, save_visuals, write_csv
from tools.experiment_common import atomic_json, sha256_file, torch_load

CONDITIONS = (("gaussian", 15), ("gaussian", 25), ("gaussian", 50),
              ("spatial_gaussian", 45), ("spatial_gaussian", 55),
              ("poisson", 2.5), ("poisson", 3.5))


def fixed_split(split):
    files = image_files(ROOT / "data/CBSD68")
    if len(files) != 68:
        raise ValueError("CBSD68 must contain exactly 68 images")
    for name, entries in (("dev", files[:20]), ("holdout", files[20:])):
        target = ROOT / f"splits/plan_a_{name}.txt"
        body = "".join(path.relative_to(ROOT).as_posix() + "\n" for path in entries)
        if target.exists() and target.read_text() != body:
            raise RuntimeError(f"Fixed evaluation split changed: {target}")
        if not target.exists():
            target.write_text(body, encoding="utf-8")
    if split == "kodak":
        files = image_files(ROOT / "data/Kodak24")
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
    return {"psnr": float(peak_signal_noise_ratio(clean, restored, data_range=1)),
            "ssim": float(structural_similarity(clean, restored, data_range=1, channel_axis=2,
                                                 gaussian_weights=True, sigma=1.5,
                                                 use_sample_covariance=False))}


def summarize(rows):
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
    if "B0" in methods and "B1" in methods:
        for category in ("id", "ood", "spatial"):
            differences = np.array([value - image_means[("B0", image, category)]
                                    for (method, image, c), value in image_means.items()
                                    if method == "B1" and c == category])
            samples = np.random.RandomState(2026).choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
            paired[category] = {"B1_minus_B0_db": float(differences.mean()),
                                "paired_image_bootstrap_95_ci": np.percentile(samples, [2.5, 97.5]).tolist(),
                                "independent_images": len(differences),
                                "scope": "Image uncertainty only; not training-seed variability"}
    return {"methods": summary, "paired": paired,
            "conditions": [{"method": m, "family": f, "level": l, "psnr": float(np.mean(v))}
                           for (m, f, l), v in conditions.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0", type=Path)
    parser.add_argument("--b1", type=Path)
    parser.add_argument("--split", choices=("dev", "holdout", "kodak"), default="dev")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, help="Diagnostic subset only; never full benchmark evidence")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if len(args.seeds) != len(set(args.seeds)) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("Noise seeds must be unique uint32 values")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(4)
    files = fixed_split(args.split)
    if args.limit:
        files = files[:args.limit]
    output = args.output or ROOT / f"reports/plan_a/evaluation_{args.split}"
    checkpoints = {"C0": ROOT / "pretrained/synthetic_baseline.pth"}
    checkpoints.update({name: path for name, path in (("B0", args.b0), ("B1", args.b1)) if path})
    rows, identities = [], {}
    for name, path in checkpoints.items():
        checkpoint = torch_load(path)
        if name != "C0":
            if checkpoint.get("experiment") != name or checkpoint.get("pfa_rng") is None:
                raise ValueError(f"Not a proposal A {name} recovery checkpoint: {path}")
            config = json.loads((path.parent.parent / "effective_config.json").read_text())
            if checkpoint["config_hash"] != config["config_hash"]:
                raise ValueError("Checkpoint/config mismatch")
            if checkpoint["update"] != config["train"]["total_iter"] and not args.allow_partial:
                raise ValueError("Incomplete training; use --allow-partial only for diagnostic evaluation")
        identities[name] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                            "stage": checkpoint.get("stage", "original C0"), "update": checkpoint.get("update")}
        model = CLIPDenoising(num_blocks=[3, 4, 6, 3], depth=5, wf=64, bias=False,
                              model_path=str(ROOT / "pretrained/RN50.pt"), aug_level=0.025)
        model.load_state_dict(checkpoint["params"], strict=True)
        del checkpoint
        model = model.to(args.device).eval()
        for image_index, file in enumerate(files):
            clean = read_rgb(file)
            for family, level in CONDITIONS:
                for seed in args.seeds:
                    noisy = noisy_input(clean, family, level, seed)
                    tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(args.device)
                    h, w = tensor.shape[-2:]
                    tensor = F.pad(tensor, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
                    with torch.inference_mode():
                        restored = model(tensor)[..., :h, :w].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                    rows.append({"method": name, "image": file.name, "family": family, "level": level,
                                 "seed": seed, "input_sha256": hashlib.sha256(noisy.tobytes()).hexdigest(),
                                 **metrics(clean, restored)})
                    if image_index == 0 and seed == args.seeds[0]:
                        save_visuals(output / "visuals" / name / f"{family}_{level:g}", clean, noisy, restored, None)
            print(f"{name}: {image_index + 1}/{len(files)}", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    write_csv(output / "per_image.csv", rows)
    report = summarize(rows)
    report.update(split=args.split, images=len(files), noise_seeds=args.seeds, checkpoints=identities,
                  protocol="RGB float, data_range=1, crop_border=0, output clamp; Gaussian-window RGB SSIM",
                  selection_disclosure="CBSD68 first 20 are OOD-assisted development; remaining 48 are held out",
                  diagnostic=bool(args.limit or args.allow_partial or any(v["stage"] == "smoke" for v in identities.values())))
    atomic_json(output / "summary.json", report)
    print(f"Evaluation saved to {output}")


if __name__ == "__main__":
    main()
