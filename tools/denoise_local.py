"""Run the available upstream synthetic baseline on one real RGB image.

This is a local inference demonstration, not the archived G1/G2/G3 evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from tools.evaluation_utils import calculate_metrics, read_rgb, save_rgb
from tools.experiment_common import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Clean RGB image; default: first development-validation image")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained/synthetic_baseline.pth")
    parser.add_argument("--weights", type=Path, default=ROOT / "pretrained/RN50.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/local_demo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.sigma <= 0 or args.threads < 1:
        parser.error("--sigma and --threads must be positive")
    if args.input is None:
        entry = (ROOT / "splits/dev_val.txt").read_text().splitlines()[0]
        args.input = ROOT / entry
    for path in (args.input, args.checkpoint, args.weights):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    clean = read_rgb(args.input)
    noisy = clean + np.random.RandomState(args.seed).normal(0, args.sigma / 255, clean.shape).astype(np.float32)
    model = CLIPDenoising(
        num_blocks=[3, 4, 6, 3], inp_channels=3, out_channels=3,
        depth=5, wf=64, bias=False, model_path=str(args.weights),
        aug_level=0.025, adapter_mode="none",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)
    del checkpoint
    model = model.to(device).eval()
    tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    tensor = F.pad(tensor, (0, (-width) % 32, 0, (-height) % 32), mode="reflect")
    with torch.inference_mode():
        output = model(tensor)[..., :height, :width].clamp(0, 1)
    restored = output[0].permute(1, 2, 0).cpu().numpy()
    noisy_metrics = calculate_metrics(clean, noisy)
    restored_metrics = calculate_metrics(clean, restored)
    for name, array in (("clean", clean), ("noisy", noisy), ("restored", restored)):
        save_rgb(args.output / f"{name}.png", array)
    comparison = Image.new("RGB", (width * 3, height + 28), "white")
    draw = ImageDraw.Draw(comparison)
    for i, name in enumerate(("clean", "noisy", "restored")):
        with Image.open(args.output / f"{name}.png") as panel:
            comparison.paste(panel, (i * width, 28))
        draw.text((i * width + 8, 7), name, fill="black")
    comparison.save(args.output / "comparison.png")
    report = {
        "scope": "Single development image, upstream synthetic B0 baseline; not archived calibration results",
        "input": str(args.input.resolve()), "device": device,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "image_shape": list(clean.shape), "sigma": args.sigma, "seed": args.seed,
        "noisy": noisy_metrics, "restored": restored_metrics,
        "seconds": time.perf_counter() - start,
    }
    atomic_json(args.output / "metrics.json", report)
    print(f"Device: {device}; noisy PSNR: {noisy_metrics['psnr']:.3f} dB; restored PSNR: {restored_metrics['psnr']:.3f} dB")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
