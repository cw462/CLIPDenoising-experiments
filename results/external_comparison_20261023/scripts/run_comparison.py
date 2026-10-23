"""Infer public denoisers on the exact saved noisy arrays of the report examples."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "runtime_deps"))
sys.path.insert(0, str(TASK / "scripts"))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(method, device):
    if method == "DnCNN":
        from dncnn_adapter import load_dncnn
        model = load_dncnn(TASK / "external/DnCNN/color_sigma=15.mat", device=device)
    elif method == "Restormer":
        module = import_file("public_restormer", TASK / "external/Restormer/restormer_arch.py")
        model = module.Restormer(inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)
        weights = torch.load(TASK / "external/Restormer/gaussian_color_denoising_sigma15.pth",
                             map_location="cpu", weights_only=True)
        model.load_state_dict(weights["params"], strict=True)
        model = model.to(device).eval().requires_grad_(False)
    elif method == "MaskDenoising":
        from masked_adapter import load_model as load_masked
        model = load_masked(device=device)
    else:
        raise ValueError(method)
    return model


def save_png(path, array):
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def run(method, device, repository):
    sys.path.insert(0, str(repository))
    from tools.evaluation_utils import calculate_metrics

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(method, device)
    rows = json.loads((TASK / "selected_rows.json").read_text(encoding="utf-8"))
    result_rows = []
    for index, row in enumerate(rows):
        base = row["existing"]["B0"]
        saved = Path(base["visual_directory"])
        with np.load(saved / "raw_arrays.npz", allow_pickle=False) as arrays:
            noisy = np.ascontiguousarray(arrays["noisy"], dtype=np.float32)
        digest = hashlib.sha256(noisy.tobytes()).hexdigest()
        if digest != base["noise_sha256"] or digest != row["existing"]["G1"]["noise_sha256"]:
            raise ValueError("Saved noisy input does not match the selected report example")
        clean = np.asarray(Image.open(saved / "clean.png").convert("RGB"), dtype=np.float32) / 255.0
        crop = json.loads((saved / "display_protocol.json").read_text())
        tensor = torch.from_numpy(noisy.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        height, width = noisy.shape[:2]
        if method == "Restormer":
            tensor = F.pad(tensor, (0, (-width) % 8, 0, (-height) % 8), mode="reflect")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor, mask_seed=index) if method == "MaskDenoising" else model(tensor)
            output = output[..., :height, :width].clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        restored = output[0].permute(1, 2, 0).cpu().numpy()
        if restored.shape != clean.shape or not np.isfinite(restored).all():
            raise ValueError("Restored image shape or values are invalid")
        y, x, h, w = (crop[key] for key in ("y", "x", "height", "width"))
        metrics = calculate_metrics(clean, restored)
        crop_metrics = calculate_metrics(clean[y:y+h, x:x+w], restored[y:y+h, x:x+w])
        metrics.update(crop_psnr=crop_metrics["psnr"], crop_ssim=crop_metrics["ssim"],
            method=method, row_id=row["id"], dataset=row["dataset"], image_id=row["image_id"],
            noise_type=row["noise_type"], noise_level=row["noise_level"], noise_sha256=digest,
            mask_seed=index if method == "MaskDenoising" else None,
            forward_seconds=seconds, crop=crop,
            metric_scope="Full image RGB PSNR/SSIM from float output, border=0; crop metrics recorded separately",
            source_weights="Official public RGB Gaussian sigma15 checkpoint",
            input_protocol="Exact saved report noisy float32 array; input not clipped or regenerated",
            speckle_note="The report uses Gaussian multiplicative Speckle, no clipping; this is its existing extension protocol, not MATLAB imnoise",
            training_performed=False)
        destination = TASK / "inference" / method / row["id"]
        destination.mkdir(parents=True, exist_ok=True)
        save_png(destination / "restored.png", restored)
        zoom = Image.fromarray(np.rint(restored[y:y+h, x:x+w] * 255).astype(np.uint8))
        zoom.resize((w*3, h*3), resample=Image.Resampling.NEAREST).save(destination / "restored_zoom.png")
        np.save(destination / "restored.npy", restored, allow_pickle=False)
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        result_rows.append(metrics)
        print(json.dumps({k: metrics[k] for k in ("method", "row_id", "psnr", "ssim", "forward_seconds")}), flush=True)
    summary = {"completed": len(result_rows) == 4, "method": method, "images": len(result_rows),
               "training_performed": False, "results": result_rows,
               "torch": torch.__version__, "cuda": torch.version.cuda}
    (TASK / "inference" / method / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("DnCNN", "Restormer", "MaskDenoising"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    run(args.method, torch.device(args.device), args.repository)
