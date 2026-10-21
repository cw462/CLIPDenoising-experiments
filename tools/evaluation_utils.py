"""Fixed evaluation protocol; no benchmark images are read at import time."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from Denoising import utils as official


OFFICIAL_CONDITIONS = (
    ("gaussian", 15), ("gaussian", 25), ("gaussian", 50),
    ("spatial_gaussian", 45), ("spatial_gaussian", 50),
    ("spatial_gaussian", 55), ("poisson", 2.5),
    ("poisson", 3.0), ("poisson", 3.5),
)
EXTENSION_CONDITIONS = (
    ("speckle", 0.02), ("speckle", 0.03), ("speckle", 0.04),
    ("salt_pepper", 0.012), ("salt_pepper", 0.016),
    ("salt_pepper", 0.02),
)
CONDITIONS = OFFICIAL_CONDITIONS + EXTENSION_CONDITIONS
FULL_QUEUE = (("B0", 0), ("G1", 0), ("B0", 1), ("G1", 1),
              ("B0", 2), ("G1", 2), ("G2", 0), ("G3", 0))
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
VISUAL_CONDITIONS = {
    ("gaussian", 15.0), ("gaussian", 50.0),
    ("spatial_gaussian", 50.0), ("poisson", 3.0),
    ("speckle", 0.03), ("salt_pepper", 0.016),
}
PROTOCOL = {
    "version": "clipdenoising-local-v1",
    "official_source": "Denoising/utils.py and Denoising/test_synthetic_denoising.py",
    "official_noise": "np.random.seed(0) separately before every image/condition",
    "spatial_gaussian": "per-channel scipy.ndimage.convolve, 3x3 mean, mode=reflect",
    "poisson": "official.add_poisson_noise(scale=level), clip=True, rounds=False",
    "extension_noise": "stable SHA256(dataset/image_id/noise_type/level) uint32 seed",
    "extension_status": "Project extension, not a claim of exact paper reproduction",
    "speckle": "x + x * N(0, variance), no clipping",
    "salt_pepper": "one spatial uniform draw, all RGB channels jointly 0 or 1",
    "input": "RGB /255; Gaussian/spatial/speckle not clipped; no CLIP normalization",
    "inference": "full-resolution batch1, right/bottom reflect-pad to multiple of 32",
    "metrics": "crop original size, clamp output [0,1], official RGB float *255 PSNR/SSIM, border0",
    "aggregation": "images equal; strengths equal within family; families equal; datasets equal",
    "ood": "Exclude Gaussian sigma15; official three-family and five-family extension scores separate",
    "visual_selection": "first two lexicographically sorted images per dataset, six fixed conditions",
    "error_color_scale": "mean absolute RGB error [0,0.20], shared fixed grayscale",
    "signed_correction_scale": "[-0.10,+0.10], shared fixed display; raw arrays preserved",
    "runtime": "256x256 seeded tensor, batch1, 3 warmups, 10 forwards; synchronized CUDA",
}


def stable_seed(dataset: str, image_id: str, noise_type: str, level: float) -> int:
    key = f"{dataset}/{image_id}/{noise_type}/{level:g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def make_noise(clean: np.ndarray, noise_type: str, level: float,
               dataset: str, image_id: str) -> tuple[np.ndarray, int, str]:
    """Match the official RNG exactly while leaving the caller RNG unchanged."""
    extension = noise_type in {"speckle", "salt_pepper"}
    seed = stable_seed(dataset, image_id, noise_type, level) if extension else 0
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        if noise_type == "gaussian":
            noisy = clean + np.random.normal(0, level / 255.0, clean.shape)
        elif noise_type == "spatial_gaussian":
            noise = np.random.normal(0, level / 255.0, clean.shape)
            kernel = np.ones((3, 3)) / 9.0
            for channel in range(clean.shape[-1]):
                noise[..., channel] = convolve(noise[..., channel], kernel, mode="reflect")
            noisy = clean + noise
        elif noise_type == "poisson":
            noisy = official.add_poisson_noise(clean, scale=level)
        elif noise_type == "speckle":
            noisy = clean + clean * np.random.normal(0, math.sqrt(level), clean.shape)
        elif noise_type == "salt_pepper":
            noisy = clean.copy()
            draws = np.random.uniform(size=clean.shape[:2])
            noisy[draws < level / 2.0] = 0
            noisy[draws >= 1.0 - level / 2.0] = 1
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    finally:
        np.random.set_state(saved)
    # This is the tensor input precision used in the official script.
    noisy = np.ascontiguousarray(noisy, dtype=np.float32)
    digest = hashlib.sha256(noisy.tobytes()).hexdigest()
    return noisy, seed, digest


def image_files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: p.relative_to(directory).as_posix())


def read_rgb(path: Path) -> np.ndarray:
    image = official.load_img(str(path))
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Not an RGB image: {path}")
    if min(image.shape[:2]) < 32:
        raise ValueError(f"Image too small for official reflect-pad/SSIM: {path}")
    return np.asarray(image, dtype=np.float32) / 255.0


def calculate_metrics(clean: np.ndarray, restored: np.ndarray) -> dict:
    if not np.isfinite(restored).all():
        raise FloatingPointError("Non-finite restored image")
    psnr = float(official.calculate_psnr(clean * 255.0, restored * 255.0))
    ssim = float(official.calculate_ssim(clean * 255.0, restored * 255.0))
    if not math.isfinite(ssim):
        raise FloatingPointError("Non-finite SSIM")
    return {"psnr": psnr, "ssim": ssim}


def region_errors(clean: np.ndarray, restored: np.ndarray) -> dict:
    """GT is used only after inference, for explicitly heuristic offline masks."""
    gray = clean.mean(axis=2)
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    masks = {"flat": magnitude < 0.02,
             "texture": (magnitude >= 0.02) & (magnitude < 0.08),
             "edge": magnitude >= 0.08}
    error = np.abs(clean - restored).mean(axis=2)
    result = {}
    for name, mask in masks.items():
        result[name + "_pixels"] = int(mask.sum())
        result[name + "_mae"] = float(error[mask].mean()) if mask.any() else ""
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path, format="PNG")


def save_visuals(directory: Path, clean: np.ndarray, noisy: np.ndarray,
                 restored: np.ndarray, maps: dict | None) -> dict:
    """All models use identical coordinates and fixed scales, never auto-contrast."""
    directory.mkdir(parents=True, exist_ok=True)
    height, width = clean.shape[:2]
    side = min(96, height, width)
    y, x = (height - side) // 2, (width - side) // 2
    arrays = {"clean": clean, "noisy_display_clipped": noisy, "restored": restored,
              "absolute_error": np.abs(restored - clean).mean(axis=2) / 0.20}
    for name, array in arrays.items():
        save_rgb(directory / (name + ".png"), array)
        crop = array[y:y + side, x:x + side]
        pixels = np.rint(np.clip(crop, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).resize((side * 3, side * 3),
                                     resample=Image.Resampling.NEAREST).save(
                                         directory / (name + "_zoom.png"))
    raw = {"noisy": noisy, "restored": restored}
    if maps:
        for name, tensor in maps.items():
            array = tensor[0, :, :height, :width].cpu().numpy().transpose(1, 2, 0)
            raw[name] = array
            if name == "g":
                display = array[..., 0]
            else:
                display = 0.5 + array / 0.20
            save_rgb(directory / (name + ".png"), display)
    np.savez_compressed(directory / "raw_arrays.npz", **raw)
    coordinates = {"x": x, "y": y, "width": side, "height": side,
                   "error_scale": [0, 0.20], "signed_maps_scale": [-0.10, 0.10],
                   "warning": "Gate is correction modulation, not a noise probability",
                   "offline_regions": "GT grayscale gradient: flat <.02; texture [.02,.08); edge >=.08"}
    (directory / "display_protocol.json").write_text(
        json.dumps(coordinates, indent=2), encoding="utf-8")
    return coordinates
