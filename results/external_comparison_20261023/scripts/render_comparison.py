"""Render real denoising outputs only; no simulated panels or missing placeholders.

Usage: python scripts/render_comparison.py --root <comparison-directory>
Contract: four fixed images, seven columns, common 96-pixel center ROI, full-image
PSNR/SSIM labels, B0/G1 seed 0, and public pretrained external checkpoints.
The figure is a qualitative comparison, not a claim of universal superiority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


WIDTH_MM = 183.0
DPI = 600
COLS = ("Noisy", "DnCNN", "Restormer", "MaskDenoising", "B0", "G1", "GT")
DISPLAY = ("Noisy", "DnCNN [1]", "Restormer [3]", "MaskDenoising [10]", "B0", "Ours (G1)", "GT")
EXTERNAL = ("DnCNN", "Restormer", "MaskDenoising")

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.4,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_from_zoom(pixels: np.ndarray, size: int = 96) -> np.ndarray:
    """Recover unaltered ROI pixels from a 1x or integer nearest-neighbor zoom."""
    height, width = pixels.shape[:2]
    if height != width or height % size:
        raise ValueError(f"Expected square nearest-neighbor zoom of {size} pixels; got {pixels.shape}.")
    scale = height // size
    original = pixels[::scale, ::scale]
    if not np.array_equal(np.repeat(np.repeat(original, scale, 0), scale, 1), pixels):
        raise ValueError("Zoom image is not an exact nearest-neighbor expansion; image adjustments are prohibited.")
    return original


def noise_label(row: dict) -> str:
    level = float(row["noise_level"])
    labels = {
        "gaussian": f"Gaussian σ = {level:g}",
        "spatial_gaussian": f"Spatial Gauss σ = {level:g}",
        "poisson": f"Poisson α = {level:.1f}",
        "speckle": f"Speckle σ² = {level:g}",
    }
    if row["noise_type"] not in labels:
        raise ValueError(f"Unknown noise type: {row['noise_type']}")
    return labels[row["noise_type"]]


def short_image_name(row: dict) -> str:
    name = Path(row["image_id"]).stem
    return name.removesuffix("_SRF_2_HR")


def required_inputs(root: Path, rows: list[dict]) -> list[Path]:
    required = []
    for row in rows:
        for method in ("B0", "G1"):
            base = root / "existing_assets" / row["id"] / method
            required.extend(base / name for name in (
                "restored_zoom.png", "clean_zoom.png", "noisy_display_clipped_zoom.png", "display_protocol.json"
            ))
        for method in EXTERNAL:
            base = root / "inference" / method / row["id"]
            required.extend(base / name for name in ("restored.png", "restored_zoom.png", "metrics.json"))
    return required


def load_panels(root: Path):
    source_rows = root / "selected_rows.json"
    if not source_rows.is_file():
        raise FileNotFoundError(f"Required source metadata missing: {source_rows}")
    rows = read_json(source_rows)
    if [row["id"] for row in rows] != ["row1", "row2", "row3", "row4"]:
        raise ValueError("Expected exactly the four fixed rows row1 through row4; no silent selection is permitted.")
    required = required_inputs(root, rows)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot render incomplete comparison. Missing required inputs:\n" + "\n".join(missing))
    scores = []
    panels = []
    manifest = []
    for row in rows:
        existing = root / "existing_assets" / row["id"]
        protocol = read_json(existing / "G1/display_protocol.json")
        crop = {key: int(protocol[key]) for key in ("x", "y", "width", "height")}
        if crop["width"] != 96 or crop["height"] != 96:
            raise ValueError(f"{row['id']}: expected the recorded 96×96 ROI.")
        old_protocol = read_json(existing / "B0/display_protocol.json")
        if any(int(old_protocol[key]) != crop[key] for key in crop):
            raise ValueError(f"{row['id']}: B0 and G1 crop coordinates differ.")
        if any(str(row["existing"][method]["train_seed"]) != "0" for method in ("B0", "G1")):
            raise ValueError("The caption specifies seed 0; metadata contains a different training seed.")
        if row["existing"]["B0"]["noise_sha256"] != row["existing"]["G1"]["noise_sha256"]:
            raise ValueError(f"{row['id']}: B0 and G1 used different noisy inputs.")
        for filename in ("clean_zoom.png", "noisy_display_clipped_zoom.png"):
            if not np.array_equal(read_rgb(existing / "B0" / filename), read_rgb(existing / "G1" / filename)):
                raise ValueError(f"{row['id']}: common {filename} differs between existing methods.")
        paths = {
            "Noisy": existing / "G1/noisy_display_clipped_zoom.png",
            "GT": existing / "G1/clean_zoom.png",
            "B0": existing / "B0/restored_zoom.png",
            "G1": existing / "G1/restored_zoom.png",
        }
        metrics = {}
        for method in ("B0", "G1"):
            metrics[method] = row["existing"][method]
        for method in EXTERNAL:
            base = root / "inference" / method / row["id"]
            paths[method] = base / "restored_zoom.png"
            metrics[method] = read_json(base / "metrics.json")
            full = read_rgb(base / "restored.png")
            roi = full[crop["y"]:crop["y"] + 96, crop["x"]:crop["x"] + 96]
            if not np.array_equal(roi, crop_from_zoom(read_rgb(paths[method]))):
                raise ValueError(f"{row['id']}/{method}: zoom does not match the recorded ROI in restored.png.")
        pictures = {method: crop_from_zoom(read_rgb(paths[method])) for method in COLS}
        formatted = {}
        for method in ("DnCNN", "Restormer", "MaskDenoising", "B0", "G1"):
            psnr, ssim = float(metrics[method]["psnr"]), float(metrics[method]["ssim"])
            if not math.isfinite(psnr) or not math.isfinite(ssim):
                raise ValueError(f"Non-finite score in {row['id']}/{method}.")
            formatted[method] = f"{psnr:.2f}/{ssim:.3f}"
            scores.append({
                "row": row["id"], "dataset": row["dataset"], "image_id": row["image_id"],
                "noise_type": row["noise_type"], "noise_level": row["noise_level"],
                "method": method, "psnr": psnr, "ssim": ssim,
                "metric_scope": "full evaluated image", "roi_x": crop["x"], "roi_y": crop["y"],
                "roi_width": 96, "roi_height": 96,
                "checkpoint_replicate": "train_seed_0" if method in ("B0", "G1") else "public_checkpoint",
            })
        for method in COLS:
            manifest.append({
                "row": row["id"], "method": method,
                "source_image": str(paths[method].relative_to(root)),
                "sha256": digest(paths[method]), "crop": crop,
                "display_adjustments": "none; common nearest-neighbor rendering",
            })
        panels.append({"row": row, "images": pictures, "scores": formatted})
    return panels, scores, manifest


def create_figure(panels: list[dict]):
    margin_mm, gap_mm = 1.5, 0.7
    tile_mm = (WIDTH_MM - 2 * margin_mm - 6 * gap_mm) / 7
    header_mm, row_extra_mm, footer_mm = 5.0, 8.1, 4.0
    row_height_mm = tile_mm + row_extra_mm
    height_mm = header_mm + len(panels) * row_height_mm + footer_mm
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, height_mm / 25.4), dpi=DPI)
    text_boxes = []
    for column, label in enumerate(DISPLAY):
        left = margin_mm + column * (tile_mm + gap_mm)
        text = fig.text((left + tile_mm / 2) / WIDTH_MM, 1 - 2.2 / height_mm,
                        label, ha="center", va="center", fontsize=6.0, fontweight="bold")
        text_boxes.append((text, tile_mm, "column_header"))
    for index, panel in enumerate(panels):
        row = panel["row"]
        top_mm = header_mm + index * row_height_mm
        for column, method in enumerate(COLS):
            left_mm = margin_mm + column * (tile_mm + gap_mm)
            axes = fig.add_axes([left_mm / WIDTH_MM, 1 - (top_mm + tile_mm) / height_mm,
                                 tile_mm / WIDTH_MM, tile_mm / height_mm])
            axes.imshow(panel["images"][method], interpolation="nearest", vmin=0, vmax=255)
            axes.set_axis_off()
            center_x = (left_mm + tile_mm / 2) / WIDTH_MM
            if method in panel["scores"]:
                text = fig.text(center_x, 1 - (top_mm + tile_mm + 1.0) / height_mm,
                                panel["scores"][method], ha="center", va="top", fontsize=5.8)
                text_boxes.append((text, tile_mm, f"{row['id']}_{method}_score"))
            elif method == "Noisy":
                line1 = f"{row['dataset']}, {short_image_name(row)}"
                line2 = noise_label(row)
                text = fig.text(center_x, 1 - (top_mm + tile_mm + 0.8) / height_mm,
                                line1 + "\n" + line2, ha="center", va="top",
                                fontsize=5.2, linespacing=1.10)
                text_boxes.append((text, tile_mm + gap_mm, f"{row['id']}_identity"))
    fig.text(0.5, 1.5 / height_mm,
             "Full-image PSNR (dB)/SSIM; B0 and Ours: seed 0; fixed 96 × 96 crops",
             ha="center", va="center", fontsize=5.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text, available_mm, identity in text_boxes:
        width_mm = text.get_window_extent(renderer).width / fig.dpi * 25.4
        if width_mm > available_mm:
            raise ValueError(f"Text exceeds column: {identity} ({width_mm:.2f} > {available_mm:.2f} mm).")
    return fig, height_mm


def caption_text() -> str:
    return (
        "Qualitative comparison of available denoising checkpoints on four fixed benchmark examples. "
        "Columns show the noisy input, DnCNN [1], Restormer [3], MaskDenoising [10], B0, Ours (G1), "
        "and ground truth. Every method receives the same original floating-point noisy image within "
        "a row. All displayed panels use the same recorded 96 × 96 center crop, with nearest-neighbor "
        "magnification and no brightness, contrast, gamma, or local image adjustment. Noisy panels "
        "are clipped to the display range only; this does not clip model inputs. Labels report PSNR "
        "(dB)/SSIM for each whole evaluated image, not for the displayed crop. B0 and G1 use their "
        "300k-step training seed 0 checkpoints; these are individual examples, not three-seed means. "
        "The external methods use released pretrained checkpoints, whose training regimes may differ "
        "from the exact checkpoints underlying Table I. This is a newly evaluated visual supplement, "
        "not a reconstruction of the published table scores. The Speckle row retains the earlier "
        "Gaussian multiplicative-noise extension (variance 0.03), which differs from MATLAB's uniform "
        "multiplicative-noise protocol. MaskDenoising retains its published stochastic attention masking "
        "at inference with the per-image mask seed recorded in the inference metrics. HAT [11] and "
        "DIL [12] are omitted because matching weights were unavailable; no substitute or fabricated "
        "result panels are included. The rows were inherited from the existing report rather than "
        "selected according to the new methods' scores. No ranking, significance test, error bars, "
        "or cross-seed uncertainty is inferred from these individual images.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    panels, scores, manifest = load_panels(root)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure, height_mm = create_figure(panels)
    stem = output / "comparison_available_methods"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI)
    figure.savefig(stem.with_suffix(".pdf"), dpi=DPI)
    figure.savefig(stem.with_suffix(".svg"), dpi=DPI)
    plt.close(figure)
    for panel in panels:
        figure, _ = create_figure([panel])
        figure.savefig(output / f"comparison_{panel['row']['id']}.png", dpi=DPI)
        plt.close(figure)
    with (output / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    (output / "comparison_caption.txt").write_text(caption_text(), encoding="utf-8")
    (output / "comparison_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "render_checks.json").write_text(json.dumps({
        "width_mm": WIDTH_MM, "height_mm": height_mm, "dpi": DPI,
        "row_count": 4, "column_count": 7, "scored_outputs": len(scores),
        "before_external_methods": 5, "after_external_methods": 3,
        "excluded_methods": ["HAT [11]", "DIL [12]"],
        "exclusion_reason": "matching pretrained weights unavailable",
        "shared_crop_check": "passed", "nearest_zoom_integrity_check": "passed",
        "required_inputs_check": "passed", "label_column_width_check": "passed",
        "minimum_source_font_pt": 5.2,
        "source_data": "comparison_metrics.csv",
        "image_provenance": "comparison_sources.json",
        "uncertainty": "none: individual image results, no averaging",
        "visual_review": "required after rendering; automated checks do not replace visual inspection",
        "accepted_source_warning": "PNG preview plus PDF/SVG requested; no TIFF journal-submission export requested",
    }, indent=2), encoding="utf-8")
    print(f"Rendered real-data comparison: {stem}.png ({WIDTH_MM:g} × {height_mm:.2f} mm)")


if __name__ == "__main__":
    main()
