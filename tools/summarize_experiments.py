"""Summarize completed evaluations; missing experiments never become zero scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_common import load_paths, read_json, atomic_json, sha256_file
from tools.evaluation_utils import FULL_QUEUE, PROTOCOL, read_csv, write_csv


def aggregate(rows: list[dict], keys: tuple[str, ...], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, members in sorted(groups.items()):
        output = dict(zip(keys, values))
        output["n"] = len(members)
        for field in fields:
            output[field + "_mean"] = statistics.mean(float(row[field]) for row in members)
            output[field + "_std"] = statistics.stdev(float(row[field]) for row in members) if len(members) > 1 else ""
        result.append(output)
    return result


def completed_evaluations(output_root: Path) -> tuple[list[dict], list[dict], list[str]]:
    selected, issues = {}, []
    for file in sorted((output_root / "full").glob("*/seed*/evaluations/benchmark_*/manifest.json")):
        manifest = read_json(file)
        if manifest.get("status") != "completed" or manifest.get("development"):
            continue
        csv_file = file.parent / "per_image_results.csv"
        if not csv_file.is_file() or sha256_file(csv_file) != manifest.get("metrics_csv_sha256"):
            issues.append(f"Corrupt metrics file: {csv_file}")
            continue
        if int(manifest.get("update", -1)) != 300000:
            issues.append(f"Non-final checkpoint rejected: {file}")
            continue
        key = (manifest["experiment"], int(manifest["train_seed"]))
        # Lexicographic UTC timestamp directories make the latest complete re-evaluation explicit.
        selected[key] = file, manifest
    per_image, runtime = [], []
    for key in FULL_QUEUE:
        if key not in selected:
            issues.append(f"Missing completed benchmark: {key[0]} seed{key[1]}")
            continue
        file, manifest = selected[key]
        per_image.extend(read_csv(file.parent / "per_image_results.csv"))
        runtime.extend(read_csv(file.parent / "runtime.csv"))
    return per_image, runtime, issues


def paired_differences(rows: list[dict]) -> list[dict]:
    key_fields = ("train_seed", "dataset", "image_id", "noise_type", "noise_level")
    baselines = {tuple(row[key] for key in key_fields): row for row in rows if row["experiment"] == "B0"}
    result = []
    for row in rows:
        if row["experiment"] == "B0":
            continue
        key = tuple(row[field] for field in key_fields)
        base = baselines.get(key)
        if base is None:
            continue
        for identity in ("noise_sha256", "dataset_manifest_hash", "source_hash"):
            if row[identity] != base[identity]:
                raise ValueError(f"Paired {identity} mismatch: {key}")
        result.append({
            "experiment": row["experiment"], **dict(zip(key_fields, key)),
            "protocol_group": row["protocol_group"], "domain": row["domain"],
            "delta_psnr": float(row["psnr"]) - float(base["psnr"]),
            "delta_ssim": float(row["ssim"]) - float(base["ssim"]),
            "psnr": row["psnr"], "baseline_psnr": base["psnr"],
            "ssim": row["ssim"], "baseline_ssim": base["ssim"],
            "checkpoint": row["checkpoint"], "baseline_checkpoint": base["checkpoint"],
            "noise_sha256": row["noise_sha256"],
            "visual_directory": row.get("visual_directory", ""),
            "baseline_visual_directory": base.get("visual_directory", ""),
        })
    return result


def hierarchical_ood(condition_rows: list[dict]) -> list[dict]:
    """Images -> strengths -> families -> datasets, each level equally weighted."""
    result = []
    for table in ("official_three_families", "five_families_with_extensions"):
        selected = [row for row in condition_rows
                    if not (row["noise_type"] == "gaussian" and float(row["noise_level"]) == 15)
                    and (table == "five_families_with_extensions" or row["protocol_group"] == "official")]
        families = aggregate(selected, ("experiment", "train_seed", "dataset", "noise_type"),
                             ("psnr_mean", "ssim_mean"))
        datasets = aggregate(families, ("experiment", "train_seed", "dataset"),
                             ("psnr_mean_mean", "ssim_mean_mean"))
        experiments = aggregate(datasets, ("experiment", "train_seed"),
                                ("psnr_mean_mean_mean", "ssim_mean_mean_mean"))
        for row in experiments:
            result.append({"experiment": row["experiment"], "train_seed": row["train_seed"],
                           "table": table, "dataset_count": row["n"],
                           "ood_psnr": row["psnr_mean_mean_mean_mean"],
                           "ood_ssim": row["ssim_mean_mean_mean_mean"]})
    return result


def comparison_panels(rows: list[dict], report_dir: Path) -> int:
    """Assemble already selected images; retain identical coordinates and scales."""
    from PIL import Image, ImageDraw
    baselines = {
        (r["train_seed"], r["dataset"], r["image_id"], r["noise_type"], r["noise_level"]): r
        for r in rows if r["experiment"] == "B0" and r.get("visual_directory")
    }
    count = 0
    for row in rows:
        if row["experiment"] != "G1" or row["train_seed"] != "0" or not row.get("visual_directory"):
            continue
        key = (row["train_seed"], row["dataset"], row["image_id"], row["noise_type"], row["noise_level"])
        base = baselines.get(key)
        if base is None:
            continue
        bdir, gdir = Path(base["visual_directory"]), Path(row["visual_directory"])
        items = [
            ("Clean", gdir / "clean_zoom.png"),
            ("Noisy (display clipped)", gdir / "noisy_display_clipped_zoom.png"),
            ("B0", bdir / "restored_zoom.png"),
            ("G1", gdir / "restored_zoom.png"),
            ("B0 abs.error [0,.20]", bdir / "absolute_error_zoom.png"),
            ("G1 abs.error [0,.20]", gdir / "absolute_error_zoom.png"),
        ]
        if not all(path.is_file() for _, path in items):
            continue
        images = [Image.open(path).convert("RGB") for _, path in items]
        width, height = images[0].size
        canvas = Image.new("RGB", (width * 3, (height + 30) * 2), "white")
        draw = ImageDraw.Draw(canvas)
        for i, ((label, _), image) in enumerate(zip(items, images)):
            x, y = (i % 3) * width, (i // 3) * (height + 30)
            draw.text((x + 5, y + 7), label, fill="black")
            canvas.paste(image, (x, y + 30))
            image.close()
        import hashlib
        stem = hashlib.sha256("/".join(key).encode()).hexdigest()[:16]
        destination = report_dir / "figures" / f"{row['dataset']}_{stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        count += 1
    return count


def summarize(paths: dict) -> dict:
    report_dir = Path(paths["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, runtime, issues = completed_evaluations(Path(paths["output_root"]))
    fields = ["experiment", "train_seed", "stage", "checkpoint", "update", "git_commit",
              "config_hash", "dataset", "image_id", "noise_type", "noise_level",
              "noise_seed", "psnr", "ssim", "forward_ms", "peak_memory_mb"]
    if not rows:
        for filename in ("main_results.csv", "per_image_results.csv",
                         "ablation_results.csv", "runtime_results.csv"):
            # Header-only files communicate the schema without inventing values.
            write_csv(report_dir / filename, [], fields)
        status = {"status": "blocked", "formal_results_available": False,
                  "reasons": issues, "protocol": PROTOCOL, "actual_metric_rows": 0}
        atomic_json(report_dir / "evaluation_summary.json", status)
        (report_dir / "evaluation_summary.md").write_text(
            "# Evaluation status\n\nNo completed 300000-update formal benchmark exists. "
            "No denoising scores, improvements, or multi-seed claims are available.\n\n"
            + "\n".join("- " + item for item in issues)
            + "\n\nThis status does not treat unit tests, synthetic checks, smoke runs, "
            "or pilot runs as final experimental results.\n", encoding="utf-8")
        return status
    write_csv(report_dir / "per_image_results.csv", rows)
    write_csv(report_dir / "runtime_results.csv", runtime)
    keys = ("experiment", "train_seed", "dataset", "noise_type", "noise_level",
            "protocol_group", "domain")
    condition_rows = aggregate(rows, keys, ("psnr", "ssim"))
    write_csv(report_dir / "condition_results.csv", condition_rows)
    main = aggregate(condition_rows,
                     ("experiment", "dataset", "noise_type", "noise_level", "protocol_group", "domain"),
                     ("psnr_mean", "ssim_mean"))
    for row in main:
        row["train_seed_count"] = row.pop("n")
        row["psnr"] = row.pop("psnr_mean_mean")
        row["ssim"] = row.pop("ssim_mean_mean")
        row["psnr_seed_std"] = row.pop("psnr_mean_std")
        row["ssim_seed_std"] = row.pop("ssim_mean_std")
        row["seed_limitation"] = "single seed; seed variability unavailable" if row["train_seed_count"] == 1 else ""
    write_csv(report_dir / "main_results.csv", [r for r in main if r["experiment"] in {"B0", "G1"}])
    paired = paired_differences(rows)
    write_csv(report_dir / "paired_per_image_results.csv", paired)
    delta_conditions = aggregate(paired, keys, ("delta_psnr", "delta_ssim"))
    write_csv(report_dir / "paired_condition_results.csv", delta_conditions)
    delta_seeds = aggregate(delta_conditions,
                            ("experiment", "dataset", "noise_type", "noise_level", "protocol_group", "domain"),
                            ("delta_psnr_mean", "delta_ssim_mean"))
    for row in delta_seeds:
        row["train_seed_count"] = row.pop("n")
        row["paired_delta_psnr"] = row.pop("delta_psnr_mean_mean")
        row["paired_delta_ssim"] = row.pop("delta_ssim_mean_mean")
        row["delta_psnr_seed_std"] = row.pop("delta_psnr_mean_std")
        row["delta_ssim_seed_std"] = row.pop("delta_ssim_mean_std")
    write_csv(report_dir / "paired_seed_summary.csv", delta_seeds)
    write_csv(report_dir / "ablation_results.csv", [r for r in delta_seeds if r["experiment"] in {"G2", "G3"}])
    ood = hierarchical_ood(condition_rows)
    write_csv(report_dir / "ood_aggregate_results.csv", ood)
    failures = sorted((row for row in paired if row["delta_psnr"] < 0), key=lambda r: r["delta_psnr"])
    write_csv(report_dir / "failure_cases.csv", failures)
    figures = comparison_panels(rows, report_dir)
    status = {"status": "completed" if not issues else "incomplete", "formal_results_available": True,
              "actual_metric_rows": len(rows), "reasons": issues, "protocol": PROTOCOL,
              "fixed_example_panels": figures, "negative_paired_cases": len(failures)}
    atomic_json(report_dir / "evaluation_summary.json", status)
    lines = [
        "# Fixed-checkpoint evaluation", "",
        f"Status: {status['status']}. Actual per-image measurements: {len(rows)}.",
        "All included checkpoints are at update 300000. B0/G1 seed variation is the "
        "sample standard deviation over seed-level condition means. G2/G3 have one seed; "
        "no multi-seed stability claim is justified.", "",
        "Every dataset/noise/strength condition is in condition_results.csv; "
        "paired_per_image_results.csv links each difference to matching image and noise hashes. "
        "Official three-family and extended five-family OOD aggregates are separate. "
        "Each OOD aggregate weights images, within-family strengths, noise families, and datasets equally "
        "at successive levels. Gaussian sigma15 is excluded.", "",
        "runtime_results.csv uses the same 256x256 batch1 input size, 3 warmups, and 10 timed forwards. "
        "Per-image file I/O, noise generation, inference, and visual saving are recorded separately. "
        "CPU runs have no CUDA peak-memory measurement.", "",
        f"Negative image/condition/seed comparisons retained: {len(failures)}. "
        "failure_cases.csv lists every negative paired difference, worst first. "
        f"Fixed-example comparison panels: {figures}. Display scales and crop coordinates are "
        "identical across B0/G1. Gate maps indicate modulation of a correction, not calibrated noise probability.", "",
        "The flat/texture/edge masks are post-inference heuristic GT-gradient bins; "
        "GT is never an input to the network. A visual pattern is a mechanism hypothesis, "
        "not proof of denoising causality.", "",
    ]
    if issues:
        lines += ["Missing or invalid work:", ""] + ["- " + issue for issue in issues] + [""]
    (report_dir / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default=str(ROOT / "configs/local_paths.yaml"))
    args = parser.parse_args(argv)
    status = summarize(load_paths(args.paths))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
