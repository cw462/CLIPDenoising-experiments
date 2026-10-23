"""Summarize the paired three-seed, channel-independent salt-and-pepper rerun.

Uses only the Python standard library. Historical results are read, never edited.
Run after all three evaluation/seedN/per_image_results.csv files are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "scripts" and SCRIPT_PATH.parent.parent.name == "table1_channel_salt_3seed_20261022":
    DEFAULT_ROOT = SCRIPT_PATH.parent.parent
    PROJECT = DEFAULT_ROOT.parent.parent
else:
    PROJECT = Path(r"D:\projects\CLIPDenoising_G1C_Seeds_20261021")
    DEFAULT_ROOT = PROJECT / "results/table1_channel_salt_3seed_20261022"
DEFAULT_HISTORY = PROJECT / "results/g1c_standard_300k_20261022/reports/g1c_standard/final/same_seed_condition_summary.csv"
SEEDS = (0, 1, 2)
DATASETS = {"CBSD68": 68, "McMaster": 18, "Kodak24": 24, "Urban100": 100}
CONDITIONS = (
    ("gaussian", 50.0, "Gauss，σ=50"),
    ("spatial_gaussian", 55.0, "Spatial Gauss，σ=55"),
    ("poisson", 3.5, "Poisson，α=3.5"),
    ("salt_pepper", 0.02, "Salt&Pepper，d=0.02"),
)
SCREENSHOT = {
    "gaussian": [(26.69, .731), (27.43, .727), (27.39, .723), (26.27, .769)],
    "spatial_gaussian": [(27.60, .797), (28.31, .775), (28.29, .786), (27.68, .822)],
    "poisson": [(27.67, .818), (28.81, .820), (28.66, .813), (27.15, .838)],
    "salt_pepper": [(29.81, .844), (29.79, .807), (30.61, .837), (29.40, .869)],
}
REQUIRED = {
    "experiment", "train_seed", "dataset", "image_id", "noise_type", "noise_level",
    "psnr", "ssim", "noise_seed", "noise_sha256", "checkpoint",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def load_salt(root: Path) -> tuple[list[dict], dict]:
    """Check completeness and pair joins for this new evaluation only."""
    seedwise = []
    provenance = {"files": [], "rows_per_seed": {}, "checkpoints": {}}
    for seed in SEEDS:
        path = root / f"evaluation/seed{seed}/per_image_results.csv"
        rows = read_csv(path)
        if len(rows) != 2 * sum(DATASETS.values()):
            raise ValueError(f"{path}: expected 420 rows, got {len(rows)}")
        if not REQUIRED.issubset(rows[0]):
            raise ValueError(f"{path}: missing columns {sorted(REQUIRED - rows[0].keys())}")
        paired = defaultdict(dict)
        checkpoints = defaultdict(set)
        for row in rows:
            experiment = row["experiment"]
            if experiment not in ("G1", "G1C") or int(row["train_seed"]) != seed:
                raise ValueError(f"{path}: unexpected experiment or seed: {row}")
            if row["dataset"] not in DATASETS or row["noise_type"] != "salt_pepper" or not math.isclose(float(row["noise_level"]), .02, abs_tol=1e-12):
                raise ValueError(f"{path}: unexpected dataset or condition: {row}")
            if not all(row[field].strip() for field in ("image_id", "noise_seed", "noise_sha256", "checkpoint")):
                raise ValueError(f"{path}: empty pairing/provenance field: {row}")
            for metric in ("psnr", "ssim"):
                if not math.isfinite(float(row[metric])):
                    raise ValueError(f"{path}: non-finite {metric}: {row}")
            key = row["dataset"], row["image_id"]
            if experiment in paired[key]:
                raise ValueError(f"{path}: duplicate {experiment} image key {key}")
            paired[key][experiment] = row
            checkpoints[experiment].add(row["checkpoint"])
        by_dataset = defaultdict(list)
        for key, pair in paired.items():
            if set(pair) != {"G1", "G1C"}:
                raise ValueError(f"{path}: incomplete G1/G1C pair {key}")
            for field in ("noise_seed", "noise_sha256"):
                if pair["G1"][field] != pair["G1C"][field]:
                    raise ValueError(f"{path}: paired new-run {field} mismatch for {key}")
            by_dataset[key[0]].append(pair)
        for dataset, expected in DATASETS.items():
            pairs = by_dataset[dataset]
            if len(pairs) != expected:
                raise ValueError(f"{path}: {dataset} expected {expected} paired images, got {len(pairs)}")
            result = {
                "train_seed": seed, "dataset": dataset, "noise_type": "salt_pepper",
                "noise_level": .02, "image_count_per_model": expected,
            }
            for metric in ("psnr", "ssim"):
                for experiment in ("G1", "G1C"):
                    result[f"{experiment.lower()}_{metric}"] = statistics.mean(float(pair[experiment][metric]) for pair in pairs)
                result[f"delta_{metric}"] = statistics.mean(float(pair["G1C"][metric]) - float(pair["G1"][metric]) for pair in pairs)
            seedwise.append(result)
        provenance["files"].append(str(path.resolve()))
        provenance["rows_per_seed"][str(seed)] = len(rows)
        provenance["checkpoints"][str(seed)] = {key: sorted(value) for key, value in checkpoints.items()}
    return seedwise, provenance


def build_conditions(history_path: Path, seedwise: list[dict]) -> list[dict]:
    historical = {}
    for row in read_csv(history_path):
        key = row["noise_type"], float(row["noise_level"]), row["dataset"]
        if key in historical:
            raise ValueError(f"Duplicate historical summary condition: {key}")
        historical[key] = row
    result = []
    for noise, level, label in CONDITIONS:
        for index, (dataset, image_count) in enumerate(DATASETS.items()):
            source = "new_channel_independent_salt_rerun" if noise == "salt_pepper" else "historical_summary_reused"
            ours_psnr, ours_ssim = SCREENSHOT[noise][index]
            record = {
                "noise_type": noise, "noise_level": level, "noise_label": label,
                "dataset": dataset, "training_seed_count": 3, "images_per_dataset_per_seed": image_count,
                "screenshot_ours_psnr": ours_psnr, "screenshot_ours_ssim": ours_ssim,
                "result_source": source,
            }
            if noise == "salt_pepper":
                group = [row for row in seedwise if row["dataset"] == dataset]
                if sorted(row["train_seed"] for row in group) != list(SEEDS):
                    raise ValueError(f"Incomplete seeds for {dataset}")
                for metric in ("psnr", "ssim"):
                    for experiment in ("g1", "g1c"):
                        record[f"{experiment}_{metric}_mean"], record[f"{experiment}_{metric}_std"] = stats([row[f"{experiment}_{metric}"] for row in group])
                    record[f"delta_{metric}_mean"], record[f"delta_{metric}_std"] = stats([row[f"delta_{metric}"] for row in group])
            else:
                old = historical[(noise, level, dataset)]
                if int(old["training_seed_count"]) != 3:
                    raise ValueError(f"Historical summary is not three-seed: {noise}, {level}, {dataset}")
                for metric in ("psnr", "ssim"):
                    for suffix in ("mean", "std"):
                        record[f"g1_{metric}_{suffix}"] = float(old[f"baseline_{metric}_{suffix}"])
                        record[f"g1c_{metric}_{suffix}"] = float(old[f"{metric}_{suffix}"])
                    for suffix in ("mean", "std"):
                        record[f"delta_{metric}_{suffix}"] = float(old[f"delta_{metric}_{suffix}"])
            result.append(record)
    return result


def markdown_table(conditions: list[dict]) -> str:
    lines = [
        "数值为 **PSNR（dB）/SSIM**。G1、G1C 均为 seed 0、1、2 各训练 300k 后的均值；ΔPSNR 为 G1C − G1，按未舍入的同 seed 差值计算。",
        "",
        "| 噪声类型 | 数据集 | 截图 Ours | G1 三 seed | G1C 三 seed | ΔPSNR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in conditions:
        label = row["noise_label"] if row["dataset"] == "CBSD68" else ""
        lines.append(
            f"| {label} | {row['dataset']} | {row['screenshot_ours_psnr']:.2f}/{row['screenshot_ours_ssim']:.3f} "
            f"| {row['g1_psnr_mean']:.2f}/{row['g1_ssim_mean']:.3f} "
            f"| **{row['g1c_psnr_mean']:.2f}/{row['g1c_ssim_mean']:.3f}** | {row['delta_psnr_mean']:+.3f} |"
        )
    lines.extend([
        "",
        "G1：原门控模型。G1C：门控＋ L1 + 0.05 × L_structure。截图 Ours 直接录入用户提供的论文表格。",
        "",
        "椒盐噪声的 G1 与 G1C 均已在本轮重新推理，使用 RGB 各通道独立的盐/椒噪声，d=0.02；每个 seed、图像的 G1/G1C 使用相同噪声。其分布遵循 imnoise 的逐通道方式，但不是 MATLAB 随机噪声的逐像素复现。",
        "",
        "Gauss σ=50、Spatial Gauss σ=55、Poisson α=3.5 复用既有三 seed 汇总。先逐 seed 平均数据集内图像，再对三个 seed 求均值和样本标准差（ddof=1）；Δ 的标准差由同 seed 差值计算。未舍入的均值、标准差和差值保存在 table1_conditions.csv；逐 seed 椒盐结果保存在 table1_seedwise_salt.csv。",
        "",
        "本轮只检查新结果的行数、条件、seed 完整性及 G1/G1C 配对。此前按用户要求跳过的历史输入/噪声一致性等校验记录继续适用，不把历史比较标记为已验证。旧的 RGB 同时损坏结果保持原样；先前的单 seed 诊断不参与本表。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    missing = [args.root / f"evaluation/seed{seed}/per_image_results.csv" for seed in SEEDS]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        raise SystemExit("Waiting for complete new three-seed evaluation; missing:\n" + "\n".join(str(path) for path in missing))
    seedwise, provenance = load_salt(args.root)
    conditions = build_conditions(args.history, seedwise)
    final_dir = args.root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_csv(final_dir / "table1_conditions.csv", conditions)
    write_csv(final_dir / "table1_seedwise_salt.csv", seedwise)
    table = markdown_table(conditions)
    (final_dir / "Table1.md").write_text(table, encoding="utf-8")
    summary = {
        "completed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seeds": list(SEEDS), "training_steps_per_seed": 300000,
        "condition_count": len(conditions), "new_evaluation_image_rows": 1260,
        "new_evaluation_rows_per_model": 630,
        "mean_rule": "mean images within each seed and dataset, then mean the three seed means",
        "std_rule": "sample standard deviation over three seed means, ddof=1",
        "delta_rule": "G1C minus G1 within matched seed, then three-seed mean and sample standard deviation",
        "salt_pepper": {
            "noise_level": .02,
            "noise_protocol": "channel-independent salt-and-pepper; equal salt/pepper probabilities",
            "matlab_pixelwise_replication": False,
            "both_models_rerun": True, "single_seed_diagnostic_reused": False,
            "new_result_checks": "row counts, expected conditions/seeds, unique paired image keys, matched noise_seed and noise_sha256",
            **provenance,
        },
        "historical": {
            "source": str(args.history.resolve()),
            "reused_conditions": ["gaussian 50", "spatial_gaussian 55", "poisson 3.5"],
            "prior_skipped_checks_continue_to_apply": True,
            "historical_input_hash_gate_added": False,
            "old_shared_rgb_salt_results_modified": False,
        },
        "positive_delta_condition_counts": {
            metric: sum(row[f"delta_{metric}_mean"] > 0 for row in conditions)
            for metric in ("psnr", "ssim")
        },
        "conditions": conditions,
        "outputs": ["final/table1_conditions.csv", "final/table1_seedwise_salt.csv", "final/Table1.md", "final/summary.json", "README.md"],
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = (
        "# Table 1：三 seed 逐通道椒盐噪声评测\n\n"
        "本目录保存对齐逐通道椒盐噪声协议后的 G1/G1C 三 seed 结果。原历史实验目录保持原样。\n\n"
        "- [最终 Table 1](final/Table1.md)\n"
        "- [16 条件未舍入均值、标准差和差值](final/table1_conditions.csv)\n"
        "- [椒盐噪声逐 seed 汇总](final/table1_seedwise_salt.csv)\n"
        "- [汇总与来源记录](final/summary.json)\n"
        "- 逐图原始结果：evaluation/seed0、evaluation/seed1、evaluation/seed2 下的 per_image_results.csv。\n\n"
        + table
    )
    (args.root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "complete", "final_dir": str(final_dir.resolve()), "condition_count": len(conditions), "new_image_rows": 1260, "positive_delta_condition_counts": summary["positive_delta_condition_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
