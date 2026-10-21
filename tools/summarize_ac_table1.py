"""Summarize the extra historical benchmark and six highest-severity Table 1 cells."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_ac_table1 import (DATASET_COUNTS, EXPECTED_JOBS, PROTOCOL, TABLE1_CONDITIONS,
                                     TRAINING_SEEDS, historical_index)
from tools.evaluation_utils import CONDITIONS, read_csv, write_csv
from tools.experiment_common import atomic_json, digest, read_json, sha256_file, utc_now
from tools.summarize_experiments import aggregate, hierarchical_ood


def validate_rows(rows, reference, manifests):
    """Complete 7-model x 210-image x 15-condition grid with exact historical inputs."""
    expected_jobs = set(EXPECTED_JOBS)
    seen, groups, clean_hashes = set(), defaultdict(set), {}
    cleaned = []
    for source in rows:
        row = dict(source)
        method, seed = row["experiment"], str(row["train_seed"])
        if (method, seed) not in expected_jobs:
            raise ValueError("Unexpected model/training seed")
        row["train_seed"] = seed
        row["noise_level"], row["noise_seed"] = float(row["noise_level"]), int(row["noise_seed"])
        for metric in ("psnr", "ssim"):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError("Nonfinite metric")
        key = row["dataset"], row["image_id"], row["noise_type"], row["noise_level"]
        if key not in reference:
            raise ValueError("Unexpected dataset/image/condition")
        full_key = method, seed, *key
        if full_key in seen:
            raise ValueError("Duplicate per-image measurement")
        seen.add(full_key)
        groups[(method, seed)].add(key)
        expected = reference[key]
        if (row["noise_sha256"], row["noise_seed"]) != (expected["noise_sha256"], expected["noise_seed"]):
            raise ValueError("Noisy input differs from historical benchmark")
        if (int(row["height"]), int(row["width"])) != (expected["height"], expected["width"]):
            raise ValueError("Image dimensions differ")
        image_key = key[:2]
        if image_key in clean_hashes and clean_hashes[image_key] != row["clean_sha256"]:
            raise ValueError("Paired methods use different clean images")
        clean_hashes[image_key] = row["clean_sha256"]
        metadata = manifests[(method, seed)]
        identity = metadata["identity"]
        config_hash = identity["config_hash"] or ""
        if (row["checkpoint_sha256"], row.get("config_hash") or "", row["source_hash"], row["dataset_manifest_hash"]) != (
                identity["checkpoint_sha256"], config_hash, identity["evaluation_source_hash"], metadata["dataset_manifest_hash"]):
            raise ValueError("CSV/checkpoint/source/data identity mismatch")
        expected_domain = "ID" if key[2:] == ("gaussian", 15.0) else "OOD"
        expected_group = "extension" if key[2] in {"speckle", "salt_pepper"} else "official"
        if row["domain"] != expected_domain or row["protocol_group"] != expected_group:
            raise ValueError("Incorrect ID/OOD or official/extension label")
        if row["evaluation_scope"] != PROTOCOL["evaluation_scope"] or row["stage"] != "confirm":
            raise ValueError("Incorrect extra-benchmark scope/stage")
        cleaned.append(row)
    if set(groups) != expected_jobs or any(keys != set(reference) for keys in groups.values()):
        raise ValueError("Missing model/image/condition cells")
    if len(cleaned) != 22050:
        raise ValueError(f"Expected 22050 measurements, got {len(cleaned)}")
    for metadata in manifests.values():
        expected_clean = {(row["dataset"], row["image_id"]): row["sha256"] for row in metadata["dataset_manifest"]}
        if clean_hashes != expected_clean:
            raise ValueError("Raw image hashes differ from bound data manifest")
    return cleaned


def paired_rows(rows):
    baseline = {(r["train_seed"], r["dataset"], r["image_id"], r["noise_type"], r["noise_level"]): r
                for r in rows if r["experiment"] == "B0"}
    output = []
    for row in rows:
        if row["experiment"] not in {"B1", "B3"}:
            continue
        key = row["train_seed"], row["dataset"], row["image_id"], row["noise_type"], row["noise_level"]
        base = baseline[key]
        for field in ("noise_sha256", "noise_seed", "clean_sha256", "source_hash", "dataset_manifest_hash"):
            if row[field] != base[field]:
                raise ValueError(f"Unpaired {field}")
        output.append({**{key: row[key] for key in ("experiment", "train_seed", "dataset", "image_id", "noise_type", "noise_level", "domain", "protocol_group")},
                       "baseline": "B0", "delta_psnr": row["psnr"] - base["psnr"],
                       "delta_ssim": row["ssim"] - base["ssim"], "noise_sha256": row["noise_sha256"],
                       "checkpoint_sha256": row["checkpoint_sha256"], "baseline_checkpoint_sha256": base["checkpoint_sha256"]})
    return output


def table1_tables(condition_rows):
    lookup = {(r["experiment"], str(r["train_seed"]), r["dataset"], r["noise_type"], float(r["noise_level"])): r
              for r in condition_rows}
    wide, seedwise, paired = [], [], []
    for dataset, family, level in TABLE1_CONDITIONS:
        common = {"dataset": dataset, "noise_type": family, "noise_level": level}
        row = dict(common)
        for method in ("C0", "B0", "B1", "B3"):
            seeds = ["C0"] if method == "C0" else [str(seed) for seed in TRAINING_SEEDS]
            entries = [lookup[(method, seed, dataset, family, level)] for seed in seeds]
            row[method + "_seed_count"] = 1 if method == "C0" else len(seeds)
            for metric in ("psnr", "ssim"):
                values = [entry[metric + "_mean"] for entry in entries]
                row[method + "_" + metric] = statistics.mean(values)
                row[method + "_" + metric + "_seed_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        for seed in TRAINING_SEEDS:
            seed_row = dict(common, train_seed=seed)
            for method in ("B0", "B1", "B3"):
                entry = lookup[(method, str(seed), dataset, family, level)]
                for metric in ("psnr", "ssim"):
                    seed_row[method + "_" + metric] = entry[metric + "_mean"]
                    if method != "B0":
                        seed_row[method + "_minus_B0_" + metric] = entry[metric + "_mean"] - lookup[("B0", str(seed), dataset, family, level)][metric + "_mean"]
            seedwise.append(seed_row)
        for method in ("B1", "B3"):
            pair = dict(common, experiment=method, baseline="B0", train_seed_count=2)
            for metric in ("psnr", "ssim"):
                differences = [lookup[(method, str(seed), dataset, family, level)][metric + "_mean"] -
                               lookup[("B0", str(seed), dataset, family, level)][metric + "_mean"] for seed in TRAINING_SEEDS]
                pair["delta_" + metric] = statistics.mean(differences)
                pair["delta_" + metric + "_seed_sd"] = statistics.stdev(differences)
                pair[metric + "_both_seeds_positive"] = all(value > 0 for value in differences)
                row[method + "_minus_B0_" + metric] = pair["delta_" + metric]
            paired.append(pair)
        wide.append(row)
    return wide, seedwise, paired


def report_text(wide, table_pairs, ood):
    lines = ["# A/C 历史完整基准与 Table 1 条件对照", "",
             "本结果是追加的历史基准复用评测：四数据集、15 个条件、7 份权重，共 22,050 条逐图结果。", "",
             "完整 CBSD68 包含本轮已用于开发的前 20 张图；四个数据集在前轮实验中也已评测。因此这里不能称为未见测试集。本报告与原主终点（CBSD68 剩余 48 张＋Kodak24、七条件、三个噪声种子）分开保存，不能替代原主终点结论。", "",
             "训练前可读取原始图像进行数据/噪声哈希核验，该检查不运行模型、不计算去噪指标。正式模型评测需等全部确认检查点通过完整性与同步训练验证。", "",
             "九个原协议条件固定噪声 seed=0；六个扩展条件按旧版 dataset/image_id/noise_type/level 的 SHA256 生成固定种子。所有 3,150 个唯一噪声输入与历史 CSV 的 noise_sha256 完全一致，原始图像文件清单 SHA256 也一致。PSNR/SSIM 复用旧版 RGB×255 官方实现，未混用原主终点的 skimage 版本。", "",
             "B0 为匹配续训基线，A/B1 为深层相关 PFA，C/B3 为结构损失。B0/A/C 各包含续训种子 101/102，并使用通过严格同步三组检查的新检查点。C0 仅展示共同初始权重的一份结果。以下数字不与论文或截图参考值作比较。", "",
             "## 六个最高强度条件", "", "表中 PSNR/SSIM 是两个续训种子的均值；Δ 为同种子配对后再平均，C0 不重复计入训练种子。", "",
             "| 数据集 | 条件 | C0 PSNR/SSIM | B0 PSNR/SSIM | A PSNR/SSIM | A−B0 dB | C PSNR/SSIM | C−B0 dB |",
             "|---|---|---|---|---|---:|---|---:|"]
    for row in wide:
        score = lambda method: f"{row[method + '_psnr']:.3f}/{row[method + '_ssim']:.4f}"
        lines.append(f"| {row['dataset']} | {row['noise_type']} {row['noise_level']:g} | {score('C0')} | {score('B0')} | {score('B1')} | {row['B1_minus_B0_psnr']:+.3f} | {score('B3')} | {row['B3_minus_B0_psnr']:+.3f} |")
    lines += ["", "每个条件的两个种子及配对差值见 [table1_seedwise.csv](table1_seedwise.csv)，均值和样本标准差见 [table1_paired.csv](table1_paired.csv)。两个种子不足以建立强统计显著性结论。"]
    for method in ("B1", "B3"):
        selected = [row for row in table_pairs if row["experiment"] == method]
        positive = sum(row["delta_psnr"] > 0 for row in selected)
        both = sum(row["psnr_both_seeds_positive"] for row in selected)
        negative = sum(row["delta_psnr"] < 0 for row in selected)
        lines.append(f"- {method}：六条件中平均 PSNR 提升 {positive} 项、下降 {negative} 项；两个种子均提升 {both} 项。")
    lines += ["", "## 完整 OOD 汇总", "", "排除 Gaussian15 后，先平均图像，再平均同噪声族内强度，再等权平均噪声族和数据集。三噪声族与含扩展的五噪声族分别报告。", "",
              "| 范围 | 方法 | 训练种子 | OOD PSNR | OOD SSIM |", "|---|---|---|---:|---:|"]
    for row in ood:
        lines.append(f"| {row['table']} | {row['experiment']} | {row['train_seed']} | {row['ood_psnr']:.4f} | {row['ood_ssim']:.6f} |")
    lines += ["", "## 完整性与文件", "", "原始逐图数据和所有负差值均保留；未按结果筛选条件或替换基线。公共 C0 使两种子的变化只覆盖续训随机性，不覆盖独立初始化变化。", "",
              "- [逐图结果](per_image_results.csv)", "- [全部条件逐种子均值](condition_results.csv)",
              "- [全部条件种子均值与波动](all_condition_seed_summary.csv)", "- [同种子逐图配对差值](paired_per_image_results.csv)",
              "- [全部条件配对均值](paired_condition_results.csv)", "- [原协议 OOD 汇总](ood_aggregate_results.csv)",
              "- [六条件表](table1_conditions.csv)", "- [全部负 PSNR 差值](negative_cases.csv)",
              "- [文件哈希与身份核验](summary.json)", ""]
    return "\n".join(lines)


def summarize(evaluation_root, output, historical_csv=None):
    manifests, rows = {}, []
    for method, seed in EXPECTED_JOBS:
        directory = evaluation_root / method / f"seed{seed}"
        summary = read_json(directory / "summary.json")
        if not summary.get("completed") or summary.get("rows") != 3150 or summary.get("matched_historical_inputs") != 3150:
            raise ValueError(f"Incomplete historical benchmark: {directory}")
        identity = summary["identity"]
        if (identity["method"], str(identity["training_seed"])) != (method, seed):
            raise ValueError("Evaluation method/seed identity mismatch")
        if summary["protocol_hash"] != digest(PROTOCOL) or digest(summary["protocol"]) != digest(PROTOCOL):
            raise ValueError("Evaluation protocol mismatch")
        if summary["contract_hash"] != digest(summary["contract"]):
            raise ValueError("Training contract hash mismatch")
        if identity["evaluation_source_hash"] != summary["contract"]["source_hash"]:
            raise ValueError("Evaluator/training source identity mismatch")
        csv_path = directory / "per_image_results.csv"
        if sha256_file(csv_path) != summary["metrics_csv_sha256"]:
            raise ValueError("Evaluation CSV SHA256 mismatch")
        actual_data_hash = hashlib.sha256(json.dumps(summary["dataset_manifest"], sort_keys=True).encode()).hexdigest()
        if actual_data_hash != summary["dataset_manifest_hash"]:
            raise ValueError("Dataset manifest content/hash mismatch")
        manifests[(method, seed)] = summary
        rows.extend(read_csv(csv_path))
    for field in ("dataset_manifest_hash", "contract_hash", "protocol_hash"):
        if len({item[field] for item in manifests.values()}) != 1:
            raise ValueError(f"Unpaired evaluation {field}")
    historical_csv = historical_csv or Path(manifests[("C0", "C0")]["historical_csv"]["path"])
    reference, historical = historical_index(historical_csv)
    if any(item["historical_csv"]["sha256"] != historical["sha256"] for item in manifests.values()):
        raise ValueError("Different historical reference CSV")
    c0 = manifests[("C0", "C0")]
    if c0["identity"]["checkpoint_sha256"] != historical["c0_checkpoint_sha256"]:
        raise ValueError("C0 is not the bound historical initialization")
    if c0["dataset_manifest_hash"] != historical["dataset_manifest_hash"]:
        raise ValueError("Raw benchmark data differs from history")
    if c0["contract"]["c0_sha256"] != historical["c0_checkpoint_sha256"]:
        raise ValueError("Training C0 is not the bound historical initialization")
    rows = validate_rows(rows, reference, manifests)
    keys = ("experiment", "train_seed", "dataset", "noise_type", "noise_level", "protocol_group", "domain")
    condition_rows = aggregate(rows, keys, ("psnr", "ssim"))
    pairs = paired_rows(rows)
    pair_conditions = aggregate(pairs, keys, ("delta_psnr", "delta_ssim"))
    seed_summary = aggregate(condition_rows, ("experiment", "dataset", "noise_type", "noise_level", "protocol_group", "domain"), ("psnr_mean", "ssim_mean"))
    wide, seedwise, table_pairs = table1_tables(condition_rows)
    ood = hierarchical_ood(condition_rows)
    artifacts = {"per_image_results.csv": rows, "condition_results.csv": condition_rows,
                 "all_condition_seed_summary.csv": seed_summary, "paired_per_image_results.csv": pairs,
                 "paired_condition_results.csv": pair_conditions, "table1_conditions.csv": wide,
                 "table1_seedwise.csv": seedwise, "table1_paired.csv": table_pairs, "ood_aggregate_results.csv": ood,
                 "negative_cases.csv": [row for row in pairs if row["delta_psnr"] < 0]}
    for filename, values in artifacts.items():
        write_csv(output / filename, values, fields=list(pairs[0]) if filename == "negative_cases.csv" else None)
    (output / "experiment_report.md").write_text(report_text(wide, table_pairs, ood), encoding="utf-8")
    artifact_hashes = {filename: sha256_file(output / filename) for filename in [*artifacts, "experiment_report.md"]}
    result = {"completed": True, "completed_at": utc_now(), "rows": len(rows), "models": len(manifests),
              "unique_historical_inputs": len(reference), "all_inputs_match_history": True,
              "full_condition_rows": len(condition_rows), "table1_rows": len(wide),
              "table1_seedwise_rows": len(seedwise), "table1_paired_rows": len(table_pairs),
              "negative_paired_images": len(artifacts["negative_cases.csv"]),
              "protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL), "historical_csv": historical,
              "dataset_manifest_hash": historical["dataset_manifest_hash"], "contract_hash": c0["contract_hash"],
              "evaluation_source_hash": c0["identity"]["evaluation_source_hash"],
              "checkpoint_identities": [item["identity"] for item in manifests.values()],
              "source_metrics_csv_sha256": {f"{method}/seed{seed}": item["metrics_csv_sha256"] for (method, seed), item in manifests.items()},
              "artifact_sha256": artifact_hashes}
    atomic_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "reports/ac/benchmark_table1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ac/table1")
    parser.add_argument("--historical-csv", type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.output, args.historical_csv)
    print(f"Additional reused benchmark complete: {result['rows']} rows; six Table1 cells; {args.output / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
