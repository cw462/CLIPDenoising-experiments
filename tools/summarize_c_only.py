"""Report C/G1C full seed0 results with explicitly reused B0/G1 baselines."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_c_only import (DATASET_COUNTS, HISTORICAL_SOURCE, PROTOCOL, TABLE1_CONDITIONS,
                                    UPDATES, VARIANTS, cell_key, historical_baseline, historical_index)
from tools.evaluation_utils import read_csv, write_csv
from tools.experiment_common import atomic_json, digest, read_json, sha256_file, utc_now
from tools.summarize_experiments import aggregate, hierarchical_ood


def validate_rows(rows, reference, identity, data_manifest, manifest_hash):
    clean_hashes = {(r["dataset"], r["image_id"]): r["sha256"] for r in data_manifest}
    seen, result = set(), []
    method = identity["method"]
    for source in rows:
        row = dict(source)
        key = cell_key(row)
        if key in seen:
            raise ValueError("Duplicate per-image measurement")
        if key not in reference:
            raise ValueError("Unexpected dataset/image/condition")
        seen.add(key)
        if (row["experiment"], int(row["train_seed"]), row["stage"], int(row["update"])) != (method, 0, "full", UPDATES):
            raise ValueError("Wrong per-image method/seed/budget identity")
        for field in ("checkpoint_sha256", "config_hash", "source_hash", "git_commit", "measurement_origin"):
            if row[field] != identity[field]:
                raise ValueError(f"Per-image identity mismatch: {field}")
        expected = reference[key]
        if (row["noise_sha256"], int(row["noise_seed"]), int(row["height"]), int(row["width"])) != (
                expected["noise_sha256"], expected["noise_seed"], expected["height"], expected["width"]):
            raise ValueError("Noisy input differs from historical benchmark")
        if row["dataset_manifest_hash"] != manifest_hash:
            raise ValueError("Raw dataset manifest mismatch")
        if method in VARIANTS and row.get("clean_sha256") != clean_hashes[(key[0], key[1])]:
            raise ValueError("Clean image SHA256 mismatch")
        domain = "ID" if key[2:] == ("gaussian", 15.0) else "OOD"
        group = "extension" if key[2] in {"speckle", "salt_pepper"} else "official"
        if row["domain"] != domain or row["protocol_group"] != group:
            raise ValueError("Wrong domain/protocol group")
        row.update(train_seed="0", noise_level=key[3], noise_seed=int(row["noise_seed"]), update=UPDATES)
        for metric in ("psnr", "ssim"):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError("Nonfinite metric")
        result.append(row)
    if seen != set(reference) or len(result) != 3150:
        raise ValueError("Missing historical benchmark cells")
    return result


def paired_rows(rows, comparisons=None):
    comparisons = comparisons or tuple(VARIANTS.items())
    indexed = {(row["experiment"], cell_key(row)): row for row in rows}
    result = []
    for method, baseline in comparisons:
        for (name, key), row in indexed.items():
            if name != method:
                continue
            base = indexed[(baseline, key)]
            if any(row[field] != base[field] for field in ("noise_sha256", "noise_seed", "dataset_manifest_hash")):
                raise ValueError("Unpaired input identity")
            result.append({"experiment": method, "baseline": baseline, "comparison": f"{method}-{baseline}",
                           "train_seed": "0", "dataset": key[0], "image_id": key[1], "noise_type": key[2],
                           "noise_level": key[3], "noise_seed": row["noise_seed"], "noise_sha256": row["noise_sha256"],
                           "protocol_group": row["protocol_group"], "domain": row["domain"],
                           "psnr": row["psnr"], "baseline_psnr": base["psnr"], "delta_psnr": row["psnr"] - base["psnr"],
                           "ssim": row["ssim"], "baseline_ssim": base["ssim"], "delta_ssim": row["ssim"] - base["ssim"],
                           "checkpoint_sha256": row["checkpoint_sha256"], "baseline_checkpoint_sha256": base["checkpoint_sha256"]})
    return result


def table1_tables(condition_rows):
    lookup = {(row["experiment"], row["dataset"], row["noise_type"], float(row["noise_level"])): row
              for row in condition_rows}
    wide, pairs = [], []
    for dataset, family, level in TABLE1_CONDITIONS:
        item = {"dataset": dataset, "noise_type": family, "noise_level": level, "train_seed": 0}
        for method in ("B0", "C", "G1", "G1C"):
            row = lookup[(method, dataset, family, level)]
            item.update({method + "_psnr": row["psnr_mean"], method + "_ssim": row["ssim_mean"], method + "_n_images": row["n"]})
        for method, baseline in (*VARIANTS.items(), ("G1C", "C")):
            comparison = method + "_minus_" + baseline
            item[comparison + "_psnr"] = item[method + "_psnr"] - item[baseline + "_psnr"]
            item[comparison + "_ssim"] = item[method + "_ssim"] - item[baseline + "_ssim"]
            if baseline != "C":
                pairs.append({"experiment": method, "baseline": baseline, "train_seed": 0,
                              "dataset": dataset, "noise_type": family, "noise_level": level,
                              "psnr": item[method + "_psnr"], "baseline_psnr": item[baseline + "_psnr"],
                              "delta_psnr": item[comparison + "_psnr"], "ssim": item[method + "_ssim"],
                              "baseline_ssim": item[baseline + "_ssim"], "delta_ssim": item[comparison + "_ssim"],
                              "training_seed_count": 1})
        wide.append(item)
    return wide, pairs


def ood_differences(ood):
    lookup = {(row["experiment"], row["table"]): row for row in ood}
    return [{"comparison": f"{method}-{base}", "table": table,
             "delta_ood_psnr": lookup[(method, table)]["ood_psnr"] - lookup[(base, table)]["ood_psnr"],
             "delta_ood_ssim": lookup[(method, table)]["ood_ssim"] - lookup[(base, table)]["ood_ssim"]}
            for method, base in (*VARIANTS.items(), ("G1C", "C"))
            for table in ("official_three_families", "five_families_with_extensions")]


def report_text(wide, table_pairs, ood, differences, negatives, panel_names):
    lines = ["# C / G1C 完整 300k 训练与历史基线对照", "",
             "C 和 G1C 均从 RN50 预训练编码器及 seed0 解码器初始化开始，完成 300,000 次更新；不是从旧模型最终权重续训。"
             "C 对照旧 B0 seed0，G1C 对照旧 G1 seed0（含 decoder guidance 的 gated 模型）；两组新训练均采用相同结构损失。", "",
             "旧 B0/G1 的 300k 结果直接复用已锁定 SHA256 的历史 CSV，没有重新训练或重新推理。"
             "本报告包括 6,300 条新测结果和 6,300 条历史结果；全部四数据集、15 条件逐图输入和噪声哈希一致，使用原始 RGB PSNR/SSIM 实现。", "",
             "只有一个训练种子（seed0），不报告跨种子波动、统计稳定性或显著性。更新预算相同不等于已经逐步验证两轮训练轨迹相同；"
             "历史基线与新模型的源码、执行时间及数值实现不同。结果是这几次具体训练之间的性能比较，不能单独确立创新的因果收益或 SOTA。", "",
             "四个数据集此前均已评测；完整 CBSD68 含先前 20 张开发图像，不能称为未见测试集。训练前输入审计可读取测试图像，但不运行新模型或读取新模型结果。", "",
             "## 六个预设最高强度条件", "",
             "每项为该数据集逐图均值；Δ 是相同图像和相同噪声输入的差值。没有挑选更好的训练种子或隐藏负结果。", "",
             "| 数据集 | 条件 | B0 PSNR/SSIM | C PSNR/SSIM | C−B0 dB | G1 PSNR/SSIM | G1C PSNR/SSIM | G1C−G1 dB |",
             "|---|---|---|---|---:|---|---|---:|"]
    for row in wide:
        score = lambda method: f"{row[method + '_psnr']:.3f}/{row[method + '_ssim']:.4f}"
        lines.append(f"| {row['dataset']} | {row['noise_type']} {row['noise_level']:g} | {score('B0')} | {score('C')} | {row['C_minus_B0_psnr']:+.3f} | {score('G1')} | {score('G1C')} | {row['G1C_minus_G1_psnr']:+.3f} |")
    lines += ["", "六行宽表同时保留 G1C−C，作为新结构损失下加入 gated 模块的探索性比较；不将它混入 C−B0 或 G1C−G1 的主对照。", ""]
    for method, baseline in VARIANTS.items():
        selected = [row for row in table_pairs if row["experiment"] == method]
        positive = sum(row["delta_psnr"] > 0 for row in selected)
        negative = sum(row["delta_psnr"] < 0 for row in selected)
        lines.append(f"- {method}−{baseline}：六条件 PSNR 上升 {positive} 项、下降 {negative} 项；完整逐图结果中负差值 {negatives[method]} 项。")
    lines += ["", "## 全条件 OOD 与 ID", "",
              "OOD 排除 Gaussian15，先平均图像，再等权平均同族强度、噪声族和数据集。原三族与含 speckle/salt-pepper 扩展的五族分别报告；"
              "各数据集单独汇总见 ood_dataset_results.csv，ID 的 Gaussian15 单列。条件表中的标准差表示图像间离散程度，不是训练种子波动。", "",
              "| 范围 | 方法 | OOD PSNR | OOD SSIM |", "|---|---|---:|---:|"]
    for row in ood:
        lines.append(f"| {row['table']} | {row['experiment']} seed0 | {row['ood_psnr']:.4f} | {row['ood_ssim']:.6f} |")
    lines += ["", "| 范围 | 对照 | Δ OOD PSNR | Δ OOD SSIM |", "|---|---|---:|---:|"]
    for row in differences:
        lines.append(f"| {row['table']} | {row['comparison']} | {row['delta_ood_psnr']:+.4f} | {row['delta_ood_ssim']:+.6f} |")
    lines += ["", "## 文件与固定可视化", "",
              "[逐图结果](per_image_results.csv)、[全部条件](condition_results.csv)、[配对逐图差值](paired_per_image_results.csv)、"
              "[配对条件差值](paired_condition_results.csv)、[六条件表](table1_conditions.csv)、[seed0 对照表](table1_seedwise.csv)、"
              "[ID](id_condition_results.csv)、[全部负 PSNR 差值](negative_cases.csv)、[G1C−C 探索比较](exploratory_g1c_minus_c.csv)、"
              "[来源与文件 SHA256](summary.json)。", "",
              "下列图固定选择每个 Table1 条件中字典序第一张图，列为 clean、显示截断 noisy、新模型输出、绝对误差（固定 0–0.20）。"
              "它们展示新模型恢复效果，不包含未重新生成的历史基线图像；未使用 AI 生成图片。", ""]
    for name in panel_names:
        lines.append(f"![{Path(name).stem}]({name})")
    return "\n".join(lines) + "\n"


def summarize(evaluation_root, output, historical_csv=None):
    summaries = {method: read_json(evaluation_root / method / "seed0/summary.json") for method in VARIANTS}
    historical_csv = Path(historical_csv or summaries["C"]["historical_csv"]["path"])
    reference, historical = historical_index(historical_csv)
    rows, identities, sources, contracts = [], [], set(), []
    panel_sources = []
    for method, baseline in VARIANTS.items():
        directory, item = evaluation_root / method / "seed0", summaries[method]
        if not item.get("completed") or (item.get("rows"), item.get("baseline_rows"), item.get("matched_historical_inputs")) != (3150, 3150, 3150):
            raise ValueError("Incomplete variant evaluation")
        if item.get("protocol_hash") != digest(PROTOCOL) or digest(item["protocol"]) != digest(PROTOCOL):
            raise ValueError("Evaluation protocol mismatch")
        if item["historical_csv"]["sha256"] != historical["sha256"]:
            raise ValueError("Historical CSV SHA256 mismatch")
        if item["contract_hash"] != digest(item["contract"]):
            raise ValueError("Training contract hash mismatch")
        identity, contract = item["identity"], item["contract"]
        if (identity["method"], identity["training_seed"], identity["stage"], identity["update"]) != (method, 0, "full", UPDATES):
            raise ValueError("Evaluation model identity mismatch")
        if (contract["variant"], contract["baseline"], contract["training_seed"], contract["training_updates"]) != (method, baseline, 0, UPDATES):
            raise ValueError("Training contract identity mismatch")
        if identity["source_hash"] != contract["source_hash"] or identity["config_hash"] != contract["config_hash"]:
            raise ValueError("Training/evaluation source/config mismatch")
        proof = item["baseline_checkpoint_proof"]
        if proof.get("identity_verified") is not True or proof["checkpoint_sha256"] != item["baseline_identity"]["checkpoint_sha256"]:
            raise ValueError("Historical checkpoint identity was not verified")
        sources.add(identity["source_hash"])
        contracts.append(contract)
        required_artifacts = {"per_image_results.csv", "baseline_per_image_results.csv", "access_gate.json"}
        if not required_artifacts.issubset(item["artifact_sha256"]):
            raise ValueError("Missing SHA-bound evaluation artifacts")
        for name, expected in item["artifact_sha256"].items():
            file = directory / name
            if file.resolve().is_relative_to(directory.resolve()) is False or sha256_file(file) != expected:
                raise ValueError("Evaluation artifact SHA256/path mismatch")
        manifest_hash = hashlib.sha256(json.dumps(item["dataset_manifest"], sort_keys=True).encode()).hexdigest()
        if manifest_hash != item["dataset_manifest_hash"] or manifest_hash != historical["dataset_manifest_hash"]:
            raise ValueError("Dataset manifest identity mismatch")
        old_rows, old_identity = historical_baseline(historical_csv, baseline, reference)
        if old_identity != item["baseline_identity"]:
            raise ValueError("Historical baseline identity mismatch")
        new_rows = validate_rows(read_csv(directory / "per_image_results.csv"), reference, identity,
                                 item["dataset_manifest"], manifest_hash)
        copied_rows = validate_rows(read_csv(directory / "baseline_per_image_results.csv"), reference, old_identity,
                                    item["dataset_manifest"], manifest_hash)
        old_index = {cell_key(row): row for row in old_rows}
        for row in copied_rows:
            original = old_index[cell_key(row)]
            if any(row[field] != original[field] for field in ("psnr", "ssim", "checkpoint_sha256", "config_hash", "source_hash")):
                raise ValueError("Reused baseline scores or identity were altered")
        rows.extend(copied_rows + new_rows)
        identities.extend((old_identity, identity))
        for dataset, family, level in TABLE1_CONDITIONS:
            name = f"panels/{dataset}_{family}_{level:g}_panel.png"
            if name not in item["artifact_sha256"]:
                raise ValueError("Missing fixed six-cell visualization")
            panel_sources.append((directory / name, f"panels/{method}_{Path(name).name}"))
    if len(sources) != 1:
        raise ValueError("New model evaluation sources differ")
    for field in ("pretrained_sha256", "data_manifest_hash", "pixel_opt"):
        if len({digest(contract[field]) for contract in contracts}) != 1:
            raise ValueError(f"New training configurations differ: {field}")
    keys = ("experiment", "train_seed", "dataset", "noise_type", "noise_level", "protocol_group", "domain")
    conditions = aggregate(rows, keys, ("psnr", "ssim"))
    pairs = paired_rows(rows)
    pair_conditions = aggregate(pairs, ("experiment", "baseline", "comparison", *keys[1:]), ("delta_psnr", "delta_ssim"))
    wide, table_pairs = table1_tables(conditions)
    ood = hierarchical_ood(conditions)
    ood_datasets = []
    for dataset in DATASET_COUNTS:
        ood_datasets.extend(dict(row, dataset=dataset) for row in hierarchical_ood([row for row in conditions if row["dataset"] == dataset]))
    differences = ood_differences(ood)
    negatives = [row for row in pairs if row["delta_psnr"] < 0]
    artifacts = {"per_image_results.csv": rows, "condition_results.csv": conditions,
                 "paired_per_image_results.csv": pairs, "paired_condition_results.csv": pair_conditions,
                 "table1_conditions.csv": wide, "table1_seedwise.csv": table_pairs,
                 "ood_aggregate_results.csv": ood, "ood_dataset_results.csv": ood_datasets,
                 "ood_paired_results.csv": differences, "id_condition_results.csv": [row for row in conditions if row["domain"] == "ID"],
                 "negative_cases.csv": negatives, "negative_condition_results.csv": [row for row in pair_conditions if row["delta_psnr_mean"] < 0],
                 "exploratory_g1c_minus_c.csv": paired_rows(rows, (("G1C", "C"),))}
    for name, values in artifacts.items():
        fields = list(pairs[0]) if name == "negative_cases.csv" else list(pair_conditions[0]) if name == "negative_condition_results.csv" else None
        write_csv(output / name, values, fields=fields)
    for source, destination in panel_sources:
        (output / destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output / destination)
    panel_names = [destination for _, destination in panel_sources]
    negative_counts = {method: sum(row["experiment"] == method for row in negatives) for method in VARIANTS}
    (output / "experiment_report.md").write_text(report_text(wide, table_pairs, ood, differences, negative_counts, panel_names), encoding="utf-8")
    filenames = [*artifacts, *panel_names, "experiment_report.md"]
    result = {"completed": True, "completed_at": utc_now(), "rows": len(rows), "new_measurement_rows": 6300,
              "reused_baseline_rows": 6300, "paired_rows": len(pairs), "unique_historical_inputs": len(reference),
              "all_inputs_match_history": True, "full_condition_rows": len(conditions), "table1_rows": len(wide),
              "table1_seedwise_rows": len(table_pairs), "training_seeds_per_method": 1,
              "negative_paired_images": negative_counts, "fixed_panels": len(panel_names),
              "protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL), "historical_csv": historical,
              "dataset_manifest_hash": historical["dataset_manifest_hash"], "checkpoint_identities": identities,
              "evaluation_source_hash": sources.pop(), "contract_hashes": {method: value["contract_hash"] for method, value in summaries.items()},
              "source_summary_sha256": {method: sha256_file(evaluation_root / method / "seed0/summary.json") for method in VARIANTS},
              "artifact_sha256": {name: sha256_file(output / name) for name in filenames}}
    atomic_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--root", dest="input", type=Path, default=ROOT / "reports/c_only/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/c_only/final")
    parser.add_argument("--historical-csv", type=Path)
    args = parser.parse_args()
    result = summarize(args.input, args.output, args.historical_csv)
    print(f"Completed: {result['new_measurement_rows']} new + {result['reused_baseline_rows']} reused rows; {args.output / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
