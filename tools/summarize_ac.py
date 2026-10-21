"""Audit and summarize the shared A/C experiment without noise-seed pseudoreplication."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from tools.evaluate_pfa import CONDITIONS, PROTOCOL, SPLIT_DATASET, STAGE_SEEDS
from tools.evaluation_utils import read_csv, write_csv
from tools.experiment_common import atomic_json, digest, read_json, utc_now


def mean(values):
    return float(np.mean(list(values)))


def audit_rows(rows, stage, expected_images=None):
    """Validate pairing, full coverage, finite metrics and identical generated inputs."""
    expected_images = expected_images or ({"CBSD68_dev20": 20} if stage == "screen" else
                                           {"CBSD68_holdout48": 48, "Kodak24": 24})
    methods = {("C0", "C0")} | {(m, str(s)) for m in ("B0", "B1", "B3") for s in STAGE_SEEDS[stage]}
    actual_methods = {(row["method"], str(row["training_seed"])) for row in rows}
    if actual_methods != methods:
        raise ValueError(f"Wrong method/training-seed set: {actual_methods}")
    expected_conditions = {(f, float(l)) for f, l in CONDITIONS}
    keys, inputs, seen_images, noise_groups = set(), {}, defaultdict(set), defaultdict(set)
    cleaned = []
    for source in rows:
        row = dict(source)
        row["training_seed"] = str(row["training_seed"])
        row["level"] = float(row["level"])
        row["noise_seed"] = int(row["noise_seed"])
        for metric in ("psnr", "ssim"):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError("Nonfinite metric")
        if row["dataset"] not in expected_images or (row["family"], row["level"]) not in expected_conditions:
            raise ValueError("Unexpected dataset/condition")
        if row["noise_seed"] not in (0, 1, 2):
            raise ValueError("Unexpected noise seed")
        paired_key = (row["dataset"], row["image"], row["family"], row["level"], row["noise_seed"])
        key = (row["method"], row["training_seed"], *paired_key)
        if key in keys:
            raise ValueError(f"Duplicate measurement: {key}")
        keys.add(key)
        provenance = (row["input_sha256"], row["clean_sha256"])
        if paired_key in inputs and inputs[paired_key] != provenance:
            raise ValueError("Unpaired noisy input or clean image")
        inputs[paired_key] = provenance
        seen_images[(row["method"], row["training_seed"], row["dataset"])].add(row["image"])
        noise_groups[key[:-1]].add(row["noise_seed"])
        cleaned.append(row)
    for method, seed in methods:
        for dataset, count in expected_images.items():
            images = seen_images[(method, seed, dataset)]
            if len(images) != count or images != seen_images[("C0", "C0", dataset)]:
                raise ValueError("Missing image or mismatched dataset")
    expected_count = len(methods) * sum(expected_images.values()) * len(CONDITIONS) * 3
    if len(rows) != expected_count or any(values != {0, 1, 2} for values in noise_groups.values()):
        raise ValueError(f"Incomplete evaluation: {len(rows)} != {expected_count}")
    return cleaned


def average_noise_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in ("method", "training_seed", "dataset", "image", "family", "level"))
        groups[key].append(row)
    return [{**dict(zip(("method", "training_seed", "dataset", "image", "family", "level"), key)),
             "noise_repeats": len(items), "psnr": mean(r["psnr"] for r in items),
             "ssim": mean(r["ssim"] for r in items)} for key, items in sorted(groups.items())]


def condition_table(averaged):
    groups = defaultdict(list)
    for row in averaged:
        groups[(row["method"], row["training_seed"], row["dataset"], row["family"], row["level"])].append(row)
    return [{**dict(zip(("method", "training_seed", "dataset", "family", "level"), key)),
             "images": len(items), **{metric: mean(r[metric] for r in items) for metric in ("psnr", "ssim")}}
            for key, items in sorted(groups.items())]


def aggregate_scores(conditions):
    """Absolute scores preserve dataset equality and separate ID/OOD."""
    groups = defaultdict(list)
    for row in conditions:
        for category in categories(row):
            groups[(row["method"], row["training_seed"], row["dataset"], category)].append(row)
    values = {key: {metric: mean(r[metric] for r in items) for metric in ("psnr", "ssim")}
              for key, items in groups.items()}
    datasets = defaultdict(list)
    for (method, seed, dataset, category), value in values.items():
        datasets[(method, seed, "equal_dataset", category)].append(value)
    values.update({key: {metric: mean(r[metric] for r in items) for metric in ("psnr", "ssim")}
                   for key, items in datasets.items()})
    seeds = defaultdict(list)
    for (method, seed, dataset, category), value in values.items():
        if method != "C0":
            seeds[(method, "mean", dataset, category)].append(value)
    values.update({key: {metric: mean(r[metric] for r in items) for metric in ("psnr", "ssim")}
                   for key, items in seeds.items()})
    return [{**dict(zip(("method", "training_seed", "dataset", "category"), key)), **value}
            for key, value in sorted(values.items())]


def paired_image_table(averaged):
    lookup = {(r["method"], r["training_seed"], r["dataset"], r["image"], r["family"], r["level"]): r
              for r in averaged}
    pairs = []
    for row in averaged:
        if row["method"] == "C0":
            continue
        for baseline in (("C0",) if row["method"] == "B0" else ("B0", "C0")):
            base_seed = "C0" if baseline == "C0" else row["training_seed"]
            base = lookup[(baseline, base_seed, row["dataset"], row["image"], row["family"], row["level"])]
            pairs.append({**{k: row[k] for k in ("method", "training_seed", "dataset", "image", "family", "level")},
                          "baseline": baseline, "noise_repeats_averaged": row["noise_repeats"],
                          "delta_psnr": row["psnr"] - base["psnr"],
                          "delta_ssim": row["ssim"] - base["ssim"]})
    return pairs


def categories(row):
    if row["family"] == "gaussian" and row["level"] == 15:
        return ("id",)
    return ("ood", "spatial") if row["family"] == "spatial_gaussian" else ("ood",)


def bootstrap_images(values_by_dataset, repeats=2000):
    """Stratified paired image bootstrap; average dataset means with equal weight."""
    rng = np.random.RandomState(2026)
    estimates, means = [], []
    for dataset, values in sorted(values_by_dataset.items()):
        values = np.asarray(values, dtype=float)
        if not len(values):
            raise ValueError("Empty image stratum")
        estimates.append(rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1))
        means.append(float(values.mean()))
    ci = np.percentile(np.mean(estimates, axis=0), [2.5, 97.5])
    return float(np.mean(means)), float(ci[0]), float(ci[1])


def paired_statistics(pairs):
    # Each unit is one physical image. First average conditions, then training seeds
    # for the across-seed estimate; never treat repeats as additional images.
    image_groups = defaultdict(list)
    for row in pairs:
        for category in categories(row):
            key = (row["method"], row["baseline"], row["training_seed"], row["dataset"], row["image"], category)
            image_groups[key].append(row)
    image_values = {key: {metric: mean(r["delta_" + metric] for r in items) for metric in ("psnr", "ssim")}
                    for key, items in image_groups.items()}
    across_seed = defaultdict(list)
    for (method, baseline, seed, dataset, image, category), value in image_values.items():
        across_seed[(method, baseline, "mean", dataset, image, category)].append(value)
    image_values.update({key: {metric: mean(r[metric] for r in items) for metric in ("psnr", "ssim")}
                         for key, items in across_seed.items()})
    groups = defaultdict(lambda: defaultdict(list))
    for (method, baseline, seed, dataset, image, category), values in image_values.items():
        for metric in ("psnr", "ssim"):
            for scope in (dataset, "equal_dataset"):
                groups[(method, baseline, seed, scope, category, metric)][dataset].append(values[metric])
    summary = []
    for key, strata in sorted(groups.items()):
        estimate, low, high = bootstrap_images(strata)
        summary.append({**dict(zip(("method", "baseline", "training_seed", "dataset", "category", "metric"), key)),
                        "difference": estimate, "ci95_low": low, "ci95_high": high,
                        "independent_images": sum(map(len, strata.values())), "bootstrap_repeats": 2000,
                        "uncertainty": "paired image bootstrap only; training-seed variation reported separately"})
    return summary


def seed_variability(statistics):
    groups = defaultdict(list)
    for row in statistics:
        if row["training_seed"] != "mean":
            groups[(row["method"], row["baseline"], row["dataset"], row["category"], row["metric"])].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r["training_seed"]))
        values = [r["difference"] for r in rows]
        result.append({**dict(zip(("method", "baseline", "dataset", "category", "metric"), key)),
                       "training_seeds": ";".join(r["training_seed"] for r in rows), "n_training_seeds": len(values),
                       "mean_difference": mean(values), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else "",
                       "minimum_difference": min(values), "maximum_difference": max(values),
                       "all_seeds_positive": all(value > 0 for value in values)})
    return result


def decisions(statistics, stage):
    lookup = {(r["method"], r["baseline"], r["training_seed"], r["dataset"], r["category"], r["metric"]): r
              for r in statistics}
    results = {}
    for method in ("B1", "B3"):
        ood = lookup[(method, "B0", "mean", "equal_dataset", "ood", "psnr")]
        id_rows = [lookup[(method, "B0", str(seed), "equal_dataset", "id", "psnr")] for seed in STAGE_SEEDS[stage]]
        seed_ood = [lookup[(method, "B0", str(seed), "equal_dataset", "ood", "psnr")]["difference"] for seed in STAGE_SEEDS[stage]]
        threshold = ood["difference"] >= 0.1
        direction = all(value > 0 for value in seed_ood)
        id_guard = all(row["difference"] >= -0.05 for row in id_rows)
        results[method] = {"ood_gain_ge_0_1_db": threshold, "all_training_seeds_ood_positive": direction,
                           "every_training_seed_id_ge_minus_0_05_db": id_guard,
                           "criterion_met": threshold and direction and id_guard,
                           "final_confirmation": stage == "confirm",
                           "paired_image_ci_excludes_zero": ood["ci95_low"] > 0,
                           "interpretation": "Predefined engineering threshold, not a multi-seed significance test"}
    return results


def build_visual_panels(jobs, output, stage):
    seed = str(STAGE_SEEDS[stage][0])
    selected = {m: jobs[(m, "C0" if m == "C0" else seed)] for m in ("C0", "B0", "B1", "B3")}
    manifests = {}
    for method, job in selected.items():
        manifests[method] = {(r["dataset"], r["image"], r["family"], float(r["level"])): r
                              for r in read_csv(job / "visual_manifest.csv")}
    keys = sorted(manifests["C0"])
    if any(set(items) != set(keys) for items in manifests.values()) or len(keys) < 6:
        raise ValueError("Missing fixed visual comparison samples")
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    index = ["# 固定图像对比", "", f"训练种子 {seed}，噪声种子 0；图像按文件名固定选前两张，非按提升挑选。",
             "上排完整图像，下排相同坐标 96×96 中心局部。仅为展示缩放，指标来自全分辨率浮点输出。", ""]
    for key in keys:
        reference = manifests["C0"][key]
        clean_root = selected["C0"] / reference["directory"]
        entries = [("Clean", clean_root / "clean.png", clean_root / "clean_crop.png"),
                   ("Noisy (display clipped)", clean_root / "noisy.png", clean_root / "noisy_crop.png")]
        for method in ("C0", "B0", "B1", "B3"):
            row = manifests[method][key]
            root = selected[method] / row["directory"]
            label = f"{method}: {float(row['psnr']):.3f} dB / {float(row['ssim']):.4f}"
            entries.append((label, root / "restored.png", root / "restored_crop.png"))
        cell_w, cell_h, label_h = 280, 260, 30
        canvas = Image.new("RGB", (cell_w * len(entries), (cell_h + label_h) * 2), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (label, full, crop) in enumerate(entries):
            for row_index, path in enumerate((full, crop)):
                picture = Image.open(path).convert("RGB")
                picture.thumbnail((cell_w - 8, cell_h - 8), Image.Resampling.LANCZOS)
                x = column * cell_w + (cell_w - picture.width) // 2
                y = row_index * (cell_h + label_h)
                draw.text((column * cell_w + 5, y + 5), label if row_index == 0 else "Fixed center crop", fill="black")
                canvas.paste(picture, (x, y + label_h + (cell_h - picture.height) // 2))
        filename = f"{key[0]}_{digest(key)[:16]}.png"
        canvas.save(directory / filename)
        index.append(f"- {key[0]} / {key[1]} / {key[2]} {key[3]:g}: [{filename}]({filename})")
    (directory / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return len(keys)


def render_report(output, stage, rows, scores, statistics, variability, verdicts, panels):
    title = "筛选阶段" if stage == "screen" else "确认阶段"
    lines = [f"# A/C 联合实验：{title}结果", "", f"生成时间：{utc_now()}。完整评测 {len(rows):,} 条，固定比较拼图 {panels} 张。", "",
             "B0：从 C0 重新建立优化器、仅 L1 的共同续训基线；A/B1：深层相关 PFA；C/B3：原始 PFA 加平滑结构损失。C0 为前轮 B0 seed0 的固定最终检查点，仅评测一次。", "",
             "噪声种子 0/1/2 先在每张图、每个条件内部平均；Gaussian15 单独报告为 ID，其余六条件等权形成 OOD。各数据集单独计算，合并值对数据集等权，不能按图数混合替代。", "",
             "CBSD68 前 20 张是本轮开发集，剩余 48 张与 Kodak24 是本轮保留集。本研究上一轮已评测完整 CBSD68/Kodak24，因此这些不是整个研究从未见过的测试数据。确认阶段在全部六组最终检查点固定后才访问保留集。", "",
             "## 绝对得分", "", "C0 只计一次；B0/B1/B3 的下表数值为续训种子均值。", "",
             "| 方法 | 数据集 | ID PSNR | OOD PSNR | OOD SSIM |", "|---|---|---:|---:|---:|"]
    score_lookup = {(r["method"], r["training_seed"], r["dataset"], r["category"]): r for r in scores}
    for method in ("C0", "B0", "B1", "B3"):
        for dataset in sorted({r["dataset"] for r in scores}):
            seed = "C0" if method == "C0" else "mean"
            ident = score_lookup[(method, seed, dataset, "id")]
            ood = score_lookup[(method, seed, dataset, "ood")]
            lines.append(f"| {method} | {dataset} | {ident['psnr']:.4f} | {ood['psnr']:.4f} | {ood['ssim']:.6f} |")
    lines.extend(["", "## 配对改善", "", "下表为相对同训练种子 B0 的差值。区间是配对图像 bootstrap 95% 区间，不能解释为训练种子总体显著性。均值行先在同一图像上平均训练种子，再以图像为独立单位抽样。", "",
             "| 方法 | 数据集 | 训练种子 | OOD ΔPSNR (dB) | 图像 95% 区间 | ID ΔPSNR (dB) | OOD ΔSSIM |", "|---|---|---|---:|---|---:|---:|"]
    )
    lookup = {(r["method"], r["baseline"], r["training_seed"], r["dataset"], r["category"], r["metric"]): r for r in statistics}
    for method in ("B1", "B3"):
        for dataset in sorted({r["dataset"] for r in statistics}):
            for seed in [*(str(s) for s in STAGE_SEEDS[stage]), "mean"]:
                prefix = (method, "B0", seed, dataset)
                ood, id_value, ssim = [lookup[(*prefix, cat, met)] for cat, met in (("ood", "psnr"), ("id", "psnr"), ("ood", "ssim"))]
                lines.append(f"| {method} | {dataset} | {seed} | {ood['difference']:+.4f} | [{ood['ci95_low']:+.4f}, {ood['ci95_high']:+.4f}] | {id_value['difference']:+.4f} | {ssim['difference']:+.6f} |")
    lines.extend(["", "## 预设判断与负结果", "", "预设工程判据：合并 OOD 平均 PSNR 至少增加 0.1 dB；确认阶段两个训练种子均同向增加；各训练种子的合并 ID 降幅不超过 0.05 dB。图像区间另列，不将其替代训练种子重复。"])
    for method, verdict in verdicts.items():
        result = "达到" if verdict["criterion_met"] else "未达到"
        qualifier = "本轮确认判据" if stage == "confirm" else "单种子筛选数值门槛，仍需两个种子确认"
        lines.append(f"- {method}：{result}{qualifier}；OOD≥0.1 dB={verdict['ood_gain_ge_0_1_db']}，所有种子 OOD>0={verdict['all_training_seeds_ood_positive']}，ID 保护条件={verdict['every_training_seed_id_ge_minus_0_05_db']}。")
        values = [r for r in variability if r["method"] == method and r["baseline"] == "B0" and r["dataset"] == "equal_dataset" and r["category"] == "ood" and r["metric"] == "psnr"][0]
        sd = values["sample_sd"]
        lines.append(f"  OOD 配对差值训练种子均值 {values['mean_difference']:+.4f} dB；样本标准差 " + (f"{sd:.4f} dB。" if sd != "" else "单种子不定义。"))
    paired = read_csv(output / "paired_per_image.csv")
    negatives = [r for r in paired if r["baseline"] == "B0" and float(r["delta_psnr"]) < 0]
    lines.extend(["", f"相对 B0，保留 {len(negatives):,} 条负 PSNR 差值（按训练种子×图像×条件，已平均噪声重复）。见 [negative_cases.csv](negative_cases.csv)，不删除退化样本。",
                  "", "## 证据范围", "", "本轮是从同一个已训练 C0 继续训练的比较；不是独立从头重训。两个确认种子改变续训随机性，但共同初始化来自同一个 C0，不能宣称覆盖初始化不确定性。有限种子、开发集筛选和先前测试暴露均限制结论。", "",
                  "结构损失的训练标量不能与纯 L1 总损失直接比较。A/C 不改变推理网络的参数结构，结果只依据统一图像指标与固定样例判断，不从训练损失推导去噪效果。", "",
                  "## 文件", "", "- [完整逐图数据](per_image.csv)", "- [条件均值](per_condition.csv)",
                  "- [ID/OOD绝对得分](aggregate_scores.csv)", "- [逐图配对差值](paired_per_image.csv)", "- [配对图像区间](paired_summary.csv)",
                  "- [训练种子波动](training_seed_variability.csv)", "- [固定样例和局部](figures/index.md)",
                  "- [机器可读核验与判据](summary.json)", ""])
    (output / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_SEEDS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "reports/ac/evaluations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected_jobs = [("C0", "C0")] + [(m, str(s)) for m in ("B0", "B1", "B3") for s in STAGE_SEEDS[args.stage]]
    output = args.output or ROOT / "reports/ac" / args.stage
    jobs, summaries, rows = {}, [], []
    for method, seed in expected_jobs:
        job = args.root / args.stage / method / f"seed{seed}"
        summary = read_json(job / "summary.json")
        if not summary.get("completed") or summary.get("diagnostic") or summary.get("stage") != args.stage:
            raise ValueError(f"Incomplete/diagnostic/wrong-stage evaluation: {job}")
        identity = summary["identity"]
        if identity["method"] != method or str(identity["training_seed"]) != seed:
            raise ValueError("Incorrect method/training-seed identity")
        jobs[(method, seed)] = job
        summaries.append(summary)
        job_rows = read_csv(job / "per_image.csv")
        if len(job_rows) != summary["rows"] or any(r["method"] != method or str(r["training_seed"]) != seed for r in job_rows):
            raise ValueError("CSV/evaluation identity mismatch")
        rows.extend(job_rows)
    for field in ("data_hash", "contract_hash", "protocol_hash"):
        if len({s[field] for s in summaries}) != 1:
            raise ValueError(f"Evaluation {field} differs across methods")
    if summaries[0]["protocol_hash"] != digest(PROTOCOL):
        raise ValueError("Changed evaluation protocol")
    rows = audit_rows(rows, args.stage)
    averaged = average_noise_seeds(rows)
    conditions = condition_table(averaged)
    scores = aggregate_scores(conditions)
    pairs = paired_image_table(averaged)
    statistics = paired_statistics(pairs)
    variability = seed_variability(statistics)
    verdicts = decisions(statistics, args.stage)
    write_csv(output / "per_image.csv", rows)
    write_csv(output / "per_image_noise_averaged.csv", averaged)
    write_csv(output / "per_condition.csv", conditions)
    write_csv(output / "aggregate_scores.csv", scores)
    write_csv(output / "paired_per_image.csv", pairs)
    write_csv(output / "paired_summary.csv", statistics)
    write_csv(output / "training_seed_variability.csv", variability)
    negative = [r for r in pairs if r["baseline"] == "B0" and r["delta_psnr"] < 0]
    write_csv(output / "negative_cases.csv", negative, fields=list(pairs[0]))
    panels = build_visual_panels(jobs, output, args.stage)
    render_report(output, args.stage, rows, scores, statistics, variability, verdicts, panels)
    atomic_json(output / "summary.json", {"completed": True, "stage": args.stage, "completed_at": utc_now(),
                "rows": len(rows), "expected_rows": 1680 if args.stage == "screen" else 10584,
                "distinct_checkpoints": len(jobs), "visual_panels": panels, "decisions": verdicts,
                "protocol": PROTOCOL, "training_seeds": STAGE_SEEDS[args.stage],
                "training_seed_uncertainty": "sample SD of paired training-seed differences, separate from image bootstrap",
                "sources": [{"directory": str(jobs[(s['identity']['method'], str(s['identity']['training_seed']))]),
                             "identity": s["identity"], "contract_hash": s["contract_hash"], "data_hash": s["data_hash"]}
                            for s in summaries]})
    print(f"Complete {args.stage}: {len(rows)} rows, {panels} fixed panels; report {output / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
