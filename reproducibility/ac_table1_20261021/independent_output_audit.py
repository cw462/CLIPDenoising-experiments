"""Independent read-only audit. Kept outside the active experiment source tree."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

EXPECTED_SOURCE = "733f3f9b0ace567f89121b15f10db812515179b63bf7953f35f11720251babf7"
EXPECTED_C0 = "123b9091a9c13b17da0e40f25f706efc8e5044ce5d8295bf5701eb1dd8f99675"
CONDITIONS = {("gaussian", 15.0), ("gaussian", 25.0), ("gaussian", 50.0),
              ("spatial_gaussian", 45.0), ("spatial_gaussian", 55.0),
              ("poisson", 2.5), ("poisson", 3.5)}
SEEDS = {"screen": [100], "confirm": [101, 102]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def contract(config):
    train, network, runtime = (copy.deepcopy(config[field]) for field in ("train", "network_g", "runtime"))
    train.pop("pixel_opt")
    network.pop("pfa_rho", None)
    network.pop("pfa_seed", None)
    for key in ("data_rng_seed", "sampler_seed", "independent_pfa_seed"):
        runtime.pop(key, None)
    plan = config.get("plan_a", config.get("plan_c"))
    return {"c0_sha256": plan["c0_sha256"], "new_optimizer": plan["new_optimizer"],
            "source_hash": config["source_hash"], "data_manifest_hash": config["data_manifest_hash"],
            "pretrained_sha256": config["pretrained_sha256"], "network": network,
            "train": train, "train_dataset": config["datasets"]["train"], "runtime": runtime}


def load_log(path):
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def training_audit(root, stage, final):
    files = sorted([*root.glob("basicsr/**/*.py"), root / "Denoising/utils.py",
                    *root.glob("tools/*.py"), *root.glob("tests/*.py")])
    actual_source = digest({p.relative_to(root).as_posix(): sha(p) for p in files if p.name != "remote_control.py"})
    assert actual_source == EXPECTED_SOURCE, ("source changed", actual_source)
    c0 = root / "pretrained/synthetic_baseline.pth"
    actual_c0 = sha(c0)
    assert actual_c0 == EXPECTED_C0, ("C0 mismatch", actual_c0)
    results, configs, sequences, completion_times, checkpoint_sha = [], {}, {}, {}, {}
    budget = 5000 if stage == "screen" else 20000
    common = []
    for seed in SEEDS[stage]:
        for method in ("B0", "B1", "B3"):
            run = root / "runs/ac" / stage / method / f"seed{seed}"
            if not (run / "effective_config.json").exists() and not final:
                results.append({"method": method, "seed": seed, "status": "not_started"})
                continue
            config, manifest, status = (read(run / f"{name}.json") for name in ("effective_config", "manifest", "status"))
            raw = dict(config)
            expected = raw.pop("config_hash")
            assert digest(raw) == expected, "Effective config hash"
            assert config["source_hash"] == EXPECTED_SOURCE
            assert config["seed"] == seed and config["stage"] == stage and config["exp"] == method
            assert config["train"]["total_iter"] == budget
            assert config["runtime"]["micro_batch"] == 16 and not config["runtime"]["amp"]
            assert config["runtime"]["train_image_count"] == 432
            assert config["runtime"]["accumulation"] == 1 and config["runtime"]["cache_decoded_images"]
            assert config["network_g"]["pfa_rho"] == (0.5 if method == "B1" else 0)
            assert config["network_g"]["pfa_seed"] == seed + 30000
            assert config["train"]["optim_g"]["lr"] == 1e-5
            assert config["train"]["scheduler"]["eta_mins"] == [1e-6]
            assert config["train"]["pixel_opt"]["type"] == ("L1StructureLoss" if method == "B3" else "L1Loss")
            if method == "B3":
                assert config["train"]["pixel_opt"]["structure_weight"] == 0.05
                assert config["train"]["pixel_opt"]["eps"] == 0.001
            assert config["paired_protocol_hash"] == digest(config["paired_protocol"])
            assert manifest["paired_protocol_hash"] == config["paired_protocol_hash"]
            assert not config["continuation"]["optimizer_state_from_c0"]
            assert config["data_manifest_hash"] == sha(root / "data/manifest.json")
            common.append(contract(config))
            assert common[-1]["c0_sha256"] == EXPECTED_C0 and common[-1]["new_optimizer"]
            configs[(method, seed)] = config
            logs = load_log(run / "train.jsonl")
            assert [r["update"] for r in logs] == list(range(1, len(logs) + 1)), "Noncontiguous training updates"
            assert all(math.isfinite(r["loss"]) and r["seconds"] > 0 for r in logs)
            sequences[(method, seed)] = [(r["update"], r["input_sha256"], r["target_sha256"]) for r in logs]
            entry = {"method": method, "seed": seed, "status": status["status"], "logged_updates": len(logs),
                     "budget": budget, "config_hash": expected, "paired_protocol_hash": config["paired_protocol_hash"]}
            if final:
                assert status["status"] == "completed" and status["update"] == budget
                assert len(logs) == budget
                checks, summary, latest = (read(run / name) for name in
                                          ("training_checks.json", "summary.json", "checkpoints/latest.json"))
                for field in ("strict_c0_params", "fresh_adamw", "fresh_scheduler", "encoder_parameters_frozen", "fp32",
                              "encoder_parameters_unchanged", "encoder_gradients_absent", "last_checkpoint_sha256_verified", "complete"):
                    assert checks[field], (method, seed, field)
                metadata = checks["c0_metadata"]
                assert (metadata["experiment"], metadata["stage"], metadata["seed"], metadata["update"]) == ("B0", "full", 0, 300000)
                assert metadata["source_hash"] == "40a2b826983afbcac68e0b4419d759272bbd0703f6614f092301bc3148489eef"
                checkpoint = Path(latest["path"])
                assert checkpoint.resolve().parent == (run / "checkpoints").resolve()
                assert sha(checkpoint) == latest["sha256"] == summary["checkpoint_sha256"]
                assert latest["update"] == budget
                assert summary["completed"] and summary["updates"] == budget
                assert math.isclose(logs[0]["lr"], 1e-5, abs_tol=1e-15)
                assert math.isclose(logs[-1]["lr"], 1e-6, abs_tol=1e-15)
                assert all(a["lr"] > b["lr"] for a, b in zip(logs, logs[1:]))
                completion_times[(method, seed)] = summary["ended_at"]
                checkpoint_sha[(method, str(seed))] = latest["sha256"]
                entry.update(ended_at=summary["ended_at"], checkpoint_sha256=latest["sha256"])
            results.append(entry)
    assert len({digest(item) for item in common}) <= 1, "Common eval/training contract mismatch"
    for seed in SEEDS[stage]:
        present = [method for method in ("B0", "B1", "B3") if (method, seed) in configs]
        assert len({configs[(method, seed)]["paired_protocol_hash"] for method in present}) <= 1
        common_steps = min((len(sequences[(m, seed)]) for m in present), default=0)
        assert len({digest(sequences[(m, seed)][:common_steps]) for m in present}) <= 1, "Unpaired input/target sequence"
    eval_root = root / "reports/ac/evaluations" / stage
    observed_access = list(eval_root.glob("*/seed*/access_gate.json"))
    if observed_access:
        assert all(item["status"] == "completed" for item in results), "Evaluation before all runs completed"
    return {"source_hash": actual_source, "c0_sha256": actual_c0, "runs": results,
            "common_contract_hash": digest(common[0]) if common else None,
            "observed_evaluation_gates": len(observed_access)}, completion_times, checkpoint_sha


def final_output_audit(root, stage, training, completion_times, checkpoint_sha):
    import numpy as np

    datasets = {"CBSD68_dev20": 20} if stage == "screen" else {"CBSD68_holdout48": 48, "Kodak24": 24}
    expected_jobs = [("C0", "C0")] + [(m, str(s)) for m in ("B0", "B1", "B3") for s in SEEDS[stage]]
    rows, evaluation_summaries, gate_times = [], [], []
    for method, seed in expected_jobs:
        directory = root / "reports/ac/evaluations" / stage / method / f"seed{seed}"
        summary, gate = read(directory / "summary.json"), read(directory / "access_gate.json")
        assert summary["completed"] and not summary["diagnostic"] and summary["stage"] == stage
        assert summary["identity"]["method"] == method and str(summary["identity"]["training_seed"]) == seed
        assert summary["identity"]["sha256"] == (EXPECTED_C0 if method == "C0" else checkpoint_sha[(method, seed)])
        assert summary["contract_hash"] == training["common_contract_hash"] == digest(summary["contract"])
        assert summary["data_hash"] == digest(summary["images"])
        assert summary["protocol_hash"] == digest(summary["protocol"])
        assert summary["protocol"]["noise_seeds"] == [0, 1, 2]
        assert gate["source_hash"] == EXPECTED_SOURCE
        assert len(gate["all_runs_complete"]) == 3 * len(SEEDS[stage])
        assert gate["timestamp"] >= max(completion_times.values()), "Holdout was accessed early"
        gate_times.append(gate["timestamp"])
        job_rows = csv_rows(directory / "per_image.csv")
        assert len(job_rows) == summary["rows"] == sum(datasets.values()) * 7 * 3
        assert all(r["method"] == method and r["training_seed"] == seed for r in job_rows)
        rows.extend(job_rows)
        evaluation_summaries.append(summary)
    assert len({s["data_hash"] for s in evaluation_summaries}) == 1
    assert len({s["protocol_hash"] for s in evaluation_summaries}) == 1
    assert len(rows) == (1680 if stage == "screen" else 10584)
    cells, provenance, image_sets = set(), {}, defaultdict(set)
    sums = defaultdict(list)
    for row in rows:
        method, seed, dataset, image = (row[key] for key in ("method", "training_seed", "dataset", "image"))
        family, level, noise_seed = row["family"], float(row["level"]), int(row["noise_seed"])
        assert dataset in datasets and (family, level) in CONDITIONS and noise_seed in (0, 1, 2)
        assert all(math.isfinite(float(row[metric])) for metric in ("psnr", "ssim"))
        key = (method, seed, dataset, image, family, level, noise_seed)
        assert key not in cells, "Duplicate cell"
        cells.add(key)
        input_key = key[2:]
        hashes = row["input_sha256"], row["clean_sha256"]
        assert input_key not in provenance or provenance[input_key] == hashes, "Unpaired noisy input or GT"
        provenance[input_key] = hashes
        image_sets[(method, seed, dataset)].add(image)
        sums[key[:-1]].append((float(row["psnr"]), float(row["ssim"])))
    for (method, seed) in expected_jobs:
        for dataset, count in datasets.items():
            assert len(image_sets[(method, seed, dataset)]) == count
            assert image_sets[(method, seed, dataset)] == image_sets[("C0", "C0", dataset)]
    assert all(len(items) == 3 for items in sums.values()), "Missing noise repeat"
    averaged = {key: tuple(statistics.mean(v[i] for v in items) for i in (0, 1)) for key, items in sums.items()}
    result = root / "reports/ac" / stage
    merged = csv_rows(result / "per_image.csv")
    def normalized(row):
        return {key: float(value) if key in ("level", "psnr", "ssim") else int(value) if key == "noise_seed" else value
                for key, value in row.items()}
    assert list(map(normalized, merged)) == list(map(normalized, rows)), "Merged per_image output differs from job outputs"
    condition_values = defaultdict(list)
    for (method, seed, dataset, image, family, level), values in averaged.items():
        condition_values[(method, seed, dataset, family, level)].append(values)
    conditions = {key: tuple(statistics.mean(v[i] for v in items) for i in (0, 1)) for key, items in condition_values.items()}
    cond_rows = csv_rows(result / "per_condition.csv")
    assert len(cond_rows) == len(conditions)
    for row in cond_rows:
        key = row["method"], row["training_seed"], row["dataset"], row["family"], float(row["level"])
        assert all(math.isclose(float(row[metric]), conditions[key][index], abs_tol=1e-10)
                   for index, metric in enumerate(("psnr", "ssim")))
    pair_expected = {}
    for (method, seed, dataset, image, family, level), values in averaged.items():
        if method == "C0":
            continue
        for baseline in (["C0"] if method == "B0" else ["B0", "C0"]):
            base = averaged[(baseline, "C0" if baseline == "C0" else seed, dataset, image, family, level)]
            pair_expected[(method, baseline, seed, dataset, image, family, level)] = tuple(v - b for v, b in zip(values, base))
    pair_rows = csv_rows(result / "paired_per_image.csv")
    assert len(pair_rows) == len(pair_expected)
    for row in pair_rows:
        key = tuple(row[k] for k in ("method", "baseline", "training_seed", "dataset", "image", "family")) + (float(row["level"]),)
        assert all(math.isclose(float(row["delta_" + metric]), pair_expected[key][i], abs_tol=1e-10)
                   for i, metric in enumerate(("psnr", "ssim")))
    units = defaultdict(list)
    for (method, baseline, seed, dataset, image, family, level), values in pair_expected.items():
        categories = ["id"] if family == "gaussian" and level == 15 else (["ood", "spatial"] if family == "spatial_gaussian" else ["ood"])
        for category in categories:
            units[(method, baseline, seed, dataset, image, category)].append(values)
    image_values = {key: tuple(statistics.mean(v[i] for v in items) for i in (0, 1)) for key, items in units.items()}
    train_averaged = defaultdict(list)
    for (method, baseline, seed, dataset, image, category), values in image_values.items():
        train_averaged[(method, baseline, "mean", dataset, image, category)].append(values)
    image_values.update({key: tuple(statistics.mean(v[i] for v in items) for i in (0, 1)) for key, items in train_averaged.items()})
    bootstrap_units = defaultdict(dict)
    for (method, baseline, seed, dataset, image, category), values in image_values.items():
        for index, metric in enumerate(("psnr", "ssim")):
            for scope in (dataset, "equal_dataset"):
                bootstrap_units[(method, baseline, seed, scope, category, metric)][(dataset, image)] = values[index]
    calculated = {}
    for key, image_map in bootstrap_units.items():
        by_dataset = defaultdict(list)
        for (dataset, image), value in sorted(image_map.items()):
            by_dataset[dataset].append(value)
        rng = np.random.RandomState(2026)
        draws = [rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                 for dataset, values in sorted(by_dataset.items())]
        interval = np.percentile(np.mean(draws, axis=0), [2.5, 97.5])
        calculated[key] = (statistics.mean(statistics.mean(v) for v in by_dataset.values()), *interval, len(image_map))
    summary_rows = csv_rows(result / "paired_summary.csv")
    assert len(summary_rows) == len(calculated)
    for row in summary_rows:
        key = tuple(row[k] for k in ("method", "baseline", "training_seed", "dataset", "category", "metric"))
        values = calculated[key]
        assert all(math.isclose(float(row[field]), values[i], abs_tol=1e-10)
                   for i, field in enumerate(("difference", "ci95_low", "ci95_high"))), (key, values)
        assert int(row["independent_images"]) == values[3], "Pseudoreplication"
    variability = csv_rows(result / "training_seed_variability.csv")
    for row in variability:
        values = [calculated[(row["method"], row["baseline"], str(seed), row["dataset"], row["category"], row["metric"])][0] for seed in SEEDS[stage]]
        assert int(row["n_training_seeds"]) == len(values)
        assert math.isclose(float(row["mean_difference"]), statistics.mean(values), abs_tol=1e-10)
        assert row["all_seeds_positive"] == str(all(value > 0 for value in values))
        if len(values) > 1:
            assert math.isclose(float(row["sample_sd"]), statistics.stdev(values), abs_tol=1e-10)
        else:
            assert row["sample_sd"] == ""
    summary = read(result / "summary.json")
    assert summary["completed"] and summary["rows"] == len(rows)
    for method in ("B1", "B3"):
        ood = calculated[(method, "B0", "mean", "equal_dataset", "ood", "psnr")]
        ood_values = [calculated[(method, "B0", str(seed), "equal_dataset", "ood", "psnr")][0] for seed in SEEDS[stage]]
        id_values = [calculated[(method, "B0", str(seed), "equal_dataset", "id", "psnr")][0] for seed in SEEDS[stage]]
        decision = summary["decisions"][method]
        expected_flags = {"ood_gain_ge_0_1_db": ood[0] >= .1,
                          "all_training_seeds_ood_positive": all(value > 0 for value in ood_values),
                          "every_training_seed_id_ge_minus_0_05_db": all(value >= -.05 for value in id_values),
                          "paired_image_ci_excludes_zero": ood[1] > 0}
        assert all(decision[key] == value for key, value in expected_flags.items())
        assert decision["criterion_met"] == all(expected_flags[key] for key in expected_flags if key != "paired_image_ci_excludes_zero")
        assert decision["final_confirmation"] == (stage == "confirm")
    negative = csv_rows(result / "negative_cases.csv")
    expected_negative = [row for row in pair_rows if row["baseline"] == "B0" and float(row["delta_psnr"]) < 0]
    assert negative == expected_negative
    expected_panels = 12 if stage == "screen" else 24
    panels = list((result / "figures").glob("*.png"))
    assert len(panels) == expected_panels == summary["visual_panels"]
    for panel in panels:
        with panel.open("rb") as stream:
            assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
    report = (result / "experiment_report.md").read_text(encoding="utf-8")
    assert "整个研究从未见过" in report and "不能解释为训练种子总体显著性" in report
    return {"rows": len(rows), "unique_cells": len(cells), "noise_averaged_cells": len(averaged),
            "condition_rows": len(conditions), "paired_rows": len(pair_expected), "paired_summary_rows": len(calculated),
            "negative_rows": len(negative), "panels": len(panels), "decisions": summary["decisions"],
            "last_training_end_utc": max(completion_times.values()), "first_evaluation_access_utc": min(gate_times),
            "paired_metrics_and_bootstrap_recomputed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/imss/cw/CLIPDenoising_AC_20261021"))
    parser.add_argument("--stage", choices=tuple(SEEDS), default="screen")
    parser.add_argument("--mode", choices=("preflight", "final"), default="preflight")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    training, completion, checksums = training_audit(args.root, args.stage, args.mode == "final")
    result = {"status": "passed", "mode": args.mode, "stage": args.stage, "training": training}
    if args.mode == "final":
        result["evaluation"] = final_output_audit(args.root, args.stage, training, completion, checksums)
    body = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
