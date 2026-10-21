"""Summarize completed G1C seeds 0/1/2 against same-seed historical G1."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_c_only import (DATASET_COUNTS, TABLE1_CONDITIONS, UPDATES, cell_key,
                                    historical_baseline, historical_index, protocol_for_seed,
                                    validate_recovery_disclosure)
from tools.evaluation_utils import read_csv, write_csv
from tools.experiment_common import atomic_json, digest, read_json, sha256_file, utc_now
from tools.summarize_experiments import aggregate, hierarchical_ood

SEEDS = (0, 1, 2)
PROTOCOL = {
    "version": "g1c-three-seed-historical-v1", "variant": "G1C", "baseline": "G1",
    "training_seeds": list(SEEDS), "training_updates": UPDATES,
    "pairing": "G1C seed k versus historical G1 seed k; identical evaluation image/noise bytes",
    "initialization": "Same named initialization seed; fresh RN50-based training for every new seed",
    "trajectory_limit": "Historical per-update samples and RNG trajectories were not recorded for this comparison and cannot be retrospectively verified bitwise. Same-seed pairing does not establish identical full training trajectories.",
    "historical_limit": "Baseline scores are reused historical measurements, not newly trained concurrent controls; numerical execution/source differences remain.",
    "seed_statistics": "First average images within each seed/condition, then equal-weight the three training seeds; sample standard deviation uses ddof=1. Paired delta standard deviation is computed from the three same-seed deltas.",
    "ood": "Exclude Gaussian15; equal-weight strengths within families, then families and datasets; official three-family and five-family extension results are separate.",
    "data_reuse": "All four datasets were evaluated previously; full CBSD68 includes development images. This is not an unseen holdout.",
    "claims": "Descriptive three-seed results only; no automatic significance, causal innovation, SOTA or screenshot-reference claim.",
    "readback_recovery": "Every enabled observation-copy recovery is counted per seed and SHA-bound to completed training. Root cause remains unresolved; passing protection gates does not mean zero readback anomalies.",
    "table1": TABLE1_CONDITIONS,
}


def validate_rows(rows, reference, identity, manifest, manifest_hash):
    clean = {(x["dataset"], x["image_id"]): x["sha256"] for x in manifest}
    seen, result = set(), []
    method, seed = identity["method"], identity["training_seed"]
    if method not in ("G1", "G1C") or seed not in SEEDS:
        raise ValueError("Wrong multi-seed evaluation identity")
    for source in rows:
        row = dict(source)
        key = cell_key(row)
        if key in seen:
            raise ValueError("Duplicate per-image measurement")
        if key not in reference:
            raise ValueError("Unexpected historical input cell")
        seen.add(key)
        if (row["experiment"], int(row["train_seed"]), row["stage"], int(row["update"])) != (method, seed, "full", UPDATES):
            raise ValueError("Wrong per-image method/seed/budget identity")
        for field in ("checkpoint_sha256", "config_hash", "source_hash", "git_commit", "measurement_origin"):
            if row[field] != identity[field]:
                raise ValueError(f"Per-image identity mismatch: {field}")
        expected = reference[key]
        if (row["noise_sha256"], int(row["noise_seed"]), int(row["height"]), int(row["width"])) != (
                expected["noise_sha256"], expected["noise_seed"], expected["height"], expected["width"]):
            raise ValueError("Noisy input differs from historical benchmark")
        if row["dataset_manifest_hash"] != manifest_hash:
            raise ValueError("Dataset manifest identity mismatch")
        if method == "G1C" and row.get("clean_sha256") != clean[(key[0], key[1])]:
            raise ValueError("Clean image SHA256 mismatch")
        domain = "ID" if key[2:] == ("gaussian", 15.0) else "OOD"
        group = "extension" if key[2] in {"speckle", "salt_pepper"} else "official"
        if row["domain"] != domain or row["protocol_group"] != group:
            raise ValueError("Wrong domain/protocol group")
        row.update(train_seed=seed, noise_level=key[3], noise_seed=int(row["noise_seed"]), update=UPDATES)
        for metric in ("psnr", "ssim"):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError("Nonfinite metric")
        result.append(row)
    if len(result) != 3150 or seen != set(reference):
        raise ValueError("Missing historical benchmark cells")
    return result


def validate_completion_contract(contract, identity, seed):
    if (identity["method"], identity["training_seed"], identity["stage"], identity["update"]) != ("G1C", seed, "full", UPDATES):
        raise ValueError("Evaluation model identity mismatch")
    if (contract["variant"], contract["baseline"], contract["training_seed"], contract["training_updates"]) != ("G1C", "G1", seed, UPDATES):
        raise ValueError("Training contract identity mismatch")
    if any(identity[field] != contract[field] for field in ("source_hash", "config_hash")):
        raise ValueError("Training/evaluation source/config mismatch")
    checks, proof = contract.get("training_checks", {}), contract.get("input_protection", {})
    if (checks.get("complete"), checks.get("update"), checks.get("verified_updates")) != (True, UPDATES, UPDATES):
        raise ValueError("Incomplete final training checks")
    for field in ("encoder_parameters_unchanged", "encoder_gradients_absent", "last_checkpoint_sha256_verified", "input_protection_passed"):
        if checks.get(field) is not True:
            raise ValueError("Training protection checks failed")
    if (proof.get("schema"), proof.get("status"), proof.get("update"), proof.get("verified_updates")) != (
            "c_only_input_protection_v1", "passed", UPDATES, UPDATES):
        raise ValueError("Incomplete final input protection proof")
    if any(proof.get(field) != contract[field] for field in ("source_hash", "config_hash")):
        raise ValueError("Input proof/source/config mismatch")
    validate_recovery_disclosure(contract)


def across_seeds(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, members in sorted(grouped.items()):
        if sorted(int(x["train_seed"]) for x in members) != list(SEEDS):
            raise ValueError("Every aggregate requires exactly seeds 0, 1, 2")
        row = dict(zip(keys, key), training_seed_count=3)
        for field in metrics:
            values = [float(x[field]) for x in members]
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Nonfinite seed statistic")
            row[field + "_mean"] = statistics.mean(values)
            row[field + "_std"] = statistics.stdev(values)
        result.append(row)
    return result


def summarize(evaluation_root, output, historical_csv=None):
    evaluation_root, output = Path(evaluation_root), Path(output)
    summaries = {seed: read_json(evaluation_root / "G1C" / f"seed{seed}" / "summary.json") for seed in SEEDS}
    historical_csv = Path(historical_csv or summaries[0]["historical_csv"]["path"])
    reference, historical = historical_index(historical_csv)
    rows, pairs, identities, contracts, panels, recovery_rows = [], [], [], [], [], []
    sources, new_checkpoints, baseline_checkpoints = set(), set(), set()
    for seed in SEEDS:
        directory, item = evaluation_root / "G1C" / f"seed{seed}", summaries[seed]
        if not item.get("completed") or (item.get("rows"), item.get("baseline_rows"), item.get("matched_historical_inputs")) != (3150, 3150, 3150):
            raise ValueError("Incomplete seed evaluation")
        protocol = protocol_for_seed(seed)
        if item.get("protocol_hash") != digest(protocol) or digest(item["protocol"]) != digest(protocol):
            raise ValueError("Evaluation protocol mismatch")
        if item["historical_csv"]["sha256"] != historical["sha256"]:
            raise ValueError("Historical CSV SHA256 mismatch")
        contract, identity = item["contract"], item["identity"]
        if item["contract_hash"] != digest(contract):
            raise ValueError("Training contract hash mismatch")
        validate_completion_contract(contract, identity, seed)
        recovery = validate_recovery_disclosure(contract)
        recovery_rows.append({"train_seed": seed, "policy": recovery["policy"],
                              "recovered_observation_events": recovery["events_total"],
                              "initialization_events": recovery["initialization_events"],
                              "training_events": recovery["training_events"],
                              "affected_training_updates": recovery["affected_training_updates"],
                              "event_log_sha256": recovery["log_sha256"] or "",
                              "all_logged_events_independently_confirmed": True,
                              "root_cause_established": False})
        sources.add(identity["source_hash"])
        contracts.append(contract)
        new_checkpoints.add(identity["checkpoint_sha256"])
        artifacts = item["artifact_sha256"]
        if not {"per_image_results.csv", "baseline_per_image_results.csv", "access_gate.json"}.issubset(artifacts):
            raise ValueError("Missing SHA-bound evaluation artifacts")
        for name, expected in artifacts.items():
            file = directory / name
            if not file.resolve().is_relative_to(directory.resolve()) or sha256_file(file) != expected:
                raise ValueError("Evaluation artifact SHA256/path mismatch")
        gate = read_json(directory / "access_gate.json")
        if (gate.get("variant"), gate.get("training_seed", 0), gate.get("source_hash"), gate.get("contract_hash")) != (
                "G1C", seed, identity["source_hash"], item["contract_hash"]):
            raise ValueError("Model access gate identity mismatch")
        manifest_hash = hashlib.sha256(json.dumps(item["dataset_manifest"], sort_keys=True).encode()).hexdigest()
        if manifest_hash != item["dataset_manifest_hash"] or manifest_hash != historical["dataset_manifest_hash"]:
            raise ValueError("Dataset manifest identity mismatch")
        old_rows, old_identity = historical_baseline(historical_csv, "G1", reference, seed=seed)
        if old_identity != item["baseline_identity"]:
            raise ValueError("Historical same-seed baseline identity mismatch")
        proof = item["baseline_checkpoint_proof"]
        if proof.get("identity_verified") is not True or proof["checkpoint_sha256"] != old_identity["checkpoint_sha256"]:
            raise ValueError("Historical checkpoint identity was not verified")
        baseline_checkpoints.add(old_identity["checkpoint_sha256"])
        new = validate_rows(read_csv(directory / "per_image_results.csv"), reference, identity, item["dataset_manifest"], manifest_hash)
        old = validate_rows(read_csv(directory / "baseline_per_image_results.csv"), reference, old_identity, item["dataset_manifest"], manifest_hash)
        original = {cell_key(x): x for x in old_rows}
        old_index = {cell_key(x): x for x in old}
        for row in old:
            if any(row[field] != original[cell_key(row)][field] for field in ("psnr", "ssim", "checkpoint_sha256", "config_hash", "source_hash")):
                raise ValueError("Reused baseline scores or identity were altered")
        for row in new:
            base = old_index[cell_key(row)]
            if any(row[field] != base[field] for field in ("noise_sha256", "noise_seed", "dataset_manifest_hash")):
                raise ValueError("Unpaired evaluation inputs")
            pairs.append({"experiment": "G1C", "baseline": "G1", "train_seed": seed,
                          **{k: row[k] for k in ("dataset", "image_id", "noise_type", "noise_level", "noise_seed", "noise_sha256", "protocol_group", "domain")},
                          "psnr": row["psnr"], "baseline_psnr": base["psnr"], "delta_psnr": row["psnr"] - base["psnr"],
                          "ssim": row["ssim"], "baseline_ssim": base["ssim"], "delta_ssim": row["ssim"] - base["ssim"],
                          "checkpoint_sha256": row["checkpoint_sha256"], "baseline_checkpoint_sha256": base["checkpoint_sha256"]})
        rows.extend(old + new)
        identities.extend((old_identity, identity))
        for dataset, family, level in TABLE1_CONDITIONS:
            name = f"panels/{dataset}_{family}_{level:g}_panel.png"
            if name not in artifacts:
                raise ValueError("Missing fixed Table1 visualization")
            panels.append((directory / name, f"panels/seed{seed}/{Path(name).name}"))
    if len(sources) != 1 or len(new_checkpoints) != 3 or len(baseline_checkpoints) != 3:
        raise ValueError("Seed models must be distinct and new evaluation sources must agree")
    for field in ("pretrained_sha256", "data_manifest_hash", "pixel_opt"):
        if len({digest(c[field]) for c in contracts}) != 1:
            raise ValueError(f"New seed configurations differ: {field}")
    keys = ("dataset", "noise_type", "noise_level", "protocol_group", "domain")
    metrics = ("psnr", "baseline_psnr", "delta_psnr", "ssim", "baseline_ssim", "delta_ssim")
    conditions = aggregate(rows, ("experiment", "train_seed", *keys), ("psnr", "ssim"))
    condition_seeds = [{**{k: r[k] for k in ("experiment", "train_seed", *keys)}, "psnr": r["psnr_mean"], "ssim": r["ssim_mean"]} for r in conditions]
    condition_summary = across_seeds(condition_seeds, ("experiment", *keys), ("psnr", "ssim"))
    grouped_pairs = aggregate(pairs, ("train_seed", *keys), metrics)
    pair_seeds = [{**{k: r[k] for k in ("train_seed", *keys)}, "n_images": r["n"], **{k: r[k + "_mean"] for k in metrics}} for r in grouped_pairs]
    pair_summary = across_seeds(pair_seeds, keys, metrics)
    table_seed, table_summary = [], []
    for dataset, family, level in TABLE1_CONDITIONS:
        selected = lambda r: (r["dataset"], r["noise_type"], float(r["noise_level"])) == (dataset, family, level)
        table_seed.extend(sorted((r for r in pair_seeds if selected(r)), key=lambda r: r["train_seed"]))
        table_summary.extend(r for r in pair_summary if selected(r))
    ood = hierarchical_ood(conditions)
    ood_summary = across_seeds(ood, ("experiment", "table"), ("ood_psnr", "ood_ssim"))
    ood_index = {(r["experiment"], r["train_seed"], r["table"]): r for r in ood}
    ood_pairs = []
    for seed in SEEDS:
        for table in ("official_three_families", "five_families_with_extensions"):
            new, old = (ood_index[(m, seed, table)] for m in ("G1C", "G1"))
            ood_pairs.append({"train_seed": seed, "table": table, "delta_ood_psnr": new["ood_psnr"] - old["ood_psnr"], "delta_ood_ssim": new["ood_ssim"] - old["ood_ssim"]})
    artifacts = {"per_image_results.csv": rows, "paired_per_image_results.csv": pairs,
                 "readback_recovery_seedwise.csv": recovery_rows,
                 "condition_seedwise.csv": conditions, "condition_summary.csv": condition_summary,
                 "paired_condition_seedwise.csv": pair_seeds, "paired_condition_summary.csv": pair_summary,
                 "table1_seedwise.csv": table_seed, "table1_conditions.csv": table_summary,
                 "ood_seedwise.csv": ood, "ood_summary.csv": ood_summary,
                 "ood_paired_seedwise.csv": ood_pairs, "ood_paired_summary.csv": across_seeds(ood_pairs, ("table",), ("delta_ood_psnr", "delta_ood_ssim")),
                 "id_seedwise.csv": [r for r in pair_seeds if r["domain"] == "ID"],
                 "id_summary.csv": [r for r in pair_summary if r["domain"] == "ID"],
                 "negative_cases.csv": [r for r in pairs if r["delta_psnr"] < 0]}
    for name, values in artifacts.items():
        write_csv(output / name, values, fields=list(pairs[0]) if name == "negative_cases.csv" else None)
    for source, destination in panels:
        (output / destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output / destination)
    recovery_total = sum(row["recovered_observation_events"] for row in recovery_rows)
    recovery_updates = sum(row["affected_training_updates"] for row in recovery_rows)
    recovery_initial = sum(row["initialization_events"] for row in recovery_rows)
    lines = ["# G1C vs historical G1: seeds 0, 1, 2", "",
             "All six checkpoints completed 300,000 updates. The three G1C runs passed their full per-update input protection gates; G1 scores are reused historical measurements.", "",
             f"The G1C logs record {recovery_total} independently confirmed observation-copy recovery events: {recovery_initial} during initialization and {recovery_total - recovery_initial} during training, affecting {recovery_updates} seed-specific training updates. Per-seed counts and event-log SHA256 values are in readback_recovery_seedwise.csv. Original dataset replay, Mixup replay, snapshot and optimizer input checks remained mandatory. These recoveries do not establish the root cause, which remains unresolved; passing the gates must not be described as zero anomalies.", "",
             "Each G1C seed is paired with the same historical G1 initialization seed and identical evaluation noise. Historical per-update samples/RNG trajectories were not recorded for this comparison, so full training trajectories cannot be verified as identical.", "",
             "Values below are the mean and sample standard deviation across three training seeds (ddof=1), after averaging images within each seed. Delta variability is computed from paired seed differences. All four datasets were previously evaluated; no unseen-holdout, significance, causal or SOTA claim is made.", "",
             "| Dataset | Noise | G1 PSNR mean ± SD | G1C PSNR mean ± SD | Paired delta mean ± SD (dB) |",
             "|---|---|---:|---:|---:|"]
    for r in table_summary:
        lines.append(f"| {r['dataset']} | {r['noise_type']} {r['noise_level']:g} | {r['baseline_psnr_mean']:.4f} ± {r['baseline_psnr_std']:.4f} | {r['psnr_mean']:.4f} ± {r['psnr_std']:.4f} | {r['delta_psnr_mean']:+.4f} ± {r['delta_psnr_std']:.4f} |")
    lines += ["", "SSIM, all 60 conditions, every seed, OOD/ID and negative per-image differences are retained in the CSV artifacts. Condition_seedwise standard deviations describe images; summary-file standard deviations describe training seeds. Fixed panels use the first lexicographic image in each predeclared Table1 cell for every G1C seed; no historical reconstruction images were generated.", ""]
    (output / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    files = [*artifacts, *(name for _, name in panels), "experiment_report.md"]
    result = {"completed": True, "completed_at": utc_now(), "training_seeds": list(SEEDS),
              "rows": len(rows), "new_measurement_rows": 9450, "reused_baseline_rows": 9450, "paired_rows": len(pairs),
              "condition_seedwise_rows": len(conditions), "condition_summary_rows": len(condition_summary),
              "table1_rows": len(table_summary), "table1_seedwise_rows": len(table_seed), "fixed_panels": len(panels),
              "readback_recovery_events_total": recovery_total,
              "readback_recovery_initialization_events": recovery_initial,
              "readback_recovery_affected_training_updates": recovery_updates,
              "readback_recovery_seedwise": recovery_rows, "readback_root_cause_established": False,
              "historical_csv": historical, "protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL),
              "evaluation_source_hash": sources.pop(), "checkpoint_identities": identities,
              "source_summary_sha256": {f"seed{seed}": sha256_file(evaluation_root / "G1C" / f"seed{seed}/summary.json") for seed in SEEDS},
              "artifact_sha256": {name: sha256_file(output / name) for name in files}}
    atomic_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--root", dest="input", type=Path, default=ROOT / "reports/g1c_multiseed/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/g1c_multiseed/final")
    parser.add_argument("--historical-csv", type=Path)
    args = parser.parse_args()
    result = summarize(args.input, args.output, args.historical_csv)
    print(f"Completed: {result['new_measurement_rows']} new + {result['reused_baseline_rows']} historical rows")


if __name__ == "__main__":
    main()
