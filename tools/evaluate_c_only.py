"""Evaluate full-budget C/G1C against the same-seed historical baseline."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw

from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from tools.evaluate_ac_table1 import (DATASET_COUNTS, TABLE1_CONDITIONS, benchmark_data,
                                     checked_noise, historical_index, verify_all_inputs)
from tools.evaluate_experiment import dependency_versions, infer
from tools.evaluate_pfa import save_fixed_visual
from tools.evaluation_utils import (CONDITIONS, PROTOCOL as HISTORICAL_PROTOCOL,
                                    calculate_metrics, read_csv, read_rgb, write_csv)
from tools.experiment_common import (RunLock, atomic_json, digest, load_paths, read_json,
                                     sha256_file, source_hash, torch_load, utc_now)

VARIANTS = {"C": "B0", "G1C": "G1"}
HISTORICAL_SHA256 = "d3fead06b10e8b86363fe8f17484fccbce2a3a723b96cfeb49db75524c900635"
HISTORICAL_SOURCE = "40a2b826983afbcac68e0b4419d759272bbd0703f6614f092301bc3148489eef"
UPDATES = 300000
PROTOCOL = {
    "version": "c-g1c-full300k-historical-v1", "historical": HISTORICAL_PROTOCOL,
    "evaluation_scope": "reused_four_dataset_full_historical_benchmark",
    "variants": VARIANTS, "training_seeds": [0], "training_updates": UPDATES,
    "new_rows_per_variant": 3150, "historical_rows_per_baseline": 3150,
    "table1": TABLE1_CONDITIONS,
    "baseline": "Reuse B0/G1 full seed0 final 300000-update CSV metrics; no baseline retraining or re-inference",
    "initialization": "New full training from RN50 and seed0 decoder initialization, not continuation from old final weights",
    "comparison": "C-B0 and G1C-G1 use matched evaluation inputs and equal nominal budgets; historical baseline is not a new concurrent randomized control",
    "limitations": "One seed per method; no across-seed variability, statistical stability, causal innovation, SOTA or paper-reference claims. Historical stepwise training inputs are not independently proven identical.",
    "data_reuse": "All four datasets were evaluated previously; full CBSD68 includes previous 20 development images. Not an unseen holdout.",
    "preflight": "Raw test images and existing historical CSV may be read for input provenance before training; no new model outcomes accessed",
    "visuals": "First lexicographic image per each of six predefined Table1 cells; C/G1C reconstructions only, not historical baseline reconstructions",
}


def validate_seed(variant, seed):
    if variant not in VARIANTS or type(seed) is not int or seed not in (0, 1, 2) or (variant == "C" and seed != 0):
        raise ValueError("Supported evaluation: C seed0 or G1C seeds 0, 1, 2")


def protocol_for_seed(seed):
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Historical training seed must be 0, 1 or 2")
    if seed == 0:
        return PROTOCOL
    return dict(PROTOCOL, version="c-g1c-full300k-historical-per-seed-v2", training_seeds=[seed],
                baseline=f"Reuse same-seed full G1 seed{seed} final 300000-update CSV metrics; no baseline retraining or re-inference",
                initialization=f"New full training from RN50 and seed{seed} decoder initialization, not continuation from old final weights",
                comparison="G1C-G1 pairs the same training seed and exact historical evaluation inputs at equal nominal budgets; the historical baseline is not a new concurrent randomized control",
                limitations="This artifact contains one seed. Across-seed variation requires all three completed seed artifacts; no causal innovation, SOTA or paper-reference claims. Historical stepwise training inputs are not proven identical.")


def cell_key(row):
    return row["dataset"], row["image_id"], row["noise_type"], float(row["noise_level"])


def resolve_path(paths, value):
    path = Path(value)
    return path if path.is_absolute() else Path(paths["project_root"]) / path


def load_historical(paths, csv_path=None):
    csv_path = resolve_path(paths, csv_path or paths["historical_results_csv"])
    expected = paths.get("historical_results_csv_sha256", HISTORICAL_SHA256)
    if not expected or sha256_file(csv_path) != expected:
        raise ValueError("Historical CSV differs from its pinned SHA256")
    reference, metadata = historical_index(csv_path)
    return csv_path, reference, metadata


def historical_baseline(csv_path, baseline, reference, expected_source=HISTORICAL_SOURCE, seed=0):
    """Extract one exact complete historical seed, preserving its measured scores."""
    protocol_for_seed(seed)
    if baseline not in VARIANTS.values():
        raise ValueError("Unknown historical baseline")
    selected, seen, identities = [], set(), set()
    fields = ("experiment", "train_seed", "stage", "checkpoint_sha256", "update", "git_commit",
              "config_hash", "source_hash", "dataset", "image_id", "noise_type", "noise_level",
              "noise_seed", "noise_sha256", "protocol_group", "domain", "dataset_manifest_hash",
              "height", "width", "psnr", "ssim")
    for source in read_csv(csv_path):
        if source["experiment"] != baseline or int(source["train_seed"]) != seed:
            continue
        row = {field: source[field] for field in fields}
        if row["stage"] != "full" or int(row["update"]) != UPDATES or row["source_hash"] != expected_source:
            raise ValueError("Historical baseline is not the pinned full seed/300000 source")
        key = cell_key(row)
        if key in seen:
            raise ValueError("Duplicate historical baseline measurement")
        if key not in reference:
            raise ValueError("Unexpected historical baseline cell")
        seen.add(key)
        for metric in ("psnr", "ssim"):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError("Nonfinite historical baseline metric")
        identities.add(tuple(row[field] for field in ("checkpoint_sha256", "config_hash", "source_hash", "git_commit")))
        row["measurement_origin"] = "reused_historical_csv"
        selected.append(row)
    if seen != set(reference) or len(selected) != 3150 or len(identities) != 1:
        raise ValueError("Missing or ambiguous historical baseline grid/identity")
    identity = dict(zip(("checkpoint_sha256", "config_hash", "source_hash", "git_commit"), identities.pop()))
    identity.update(method=baseline, training_seed=seed, stage="full", update=UPDATES,
                    measurement_origin="reused_historical_csv")
    return selected, identity


def verify_baseline_checkpoint(paths, baseline, identity):
    seed = identity["training_seed"]
    configured = paths["historical_baseline_checkpoints"][baseline]
    if isinstance(configured, dict):
        if f"seed{seed}" not in configured:
            raise ValueError("Missing explicit same-seed historical checkpoint path")
        configured = configured[f"seed{seed}"]
    elif seed != 0:
        raise ValueError("A single historical checkpoint path is valid only for seed0")
    path = resolve_path(paths, configured)
    if sha256_file(path) != identity["checkpoint_sha256"]:
        raise ValueError("Historical baseline checkpoint SHA256 mismatch")
    checkpoint = torch_load(path)
    for key, expected in (("experiment", baseline), ("seed", seed), ("stage", "full"), ("update", UPDATES),
                          ("source_hash", identity["source_hash"]), ("config_hash", identity["config_hash"])):
        if checkpoint.get(key) != expected:
            raise ValueError(f"Historical baseline checkpoint identity mismatch: {key}")
    if "params" not in checkpoint or not checkpoint.get("complete_update_boundary"):
        raise ValueError("Incomplete historical baseline checkpoint")
    del checkpoint
    gc.collect()
    return {"checkpoint_sha256": identity["checkpoint_sha256"], "identity_verified": True}


def validate_training_documents(config, status, manifest, checks, summary, latest, variant, expected_source, seed=0):
    """Pure validation shared by the filesystem gate and CPU rejection tests."""
    validate_seed(variant, seed)
    if digest({k: v for k, v in config.items() if k != "config_hash"}) != config.get("config_hash"):
        raise ValueError("Effective config hash mismatch")
    if (config.get("exp"), config.get("stage"), config.get("seed"), config.get("train", {}).get("total_iter")) != (variant, "full", seed, UPDATES):
        raise ValueError("Only the requested full 300000-update training seed is accepted")
    if config.get("source_hash") != expected_source:
        raise ValueError("Training/evaluation source mismatch")
    if config.get("train", {}).get("pixel_opt", {}).get("type") != "L1StructureLoss":
        raise ValueError("Variant must use the fixed structure loss")
    mode = "none" if variant == "C" else "gated"
    network = config.get("network_g", {})
    if network.get("adapter_mode") != mode or network.get("pfa_rho", 0) != 0 or network.get("pfa_seed") is not None:
        raise ValueError("Wrong adapter/PFA training protocol")
    if config.get("continuation") or config.get("plan_c", {}).get("c0_sha256"):
        raise ValueError("Full training cannot use the earlier C0 continuation")
    full = config.get("full_training", {})
    if (full.get("schema"), full.get("variant"), full.get("baseline_experiment"), full.get("budget")) != (
            "c_only_full_v1", variant, VARIANTS[variant], UPDATES):
        raise ValueError("Wrong full-training initialization contract")
    if full.get("baseline_training_seed", 0) != seed:
        raise ValueError("Historical baseline training seed mismatch")
    if full.get("from_c0") is not False or full.get("legacy_global_pfa") is not True:
        raise ValueError("Full training requires fresh initialization and legacy global PFA")
    if full.get("diagnostic_tag") is not None or config.get("path", {}).get("pretrain_network_g"):
        raise ValueError("Diagnostic or pretrained-final runs cannot enter formal evaluation")
    if config.get("execution_contract", {}).get("schema") != "c_only_input_protection_v1":
        raise ValueError("Missing per-update input protection contract")
    if status.get("status") != "completed" or status.get("update") != UPDATES:
        raise ValueError("Training status is not completed at 300000")
    for field in ("config_hash", "source_hash"):
        if any(document.get(field) != config[field] for document in (manifest, status, summary)):
            raise ValueError(f"Training manifest {field} mismatch")
    if not summary.get("completed") or summary.get("updates") != UPDATES:
        raise ValueError("Training summary is incomplete")
    if latest.get("update") != UPDATES or not latest.get("sha256"):
        raise ValueError("Final checkpoint pointer is incomplete")
    if summary.get("checkpoint_sha256") != latest["sha256"]:
        raise ValueError("Training summary/checkpoint SHA mismatch")
    if manifest.get("completed_optimizer_updates") != UPDATES or manifest.get("final_checkpoint_sha256") != latest["sha256"]:
        raise ValueError("Manifest does not bind the final checkpoint")
    if checks.get("update") != UPDATES or not checks.get("complete"):
        raise ValueError("Training checks are incomplete")
    initial = checks.get("initialization_contract", {})
    if initial.get("seed", 0) != seed:
        raise ValueError("Initialization seed proof mismatch")
    for field in ("fresh_adamw", "fresh_scheduler", "encoder_parameters_frozen", "encoder_bn_training", "fp32", "legacy_global_pfa"):
        if initial.get(field) is not True:
            raise ValueError(f"Initialization protection proof failed/missing: {field}")
    if not isinstance(initial.get("initial_network_sha256"), str) or len(initial["initial_network_sha256"]) != 64:
        raise ValueError("Missing initial model fingerprint")
    for field in ("encoder_parameters_unchanged", "encoder_gradients_absent", "last_checkpoint_sha256_verified", "input_protection_passed"):
        if checks.get(field) is not True:
            raise ValueError(f"Training protection proof failed/missing: {field}")
    if checks.get("verified_updates") != UPDATES:
        raise ValueError("Incomplete per-update protection proof")
    if manifest.get("initialization_contract") != initial or summary.get("training_checks") != checks:
        raise ValueError("Inconsistent initialization/training checks documents")


def validate_input_protection(path, proof, config, training_log=None):
    if (proof.get("schema"), proof.get("status"), proof.get("update"), proof.get("verified_updates")) != (
            "c_only_input_protection_v1", "passed", UPDATES, UPDATES):
        raise ValueError("Incomplete input protection summary")
    if any(proof.get(field) != config[field] for field in ("config_hash", "source_hash")):
        raise ValueError("Input protection source/config mismatch")
    if proof.get("proof_sha256") != sha256_file(path):
        raise ValueError("Input protection SHA256 mismatch")
    count = 0
    def records(file):
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    log = records(training_log) if training_log is not None else ()
    for row, log_row in zip_longest(records(path), log):
        if row is None or (training_log is not None and log_row is None):
            raise ValueError("Training log/protection coverage differs")
        count += 1
        if row.get("update") != count or row.get("dataset_replay_count") != 16:
            raise ValueError("Missing/duplicate input protection update or wrong batch count")
        for field in ("dataset_replay_exact", "dataset_rng_exact", "feed_replay_exact", "raw_unchanged", "feed_rng_exact", "optimizer_inputs_unchanged", "snapshot_copies_exact"):
            if row.get(field) is not True:
                raise ValueError(f"Input protection failed: {field}")
        for field in ("input_sha256", "target_sha256"):
            if not isinstance(row.get(field), str) or len(row[field]) != 64 or row[field] != row.get("post_" + field):
                raise ValueError("Training input/target was modified")
        if log_row is not None:
            if log_row.get("update") != count or not math.isfinite(float(log_row["loss"])):
                raise ValueError("Training log has missing/duplicate updates or nonfinite loss")
            if any(log_row.get(field) != row[field] for field in ("input_sha256", "target_sha256")):
                raise ValueError("Training log inputs differ from input protection proof")
    if count != UPDATES:
        raise ValueError("Input protection does not cover all 300000 updates")
    return count


def validated_gate(paths, variant, seed=0):
    validate_seed(variant, seed)
    run = Path(paths["output_root"]) / "full" / variant / f"seed{seed}"
    if (run / "input_failure").exists():
        raise ValueError("Run contains a preserved input-protection failure")
    names = ("effective_config.json", "status.json", "manifest.json", "training_checks.json", "summary.json", "checkpoints/latest.json")
    config, status, manifest, checks, summary, latest = [read_json(run / name) for name in names]
    validate_training_documents(config, status, manifest, checks, summary, latest, variant, source_hash(), seed)
    protection = read_json(run / "input_protection.json")
    validate_input_protection(run / "input_protection.jsonl", protection, config, run / "train.jsonl")
    if any(document.get("input_protection_sha256") != protection["proof_sha256"] for document in (manifest, summary, checks)):
        raise ValueError("Input protection proof is not bound to training completion")
    for field, path_key in (("pretrained_sha256", "pretrained_rn50"), ("data_manifest_hash", "data_manifest")):
        if config.get(field) != sha256_file(paths[path_key]):
            raise ValueError(f"Training asset identity mismatch: {field}")
    if config["train"]["pixel_opt"] != paths["pixel_opt"]:
        raise ValueError("Structure loss configuration differs from fixed paths")
    checkpoint_path = Path(latest["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(paths["project_root"]) / checkpoint_path
    if checkpoint_path.resolve().parent != (run / "checkpoints").resolve():
        raise ValueError("Final checkpoint is outside the expected run")
    if sha256_file(checkpoint_path) != latest["sha256"]:
        raise ValueError("Final checkpoint SHA256 mismatch")
    checkpoint = torch_load(checkpoint_path)
    for field, expected in (("experiment", variant), ("stage", "full"), ("seed", seed), ("update", UPDATES),
                            ("config_hash", config["config_hash"]), ("source_hash", config["source_hash"])):
        if checkpoint.get(field) != expected:
            raise ValueError(f"Final checkpoint identity mismatch: {field}")
    required = {"params", "optimizer", "scheduler", "rng", "sampler", "scaler", "ema", "pfa_rng"}
    if not required.issubset(checkpoint) or not checkpoint.get("complete_update_boundary") or checkpoint["pfa_rng"] is not None:
        raise ValueError("Incomplete final recovery checkpoint")
    contract = {"variant": variant, "baseline": VARIANTS[variant], "source_hash": config["source_hash"],
                "config_hash": config["config_hash"], "pretrained_sha256": config["pretrained_sha256"],
                "data_manifest_hash": config["data_manifest_hash"], "training_updates": UPDATES,
                "training_seed": seed, "training_checks": checks, "input_protection": protection,
                "pixel_opt": config["train"]["pixel_opt"], "full_training": config["full_training"],
                "proof_sha256": {name: sha256_file(run / name) for name in (*names, "train.jsonl", "input_protection.json", "input_protection.jsonl")}}
    return config, checkpoint_path, checkpoint, contract


def save_panel(directory, clean, noisy, restored, variant, dataset, family, level):
    name = f"{dataset}_{family}_{level:g}"
    destination = directory / name
    save_fixed_visual(destination, clean, noisy, restored)
    images = [Image.open(destination / f"{field}_crop.png").convert("RGB")
              for field in ("clean", "noisy", "restored", "error")]
    width, height = images[0].size
    panel = Image.new("RGB", (width * 4, height + 24), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, image) in enumerate(zip(("Clean", "Noisy clipped", variant, "Abs.err [0,.20]"), images)):
        panel.paste(image, (index * width, 24))
        draw.text((index * width + 2, 5), label, fill="black")
        image.close()
    panel.save(directory / f"{name}_panel.png")
    panel.close()


def evaluate_variant(paths, variant, output_root, device, seed=0):
    validate_seed(variant, seed)
    protocol = protocol_for_seed(seed)
    historical_csv, reference, historical = load_historical(paths)
    baseline_rows, baseline_identity = historical_baseline(historical_csv, VARIANTS[variant], reference,
                                                          paths.get("baseline_source_hash", HISTORICAL_SOURCE), seed)
    baseline_proof = verify_baseline_checkpoint(paths, VARIANTS[variant], baseline_identity)
    config, checkpoint_path, checkpoint, contract = validated_gate(paths, variant, seed)
    output = output_root / variant / f"seed{seed}"
    identity = {"method": variant, "training_seed": seed, "stage": "full", "update": UPDATES,
                "checkpoint_sha256": sha256_file(checkpoint_path), "config_hash": config["config_hash"],
                "source_hash": config["source_hash"], "git_commit": config["git_commit"],
                "measurement_origin": "new_model_inference"}
    with RunLock(output / "process.lock"):
        if (output / "summary.json").exists():
            old = read_json(output / "summary.json")
            if (old.get("completed") and old.get("identity") == identity
                    and old.get("protocol_hash") == digest(protocol)
                    and old.get("contract_hash") == digest(contract)
                    and old.get("historical_csv", {}).get("sha256") == historical["sha256"]
                    and all(sha256_file(output / name) == value for name, value in old["artifact_sha256"].items())):
                print(f"Already evaluated and SHA verified: {output}", flush=True)
                return old
            raise FileExistsError(f"Incompatible evaluation output: {output}")
        datasets, manifest, manifest_hash = benchmark_data(paths, reference, historical)
        atomic_json(output / "access_gate.json", {"at": utc_now(), "variant": variant, "training_seed": seed,
                    "source_hash": source_hash(), "contract_hash": digest(contract), "baseline_proof": baseline_proof})
        network = dict(config["network_g"])
        network.pop("type", None)
        network["model_path"] = paths["pretrained_rn50"]
        model = CLIPDenoising(**network)
        model.load_state_dict(checkpoint["params"], strict=True)
        del checkpoint
        gc.collect()
        model.to(device).eval()
        rows = []
        clean_hashes = {(row["dataset"], row["image_id"]): row["sha256"] for row in manifest}
        for dataset, (base, files) in datasets.items():
            for file_index, file in enumerate(files):
                image_id, clean = file.relative_to(base).as_posix(), read_rgb(file)
                for family, level in CONDITIONS:
                    noisy, noise_seed, noise_hash = checked_noise(clean, family, level, dataset, image_id, reference)
                    restored, elapsed, _ = infer(model, noisy, torch.device(device))
                    scores = calculate_metrics(clean, restored)
                    if not all(math.isfinite(value) for value in scores.values()):
                        raise FloatingPointError("Nonfinite metric")
                    rows.append({"experiment": variant, "train_seed": seed, "stage": "full", "update": UPDATES,
                                 **{key: identity[key] for key in ("checkpoint_sha256", "config_hash", "source_hash", "git_commit", "measurement_origin")},
                                 "dataset": dataset, "image_id": image_id, "noise_type": family, "noise_level": level,
                                 "noise_seed": noise_seed, "noise_sha256": noise_hash,
                                 "clean_sha256": clean_hashes[(dataset, image_id)], "dataset_manifest_hash": manifest_hash,
                                 "protocol_group": "extension" if family in {"speckle", "salt_pepper"} else "official",
                                 "domain": "ID" if family == "gaussian" and level == 15 else "OOD",
                                 "height": clean.shape[0], "width": clean.shape[1], **scores, "forward_ms": elapsed})
                    if file_index == 0 and (dataset, family, float(level)) in TABLE1_CONDITIONS:
                        save_panel(output / "panels", clean, noisy, restored, variant, dataset, family, level)
                print(f"{variant} seed{seed}: {len(rows)}/3150", flush=True)
        if len(rows) != 3150:
            raise ValueError("Incomplete four-dataset benchmark")
        write_csv(output / "per_image_results.csv", rows)
        write_csv(output / "baseline_per_image_results.csv", baseline_rows)
        artifacts = {path.relative_to(output).as_posix(): sha256_file(path)
                     for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"process.lock", "summary.json"}}
        result = {"completed": True, "completed_at": utc_now(), "rows": len(rows), "baseline_rows": len(baseline_rows),
                  "identity": identity, "baseline_identity": baseline_identity, "baseline_checkpoint_proof": baseline_proof,
                  "historical_csv": historical, "matched_historical_inputs": len(rows),
                  "dataset_manifest": manifest, "dataset_manifest_hash": manifest_hash,
                  "protocol": protocol, "protocol_hash": digest(protocol), "contract": contract,
                  "contract_hash": digest(contract), "artifact_sha256": artifacts, "dependencies": dependency_versions()}
        atomic_json(output / "summary.json", result)
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/plan_c_only_server.yaml")
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/c_only/evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-inputs-only", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.variant:
        validate_seed(args.variant, args.seed)
    paths = load_paths(args.paths)
    historical_csv, reference, historical = load_historical(paths)
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    if args.verify_inputs_only:
        identities = {baseline: historical_baseline(historical_csv, baseline, reference,
                      paths.get("baseline_source_hash", HISTORICAL_SOURCE), args.seed)[1]
                      for baseline in ((VARIANTS[args.variant],) if args.variant else (("G1",) if args.seed else VARIANTS.values()))}
        report = verify_all_inputs(paths, historical_csv)
        report.update(completed=True, status="passed", accepted=True, completed_at=utc_now(), new_model_outcome_access=False,
                      existing_historical_csv_read=True,
                      test_images_opened_for_input_provenance=True, source_hash=source_hash(),
                      historical_baseline_identities=identities,
                      purpose="input provenance and baseline CSV identity only; no model inference or new denoising scores")
        if args.audit_output:
            atomic_json(args.audit_output, report)
        print(json.dumps(report, indent=2))
        return
    if not args.variant:
        parser.error("--variant is required for model evaluation")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    evaluate_variant(paths, args.variant, args.output, args.device, args.seed)


if __name__ == "__main__":
    main()
