"""Additional reused-benchmark evaluation matching the historical 4 x 15 protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from tools.evaluate_experiment import dependency_versions, infer
from tools.evaluate_pfa import resolve_c0, validate_stage_gate
from tools.evaluation_utils import (CONDITIONS, PROTOCOL as HISTORICAL_PROTOCOL, calculate_metrics,
                                    image_files, make_noise, read_csv, read_rgb, write_csv)
from tools.experiment_common import (RunLock, atomic_json, digest, load_paths, read_json,
                                     sha256_file, source_hash, torch_load, utc_now)

DATASET_COUNTS = {"CBSD68": 68, "Kodak24": 24, "McMaster": 18, "Urban100": 100}
TRAINING_SEEDS = (101, 102)
EXPECTED_JOBS = [("C0", "C0")] + [(method, str(seed)) for method in ("B0", "B1", "B3") for seed in TRAINING_SEEDS]
TABLE1_CONDITIONS = (("CBSD68", "gaussian", 50.0), ("Urban100", "gaussian", 50.0),
                     ("CBSD68", "spatial_gaussian", 55.0), ("McMaster", "spatial_gaussian", 55.0),
                     ("CBSD68", "poisson", 3.5), ("Kodak24", "salt_pepper", 0.02))
PROTOCOL = {"version": "ac-historical-table1-v1", "historical": HISTORICAL_PROTOCOL,
            "evaluation_scope": "additional_reused_historical_benchmark",
            "disclosure": "Full CBSD68 includes the 20 development images. All four datasets were evaluated in the earlier study. This is a reused benchmark, separate from the original primary 48-image CBSD68 plus Kodak24 seven-condition/three-noise-seed evaluation.",
            "input_preflight": "Raw test images may be opened before training solely to verify image/noise hashes; no model inference or outcome metrics. Model evaluation waits for all fixed valid confirmation checkpoints.",
            "training_seeds": TRAINING_SEEDS, "C0": "single fixed initialization, never duplicated as training seeds",
            "table1": TABLE1_CONDITIONS, "reference_claims": "No paper or screenshot numerical-reference comparison",
            "expected_models": 7, "expected_rows": 22050}


def historical_index(csv_path):
    """Bind all old noisy-input identities; repeated old model rows must agree."""
    lookup, manifests, c0_hashes = {}, set(), set()
    for row in read_csv(csv_path):
        key = row["dataset"], row["image_id"], row["noise_type"], float(row["noise_level"])
        if key[0] not in DATASET_COUNTS or (key[2], key[3]) not in CONDITIONS:
            raise ValueError(f"Unexpected historical condition: {key}")
        value = {"noise_sha256": row["noise_sha256"], "noise_seed": int(row["noise_seed"]),
                 "height": int(row["height"]), "width": int(row["width"])}
        if key in lookup and lookup[key] != value:
            raise ValueError(f"Historical models used different noisy inputs: {key}")
        lookup[key] = value
        manifests.add(row["dataset_manifest_hash"])
        if row["experiment"] == "B0" and int(row["train_seed"]) == 0:
            if row["stage"] != "full" or int(row["update"]) != 300000:
                raise ValueError("Historical C0 is not the original B0/full/seed0/300000 checkpoint")
            c0_hashes.add(row["checkpoint_sha256"])
    if len(manifests) != 1 or len(c0_hashes) != 1:
        raise ValueError("Historical dataset/C0 identity is ambiguous")
    for dataset, count in DATASET_COUNTS.items():
        images = {key[1] for key in lookup if key[0] == dataset}
        if len(images) != count:
            raise ValueError(f"Historical image count differs for {dataset}")
        expected = {(dataset, image, family, float(level)) for image in images for family, level in CONDITIONS}
        if {key for key in lookup if key[0] == dataset} != expected:
            raise ValueError(f"Missing historical condition: {dataset}")
    return lookup, {"path": str(Path(csv_path).resolve()), "sha256": sha256_file(csv_path),
                    "unique_inputs": len(lookup), "dataset_manifest_hash": manifests.pop(),
                    "c0_checkpoint_sha256": c0_hashes.pop()}


def benchmark_data(paths, reference, historical):
    """Recreate the exact historical ordered file-hash manifest before inference."""
    datasets, manifest = {}, []
    for dataset, expected_count in DATASET_COUNTS.items():
        base = Path(paths["test_datasets"][dataset])
        files = image_files(base)
        if len(files) != expected_count:
            raise ValueError(f"{dataset}: {len(files)} images != {expected_count}")
        for file in files:
            image_id = file.relative_to(base).as_posix()
            if (dataset, image_id, "gaussian", 15.0) not in reference:
                raise ValueError(f"Different historical image: {dataset}/{image_id}")
            manifest.append({"dataset": dataset, "image_id": image_id, "sha256": sha256_file(file)})
        datasets[dataset] = base, files
    # Historical evaluate_experiment uses json.dumps(sort_keys=True) with defaults.
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if manifest_hash != historical["dataset_manifest_hash"]:
        raise ValueError("Raw benchmark images/order differ from the historical dataset manifest")
    return datasets, manifest, manifest_hash


def checked_noise(clean, family, level, dataset, image_id, reference):
    noisy, noise_seed, noise_hash = make_noise(clean, family, level, dataset, image_id)
    expected = reference[(dataset, image_id, family, float(level))]
    if (noise_seed, noise_hash, clean.shape[0], clean.shape[1]) != (
            expected["noise_seed"], expected["noise_sha256"], expected["height"], expected["width"]):
        raise ValueError(f"Historical noisy-input mismatch: {dataset}/{image_id}/{family}/{level}")
    return noisy, noise_seed, noise_hash


def verify_all_inputs(paths, historical_csv):
    """CPU-only parity check for tests; does not run a model or compute denoising scores."""
    reference, historical = historical_index(historical_csv)
    datasets, _, manifest_hash = benchmark_data(paths, reference, historical)
    count = 0
    for dataset, (base, files) in datasets.items():
        for file in files:
            clean = read_rgb(file)
            for family, level in CONDITIONS:
                checked_noise(clean, family, level, dataset, file.relative_to(base).as_posix(), reference)
                count += 1
    return {"matched_inputs": count, "matched_raw_images": sum(DATASET_COUNTS.values()),
            "dataset_counts": DATASET_COUNTS, "dataset_manifest_hash": manifest_hash, "historical_csv": historical}


def validated_gate(paths):
    gate = validate_stage_gate(paths, "confirm", source_hash())
    configs, checkpoints, contract = gate
    # Explicitly exclude the earlier failed independent-process runs.
    for config in configs.values():
        if config.get("execution_contract", {}).get("schema") != "ac_triplet_v1":
            raise ValueError("Additional benchmark requires validated triplet training, not earlier failed runs")
    return gate


def evaluate_method(method, training_seed, paths, historical_csv, output_root, device, gate):
    configs, checkpoints, contract = gate
    seed_label = "C0" if method == "C0" else str(training_seed)
    checkpoint_path = resolve_c0(paths) if method == "C0" else checkpoints[(method, training_seed)]
    reference, historical = historical_index(historical_csv)
    if historical["c0_checkpoint_sha256"] != contract["c0_sha256"]:
        raise ValueError("Historical C0 checkpoint differs from the current common initialization")
    output = output_root / method / f"seed{seed_label}"
    config = configs[("B0", TRAINING_SEEDS[0])] if method == "C0" else configs[(method, training_seed)]
    checkpoint = torch_load(checkpoint_path)
    if method != "C0":
        for key in ("config_hash", "source_hash", "seed", "stage"):
            if checkpoint.get(key) != config[key]:
                raise ValueError(f"Checkpoint/config {key} mismatch")
        if checkpoint.get("experiment") != method or checkpoint.get("update") != config["train"]["total_iter"]:
            raise ValueError("Wrong checkpoint identity/update")
        if checkpoint.get("pfa_rng") is None or not checkpoint.get("complete_update_boundary"):
            raise ValueError("Incomplete recovery checkpoint")
    else:
        for key, expected in paths.get("c0_expected", {}).items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"Wrong C0 provenance: {key}")
    identity = {"method": method, "training_seed": seed_label, "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path), "update": checkpoint.get("update"),
                "config_hash": config["config_hash"] if method != "C0" else None,
                "checkpoint_source_hash": checkpoint.get("source_hash"), "evaluation_source_hash": source_hash()}
    with RunLock(output / "process.lock"):
        completion = output / "summary.json"
        if completion.exists():
            old = read_json(completion)
            if (old.get("completed") and old.get("identity") == identity and old.get("protocol_hash") == digest(PROTOCOL)
                    and old.get("historical_csv", {}).get("sha256") == historical["sha256"]
                    and old.get("contract_hash") == digest(contract)
                    and sha256_file(output / "per_image_results.csv") == old.get("metrics_csv_sha256")):
                print(f"Already evaluated: {output}", flush=True)
                return
            raise FileExistsError(f"Incompatible/completed evaluation output: {output}")
        atomic_json(output / "access_gate.json", {"at": utc_now(), "purpose": PROTOCOL["evaluation_scope"],
                    "fixed_runs": sorted(f"{method}/seed{seed}" for method, seed in configs),
                    "source_hash": source_hash(), "contract_hash": digest(contract)})
        datasets, manifest, manifest_hash = benchmark_data(paths, reference, historical)
        network = dict(config["network_g"])
        network.pop("type", None)
        network["model_path"] = paths["pretrained_rn50"]
        model = CLIPDenoising(**network)
        model.load_state_dict(checkpoint["params"], strict=True)
        del checkpoint
        model.to(device).eval()
        rows = []
        clean_hashes = {(r["dataset"], r["image_id"]): r["sha256"] for r in manifest}
        for dataset, (base, files) in datasets.items():
            for file in files:
                image_id = file.relative_to(base).as_posix()
                clean = read_rgb(file)
                for family, level in CONDITIONS:
                    noisy, noise_seed, noise_hash = checked_noise(clean, family, level, dataset, image_id, reference)
                    restored, forward_ms, _ = infer(model, noisy, torch.device(device))
                    scores = calculate_metrics(clean, restored)
                    if not all(math.isfinite(v) for v in scores.values()):
                        raise FloatingPointError("Nonfinite historical-protocol metric")
                    rows.append({"experiment": method, "train_seed": seed_label, "stage": "confirm",
                                 "checkpoint_sha256": identity["checkpoint_sha256"],
                                 "config_hash": identity["config_hash"], "source_hash": identity["evaluation_source_hash"],
                                 "dataset": dataset, "image_id": image_id, "noise_type": family, "noise_level": level,
                                 "noise_seed": noise_seed, "noise_sha256": noise_hash,
                                 "clean_sha256": clean_hashes[(dataset, image_id)], "dataset_manifest_hash": manifest_hash,
                                 "protocol_group": "extension" if family in {"speckle", "salt_pepper"} else "official",
                                 "domain": "ID" if family == "gaussian" and level == 15 else "OOD",
                                 "height": clean.shape[0], "width": clean.shape[1], **scores,
                                 "forward_ms": forward_ms, "evaluation_scope": PROTOCOL["evaluation_scope"]})
                print(f"Table1 reused benchmark {method} seed{seed_label}: {len(rows)}/3150", flush=True)
        if len(rows) != 3150:
            raise ValueError("Incomplete four-dataset benchmark")
        write_csv(output / "per_image_results.csv", rows)
        atomic_json(output / "summary.json", {"completed": True, "completed_at": utc_now(), "rows": len(rows),
                    "identity": identity, "historical_csv": historical, "matched_historical_inputs": len(rows),
                    "dataset_manifest": manifest, "dataset_manifest_hash": manifest_hash,
                    "metrics_csv_sha256": sha256_file(output / "per_image_results.csv"),
                    "protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL), "contract": contract,
                    "contract_hash": digest(contract), "dependencies": dependency_versions()})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/plan_a_server.yaml")
    parser.add_argument("--method", choices=("C0", "B0", "B1", "B3"))
    parser.add_argument("--training-seed", type=int, default=101)
    parser.add_argument("--historical-csv", type=Path)
    parser.add_argument("--historical-sha256")
    parser.add_argument("--verify-inputs-only", action="store_true",
                        help="CPU-only check of 210 raw images and all 3150 historical noisy-input hashes; no model/metrics")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ac/benchmark_table1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not args.verify_inputs_only and not args.method:
        parser.error("--method is required unless --verify-inputs-only is selected")
    if not args.verify_inputs_only and args.method != "C0" and args.training_seed not in TRAINING_SEEDS:
        parser.error("Only validated confirmation seeds 101 and 102 are accepted")
    if not args.verify_inputs_only and args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    paths = load_paths(args.paths)
    historical_csv = args.historical_csv or paths.get("historical_results_csv")
    if not historical_csv:
        parser.error("--historical-csv or historical_results_csv in paths is required")
    historical_csv = Path(historical_csv)
    if not historical_csv.is_absolute():
        historical_csv = Path(paths["project_root"]) / historical_csv
    expected_sha = args.historical_sha256 or paths.get("historical_results_csv_sha256")
    if expected_sha and sha256_file(historical_csv) != expected_sha:
        raise ValueError("Historical CSV does not match its pinned SHA256")
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    if args.verify_inputs_only:
        report = verify_all_inputs(paths, historical_csv)
        report.update(completed=True, completed_at=utc_now(), purpose="input parity only; no model inference or denoising scores",
                      model_outcome_access=False, test_images_opened_for_input_provenance=True,
                      source_hash=source_hash(), dependencies=dependency_versions())
        if args.audit_output:
            atomic_json(args.audit_output, report)
        print(json.dumps(report, indent=2))
        return
    gate = validated_gate(paths)
    evaluate_method(args.method, args.training_seed, paths, historical_csv, args.output, args.device, gate)


if __name__ == "__main__":
    main()
