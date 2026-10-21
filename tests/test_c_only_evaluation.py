"""CPU complete-grid, immutable historical-baseline and training-gate rejection tests."""
import copy
import hashlib
import json

import pytest
from PIL import Image

from tools import evaluate_c_only as evaluation
from tools.evaluate_c_only import (CONDITIONS, DATASET_COUNTS, HISTORICAL_SOURCE, PROTOCOL,
                                    TABLE1_CONDITIONS, VARIANTS, historical_baseline, historical_index)
from tools.evaluation_utils import read_csv, write_csv
from tools.experiment_common import atomic_json, digest, read_json, sha256_file
from tools.summarize_c_only import summarize, validate_rows


@pytest.fixture
def benchmark(tmp_path):
    manifest = [{"dataset": dataset, "image_id": f"image{i:03d}.png", "sha256": digest((dataset, i))}
                for dataset, count in DATASET_COUNTS.items() for i in range(count)]
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    old_rows = []
    for baseline in VARIANTS.values():
        for image in manifest:
            for family, level in CONDITIONS:
                old_rows.append({"experiment": baseline, "train_seed": 0, "stage": "full", "update": 300000,
                                 "checkpoint_sha256": digest(baseline), "config_hash": digest((baseline, "config")),
                                 "source_hash": HISTORICAL_SOURCE, "git_commit": "old-commit",
                                 "dataset": image["dataset"], "image_id": image["image_id"],
                                 "noise_type": family, "noise_level": level, "noise_seed": 0,
                                 "noise_sha256": digest((image["dataset"], image["image_id"], family, level)),
                                 "height": 64, "width": 64, "dataset_manifest_hash": manifest_hash,
                                 "protocol_group": "extension" if family in {"speckle", "salt_pepper"} else "official",
                                 "domain": "ID" if family == "gaussian" and level == 15 else "OOD",
                                 "psnr": 30 + int(image["image_id"][5:8]) / 100 + (0.2 if baseline == "G1" else 0),
                                 "ssim": .8 + (.002 if baseline == "G1" else 0)})
    history = tmp_path / "historical.csv"
    write_csv(history, old_rows)
    reference, historical = historical_index(history)
    root, summaries = tmp_path / "evaluation", {}
    for method, baseline in VARIANTS.items():
        baseline_rows, baseline_identity = historical_baseline(history, baseline, reference)
        directory = root / method / "seed0"
        identity = {"method": method, "training_seed": 0, "stage": "full", "update": 300000,
                    "checkpoint_sha256": digest(method), "config_hash": digest((method, "config")),
                    "source_hash": "new-source", "git_commit": "new-commit", "measurement_origin": "new_model_inference"}
        delta = .12 if method == "C" else -.07
        new_rows = []
        clean_hashes = {(row["dataset"], row["image_id"]): row["sha256"] for row in manifest}
        for old in baseline_rows:
            row = dict(old, experiment=method, psnr=old["psnr"] + delta, ssim=old["ssim"] + delta / 100,
                       clean_sha256=clean_hashes[(old["dataset"], old["image_id"])])
            row.update({field: identity[field] for field in ("checkpoint_sha256", "config_hash", "source_hash", "git_commit", "measurement_origin")})
            new_rows.append(row)
        write_csv(directory / "per_image_results.csv", new_rows)
        write_csv(directory / "baseline_per_image_results.csv", baseline_rows)
        atomic_json(directory / "access_gate.json", {"fixture": True})
        for dataset, family, level in TABLE1_CONDITIONS:
            path = directory / f"panels/{dataset}_{family}_{level:g}_panel.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4)).save(path)
        contract = {"variant": method, "baseline": baseline, "training_seed": 0, "training_updates": 300000,
                    "source_hash": "new-source", "config_hash": identity["config_hash"],
                    "pretrained_sha256": "rn50", "data_manifest_hash": "train432", "pixel_opt": {"type": "L1StructureLoss"}}
        metadata = {"completed": True, "rows": 3150, "baseline_rows": 3150, "matched_historical_inputs": 3150,
                    "identity": identity, "baseline_identity": baseline_identity,
                    "baseline_checkpoint_proof": {"identity_verified": True, "checkpoint_sha256": baseline_identity["checkpoint_sha256"]},
                    "historical_csv": historical, "protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL),
                    "contract": contract, "contract_hash": digest(contract), "dataset_manifest": manifest,
                    "dataset_manifest_hash": manifest_hash,
                    "artifact_sha256": {file.relative_to(directory).as_posix(): sha256_file(file)
                                        for file in directory.rglob("*") if file.is_file()}}
        atomic_json(directory / "summary.json", metadata)
        summaries[method] = metadata
    return {"root": root, "history": history, "summaries": summaries, "reference": reference}


def test_full_report_four_methods_and_historical_baselines(benchmark, tmp_path):
    result = summarize(benchmark["root"], tmp_path / "report", benchmark["history"])
    assert result["completed"] and result["rows"] == 12600
    assert result["new_measurement_rows"] == result["reused_baseline_rows"] == result["paired_rows"] == 6300
    assert result["full_condition_rows"] == 240 and result["table1_rows"] == 6
    assert result["table1_seedwise_rows"] == 12 and result["training_seeds_per_method"] == 1
    assert result["negative_paired_images"] == {"C": 0, "G1C": 3150}
    assert result["fixed_panels"] == 12
    output = tmp_path / "report"
    table = read_csv(output / "table1_conditions.csv")
    assert [(row["dataset"], row["noise_type"], float(row["noise_level"])) for row in table] == list(TABLE1_CONDITIONS)
    for row in table:
        assert float(row["C_minus_B0_psnr"]) == pytest.approx(.12)
        assert float(row["G1C_minus_G1_psnr"]) == pytest.approx(-.07)
        assert float(row["G1C_minus_C_psnr"]) == pytest.approx(.01)
    assert len(read_csv(output / "id_condition_results.csv")) == 16
    assert len(read_csv(output / "ood_aggregate_results.csv")) == 8
    assert len(read_csv(output / "ood_dataset_results.csv")) == 32
    assert len(read_csv(output / "negative_cases.csv")) == 3150
    assert len(read_csv(output / "exploratory_g1c_minus_c.csv")) == 3150
    for name, value in result["artifact_sha256"].items():
        assert sha256_file(output / name) == value
    report = (output / "experiment_report.md").read_text(encoding="utf-8")
    assert "只有一个训练种子" in report and "不能单独确立创新的因果收益或 SOTA" in report
    assert "不能称为未见测试集" in report and "没有重新训练或重新推理" in report


@pytest.mark.parametrize("mutation,message", [("missing", "Missing"), ("duplicate", "Duplicate"),
                                                ("noise", "Noisy input"), ("nan", "Nonfinite"), ("budget", "identity")])
def test_invalid_grid_rejected(benchmark, mutation, message):
    metadata = benchmark["summaries"]["C"]
    rows = read_csv(benchmark["root"] / "C/seed0/per_image_results.csv")
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "noise":
        rows[0]["noise_sha256"] = "different"
    elif mutation == "nan":
        rows[0]["psnr"] = "nan"
    else:
        rows[0]["update"] = "20000"
    with pytest.raises(ValueError, match=message):
        validate_rows(rows, benchmark["reference"], metadata["identity"], metadata["dataset_manifest"], metadata["dataset_manifest_hash"])


def test_tampered_artifact_rejected(benchmark, tmp_path):
    path = benchmark["root"] / "G1C/seed0/per_image_results.csv"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifact SHA256"):
        summarize(benchmark["root"], tmp_path / "report", benchmark["history"])


def test_baseline_cannot_be_changed_even_with_rewritten_copy_hash(benchmark, tmp_path):
    directory = benchmark["root"] / "C/seed0"
    rows = read_csv(directory / "baseline_per_image_results.csv")
    rows[0]["psnr"] = float(rows[0]["psnr"]) + 1
    write_csv(directory / "baseline_per_image_results.csv", rows)
    metadata = read_json(directory / "summary.json")
    metadata["artifact_sha256"]["baseline_per_image_results.csv"] = sha256_file(directory / "baseline_per_image_results.csv")
    atomic_json(directory / "summary.json", metadata)
    with pytest.raises(ValueError, match="baseline scores"):
        summarize(benchmark["root"], tmp_path / "report", benchmark["history"])


def test_missing_second_variant_blocks_final_report(benchmark, tmp_path):
    (benchmark["root"] / "G1C/seed0/summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize(benchmark["root"], tmp_path / "report", benchmark["history"])


def valid_documents():
    initial = {field: True for field in ("fresh_adamw", "fresh_scheduler", "encoder_parameters_frozen", "encoder_bn_training", "fp32", "legacy_global_pfa")}
    initial["initial_network_sha256"] = "a" * 64
    checks = {"initialization_contract": initial, "complete": True, "update": 300000, "verified_updates": 300000}
    checks.update({field: True for field in ("encoder_parameters_unchanged", "encoder_gradients_absent", "last_checkpoint_sha256_verified", "input_protection_passed")})
    config = {"exp": "C", "stage": "full", "seed": 0, "source_hash": "new-source",
              "train": {"total_iter": 300000, "pixel_opt": {"type": "L1StructureLoss"}},
              "network_g": {"adapter_mode": "none", "pfa_rho": 0, "pfa_seed": None},
              "full_training": {"schema": "c_only_full_v1", "variant": "C", "baseline_experiment": "B0", "budget": 300000,
                                "from_c0": False, "legacy_global_pfa": True},
              "execution_contract": {"schema": "c_only_input_protection_v1"}}
    config["config_hash"] = digest(config)
    common = {"source_hash": "new-source", "config_hash": config["config_hash"]}
    status = dict(common, status="completed", update=300000)
    manifest = dict(common, completed_optimizer_updates=300000, final_checkpoint_sha256="final-sha", initialization_contract=copy.deepcopy(initial))
    summary = dict(common, completed=True, updates=300000, checkpoint_sha256="final-sha", training_checks=copy.deepcopy(checks))
    latest = {"update": 300000, "sha256": "final-sha"}
    return [config, status, manifest, checks, summary, latest]


def test_training_gate_rejects_incomplete_wrong_source_and_invalid_initialization():
    evaluation.validate_training_documents(*valid_documents(), "C", "new-source")
    for index, field, bad, message in ((1, "status", "failed", "not completed"),
                                      (3, "input_protection_passed", False, "protection proof"),
                                      (3, "verified_updates", 20, "per-update"),
                                      (5, "update", 20000, "checkpoint pointer")):
        documents = valid_documents()
        documents[index][field] = bad
        with pytest.raises(ValueError, match=message):
            evaluation.validate_training_documents(*documents, "C", "new-source")
    with pytest.raises(ValueError, match="source mismatch"):
        evaluation.validate_training_documents(*valid_documents(), "C", "wrong-source")
    documents = valid_documents()
    documents[0]["full_training"]["from_c0"] = True
    documents[0]["config_hash"] = digest({key: value for key, value in documents[0].items() if key != "config_hash"})
    with pytest.raises(ValueError, match="fresh initialization"):
        evaluation.validate_training_documents(*documents, "C", "new-source")


def test_per_step_proof_checks_real_flags_hashes_and_training_log(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "UPDATES", 3)
    rows = []
    for update in range(1, 4):
        rows.append({"update": update, "dataset_replay_count": 16,
                     **{field: True for field in ("dataset_replay_exact", "dataset_rng_exact", "feed_replay_exact", "raw_unchanged", "feed_rng_exact", "optimizer_inputs_unchanged", "snapshot_copies_exact")},
                     "input_sha256": "a" * 64, "post_input_sha256": "a" * 64,
                     "target_sha256": "b" * 64, "post_target_sha256": "b" * 64})
    log_rows = [{"update": row["update"], "loss": .1, "input_sha256": row["input_sha256"], "target_sha256": row["target_sha256"]} for row in rows]
    path, log = tmp_path / "proof.jsonl", tmp_path / "train.jsonl"
    def save():
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        log.write_text("".join(json.dumps(row) + "\n" for row in log_rows))
        return {"schema": "c_only_input_protection_v1", "status": "passed", "update": 3, "verified_updates": 3,
                "source_hash": "source", "config_hash": "config", "proof_sha256": sha256_file(path)}
    config = {"source_hash": "source", "config_hash": "config"}
    assert evaluation.validate_input_protection(path, save(), config, log) == 3
    rows[1]["snapshot_copies_exact"] = False
    with pytest.raises(ValueError, match="snapshot_copies_exact"):
        evaluation.validate_input_protection(path, save(), config, log)
    rows[1]["snapshot_copies_exact"] = True
    log_rows[1]["input_sha256"] = "different"
    with pytest.raises(ValueError, match="inputs differ"):
        evaluation.validate_input_protection(path, save(), config, log)
    log_rows[1]["input_sha256"] = "a" * 64
    rows.pop()
    with pytest.raises(ValueError, match="coverage differs"):
        evaluation.validate_input_protection(path, save(), config, log)


def test_input_only_cli_does_not_require_trained_models(monkeypatch, tmp_path):
    history = tmp_path / "old.csv"
    monkeypatch.setattr(evaluation, "load_paths", lambda _: {"project_root": str(tmp_path)})
    monkeypatch.setattr(evaluation, "load_historical", lambda _: (history, {}, {}))
    monkeypatch.setattr(evaluation, "historical_baseline", lambda *args: ([], {"method": args[1]}))
    monkeypatch.setattr(evaluation, "verify_all_inputs", lambda *args: {"matched_inputs": 3150, "matched_raw_images": 210})
    def forbidden(*args):
        raise AssertionError("Input audit cannot access a new model")
    monkeypatch.setattr(evaluation, "validated_gate", forbidden)
    output = tmp_path / "input_audit.json"
    monkeypatch.setattr("sys.argv", ["evaluate_c_only", "--verify-inputs-only", "--audit-output", str(output)])
    evaluation.main()
    result = read_json(output)
    assert result["completed"] and result["matched_inputs"] == 3150
    assert result["new_model_outcome_access"] is False and result["existing_historical_csv_read"] is True
