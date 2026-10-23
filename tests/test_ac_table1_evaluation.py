"""Historical-input parity, complete-grid and independent A/C Table1 aggregation tests."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.evaluate_ac_table1 import (CONDITIONS, DATASET_COUNTS, EXPECTED_JOBS, PROTOCOL,
                                     TABLE1_CONDITIONS, historical_index, validated_gate, verify_all_inputs)
from tools.evaluation_utils import write_csv, read_csv
from tools.experiment_common import atomic_json, digest, read_json, sha256_file
from tools.summarize_ac_table1 import summarize, table1_tables, validate_rows


@pytest.fixture
def benchmark_fixture(tmp_path):
    manifest = [{"dataset": dataset, "image_id": f"image{i:03d}.png", "sha256": f"clean-{dataset}-{i}"}
                for dataset, count in DATASET_COUNTS.items() for i in range(count)]
    data_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    historical_rows, reference = [], {}
    for image in manifest:
        for family, level in CONDITIONS:
            key = image["dataset"], image["image_id"], family, float(level)
            noise_hash = digest(key)
            historical_rows.append({"experiment": "B0", "train_seed": 0, "stage": "full", "update": 300000,
                                    "checkpoint_sha256": "C0-checkpoint", "dataset": key[0], "image_id": key[1],
                                    "noise_type": family, "noise_level": level, "noise_seed": 0,
                                    "noise_sha256": noise_hash, "height": 64, "width": 64,
                                    "dataset_manifest_hash": data_hash})
            reference[key] = {"noise_seed": 0, "noise_sha256": noise_hash, "height": 64, "width": 64}
    history = tmp_path / "historical.csv"
    write_csv(history, historical_rows)
    _, history_meta = historical_index(history)
    contract = {"c0_sha256": "C0-checkpoint", "source_hash": "new-source"}
    manifests, all_rows = {}, []
    evaluations = tmp_path / "evaluations"
    for method, seed in EXPECTED_JOBS:
        identity = {"method": method, "training_seed": seed, "checkpoint_sha256": "C0-checkpoint" if method == "C0" else f"{method}-{seed}",
                    "config_hash": None if method == "C0" else f"config-{method}-{seed}", "evaluation_source_hash": "new-source"}
        rows = []
        for image in manifest:
            image_index = int(image["image_id"][5:8])
            for family, level in CONDITIONS:
                key = image["dataset"], image["image_id"], family, float(level)
                # Different paired effects by training seed test paired SD and signs.
                effect = 0 if method in ("C0", "B0") else (.12 if seed == "101" else .18) if method == "B1" else (.02 if seed == "101" else -.10)
                row = {"experiment": method, "train_seed": seed, "stage": "confirm",
                       "checkpoint_sha256": identity["checkpoint_sha256"], "config_hash": identity["config_hash"], "source_hash": "new-source",
                       "dataset": key[0], "image_id": key[1], "noise_type": family, "noise_level": level,
                       "noise_seed": 0, "noise_sha256": reference[key]["noise_sha256"], "clean_sha256": image["sha256"],
                       "dataset_manifest_hash": data_hash, "height": 64, "width": 64,
                       "protocol_group": "extension" if family in {"speckle", "salt_pepper"} else "official",
                       "domain": "ID" if family == "gaussian" and level == 15 else "OOD",
                       "evaluation_scope": PROTOCOL["evaluation_scope"], "psnr": 30 + image_index / 100 + effect, "ssim": .8 + effect / 100}
                rows.append(row)
        directory = evaluations / method / f"seed{seed}"
        write_csv(directory / "per_image_results.csv", rows)
        metadata = {"completed": True, "rows": 3150, "matched_historical_inputs": 3150, "identity": identity,
                    "historical_csv": history_meta, "protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL),
                    "contract": contract, "contract_hash": digest(contract), "dataset_manifest": manifest,
                    "dataset_manifest_hash": data_hash, "metrics_csv_sha256": sha256_file(directory / "per_image_results.csv")}
        atomic_json(directory / "summary.json", metadata)
        manifests[(method, seed)] = metadata
        all_rows.extend(rows)
    return {"root": evaluations, "history": history, "rows": all_rows, "reference": reference, "manifests": manifests}


def test_full_grid_does_not_accept_missing_duplicate_wrong_noise_or_failed_identity(benchmark_fixture):
    data = benchmark_fixture
    assert len(validate_rows(data["rows"], data["reference"], data["manifests"])) == 22050
    with pytest.raises(ValueError, match="Missing"):
        validate_rows(data["rows"][:-1], data["reference"], data["manifests"])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(data["rows"] + data["rows"][:1], data["reference"], data["manifests"])
    for field, bad, message in (("noise_sha256", "unpaired", "Noisy input"),
                                ("source_hash", "failed-old-source", "identity"),
                                ("checkpoint_sha256", "failed-old-checkpoint", "identity"),
                                ("psnr", float("nan"), "Nonfinite")):
        rows = list(data["rows"])
        rows[-1] = dict(rows[-1], **{field: bad})
        with pytest.raises(ValueError, match=message):
            validate_rows(rows, data["reference"], data["manifests"])


def test_complete_summary_matches_six_cells_and_keeps_seedwise_negative_results(benchmark_fixture, tmp_path):
    data = benchmark_fixture
    output = tmp_path / "report"
    result = summarize(data["root"], output, data["history"])
    assert result["completed"] and result["rows"] == 22050
    assert result["unique_historical_inputs"] == 3150
    assert result["full_condition_rows"] == 420
    assert result["table1_rows"] == 6 and result["table1_seedwise_rows"] == 12 and result["table1_paired_rows"] == 12
    wide = read_csv(output / "table1_conditions.csv")
    assert [(r["dataset"], r["noise_type"], float(r["noise_level"])) for r in wide] == list(TABLE1_CONDITIONS)
    assert all(float(r["B1_minus_B0_psnr"]) == pytest.approx(.15) for r in wide)
    assert all(float(r["B3_minus_B0_psnr"]) == pytest.approx(-.04) for r in wide)
    assert all(int(r["C0_seed_count"]) == 1 and int(r["B0_seed_count"]) == 2 for r in wide)
    pairs = read_csv(output / "table1_paired.csv")
    assert all(r["psnr_both_seeds_positive"] == ("True" if r["experiment"] == "B1" else "False") for r in pairs)
    assert len(read_csv(output / "negative_cases.csv")) == 3150
    assert len(read_csv(output / "ood_aggregate_results.csv")) == 14
    for name, expected_hash in result["artifact_sha256"].items():
        assert sha256_file(output / name) == expected_hash
    report = (output / "experiment_report.md").read_text(encoding="utf-8")
    assert "不能称为未见测试集" in report and "未混用原主终点的 skimage" in report
    assert "screenshot_reference_psnr" not in (output / "table1_conditions.csv").read_text()


def test_summary_detects_tampered_artifact(benchmark_fixture, tmp_path):
    data = benchmark_fixture
    csv_path = data["root"] / "B1/seed101/per_image_results.csv"
    with csv_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="CSV SHA256"):
        summarize(data["root"], tmp_path / "report", data["history"])


def test_gate_rejects_old_non_triplet_runs(monkeypatch):
    from tools import evaluate_ac_table1
    monkeypatch.setattr(evaluate_ac_table1, "validate_stage_gate", lambda *args: ({("B0", 101): {}}, {}, {}))
    with pytest.raises(ValueError, match="triplet"):
        validated_gate({})


def test_input_only_cli_requires_no_model_or_completed_training(monkeypatch, tmp_path):
    from tools import evaluate_ac_table1
    history = tmp_path / "historical.csv"
    history.write_text("fixture")
    output = tmp_path / "input_parity.json"
    monkeypatch.setattr(evaluate_ac_table1, "load_paths", lambda path: {"project_root": str(tmp_path), "historical_results_csv": str(history)})
    monkeypatch.setattr(evaluate_ac_table1, "verify_all_inputs", lambda *args: {"matched_inputs": 3150, "matched_raw_images": 210})
    def reject_gate(*args):
        raise AssertionError("Input-only parity must not require trained checkpoints")
    monkeypatch.setattr(evaluate_ac_table1, "validated_gate", reject_gate)
    monkeypatch.setattr("sys.argv", ["evaluate_ac_table1", "--verify-inputs-only", "--audit-output", str(output)])
    evaluate_ac_table1.main()
    report = read_json(output)
    assert report["completed"] and report["matched_inputs"] == 3150
    assert "no model inference" in report["purpose"]


@pytest.mark.skipif(not os.environ.get("AC_TABLE1_HISTORICAL_CSV") or not os.environ.get("AC_TABLE1_DATA_ROOT"),
                    reason="Explicit historical CSV/data paths required for full 3150-input parity audit")
def test_all_3150_actual_historical_noisy_inputs_and_raw_data_hashes_match():
    data_root = Path(os.environ["AC_TABLE1_DATA_ROOT"])
    paths = {"test_datasets": {name: str(data_root / name) for name in DATASET_COUNTS}}
    result = verify_all_inputs(paths, Path(os.environ["AC_TABLE1_HISTORICAL_CSV"]))
    assert result["matched_inputs"] == 3150
    assert result["historical_csv"]["unique_inputs"] == 3150
