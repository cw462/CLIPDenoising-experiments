"""A/C evaluation checks: no held-out leakage, unpaired input or pseudoreplication."""
import copy
from pathlib import Path

import numpy as np
import pytest

from tools.evaluate_pfa import CONDITIONS, common_training_contract, validate_stage_gate
from tools.evaluate_pfa import PROTOCOL, VISUAL_CONDITIONS, visual_key
from tools.evaluation_utils import write_csv
from tools.experiment_common import atomic_json, digest, read_json, sha256_file
from tools.summarize_ac import (audit_rows, average_noise_seeds, bootstrap_images,
                                decisions, paired_image_table, paired_statistics, seed_variability)


def fake_rows(stage="confirm"):
    seeds = (100,) if stage == "screen" else (101, 102)
    methods = [("C0", "C0")] + [(m, str(s)) for m in ("B0", "B1", "B3") for s in seeds]
    rows = []
    for method, training_seed in methods:
        for dataset, count in (("small", 2), ("large", 4)):
            for i in range(count):
                for family, level in CONDITIONS:
                    for noise_seed in (0, 1, 2):
                        # Different effect by dataset ensures equal-dataset aggregation is tested.
                        delta = (0.1 if dataset == "small" else 0.3) if method == "B1" else -0.2 if method == "B3" else 0
                        if training_seed == "102" and method == "B1":
                            delta += 0.02
                        rows.append(dict(method=method, training_seed=training_seed, dataset=dataset,
                                         image=f"image{i}", family=family, level=level, noise_seed=noise_seed,
                                         input_sha256=f"input:{dataset}/{i}/{family}/{level}/{noise_seed}",
                                         clean_sha256=f"clean:{dataset}/{i}", psnr=25 + noise_seed + delta,
                                         ssim=0.8 + delta / 100))
    return rows


def test_complete_grid_pairing_rejects_missing_duplicate_and_different_noise():
    rows = fake_rows()
    expected = {"small": 2, "large": 4}
    assert len(audit_rows(rows, "confirm", expected)) == 7 * 6 * 7 * 3
    for mutation, message in ((rows[:-1], "Incomplete"), (rows + rows[:1], "Duplicate")):
        with pytest.raises(ValueError, match=message):
            audit_rows(mutation, "confirm", expected)
    bad = copy.deepcopy(rows)
    bad[-1]["input_sha256"] = "different random draw"
    with pytest.raises(ValueError, match="Unpaired"):
        audit_rows(bad, "confirm", expected)
    bad = copy.deepcopy(rows)
    bad[-1]["psnr"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        audit_rows(bad, "confirm", expected)


def test_noise_and_training_repeats_not_independent_images_and_equal_datasets():
    averaged = average_noise_seeds(fake_rows())
    assert all(row["noise_repeats"] == 3 for row in averaged)
    assert len(averaged) == 7 * 6 * 7
    statistics = paired_statistics(paired_image_table(averaged))
    row = next(r for r in statistics if (r["method"], r["baseline"], r["training_seed"], r["dataset"], r["category"], r["metric"])
               == ("B1", "B0", "mean", "equal_dataset", "ood", "psnr"))
    assert row["independent_images"] == 6  # not 6 images * 3 noise seeds * 2 train seeds
    assert row["difference"] == pytest.approx(0.21)
    assert row["ci95_low"] == pytest.approx(0.21)
    assert row["ci95_high"] == pytest.approx(0.21)
    variability = next(r for r in seed_variability(statistics) if (r["method"], r["baseline"], r["dataset"], r["category"], r["metric"])
                       == ("B1", "B0", "equal_dataset", "ood", "psnr"))
    assert variability["sample_sd"] == pytest.approx(np.std([0.2, 0.22], ddof=1))
    assert decisions(statistics, "confirm")["B1"]["criterion_met"]
    assert not decisions(statistics, "confirm")["B3"]["criterion_met"]


def test_bootstrap_preserves_image_variability_and_is_reproducible():
    first = bootstrap_images({"one": [-0.2, 0.4, 0.1]})
    assert first == bootstrap_images({"one": [-0.2, 0.4, 0.1]})
    assert first[0] == pytest.approx(0.1)
    assert first[1] < 0 < first[2]


def stage_fixture(tmp_path):
    c0 = tmp_path / "c0.pth"
    c0.write_bytes(b"C0")
    rn50 = tmp_path / "rn50.pt"
    rn50.write_bytes(b"rn50")
    data = tmp_path / "data.json"
    data.write_text("{}")
    paths = {"project_root": str(tmp_path), "output_root": str(tmp_path / "runs"), "c0": str(c0),
             "data_manifest": str(data), "pretrained_rn50": str(rn50), "budgets": {"confirm": 20000}}
    for seed in (101, 102):
        for method in ("B0", "B1", "B3"):
            run = Path(paths["output_root"]) / "confirm" / method / f"seed{seed}"
            checkpoint = run / "checkpoints/final.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"{method}{seed}".encode())
            config = {"seed": seed, "exp": method, "stage": "confirm", "source_hash": "source",
                      "data_manifest_hash": sha256_file(data), "pretrained_sha256": sha256_file(rn50),
                      "network_g": {"depth": 5, "pfa_rho": 0.5 if method == "B1" else 0, "pfa_seed": seed + 30000},
                      "train": {"total_iter": 20000, "pixel_opt": {"type": "L1StructureLoss" if method == "B3" else "L1Loss"}},
                      "datasets": {"train": {"batch_size": 16}}, "runtime": {"data_rng_seed": seed + 10000},
                      "paired_protocol_hash": f"paired-{seed}",
                      "plan_a": {"c0_sha256": sha256_file(c0), "new_optimizer": True}}
            config["config_hash"] = digest(config)
            atomic_json(run / "effective_config.json", config)
            atomic_json(run / "status.json", {"status": "completed", "update": 20000})
            atomic_json(run / "checkpoints/latest.json", {"path": str(checkpoint), "update": 20000, "sha256": sha256_file(checkpoint)})
    return paths


def test_holdout_gate_requires_all_six_completed_and_matching_provenance(tmp_path):
    paths = stage_fixture(tmp_path)
    configs, checkpoints, contract = validate_stage_gate(paths, "confirm", "source")
    assert len(configs) == 6 and len(checkpoints) == 6
    assert contract["new_optimizer"]
    status = Path(paths["output_root"]) / "confirm/B3/seed102/status.json"
    atomic_json(status, {"status": "running", "update": 19999})
    with pytest.raises(RuntimeError, match="unfinished"):
        validate_stage_gate(paths, "confirm", "source")
    atomic_json(status, {"status": "completed", "update": 20000})
    with pytest.raises(ValueError, match="source"):
        validate_stage_gate(paths, "confirm", "changed-source")
    Path(paths["c0"]).write_bytes(b"changed C0")
    with pytest.raises(ValueError, match="C0"):
        validate_stage_gate(paths, "confirm", "source")


def test_common_contract_detects_non_prescribed_optimizer_changes(tmp_path):
    paths = stage_fixture(tmp_path)
    configs, _, _ = validate_stage_gate(paths, "confirm", "source")
    baseline = configs[("B0", 101)]
    changed = copy.deepcopy(configs[("B3", 102)])
    assert common_training_contract(baseline) == common_training_contract(changed)
    changed["train"]["lr"] = 1
    assert common_training_contract(baseline) != common_training_contract(changed)


def test_screen_summary_end_to_end_including_csv_report_and_fixed_panels(tmp_path, monkeypatch):
    from PIL import Image
    from tools import summarize_ac

    evaluation_root, output = tmp_path / "evaluations", tmp_path / "report"
    dataset = "CBSD68_dev20"
    for method, offset in (("C0", 0), ("B0", 0), ("B1", 0.2), ("B3", -0.1)):
        seed = "C0" if method == "C0" else "100"
        job = evaluation_root / "screen" / method / f"seed{seed}"
        rows, visuals = [], []
        for image_index in range(20):
            image_id = f"image{image_index:02d}.png"
            for family, level in CONDITIONS:
                for noise_seed in (0, 1, 2):
                    row = dict(method=method, training_seed=seed, dataset=dataset, image=image_id,
                               family=family, level=level, noise_seed=noise_seed,
                               input_sha256=f"{image_id}/{family}/{level}/{noise_seed}", clean_sha256=image_id,
                               psnr=30 + offset + noise_seed, ssim=0.8 + offset / 100)
                    rows.append(row)
                    if image_index < 2 and noise_seed == 0 and (family, level) in VISUAL_CONDITIONS:
                        relative = Path("visuals") / dataset / visual_key(image_id, family, level)
                        destination = job / relative
                        destination.mkdir(parents=True)
                        for name in ("clean", "noisy", "restored"):
                            for suffix in ("", "_crop"):
                                Image.new("RGB", (32, 32), (80, 120, 160)).save(destination / f"{name}{suffix}.png")
                        visuals.append(dict(row, directory=relative.as_posix()))
        write_csv(job / "per_image.csv", rows)
        write_csv(job / "visual_manifest.csv", visuals)
        atomic_json(job / "summary.json", {"completed": True, "diagnostic": False, "stage": "screen",
                    "identity": {"method": method, "training_seed": seed}, "rows": len(rows),
                    "data_hash": "shared", "contract_hash": "shared", "protocol_hash": digest(PROTOCOL)})
    monkeypatch.setattr("sys.argv", ["summarize_ac", "--stage", "screen", "--root", str(evaluation_root), "--output", str(output)])
    summarize_ac.main()
    summary = read_json(output / "summary.json")
    assert summary["completed"] and summary["rows"] == 1680
    assert summary["visual_panels"] == 12
    assert summary["decisions"]["B1"]["criterion_met"]
    assert not summary["decisions"]["B3"]["criterion_met"]
    report = (output / "experiment_report.md").read_text(encoding="utf-8")
    assert "未达到" in report and "绝对得分" in report and "整个研究从未见过" in report
    assert len(list((output / "figures").glob("*.png"))) == 12
    assert (output / "aggregate_scores.csv").is_file()
