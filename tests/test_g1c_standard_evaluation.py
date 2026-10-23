"""Standard full-budget evaluation is explicit and never invents guard proofs."""
import copy

import pytest
import torch

from tools import evaluate_c_only as evaluation
from tools.experiment_common import atomic_json, digest, read_json, sha256_file
from tools.summarize_g1c_multiseed import validate_completion_contract


@pytest.fixture
def completed_standard_run(tmp_path, monkeypatch):
    mode = evaluation.STANDARD_MODE
    disclosure = {"training_integrity_mode": mode, "per_update_input_consistency_checks": False}
    for name in ("rn50.bin", "data_manifest.json"):
        (tmp_path / name).write_bytes(name.encode())
    paths = {**disclosure, "project_root": str(tmp_path), "output_root": str(tmp_path / "runs"),
             "pretrained_rn50": str(tmp_path / "rn50.bin"), "data_manifest": str(tmp_path / "data_manifest.json"),
             "pixel_opt": {"type": "L1StructureLoss"}}
    run = tmp_path / "runs/full/G1C/seed1"
    config = {**disclosure, "exp": "G1C", "stage": "full", "seed": 1, "source_hash": "new-source",
              "git_commit": "fixture", "train": {"total_iter": 300000, "pixel_opt": paths["pixel_opt"]},
              "network_g": {"adapter_mode": "gated", "pfa_rho": 0, "pfa_seed": None},
              "runtime": {"integrity_mode": mode, "per_update_input_consistency_checks": False},
              "execution_contract": {"integrity_mode": mode, "per_update_input_consistency_checks": False},
              "full_training": {"variant": "G1C", "baseline_experiment": "G1", "baseline_training_seed": 1,
                                "budget": 300000, "from_c0": False, "diagnostic_tag": None},
              "pretrained_sha256": sha256_file(paths["pretrained_rn50"]),
              "data_manifest_hash": sha256_file(paths["data_manifest"])}
    config["config_hash"] = digest(config)
    common = {"source_hash": config["source_hash"], "config_hash": config["config_hash"]}
    checkpoint = {**common, "experiment": "G1C", "stage": "full", "seed": 1, "update": 300000,
                  "complete_update_boundary": True, "params": {"fixture": torch.zeros(1)}, "optimizer": {},
                  "scheduler": [], "rng": {}, "sampler": {}, "scaler": None, "ema": None, "pfa_rng": None}
    destination = run / "checkpoints/final.pth"
    destination.parent.mkdir(parents=True)
    torch.save(checkpoint, destination)
    sha = sha256_file(destination)
    checks = {**disclosure, "update": 300000, "loss_finite": True, "frozen_encoder_unchanged": True}
    documents = {
        "effective_config.json": config,
        "status.json": {**common, **disclosure, "status": "completed", "update": 300000},
        "manifest.json": {**common, **disclosure, "completed_optimizer_updates": 300000, "final_checkpoint_sha256": sha},
        "training_checks.json": checks,
        "summary.json": {**common, **disclosure, "completed": True, "updates": 300000,
                         "checkpoint_sha256": sha, "training_checks": checks},
        "checkpoints/latest.json": {"path": str(destination), "sha256": sha, "update": 300000}}
    for name, value in documents.items():
        atomic_json(run / name, value)
    monkeypatch.setattr(evaluation, "source_hash", lambda: "new-source")
    return paths, run


def test_completed_standard_run_evaluates_without_input_or_recovery_proofs(completed_standard_run, monkeypatch):
    paths, run = completed_standard_run
    def forbidden(*args, **kwargs):
        raise AssertionError("Standard mode cannot invoke disabled training input checks")
    monkeypatch.setattr(evaluation, "validate_input_protection", forbidden)
    monkeypatch.setattr(evaluation, "validate_readback_recovery", forbidden)
    config, path, checkpoint, contract = evaluation.validated_gate(paths, "G1C", 1)
    assert not (run / "input_protection.jsonl").exists()
    assert not (run / "readback_recovery.jsonl").exists()
    assert contract["training_integrity_mode"] == evaluation.STANDARD_MODE
    assert contract["per_update_input_consistency_checks"] is False
    assert "input_protection" not in contract and "readback_recovery" not in contract
    identity = {"method": "G1C", "training_seed": 1, "stage": "full", "update": 300000,
                "source_hash": config["source_hash"], "config_hash": config["config_hash"]}
    validate_completion_contract(contract, identity, 1)
    protocol = evaluation.protocol_for_seed(1, evaluation.STANDARD_MODE)
    assert protocol["per_update_input_consistency_checks"] is False
    assert "root cause remains unresolved" in protocol["known_host_diagnostics"]


@pytest.mark.parametrize("document,field,value", [
    ("status.json", "update", 299999), ("status.json", "status", "failed"),
    ("summary.json", "updates", 10000), ("summary.json", "completed", False),
    ("training_checks.json", "loss_finite", False),
    ("training_checks.json", "per_update_input_consistency_checks", True)])
def test_standard_mode_still_requires_normal_full_completion(completed_standard_run, document, field, value):
    paths, run = completed_standard_run
    changed = read_json(run / document)
    changed[field] = value
    atomic_json(run / document, changed)
    with pytest.raises(ValueError):
        evaluation.validated_gate(paths, "G1C", 1)


def test_standard_mode_cannot_silently_replace_guarded_gate(completed_standard_run):
    paths, _ = completed_standard_run
    paths = copy.deepcopy(paths)
    paths.pop("training_integrity_mode")
    with pytest.raises(ValueError, match="explicitly selected"):
        evaluation.validated_gate(paths, "G1C", 1)
