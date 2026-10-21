"""A/C continuation execution contracts using real optimizer, Mixup and data code.

The fast CPU tests substitute only a small network and deterministic fixture
images. The optional CLI exercises the actual RN50/C0 and real CBSD432 on the
server, with exact restoration and FP32 CUDA next-update acceptance.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.models.archs.spatial_pfa import FeaturePerturbation
from basicsr.models import image_restoration_model
from basicsr.utils import get_root_logger
from tests.test_resume_contract import compare_states
from tools.experiment_common import atomic_json, digest, load_paths, read_json, seed_all, torch_load
from tools.run_experiment import (
    ExactSampler, build_model, checkpoint_payload, restore_checkpoint, train_update,
)
from tools import run_pfa_experiment as runner
from tools.run_structure_experiment import configuration as structure_configuration


class TinyNetwork(nn.Module):
    """Keep encoder BN, frozen weights, independent PFA and learned decoder."""
    def __init__(self, opt):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.last = nn.Conv2d(4, 3, 1)
        self.pfa = FeaturePerturbation(opt.get("pfa_rho", 0), opt.get("pfa_seed"))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, value):
        features = self.encoder(value)
        if self.training:
            for level in (1, 2, 3, 4):
                features = features * (1 + self.pfa(features, level) * (0.025 * level))
        return self.last(features)


@pytest.fixture
def continuation_paths(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
    monkeypatch.setattr(image_restoration_model, "define_network", TinyNetwork)
    images = tmp_path / "train"
    images.mkdir()
    fixture = np.random.RandomState(20).randint(0, 256, (130, 132, 3), dtype=np.uint8)
    fixture_bytes = cv2.imencode(".png", fixture)[1].tobytes()
    for index in range(432):
        (images / f"{index:03d}.png").write_bytes(fixture_bytes)
    manifest, split, rn50 = [tmp_path / name for name in ("manifest.json", "split.txt", "RN50.pt")]
    manifest.write_text('{"fixture": true}', encoding="utf-8")
    split.write_text("fixture\n", encoding="utf-8")
    rn50.write_bytes(b"tiny network replaces only this diagnostic RN50")
    seed_all(22)
    c0 = tmp_path / "c0.pth"
    torch.save({"params": TinyNetwork({}).state_dict(), "experiment": "B0", "stage": "full",
                "seed": 0, "update": 300000, "optimizer": {"must_not_load": True}}, c0)
    return {"project_root": str(tmp_path), "pretrained_rn50": str(rn50), "train_dir": str(images),
            "output_root": str(tmp_path / "runs"), "data_manifest": str(manifest),
            "dev_train": str(split), "dev_val": str(split), "c0": str(c0), "device": "cpu",
            "micro_batch": 2, "accumulation": 1, "amp": False, "cpu_threads": 2,
            "cache_decoded_images": True, "minimum_free_gb": 0, "rho": 0.5,
            "learning_rate": 1e-5, "minimum_learning_rate": 1e-6, "weight_decay": 1e-4,
            "budgets": {"smoke": 4}, "checkpoint_every": 2, "checkpoint_keep": 1,
            "status_every": 1,
            "c0_expected": {"experiment": "B0", "stage": "full", "seed": 0, "update": 300000},
            "pixel_opt": {"type": "L1StructureLoss", "loss_weight": 1.0, "reduction": "mean",
                          "structure_weight": 0.05, "eps": 0.001}}


def load_final(directory):
    return torch_load(read_json(directory / "checkpoints/latest.json")["path"])


def test_run_resume_replays_input_mixup_optimizer_scheduler_and_pfa_exactly(continuation_paths, monkeypatch):
    paths = continuation_paths
    statuses = []
    real_atomic_json = runner.atomic_json

    def observe(path, value):
        if Path(path).name == "status.json":
            statuses.append(copy.deepcopy(value))
        return real_atomic_json(path, value)

    monkeypatch.setattr(runner, "atomic_json", observe)
    directory = runner.run("B1", "smoke", 100, paths)
    expected = load_final(directory)
    expected_rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
    assert any(row["status"] == "running" and row["durable_update"] == 2 for row in statuses)
    assert not any(row["status"] == "paused_at_checkpoint" for row in statuses)
    assert expected_rows[0]["lr"] == 1e-5 and expected_rows[-1]["lr"] == 1e-6
    assert all(first["lr"] > second["lr"] for first, second in zip(expected_rows, expected_rows[1:]))
    # This directory contains only this test's generated fixture run.
    assert directory.is_relative_to(Path(paths["project_root"]))
    shutil.rmtree(directory)
    statuses.clear()
    runner.run("B1", "smoke", 100, paths, stop_after=2)
    paused = read_json(directory / "status.json")
    assert paused["status"] == "paused_at_checkpoint" and paused["durable_update"] == 2
    assert paused["ended_at"] and paused["process"]["pid"] > 0
    # A crash after the durable checkpoint can leave an uncommitted partial row.
    with (directory / "train.jsonl").open("a") as stream:
        stream.write('{"update": 3, "loss":')
    runner.run("B1", "smoke", 100, paths, resume=True)
    actual = load_final(directory)
    expected["elapsed_seconds"] = actual["elapsed_seconds"] = 0
    comparison = compare_states(expected, actual, device="cpu")
    assert comparison["accepted"] and comparison["bitwise_equal"], comparison["failures"]
    rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
    assert [{k: v for k, v in row.items() if k != "seconds"} for row in rows] == [
        {k: v for k, v in row.items() if k != "seconds"} for row in expected_rows]
    assert len(list((directory / "checkpoints").glob("update_*.pth"))) == 1
    completed = read_json(directory / "status.json")
    assert completed["status"] == "completed" and completed["update"] == 4
    assert completed["ended_at"] and completed["checkpoint_sha256"]
    checks = read_json(directory / "training_checks.json")
    assert checks["fresh_adamw"] and checks["encoder_parameters_unchanged"]
    assert checks["encoder_gradients_absent"] and checks["encoder_bn_training_after"]
    assert runner.run("B1", "smoke", 100, paths, resume=True) == directory


def test_three_groups_share_c0_inputs_and_only_declared_interventions(continuation_paths):
    paths = continuation_paths
    rows, configurations = {}, {}
    for group in ("B0", "B1", "B3"):
        config_builder = structure_configuration if group == "B3" else runner.configuration
        method_paths = dict(paths, rho=0 if group == "B3" else 0.5)
        directory = runner.run(group, "smoke", 100, method_paths, configuration_builder=config_builder)
        rows[group] = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        configurations[group] = read_json(directory / "effective_config.json")
    assert len({value["paired_protocol_hash"] for value in configurations.values()}) == 1
    for field in ("input_sha256", "target_sha256", "lr"):
        assert len({tuple(row[field] for row in value) for value in rows.values()}) == 1
    assert rows["B1"][-1]["loss"] != rows["B0"][-1]["loss"]
    for row in rows["B3"]:
        assert row["structure"] > 0 and row["weighted_structure"] > 0
        assert np.isclose(row["loss"], row["pixel_l1"] + row["weighted_structure"], atol=1e-7)


def test_checkpoint_default_keeps_two_and_corruption_is_rejected(continuation_paths):
    paths = dict(continuation_paths)
    paths.pop("checkpoint_keep")
    directory = runner.run("B0", "smoke", 100, paths)
    assert len(list((directory / "checkpoints").glob("update_*.pth"))) == 2
    latest = directory / "checkpoints/latest.json"
    reference = read_json(latest)
    reference["sha256"] = "0" * 64
    atomic_json(latest, reference)
    with pytest.raises(RuntimeError, match="SHA256"):
        runner.run("B0", "smoke", 100, paths, resume=True)
    assert read_json(directory / "status.json")["status"] == "failed"


def test_wrong_c0_identity_or_missing_pfa_resume_fails_closed(continuation_paths):
    paths = dict(continuation_paths)
    paths["c0_expected"] = dict(paths["c0_expected"], update=299999)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        runner.run("B0", "smoke", 100, paths)
    assert not (Path(paths["output_root"]) / "smoke/B0/seed100/train.jsonl").exists()
    paths = continuation_paths
    directory = runner.run("B1", "smoke", 100, paths, stop_after=2)
    latest = directory / "checkpoints/latest.json"
    reference = read_json(latest)
    checkpoint = torch_load(reference["path"])
    checkpoint.pop("pfa_rng")
    torch.save(checkpoint, reference["path"])
    reference["sha256"] = runner.sha256_file(reference["path"])
    atomic_json(latest, reference)
    with pytest.raises(RuntimeError, match="PFA RNG"):
        runner.run("B1", "smoke", 100, paths, resume=True)


def actual_model_contract(paths, structure_paths, device, batch=2):
    """No experiment is trained here; only three diagnostic updates per group."""
    torch.set_num_threads(int(paths.get("cpu_threads", 4)))
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("The actual CUDA contract requires a visible GPU")
        torch.cuda.set_device(int(paths.get("gpu_index", 0)))
    result = {"device": device, "diagnostic_batch": batch, "groups": {}, "accepted": False,
              "cpu_requires_bitwise": True, "cuda_next_update_atol": 1e-6, "cuda_next_update_rtol": 1e-5}
    paired, protocols = {}, []
    for group in ("B0", "B1", "B3"):
        current_paths = dict(structure_paths if group == "B3" else paths, device=device)
        builder = structure_configuration if group == "B3" else runner.configuration
        opt, _ = builder(group, "smoke", 100, current_paths)
        seed_all(100)
        model = build_model(opt)
        checks = runner.initialization_contract(model, opt)
        dataset = Dataset_GaussianDenoising(copy.deepcopy(opt["datasets"]["train"]))
        assert len(dataset) == 432
        sampler = ExactSampler(len(dataset), 20100)
        seed_all(10100)
        observed = []
        for update in (1, 2):
            train_update(model, dataset, sampler, update, batch, 1)
            observed.append((runner.tensor_hash(model.lq), runner.tensor_hash(model.gt)))
        with tempfile.TemporaryDirectory(prefix="clip_ac_contract_") as temporary:
            checkpoint_path = Path(temporary) / "diagnostic.pth"
            torch.save(checkpoint_payload(model, sampler, opt, 2, 0), checkpoint_path)
            expected_loss = train_update(model, dataset, sampler, 3, batch, 1)
            expected_inputs = (runner.tensor_hash(model.lq), runner.tensor_hash(model.gt))
            expected = copy.deepcopy(checkpoint_payload(model, sampler, opt, 3, 0))
            observed.append(expected_inputs)
            frozen = runner.encoder_hash(model) == checks["encoder_parameter_sha256"]
            assert frozen and all(p.grad is None for p in model.net_g.encoder.parameters())
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            restored = build_model(opt)
            restored_sampler = ExactSampler(len(dataset), 1)
            saved = torch_load(checkpoint_path)
            restore_checkpoint(restored, restored_sampler, saved)
            immediate = compare_states(saved, checkpoint_payload(restored, restored_sampler, opt, 2, 0), device="cpu")
            assert immediate["accepted"], immediate["failures"]
            del saved
            actual_loss = train_update(restored, dataset, restored_sampler, 3, batch, 1)
            assert (runner.tensor_hash(restored.lq), runner.tensor_hash(restored.gt)) == expected_inputs
            assert np.isclose(expected_loss, actual_loss, atol=1e-6 if device == "cuda" else 0,
                              rtol=1e-5 if device == "cuda" else 0)
            comparison = compare_states(expected, checkpoint_payload(restored, restored_sampler, opt, 3, 0), device=device)
            assert comparison["accepted"], comparison["failures"]
            result["groups"][group] = {"initialization": checks, "encoder_unchanged": frozen,
                                       "immediate_restore": immediate, "next_update": comparison}
            paired[group] = observed
            protocols.append(opt["paired_protocol_hash"])
            del restored, expected, dataset
            if device == "cuda":
                torch.cuda.empty_cache()
    assert paired["B0"] == paired["B1"] == paired["B3"]
    assert len(set(protocols)) == 1
    result.update(accepted=True, paired_input_and_target_hashes=paired, paired_protocol_hash=protocols[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--structure-paths", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = actual_model_contract(load_paths(args.paths), load_paths(args.structure_paths), args.device, args.batch)
    atomic_json(args.report, report)
    print(json.dumps({"accepted": report["accepted"], "report": args.report, "device": args.device}), flush=True)


if __name__ == "__main__":
    main()
