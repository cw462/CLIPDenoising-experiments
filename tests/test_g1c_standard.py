"""Standard G1C execution: real data/Mixup/optimizer with a tiny test network.

These tests check execution count, full-budget configuration and recovery. They
do not add input-replay guards or claim that a short CPU check proves hardware
integrity or long-run CUDA reproducibility.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch

from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.models.image_restoration_model import ImageCleanModel
from tests.test_ac_training_contract import continuation_paths
from tests.test_resume_contract import compare_states
from tools.experiment_common import digest, read_json, torch_load
from tools import run_g1c_standard as runner

MODE = "standard_user_requested_v1"


@pytest.fixture(autouse=True)
def restore_execution_flags():
    prior = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    yield
    torch.use_deterministic_algorithms(prior["deterministic"], warn_only=prior["warn_only"])
    torch.backends.cudnn.benchmark = prior["benchmark"]
    torch.backends.cudnn.deterministic = prior["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = prior["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = prior["matmul_tf32"]


@pytest.fixture
def standard_paths(continuation_paths, monkeypatch):
    original_build = runner.legacy.build_model

    def build_fixture_model(opt):
        model = original_build(opt)
        # Supply RN50 metadata names without registering a duplicate BN layer.
        object.__setattr__(model.net_g.encoder, "bn1", model.net_g.encoder[1])
        model.net_g.encoder.pretrain_load_report = {"test_fixture": True}
        return model

    monkeypatch.setattr(runner.legacy, "build_model", build_fixture_model)
    monkeypatch.setattr(runner.legacy, "source_hash", lambda: "fixture-source")
    monkeypatch.setattr(runner.legacy, "git_commit", lambda: "fixture-commit")
    return dict(continuation_paths, micro_batch=16, cpu_threads=1,
                training_integrity_mode=MODE, formal_seeds=[0, 1, 2],
                formal_variants=["G1C"], checkpoint_every=2, checkpoint_keep=2)


def rows(directory):
    return [json.loads(line) for line in (directory / "train.jsonl").read_text(encoding="utf-8").splitlines()]


def checkpoint(directory):
    latest = read_json(directory / "checkpoints" / "latest.json")
    return torch_load(latest["path"])


def assert_recovery_state(expected, observed):
    """Computed floats may be close; all controls and optimizer counters exact."""
    floating_state = {"params", "optimizer", "elapsed_seconds"}
    controls = compare_states({k: v for k, v in expected.items() if k not in floating_state},
                              {k: v for k, v in observed.items() if k not in floating_state}, device="cpu")
    assert controls["accepted"] and controls["bitwise_equal"], controls["failures"]
    torch.testing.assert_close(expected["params"], observed["params"], rtol=1e-5, atol=1e-6)
    assert expected["optimizer"]["param_groups"] == observed["optimizer"]["param_groups"]
    assert expected["optimizer"]["state"].keys() == observed["optimizer"]["state"].keys()
    assert expected["optimizer"]["state"], "AdamW state must exist after actual updates"
    for key, wanted in expected["optimizer"]["state"].items():
        actual = observed["optimizer"]["state"][key]
        torch.testing.assert_close(wanted, actual, rtol=1e-5, atol=1e-6)
        assert float(wanted["step"]) == float(actual["step"]) == 4.0
    assert observed["update"] == 4 and observed["sampler"]["cursor"] == 64
    assert observed["scheduler"][0]["last_epoch"] == 3
    assert "pfa_rng" in observed and observed["pfa_rng"] is None


def test_standard_uses_one_legacy_update_without_input_replay(standard_paths, monkeypatch):
    counts = {"dataset": 0, "feed": 0, "optimize": 0}
    getitem = Dataset_GaussianDenoising.__getitem__
    feed = ImageCleanModel.feed_train_data
    optimize = ImageCleanModel.optimize_parameters

    def counted_getitem(self, index):
        counts["dataset"] += 1
        return getitem(self, index)

    def counted_feed(self, data):
        counts["feed"] += 1
        return feed(self, data)

    def counted_optimize(self, update):
        counts["optimize"] += 1
        return optimize(self, update)

    monkeypatch.setattr(Dataset_GaussianDenoising, "__getitem__", counted_getitem)
    monkeypatch.setattr(ImageCleanModel, "feed_train_data", counted_feed)
    monkeypatch.setattr(ImageCleanModel, "optimize_parameters", counted_optimize)
    directory = runner.run(0, standard_paths, stop_after=1, tag="one_update")
    assert counts == {"dataset": 16, "feed": 1, "optimize": 1}
    assert [row["update"] for row in rows(directory)] == [1]
    assert not list(directory.glob("input_protection*"))
    assert not (directory / "input_failure").exists()
    state = checkpoint(directory)
    assert "pfa_rng" in state and state["pfa_rng"] is None
    assert state["sampler"]["cursor"] == 16 and state["optimizer"]["state"]
    status = read_json(directory / "status.json")
    assert status["status"] == "paused_at_checkpoint" and status["update"] == 1
    # Importing the guarded runner changes execution policy at import time.
    imports = []
    for node in ast.walk(ast.parse(inspect.getsource(runner))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend([node.module or "", *(alias.name for alias in node.names)])
    assert not any("run_c_only_experiment" in name for name in imports)


def test_standard_resume_restores_full_state_and_archives_uncommitted_tail(standard_paths):
    paths = standard_paths
    directory = runner.run(0, paths, stop_after=4, tag="resume_contract")
    expected = checkpoint(directory)
    original_config = read_json(directory / "effective_config.json")
    expected_rows = rows(directory)
    # Remove only this test's generated run; keep identical paths/config on replay.
    assert directory.resolve().is_relative_to(Path(paths["project_root"]).resolve())
    shutil.rmtree(directory)
    runner.run(0, paths, stop_after=2, tag="resume_contract")
    assert checkpoint(directory)["update"] == 2
    with (directory / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"update": 3, "loss":')
    original_tail = (directory / "train.jsonl").read_text(encoding="utf-8")
    runner.run(0, paths, resume=True, stop_after=4, tag="resume_contract")
    assert_recovery_state(expected, checkpoint(directory))
    assert read_json(directory / "effective_config.json") == original_config
    observed_rows = rows(directory)
    assert [row["update"] for row in observed_rows] == [1, 2, 3, 4]
    assert [row["lr"] for row in observed_rows] == [row["lr"] for row in expected_rows]
    for wanted, actual in zip(expected_rows, observed_rows):
        assert actual["loss"] == pytest.approx(wanted["loss"], rel=1e-5, abs=1e-6)
    archives = [path for path in (directory / "resume_archive").rglob("*") if path.is_file()]
    assert archives, "Interrupted log evidence must be retained"
    assert any(original_tail == path.read_text(encoding="utf-8") for path in archives)
    status = read_json(directory / "status.json")
    assert status["status"] == "paused_at_checkpoint" and status["update"] == 4


def test_standard_keeps_full_training_science_and_historical_execution_flags(standard_paths):
    paths = standard_paths
    for seed in (0, 1, 2):
        opt, directory = runner.configuration(seed, paths, tag="diagnostic")
        assert directory == Path(paths["output_root"]) / "full" / "G1C" / f"seed{seed}" / "diagnostic"
        assert opt["experiment"] == "G1C" and opt["stage"] == "full" and opt["seed"] == seed
        assert opt["train"]["total_iter"] == 300000
        assert opt["path"]["pretrain_network_g"] is None
        assert opt["network_g"]["adapter_mode"] == "gated"
        assert opt["network_g"]["pfa_rho"] == 0 and opt["network_g"]["pfa_seed"] is None
        assert opt["runtime"]["micro_batch"] == 16 and opt["runtime"]["accumulation"] == 1
        assert opt["runtime"]["amp"] is False and opt["runtime"]["train_image_count"] == 432
        assert opt["runtime"]["freeze_bn"] is False
        for metadata in (opt["runtime"], opt["execution_contract"]):
            assert metadata["integrity_mode"] == MODE
            assert metadata["per_update_input_consistency_checks"] is False
        assert opt["datasets"]["train"]["sigma_range"] == 15
        assert opt["datasets"]["train"]["gt_size"] == 128
        assert opt["train"]["mixing_augs"] == {"mixup": True, "mixup_beta": 1.2, "use_identity": True}
        assert opt["train"]["optim_g"] == {"type": "AdamW", "lr": 3e-4,
                                           "weight_decay": 1e-4, "betas": [0.9, 0.999]}
        assert opt["train"]["scheduler"] == {"type": "CosineAnnealingRestartCyclicLR",
                                             "periods": [92000, 208000], "restart_weights": [1, 1],
                                             "eta_mins": [3e-4, 1e-6]}
        loss = opt["train"]["pixel_opt"]
        assert loss["type"] == "L1StructureLoss" and loss["loss_weight"] == 1
        assert loss["reduction"] == "mean" and loss["structure_weight"] == 0.05 and loss["eps"] == 0.001
        assert opt["config_hash"] == digest({key: value for key, value in opt.items() if key != "config_hash"})
    for invalid in ({"training_integrity_mode": "protected"}, {"micro_batch": 8},
                    {"accumulation": 2}, {"amp": True}):
        with pytest.raises((ValueError, RuntimeError)):
            runner.configuration(0, dict(paths, **invalid))
    missing_mode = dict(paths)
    missing_mode.pop("training_integrity_mode")
    with pytest.raises((ValueError, RuntimeError)):
        runner.configuration(0, missing_mode)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = True
    runner.configure_execution()
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark and not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32


def test_standard_nonfinite_loss_fails_without_publishing_checkpoint(standard_paths, monkeypatch):
    original_optimize = ImageCleanModel.optimize_parameters

    def nonfinite_report(self, update):
        original_optimize(self, update)
        self.log_dict["l_pix"] = float("nan")

    monkeypatch.setattr(ImageCleanModel, "optimize_parameters", nonfinite_report)
    _, directory = runner.configuration(0, standard_paths, tag="nonfinite")
    with pytest.raises(FloatingPointError, match="(?i)non.?finite"):
        runner.run(0, standard_paths, stop_after=1, tag="nonfinite")
    assert read_json(directory / "status.json")["status"] == "failed"
    assert not (directory / "checkpoints" / "latest.json").exists()
    assert not (directory / "checkpoints" / "final.pth").exists()
