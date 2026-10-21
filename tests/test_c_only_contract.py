"""CPU contracts for guarded full C/G1C; no shortened scientific schedule."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from basicsr.models.archs.spatial_pfa import FeaturePerturbation
from tests.test_ac_training_contract import continuation_paths
from tools import run_c_only_experiment as runner
from tools.experiment_common import read_json, restore_rng, rng_state, seed_all, torch_load


@pytest.fixture
def full_paths(continuation_paths, monkeypatch):
    # The synthetic fixture substitutes the network only; real dataset, Mixup,
    # AdamW, cyclic scheduler and L1StructureLoss are exercised with batch 16.
    paths = dict(continuation_paths, micro_batch=16, cpu_threads=1)
    monkeypatch.setattr(runner.legacy, "source_hash", lambda: "fixture-source")
    monkeypatch.setattr(runner.legacy, "git_commit", lambda: "fixture-commit")
    torch.set_num_threads(1)
    return paths


@pytest.mark.parametrize("variant,baseline,mode,seed", [("C", "B0", "none", 0),
    ("G1C", "G1", "gated", 0), ("G1C", "G1", "gated", 1), ("G1C", "G1", "gated", 2)])
def test_full_configuration_retains_legacy_protocol(full_paths, variant, baseline, mode, seed):
    opt, directory = runner.configuration(variant, "full", seed, full_paths)
    original, _ = runner.legacy.make_config(baseline, "full", seed, full_paths)
    assert directory == Path(full_paths["output_root"]) / "full" / variant / f"seed{seed}"
    assert opt["train"]["total_iter"] == 300000
    assert opt["train"]["optim_g"] == original["train"]["optim_g"]
    assert opt["train"]["optim_g"]["lr"] == 3e-4
    assert opt["train"]["scheduler"] == original["train"]["scheduler"]
    assert opt["train"]["scheduler"]["periods"] == [92000, 208000]
    assert opt["datasets"]["train"] == original["datasets"]["train"]
    assert opt["network_g"]["adapter_mode"] == mode
    assert opt["network_g"]["pfa_seed"] is None and opt["network_g"]["pfa_rho"] == 0
    assert opt["path"]["pretrain_network_g"] is None and not opt["full_training"]["from_c0"]
    assert opt["train"]["pixel_opt"]["structure_weight"] == 0.05
    assert opt["train"]["pixel_opt"]["eps"] == 0.001
    assert opt["train"]["pixel_opt"]["deterministic_reflect_padding"]
    assert opt["runtime"]["data_rng_seed"] == seed + 10000 and opt["runtime"]["sampler_seed"] == seed + 20000
    assert opt["full_training"]["baseline_training_seed"] == seed


def test_multiseed_scope_rejects_unplanned_variants_and_seeds(full_paths):
    paths = dict(full_paths, formal_variants=["G1C"], formal_seeds=[0, 1, 2])
    for variant, seed in (("C", 0), ("G1C", 3), ("G1C", -1), ("G1C", True)):
        with pytest.raises(ValueError):
            runner.configuration(variant, "full", seed, paths)
    with pytest.raises(ValueError, match="configured training scope"):
        runner.configuration("G1C", "full", 1, dict(paths, formal_seeds=[0]))


def test_each_seed_matches_corresponding_g1_initialization_and_rng(full_paths):
    fingerprints, samples = set(), set()
    for seed in (0, 1, 2):
        opt, _ = runner.configuration("G1C", "full", seed, full_paths)
        old_opt, _ = runner.legacy.make_config("G1", "full", seed, full_paths)
        seed_all(seed)
        old = runner.legacy.build_model(old_opt)
        old_rng = rng_state()
        seed_all(seed)
        candidate = runner.legacy.build_model(opt)
        assert runner.exact_equal(old.net_g.state_dict(), candidate.net_g.state_dict())
        assert runner.exact_equal(old_rng, rng_state())
        initial = runner.initialization_contract(candidate, opt)
        assert initial["seed"] == seed and initial["baseline_experiment"] == "G1"
        fingerprints.add(initial["initial_network_sha256"])
        sampler = runner.legacy.ExactSampler(432, seed + 20000)
        samples.add(tuple(sampler.take(16)))
    assert len(fingerprints) == len(samples) == 3


def test_global_pfa_exactly_matches_legacy_randn_like_and_rng():
    seed_all(37)
    reference = torch.ones(2, 3, 4, 5)
    before = rng_state()
    expected = [torch.randn_like(reference) for _ in range(4)]
    after = rng_state()
    restore_rng(before)
    pfa = FeaturePerturbation(0.0, None)
    assert runner.exact_equal(before, rng_state())  # construction draws nothing
    actual = [pfa(reference, level) for level in range(1, 5)]
    assert runner.exact_equal(expected, actual) and runner.exact_equal(after, rng_state())
    assert pfa.state_dict() is None


def test_execution_retains_historical_tf32_and_requires_determinism():
    runner.configure_execution(False)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32


class RandomDataset:
    def __len__(self):
        return 432

    def __getitem__(self, index):
        return {"lq": torch.randn(3, 4, 4) + random.random(),
                "gt": torch.ones(3, 4, 4) * np.random.rand(), "index": index}


def protection(tmp_path):
    return runner.Protection(tmp_path, {"config_hash": "fixture", "source_hash": "fixture"})


def test_dataset_replay_consumes_rng_once_and_preserves_result(tmp_path):
    seed_all(4)
    initial = rng_state()
    expected = RandomDataset()[8]
    expected_rng = rng_state()
    restore_rng(initial)
    actual = runner.ProtectedDataset(RandomDataset(), protection(tmp_path))[8]
    assert runner.exact_equal(expected, actual)
    assert runner.exact_equal(expected_rng, rng_state())


def test_dataset_difference_fails_and_preserves_both_outputs(tmp_path):
    class BadDataset(RandomDataset):
        calls = 0

        def __getitem__(self, index):
            value = super().__getitem__(index)
            self.calls += 1
            if self.calls == 2:
                value["gt"].view(torch.int32)[0, 0, 0] ^= 0x02000000
            return value

    with pytest.raises(runner.InputProtectionError, match="Dataset replay differs"):
        runner.ProtectedDataset(BadDataset(), protection(tmp_path))[2]
    manifest = read_json(tmp_path / "input_failure/failure.json")
    evidence = torch_load(manifest["evidence"])["evidence"]
    assert not runner.exact_equal(evidence["first"]["gt"], evidence["second"]["gt"])
    assert manifest["retry_allowed"] is False


def test_snapshot_detects_corrupted_clone_and_signed_zero(tmp_path, monkeypatch):
    original = torch.Tensor.clone

    def bad_clone(tensor, *args, **kwargs):
        result = original(tensor, *args, **kwargs)
        if tensor.dtype == torch.float32:
            result.reshape(-1).view(torch.int32)[0] ^= 0x02000000
        return result

    assert not runner.exact_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    monkeypatch.setattr(torch.Tensor, "clone", bad_clone)
    with pytest.raises(runner.InputProtectionError, match="Snapshot clone/source bytes differ"):
        runner.ProtectedDataset(RandomDataset(), protection(tmp_path))[2]
    assert (tmp_path / "input_failure/failure.json").exists()


class ToyModel:
    def feed_train_data(self, data):
        lam = random.random() + torch.rand(()).item() + np.random.rand()
        self.lq, self.gt = data["lq"] * lam, data["gt"] * lam

    def optimize_parameters(self, update):
        # Legacy PFA is supposed to consume global RNG in the optimizer phase.
        self.noise = torch.randn_like(self.lq)


def raw_batch():
    return {"lq": torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4),
            "gt": torch.ones(1, 3, 4, 4)}


def test_mixup_replay_consumes_rng_once_and_optimizer_may_consume_rng(tmp_path):
    seed_all(11)
    before = rng_state()
    expected = ToyModel()
    expected.feed_train_data(raw_batch())
    after_feed = rng_state()
    expected.optimize_parameters(1)
    after_optimize = rng_state()
    restore_rng(before)
    guard = protection(tmp_path)
    actual = runner.protect_model(ToyModel(), guard)
    actual.feed_train_data(raw_batch())
    assert runner.exact_equal(expected.lq, actual.lq)
    assert runner.exact_equal(after_feed, rng_state())
    actual.optimize_parameters(1)
    assert runner.exact_equal(after_optimize, rng_state())
    assert runner.exact_equal(expected.noise, actual.noise)
    assert guard.optimized


@pytest.mark.parametrize("fault", ["mixup", "raw", "optimizer"])
def test_feed_and_optimizer_faults_are_not_retried(tmp_path, fault):
    class BadModel(ToyModel):
        calls = 0

        def feed_train_data(self, data):
            super().feed_train_data(data)
            self.calls += 1
            if fault == "mixup" and self.calls == 2:
                self.lq.view(torch.int32)[0, 0, 0, 0] ^= 1
            if fault == "raw":
                data["gt"].add_(1)

        def optimize_parameters(self, update):
            super().optimize_parameters(update)
            if fault == "optimizer":
                self.gt.view(torch.int32)[0, 0, 0, 0] ^= 1

    actual = runner.protect_model(BadModel(), protection(tmp_path))
    with pytest.raises(runner.InputProtectionError):
        actual.feed_train_data(raw_batch())
        actual.optimize_parameters(1)
    assert actual.calls <= 2
    assert (tmp_path / "input_failure/failure.json").exists()


@pytest.mark.parametrize("variant,seed", [("C", 0), ("G1C", 2)])
def test_full_scheduler_four_updates_equal_two_save_restore_two(full_paths, variant, seed):
    first = runner.run(variant, "full", seed, full_paths, stop_after=4, tag="continuous")
    second = runner.run(variant, "full", seed, full_paths, stop_after=2, tag="resumed")
    assert read_json(second / "status.json")["status"] == "paused_at_checkpoint"
    # An interrupted, uncommitted tail must be archived and ignored on resume.
    with (second / "input_protection.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"update":3')
    runner.run(variant, "full", seed, full_paths, resume=True, stop_after=4, tag="resumed")
    expected = torch_load(read_json(first / "checkpoints/latest.json")["path"])
    actual = torch_load(read_json(second / "checkpoints/latest.json")["path"])
    for key in ("params", "optimizer", "scheduler", "rng", "sampler", "ema", "scaler", "pfa_rng"):
        assert runner.exact_equal(expected[key], actual[key]), key
    assert expected["pfa_rng"] is None
    first_rows = [json.loads(line) for line in (first / "input_protection.jsonl").read_text().splitlines()]
    second_rows = [json.loads(line) for line in (second / "input_protection.jsonl").read_text().splitlines()]
    assert first_rows == second_rows and all(runner.validate_proof_row(row) for row in first_rows)
    assert len(list((second / "interrupted_tails").iterdir())) == 1
    assert len(list((second / "checkpoints").glob("update_*.pth"))) == 1
    assert read_json(second / "effective_config.json")["train"]["total_iter"] == 300000
    assert not read_json(second / "summary.json")["completed"]
    assert read_json(second / "training_checks.json")["verified_updates"] == 4
    # A tampered committed proof must not be silently regenerated.
    second_rows[0]["raw_unchanged"] = False
    (second / "input_protection.jsonl").write_text("".join(json.dumps(row) + "\n" for row in second_rows))
    with pytest.raises(RuntimeError, match="proof is invalid"):
        runner.run(variant, "full", seed, full_paths, resume=True, stop_after=4, tag="resumed")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_protection_preserves_ordinary_training_state_and_rng(full_paths, seed):
    opt, directory = runner.configuration("G1C", "full", seed, full_paths)
    seed_all(seed)
    ordinary = runner.legacy.build_model(opt)
    ordinary_ds = runner.legacy.build_dataset(opt, full_paths)
    ordinary_sampler = runner.legacy.ExactSampler(len(ordinary_ds), seed + 20000)
    seed_all(seed + 10000)
    ordinary_loss = runner.legacy.train_update(ordinary, ordinary_ds, ordinary_sampler, 1, 16, 1)
    expected = copy.deepcopy(runner.legacy.checkpoint_payload(ordinary, ordinary_sampler, opt, 1, 0))
    expected_input = runner.checked_clone({"lq": ordinary.lq, "gt": ordinary.gt})
    seed_all(seed)
    guard = runner.Protection(directory, opt)
    protected = runner.build_protected_model(opt, guard)
    protected_ds = runner.build_protected_dataset(opt, full_paths, guard)
    protected_sampler = runner.legacy.ExactSampler(len(protected_ds), seed + 20000)
    seed_all(seed + 10000)
    loss, proof = runner.guarded_train_update(protected, protected_ds, protected_sampler, 1, guard)
    actual = runner.legacy.checkpoint_payload(protected, protected_sampler, opt, 1, 0)
    assert ordinary_loss == loss
    assert runner.exact_equal(expected, actual)
    assert runner.exact_equal(expected_input, {"lq": protected.lq, "gt": protected.gt})
    assert runner.validate_proof_row(proof)


def test_failed_input_directory_cannot_resume(full_paths):
    _, directory = runner.configuration("C", "full", 0, full_paths)
    (directory / "input_failure").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Input protection failed"):
        runner.run("C", "full", 0, full_paths, resume=True, stop_after=1)
