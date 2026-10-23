"""Contracts for one materialized batch shared by independent B0/A/C models.

CPU tests substitute only the network. The optional CLI uses the actual RN50,
fixed C0 and CBSD432 at batch 16 in a temporary four-update diagnostic run.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch

from basicsr.models.image_restoration_model import ImageCleanModel, Mixing_Augment
from basicsr.models.losses.losses import L1StructureLoss, deterministic_reflect_pad_one
from tests.test_ac_training_contract import continuation_paths
from tests.test_resume_contract import CUDA_ATOL, CUDA_RTOL, compare_states
from tools.experiment_common import atomic_json, load_paths, read_json, torch_load
from tools.run_experiment import checkpoint_payload
from tools import run_ac_triplet as triplet


@pytest.fixture(autouse=True)
def restore_backend_settings():
    settings = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    yield
    torch.use_deterministic_algorithms(settings[0], warn_only=settings[1])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = settings[2:4]
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = settings[4:6]


@pytest.mark.parametrize("shape", [(2, 3, 2, 2), (2, 3, 7, 11)])
def test_deterministic_reflection_preserves_values_and_gradient(shape):
    generator = torch.Generator().manual_seed(728)
    image = torch.randn(shape, generator=generator, dtype=torch.float64).transpose(-1, -2)
    reference_input = image.detach().requires_grad_(True)
    actual_input = image.detach().requires_grad_(True)
    expected = torch.nn.functional.pad(reference_input, (1, 1, 1, 1), mode="reflect")
    actual = deterministic_reflect_pad_one(actual_input)
    assert torch.equal(expected, actual)
    cotangent = torch.randn(expected.shape, generator=generator, dtype=torch.float64)
    expected.backward(cotangent)
    actual.backward(cotangent)
    assert torch.allclose(reference_input.grad, actual_input.grad, atol=1e-12, rtol=1e-12)


def test_deterministic_structure_loss_preserves_definition_and_gradient():
    generator = torch.Generator().manual_seed(729)
    image = torch.randn((2, 3, 7, 11), generator=generator, dtype=torch.float64)
    target = torch.randn(image.shape, generator=generator, dtype=torch.float64)
    reference_input = image.clone().requires_grad_(True)
    actual_input = image.clone().requires_grad_(True)
    reference = L1StructureLoss()
    actual = L1StructureLoss(deterministic_reflect_padding=True)
    expected_loss, actual_loss = reference(reference_input, target), actual(actual_input, target)
    assert torch.equal(expected_loss, actual_loss)
    assert reference.last_components == actual.last_components
    expected_loss.backward()
    actual_loss.backward()
    assert torch.allclose(reference_input.grad, actual_input.grad, atol=1e-12, rtol=1e-12)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    assert all(torch.equal(value, actual.state_dict()[key]) for key, value in reference.state_dict().items())


def test_deterministic_execution_fails_closed_on_incompatible_cublas_setting(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG must equal"):
        triplet.configure_deterministic_execution(cuda=False)


def test_actual_contract_failure_preserves_partial_report_and_evidence(tmp_path, monkeypatch):
    report_path, report = tmp_path / "diagnostic.json", {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def interrupted(*args, **kwargs):
        directory = Path(args[2]["output_root"])
        directory.mkdir()
        (directory / "failure_evidence.txt").write_text("preserve failure", encoding="utf-8")
        raise RuntimeError("injected diagnostic failure")

    monkeypatch.setattr(triplet, "run", interrupted)
    paths = {"micro_batch": 16, "budgets": {"smoke": 200}}
    with pytest.raises(RuntimeError, match="injected diagnostic failure"):
        actual_triplet_contract(paths, paths, report_path, report=report)
    persisted = read_json(report_path)
    assert persisted == report
    assert persisted["accepted"] is False and persisted["evidence_preserved"] is True
    assert persisted["phase"] == "continuous_four_updates"
    assert persisted["execution_contract"] == triplet.EXECUTION_CONTRACT
    assert (Path(persisted["diagnostic_directory"]) / "runs" / "failure_evidence.txt").is_file()
    assert "injected diagnostic failure" in persisted["error"]


def _structure_paths(paths):
    return dict(paths, rho=0.0)


def _coordinator(paths):
    return Path(paths["output_root"]) / "smoke" / "_triplet" / "seed100"


def _rows(directory):
    return [json.loads(line) for line in (directory / "train.jsonl").read_text(encoding="utf-8").splitlines()]


def _checkpoint(directory):
    return torch_load(read_json(directory / "checkpoints" / "latest.json")["path"])


def _without_elapsed(checkpoint):
    checkpoint["elapsed_seconds"] = 0.0
    return checkpoint


def _assert_rows_paired(rows):
    expected = rows["B0"]
    for group in triplet.GROUPS:
        assert [row["update"] for row in rows[group]] == list(range(1, len(expected) + 1))
        for row, baseline in zip(rows[group], expected):
            assert row["exact_reference_and_unchanged"] is True
            assert row["shared_rng_unchanged"] is True
            for kind in ("input", "target"):
                assert {row[f"{prefix}{kind}_sha256"] for prefix in ("", "reference_", "pre_", "post_")} == {
                    baseline[f"{kind}_sha256"]}
            assert row["lr"] == baseline["lr"]


def _remove_fixture_runs(paths):
    root = Path(paths["output_root"]).resolve()
    assert root.is_relative_to(Path(paths["project_root"]).resolve())
    shutil.rmtree(root)


def test_triplet_materializes_once_with_independent_models_and_declared_interventions(continuation_paths, monkeypatch):
    paths = continuation_paths
    calls = {"data": 0, "mixup": 0}
    models = []
    original_getitem = triplet.Dataset_GaussianDenoising.__getitem__
    original_mixup = Mixing_Augment.__call__
    original_build = triplet.build_model

    def getitem(self, index):
        calls["data"] += 1
        return original_getitem(self, index)

    def mixup(self, target, value):
        calls["mixup"] += 1
        return original_mixup(self, target, value)

    def build(opt):
        model = original_build(opt)
        models.append(model)
        return model

    monkeypatch.setattr(triplet.Dataset_GaussianDenoising, "__getitem__", getitem)
    monkeypatch.setattr(Mixing_Augment, "__call__", mixup)
    monkeypatch.setattr(triplet, "build_model", build)
    directories = triplet.run("smoke", 100, paths, _structure_paths(paths))
    rows = {group: _rows(directory) for group, directory in directories.items()}
    _assert_rows_paired(rows)
    assert calls == {"data": 4 * paths["micro_batch"], "mixup": 4}
    assert len(models) == 3
    assert len({id(model.optimizer_g) for model in models}) == 3
    assert len({id(model.net_g.pfa) for model in models}) == 3
    assert len({model.lq.data_ptr() for model in models}) == 3
    assert len({model.gt.data_ptr() for model in models}) == 3
    assert all(model.optimizer_g.state for model in models)
    optimizer_storages = [{value.data_ptr() for state in model.optimizer_g.state.values()
                           for value in state.values() if isinstance(value, torch.Tensor)} for model in models]
    assert all(not first.intersection(second) for index, first in enumerate(optimizer_storages)
               for second in optimizer_storages[index + 1:])
    configs = {group: read_json(directory / "effective_config.json") for group, directory in directories.items()}
    assert len({opt["paired_protocol_hash"] for opt in configs.values()}) == 1
    assert [configs[group]["network_g"]["pfa_rho"] for group in triplet.GROUPS] == [0.0, 0.5, 0.0]
    assert [configs[group]["train"]["pixel_opt"]["type"] for group in triplet.GROUPS] == ["L1Loss", "L1Loss", "L1StructureLoss"]
    assert configs["B3"]["train"]["pixel_opt"]["deterministic_reflect_padding"] is True
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert rows["B0"][0]["lr"] == 1e-5 and rows["B0"][-1]["lr"] == 1e-6
    assert all(a["lr"] > b["lr"] for a, b in zip(rows["B0"], rows["B0"][1:]))
    assert rows["B1"][-1]["loss"] != rows["B0"][-1]["loss"]
    for row in rows["B3"]:
        assert row["structure"] > 0 and row["weighted_structure"] > 0
        assert math.isclose(row["loss"], row["pixel_l1"] + row["weighted_structure"], abs_tol=1e-7)
    for group, directory in directories.items():
        checks = read_json(directory / "training_checks.json")
        assert checks["strict_c0_params"] and checks["fresh_adamw"] and checks["fresh_scheduler"]
        assert checks["encoder_parameters_unchanged"] and checks["encoder_gradients_absent"]
        assert checks["encoder_bn_training_after"] and checks["exact_reference_pre_post_every_update"]
        assert checks["shared_rng_unchanged_every_model_step"]
        assert read_json(directory / "status.json")["status"] == "completed"
        assert len(list((directory / "checkpoints").glob("update_*.pth"))) == 1
    assert read_json(_coordinator(paths) / "commit.json")["update"] == 4


def test_triplet_resume_replays_every_group_exactly_and_trims_uncommitted_logs(continuation_paths):
    paths = continuation_paths
    directories = triplet.run("smoke", 100, paths, _structure_paths(paths))
    expected = {group: _without_elapsed(_checkpoint(directory)) for group, directory in directories.items()}
    expected_rows = {group: _rows(directory) for group, directory in directories.items()}
    _remove_fixture_runs(paths)
    triplet.run("smoke", 100, paths, _structure_paths(paths), stop_after=2)
    assert read_json(_coordinator(paths) / "commit.json")["update"] == 2
    for directory in directories.values():
        assert read_json(directory / "status.json")["status"] == "paused_at_checkpoint"
        with (directory / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"update": 3, "loss":')
    triplet.run("smoke", 100, paths, _structure_paths(paths), resume=True)
    actual_rows = {group: _rows(directory) for group, directory in directories.items()}
    _assert_rows_paired(actual_rows)
    for group, directory in directories.items():
        comparison = compare_states(expected[group], _without_elapsed(_checkpoint(directory)), device="cpu")
        assert comparison["accepted"] and comparison["bitwise_equal"], comparison["failures"]
        assert [{k: v for k, v in row.items() if k not in {"seconds", "triplet_wall_seconds"}} for row in actual_rows[group]] == [
            {k: v for k, v in row.items() if k not in {"seconds", "triplet_wall_seconds"}} for row in expected_rows[group]]
    assert triplet.run("smoke", 100, paths, _structure_paths(paths), resume=True) == directories


def test_triplet_failed_transaction_retains_previous_commit_and_recovers(continuation_paths, monkeypatch):
    paths = continuation_paths
    directories = triplet.run("smoke", 100, paths, _structure_paths(paths))
    expected = {group: _without_elapsed(_checkpoint(directory)) for group, directory in directories.items()}
    _remove_fixture_runs(paths)
    triplet.run("smoke", 100, paths, _structure_paths(paths), stop_after=2)
    committed = read_json(_coordinator(paths) / "commit.json")
    real_atomic_json = triplet.atomic_json
    failures = []

    def interrupt_commit(path, value):
        if Path(path).name == "commit.json" and value["update"] == 4 and not failures:
            failures.append(True)
            raise OSError("injected interruption before coordinator commit")
        return real_atomic_json(path, value)

    monkeypatch.setattr(triplet, "atomic_json", interrupt_commit)
    with pytest.raises(OSError, match="injected interruption"):
        triplet.run("smoke", 100, paths, _structure_paths(paths), resume=True)
    assert read_json(_coordinator(paths) / "commit.json") == committed
    for group, directory in directories.items():
        assert Path(committed["groups"][group]["path"]).is_file()
        assert (directory / "checkpoints" / "update_000004.pth").is_file()
        assert read_json(directory / "checkpoints" / "latest.json")["update"] == 2
        assert read_json(directory / "status.json")["status"] == "failed"
    triplet.run("smoke", 100, paths, _structure_paths(paths), resume=True)
    for group, directory in directories.items():
        comparison = compare_states(expected[group], _without_elapsed(_checkpoint(directory)), device="cpu")
        assert comparison["accepted"] and comparison["bitwise_equal"], comparison["failures"]
        assert [row["update"] for row in _rows(directory)] == [1, 2, 3, 4]
        assert len(list((directory / "checkpoints").glob("update_*.pth"))) == 1


@pytest.mark.parametrize("fault", ["target_mutation", "shared_rng"])
def test_triplet_model_side_effects_fail_closed_and_preserve_evidence(continuation_paths, monkeypatch, fault):
    paths = continuation_paths
    original_optimize = ImageCleanModel.optimize_parameters

    def contaminated_step(self, current):
        original_optimize(self, current)
        if fault == "target_mutation":
            self.gt[0, 0, 0, 0].add_(1)
        else:
            torch.rand(1)

    monkeypatch.setattr(ImageCleanModel, "optimize_parameters", contaminated_step)
    with pytest.raises(RuntimeError, match="Exact triplet pairing failed at update 1"):
        triplet.run("smoke", 100, paths, _structure_paths(paths))
    coordinator = _coordinator(paths)
    assert not (coordinator / "commit.json").exists()
    evidence_path = coordinator / "failure_update_000001.pth"
    assert evidence_path.is_file()
    evidence = torch_load(evidence_path)
    assert evidence["update"] == 1 and evidence["group"] == "B0"
    assert all(key in evidence for key in ("raw_batch", "reference", "observed", "rng_before_model", "rng_after_model", "sampler"))
    if fault == "target_mutation":
        assert not torch.equal(evidence["reference"]["gt"], evidence["observed"]["gt"])
        assert torch.equal(evidence["reference"]["lq"], evidence["observed"]["lq"])
    else:
        assert not triplet.exact_state_equal(evidence["rng_before_model"], evidence["rng_after_model"])
    assert read_json(coordinator / "status.json")["status"] == "failed"


def actual_triplet_contract(paths, structure_paths, report_path, report=None):
    """Actual four-step triplet and interrupted replay, isolated from experiments."""
    if report is None:
        report = {}
    report.update({"accepted": False, "diagnostic_only": True, "device": "cuda", "batch_size": 16,
              "updates": 4, "checkpoint_at": 2, "groups": {}, "immediate_restore": {},
              "phase_resources": [], "initialization_checks": {},
              "execution_contract": copy.deepcopy(triplet.EXECUTION_CONTRACT),
              "float_acceptance": {"atol": CUDA_ATOL, "rtol": CUDA_RTOL,
                  "scope": "computed params/optimizer/ema floats only; inputs/control states exact"}})
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ac_triplet_contract_", dir=report_path.parent))
    report.update(diagnostic_directory=str(root), evidence_preserved=True, phase="initializing")
    atomic_json(report_path, report)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Actual triplet diagnostic requires CUDA")
        if paths["micro_batch"] != 16 or structure_paths["micro_batch"] != 16:
            raise ValueError("Actual triplet diagnostic must preserve the authorized batch size 16")
        diagnostic_paths = []
        for source in (paths, structure_paths):
            value = copy.deepcopy(source)
            value.update(device="cuda", output_root=str(root / "runs"), checkpoint_every=2,
                         checkpoint_keep=1, status_every=1, budgets={**source["budgets"], "smoke": 4})
            diagnostic_paths.append(value)
        first_paths, third_paths = diagnostic_paths
        report["phase"] = "continuous_four_updates"
        atomic_json(report_path, report)
        directories = triplet.run("smoke", 100, first_paths, third_paths)

        def record_resources(phase):
            summaries = [read_json(directory / "summary.json") for directory in directories.values()]
            report["phase_resources"].append({"phase": phase,
                "peak_allocated_mib": max(value["peak_allocated_mib"] for value in summaries),
                "peak_reserved_mib": max(value["peak_reserved_mib"] for value in summaries),
                "invocation_wall_seconds": max(value["invocation_wall_seconds"] for value in summaries)})
            for key in ("peak_allocated_mib", "peak_reserved_mib"):
                report[key] = max(value[key] for value in report["phase_resources"])
            atomic_json(report_path, report)

        record_resources("continuous_four_updates")
        for group, directory in directories.items():
            manifest = read_json(directory / "manifest.json")
            report["initialization_checks"][group] = manifest["initialization_checks"]
            report["source_hash"] = manifest["source_hash"]
            report["c0_sha256"] = manifest["c0_sha256"]
        expected_rows = {group: _rows(directory) for group, directory in directories.items()}
        _assert_rows_paired(expected_rows)
        expected_files = {}
        for group, directory in directories.items():
            expected_files[group] = root / f"expected_{group}.pth"
            shutil.copyfile(read_json(directory / "checkpoints" / "latest.json")["path"], expected_files[group])
        run_root = Path(first_paths["output_root"]).resolve()
        assert run_root.parent == root.resolve()
        shutil.rmtree(run_root)
        torch.cuda.empty_cache()
        report["phase"] = "first_two_updates"
        atomic_json(report_path, report)
        triplet.run("smoke", 100, first_paths, third_paths, stop_after=2)
        record_resources("first_two_updates")
        restore = triplet.restore_checkpoint

        def checked_restore(model, sampler, checkpoint):
            restore(model, sampler, checkpoint)
            restored = checkpoint_payload(model, sampler, model.opt, checkpoint["update"], checkpoint["elapsed_seconds"])
            comparison = compare_states(checkpoint, restored, device="cpu")
            report["immediate_restore"][checkpoint["experiment"]] = comparison
            atomic_json(report_path, report)
            assert comparison["accepted"] and comparison["bitwise_equal"], comparison["failures"]

        triplet.restore_checkpoint = checked_restore
        report["phase"] = "resumed_updates_three_four"
        atomic_json(report_path, report)
        try:
            triplet.run("smoke", 100, first_paths, third_paths, resume=True)
        finally:
            triplet.restore_checkpoint = restore
        record_resources("resumed_updates_three_four")
        actual_rows = {group: _rows(directory) for group, directory in directories.items()}
        _assert_rows_paired(actual_rows)
        report["phase"] = "compare_resumed_states"
        for group, directory in directories.items():
            expected = _without_elapsed(torch_load(expected_files[group]))
            actual = _without_elapsed(_checkpoint(directory))
            comparison = compare_states(expected, actual, device="cuda")
            report["groups"][group] = comparison
            atomic_json(report_path, report)
            for wanted, observed in zip(expected_rows[group], actual_rows[group]):
                for key, value in wanted.items():
                    if key in {"seconds", "triplet_wall_seconds"}:
                        continue
                    if key in {"loss", "pixel_l1", "structure", "weighted_structure"}:
                        assert math.isclose(value, observed[key], abs_tol=CUDA_ATOL, rel_tol=CUDA_RTOL)
                    else:
                        assert value == observed[key], (group, wanted["update"], key)
            del expected, actual
        assert all(value["accepted"] for value in report["groups"].values()), {
            group: value["failures"] for group, value in report["groups"].items() if not value["accepted"]}
        report.update(accepted=True, paired_input_target_exact=True,
                      resumed_inputs_and_controls_exact=True, all_three_restores_exact=True,
                      phase="completed")
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    else:
        assert root.resolve().parent == report_path.parent
        shutil.rmtree(root)
        report["evidence_preserved"] = False
    finally:
        atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--structure-paths", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = {"accepted": False, "diagnostic_only": True}
    try:
        actual_triplet_contract(load_paths(args.paths), load_paths(args.structure_paths), args.report, report=report)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr)
    report["wall_seconds"] = time.perf_counter() - started
    atomic_json(args.report, report)
    print(json.dumps({"accepted": report["accepted"], "report": args.report}), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
