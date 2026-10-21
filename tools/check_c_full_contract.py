"""Actual RN50/batch-16 C or G1C initialization, legacy-PFA and resume checks.

Four diagnostic updates retain the full 300000-update configuration/schedule.
This short check does not establish a bitwise-equivalent 300000-step trajectory
or claim that the unresolved CPU snapshot anomaly has been repaired.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from basicsr.utils import get_root_logger
from basicsr.models.losses.losses import L1StructureLoss
from tests.test_resume_contract import CUDA_ATOL, CUDA_RTOL, compare_states
from tools import run_c_only_experiment as runner
from tools import run_experiment as legacy
from tools.experiment_common import (
    atomic_json, digest, load_paths, restore_rng, rng_state, seed_all,
    sha256_file, source_hash, torch_load, utc_now,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rng_fingerprint(state):
    numpy_state = state["numpy"]
    return {"python": digest(state["python"]),
            "numpy": digest({"kind": numpy_state[0],
                "keys_sha256": hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
                "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gauss": numpy_state[4]}),
            "torch": runner.tensor_hash(state["torch"]),
            "cuda": [runner.tensor_hash(value) for value in state["cuda"]]}


def cuda_rng_consumed(before, after):
    return (len(before["cuda"]) == len(after["cuda"]) > 0
            and any(not runner.exact_equal(a, b) for a, b in zip(before["cuda"], after["cuda"])))


def load_legacy_architecture(legacy_root):
    """Execute the actual old files, retaining their own imported class objects."""
    base = Path(legacy_root).resolve() / "basicsr/models/archs"
    names = ("rgb_calibration", "CLIPEncoder_util", "CLIPDenoising_arch")
    saved = {}
    modules = {}
    provenance = {}
    try:
        for name in names:
            path = base / (name + ".py")
            require(path.is_file(), f"Missing legacy architecture dependency: {path}")
            key = "basicsr.models.archs." + name
            saved[key] = sys.modules.get(key)
            spec = importlib.util.spec_from_file_location("_c_contract_legacy_" + name, path)
            require(spec is not None and spec.loader is not None, f"Cannot import {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[key] = module
            modules[name] = module
            current = ROOT / "basicsr/models/archs" / (name + ".py")
            provenance[name] = {"legacy_path": str(path), "legacy_sha256": sha256_file(path),
                                "current_sha256": sha256_file(current)}
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module
    return modules["CLIPDenoising_arch"].CLIPDenoising, provenance


def verified_save(path, model, sampler, opt, update):
    """Check the actual serialized state, including exact RNG and counters."""
    payload = legacy.checkpoint_payload(model, sampler, opt, update, 0.0)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    loaded = legacy.validate_checkpoint(path, opt, expected=update)
    comparison = compare_states(payload, loaded, device="cpu")
    require(comparison["accepted"] and comparison["bitwise_equal"],
            "Checkpoint serialization changed state: " + str(comparison["failures"]))
    require(runner.exact_equal(payload, loaded), "Repeated checkpoint reads disagree")
    return {"path": str(path), "sha256": sha256_file(path), "comparison": comparison}


def fresh_model(opt, paths, directory):
    directory.mkdir(parents=True, exist_ok=True)
    protection = runner.Protection(directory, opt)
    seed_all(opt["seed"])
    model = runner.build_protected_model(opt, protection)
    initialization = runner.initialization_contract(model, opt)
    dataset = runner.build_protected_dataset(opt, paths, protection)
    require(len(dataset) == 432, "The actual diagnostic requires all 432 training images")
    sampler = legacy.ExactSampler(len(dataset), opt["seed"] + 20000)
    seed_all(opt["seed"] + 10000)
    return model, dataset, sampler, protection, initialization


def diagnostic_updates(model, dataset, sampler, protection, first, last, report, phase, persist):
    rows = []
    for update in range(first, last + 1):
        before = runner.checked_clone(rng_state())
        loss, proof = runner.guarded_train_update(model, dataset, sampler, update, protection)
        torch.cuda.synchronize()
        after = rng_state()
        require(runner.validate_proof_row(proof), f"Incomplete protected update {update}")
        require(math.isfinite(loss), f"Nonfinite loss at update {update}")
        require(cuda_rng_consumed(before, after), "Legacy global CUDA PFA RNG did not advance")
        require(model.net_g.pfa.state_dict() is None, "A private PFA generator was introduced")
        row = {"update": update, "loss": loss, "lr": model.optimizer_g.param_groups[0]["lr"],
               "proof": proof, "rng_before": rng_fingerprint(before), "rng_after": rng_fingerprint(after),
               "global_pfa_rng_consumed": True}
        components = getattr(model.cri_pix, "last_components", {})
        row["loss_components"] = {key: float(value) for key, value in components.items()}
        rows.append(row)
        report.setdefault("training_rows", {}).setdefault(phase, []).append(row)
        persist()
    return rows


def compare_legacy(model, opt, legacy_root, report, persist):
    legacy_class, provenance = load_legacy_architecture(legacy_root)
    report["legacy_files"] = provenance
    parameters = copy.deepcopy(opt["network_g"])
    parameters.pop("type", None)
    require(parameters.pop("pfa_rho", None) == 0.0, "C must preserve rho=0 PFA")
    require(parameters.pop("pfa_seed", "missing") is None, "C must preserve global PFA RNG")
    seed_all(opt["seed"])
    old = legacy_class(**parameters).to(model.device).train()
    old_init_rng = rng_state()
    # Rebuild the protected model from the exact same seed, not from old weights.
    seed_all(opt["seed"])
    candidate = runner.build_protected_model(opt, runner.Protection(Path(report["diagnostic_directory"]) / "legacy_candidate", opt))
    new_init_rng = rng_state()
    initial = compare_states({"params": old.state_dict()}, {"params": candidate.net_g.state_dict()}, device="cpu")
    report["legacy_initialization"] = {"full_state_dict": initial,
        "global_rng_equal": runner.exact_equal(old_init_rng, new_init_rng),
        "adapter_mode": parameters["adapter_mode"],
        "new_initialization": runner.initialization_contract(candidate, opt)}
    persist()
    require(initial["accepted"] and initial["bitwise_equal"], "Legacy/new initial full state dictionaries differ")
    require(runner.exact_equal(old.state_dict(), candidate.net_g.state_dict()), "Initial state repeated reads differ")
    require(report["legacy_initialization"]["global_rng_equal"], "New construction consumes different global RNG")
    value = torch.linspace(0, 1, 16 * 3 * 128 * 128, dtype=torch.float32, device=model.device).reshape(16, 3, 128, 128)
    value_snapshot = runner.checked_clone(value)
    seed_all(opt["seed"] + 10000)
    shared_rng = runner.checked_clone(rng_state())
    original_randn_like = torch.randn_like

    def forward_with_trace(network):
        draws = []
        def tracked_randn_like(*args, **kwargs):
            result = original_randn_like(*args, **kwargs)
            draws.append({"shape": list(result.shape), "dtype": str(result.dtype),
                          "sha256": runner.tensor_hash(result)})
            return result
        restore_rng(shared_rng)
        torch.randn_like = tracked_randn_like
        try:
            with torch.no_grad():
                result = network(value)
            torch.cuda.synchronize()
        finally:
            torch.randn_like = original_randn_like
        return result, rng_state(), draws

    old_output, old_after, old_draws = forward_with_trace(old)
    new_output, new_after, new_draws = forward_with_trace(candidate.net_g)
    forward = {"input_shape": list(value.shape), "input_sha256": runner.tensor_hash(value),
        "output_bitwise_equal": runner.exact_equal(old_output, new_output),
        "max_abs_difference": float((old_output - new_output).abs().max()),
        "legacy_output_sha256": runner.tensor_hash(old_output), "new_output_sha256": runner.tensor_hash(new_output),
        "input_unchanged": runner.exact_equal(value, value_snapshot),
        "legacy_pfa_draws": old_draws, "new_pfa_draws": new_draws,
        "pfa_draws_exact": old_draws == new_draws, "global_rng_after_equal": runner.exact_equal(old_after, new_after),
        "legacy_global_cuda_rng_consumed": cuda_rng_consumed(shared_rng, old_after),
        "new_global_cuda_rng_consumed": cuda_rng_consumed(shared_rng, new_after),
        "new_private_pfa_state": candidate.net_g.pfa.state_dict(),
        "bn_and_full_state_after_forward_exact": runner.exact_equal(old.state_dict(), candidate.net_g.state_dict())}
    report["legacy_forward"] = forward
    persist()
    for key in ("output_bitwise_equal", "input_unchanged", "pfa_draws_exact", "global_rng_after_equal",
                "legacy_global_cuda_rng_consumed", "new_global_cuda_rng_consumed", "bn_and_full_state_after_forward_exact"):
        require(forward[key], "Legacy/new forward contract failed: " + key)
    require(len(old_draws) == len(new_draws) == 4, "Expected four legacy global PFA draws")
    require(forward["new_private_pfa_state"] is None, "Unexpected private PFA state")
    del old, candidate, old_output, new_output, value, value_snapshot


def check_contract(variant, paths, report_path, legacy_root, report, seed=0):
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{variant.lower()}_seed{seed}_full_contract_", dir=report_path.parent))
    report.update(accepted=False, diagnostic_only=True, variant=variant, batch_size=16, seed=seed,
        diagnostic_updates=4, checkpoint_update=2, configured_budget=300000,
        diagnostic_directory=str(temporary), evidence_preserved=True, phase="initializing",
        source_hash=source_hash(), started_at=utc_now(),
        scope="Actual short legacy initialization/forward/PFA and four-update restoration checks; not a complete 300000-step trajectory proof",
        unresolved_cpu_anomaly="One CPU thread plus fail-closed repeated generation/snapshot/transfer checks are detection measures; root cause remains unknown",
        acceptance={"atol": CUDA_ATOL, "rtol": CUDA_RTOL,
            "computed_cuda_float_states": "existing compare_states tolerance; report bitwise equality separately",
            "inputs_rng_sampler_counters_immediate_restore": "exact"})
    persist = lambda: atomic_json(report_path, report)
    persist()
    try:
        require(torch.cuda.is_available(), "Actual RN50 contract requires CUDA")
        require(paths["device"] == "cuda" and paths["micro_batch"] == 16 and paths["accumulation"] == 1,
                "Actual contract requires CUDA and true batch 16")
        require(int(paths.get("cpu_threads", 0)) == 1, "This diagnostic requires cpu_threads=1")
        torch.set_num_threads(1)
        runner.configure_execution(True)
        torch.cuda.set_device(int(paths.get("gpu_index", 0)))
        get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)
        diagnostic_paths = dict(paths, output_root=str(temporary / "runs"))
        opt, directory = runner.configuration(variant, "full", seed, diagnostic_paths, tag="contract")
        report.update(config_hash=opt["config_hash"], effective_config=opt,
                      rn50_sha256=opt["pretrained_sha256"], device_name=torch.cuda.get_device_name(),
                      cpu_threads=torch.get_num_threads(), execution_contract=opt["execution_contract"])
        report["execution_flags"] = {
            "current": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
            "historical_server_default_verified_separately": {
                "deterministic_algorithms": False, "cudnn_deterministic": False,
                "cudnn_benchmark": False, "cudnn_allow_tf32": True,
                "cuda_matmul_allow_tf32": False},
            "comparison_scope": "Legacy and new architectures run under the same current explicit flags; historical entire trajectories are not compared"}
        require(torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32,
                "Full contract must preserve historical cuDNN TF32=True and matmul TF32=False")
        require(opt["train"]["total_iter"] == 300000 and opt["path"]["pretrain_network_g"] is None,
                "Diagnostic must use fresh full-budget initialization, without C0")
        directory.mkdir(parents=True, exist_ok=True)
        seed_all(seed)
        initial_model = runner.build_protected_model(opt, runner.Protection(directory / "initialization", opt))
        loss = initial_model.cri_pix
        require(isinstance(loss, L1StructureLoss) and loss.structure_weight == 0.05 and loss.eps == 0.001
                and loss.deterministic_reflect_padding, "Actual structure-loss settings differ")
        report["loss_contract"] = {"class": type(loss).__name__, "structure_weight": loss.structure_weight,
            "eps": loss.eps, "deterministic_reflect_padding": loss.deterministic_reflect_padding,
            "loss_weight": loss.loss_weight, "trainable_loss_parameters": sum(p.numel() for p in loss.parameters())}
        report["phase"] = "legacy_initialization_and_forward"
        persist()
        compare_legacy(initial_model, opt, legacy_root, report, persist)
        del initial_model, loss
        gc.collect(); torch.cuda.empty_cache()

        report["phase"] = "continuous_four_updates"
        model, dataset, sampler, protection, initialization = fresh_model(opt, diagnostic_paths, directory / "continuous")
        report["training_initialization"] = initialization
        expected_rows = diagnostic_updates(model, dataset, sampler, protection, 1, 4, report, "continuous", persist)
        report["continuous_checkpoint"] = verified_save(temporary / "continuous_update4.pth", model, sampler, opt, 4)
        del model, dataset, sampler, protection
        gc.collect(); torch.cuda.empty_cache()

        report["phase"] = "interrupted_first_two_updates"
        model, dataset, sampler, protection, _ = fresh_model(opt, diagnostic_paths, directory / "interrupted")
        first_rows = diagnostic_updates(model, dataset, sampler, protection, 1, 2, report, "interrupted", persist)
        report["interrupted_checkpoint"] = verified_save(temporary / "interrupted_update2.pth", model, sampler, opt, 2)
        del model, dataset, sampler, protection
        gc.collect(); torch.cuda.empty_cache()

        report["phase"] = "restore_and_updates_three_four"
        model, dataset, sampler, protection, _ = fresh_model(opt, diagnostic_paths, directory / "resumed")
        saved = legacy.validate_checkpoint(temporary / "interrupted_update2.pth", opt, expected=2)
        legacy.restore_checkpoint(model, sampler, saved)
        restored = legacy.checkpoint_payload(model, sampler, opt, 2, 0.0)
        report["immediate_restore"] = compare_states(saved, restored, device="cpu")
        persist()
        require(report["immediate_restore"]["accepted"] and report["immediate_restore"]["bitwise_equal"]
                and runner.exact_equal(saved, restored), "Immediate restored state is not bitwise exact")
        del saved, restored
        resumed_rows = diagnostic_updates(model, dataset, sampler, protection, 3, 4, report, "resumed", persist)
        report["resumed_checkpoint"] = verified_save(temporary / "resumed_update4.pth", model, sampler, opt, 4)
        expected = torch_load(temporary / "continuous_update4.pth")
        actual = torch_load(temporary / "resumed_update4.pth")
        report["continuous_vs_resumed"] = compare_states(expected, actual, device="cuda")
        row_checks = []
        for wanted, observed in zip(expected_rows, first_rows + resumed_rows):
            exact_fields = all(wanted[key] == observed[key] for key in
                ("update", "lr", "proof", "rng_before", "rng_after", "global_pfa_rng_consumed"))
            components_close = wanted["loss_components"].keys() == observed["loss_components"].keys() and all(
                math.isclose(value, observed["loss_components"][key], rel_tol=CUDA_RTOL, abs_tol=CUDA_ATOL)
                for key, value in wanted["loss_components"].items())
            loss_close = math.isclose(wanted["loss"], observed["loss"], rel_tol=CUDA_RTOL, abs_tol=CUDA_ATOL)
            row_checks.append({"update": wanted["update"], "inputs_controls_exact": exact_fields,
                "loss_accepted": loss_close, "components_accepted": components_close,
                "loss_bitwise_equal": wanted["loss"] == observed["loss"]})
        report["row_checks"] = row_checks
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        persist()
        require(report["continuous_vs_resumed"]["accepted"], "Continuous/resumed computed state exceeds existing tolerance")
        require(all(row["inputs_controls_exact"] and row["loss_accepted"] and row["components_accepted"] for row in row_checks),
                "Continuous/resumed inputs, controls or loss differ")
        report.update(accepted=True, phase="completed", ended_at=utc_now())
        del model, dataset, sampler, protection, expected, actual
        gc.collect(); torch.cuda.empty_cache()
        require(temporary.parent == report_path.parent, "Diagnostic cleanup escaped report directory")
        shutil.rmtree(temporary)
        report["evidence_preserved"] = False
    except BaseException as error:
        report.update(accepted=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), ended_at=utc_now())
        raise
    finally:
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("C", "G1C"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--legacy-root", required=True)
    args = parser.parse_args()
    report = {"accepted": False, "diagnostic_only": True}
    started = time.perf_counter()
    try:
        check_contract(args.variant, load_paths(args.paths), args.report, args.legacy_root, report, args.seed)
    except BaseException as error:
        report.update(accepted=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr)
    report["wall_seconds"] = time.perf_counter() - started
    atomic_json(args.report, report)
    print(json.dumps({"accepted": report["accepted"], "variant": args.variant, "seed": args.seed, "report": args.report}), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
