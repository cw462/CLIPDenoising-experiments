"""Proposal A mathematical, compatibility and real CUDA recovery contracts."""
import copy
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from basicsr.models.archs.spatial_pfa import FeaturePerturbation, correlated_normal
from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.utils import get_root_logger
from tools.experiment_common import ROOT, load_paths, seed_all
from tools.run_experiment import ExactSampler, build_model, checkpoint_payload, restore_checkpoint, train_update
from tools.run_pfa_experiment import configuration, tensor_hash
from tools.evaluate_pfa import CONDITIONS, fixed_split, noisy_input, summarize
from tests.test_resume_contract import compare_states


@pytest.fixture(autouse=True)
def small_thread_pool():
    torch.set_num_threads(4)
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)


def correlation(x):
    a, b = x.flatten().double(), torch.roll(x, 1, -1).flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False)))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_marginals_edges_and_expected_neighbor_correlation(device):
    reference = torch.empty(512, 8, 8, 8, device=device)
    stream = FeaturePerturbation(0.5, 123)
    field = stream(reference, 3)
    assert abs(float(field.mean())) < 0.015
    assert abs(float(field.std()) - 1) < 0.015
    # Includes all circular-padding boundary locations, not only interiors.
    assert float(field.std(dim=(0, 1)).sub(1).abs().max()) < 0.08
    assert float(field.mean(dim=(0, 1)).abs().max()) < 0.07
    # Adjacent 3x3 windows share 6/9 samples; rho=0.5 => covariance 1/3.
    assert abs(correlation(field) - 1 / 3) < 0.025
    independent = FeaturePerturbation(0, 123)(reference, 3)
    assert abs(correlation(independent)) < 0.015
    fully_correlated = FeaturePerturbation(1, 123)(reference, 3)
    assert abs(correlation(fully_correlated) - 2 / 3) < 0.025


def test_only_deep_levels_change_and_global_rng_is_isolated():
    reference = torch.empty(2, 32, 8, 8)
    baseline, proposed = FeaturePerturbation(0, 99), FeaturePerturbation(0.5, 99)
    saved = torch.get_rng_state().clone()
    for level in (1, 2, 3, 4):
        b0, b1 = baseline(reference, level), proposed(reference, level)
        assert torch.equal(b0, b1) == (level in (1, 2))
    assert torch.equal(saved, torch.get_rng_state())
    state = proposed.state_dict()
    expected = proposed(reference, 4)
    restored = FeaturePerturbation(0.5, 99)
    restored.load_state_dict(state)
    assert torch.equal(expected, restored(reference, 4))
    with pytest.raises(RuntimeError, match="mismatch"):
        FeaturePerturbation(0, 99).load_state_dict(state)
    with pytest.raises(RuntimeError, match="Missing"):
        restored.load_state_dict(None)


def test_zero_rho_preserves_legacy_draw_and_rng_state():
    reference = torch.empty(2, 9, 8, 8)
    torch.manual_seed(5)
    expected = torch.randn_like(reference)
    state = torch.get_rng_state().clone()
    torch.manual_seed(5)
    assert torch.equal(expected, FeaturePerturbation()(reference, 3))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("rho", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_configuration_is_rejected(rho):
    with pytest.raises(ValueError):
        FeaturePerturbation(rho)


def test_rejects_fields_that_repeat_random_samples_in_kernel():
    with pytest.raises(ValueError, match="H,W"):
        correlated_normal(torch.empty(1, 3, 2, 8), 0.5)


def test_paired_continuation_configuration_and_evaluation_protocol():
    paths = load_paths(ROOT / "configs/plan_a.yaml")
    b0, _ = configuration("B0", "screen", 100, paths)
    b1, _ = configuration("B1", "screen", 100, paths)
    assert b0["path"]["pretrain_network_g"] == b1["path"]["pretrain_network_g"]
    assert b0["plan_a"]["c0_sha256"] == b1["plan_a"]["c0_sha256"]
    assert b0["network_g"]["adapter_mode"] == b1["network_g"]["adapter_mode"] == "none"
    assert b0["network_g"]["pfa_rho"] == 0 and b1["network_g"]["pfa_rho"] == 0.5
    assert b0["train"] == b1["train"]
    assert b0["train"]["optim_g"]["lr"] == 1e-5
    assert b0["train"]["optim_g"]["weight_decay"] == 1e-4
    assert b0["train"]["scheduler"]["eta_mins"] == [1e-6]
    assert b0["train"]["total_iter"] == 5000
    dev, holdout = fixed_split("dev"), fixed_split("holdout")
    assert len(dev) == 20 and len(holdout) == 48 and not set(dev) & set(holdout)
    clean = np.full((128, 128, 3), 0.5, dtype=np.float32)
    np.random.seed(123)
    state = np.random.get_state()
    for family, level in CONDITIONS:
        first = noisy_input(clean, family, level, 0)
        assert np.array_equal(first, noisy_input(clean, family, level, 0))
        assert not np.array_equal(first, noisy_input(clean, family, level, 1))
    actual_state = np.random.get_state()
    assert state[0] == actual_state[0] and np.array_equal(state[1], actual_state[1])
    assert state[2:] == actual_state[2:]
    assert abs(noisy_input(clean, "gaussian", 15, 0).std() - 15 / 255) < 0.002


def test_aggregation_bootstraps_images_after_averaging_noise_seeds():
    rows = []
    for method, offset in (("B0", 0), ("B1", 0.1)):
        for image in ("one", "two"):
            for family, level in CONDITIONS:
                for seed in (0, 1, 2):
                    rows.append(dict(method=method, image=image, family=family, level=level,
                                     seed=seed, psnr=25 + seed + offset))
    report = summarize(rows)
    assert report["paired"]["ood"]["independent_images"] == 2
    assert abs(report["paired"]["ood"]["B1_minus_B0_db"] - 0.1) < 1e-10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual GPU required")
def test_gpu_eval_is_identical_and_state_dict_has_no_added_parameters():
    common = dict(num_blocks=[3, 4, 6, 3], depth=5, wf=64, bias=False,
                  model_path=str(ROOT / "pretrained/RN50.pt"), aug_level=0.025)
    baseline = CLIPDenoising(**common).cuda().eval()
    proposed = CLIPDenoising(**common, pfa_rho=0.5, pfa_seed=44).cuda().eval()
    proposed.load_state_dict(baseline.state_dict(), strict=True)
    assert set(baseline.state_dict()) == set(proposed.state_dict())
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in proposed.parameters())
    image = torch.randn(1, 3, 160, 160, device="cuda")
    state = torch.cuda.get_rng_state().clone()
    with torch.no_grad():
        assert torch.equal(baseline(image), proposed(image))
    assert torch.equal(state, torch.cuda.get_rng_state())
    assert proposed.pfa.state_dict()["generators"] == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual GPU required")
def test_real_gpu_training_and_full_rng_checkpoint_recovery(tmp_path):
    paths = load_paths(ROOT / "configs/plan_a.yaml")
    opt, _ = configuration("B1", "smoke", 100, paths)
    model = build_model(opt)
    dataset = Dataset_GaussianDenoising(copy.deepcopy(opt["datasets"]["train"]))
    assert len(dataset) == 432
    assert dataset.opt["sigma_range"] == 15 and dataset.opt["gt_size"] == 128
    assert model.net_g.encoder.training
    encoder_before = model.net_g.encoder.conv1.weight.detach().clone()
    decoder_before = model.net_g.last.weight.detach().clone()
    sampler = ExactSampler(len(dataset), 20100)
    seed_all(10100)
    train_update(model, dataset, sampler, 1, 4, 1)
    assert torch.equal(encoder_before, model.net_g.encoder.conv1.weight)
    assert not torch.equal(decoder_before, model.net_g.last.weight)
    assert all(p.grad is None for p in model.net_g.encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.net_g.parameters() if p.grad is not None)
    checkpoint = tmp_path / "state.pth"
    with checkpoint.open("wb") as stream:
        torch.save(checkpoint_payload(model, sampler, opt, 1, 0), stream)
    expected_loss = train_update(model, dataset, sampler, 2, 4, 1)
    expected_input = tensor_hash(model.lq)
    expected = checkpoint_payload(model, sampler, opt, 2, 0)
    restored = build_model(opt)
    restored_dataset = Dataset_GaussianDenoising(copy.deepcopy(opt["datasets"]["train"]))
    restored_sampler = ExactSampler(len(dataset), 0)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restore_checkpoint(restored, restored_sampler, saved)
    immediate = compare_states(saved, checkpoint_payload(restored, restored_sampler, opt, 1, 0), device="cpu")
    assert immediate["accepted"]
    del saved
    actual_loss = train_update(restored, restored_dataset, restored_sampler, 2, 4, 1)
    assert tensor_hash(restored.lq) == expected_input
    assert np.isclose(expected_loss, actual_loss, atol=1e-6, rtol=1e-5)
    comparison = compare_states(expected, checkpoint_payload(restored, restored_sampler, opt, 2, 0), device="cuda")
    assert comparison["accepted"], comparison["failures"]
