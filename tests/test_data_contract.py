"""Real-data integration checks; these do not run or substitute formal training."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from tools.experiment_common import ROOT, load_paths, restore_rng, rng_state, seed_all
from tools.run_experiment import ExactSampler, build_dataset, make_config


@pytest.fixture(scope="module")
def paths():
    result = load_paths(ROOT / "configs/local_paths.yaml")
    result["device"] = "cpu"
    return result


@pytest.fixture(scope="module")
def smoke_opt(paths):
    return make_config("B0", "smoke", 0, paths)[0]


def split_paths(name):
    return {
        (ROOT / line).resolve()
        for line in (ROOT / "splits" / f"{name}.txt").read_text().splitlines()
        if line.strip()
    }


def test_real_original_image_split_and_training_selection(paths, smoke_opt):
    manifest = json.loads((ROOT / "data/manifest.json").read_text("utf-8"))
    original = {
        (ROOT / image["path"]).resolve()
        for image in manifest["images"] if image["dataset"] == "CBSD432"
    }
    final_test = {
        (ROOT / image["path"]).resolve()
        for image in manifest["images"] if image["dataset"] != "CBSD432"
    }
    train, val = split_paths("dev_train"), split_paths("dev_val")
    assert len(train) == 400 and len(val) == 32 and len(original) == 432
    assert not train & val
    assert train | val == original
    assert not (train | val) & final_test
    assert all(path.is_file() for path in original)
    actual = build_dataset(smoke_opt, paths)
    assert {Path(p).resolve() for p in actual.paths} == train
    full_opt = make_config("B0", "full", 0, paths)[0]
    full = build_dataset(full_opt, paths)
    assert {Path(p).resolve() for p in full.paths} == original
    assert smoke_opt["train"]["total_iter"] == 100
    assert full_opt["train"]["total_iter"] == 300000
    # The development horizon must not compress the official learning-rate schedule.
    assert smoke_opt["train"]["scheduler"]["periods"] == [92000, 208000]
    assert smoke_opt["train"]["scheduler"] == full_opt["train"]["scheduler"]


def test_real_batch_noise_reproducibility_and_unclamped_inputs(paths, smoke_opt):
    dataset = build_dataset(smoke_opt, paths)
    sampler = ExactSampler(len(dataset), 20000)
    seed_all(10000)
    saved_rng = rng_state()
    saved_sampler = copy.deepcopy(sampler.state_dict())
    indices = sampler.take(16)
    first = default_collate([dataset[i] for i in indices])
    restore_rng(saved_rng)
    sampler.load_state_dict(saved_sampler)
    second_indices = sampler.take(16)
    second = default_collate([dataset[i] for i in second_indices])
    assert second_indices == indices
    assert first["gt_path"] == second["gt_path"]
    assert torch.equal(first["gt"], second["gt"])
    assert torch.equal(first["lq"], second["lq"])
    assert tuple(first["gt"].shape) == (16, 3, 128, 128)
    assert tuple(first["lq"].shape) == (16, 3, 128, 128)
    assert first["gt"].min() >= 0 and first["gt"].max() <= 1
    assert first["lq"].min() < 0 and first["lq"].max() > 1
    noise = first["lq"] - first["gt"]
    assert abs(float(noise.mean())) < 0.0005
    assert abs(float(noise.std()) - 15 / 255) < 0.001
    assert paths["micro_batch"] * paths["accumulation"] == 16
    assert smoke_opt["runtime"]["noisy_views_per_update"] == 16
    assert smoke_opt["train"]["mixing_augs"] == {
        "mixup": True, "mixup_beta": 1.2, "use_identity": True
    }


def test_validation_uses_32_source_images_and_fixed_sigma15(paths, smoke_opt):
    opt = copy.deepcopy(smoke_opt["datasets"]["val"])
    dataset = Dataset_GaussianDenoising(opt)
    dataset.paths = [str(path) for path in sorted(split_paths("dev_val"))]
    assert len(dataset) == 32 and opt["sigma_test"] == 15
    # Validation is full-size; the original validation dataset resets its NumPy noise seed.
    seed_all(918)
    first = dataset[0]
    seed_all(326)
    second = dataset[0]
    assert torch.equal(first["gt"], second["gt"])
    assert torch.equal(first["lq"], second["lq"])
    assert first["gt"].shape[-2:] != (128, 128)
    assert Path(first["gt_path"]).resolve() not in split_paths("dev_train")


def test_sampler_recovery_across_epoch_boundary_and_global_rng_isolation():
    sampler = ExactSampler(432, 20000)
    for _ in range(26):
        assert len(sampler.take(16)) == 16
    state = copy.deepcopy(sampler.state_dict())
    expected = [sampler.take(16) for _ in range(3)]
    random.seed(777)
    np.random.seed(888)
    torch.manual_seed(999)
    restored = ExactSampler(432, 0)
    restored.load_state_dict(state)
    assert [restored.take(16) for _ in range(3)] == expected
    assert restored.epoch == 1
