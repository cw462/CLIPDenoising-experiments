"""Proposal C loss mathematics, paired configuration and real CUDA recovery."""
import copy
import logging

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from basicsr.data.paired_image_dataset import Dataset_GaussianDenoising
from basicsr.models.losses import L1Loss, L1StructureLoss
from basicsr.utils import get_root_logger
from tests.test_resume_contract import compare_states
from tools.experiment_common import ROOT, load_paths, seed_all
from tools.run_experiment import ExactSampler, build_model, checkpoint_payload, restore_checkpoint, train_update
from tools.run_pfa_experiment import tensor_hash
from tools.run_structure_experiment import configuration
from tools.evaluate_pfa import CONDITIONS, summarize

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual GPU required'))]


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(4)
    get_root_logger(log_level=logging.ERROR).setLevel(logging.ERROR)


@pytest.mark.parametrize('device', DEVICES)
def test_perfect_match_is_zero_with_zero_finite_gradient(device):
    target = torch.rand(2, 3, 16, 17, device=device)
    pred = target.clone().requires_grad_()
    criterion = L1StructureLoss().to(device)
    loss = criterion(pred, target)
    assert abs(float(loss)) < 1e-8
    loss.backward()
    assert torch.isfinite(pred.grad).all() and torch.count_nonzero(pred.grad) == 0
    assert not list(criterion.parameters())
    assert set(criterion.state_dict()) == {'gaussian', 'sobel_x', 'sobel_y'}


@pytest.mark.parametrize('device', DEVICES)
def test_lambda_zero_exactly_matches_original_l1_value_and_gradient_without_clamp(device):
    pred = torch.linspace(-0.8, 1.8, 3 * 8 * 9, device=device).reshape(1, 3, 8, 9).requires_grad_()
    target = torch.full_like(pred, 0.4)
    reference = L1Loss(loss_weight=2.3)(pred, target)
    expected_gradient = torch.autograd.grad(reference, pred)[0]
    actual = L1StructureLoss(loss_weight=2.3, structure_weight=0).to(device)(pred, target)
    actual_gradient = torch.autograd.grad(actual, pred)[0]
    assert torch.equal(reference, actual)
    assert torch.equal(expected_gradient, actual_gradient)
    assert actual_gradient[pred < 0].abs().min() > 0
    assert actual_gradient[pred > 1].abs().min() > 0


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('distortion', ['shift', 'blur'])
def test_structure_detects_displaced_or_blurred_edges(device, distortion):
    target = torch.zeros(1, 3, 24, 24, device=device)
    target[..., 12:] = 1
    if distortion == 'shift':
        pred = torch.roll(target, 2, -1)
    else:
        pred = F.avg_pool2d(F.pad(target, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
    pred.requires_grad_()
    criterion = L1StructureLoss().to(device)
    loss = criterion(pred, target)
    assert criterion.last_components['structure'] > 1e-4
    assert torch.allclose(loss, criterion.last_components['pixel_l1'] + criterion.last_components['weighted_structure'])
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_kernels_follow_the_formula_and_channels_do_not_mix():
    criterion = L1StructureLoss().double()
    assert float(criterion.gaussian.sum()) == 1
    assert float(criterion.sobel_x.sum()) == 0
    assert torch.equal(criterion.sobel_y, criterion.sobel_x.transpose(-1, -2))
    target = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    single = target.clone()
    single[:, 0, :, 8:] = 1
    criterion(single, target)
    one_channel = criterion.last_components['structure'].clone()
    all_channels = single[:, :1].repeat(1, 3, 1, 1)
    criterion(all_channels, target)
    assert torch.allclose(criterion.last_components['structure'], 3 * one_channel, atol=1e-12, rtol=0)
    # A normalized Sobel on an interior unit-slope horizontal ramp is exactly 1.
    ramp = torch.arange(16, dtype=torch.float64).view(1, 1, 1, 16).expand(1, 3, 16, 16)
    smoothed = criterion._filter(ramp, criterion.gaussian)
    gx = criterion._filter(smoothed, criterion.sobel_x)
    gy = criterion._filter(smoothed, criterion.sobel_y)
    assert torch.equal(gx[..., 3:-3, 3:-3], torch.ones_like(gx[..., 3:-3, 3:-3]))
    assert torch.count_nonzero(gy[..., 3:-3, 3:-3]) == 0


def test_full_loss_passes_double_precision_autograd_check():
    torch.manual_seed(77)
    pred = torch.rand(1, 3, 5, 6, dtype=torch.float64, requires_grad=True)
    target = torch.rand_like(pred)
    assert torch.autograd.gradcheck(L1StructureLoss().double(), (pred, target), fast_mode=True)


@pytest.mark.parametrize('kwargs', [dict(structure_weight=-0.1), dict(structure_weight=float('nan')),
                                    dict(eps=0), dict(eps=-1), dict(loss_weight=float('inf')),
                                    dict(reduction='sum')])
def test_invalid_loss_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        L1StructureLoss(**kwargs)


def test_invalid_shapes_and_undefined_pixel_weights_are_rejected():
    criterion = L1StructureLoss()
    with pytest.raises(ValueError, match='RGB'):
        criterion(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    with pytest.raises(ValueError, match='RGB'):
        criterion(torch.zeros(1, 3, 1, 8), torch.zeros(1, 3, 1, 8))
    with pytest.raises(ValueError, match='weights'):
        criterion(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8), weight=torch.ones(1, 3, 8, 8))


def test_b0_b3_use_identical_network_training_and_c0_except_pixel_loss():
    paths = load_paths(ROOT / 'configs/plan_c.yaml')
    b0, _ = configuration('B0', 'screen', 100, paths)
    b3, _ = configuration('B3', 'screen', 100, paths)
    assert b0['network_g'] == b3['network_g']
    assert b3['network_g']['pfa_rho'] == 0 and b3['network_g']['adapter_mode'] == 'none'
    assert b0['datasets'] == b3['datasets']
    assert b0['path']['pretrain_network_g'] == b3['path']['pretrain_network_g']
    assert b0['plan_c']['c0_sha256'] == b3['plan_c']['c0_sha256']
    train0, train3 = copy.deepcopy(b0['train']), copy.deepcopy(b3['train'])
    assert train0.pop('pixel_opt')['type'] == 'L1Loss'
    loss = train3.pop('pixel_opt')
    assert loss['type'] == 'L1StructureLoss' and loss['structure_weight'] == 0.05 and loss['eps'] == 0.001
    assert train0 == train3
    with pytest.raises(ValueError, match='original'):
        configuration('B3', 'screen', 100, dict(paths, rho=0.5))
    rows = [dict(method=m, image='fixture', family=f, level=l, psnr=25 + offset)
            for m, offset in [('B0', 0), ('B3', 0.2)] for f, l in CONDITIONS]
    assert abs(summarize(rows, candidate='B3')['paired']['ood']['B3_minus_B0_db'] - 0.2) < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual GPU required')
def test_actual_gpu_training_logs_paired_target_components_and_restores(tmp_path):
    paths = load_paths(ROOT / 'configs/plan_c.yaml')
    opt, _ = configuration('B3', 'smoke', 100, paths)
    model = build_model(opt)
    assert isinstance(model.cri_pix, L1StructureLoss)
    dataset = Dataset_GaussianDenoising(copy.deepcopy(opt['datasets']['train']))
    sampler = ExactSampler(len(dataset), 20100)
    seed_all(10100)
    encoder = model.net_g.encoder.conv1.weight.detach().clone()
    decoder = model.net_g.last.weight.detach().clone()
    loss = train_update(model, dataset, sampler, 1, 4, 1)
    terms = model.get_current_log()
    assert np.isclose(loss, terms['pixel_l1'] + terms['weighted_structure'], atol=1e-7)
    assert np.isclose(terms['weighted_structure'], 0.05 * terms['structure'], atol=1e-7)
    # This is the actual paired target *after* feed_train_data's Mixup, not raw GT or noisy LQ.
    assert np.isclose(terms['pixel_l1'], float(F.l1_loss(model.output, model.gt)), atol=1e-7)
    reference_criterion = L1StructureLoss().cuda()
    assert np.isclose(float(reference_criterion(model.output, model.gt)), loss, atol=1e-7)
    assert all(p.grad is None for p in model.net_g.encoder.parameters())
    assert torch.equal(encoder, model.net_g.encoder.conv1.weight)
    assert not torch.equal(decoder, model.net_g.last.weight)
    assert all(torch.isfinite(p.grad).all() for p in model.net_g.parameters() if p.grad is not None)
    with (tmp_path / 'checkpoint.pth').open('wb') as stream:
        torch.save(checkpoint_payload(model, sampler, opt, 1, 0), stream)
    expected_loss = train_update(model, dataset, sampler, 2, 4, 1)
    expected_input = tensor_hash(model.lq)
    expected = checkpoint_payload(model, sampler, opt, 2, 0)
    restored = build_model(opt)
    restored_dataset = Dataset_GaussianDenoising(copy.deepcopy(opt['datasets']['train']))
    restored_sampler = ExactSampler(len(dataset), 0)
    saved = torch.load(tmp_path / 'checkpoint.pth', map_location='cpu', weights_only=False)
    restore_checkpoint(restored, restored_sampler, saved)
    assert compare_states(saved, checkpoint_payload(restored, restored_sampler, opt, 1, 0), device='cpu')['accepted']
    del saved
    actual_loss = train_update(restored, restored_dataset, restored_sampler, 2, 4, 1)
    assert tensor_hash(restored.lq) == expected_input
    assert np.isclose(expected_loss, actual_loss, atol=1e-6, rtol=1e-5)
    comparison = compare_states(expected, checkpoint_payload(restored, restored_sampler, opt, 2, 0), device='cuda')
    assert comparison['accepted'], comparison['failures']
    # The criterion lives outside net_g; strict inference checkpoint loading stays unchanged.
    assert not any('gaussian' in k or 'sobel' in k for k in model.net_g.state_dict())
