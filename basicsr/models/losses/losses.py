import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import math

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class L1StructureLoss(nn.Module):
    """Pixel L1 + Gaussian-smoothed, per-RGB Sobel Charbonnier matching.

    Only mean reduction is defined by this experiment. Kernels are buffers;
    this criterion introduces no trainable or inference-model parameters.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', structure_weight=0.05, eps=1e-3):
        super().__init__()
        if reduction != 'mean':
            raise ValueError("L1StructureLoss requires reduction='mean'")
        for name, value in (("loss_weight", loss_weight), ("structure_weight", structure_weight), ("eps", eps)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if eps == 0:
            raise ValueError('eps must be positive')
        self.loss_weight, self.structure_weight, self.eps = loss_weight, structure_weight, eps
        gaussian = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16
        sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer('gaussian', gaussian[None, None])
        self.register_buffer('sobel_x', sobel[None, None])
        self.register_buffer('sobel_y', sobel.t().contiguous()[None, None])
        self.last_components = {}

    def _filter(self, image, kernel):
        return F.conv2d(F.pad(image, (1, 1, 1, 1), mode='reflect'),
                        kernel.to(image).expand(3, 1, 3, 3), groups=3)

    def forward(self, pred, target, weight=None, **kwargs):
        if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 3 or min(pred.shape[-2:]) < 2:
            raise ValueError('Expected matching NCHW RGB tensors with H,W >= 2')
        if weight is not None:
            raise ValueError('Element-wise weights are not defined for this structure loss')
        pixel = l1_loss(pred, target, reduction='mean')
        structure = pred.new_zeros(())
        if self.structure_weight:
            smooth_pred, smooth_target = self._filter(pred, self.gaussian), self._filter(target, self.gaussian)
            for kernel in (self.sobel_x, self.sobel_y):
                delta = self._filter(smooth_pred, kernel) - self._filter(smooth_target, kernel)
                structure = structure + 0.5 * (torch.sqrt(delta.square() + self.eps ** 2) - self.eps).mean()
        weighted = self.structure_weight * structure
        self.last_components = {'pixel_l1': pixel.detach(), 'structure': structure.detach(),
                                'weighted_structure': weighted.detach()}
        return self.loss_weight * (pixel + weighted)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
