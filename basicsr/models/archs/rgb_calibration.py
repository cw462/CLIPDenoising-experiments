"""Calibration of the final, cropped noisy RGB skip only."""

import torch
from torch import nn
from torch.nn import functional as F


class RGBCalibration(nn.Module):
    """Identity-initialized RGB correction guided by decoder features.

    The fixed 3 x 3 mean filter uses one-pixel reflection padding.  Its
    high-pass output is an input cue containing texture and noise, not a
    ground-truth noise estimate.  The gate modulates the correction amplitude;
    it is not a calibrated probability or confidence map.
    """

    MODES = ("gated", "ungated", "no_decoder_guidance")

    def __init__(self, decoder_channels, mode="gated"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown RGB calibration mode: {!r}".format(mode))
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive")
        self.mode = mode
        self.decoder_channels = int(decoder_channels)
        self.projection = (
            nn.Conv2d(decoder_channels, 16, kernel_size=1)
            if mode != "no_decoder_guidance" else None
        )
        input_channels = 6 + (16 if self.projection is not None else 0)
        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gate = (
            nn.Conv2d(32, 1, kernel_size=1) if mode != "ungated" else None
        )
        self.residual = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        # Enable only for a selected visualization forward, avoiding persistent
        # graph references or training-sized copies during normal execution.
        self.capture_diagnostics = False
        self.last_maps = None

    def forward(self, rgb, decoder):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Calibration requires NCHW RGB input")
        if rgb.shape[0] != decoder.shape[0] or rgb.shape[2:] != decoder.shape[2:]:
            raise ValueError("RGB and decoder feature dimensions must agree")
        mean = F.avg_pool2d(F.pad(rgb, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        high_pass = rgb - mean
        parts = [rgb, high_pass]
        if self.projection is not None:
            parts.insert(0, self.projection(decoder))
        hidden = self.trunk(torch.cat(parts, dim=1))
        residual = self.residual(hidden)
        gate = (
            torch.sigmoid(self.gate(hidden)) if self.gate is not None
            else torch.ones_like(residual[:, :1])
        )
        correction = gate * residual
        if self.capture_diagnostics:
            self.last_maps = {
                "g": gate.detach(),
                "r": residual.detach(),
                "g_times_r": correction.detach(),
            }
        else:
            self.last_maps = None
        return rgb + correction
