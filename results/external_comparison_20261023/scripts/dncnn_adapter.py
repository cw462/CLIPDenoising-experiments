"""Inference adapter for cszn/DnCNN's official RGB Gaussian sigma=15 model.

The upstream color_sigma=15.mat is MATLAB v7.3, containing 20 convolutions
and 19 ReLUs. It has no separate batch-normalization layers: load its supplied
convolution weights/biases directly, without adding or folding another BN.

MatConvNet filters have logical layout H,W,I,O. MATLAB's HDF5 representation
is read by h5py as O,I,W,H, so the last two dimensions must be exchanged for
PyTorch O,I,H,W. The official demo returns input - predicted_noise.

Dependencies: numpy, h5py, torch. No BasicSR or MATLAB runtime is required.
This adapter does not add noise, change channels, clip, quantize, or crop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn


def _text(dataset: h5py.Dataset) -> str:
    return "".join(chr(int(value)) for value in dataset[()].reshape(-1))


def _scalar(group: h5py.Group, key: str) -> float:
    values = np.asarray(group[key][()]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected scalar {key}, got {values.tolist()}")
    return float(values[0])


class MatConvNetDnCNN(nn.Module):
    """Official color DnCNN sigma15 as a frozen RGB residual denoiser."""

    def __init__(self, mat_path: str | Path) -> None:
        super().__init__()
        self.mat_path = str(Path(mat_path))
        modules: list[nn.Module] = []
        layer_types: list[str] = []
        layer_shapes: list[list[int]] = []
        with h5py.File(self.mat_path, "r") as source:
            layer_refs = source["net/layers"][()].reshape(-1)
            for index, reference in enumerate(layer_refs):
                layer = source[reference]
                kind = _text(layer["type"])
                layer_types.append(kind)
                if kind == "conv":
                    if any(_scalar(layer, key) != 1 for key in ("pad", "stride", "dilate")):
                        raise ValueError(f"Unexpected geometry at layer {index}")
                    weights = layer["weights"][()].reshape(-1)
                    if len(weights) != 2:
                        raise ValueError(f"Expected filter and bias at layer {index}")
                    # h5py's axes reverse MATLAB's logical HWIO order.
                    filter_oiwh = np.asarray(source[weights[0]][()], dtype=np.float32)
                    if filter_oiwh.ndim != 4 or filter_oiwh.shape[2:] != (3, 3):
                        raise ValueError(f"Unexpected filter shape at layer {index}: {filter_oiwh.shape}")
                    filter_oihw = np.ascontiguousarray(filter_oiwh.transpose(0, 1, 3, 2))
                    bias = np.asarray(source[weights[1]][()], dtype=np.float32).reshape(-1)
                    outputs, inputs, height, width = filter_oihw.shape
                    if bias.shape != (outputs,):
                        raise ValueError(f"Unexpected bias shape at layer {index}")
                    conv = nn.Conv2d(inputs, outputs, (height, width), padding=1, bias=True)
                    with torch.no_grad():
                        conv.weight.copy_(torch.from_numpy(filter_oihw))
                        conv.bias.copy_(torch.from_numpy(np.ascontiguousarray(bias)))
                    modules.append(conv)
                    layer_shapes.append(list(filter_oihw.shape))
                elif kind == "relu":
                    if _scalar(layer, "leak") != 0:
                        raise ValueError(f"Unexpected leaky ReLU at layer {index}")
                    modules.append(nn.ReLU(inplace=False))
                else:
                    # In particular, never silently drop a batchnorm or loss.
                    raise ValueError(f"Unsupported layer {index}: {kind}")

        expected_types = [item for _ in range(19) for item in ("conv", "relu")] + ["conv"]
        expected_shapes = [[64, 3, 3, 3]] + [[64, 64, 3, 3]] * 18 + [[3, 64, 3, 3]]
        if layer_types != expected_types or layer_shapes != expected_shapes:
            raise ValueError("This file does not match the inspected official RGB sigma15 architecture")
        self.body = nn.Sequential(*modules)
        self.requires_grad_(False)
        self.eval()
        self.provenance = {
            "architecture": "DnCNN RGB, 20 convolutions, 19 ReLUs, no separate BN",
            "training_noise": "Gaussian sigma=15/255",
            "weight_file": Path(mat_path).name,
            "sha256": hashlib.sha256(Path(mat_path).read_bytes()).hexdigest(),
            "output": "input - predicted_noise; unclipped float RGB",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, noisy_rgb: torch.Tensor) -> torch.Tensor:
        if noisy_rgb.ndim != 4 or noisy_rgb.shape[1] != 3:
            raise ValueError("Expected RGB NCHW tensor with three channels")
        return noisy_rgb - self.body(noisy_rgb)


def load_dncnn(mat_path: str | Path, device: str | torch.device = "cpu") -> MatConvNetDnCNN:
    """Load the supplied official MAT weight and return an eval-mode model."""
    return MatConvNetDnCNN(mat_path).to(device).eval()


def conversion_self_test(model: MatConvNetDnCNN) -> dict:
    """Check filter orientation against an independent logical-HWIO reference.

    The first-convolution comparison catches the otherwise silent spatial-axis
    transposition introduced by MATLAB v7.3/HDF5 storage. It also checks bias
    and zero padding. A complete forward then checks shape and finite output.
    """
    rng = np.random.default_rng(20261023)
    image = rng.random((11, 13, 3), dtype=np.float32)
    with h5py.File(model.mat_path, "r") as source:
        first = source[source["net/layers"][()].reshape(-1)[0]]
        weights = first["weights"][()].reshape(-1)
        # Restore MATLAB's original logical dimensions, independently of loader.
        matlab_hwio = np.asarray(source[weights[0]][()], dtype=np.float64).transpose(3, 2, 1, 0)
        bias = np.asarray(source[weights[1]][()], dtype=np.float64).reshape(-1)
    padded = np.pad(image.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))
    reference_hwo = np.einsum("hwcij,ijco->hwo", windows, matlab_hwio) + bias
    device = next(model.parameters()).device
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        first_actual = model.body[0](tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
        denoised = model(tensor)
    difference = float(np.max(np.abs(first_actual - reference_hwo)))
    if difference > 2e-5:
        raise AssertionError(f"MATLAB/PyTorch filter conversion mismatch: {difference}")
    if denoised.shape != tensor.shape or not bool(torch.isfinite(denoised).all()):
        raise AssertionError("Complete DnCNN forward returned invalid shape or nonfinite values")
    return {
        "first_convolution_max_abs_error_vs_numpy_hwio": difference,
        "full_forward_shape": list(denoised.shape),
        "full_forward_finite": True,
        "device": str(device),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--export", type=Path, help="Optional converted plain PyTorch state dict")
    args = parser.parse_args()
    model = load_dncnn(args.weights, args.device)
    output = dict(model.provenance)
    if args.self_test:
        output["conversion_self_test"] = conversion_self_test(model)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.cpu() for key, value in model.state_dict().items()}, args.export)
        output["converted_state_dict"] = str(args.export)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
