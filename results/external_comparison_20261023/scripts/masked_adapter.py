"""Inference adapter for the original MaskedDenoising public checkpoint.

Required packages: torch, numpy, timm, entmax.
The unmodified upstream network keeps its stochastic attention mask in eval mode.
Pass ``mask_seed`` per image for repeatable inference without changing that method.
Inputs are RGB floating tensors (N, 3, H, W); out-of-range inputs are preserved.
Outputs are cropped to the input size and clipped to [0, 1], like the official
main_test_swinir.py. No training, input masking, x8 ensemble, or input clipping.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
from typing import Iterator

import numpy as np
import torch
from torch import nn


EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "external" / "MaskedDenoising"


def _read_commented_json(path: Path) -> dict:
    """Remove // comments outside strings, retaining original JSON values."""
    source = path.read_text(encoding="utf-8")
    result = []
    in_string = escaped = False
    index = 0
    while index < len(source):
        char = source[index]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
        elif char == '"':
            in_string = True
            result.append(char)
            index += 1
        elif source[index:index + 2] == "//":
            end = source.find("\n", index)
            index = len(source) if end == -1 else end
        else:
            result.append(char)
            index += 1
    return json.loads("".join(result))


@contextmanager
def _mask_rng(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class MaskedDenoisingInference(nn.Module):
    """Whole-image official inference with explicit optional mask RNG seed.

    ``forward(image, mask_seed=123)`` reproduces the same sampled masks whenever
    the image dimensions and seed are the same. The checkpoint's 75% attention
    masking remains enabled. Do not call concurrently from multiple threads
    within a process when using ``mask_seed`` (upstream uses global NumPy RNG).
    """

    def __init__(self, network: nn.Module, checkpoint_path: Path, config: dict):
        super().__init__()
        self.network = network
        self.checkpoint_path = str(checkpoint_path)
        self.window_size = int(config["window_size"])
        self.inference_metadata = {
            "method": "MaskedDenoising",
            "source": "https://github.com/haoyuc/MaskedDenoising",
            "training_noise": "Gaussian sigma=15",
            "use_attention_mask": bool(config["use_mask"]),
            "attention_mask_ratio": [config["mask_ratio1"], config["mask_ratio2"]],
            "stochastic_in_eval": True,
            "input_clipping": False,
            "output_clipping": [0.0, 1.0],
            "padding": "official mirror-concatenation; always pad to next window multiple",
            "ensemble": "none",
        }

    @torch.inference_mode()
    def forward(self, image: torch.Tensor, mask_seed: int | None = None) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected floating RGB tensor with shape (N, 3, H, W).")
        if not image.is_floating_point():
            raise TypeError("Normalize the common noisy input to a floating tensor first.")
        _, _, height, width = image.shape
        if min(height, width) < self.window_size:
            raise ValueError("Official padding requires dimensions at least one window (8).")
        h_pad = (height // self.window_size + 1) * self.window_size - height
        w_pad = (width // self.window_size + 1) * self.window_size - width
        padded = torch.cat([image, torch.flip(image, [2])], 2)[:, :, :height + h_pad, :]
        padded = torch.cat([padded, torch.flip(padded, [3])], 3)[:, :, :, :width + w_pad]
        with _mask_rng(mask_seed):
            output = self.network(padded)
        return output[..., :height, :width].clamp(0.0, 1.0)


def load_model(
    device: str | torch.device = "cuda",
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> MaskedDenoisingInference:
    """Load public sigma-15 weights strictly; return an eval-mode RGB denoiser."""
    checkpoint_path = Path(checkpoint_path or EXTERNAL_ROOT / "model_zoo/input_mask_80_90.pth")
    config_path = Path(config_path or EXTERNAL_ROOT / "model_zoo/input_mask_80_90.json")
    config = _read_commented_json(config_path)["netG"]
    if not config["use_mask"]:
        raise ValueError("The published MaskedDenoising configuration must keep use_mask=true.")
    source_path = EXTERNAL_ROOT / "models/network_swinir.py"
    spec = importlib.util.spec_from_file_location("external_masked_denoising_swinir", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import upstream network: {source_path}")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    network = source.SwinIR(
        upscale=config["upscale"],
        in_chans=config["in_chans"],
        img_size=config["img_size"],
        window_size=config["window_size"],
        img_range=config["img_range"],
        depths=config["depths"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        upsampler=config["upsampler"],
        resi_connection=config["resi_connection"],
        talking_heads=config["talking_heads"],
        use_attn_fn=config["attn_fn"],
        head_scale=config["head_scale"],
        on_attn=config["on_attn"],
        use_mask=config["use_mask"],
        mask_ratio1=config["mask_ratio1"],
        mask_ratio2=config["mask_ratio2"],
        mask_is_diff=config["mask_is_diff"],
        type=config["type"],
        opt=config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint.get("params", checkpoint), strict=True)
    wrapper = MaskedDenoisingInference(network, checkpoint_path, config)
    return wrapper.to(device).eval().requires_grad_(False)
