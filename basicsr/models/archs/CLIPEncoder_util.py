#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet


import torch
import json
from collections import OrderedDict
from pathlib import Path
from torch import nn

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool (remove attentionpooling)
    """

    def __init__(self, layers, in_chn=3, width=64):
        super().__init__()

        # the 3-layer stem
        self.conv1 = nn.Conv2d(in_chn, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []

        x = x.type(self.conv1.weight.dtype); out.append(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); out.append(x)
        x = self.avgpool(x)
        
        x = self.layer1(x); out.append(x)
        x = self.layer2(x); out.append(x)
        x = self.layer3(x); out.append(x)
        x = self.layer4(x); out.append(x)

        return out

    def load_pretrain_model(self, model_path):
        """Load every RN50 backbone tensor; never silently leave random layers.

        OpenAI checkpoints also contain text and attention-pooling tensors.
        These are reported as ignored because this backbone has neither.
        """
        if model_path is None or not Path(model_path).is_file():
            raise FileNotFoundError(
                "A real pretrained RN50 checkpoint is required: {!r}".format(model_path)
            )
        with open(model_path, 'rb') as opened_file:
            try:
                model = torch.jit.load(opened_file, map_location="cpu").eval()
                saved_state_dict = model.state_dict()
                source_format = "torchscript"
            except RuntimeError:
                opened_file.seek(0)
                saved_state_dict = torch.load(
                    opened_file, map_location="cpu", weights_only=True
                )
                source_format = "state_dict"
        if not isinstance(saved_state_dict, dict):
            raise TypeError("RN50 checkpoint must contain a tensor state dictionary")
        if "state_dict" in saved_state_dict:
            saved_state_dict = saved_state_dict["state_dict"]
        if not isinstance(saved_state_dict, dict):
            raise TypeError("Nested RN50 state_dict is not a dictionary")
        state_dict = self.state_dict()
        # Accept the complete CLIP archive or an explicitly extracted visual
        # backbone.  A mixture of prefixes is never silently combined.
        prefix = "visual." if any(k.startswith("visual.") for k in saved_state_dict) else ""
        matched, missing, shape_mismatches = {}, [], {}
        for key, expected in state_dict.items():
            saved_key = prefix + key
            if saved_key not in saved_state_dict:
                missing.append(key)
            else:
                value = saved_state_dict[saved_key]
                if not torch.is_tensor(value) or value.shape != expected.shape:
                    shape_mismatches[key] = {
                        "expected": list(expected.shape),
                        "actual": list(value.shape) if torch.is_tensor(value) else "not_tensor",
                    }
                else:
                    matched[key] = value
        parameters = dict(self.named_parameters())
        report = {
            "checkpoint": str(Path(model_path).resolve()),
            "source_format": source_format,
            "source_prefix": prefix,
            "expected_state_key_count": len(state_dict),
            "matched_state_key_count": len(matched),
            "matched_parameter_tensor_count": sum(k in parameters for k in matched),
            "matched_parameter_count": sum(parameters[k].numel() for k in matched if k in parameters),
            "matched_state_element_count": sum(v.numel() for v in matched.values()),
            "matched_keys": list(matched),
            "missing_keys": missing,
            "shape_mismatches": shape_mismatches,
            "ignored_source_keys": sorted(set(saved_state_dict) - {prefix + k for k in matched}),
        }
        self.pretrain_load_report = report
        summary = {k: v for k, v in report.items() if k not in ("matched_keys", "ignored_source_keys")}
        summary["ignored_source_key_count"] = len(report["ignored_source_keys"])
        print("RN50_LOAD_AUDIT " + json.dumps(summary, sort_keys=True), flush=True)
        if missing or shape_mismatches:
            raise RuntimeError("Incomplete RN50 pretrained weights: " + json.dumps(summary))
        # The preflight above prevents partial mutation before a failure.
        self.load_state_dict(matched, strict=True)
        return report

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2, bias=True):
        super(UNetConvBlock, self).__init__()
        block = []
            
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=bias))
        block.append(nn.LeakyReLU(slope, inplace=bias))
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=bias))
        block.append(nn.LeakyReLU(slope, inplace=bias))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2, bias=True):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=bias)
        self.conv_block = UNetConvBlock(in_size, out_size, slope, bias)
        self.rgb_calibration = None

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        if self.rgb_calibration is not None:
            crop1 = self.rgb_calibration(crop1, up)
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

class UNetUpBlock_nocat(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2, bias=True):
        super(UNetUpBlock_nocat, self).__init__()
        self.conv_block = UNetConvBlock(in_size, out_size, slope, bias)

    def forward(self, x):

        out = self.conv_block(x)

        return out
