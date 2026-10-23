

import torch
from torch import nn

from basicsr.models.archs.CLIPEncoder_util import (ModifiedResNet,
                                                      UNetUpBlock,
                                                      UNetUpBlock_nocat,
                                                      conv3x3)
from basicsr.models.archs.rgb_calibration import RGBCalibration
from basicsr.models.archs.spatial_pfa import FeaturePerturbation

class ModifiedResNet_RemoveFinalLayer(ModifiedResNet):
   
    def __init__(self, layers, in_chn=3, width=64):
        super().__init__(layers, in_chn, width)

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
        x = self.layer4(x); 

        return out

class CLIPDenoising(nn.Module):
    def __init__(self, num_blocks, inp_channels=3, out_channels=3, depth=5, wf=64, slope=0.2,
                       bias=True, model_path=None, aug_level=0.05, adapter_mode="none",
                       pfa_rho=0.0, pfa_seed=None):

        super(CLIPDenoising, self).__init__()
        if adapter_mode not in ("none",) + RGBCalibration.MODES:
            raise ValueError("Unknown adapter_mode: {!r}".format(adapter_mode))
        if adapter_mode != "none" and (depth != 5 or inp_channels != 3):
            raise ValueError("The calibration experiment requires depth=5 and RGB input")
        self.adapter_mode = adapter_mode
        self.pfa = FeaturePerturbation(pfa_rho, pfa_seed)
        
        self.sigmas = [aug_level * i for i in range(1, 5)]
        
        self.inp_channels = inp_channels
        if inp_channels == 1: # used for 1 channel input, eg. CT images
            self.first = nn.Conv2d(inp_channels, 3, kernel_size=1, bias=bias)
            inp_channels = 3

        self.encoder = ModifiedResNet_RemoveFinalLayer(num_blocks, inp_channels, width=wf)
        self.encoder.load_pretrain_model(model_path)
            
        for params in self.encoder.parameters():
            params.requires_grad = False

        # learnable decoder
        self.up_path = nn.ModuleList()
        prev_channels = wf * 2 ** (len(num_blocks))
        for i in range(depth):
            if i == 0:
                self.up_path.append(UNetUpBlock_nocat(prev_channels, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 2:
                self.up_path.append(UNetUpBlock(prev_channels*3//2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 1: # introduce noisy image as a dense feature
                self.up_path.append(UNetUpBlock(prev_channels+inp_channels, prev_channels, slope, bias))
            else:
                self.up_path.append(UNetUpBlock(prev_channels*2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2

        self.last = conv3x3(prev_channels, out_channels, bias=bias)

        # All official layers have already consumed exactly their original RNG
        # sequence. Isolate the additional initialization so paired experiments
        # also retain identical subsequent augmentation/noise RNG states.
        if adapter_mode != "none":
            final_block = self.up_path[4]
            concatenated_channels = final_block.conv_block.block[0].in_channels
            decoder_channels = concatenated_channels - inp_channels
            with torch.random.fork_rng(devices=[]):
                final_block.rgb_calibration = RGBCalibration(
                    decoder_channels=decoder_channels, mode=adapter_mode
                )

    def load_baseline_state_dict(self, state_dict):
        """Transfer a baseline with only the newly added adapter keys absent."""
        expected = self.state_dict()
        prefix = "up_path.4.rgb_calibration."
        missing = set(expected) - set(state_dict)
        unexpected = set(state_dict) - set(expected)
        allowed = {key for key in expected if key.startswith(prefix)}
        bad_shapes = {
            key: (tuple(expected[key].shape), tuple(value.shape))
            for key, value in state_dict.items()
            if key in expected and expected[key].shape != value.shape
        }
        if missing != allowed or unexpected or bad_shapes:
            raise RuntimeError(
                "Baseline transfer mismatch: missing={!r}, unexpected={!r}, shapes={!r}; "
                "only complete calibration key absence is allowed".format(
                    sorted(missing), sorted(unexpected), bad_shapes
                )
            )
        merged = expected.copy()
        merged.update(state_dict)
        self.load_state_dict(merged, strict=True)
        return {"missing_adapter_keys": sorted(missing), "unexpected_keys": []}


    def forward(self, x):
        
        if self.inp_channels == 1:
            x = self.first(x)
            
        out = self.encoder(x)

        # progressive feature augmentation 
        if self.training:
            for idx in range(len(out)):
                if idx == 0: continue # 
                alpha = self.pfa(out[idx], idx) * self.sigmas[idx-1] + 1.0
                out[idx] = out[idx] * alpha
                
        x = out[-1]
         
        for i, up in enumerate(self.up_path):
            if i != 0: 
                x = up(x, out[-i-1])
            else:
                x = up(x)

        return self.last(x)
