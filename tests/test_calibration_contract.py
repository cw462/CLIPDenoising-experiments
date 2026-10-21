"""Critical structural checks; synthetic checks are never training results.

Run from the repository root:
    python tests/test_calibration_contract.py --weights pretrained/RN50.pt \
        --report reports/calibration_contract.json --threads 4

Without actual pretrained weights the full-network test is skipped and the
JSON report explicitly marks the pretrained acceptance gate incomplete.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "2647bf815df0299a03d16df34b896fa266ceefda"
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from basicsr.models.archs.CLIPEncoder_util import ModifiedResNet
from basicsr.models.archs.rgb_calibration import RGBCalibration

OBSERVATIONS = {"input_kind": "synthetic structural and gradient smoke", "measurements": {}}
WEIGHTS = Path(os.environ.get("CLIP_RN50_PATH", str(ROOT / "pretrained" / "RN50.pt")))
DEVICE = "cpu"


def grad_norm(module):
    return sum(float(p.grad.detach().float().norm()) for p in module.parameters() if p.grad is not None)


def infer_original_size(model, image):
    height, width = image.shape[-2:]
    padded = F.pad(image, (0, (-width) % 32, 0, (-height) % 32), mode="reflect")
    return model(padded)[..., :height, :width]


class CalibrationSyntheticContracts(unittest.TestCase):
    def test_all_modes_are_identity_and_report_parameter_counts(self):
        counts = {}
        for mode in RGBCalibration.MODES:
            module = RGBCalibration(19, mode)
            rgb, guidance = torch.randn(2, 3, 13, 17), torch.randn(2, 19, 13, 17)
            module.capture_diagnostics = True
            output = module(rgb, guidance)
            self.assertTrue(torch.equal(output, rgb), mode)
            self.assertEqual(module.last_maps["g"].shape, (2, 1, 13, 17))
            self.assertEqual(torch.count_nonzero(module.last_maps["r"]), 0)
            self.assertEqual(module.projection is None, mode == "no_decoder_guidance")
            self.assertEqual(module.gate is None, mode == "ungated")
            counts[mode] = sum(p.numel() for p in module.parameters())
        OBSERVATIONS["measurements"]["synthetic_module_parameters_decoder_channels_19"] = counts

    def test_gradient_reaches_hidden_layers_after_residual_update(self):
        torch.manual_seed(310)
        module = RGBCalibration(19)
        optimizer = torch.optim.AdamW(module.parameters(), lr=3e-4)
        rgb, guidance = torch.randn(2, 3, 13, 17), torch.randn(2, 19, 13, 17)
        target = torch.rand_like(rgb)
        gradients = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            F.l1_loss(module(rgb, guidance), target).backward()
            row = {name: grad_norm(getattr(module, name)) for name in ("projection", "trunk", "gate", "residual")}
            gradients.append(row)
            self.assertGreater(row["residual"], 0)
            if step == 0:
                self.assertEqual(row["projection"] + row["trunk"] + row["gate"], 0)
            else:
                self.assertTrue(all(value > 0 for value in row.values()), row)
            optimizer.step()
        OBSERVATIONS["measurements"]["synthetic_module_gradient_norms"] = gradients

    def test_pretrained_loader_rejects_missing_and_wrong_shapes(self):
        # Deliberately small random fixture for loader rejection only.
        encoder = ModifiedResNet([1, 1, 1, 1], width=4)
        fixture = {"visual." + key: value for key, value in encoder.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="clip_loader_synthetic_") as temp:
            path = Path(temp) / "synthetic_fixture.pt"
            missing = dict(fixture)
            missing.pop("visual.conv1.weight")
            torch.save(missing, path)
            with self.assertRaisesRegex(RuntimeError, "Incomplete RN50"):
                encoder.load_pretrain_model(path)
            self.assertIn("conv1.weight", encoder.pretrain_load_report["missing_keys"])
            wrong = dict(fixture)
            wrong["visual.conv1.weight"] = torch.empty(1)
            torch.save(wrong, path)
            with self.assertRaisesRegex(RuntimeError, "Incomplete RN50"):
                encoder.load_pretrain_model(path)
            self.assertIn("conv1.weight", encoder.pretrain_load_report["shape_mismatches"])


class PretrainedFullNetworkContracts(unittest.TestCase):
    def test_actual_rn50_full_network(self):
        if not WEIGHTS.is_file():
            self.skipTest("Actual pretrained RN50 missing: {}".format(WEIGHTS))
        measurements = OBSERVATIONS["measurements"]
        common = dict(
            num_blocks=[3, 4, 6, 3], inp_channels=3, out_channels=3,
            depth=5, wf=64, bias=False, model_path=str(WEIGHTS), aug_level=0.025,
        )

        def build(mode):
            torch.manual_seed(2026)
            model = CLIPDenoising(**common, adapter_mode=mode)
            return model.to(DEVICE), torch.get_rng_state().clone()

        b0, rng_b0 = build("none")
        g1, rng_g1 = build("gated")
        self.assertTrue(torch.equal(rng_b0, rng_g1))
        b0_state, g1_state = b0.state_dict(), g1.state_dict()
        for key, tensor in b0_state.items():
            self.assertTrue(torch.equal(tensor, g1_state[key]), key)
        adapter_keys = {key for key in g1_state if key.startswith("up_path.4.rgb_calibration.")}
        self.assertEqual(set(g1_state) - set(b0_state), adapter_keys)
        self.assertGreater(len(adapter_keys), 0)
        self.assertTrue(all(getattr(block, "rgb_calibration", None) is None for block in g1.up_path[:4]))
        transfer = g1.load_baseline_state_dict(b0_state)
        self.assertEqual(set(transfer["missing_adapter_keys"]), adapter_keys)
        with self.assertRaises(RuntimeError):
            g1.load_baseline_state_dict({k: v for k, v in b0_state.items() if k != "last.weight"})
        with self.assertRaises(RuntimeError):
            g1.load_baseline_state_dict(dict(b0_state, unexpected=torch.zeros(1)))

        audit = g1.encoder.pretrain_load_report
        self.assertEqual(audit["missing_keys"], [])
        self.assertEqual(audit["shape_mismatches"], {})
        self.assertEqual(audit["expected_state_key_count"], audit["matched_state_key_count"])
        measurements["pretrained_load_audit"] = audit
        measurements["paired_initialization_identical"] = True
        measurements["post_initialization_rng_identical"] = True
        measurements["parameters"] = {
            "B0": {"total": sum(p.numel() for p in b0.parameters()),
                   "trainable": sum(p.numel() for p in b0.parameters() if p.requires_grad)},
            "G1": {"total": sum(p.numel() for p in g1.parameters()),
                   "trainable": sum(p.numel() for p in g1.parameters() if p.requires_grad)},
        }
        decoder_channels = g1.up_path[4].rgb_calibration.decoder_channels
        measurements["final_decoder_channels"] = decoder_channels
        for label, mode in (("G2", "ungated"), ("G3", "no_decoder_guidance")):
            extra = RGBCalibration(decoder_channels, mode)
            parameters = sum(p.numel() for p in extra.parameters())
            measurements["parameters"][label] = {
                "total": measurements["parameters"]["B0"]["total"] + parameters,
                "trainable": measurements["parameters"]["B0"]["trainable"] + parameters,
            }
            del extra

        # Execute the source-locked original architecture as an independent
        # baseline path. Its imported utility's disabled adapter is a no-op.
        original_source = subprocess.check_output(
            ["git", "show", OFFICIAL_COMMIT + ":basicsr/models/archs/CLIPDenoising_arch.py"],
            cwd=str(ROOT), text=True,
        )
        original_namespace = {"__name__": "source_locked_original_architecture"}
        exec(compile(original_source, "<git-original-CLIPDenoising>", "exec"), original_namespace)
        torch.manual_seed(2026)
        original = original_namespace["CLIPDenoising"](**common).to(DEVICE).eval()
        self.assertEqual(set(original.state_dict()), set(b0_state))
        for key, tensor in original.state_dict().items():
            self.assertTrue(torch.equal(tensor, b0_state[key]), key)
        torch.manual_seed(515)
        image = torch.randn(1, 3, 128, 128, device=DEVICE)
        b0.eval()
        g1.eval()
        with torch.no_grad():
            original_output = original(image)
            b0_output = b0(image)
            g1_output = g1(image)
            self.assertTrue(torch.allclose(original_output, b0_output, atol=1e-6, rtol=1e-5))
            self.assertTrue(torch.allclose(b0_output, g1_output, atol=1e-6, rtol=1e-5))
            self.assertEqual(tuple(b0_output.shape), (1, 3, 128, 128))
            self.assertTrue(torch.equal(g1_output, g1(image)))
            measurements["identity_max_abs_difference_128"] = float((b0_output - g1_output).abs().max())
            measurements["original_vs_none_max_abs_difference"] = float((original_output - b0_output).abs().max())
            odd = torch.randn(1, 3, 137, 151, device=DEVICE)
            output_b0 = infer_original_size(b0, odd)
            output_g1 = infer_original_size(g1, odd)
            self.assertEqual(tuple(output_g1.shape), tuple(odd.shape))
            self.assertTrue(torch.allclose(output_b0, output_g1, atol=1e-6, rtol=1e-5))
            measurements["odd_input_shape"] = list(odd.shape)
            measurements["odd_padded_shape"] = [1, 3, 160, 160]
        del original, original_output, b0_output, g1_output, output_b0, output_g1
        gc.collect()

        # Verify actual final concat input: decoder first, original/corrected
        # RGB last; the calibration is called after center_crop.
        observed = {}
        block = g1.up_path[4]
        hooks = [
            block.up.register_forward_hook(lambda m, a, output: observed.update(up=output.detach().clone())),
            block.conv_block.block[0].register_forward_pre_hook(
                lambda m, args: observed.update(concat=args[0].detach().clone())
            ),
        ]
        with torch.no_grad():
            g1(image)
        for hook in hooks:
            hook.remove()
        self.assertTrue(torch.equal(observed["concat"][:, :decoder_channels], observed["up"]))
        self.assertTrue(torch.equal(observed["concat"][:, decoder_channels:], image))
        measurements["concat_order"] = ["upsampled_decoder", "cropped_noisy_RGB"]

        # BN remains in train mode despite frozen parameters. PFA retains its
        # original RNG and only runs in training. Shared first forwards match.
        b0.train()
        g1.train()
        self.assertTrue(b0.encoder.bn1.training and g1.encoder.bn1.training)
        encoder_before = {name: value.detach().clone() for name, value in g1.encoder.named_parameters()}
        bn_before = {name: value.detach().clone() for name, value in g1.encoder.named_buffers()}
        self.assertTrue(all(not parameter.requires_grad for parameter in g1.encoder.parameters()))
        torch.manual_seed(707)
        with torch.no_grad():
            paired_b0_output = b0(image)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in g1.parameters() if parameter.requires_grad],
            lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4,
        )
        target = torch.rand_like(image)
        gradients, seconds = [], []
        adapter = g1.up_path[4].rgb_calibration
        for step in range(3):
            torch.manual_seed(707 + step)
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            prediction = g1(image)
            if step == 0:
                self.assertTrue(torch.allclose(prediction, paired_b0_output, atol=1e-6, rtol=1e-5))
                for name, value in g1.encoder.named_buffers():
                    self.assertTrue(torch.equal(value, dict(b0.encoder.named_buffers())[name]), name)
            loss = F.l1_loss(prediction, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            row = {name: grad_norm(getattr(adapter, name)) for name in ("projection", "trunk", "gate", "residual")}
            row["decoder_last"] = grad_norm(g1.last)
            row["loss"] = float(loss.detach())
            self.assertGreater(row["decoder_last"], 0)
            self.assertGreater(row["residual"], 0)
            self.assertTrue(all(parameter.grad is None for parameter in g1.encoder.parameters()))
            if step == 0:
                self.assertEqual(row["projection"] + row["trunk"] + row["gate"], 0)
            else:
                self.assertTrue(all(row[name] > 0 for name in ("projection", "trunk", "gate")))
            torch.nn.utils.clip_grad_norm_(g1.parameters(), 0.01)
            optimizer.step()
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
            gradients.append(row)
        for name, value in g1.encoder.named_parameters():
            self.assertTrue(torch.equal(encoder_before[name], value), name)
        changed_buffers = [
            name for name, value in g1.encoder.named_buffers()
            if not torch.equal(value, bn_before[name])
        ]
        self.assertTrue(any(name.endswith("running_mean") for name in changed_buffers))
        self.assertTrue(any(name.startswith("layer4.") for name in changed_buffers))
        measurements["encoder_parameters_unchanged"] = True
        measurements["changed_encoder_bn_buffers"] = changed_buffers
        measurements["full_network_gradient_norms"] = gradients
        measurements["synthetic_batch1_update_seconds"] = seconds
        measurements["timing_scope"] = "synthetic batch=1 at 128x128; not official batch16 throughput"
        with torch.no_grad():
            torch.manual_seed(901)
            train_first = g1(image)
            torch.manual_seed(902)
            train_second = g1(image)
        self.assertFalse(torch.equal(train_first, train_second))
        measurements["pfa_train_outputs_differ"] = True
        measurements["pretrained_full_network_passed"] = True


def main():
    global WEIGHTS, DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "calibration_contract.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    WEIGHTS, DEVICE = args.weights.resolve(), args.device
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(CalibrationSyntheticContracts),
        unittest.defaultTestLoader.loadTestsFromTestCase(PretrainedFullNetworkContracts),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    OBSERVATIONS.update({
        "torch_version": torch.__version__,
        "device": DEVICE,
        "threads": args.threads,
        "pretrained_checkpoint": str(WEIGHTS),
        "tests_run": result.testsRun,
        "failures": [(str(test), trace) for test, trace in result.failures],
        "errors": [(str(test), trace) for test, trace in result.errors],
        "skipped": [(str(test), reason) for test, reason in result.skipped],
        "passed": result.wasSuccessful(),
        "pretrained_acceptance_complete": bool(
            OBSERVATIONS["measurements"].get("pretrained_full_network_passed", False)
        ),
        "wall_seconds": time.perf_counter() - start,
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(OBSERVATIONS, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
