"""Compare the actual decoded-image cache with a bundled upstream dataset class.

The independent uncached reference is copied verbatim from the upstream commit
below. This test requires neither git history nor tmp files on the remote host.
It tests data only on CPU; no training run, optimizer, or GPU state is touched.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
import basicsr.data.paired_image_dataset as actual_module
from tools.experiment_common import atomic_json, load_paths, sha256_file
from tools.run_experiment import ExactSampler

UPSTREAM_COMMIT = "2647bf815df0299a03d16df34b896fa266ceefda"
REFERENCE_CLASS_SHA256 = "4fd66d8ef3619569f1f3086d39e515ffe4fd3708aa2c04b8cfc2a1849bad1bc9"
REFERENCE_CLASS_SOURCE = r'''class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            if type(self.gt_folder) is list:
                self.paths = []
                for dataset in range(len(self.gt_folder)):
                    self.paths.extend(sorted(list(scandir(self.gt_folder[dataset], full_path=True))))
            else:
                self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]#['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)'''


def capture_rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(value):
    random.setstate(value[0])
    np.random.set_state(value[1])
    torch.set_rng_state(value[2])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_rng_equal(left, right):
    require(left[0] == right[0], "Python RNG differs")
    require(left[1][0] == right[1][0], "NumPy RNG algorithm differs")
    require(np.array_equal(left[1][1], right[1][1]), "NumPy RNG state differs")
    require(left[1][2:] == right[1][2:], "NumPy RNG counters/cache differ")
    require(torch.equal(left[2], right[2]), "CPU Torch RNG state differs")


def array_digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_reference_class():
    require(hashlib.sha256(REFERENCE_CLASS_SOURCE.encode()).hexdigest() ==
            REFERENCE_CLASS_SHA256, "Bundled uncached reference source hash changed")
    # Clone global names, not the actual class: exec replaces only this private
    # namespace's Dataset_GaussianDenoising. Shared decode/augmentation helpers
    # remain the same as the production module, isolating the cache change.
    namespace = dict(actual_module.__dict__)
    exec(compile(REFERENCE_CLASS_SOURCE, "<bundled-upstream-uncached-dataset>", "exec"),
         namespace)
    return namespace["Dataset_GaussianDenoising"]


def assert_samples_equal(expected, actual, label):
    require(expected["gt_path"] == actual["gt_path"], label + ": GT path differs")
    require(expected["lq_path"] == actual["lq_path"], label + ": LQ path differs")
    require(torch.equal(expected["gt"], actual["gt"]), label + ": GT differs")
    require(torch.equal(expected["lq"], actual["lq"]), label + ": LQ differs")
    require(tuple(actual["gt"].shape) == (3, 128, 128), label + ": wrong patch shape")


def check_split(paths, reference_class, selected_paths, label):
    cfg = yaml.safe_load(
        (ROOT / "Denoising/Options/CLIPDenoising_SyntheticDenoising_GaussianSigma15.yml").read_text()
    )
    opt = copy.deepcopy(cfg["datasets"]["train"])
    opt.update(dataroot_gt=paths["train_dir"], phase="train", scale=1)
    opt.pop("cache_decoded_images", None)
    baseline = reference_class(copy.deepcopy(opt))
    default_off = actual_module.Dataset_GaussianDenoising(copy.deepcopy(opt))
    enabled = copy.deepcopy(opt)
    enabled["cache_decoded_images"] = True
    cached = actual_module.Dataset_GaussianDenoising(copy.deepcopy(enabled))
    for dataset in (baseline, default_off, cached):
        dataset.paths = list(selected_paths)
    n = len(selected_paths)
    require(n in (400, 432), "Expected the real 400/432 original-image split")
    require(baseline.paths == cached.paths, "Cached/reference image order differs")
    require(not default_off.cache_decoded_images, "Cache must default off")

    random.seed(81923)
    np.random.seed(81923)
    torch.manual_seed(81923)
    sampler = ExactSampler(n, 20000)
    canonical_hashes = {}
    orders = []
    modified_outputs = 0
    unclamped_samples = 0
    cache_get_counts = []
    cold_resume_checks = 0
    real_imfrombytes = actual_module.imfrombytes
    start = time.perf_counter()
    # The reference namespace retains its original decoder. Only production
    # cache decoding is counted, with wraps executing the unchanged decoder.
    with patch.object(actual_module, "imfrombytes", wraps=real_imfrombytes) as decoder:
        for epoch in range(2):
            order = []
            for _ in range(n // 16):
                indices = sampler.take(16)
                require(len(indices) == 16, "Sampler did not produce batch 16")
                order.extend(indices)
                for index in indices:
                    before = capture_rng()
                    expected = baseline[index]
                    expected_rng = capture_rng()
                    # Check default-off against upstream on every image in
                    # epoch one, not merely against another cache-enabled path.
                    if epoch == 0:
                        restore_rng(before)
                        uncached_actual = default_off[index]
                        assert_samples_equal(expected, uncached_actual, f"{label}/default/{index}")
                        assert_rng_equal(expected_rng, capture_rng())
                        require(not default_off._decoded_cache, "Disabled cache stored an image")
                    calls_before = decoder.call_count
                    restore_rng(before)
                    actual = cached[index]
                    cache_get_counts.append(decoder.call_count - calls_before)
                    assert_samples_equal(expected, actual, f"{label}/epoch{epoch}/{index}")
                    assert_rng_equal(expected_rng, capture_rng())
                    if actual["lq"].min() < 0 or actual["lq"].max() > 1:
                        unclamped_samples += 1
                    path = cached.paths[index]
                    canonical = cached._decoded_cache[path]
                    require(canonical.dtype == np.float32, "Cache is not float32")
                    require(canonical.ndim == 3 and canonical.shape[-1] == 3,
                            "Cache is not full RGB HWC")
                    require(not canonical.flags.writeable, "Canonical cache is writeable")
                    if epoch == 0:
                        canonical_hashes[path] = array_digest(canonical)
                    actual["gt"].fill_(11.0)
                    actual["lq"].fill_(-7.0)
                    modified_outputs += 1
                    require(array_digest(canonical) == canonical_hashes[path],
                            "Returned tensor mutation changed canonical cache")
            require(sorted(order) == list(range(n)), "Epoch omitted or repeated an original image")
            orders.append(order)
            print(f"CACHE_CONTRACT {label} epoch {epoch + 1}: "
                  f"{n} GT/LQ and RNG states exact", flush=True)
        require(orders[0] != orders[1], "Two epoch permutations should differ")
        require(cache_get_counts[:n] == [1] * n, "Cold cache did not decode once per image")
        require(cache_get_counts[n:] == [0] * n, "Warm cache unexpectedly decoded images")
        require(len(cached._decoded_cache) == n, "Incomplete canonical cache")
        # Resume reconstructs a new, empty cache; check a full batch of cold
        # misses against the warm cache at identical saved RNG positions.
        cold_again = actual_module.Dataset_GaussianDenoising(copy.deepcopy(enabled))
        cold_again.paths = list(selected_paths)
        for index in orders[1][:16]:
            before = capture_rng()
            expected = cached[index]
            expected_rng = capture_rng()
            restore_rng(before)
            actual = cold_again[index]
            assert_samples_equal(expected, actual, f"{label}/cold_resume/{index}")
            assert_rng_equal(expected_rng, capture_rng())
            cold_resume_checks += 1
        del cold_again
    require(all(array_digest(value) == canonical_hashes[path]
                for path, value in cached._decoded_cache.items()),
            "Canonical hash changed by the end of testing")
    total_bytes = sum(array.nbytes for array in cached._decoded_cache.values())
    result = {
        "split": label, "images": n, "epochs": 2, "compared_cached_samples": 2 * n,
        "compared_default_off_samples": n, "different_sampler_orders": True,
        "gt_lq_bitwise_equal": True, "python_numpy_torch_rng_exact": True,
        "mutated_returned_tensor_pairs": modified_outputs,
        "canonical_cache_hashes_unchanged": True,
        "canonical_readonly_rgb_float32": True, "cold_decodes": n, "warm_decodes": 0,
        "default_off_empty_cache": True, "cold_resume_samples_exact": cold_resume_checks,
        "unclamped_samples": unclamped_samples, "cache_bytes": total_bytes,
        "cache_mib": total_bytes / 1024**2, "wall_seconds": time.perf_counter() - start,
        "passed": True,
    }
    del cached, default_off, baseline
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(ROOT / "configs/local_paths.yaml"))
    parser.add_argument("--report", type=Path, default=ROOT / "reports/cache_contract.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    start = time.perf_counter()
    report = {
        "passed": False, "scope": "actual production cache versus bundled upstream uncached class",
        "upstream_commit": UPSTREAM_COMMIT, "reference_class_sha256": REFERENCE_CLASS_SHA256,
        "device": "cpu data path only", "torch_version": torch.__version__,
        "splits": [], "no_training_run_modified": True, "no_tmp_or_git_dependency": True,
    }
    try:
        torch.set_num_threads(args.threads)
        paths = load_paths(args.paths)
        source = ROOT / "basicsr/data/paired_image_dataset.py"
        initial_sha = sha256_file(source)
        report["actual_dataset_source_sha256"] = initial_sha
        reference_class = build_reference_class()
        manifest = json.loads(Path(paths["data_manifest"]).read_text("utf-8"))
        formal = sorted(str(Path(paths["project_root"]) / image["path"])
                        for image in manifest["images"] if image["dataset"] == "CBSD432")
        development = [
            str(Path(paths["project_root"]) / line)
            for line in Path(paths["dev_train"]).read_text().splitlines() if line.strip()
        ]
        require(len(formal) == 432 and len(set(formal)) == 432, "Invalid formal image count")
        require(len(development) == 400 and len(set(development)) == 400, "Invalid dev count")
        require(set(development) <= set(formal), "Development image absent from formal set")
        require(all(Path(path).is_file() for path in formal), "Real dataset files are missing")
        for label, selected in (("development400", development), ("formal432", formal)):
            report["splits"].append(check_split(paths, reference_class, selected, label))
        require(sha256_file(source) == initial_sha, "Production source changed during the test")
        report.update(passed=True, default_off_verified=True, cold_resume_verified=True,
                      mutable_isolation_verified=True, rng_preserved=True,
                      production_source_unchanged_during_test=True)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr)
    report["wall_seconds"] = time.perf_counter() - start
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
