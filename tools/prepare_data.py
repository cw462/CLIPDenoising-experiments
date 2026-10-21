"""Fetch pinned public image datasets and OpenAI RN50, then audit and split.

Existing files are never overwritten. A verified identical file may be reused.
All image split entries are paths relative to the project root.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import time
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RN50_SHA = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
RN50_URL = f"https://openaipublic.azureedge.net/clip/models/{RN50_SHA}/RN50.pt"
SOURCES = {
    "CBSD432": {
        "repository": "drorsimon/CSCNet",
        "commit": "699d08501649979ef487c7e4b5b4a37d0a8ffdf7",
        "prefix": "datasets/CBSD432/",
        "expected_count": 432,
        "role": "development split and full formal training",
    },
    "CBSD68": {
        "repository": "cszn/FFDNet",
        "commit": "e787f46df2f374ff3591186706229cb9a0591fea",
        "prefix": "testsets/CBSD68/",
        "expected_count": 68,
        "role": "final test only",
    },
    "Kodak24": {
        "repository": "cszn/FFDNet",
        "commit": "e787f46df2f374ff3591186706229cb9a0591fea",
        "prefix": "testsets/Kodak24/",
        "expected_count": 24,
        "role": "final test only",
    },
    "McMaster": {
        "repository": "cszn/FFDNet",
        "commit": "e787f46df2f374ff3591186706229cb9a0591fea",
        "prefix": "testsets/McMaster/",
        "expected_count": 18,
        "role": "final test only",
    },
}
URBAN100_SOURCE = {
    "repository": "jbhuang0604/SelfExSR",
    "commit": "8f6dd8c1d20cb7e8792a7177b4f6fd677633f598",
    "prefix": "data/Urban100/image_SRF_2/",
    "filename_suffix": "_HR.png",
    "expected_count": 100,
    "role": "final test only; author-provided SRF2 HR version",
    "upstream_preprocessing": "SelfExSR README states images were cropped to multiples of the super-resolution factor. These are SRF2 HR files, used at their provided full resolution without further crop/resize. Pixel equivalence to CLIPDenoising authors' unspecified Urban100 copy is unverified.",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def immutable_write(path, data):
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace differing existing file: {path}")
        return
    with path.open("xb") as f:
        f.write(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def fetch(url, dest, expected_sha=None, git_blob_sha=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def verify(path):
        if expected_sha and sha256_file(path) != expected_sha:
            raise ValueError(f"SHA256 mismatch: {path}")
        if git_blob_sha:
            body = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
            if actual != git_blob_sha:
                raise ValueError(f"Git blob mismatch: {path}")

    if dest.exists():
        verify(dest)
        return dest
    temporary = dest.with_name(dest.name + f".download-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Incomplete download exists: {temporary}")
    req = urllib.request.Request(url, headers={"User-Agent": "CLIPDenoising-reproducible-experiment/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as response, temporary.open("xb") as f:
                last_progress = time.monotonic()
                total = 0
                while block := response.read(2**20):
                    f.write(block)
                    total += len(block)
                    if time.monotonic() - last_progress > 15:
                        print(f"Downloading {dest.name}: {total / 2**20:.1f} MiB", flush=True)
                        last_progress = time.monotonic()
            verify(temporary)
            # rename refuses an existing destination on Windows.
            temporary.rename(dest)
            return dest
        except Exception:
            # Only this process's newly created incomplete download is removed.
            if temporary.exists():
                temporary.unlink()
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch_tree(repo, commit):
    dest = ROOT / "data" / "_downloads" / f"{repo.replace('/', '_')}_{commit}_tree.json"
    fetch(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1", dest)
    tree = json.loads(dest.read_text("utf-8"))
    if tree.get("sha") != commit or tree.get("truncated"):
        raise ValueError(f"Invalid or truncated repository tree: {repo}")
    return tree["tree"]


def image_record(dataset, path, source_url, git_blob_sha):
    with Image.open(path) as im:
        im.load()
        original_mode = im.mode
        rgb = im.convert("RGB")
        width, height = rgb.size
        # Dimensions and representation are included to avoid shape ambiguity.
        digest = hashlib.sha256(b"RGB_UINT8\0" + struct.pack(">II", width, height) + rgb.tobytes()).hexdigest()
    return {
        "dataset": dataset,
        "image_id": path.stem,
        "path": path.relative_to(ROOT).as_posix(),
        "width": width,
        "height": height,
        "original_mode": original_mode,
        "decoded_mode": "RGB",
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "decoded_rgb_sha256": digest,
        "source_url": source_url,
        "source_git_blob_sha1": git_blob_sha,
    }


def download_dataset(dataset, source, trees, workers):
    prefix = source["prefix"]
    entries = sorted(
        (e for e in trees[(source["repository"], source["commit"])]
         if e["type"] == "blob" and e["path"].startswith(prefix)
         and e["path"].endswith(source.get("filename_suffix", ""))
         and Path(e["path"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}),
        key=lambda e: e["path"],
    )
    if len(entries) != source["expected_count"]:
        raise ValueError(f"{dataset}: expected {source['expected_count']}, got {len(entries)}")

    def process(e):
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/{e['path']}"
        path = ROOT / "data" / dataset / Path(e["path"]).name
        fetch(url, path, git_blob_sha=e["sha"])
        return image_record(dataset, path, url, e["sha"])

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for count, record in enumerate(pool.map(process, entries), 1):
            records.append(record)
            if count % 50 == 0 or count == len(entries):
                print(f"{dataset}: verified {count}/{len(entries)}", flush=True)
    return records


def duplicate_groups(records, key):
    groups = {}
    for rec in records:
        groups.setdefault(rec[key], []).append(rec["path"])
    return [paths for paths in groups.values() if len(paths) > 1]


def audit(records, sources):
    train = [r for r in records if r["dataset"] == "CBSD432"]
    test = [r for r in records if r["dataset"] != "CBSD432"]
    duplicate_files = duplicate_groups(records, "file_sha256")
    duplicate_pixels = duplicate_groups(records, "decoded_rgb_sha256")
    overlapping_ids = sorted(
        set(r["image_id"] for r in train) &
        set(r["image_id"] for r in test if r["dataset"] == "CBSD68")
    )
    split_candidates = sorted(r["path"] for r in train)
    random.Random(2026).shuffle(split_candidates)
    dev_train = sorted(split_candidates[:400])
    dev_val = sorted(split_candidates[400:])
    if len(dev_train) != 400 or len(dev_val) != 32:
        raise ValueError("Wrong development split size")
    if duplicate_files or duplicate_pixels or overlapping_ids:
        raise ValueError(json.dumps({
            "duplicate_files": duplicate_files, "duplicate_pixels": duplicate_pixels,
            "CBSD432_CBSD68_overlapping_original_ids": overlapping_ids,
        }))
    immutable_write(ROOT / "splits" / "dev_train.txt", "\n".join(dev_train) + "\n")
    immutable_write(ROOT / "splits" / "dev_val.txt", "\n".join(dev_val) + "\n")
    immutable_write(ROOT / "splits" / "full_train.txt", "\n".join(sorted(r["path"] for r in train)) + "\n")
    manifest = {
        "schema_version": 1,
        "decoded_hash_definition": "SHA256(b'RGB_UINT8\\0' + big-endian uint32(width,height) + RGB uint8 C-order bytes)",
        "sources": sources,
        "images": sorted(records, key=lambda r: r["path"]),
        "audit": {
            "same_filename_within_dataset": [],
            "duplicate_file_hash_groups": duplicate_files,
            "duplicate_decoded_rgb_hash_groups": duplicate_pixels,
            "CBSD432_CBSD68_overlapping_original_ids": overlapping_ids,
            "split_seed": 2026,
            "split_algorithm": "random.Random(2026).shuffle(sorted(project-relative image paths)); first 400 train, remaining 32 validation; output each sorted",
            "dev_train_count": 400,
            "dev_val_count": 32,
            "exact_duplicate_scope": "file bytes and decoded RGB pixels across all included datasets; does not exclude semantically similar scenes or resized/reencoded near-duplicates",
        },
        "splits": {
            name: {"path": f"splits/{name}.txt", "sha256": sha256_file(ROOT / "splits" / f"{name}.txt")}
            for name in ("dev_train", "dev_val", "full_train")
        },
    }
    immutable_write(ROOT / "data" / "manifest.json", json_bytes(manifest))
    immutable_write(ROOT / "data" / "manifest.sha256", sha256_file(ROOT / "data" / "manifest.json") + "  manifest.json\n")
    immutable_write(ROOT / "configs" / "data_sources.json", json_bytes({
        "datasets": sources,
        "pretrained": {"RN50": {"url": RN50_URL, "sha256": RN50_SHA, "path": "pretrained/RN50.pt"}},
        "CBSD432_source_provenance": "https://github.com/liutianlin0121/BSD-432-68/blob/3373c68047d0123ee79450ea929d988bf1370f0e/README.md links to the original CSCNet repository used here.",
    }))
    report = [
        "# Data preparation and audit\n",
        "All images are real public benchmark images. No synthetic data substitute was used.",
        "Downloads use immutable repository commits and Git blob verification. Original image files are preserved; training/evaluation decode RGB without resizing.\n",
        "| Dataset | Count | Role | Source |",
        "|---|---:|---|---|",
    ]
    for name, source in sources.items():
        report.append(f"| {name} | {source['expected_count']} | {source['role']} | https://github.com/{source['repository']}/tree/{source['commit']}/{source['prefix']} |")
    report += [
        "",
        f"Image manifest SHA256: `{sha256_file(ROOT / 'data' / 'manifest.json')}`.",
        "Per-file SHA256, decoded RGB SHA256, dimensions, original mode, source URL, and upstream Git blob SHA1 are recorded in `data/manifest.json`.",
        "Exact file duplicates: 0. Exact decoded RGB duplicates: 0. CBSD432/CBSD68 original image-ID overlap: 0.",
        "The audit does not claim to exclude near-duplicate scenes or differently resized/reencoded versions.",
        "",
        "Development split uses the 432 original images before patch extraction: Python `random.Random(2026)` shuffles sorted project-relative paths, assigns 400 to development training and 32 to validation, and writes sorted lists.",
        "Split hashes are in the manifest. Full training uses all 432 original images; the previous 32-image development validation set must not select a formal checkpoint.",
        "Final test datasets are excluded from model selection.",
        "",
        f"OpenAI RN50 source: {RN50_URL}. Expected SHA256: `{RN50_SHA}`. Download verification and actual availability are recorded in `pretrained/RN50.manifest.json`.",
        "",
        "Urban100 is prepared separately with `python tools/prepare_data.py --urban100-only`. Its availability, per-image hashes, and combined duplicate audit are recorded in `data/Urban100_manifest.json` when present.",
        "The Urban100 source is the original authors' SelfExSR repository at commit `8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`, `data/Urban100/image_SRF_2/*_HR.png` (100 images). The upstream README explicitly states SR-factor cropping. We use these SRF2 HR files at their supplied resolution without further crop or resize; equivalence to CLIPDenoising authors' unspecified Urban100 copy is unverified. This dataset variant must be disclosed in results.",
        "https://github.com/jbhuang0604/SelfExSR/blob/8f6dd8c1d20cb7e8792a7177b4f6fd677633f598/README.md",
        "Alternative author-hosted download attempts: Restormer Google Drive test bundle returned 'Quota exceeded'; IDF OneDrive Urban100 returned HTTP 401. No access restrictions were bypassed.",
    ]
    immutable_write(ROOT / "reports" / "data_audit.md", "\n".join(report) + "\n")
    print(f"Audit complete: {len(records)} images; manifest {sha256_file(ROOT / 'data' / 'manifest.json')}", flush=True)


def prepare_weights():
    path = fetch(RN50_URL, ROOT / "pretrained" / "RN50.pt", expected_sha=RN50_SHA)
    immutable_write(ROOT / "pretrained" / "RN50.manifest.json", json_bytes({
        "name": "OpenAI CLIP RN50",
        "source_url": RN50_URL,
        "path": "pretrained/RN50.pt",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "hash_verified": True,
        "source_registry": "https://raw.githubusercontent.com/openai/CLIP/main/clip/clip.py",
    }))
    print("RN50 downloaded and SHA256 verified", flush=True)


def prepare_urban100(workers):
    source = URBAN100_SOURCE
    key = source["repository"], source["commit"]
    trees = {key: fetch_tree(*key)}
    records = download_dataset("Urban100", source, trees, workers)
    previous = json.loads((ROOT / "data" / "manifest.json").read_text("utf-8"))
    combined = previous["images"] + records
    duplicates_file = duplicate_groups(combined, "file_sha256")
    duplicates_rgb = duplicate_groups(combined, "decoded_rgb_sha256")
    if duplicates_file or duplicates_rgb:
        raise ValueError(f"Duplicate images found: {duplicates_file}, {duplicates_rgb}")
    manifest = {
        "schema_version": 1,
        "dataset": "Urban100",
        "source": source,
        "images": records,
        "cross_dataset_audit": {
            "other_manifest": "data/manifest.json",
            "other_manifest_sha256": sha256_file(ROOT / "data" / "manifest.json"),
            "duplicate_file_hash_groups": duplicates_file,
            "duplicate_decoded_rgb_hash_groups": duplicates_rgb,
            "combined_image_count": len(combined),
        },
    }
    target = ROOT / "data" / "Urban100_manifest.json"
    immutable_write(target, json_bytes(manifest))
    immutable_write(ROOT / "data" / "Urban100_manifest.sha256", sha256_file(target) + "  Urban100_manifest.json\n")
    print(f"Urban100 audit complete: {len(records)} images; SHA256 {sha256_file(target)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--urban100-only", action="store_true")
    args = parser.parse_args()
    if args.weights_only:
        prepare_weights()
        return
    if args.urban100_only:
        prepare_urban100(args.workers)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        weights = None if args.skip_weights else pool.submit(prepare_weights)
        trees = {key: fetch_tree(*key) for key in sorted({
            (s["repository"], s["commit"]) for s in SOURCES.values()
        })}
        records = []
        for dataset, source in SOURCES.items():
            records.extend(download_dataset(dataset, source, trees, args.workers))
        audit(records, SOURCES)
        if weights:
            weights.result()


if __name__ == "__main__":
    main()
