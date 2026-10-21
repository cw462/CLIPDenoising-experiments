"""Shared provenance, filesystem and process utilities for the local experiment."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODES = {"B0": "none", "G1": "gated", "G2": "ungated", "G3": "no_decoder_guidance"}
BUDGETS = {"smoke": 100, "pilot": 5000, "full": 300000}
FULL_QUEUE = [("B0", 0), ("G1", 0), ("B0", 1), ("G1", 1),
              ("B0", 2), ("G1", 2), ("G2", 0), ("G3", 0)]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def source_hash():
    files = sorted([*ROOT.glob("basicsr/**/*.py"), ROOT / "Denoising/utils.py",
                    *ROOT.glob("tools/*.py"), *ROOT.glob("tests/*.py")])
    return digest({p.relative_to(ROOT).as_posix(): sha256_file(p)
                   for p in files if p.name != "remote_control.py"})


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return (ROOT / "UPSTREAM_COMMIT").read_text().strip()


def load_paths(path):
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base = Path(config.get("project_root", ROOT))
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    config["project_root"] = str(base)
    for k in ("pretrained_rn50", "train_dir", "output_root", "reports_dir",
              "data_manifest", "dev_train", "dev_val"):
        if k in config:
            p = Path(config[k])
            config[k] = str(p if p.is_absolute() else base / p)
    config["test_datasets"] = {
        k: str(Path(v) if Path(v).is_absolute() else base / v)
        for k, v in config.get("test_datasets", {}).items()}
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def torch_load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def process_info():
    p = psutil.Process()
    return {"pid": p.pid, "create_time": p.create_time(), "host": __import__("socket").gethostname()}


def process_alive(info):
    if not info or info.get("host") != __import__("socket").gethostname():
        return False
    try:
        p = psutil.Process(info["pid"])
        return (abs(p.create_time() - info["create_time"]) < 0.1
                and p.is_running() and p.status() != psutil.STATUS_ZOMBIE)
    except (psutil.NoSuchProcess, KeyError):
        return False


class RunLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            prior = read_json(self.path)
            if process_alive(prior):
                raise RuntimeError(f"Matching process already active: {prior}")
            self.path.unlink()
        with self.path.open("x", encoding="utf-8") as f:
            json.dump(process_info(), f)
        return self

    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
