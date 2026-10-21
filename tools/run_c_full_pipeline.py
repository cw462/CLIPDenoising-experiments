"""Run two 300k structure-loss experiments, reusing archived B0/G1 controls."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_ac_pipeline import Queue as JobQueue
from tools.ac_gpu_policy import shared_gpu_options
from tools.experiment_common import (
    RunLock, atomic_json, load_paths, process_info, read_json, sha256_file,
    source_hash, utc_now,
)


class CFullQueue(JobQueue):
    def __init__(self, config):
        self.config_a = self.config_c = str(config)
        self.a = self.c = load_paths(config)
        if (self.a["micro_batch"], self.a["accumulation"], self.a["amp"]) != (16, 1, False):
            raise ValueError("Full historical comparison requires batch16 and FP32")
        if (self.a.get("formal_variants"), self.a.get("formal_seeds"),
                self.a.get("formal_updates_per_run")) != (["C", "G1C"], [0], 300000):
            raise ValueError("This queue is fixed to C/G1C, seed0, 300000 updates each")
        self.output, self.reports = Path(self.a["output_root"]), Path(self.a["reports_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "pipeline_state.json"
        self.shared_resource_log = self.reports / "gpu_admission.jsonl"
        identity = {"source_hash": source_hash(), "config_sha256": sha256_file(config)}
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state["identity"] != identity:
                raise RuntimeError("Source/config changed; preserve this run and use a new root")
        else:
            self.state = {
                "identity": identity, "status": "created", "phase": "preflight",
                "started_at": utc_now(), "jobs": {}, "protocol": {
                    "variants": {"C": "B0 architecture + structure loss",
                                 "G1C": "G1 gated architecture + structure loss"},
                    "seeds": [0], "updates_per_run": 300000,
                    "formal_optimizer_updates": 600000, "batch": 16, "fp32": True,
                    "initialization": "RN50 pretrained encoder; original seed0 decoder initialization",
                    "not_continuation": True, "loss": "L1 + 0.05 * smoothed-gradient structure loss",
                    "baseline": "reuse archived B0/G1 full seed0; no additional baseline training",
                    "inference_rows": 6300, "reused_baseline_rows": 6300,
                    "claim_boundary": "one training seed; no multi-seed robustness claim",
                    "known_runtime_issue": "prior CPU copy/read anomaly unresolved; single CPU thread and fail-closed repeated input checks",
                },
            }
        self.python = sys.executable

    def add(self, key, command, kind="train", run_dir=None):
        threads = str(self.a["cpu_threads"])
        command = ["env", "OMP_NUM_THREADS=" + threads, "MKL_NUM_THREADS=" + threads,
                   "OPENBLAS_NUM_THREADS=1", "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command]
        return super().add(key, command, kind, run_dir)

    def gpu_options(self, kind):
        return shared_gpu_options(self, kind)

    def record(self):
        self.state["updated_at"] = utc_now()
        lines = ["# C 与门控+C 完整300k实验进度", "",
                 f"状态：{self.state['status']}；阶段：{self.state['phase']}",
                 f"更新时间（UTC）：{self.state['updated_at']}", "",
                 "| 任务 | 状态 | GPU | 完成步数 |", "|---|---|---:|---:|"]
        for key, job in self.state["jobs"].items():
            progress = ""
            if job.get("run_dir"):
                path = Path(job["run_dir"]) / "status.json"
                if path.exists():
                    status = read_json(path)
                    job["update"] = status.get("update", 0)
                    progress = f"{job['update']}/{status.get('budget', '')}"
            lines.append(f"| {key} | {job['status']} | {job.get('gpu', '')} | {progress} |")
        if self.state.get("reason"):
            lines += ["", self.state["reason"]]
        atomic_json(self.state_path, self.state)
        (self.reports / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def preflight(self):
        tests = [self.python, "-m", "pytest", "-q", "tests/test_structure_loss.py",
                 "tests/test_spatial_pfa.py", "tests/test_c_only_contract.py",
                 "tests/test_c_only_evaluation.py"]
        audit = self.reports / "input_parity_audit.json"
        keys = [self.add("unit_contracts", tests, kind="cpu"),
                self.add("historical_inputs", [self.python, "tools/evaluate_c_only.py", "--paths",
                         self.config_a, "--verify-inputs-only", "--audit-output", str(audit)], kind="cpu")]
        for variant in ("C", "G1C"):
            report = self.reports / f"actual_{variant}_contract.json"
            keys.append(self.add(f"actual_{variant}_contract", [self.python, "tools/check_c_full_contract.py",
                                  "--variant", variant, "--paths", self.config_a, "--report", str(report),
                                  "--legacy-root", "/home/imss/cw/CLIPDenoising_20261020"], kind="check"))
        self.execute(keys)
        for variant in ("C", "G1C"):
            if not read_json(self.reports / f"actual_{variant}_contract.json").get("accepted"):
                raise RuntimeError(f"Actual {variant} initialization/recovery check did not pass")

    def train(self, smoke=False):
        keys = []
        for variant in ("C", "G1C"):
            command = [self.python, "tools/run_c_only_experiment.py", "--variant", variant,
                       "--stage", "full", "--seed", "0", "--paths", self.config_a, "--resume"]
            run = self.output / "full" / variant / "seed0"
            if smoke:
                command += ["--tag", "smoke200", "--stop-after", str(self.a["smoke_updates"])]
                run = run / "smoke200"
            keys.append(self.add(("smoke_" if smoke else "full_") + variant, command, run_dir=run))
        self.execute(keys)

    def evaluate(self):
        keys = []
        for variant in ("C", "G1C"):
            keys.append(self.add("evaluation_" + variant,
                [self.python, "tools/evaluate_c_only.py", "--variant", variant, "--paths", self.config_a,
                 "--output", str(self.reports / "evaluation"), "--device", "cuda"], kind="eval"))
        self.execute(keys)
        self.execute([self.add("summarize", [self.python, "tools/summarize_c_only.py", "--input",
                     str(self.reports / "evaluation"), "--output", str(self.reports / "final")], kind="cpu")])

    def run(self):
        with RunLock(self.output / "pipeline.lock"):
            if self.state["status"] == "completed":
                print("C/G1C 300k queue already completed", flush=True)
                return
            self.state.update(status="running", process=process_info())
            self.state.pop("reason", None)
            try:
                for phase, action in (("preflight", self.preflight),
                                      ("smoke200", lambda: self.train(smoke=True)),
                                      ("full300k", self.train), ("evaluation", self.evaluate)):
                    self.state["phase"] = phase
                    self.record()
                    action()
                self.state.update(status="completed", phase="completed", ended_at=utc_now())
                self.record()
            except BaseException as error:
                self.state.update(status="failed", reason=f"{type(error).__name__}: {error}",
                                  ended_at=utc_now())
                self.record()
                (self.reports / "pipeline_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/plan_c_only_server.yaml")
    args = parser.parse_args()
    CFullQueue(args.paths).run()


if __name__ == "__main__":
    main()
