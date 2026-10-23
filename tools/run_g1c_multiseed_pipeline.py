"""Independent single-GPU G1C seeds 0/1/2, paired with archived G1 seeds.

An unresolved execution_blocked_reason prevents all GPU work. Preparing the
queue is not training completion or evidence that the CPU anomaly was fixed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tools.run_c_full_pipeline import CFullQueue
from tools.ac_gpu_policy import gpu_snapshot, audit_decision
from tools.experiment_common import (atomic_json, load_paths, process_alive, process_info,
                                     read_json, sha256_file, source_hash, utc_now)


class G1CMultiSeedQueue(CFullQueue):
    def __init__(self, config):
        self.config_a = self.config_c = str(config)
        self.a = self.c = load_paths(config)
        if (self.a.get("formal_variants"), self.a.get("formal_seeds"),
            self.a.get("formal_updates_per_run"), self.a["micro_batch"],
            self.a["accumulation"], self.a["amp"]) != (["G1C"], [0, 1, 2], 300000, 16, 1, False):
            raise ValueError("G1C requires seeds 0/1/2, 300k each, true batch16, FP32")
        if int(self.a.get("checkpoint_keep", 0)) < 2:
            raise ValueError("Retain at least two complete recovery checkpoints")
        self.output, self.reports = Path(self.a["output_root"]), Path(self.a["reports_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "pipeline_state.json"
        self.shared_resource_log = self.reports / "gpu_admission.jsonl"
        identity = {"source_hash": source_hash(), "config_sha256": sha256_file(config)}
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state["identity"] != identity:
                raise RuntimeError("Source/config changed; preserve the existing queue")
        else:
            self.state = {"identity": identity, "status": "prepared", "phase": "not_started",
                          "started_at": utc_now(), "jobs": {}, "protocol": {
                "variant": "G1C", "seeds": [0, 1, 2], "updates_per_run": 300000,
                "formal_optimizer_updates": 900000, "batch": 16, "fp32": True,
                "loss": "L1 + 0.05 * smoothed-gradient structure loss",
                "baseline": "archived G1 seed0/seed1/seed2, paired by training seed",
                "initialization": "RN50 plus original G1 initialization for each same seed",
                "fresh_training": True, "ddp": False,
                "new_inference_rows": 9450, "reused_baseline_rows": 9450,
                "historical_trajectory": "full historical per-step inputs unavailable; not claimed bitwise identical",
                "planned_gpu_assignment": {"0": 0, "1": 1, "2": 1},
                "concurrency": "one run per GPU until independent two-job throughput validation",
                "readback_recovery_policy": self.a.get("readback_recovery", "strict"),
                "readback_limitation": "Independent stable rereads can recover observation discrepancies; "
                                       "all input replay checks remain required and the root cause is unresolved.",
            }}
        self.python = sys.executable

    def prepare(self):
        reason = self.a.get("execution_blocked_reason")
        self.state.update(status="blocked" if reason else "prepared", phase="not_started")
        if reason:
            self.state["reason"] = reason
        self.record()
        return self.state

    def assert_ready(self):
        if self.a.get("execution_blocked_reason"):
            self.prepare()
            raise RuntimeError("Execution remains blocked: " + self.a["execution_blocked_reason"])

    def execute(self, keys):
        """Explicit job placement; persisted failures are never made pending."""
        self.assert_ready()
        keys = list(keys)
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate job keys in one cohort")
        while True:
            self.assert_ready()
            # Reconcile the entire cohort before starting anything else.
            for key in keys:
                job = self.state["jobs"][key]
                if job["status"] in ("failed", "interrupted"):
                    self.record()
                    raise RuntimeError("Do not retry failed task automatically: " + key)
                if job["status"] not in ("pending", "running", "completed"):
                    raise RuntimeError("Unknown persisted job status: " + key)
                if job.get("run_dir"):
                    run = Path(job["run_dir"])
                    if (run / "input_failure").exists():
                        raise RuntimeError("Preserve prior input failure: " + key)
                    if (run / "status.json").exists() and read_json(run / "status.json").get("status") in ("failed", "blocked", "interrupted"):
                        raise RuntimeError("Training already failed; do not resume automatically: " + key)
                if job["kind"] != "cpu":
                    preferred = job.get("preferred_gpu")
                    if preferred not in self.a.get("gpu_indices", [0, 1]):
                        raise RuntimeError("Missing or invalid explicit GPU placement: " + key)
                    if job["status"] == "running" and job.get("gpu") != preferred:
                        raise RuntimeError("Persisted running job has the wrong GPU placement: " + key)
                exit_file = Path(job["exit_file"])
                if exit_file.exists():
                    result = read_json(exit_file)
                    code = result.get("returncode")
                    if type(code) is not int or result.get("command") != job["command"]:
                        raise RuntimeError("Exit record belongs to another command or is incomplete: " + key)
                    job.update(status="completed" if code == 0 else "failed", returncode=code, ended_at=utc_now())
                    if code:
                        self.record()
                        raise RuntimeError("Task failed; preserve exit record without retry: " + key)
                elif job["status"] == "running" and not process_alive(job.get("process")):
                    job.update(status="interrupted", ended_at=utc_now())
                    self.record()
                    raise RuntimeError("Task interrupted; no automatic restart: " + key)
                elif job["status"] == "completed":
                    raise RuntimeError("Completed task has no durable exit record: " + key)
            if all(self.state["jobs"][key]["status"] == "completed" for key in keys):
                self.record()
                return
            for key in keys:
                job = self.state["jobs"][key]
                if job["status"] != "pending":
                    continue
                # A previously launched sibling may have failed during its
                # allocation grace period. Do not start another task after it.
                for sibling_key in keys:
                    sibling = self.state["jobs"][sibling_key]
                    exit_file = Path(sibling["exit_file"])
                    if sibling["status"] == "running" and exit_file.exists():
                        result = read_json(exit_file)
                        if result.get("returncode") != 0:
                            sibling.update(status="failed", returncode=result.get("returncode"), ended_at=utc_now())
                            self.record()
                            raise RuntimeError("Task failed before sibling admission: " + sibling_key)
                gpu = None
                if job["kind"] != "cpu":
                    choices = self.gpu_options(job["kind"], job["preferred_gpu"])
                    if not choices:
                        continue
                    gpu = choices[0][2]
                    if gpu != job["preferred_gpu"]:
                        raise RuntimeError("Admission returned a GPU outside the job's assignment")
                if shutil.disk_usage(self.output).free < float(self.a.get("minimum_free_gb", 2)) * 2**30:
                    raise RuntimeError("Insufficient free disk; preserve existing data/checkpoints")
                self.assert_ready()
                for target in (job["log"], job["exit_file"]):
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                threads = str(self.a.get("cpu_threads", 1))
                environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=threads,
                                   MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS="1",
                                   CUBLAS_WORKSPACE_CONFIG=":4096:8",
                                   CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
                wrapped = [self.python, "tools/ac_job_runner.py", "--exit-file", job["exit_file"], "--", *job["command"]]
                with Path(job["log"]).open("a", encoding="utf-8") as stream:
                    process = subprocess.Popen(wrapped, cwd=ROOT, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    child = psutil.Process(process.pid)
                    identity = {"pid": child.pid, "create_time": child.create_time(), "host": process_info()["host"]}
                except psutil.NoSuchProcess:
                    if not Path(job["exit_file"]).exists():
                        raise RuntimeError("Launched wrapper disappeared without an exit record")
                    identity = None
                job.update(status="running", gpu=gpu, started_at=utc_now(), process=identity)
                self.record()
                # Retained reservation accounting handles CUDA contexts that
                # have not allocated yet; no second job can steal this slot.
                time.sleep(3)
            self.record()
            time.sleep(10)

    def gpu_options(self, kind, preferred_gpu):
        # Only independent runs, never DDP. Do not compete with unrelated jobs.
        if kind not in ("train", "check", "eval") or preferred_gpu not in self.a.get("gpu_indices", [0, 1]):
            raise ValueError("Invalid job kind or explicit GPU assignment")
        required = 8192 if kind in ("train", "check") else 6144
        verified = self.a.get("verified_train_capacity_per_gpu", {0: 1, 1: 1})
        limit = verified.get(preferred_gpu, verified.get(str(preferred_gpu), 1)) if kind == "train" else 1
        if type(limit) is not int or limit not in (1, 2):
            raise ValueError("Only one or two independent jobs per GPU supported")
        if limit == 2 and self.a.get("concurrent_training_benchmark_required", True):
            raise RuntimeError("Two-job concurrency has not passed a throughput comparison")
        try:
            first = gpu_snapshot(self)
            time.sleep(3)
            second = gpu_snapshot(self)
            choices = []
            if preferred_gpu not in first or preferred_gpu not in second:
                audit_decision(self, {"kind": kind, "preferred_gpu": preferred_gpu, "status": "waiting", "choices": []})
                return []
            samples = (first[preferred_gpu], second[preferred_gpu])
            if all(not d["external"] and d["own_count"] < limit and
                   d["free_mib"] - d["unallocated_reservation_mib"] >= required + 2048 for d in samples):
                choices.append((max(d["own_count"] for d in samples),
                                -min(d["free_mib"] for d in samples), preferred_gpu))
            audit_decision(self, {"kind": kind, "preferred_gpu": preferred_gpu,
                                 "first": first, "second": second, "choices": choices})
            return choices
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, RuntimeError, psutil.Error) as error:
            audit_decision(self, {"kind": kind, "preferred_gpu": preferred_gpu, "status": "query_error",
                                 "choices": [], "error_type": type(error).__name__, "error": str(error)})
            return []

    def add_seed_job(self, key, command, seed, kind="train", run_dir=None):
        if type(seed) is not int or seed not in (0, 1, 2):
            raise ValueError("GPU placement only defined for seeds 0/1/2")
        result = self.add(key, command, kind, run_dir)
        preferred = 0 if seed == 0 else 1
        if self.state["jobs"][key].get("preferred_gpu", preferred) != preferred:
            raise RuntimeError("Persisted job placement changed: " + key)
        self.state["jobs"][key]["preferred_gpu"] = preferred
        return result

    def preflight(self):
        self.assert_ready()
        keys = [self.add("historical_inputs", [self.python, "tools/evaluate_c_only.py", "--paths",
            self.config_a, "--verify-inputs-only", "--audit-output", str(self.reports / "input_parity_audit.json")], "cpu")]
        for seed in (0, 1, 2):
            report = self.reports / f"actual_G1C_seed{seed}_contract.json"
            keys.append(self.add_seed_job(f"contract_seed{seed}", [self.python,
                "tools/check_c_full_contract.py", "--variant", "G1C", "--seed", str(seed),
                "--paths", self.config_a, "--report", str(report),
                "--legacy-root", "/home/imss/cw/CLIPDenoising_20261020"], seed, "check"))
        self.execute(keys)
        for seed in (0, 1, 2):
            if not read_json(self.reports / f"actual_G1C_seed{seed}_contract.json").get("accepted"):
                raise RuntimeError("Actual G1C contract failed for seed " + str(seed))

    def train(self, smoke=False):
        self.assert_ready()
        keys = []
        for seed in (0, 1, 2):
            command = [self.python, "tools/run_c_only_experiment.py", "--variant", "G1C", "--stage", "full",
                       "--seed", str(seed), "--paths", self.config_a, "--resume"]
            run = self.output / "full" / "G1C" / f"seed{seed}"
            if smoke:
                command += ["--tag", "smoke200", "--stop-after", "200"]
                run /= "smoke200"
            keys.append(self.add_seed_job(("smoke_" if smoke else "full_") + f"G1C_seed{seed}",
                                          command, seed, run_dir=run))
        self.execute(keys)

    def evaluate(self):
        self.assert_ready()
        keys = []
        for seed in (0, 1, 2):
            keys.append(self.add_seed_job(f"evaluation_G1C_seed{seed}", [self.python,
                "tools/evaluate_c_only.py", "--variant", "G1C", "--seed", str(seed), "--paths", self.config_a,
                "--output", str(self.reports / "evaluation"), "--device", "cuda"], seed, "eval"))
        self.execute(keys)
        self.execute([self.add("summarize", [self.python, "tools/summarize_g1c_multiseed.py", "--input",
            str(self.reports / "evaluation"), "--output", str(self.reports / "final")], "cpu")])

    def run(self):
        self.assert_ready()
        return super().run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/plan_g1c_multiseed_server.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    queue = G1CMultiSeedQueue(args.paths)
    if args.prepare_only:
        queue.prepare()
    else:
        queue.run()


if __name__ == "__main__":
    main()
