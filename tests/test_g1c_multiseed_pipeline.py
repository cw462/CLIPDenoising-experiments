"""Mock process/GPU tests: explicit placement and fail-closed scheduling."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_g1c_multiseed_pipeline as pipeline


@pytest.fixture
def queue(tmp_path, monkeypatch):
    value = pipeline.G1CMultiSeedQueue.__new__(pipeline.G1CMultiSeedQueue)
    value.a = {"gpu_indices": [0, 1], "minimum_free_gb": 0, "cpu_threads": 1,
               "verified_train_capacity_per_gpu": {0: 1, 1: 1},
               "concurrent_training_benchmark_required": True}
    value.output, value.reports = tmp_path / "runs", tmp_path / "reports"
    value.output.mkdir()
    value.reports.mkdir()
    value.shared_resource_log = value.reports / "gpu.jsonl"
    value.state = {"jobs": {}, "status": "prepared"}
    value.python = "fixture-python"
    monkeypatch.setattr(value, "record", lambda: None)
    monkeypatch.setattr(pipeline, "process_info", lambda: {"host": "fixture"})
    monkeypatch.setattr(pipeline, "process_alive", lambda info: info is not None)
    monkeypatch.setattr(pipeline.psutil, "Process", lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 10.0))
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    return value


def add_jobs(queue, seeds=(0, 1, 2)):
    return [queue.add_seed_job(f"seed{seed}", ["python", "train", "--seed", str(seed)], seed)
            for seed in seeds]


def write_exit(job, code=0):
    path = Path(job["exit_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"returncode": code, "command": job["command"]}), encoding="utf-8")


def snapshots(queue, external0=False):
    return {index: {"index": index, "free_mib": 24000, "external": [99] if index == 0 and external0 else [],
                    "own_count": sum(j["status"] == "running" and j.get("gpu") == index for j in queue.state["jobs"].values()),
                    "unallocated_reservation_mib": 0}
            for index in (0, 1)}


def test_explicit_mapping_when_gpu0_initially_busy_and_gpu1_capacity_two(queue, monkeypatch):
    keys = add_jobs(queue)
    queue.a.update(verified_train_capacity_per_gpu={0: 1, 1: 2}, concurrent_training_benchmark_required=False)
    launches, round_number = [], [0]

    def snapshot(_):
        return snapshots(queue, external0=round_number[0] == 0)

    def launch(command, **kwargs):
        key = next(k for k, job in queue.state["jobs"].items() if job["exit_file"] == command[3])
        gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        launches.append((key, gpu))
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        return SimpleNamespace(pid=9000 + len(launches))

    def advance(seconds):
        if seconds == 10:
            round_number[0] += 1
            for job in queue.state["jobs"].values():
                if job["status"] == "running":
                    write_exit(job)

    monkeypatch.setattr(pipeline, "gpu_snapshot", snapshot)
    monkeypatch.setattr(pipeline.subprocess, "Popen", launch)
    monkeypatch.setattr(pipeline.time, "sleep", advance)
    queue.execute(keys)
    assert launches == [("seed1", 1), ("seed2", 1), ("seed0", 0)]
    assert all(job["status"] == "completed" for job in queue.state["jobs"].values())


def test_default_gpu1_capacity_one_defers_seed2_until_seed1_finished(queue, monkeypatch):
    keys = add_jobs(queue)
    launches, rounds = [], [0]
    monkeypatch.setattr(pipeline, "gpu_snapshot", lambda _: snapshots(queue))

    def launch(command, **kwargs):
        key = next(k for k, job in queue.state["jobs"].items() if job["exit_file"] == command[3])
        launches.append((key, int(kwargs["env"]["CUDA_VISIBLE_DEVICES"]), rounds[0]))
        return SimpleNamespace(pid=9100 + len(launches))

    def advance(seconds):
        if seconds == 10:
            rounds[0] += 1
            for job in queue.state["jobs"].values():
                if job["status"] == "running":
                    write_exit(job)

    monkeypatch.setattr(pipeline.subprocess, "Popen", launch)
    monkeypatch.setattr(pipeline.time, "sleep", advance)
    queue.execute(keys)
    assert launches == [("seed0", 0, 0), ("seed1", 1, 0), ("seed2", 1, 1)]


@pytest.mark.parametrize("failure", ["failed", "interrupted", "input_failure", "exit_failure"])
def test_existing_failure_cannot_restart_or_launch_siblings(queue, monkeypatch, failure):
    keys = add_jobs(queue)
    failed = queue.state["jobs"]["seed1"]
    if failure in ("failed", "interrupted"):
        failed["status"] = failure
    elif failure == "input_failure":
        run = queue.output / "seed1"
        (run / "input_failure").mkdir(parents=True)
        failed["run_dir"] = str(run)
    else:
        write_exit(failed, 2)
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: pytest.fail("Failure must block all launches"))
    with pytest.raises(RuntimeError):
        queue.execute(keys)
    if failure == "exit_failure":
        assert Path(failed["exit_file"]).exists()  # The failure record is retained.


def test_child_failure_during_allocation_grace_stops_sibling_launch(queue, monkeypatch):
    keys = add_jobs(queue)
    launches = []
    monkeypatch.setattr(pipeline, "gpu_snapshot", lambda _: snapshots(queue))

    def launch(command, **kwargs):
        key = next(k for k, job in queue.state["jobs"].items() if job["exit_file"] == command[3])
        launches.append(key)
        write_exit(queue.state["jobs"][key], 1)
        return SimpleNamespace(pid=9200)

    monkeypatch.setattr(pipeline.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="before sibling admission"):
        queue.execute(keys)
    assert launches == ["seed0"]


def test_pending_success_exit_is_consumed_without_duplicate_process(queue, monkeypatch):
    key = add_jobs(queue, (2,))[0]
    write_exit(queue.state["jobs"][key])
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: pytest.fail("Completed command relaunched"))
    queue.execute([key])
    assert queue.state["jobs"][key]["status"] == "completed"


def test_cpu_blocker_applies_before_execute_evaluate_and_run(queue, monkeypatch):
    queue.a["execution_blocked_reason"] = "Unresolved CPU readback anomaly"
    key = queue.add("cpu", ["python", "audit"], "cpu")
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: pytest.fail("Blocked queue launched a process"))
    monkeypatch.setattr(pipeline, "gpu_snapshot", lambda *a: pytest.fail("Blocked queue queried a GPU"))
    for operation in (lambda: queue.execute([key]), queue.evaluate, queue.run):
        with pytest.raises(RuntimeError, match="Execution remains blocked"):
            operation()
    assert queue.state["status"] == "blocked"


def test_gpu_query_error_never_means_free_and_two_jobs_need_validation(queue, monkeypatch):
    def unavailable(_):
        raise RuntimeError("nvidia-smi unavailable")
    monkeypatch.setattr(pipeline, "gpu_snapshot", unavailable)
    assert queue.gpu_options("train", 1) == []
    queue.a["verified_train_capacity_per_gpu"][1] = 2
    with pytest.raises(RuntimeError, match="throughput comparison"):
        queue.gpu_options("train", 1)


def test_placement_cannot_silently_change_in_persisted_state(queue):
    key = add_jobs(queue, (1,))[0]
    queue.state["jobs"][key]["preferred_gpu"] = 0
    with pytest.raises(RuntimeError, match="placement changed"):
        queue.add_seed_job(key, ["python", "train", "--seed", "1"], 1)
