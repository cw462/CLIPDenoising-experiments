"""Independent GPU experiment slots with one durable pipeline state writer."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import psutil
from tools.experiment_common import (
    ROOT, FULL_QUEUE, BUDGETS, RunLock, atomic_json, load_paths, process_info,
    process_alive, read_json, sha256_file, source_hash, utc_now, digest)
from tools.run_experiment import make_config, validate_checkpoint


def report_progress(state, reports):
    jobs = state.get("jobs_per_gpu", 1)
    active = sum(task["status"] == "running" for task in state["queue"])
    lines = ["# CLIPDenoising execution progress", "",
             f"Updated: {utc_now()}", f"Pipeline status: **{state['status']}**", "",
             f"Source hash: `{state['source_hash']}`", "",
             "The user authorized two GPUs to run independent experiments in parallel.",
             f"Formal capacity: {jobs} independent jobs per GPU, {2 * jobs} jobs total. "
             "Smoke and pilot retain two jobs total.",
             f"Currently monitored training jobs: {active}; peak concurrently monitored: "
             f"{state.get('peak_concurrent_training_runs', 0)}.",
             "Each child sees one GPU and retains effective batch 16; no DataParallel/DDP is used.",
             "Only the pipeline controller writes this shared queue state.", "",
             "| Stage | Experiment | Seed | Physical GPU | GPU slot | State | Optimizer updates |",
             "|---|---|---:|---:|---:|---|---:|"]
    for task in state["queue"]:
        lines.append(f"| {task['stage']} | {task['experiment']} | {task['seed']} | "
                     f"{task.get('gpu_index', '')} | {task.get('gpu_slot', '')} | {task['status']} | "
                     f"{task.get('update', 0)}/{task['budget']} |")
    if state.get("reason"):
        lines += ["", "Reason: " + state["reason"]]
    lines += ["", "Only a validated final checkpoint at the prescribed update count marks training completed.",
              "Final test sets are accessed after all eight full runs are completed.",
              "Short-run validation uses only the fixed 32-image CBSD432 development split."]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = ["# CLIPDenoising experiment report", "",
               f"Execution status: **{state['status']}**", "",
               "This is a live execution report; no unrun benchmark scores are asserted.", "",
               f"Formal runs completed: {sum(t['status']=='completed' and t['stage']=='full' for t in state['queue'])}/8.",
               "Required formal budget: 2,400,000 optimizer updates, plus B0/G1 smoke and pilot.", "",
               f"User-authorized execution: two GPUs, up to {jobs} independent formal experiments "
               f"per GPU ({2 * jobs} total); smoke/pilot use one per GPU. "
               "Every experiment retains its own single-GPU model, optimizer, seed, "
               "checkpoint directory and effective batch 16.",
               f"Peak concurrently monitored training processes: "
               f"{state.get('peak_concurrent_training_runs', 0)}.", "",
               "Progress and complete failure details: progress.md and runs/pipeline_state.json.",
               "Correctness gates: pipeline_capacity.json, cache_contract.json, "
               "calibration_contract.json and resume_contract.json.",
               "Development scores, when available, are not final benchmark results.", "",
               "Mechanism hypothesis: final RGB branch calibration may improve transfer; "
               "the hypothesis remains unverified until the prescribed evaluations are complete."]
    if state.get("reason"):
        summary += ["", "Current blocker: " + state["reason"]]
    if state["status"] == "completed":
        summary += ["", "All requested full training and evaluation stages have completed.",
                    "Detailed measured results: evaluation_summary.md and the CSV tables."]
        eval_summary = reports / "evaluation_summary.md"
        if eval_summary.exists():
            summary += ["", eval_summary.read_text(encoding="utf-8")]
    (reports / "experiment_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def preflight(paths):
    gpu_slots(paths)
    if not torch.cuda.is_available() or paths["device"] != "cuda":
        raise RuntimeError("Main queue requires the authorized CUDA GPU server")
    expected = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
    if sha256_file(paths["pretrained_rn50"]) != expected:
        raise RuntimeError("Official RN50 digest mismatch")
    manifest = read_json(paths["data_manifest"])
    for item in manifest["images"]:
        p = Path(paths["project_root"]) / item["path"]
        if sha256_file(p) != item["file_sha256"]:
            raise RuntimeError(f"Dataset content changed: {p}")
    train = Path(paths["dev_train"]).read_text().splitlines()
    val = Path(paths["dev_val"]).read_text().splitlines()
    if len(train) != 400 or len(val) != 32 or set(train) & set(val):
        raise RuntimeError("Invalid source image development split")
    if shutil.disk_usage(ROOT).free < paths.get("minimum_free_gb", 2) * 1024**3:
        raise RuntimeError("Insufficient disk space for checkpointing")
    if paths["micro_batch"] * paths["accumulation"] != 16:
        raise RuntimeError("Effective batch must remain 16")
    if torch.cuda.device_count() <= max(paths.get("gpu_indices", [0, 1])):
        raise RuntimeError("Both authorized GPUs must be visible to the pipeline controller")


def child_environment(gpu_index):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def describe_process(pid):
    process = psutil.Process(pid)
    return {"pid": pid, "create_time": process.create_time(),
            "host": __import__("socket").gethostname()}


def matching_process_alive(record):
    if not process_alive(record):
        return False
    try:
        return psutil.Process(record["pid"]).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def read_status(run):
    path = Path(run) / "status.json"
    return read_json(path) if path.is_file() else {}


def existing_training_process(task, run):
    status = read_status(run)
    candidates = [status.get("process")]
    lock = Path(run) / "process.lock"
    if lock.is_file():
        candidates.append(read_json(lock))
    candidates.append(task.get("process"))
    return next((p for p in candidates if matching_process_alive(p)), None)


def start_logged_child(command, log_path, gpu_index):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as log:
        return subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=child_environment(gpu_index), start_new_session=os.name != "nt",
        )


def gpu_slots(paths, development=False):
    """Round-robin physical GPUs, then advance each GPU's independent slot."""
    jobs = paths.get("jobs_per_gpu", 1)
    if type(jobs) is not int or jobs not in (1, 2, 4):
        raise RuntimeError("jobs_per_gpu must be one of 1, 2 or 4")
    indices = paths.get("gpu_indices", [0, 1])
    if (len(indices) != 2 or any(type(gpu) is not int or gpu < 0 for gpu in indices)
            or len(set(indices)) != 2):
        raise RuntimeError("Two distinct nonnegative integer gpu_indices are required")
    return [(gpu, slot) for slot in range(1 if development else jobs) for gpu in indices]


def training_waves(queue, jobs_per_gpu=1):
    """Keep the fixed experiment order; only the formal wave capacity changes."""
    capacity = len(gpu_slots({"jobs_per_gpu": jobs_per_gpu}))
    by_id = {(t["stage"], t["experiment"], t["seed"]): t for t in queue}
    formal = [("full", experiment, seed) for experiment, seed in FULL_QUEUE]
    specifications = [
        [("smoke", "B0", 0), ("smoke", "G1", 0)],
        [("pilot", "B0", 0), ("pilot", "G1", 0)],
        *[formal[start:start + capacity] for start in range(0, len(formal), capacity)],
    ]
    if (len(by_id) != len(queue)
            or set(by_id) != {item for wave in specifications for item in wave}):
        raise RuntimeError("Queue does not match the fixed twelve development/formal runs")
    return [[by_id[item] for item in wave] for wave in specifications]


def wave_assignments(wave, paths):
    if not wave or len({task["stage"] for task in wave}) != 1:
        raise RuntimeError("A wave must contain runs from exactly one stage")
    stage = wave[0]["stage"]
    if stage not in ("smoke", "pilot", "full"):
        raise RuntimeError(f"Unknown wave stage: {stage}")
    identifiers = {(task["stage"], task["experiment"], task["seed"]) for task in wave}
    if len(identifiers) != len(wave):
        raise RuntimeError("Duplicate experiment in a training wave")
    slots = gpu_slots(paths, development=stage != "full")
    if len(wave) > len(slots):
        raise RuntimeError(f"Wave has {len(wave)} runs but capacity is only {len(slots)}")
    assignments = [(task, *slots[index]) for index, task in enumerate(wave)]
    for task, gpu, slot in assignments:
        for key, expected in (("gpu_index", gpu), ("gpu_slot", slot)):
            if key in task and task[key] != expected:
                raise RuntimeError(f"Existing {task['experiment']} seed{task['seed']} "
                                   f"{key} assignment changed: {task[key]} != {expected}")
    return assignments


def validate_execution_policy(state, paths):
    """Reject altered capacity or slot placement before any child is launched."""
    gpu_slots(paths)
    if state.get("gpu_indices", paths["gpu_indices"]) != paths["gpu_indices"]:
        raise RuntimeError("GPU assignment changed for an existing queue")
    if state.get("jobs_per_gpu", 1) != paths.get("jobs_per_gpu", 1):
        raise RuntimeError("jobs_per_gpu changed for an existing queue; use a new queue")
    for wave in training_waves(state["queue"], paths.get("jobs_per_gpu", 1)):
        wave_assignments(wave, paths)


def run_training_wave(wave, paths, paths_file, reports, state, persist, poll_seconds=5):
    """Start independent runs, or attach to their live matching processes.

    If one run fails, the sibling is allowed to reach its normal terminal
    checkpoint. Its process is never killed or duplicated. No following wave
    starts until every active child in this wave has reached a terminal state.
    """
    assignments = wave_assignments(wave, paths)
    active, failures = [], []
    for task, gpu, slot in assignments:
        try:
            opt, run = make_config(task["experiment"], task["stage"], task["seed"], paths)
            if opt["source_hash"] != state["source_hash"]:
                raise RuntimeError("Source changed during this queue; refusing a new mixed-source run")
            task.update(run_dir=str(run), config_hash=opt["config_hash"],
                        source_hash=state["source_hash"], gpu_index=gpu, gpu_slot=slot)
            # Persist the reservation before launching: a controller crash must
            # retain the exact GPU/slot to which a surviving child belongs.
            persist()
            final = run / "checkpoints/final.pth"
            process = existing_training_process(task, run)
            existing_config = run / "effective_config.json"
            if existing_config.exists() and read_json(existing_config)["config_hash"] != opt["config_hash"]:
                raise RuntimeError("Existing run config/source differs; refusing to attach or overwrite")
            if process is None and final.is_file():
                validate_checkpoint(final, opt, task["budget"])
                task.update(status="completed", update=task["budget"], checkpoint=str(final))
                task.pop("reason", None)
                persist()
                continue
            child = None
            if process is None:
                if ((run / "train.jsonl").exists()
                        and not (run / "checkpoints/latest.json").exists()):
                    # Preserve an interrupted attempt that never reached a
                    # recoverable boundary; restart the same logical run from
                    # its paired initialization, explicitly recorded.
                    output_root = Path(paths["output_root"]).resolve()
                    run.resolve().relative_to(output_root)
                    archive = output_root / "restarts" / (
                        f"{task['stage']}_{task['experiment']}_seed{task['seed']}_"
                        + str(time.time_ns()))
                    archive.resolve().relative_to(output_root)
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    run.rename(archive)
                    task.setdefault("restarts", []).append({
                        "at": utc_now(), "preserved_attempt": str(archive),
                        "reason": "No complete checkpoint; restart from paired initialization",
                        "discarded_attempt_updates": read_status(archive).get("update", 0)})
                    persist()
                command = [
                    sys.executable, "tools/run_experiment.py", "--exp", task["experiment"],
                    "--stage", task["stage"], "--seed", str(task["seed"]), "--paths", paths_file,
                ]
                if run.exists():
                    command.append("--resume")
                log = reports / f"{task['stage']}_{task['experiment']}_seed{task['seed']}.log"
                child = start_logged_child(command, log, gpu)
                process = describe_process(child.pid)
                task.update(command=command, log=str(log), recovery="new child launched")
            else:
                task["recovery"] = "monitoring matching existing process; no duplicate launch"
            task.update(status="running", process=process)
            task.pop("reason", None)
            active.append({"task": task, "opt": opt, "run": run,
                           "final": final, "process": process, "child": child})
            state["peak_concurrent_training_runs"] = max(
                state.get("peak_concurrent_training_runs", 0), len(active))
            persist()
        except Exception as error:
            task.update(status="blocked", reason=f"{type(error).__name__}: {error}")
            failures.append(f"{task['experiment']} seed{task['seed']}: {error}")
            persist()

    while active:
        for record in list(active):
            task = record["task"]
            info = read_status(record["run"])
            for key in ("update", "checkpoint", "log", "seconds", "elapsed_seconds"):
                if key in info:
                    task[key] = info[key]
            child = record["child"]
            alive = child.poll() is None if child is not None else matching_process_alive(record["process"])
            if alive:
                continue
            try:
                validate_checkpoint(record["final"], record["opt"], task["budget"])
                task.update(status="completed", update=task["budget"],
                            checkpoint=str(record["final"]), completed_at=utc_now())
                task.pop("reason", None)
            except Exception as error:
                reason = info.get("reason") or (
                    f"Process ended without a valid final checkpoint: {type(error).__name__}: {error}"
                )
                task.update(status="blocked", reason=reason)
                failures.append(f"{task['experiment']} seed{task['seed']}: {reason}")
            if child is not None:
                task["exit_code"] = child.returncode
            active.remove(record)
            persist()
        if active:
            if failures:
                state["wave_failure"] = {
                    "failures": failures,
                    "action": "monitoring remaining matching children until they finish; none killed",
                }
            persist()
            time.sleep(poll_seconds)
    state.pop("wave_failure", None)
    if failures:
        raise RuntimeError("Wave stopped after all active children reached a terminal state: " + "; ".join(failures))


# Auxiliary commands have their own durable exit marker, so a controller
# restart can monitor the same wrapper and recover its exit code.
AUXILIARY_WRAPPER = r"""
import json, os, pathlib, socket, subprocess, sys, time
import psutil
marker = pathlib.Path(sys.argv[1])
signature = sys.argv[2]
command = json.loads(sys.argv[3])
def identity(pid):
    p = psutil.Process(pid)
    return {"pid": pid, "create_time": p.create_time(), "host": socket.gethostname()}
def save(value):
    temporary = marker.with_name(marker.name + "." + str(os.getpid()) + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, marker)
record = {"signature": signature, "process": identity(os.getpid()), "status": "running"}
save(record)
child = subprocess.Popen(command)
record["child_process"] = identity(child.pid)
save(record)
code = child.wait()
record.update(status="completed" if code == 0 else "blocked", exit_code=code,
              ended_at=time.time())
save(record)
sys.exit(code)
"""


def run_pipeline(paths_file):
    paths_file = str(Path(paths_file).resolve())
    paths = load_paths(paths_file)
    paths.setdefault("gpu_indices", [0, 1])
    paths.setdefault("jobs_per_gpu", 1)
    gpu_slots(paths)
    reports, runs = Path(paths["reports_dir"]), Path(paths["output_root"])
    reports.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    state_path = runs / "pipeline_state.json"
    with RunLock(runs / "pipeline.lock"):
        h = source_hash()
        if state_path.exists():
            state = read_json(state_path)
            if state["source_hash"] != h:
                raise RuntimeError("Source changed since this queue was created; review and use a new queue")
            validate_execution_policy(state, paths)
        else:
            queue = [(s, e, 0) for s in ("smoke", "pilot") for e in ("B0", "G1")]
            queue += [("full", e, s) for e, s in FULL_QUEUE]
            state = {"source_hash": h, "status": "pending",
                     "queue": [{"stage": stage, "experiment": exp, "seed": seed,
                                "status": "pending", "budget": BUDGETS[stage], "update": 0}
                               for stage, exp, seed in queue],
                     "evaluation": "pending", "jobs_per_gpu": paths["jobs_per_gpu"]}
        state.update(status="running", process=process_info(), updated_at=utc_now())
        state.update(execution_policy="user-authorized independent two-GPU waves",
                     gpu_indices=paths["gpu_indices"], model_gpus_per_run=1,
                     effective_batch_per_run=16, jobs_per_gpu=paths["jobs_per_gpu"],
                     formal_concurrency_limit=2 * paths["jobs_per_gpu"],
                     development_concurrency_limit=2)
        state.pop("reason", None)

        def persist():
            state["updated_at"] = utc_now()
            state["active_training_count"] = sum(
                task["status"] == "running" for task in state["queue"])
            atomic_json(state_path, state)
            report_progress(state, reports)

        def execute(cmd, log_path, key, gpu_index=None):
            if gpu_index is None:
                gpu_index = paths["gpu_indices"][0]
            directory = reports / ".pipeline_commands"
            directory.mkdir(exist_ok=True)
            marker = directory / (key + ".json")
            signature = digest({"command": cmd, "gpu": gpu_index, "source_hash": h})
            prior = read_json(marker) if marker.exists() else {}
            if prior and prior.get("signature") != signature:
                raise RuntimeError(f"Auxiliary command signature changed: {key}")
            if prior.get("exit_code") == 0:
                return
            process = prior.get("process")
            auxiliary = state.setdefault("auxiliary_processes", {})
            if not matching_process_alive(process):
                process = auxiliary.get(key, {}).get("process")
            child = None
            if matching_process_alive(process):
                auxiliary[key] = dict(command=cmd, process=process, status="running",
                                      gpu_index=gpu_index, recovery="attached existing command")
            elif matching_process_alive(prior.get("child_process")):
                # The wrapper itself was externally killed while its command
                # survived. Monitor that child and refuse to create a duplicate.
                process = prior["child_process"]
                auxiliary[key] = dict(command=cmd, process=process, status="running",
                                      gpu_index=gpu_index, recovery="monitoring orphaned command")
            else:
                wrapper = [sys.executable, "-c", AUXILIARY_WRAPPER,
                           str(marker), signature, json.dumps(cmd)]
                child = start_logged_child(wrapper, log_path, gpu_index)
                process = describe_process(child.pid)
                auxiliary[key] = dict(command=cmd, process=process, status="running",
                                      gpu_index=gpu_index, log=str(log_path))
            persist()
            while (child.poll() is None if child is not None else matching_process_alive(process)):
                persist()
                time.sleep(5)
            result = read_json(marker) if marker.exists() else {}
            if result.get("signature") != signature or "exit_code" not in result:
                raise RuntimeError(f"Command ended without a recoverable exit status: {key}; see {log_path}")
            auxiliary[key].update(status="completed" if result["exit_code"] == 0 else "blocked",
                                  exit_code=result["exit_code"])
            persist()
            if result["exit_code"] != 0:
                raise RuntimeError(f"Command failed with exit {result['exit_code']}; see {log_path}")

        persist()
        try:
            preflight(paths)
            if state.get("acceptance") != "completed":
                state["phase"] = "acceptance"
                execute([sys.executable, "tests/test_pipeline_capacity.py",
                         "--report", str(reports / "pipeline_capacity.json")],
                        reports / "pipeline_capacity.log", "pipeline_capacity")
                if not read_json(reports / "pipeline_capacity.json").get("passed"):
                    raise RuntimeError("Independent process scheduling acceptance failed")
                execute([sys.executable, "tests/test_cache_contract.py",
                         "--paths", paths_file, "--threads",
                         str(paths.get("cpu_threads", 4)), "--report",
                         str(reports / "cache_contract.json")],
                        reports / "cache_contract.log", "cache_contract")
                if not read_json(reports / "cache_contract.json").get("passed"):
                    raise RuntimeError("Decoded image cache acceptance failed")
                execute([sys.executable, "tests/test_calibration_contract.py",
                         "--weights", paths["pretrained_rn50"], "--device", "cuda",
                         "--threads", str(paths.get("cpu_threads", 4)), "--report",
                         str(reports / "calibration_contract.json")],
                        reports / "calibration_contract.log", "calibration")
                execute([sys.executable, "tests/test_resume_contract.py",
                         "--paths", paths_file, "--device", "cuda",
                         "--report", str(reports / "resume_contract.json")],
                        reports / "resume_contract.log", "resume_contract")
            for name in ("pipeline_capacity", "cache_contract"):
                if not read_json(reports / f"{name}.json").get("passed"):
                    raise RuntimeError(f"Required acceptance is incomplete: {name}")
            result = read_json(reports / "calibration_contract.json")
            if not result.get("passed") or not result.get("pretrained_acceptance_complete"):
                raise RuntimeError("Real pretrained model acceptance is incomplete")
            if not read_json(reports / "resume_contract.json").get("passed"):
                raise RuntimeError("Exact checkpoint resume acceptance failed")
            state["acceptance"] = "completed"
            persist()
            for wave_number, wave in enumerate(
                    training_waves(state["queue"], paths["jobs_per_gpu"]), start=1):
                state.update(phase=wave[0]["stage"], wave=wave_number,
                             wave_capacity=len(gpu_slots(
                                 paths, development=wave[0]["stage"] != "full")),
                             wave_task_count=len(wave),
                             wave_experiments=[
                                 f"{task['experiment']}_seed{task['seed']}" for task in wave
                             ])
                persist()
                run_training_wave(wave, paths, paths_file, reports, state, persist)
                for task in wave:
                    if task["stage"] != "full" and not task.get("development_evaluation_completed"):
                        execute([sys.executable, "tools/evaluate_experiment.py", "--paths", paths_file,
                                 "--development", "--run-dir", task["run_dir"], "--visuals"],
                                reports / f"{task['stage']}_{task['experiment']}_development.log",
                                f"{task['stage']}_{task['experiment']}_development", task["gpu_index"])
                        task["development_evaluation_completed"] = True
                        persist()
            state.update(phase="final_evaluation", evaluation="running")
            persist()
            execute([sys.executable, "tools/evaluate_experiment.py", "--paths", paths_file, "--visuals"],
                    reports / "final_evaluation.log", "final_evaluation")
            execute([sys.executable, "tools/summarize_experiments.py", "--paths", paths_file],
                    reports / "summarize.log", "summarize")
            state.update(status="completed", phase="completed", evaluation="completed")
            persist()
        except BaseException as e:
            state.update(status="blocked", reason=f"{type(e).__name__}: {e}")
            (reports / "pipeline_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            persist()
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["main"], default="main")
    p.add_argument("--paths", default="configs/local_paths.yaml")
    a = p.parse_args()
    run_pipeline(a.paths)


if __name__ == "__main__":
    main()
