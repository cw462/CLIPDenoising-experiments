"""CPU-only scheduler contracts using temporary artifacts and mocked children.

No model, CUDA operation, dataset, or real training process is created. Mock
checkpoints exist only inside TemporaryDirectory and are never experiment data.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pipeline as pipeline


def queue():
    identities = [(stage, experiment, 0) for stage in ("smoke", "pilot")
                  for experiment in ("B0", "G1")]
    identities += [("full", experiment, seed) for experiment, seed in pipeline.FULL_QUEUE]
    return [{"stage": stage, "experiment": experiment, "seed": seed, "budget": 3,
             "status": "pending", "update": 0} for stage, experiment, seed in identities]


def identity(task):
    return task["stage"], task["experiment"], task["seed"]


class MockChild:
    def __init__(self, harness, task, pid, *, fail=False, ticks=3):
        self.harness, self.task, self.pid = harness, task, pid
        self.fail, self.ticks, self.polls = fail, ticks, 0
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            self.polls += 1
            if self.polls >= self.ticks:
                self.returncode = 1 if self.fail else 0
                self.harness.finish(self.task, failed=self.fail)
        return self.returncode

    def kill(self):
        raise AssertionError("Scheduler must never kill a sibling")

    terminate = kill


class WaveHarness:
    def __init__(self, jobs, *, fail=None):
        self.temporary = tempfile.TemporaryDirectory(prefix="clip_scheduler_mock_")
        self.root = Path(self.temporary.name)
        self.paths = {"jobs_per_gpu": jobs, "gpu_indices": [0, 1],
                      "output_root": str(self.root / "runs")}
        self.state = {"source_hash": "MOCK-SCHEDULER-ONLY", "queue": queue(),
                      "gpu_indices": [0, 1], "jobs_per_gpu": jobs}
        self.wave = pipeline.training_waves(self.state["queue"], jobs)[2]
        self.fail = fail
        self.launched, self.children, self.snapshots, self.validated = [], {}, [], []
        self.stack = ExitStack()

    def __enter__(self):
        replacements = {
            "make_config": self.make_config, "start_logged_child": self.start,
            "describe_process": self.describe, "matching_process_alive": self.alive,
            "validate_checkpoint": self.validate,
        }
        for name, replacement in replacements.items():
            self.stack.enter_context(patch.object(pipeline, name, replacement))
        self.stack.enter_context(patch.object(pipeline.time, "sleep", lambda _: None))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        self.temporary.cleanup()

    def run_path(self, task):
        return self.root / "runs" / f"{task['stage']}_{task['experiment']}_seed{task['seed']}"

    def make_config(self, experiment, stage, seed, paths):
        task = {"stage": stage, "experiment": experiment, "seed": seed}
        return ({"source_hash": self.state["source_hash"],
                 "config_hash": "/".join(map(str, identity(task)))}, self.run_path(task))

    def describe(self, pid):
        return {"pid": pid, "create_time": float(pid), "host": "mock-scheduler-only"}

    def install_child(self, task, *, fail=False, ticks=3):
        pid = 10000 + len(self.children)
        child = MockChild(self, task, pid, fail=fail, ticks=ticks)
        self.children[pid] = child
        run = self.run_path(task)
        run.mkdir(parents=True, exist_ok=True)
        opt, _ = self.make_config(task["experiment"], task["stage"], task["seed"], self.paths)
        (run / "effective_config.json").write_text(json.dumps(opt), encoding="utf-8")
        record = self.describe(pid)
        (run / "status.json").write_text(json.dumps({
            "status": "running", "process": record, "update": 1,
        }), encoding="utf-8")
        return child

    def start(self, command, log_path, gpu):
        experiment = command[command.index("--exp") + 1]
        stage = command[command.index("--stage") + 1]
        seed = int(command[command.index("--seed") + 1])
        task = next(task for task in self.wave
                    if identity(task) == (stage, experiment, seed))
        identifier = identity(task)
        if identifier in [entry[0] for entry in self.launched]:
            raise AssertionError(f"Duplicate child launch: {identifier}")
        self.launched.append((identifier, gpu, task.get("gpu_slot")))
        return self.install_child(task, fail=identifier == self.fail,
                                  ticks=1 if identifier == self.fail else 3)

    def alive(self, record):
        child = self.children.get(record["pid"]) if record else None
        return child is not None and child.poll() is None

    def finish(self, task, *, failed=False):
        run = self.run_path(task)
        run.mkdir(parents=True, exist_ok=True)
        if not failed:
            final = run / "checkpoints/final.pth"
            final.parent.mkdir(exist_ok=True)
            final.write_text(json.dumps({"kind": "MOCK-NO-MODEL",
                                         "update": task["budget"]}), encoding="utf-8")
        (run / "status.json").write_text(json.dumps({
            "status": "blocked" if failed else "completed",
            "update": 1 if failed else task["budget"],
            "reason": "intentional mocked child failure" if failed else None,
        }), encoding="utf-8")

    def validate(self, final, opt, budget):
        value = json.loads(Path(final).read_text(encoding="utf-8"))
        if value != {"kind": "MOCK-NO-MODEL", "update": budget}:
            raise ValueError("Invalid mocked terminal checkpoint")
        self.validated.append(str(final))

    def persist(self):
        self.snapshots.append(copy.deepcopy(self.state))

    def run(self):
        pipeline.run_training_wave(
            self.wave, self.paths, "MOCK-PATHS-NEVER-OPENED",
            self.root / "reports", self.state, self.persist, poll_seconds=0)


class CapacityContracts(unittest.TestCase):
    def test_fixed_order_complete_coverage_and_slots_for_all_capacities(self):
        for jobs in (1, 2, 4):
            with self.subTest(jobs_per_gpu=jobs):
                waves = pipeline.training_waves(queue(), jobs)
                self.assertEqual([len(wave) for wave in waves[:2]], [2, 2])
                self.assertEqual(len(waves[2:]), 8 // (2 * jobs))
                flat = [identity(task) for wave in waves for task in wave]
                self.assertEqual(len(flat), 12)
                self.assertEqual(len(set(flat)), 12)
                self.assertEqual(flat[4:], [("full", exp, seed)
                                            for exp, seed in pipeline.FULL_QUEUE])
                for wave in waves:
                    assignments = pipeline.wave_assignments(
                        wave, {"gpu_indices": [0, 1], "jobs_per_gpu": jobs})
                    pairs = [(gpu, slot) for _, gpu, slot in assignments]
                    self.assertEqual(len(pairs), len(set(pairs)))
                    self.assertEqual([gpu for gpu, _ in pairs], [0, 1] * (len(wave) // 2))
                    self.assertEqual([slot for _, slot in pairs],
                                     [index // 2 for index in range(len(wave))])

    def test_invalid_capacity_duplicate_queue_and_overflow_fail_before_launch(self):
        for jobs in (0, 3, 8, True, "4"):
            with self.subTest(invalid_jobs=jobs), self.assertRaises(RuntimeError):
                pipeline.training_waves(queue(), jobs)
        for indices in ([0, 0], [0], [-1, 1], [False, 1]):
            with self.subTest(invalid_gpus=indices), self.assertRaises(RuntimeError):
                pipeline.gpu_slots({"gpu_indices": indices, "jobs_per_gpu": 1})
        duplicate = queue()
        duplicate.append(copy.deepcopy(duplicate[0]))
        with self.assertRaises(RuntimeError):
            pipeline.training_waves(duplicate, 4)
        with WaveHarness(1) as harness:
            harness.wave = harness.state["queue"][4:]
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                harness.run()
            self.assertEqual(harness.launched, [])

    def test_every_task_launches_for_each_capacity(self):
        for jobs in (1, 2, 4):
            with self.subTest(jobs_per_gpu=jobs), WaveHarness(jobs) as harness:
                harness.run()
                self.assertEqual(len(harness.launched), 2 * jobs)
                self.assertEqual(harness.state["peak_concurrent_training_runs"], 2 * jobs)
                self.assertTrue(all(task["status"] == "completed" for task in harness.wave))
                self.assertEqual([(gpu, slot) for _, gpu, slot in harness.launched],
                                 pipeline.gpu_slots(harness.paths))

    def test_live_status_or_lock_attaches_without_duplicate_launch(self):
        with WaveHarness(4) as harness:
            for index, task in enumerate(harness.wave):
                child = harness.install_child(task, ticks=4)
                task.update(status="running", gpu_index=index % 2, gpu_slot=index // 2)
                if index % 2:
                    # Recover from the durable run lock if status has not been
                    # written yet and the controller lost its child record.
                    run = harness.run_path(task)
                    (run / "status.json").unlink()
                    (run / "process.lock").write_text(
                        json.dumps(harness.describe(child.pid)), encoding="utf-8")
            pipeline.validate_execution_policy(harness.state, harness.paths)
            harness.run()
            self.assertEqual(harness.launched, [])
            self.assertEqual(len(harness.validated), 8)
            self.assertTrue(all(task["status"] == "completed" for task in harness.wave))
            self.assertTrue(all("no duplicate" in task["recovery"] for task in harness.wave))

    def test_failure_waits_for_all_other_children_at_every_capacity(self):
        for jobs in (1, 2, 4):
            with self.subTest(jobs_per_gpu=jobs), WaveHarness(
                    jobs, fail=("full", "B0", 0)) as harness:
                with self.assertRaisesRegex(RuntimeError, "all active children reached"):
                    harness.run()
                self.assertEqual(len(harness.launched), 2 * jobs)
                self.assertEqual(harness.wave[0]["status"], "blocked")
                self.assertTrue(all(task["status"] == "completed" for task in harness.wave[1:]))
                self.assertTrue(all(child.returncode is not None
                                    for child in harness.children.values()))
                self.assertTrue(any(snapshot.get("wave_failure")
                                    and any(task["status"] == "running"
                                            for task in snapshot["queue"])
                                    for snapshot in harness.snapshots))

    def test_mixed_completed_running_pending_only_launches_pending(self):
        with WaveHarness(4) as harness:
            for task in harness.wave[:2]:
                harness.finish(task)
                task["status"] = "completed"
            for task in harness.wave[2:4]:
                harness.install_child(task, ticks=4)
                task["status"] = "running"
            harness.run()
            self.assertEqual([entry[0] for entry in harness.launched],
                             [identity(task) for task in harness.wave[4:]])
            self.assertEqual(len(harness.validated), 8)
            self.assertTrue(all(task["status"] == "completed" for task in harness.wave))
            self.assertEqual(harness.state["peak_concurrent_training_runs"], 6)

    def test_resume_rejects_changed_capacity_gpu_and_slot_before_launch(self):
        with WaveHarness(4) as harness:
            pipeline.validate_execution_policy(harness.state, harness.paths)
            for changes in ({"jobs_per_gpu": 2}, {"gpu_indices": [1, 0]}):
                changed = dict(harness.paths, **changes)
                with self.assertRaises(RuntimeError):
                    pipeline.validate_execution_policy(harness.state, changed)
            for key, wrong in (("gpu_index", 1), ("gpu_slot", 3)):
                task = harness.wave[0]
                task[key] = wrong
                with self.assertRaisesRegex(RuntimeError, "assignment changed"):
                    pipeline.validate_execution_policy(harness.state, harness.paths)
                with self.assertRaisesRegex(RuntimeError, "assignment changed"):
                    harness.run()
                del task[key]
            self.assertEqual(harness.launched, [])

    def test_visible_gpu_environment_preserves_independent_children(self):
        with patch.dict(pipeline.os.environ, {"SCHEDULER_MOCK": "retained"}, clear=False):
            for gpu in (0, 1):
                environment = pipeline.child_environment(gpu)
                self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], str(gpu))
                self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
                self.assertEqual(environment["SCHEDULER_MOCK"], "retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CapacityContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "jobs_per_gpu_verified": [1, 2, 4],
        "scope": "CPU-only mocked process scheduling; no real training or GPU work",
        "real_training_processes_started": 0,
        "failure_details": [{"test": str(test), "traceback": detail}
                            for test, detail in result.failures + result.errors],
    }
    if args.report:
        pipeline.atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
