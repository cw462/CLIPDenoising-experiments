"""Recover A/C with exact shared batches and the historical full benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ac_gpu_policy import POLICY, shared_gpu_options
from tools.ac_triplet_gate import validate_triplet_stage
from tools.experiment_common import atomic_json, read_json, sha256_file, source_hash
from tools.run_ac_pipeline import Queue


class Table1Queue(Queue):
    def __init__(self, config_a, config_c):
        super().__init__(config_a, config_c)
        if not self.a.get("require_triplet") or not self.c.get("require_triplet"):
            raise ValueError("Recovery requires exact triplet training")
        historical = self.a["historical_results_csv"]
        if historical != self.c.get("historical_results_csv"):
            raise ValueError("A/C historical benchmark inputs differ")
        if sha256_file(historical) != self.a["historical_results_csv_sha256"]:
            raise ValueError("Historical benchmark CSV changed")
        self.shared_resource_log = self.reports / "resource_decisions.jsonl"
        self.state["protocol"].update({
            "execution": "ac_triplet_v2; exact input/target reference checks before and after every model step",
            "failed_predecessor": "experiment/ac-server-20261021; confirmation pairing failed; no weights reused",
            "gpu_admission": POLICY,
            "additional_benchmark": {"datasets": ["CBSD68", "Kodak24", "McMaster", "Urban100"],
                "conditions": 15, "model_checkpoints": 7, "expected_rows": 22050,
                "historical_results_sha256": self.a["historical_results_csv_sha256"],
                "scope": "Historical benchmark reuse; full CBSD68 includes 20 development images; separate from primary holdout analysis"}})

    def gpu_options(self, kind):
        return shared_gpu_options(self, kind)

    def training(self, stage, seeds):
        if stage == "smoke":
            audit = self.reports / "historical_input_parity.json"
            command = [self.python, "tools/evaluate_ac_table1.py", "--paths", self.config_a,
                       "--verify-inputs-only", "--audit-output", str(audit)]
            self.execute([self.add("historical_input_parity", command, kind="cpu")])
            if not read_json(audit).get("completed"):
                raise RuntimeError("Historical benchmark input parity failed")
            command = [self.python, "-m", "pytest", "-q", "tests/test_ac_triplet_contract.py",
                       "tests/test_ac_triplet_gate.py", "tests/test_ac_table1_evaluation.py",
                       "tests/test_ac_gpu_policy.py", "--basetemp=/dev/shm/clip_ac_triplet_tests_20261021"]
            self.execute([self.add("triplet_contract_tests", command, kind="check")])
            report = self.reports / "actual_triplet_contract.json"
            command = [self.python, "-m", "tests.test_ac_triplet_contract", "--paths", self.config_a,
                       "--structure-paths", self.config_c, "--report", str(report)]
            self.execute([self.add("actual_triplet_contract", command, kind="check")])
            if not read_json(report).get("accepted"):
                raise RuntimeError("Actual triplet training/recovery contract failed")
        keys = []
        for seed in seeds:
            command = [self.python, "tools/run_ac_triplet.py", "--stage", stage, "--seed", str(seed),
                       "--paths", self.config_a, "--structure-paths", self.config_c, "--resume"]
            keys.append(self.add(f"{stage}_triplet_seed{seed}", command, kind="train",
                                 run_dir=self.output / stage / "_triplet" / f"seed{seed}"))
        self.execute(keys)
        proof = validate_triplet_stage(self.a, stage, seeds, source_hash())
        atomic_json(self.reports / f"{stage}_paired_training_check.json", proof)

    def evaluation(self, stage, seeds):
        super().evaluation(stage, seeds)
        if stage != "confirm":
            return
        self.state["phase"] = "historical_full_benchmark"
        self.record()
        keys = []
        for method, seed in [("C0", 0)] + [(m, s) for s in seeds for m in ("B0", "B1", "B3")]:
            command = [self.python, "tools/evaluate_ac_table1.py", "--paths",
                       self.config_c if method == "B3" else self.config_a,
                       "--method", method, "--training-seed", str(seed),
                       "--historical-csv", self.a["historical_results_csv"],
                       "--output", str(self.reports / "benchmark_table1")]
            keys.append(self.add(f"table1_{method}_seed{seed}", command, kind="eval"))
        self.execute(keys)
        command = [self.python, "tools/summarize_ac_table1.py", "--root", str(self.reports / "benchmark_table1"),
                   "--output", str(self.reports / "table1")]
        self.execute([self.add("summarize_table1", command, kind="cpu")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-a", default="configs/plan_a_server.yaml")
    parser.add_argument("--plan-c", default="configs/plan_c_server.yaml")
    args = parser.parse_args()
    Table1Queue(args.plan_a, args.plan_c).run()


if __name__ == "__main__":
    main()
