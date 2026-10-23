"""Run the user-requested ordinary G1C training, then evaluation and reporting."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.experiment_common import RunLock, process_info, utc_now
from tools.run_g1c_multiseed_pipeline import G1CMultiSeedQueue

MODE = "standard_user_requested_v1"


class G1CStandardQueue(G1CMultiSeedQueue):
    def __init__(self, config):
        super().__init__(config)
        if self.a.get("training_integrity_mode") != MODE:
            raise ValueError("Ordinary training requires the explicitly requested mode")
        if self.a.get("readback_recovery") or self.a.get("execution_blocked_reason"):
            raise ValueError("Use a separate standard configuration, not a failed protected run")
        protocol = self.state["protocol"]
        for field in ("readback_recovery_policy", "readback_limitation"):
            protocol.pop(field, None)
        protocol.update(
            training_integrity_mode=MODE,
            per_update_input_consistency_checks=False,
            training_entrypoint="tools/run_g1c_standard.py",
            authorization="User explicitly requested ordinary legacy-style training without additional input consistency checks",
            retained_checks="finite loss/gradients, normal checkpoints, complete update count and frozen encoder",
            concurrency="GPU0 seed0; GPU1 independent seeds1 and2; no DDP",
            known_host_issue="Prior host read/write discrepancies remain documented; ordinary training does not verify their absence",
        )

    def preflight(self):
        # Assets already exist from the historical run. Do not repeat tensor
        # readback/replay tests or gate training on their old failure records.
        for field in ("pretrained_rn50", "train_dir", "data_manifest", "historical_results_csv"):
            if not Path(self.a[field]).exists():
                raise FileNotFoundError(self.a[field])
        self.state["asset_paths_present"] = True
        self.record()

    def record(self):
        super().record()
        path = self.reports / "progress.md"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = "# 门控+C：三个 seed 各完整 300k 普通训练"
        lines[1:1] = ["", "按用户要求关闭额外逐步输入一致性检查；保留正常训练日志和断点。"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def train(self, smoke=False):
        if smoke:
            raise ValueError("This queue starts full training directly; no separate smoke cohort")
        keys = []
        for seed in (0, 1, 2):
            run = self.output / "full" / "G1C" / f"seed{seed}"
            command = [self.python, "tools/run_g1c_standard.py", "--seed", str(seed),
                       "--paths", self.config_a, "--resume"]
            keys.append(self.add_seed_job(f"full_G1C_seed{seed}", command, seed, run_dir=run))
        self.execute(keys)

    def run(self):
        with RunLock(self.output / "pipeline.lock"):
            if self.state["status"] == "completed":
                return
            self.state.update(status="running", process=process_info())
            self.state.pop("reason", None)
            try:
                for phase, action in (("assets", self.preflight), ("full300k", self.train),
                                      ("evaluation", self.evaluate)):
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
    parser.add_argument("--paths", default="configs/plan_g1c_standard_server.yaml")
    args = parser.parse_args()
    G1CStandardQueue(args.paths).run()


if __name__ == "__main__":
    main()
