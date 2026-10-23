"""Fail-closed verification of shared-input training before benchmark access."""
from __future__ import annotations

import json
from pathlib import Path

from tools.experiment_common import digest, read_json, sha256_file

GROUPS = ("B0", "B1", "B3")


def validate_triplet_stage(paths, stage, seeds, current_source=None):
    budget = int(paths["budgets"][stage])
    root = Path(paths["output_root"]) / stage
    evidence = []
    for seed in seeds:
        coordinator = root / "_triplet" / f"seed{seed}"
        commit = read_json(coordinator / "commit.json")
        status = read_json(coordinator / "status.json")
        if (commit.get("schema") != "ac_triplet_commit_v1" or commit.get("update") != budget
                or set(commit.get("groups", {})) != set(GROUPS)
                or status.get("status") != "completed" or status.get("update") != budget):
            raise RuntimeError(f"Incomplete triplet commit: {coordinator}")
        if current_source and commit.get("source_hash") != current_source:
            raise ValueError("Triplet source mismatch")
        reference = None
        for group in GROUPS:
            run = root / group / f"seed{seed}"
            config = read_json(run / "effective_config.json")
            raw = dict(config)
            stored = raw.pop("config_hash")
            latest = read_json(run / "checkpoints/latest.json")
            checks = read_json(run / "training_checks.json")
            state = read_json(run / "status.json")
            if (digest(raw) != stored or commit["config_hashes"].get(group) != stored
                    or config.get("paired_protocol_hash") != commit["paired_protocol_hash"]
                    or config.get("source_hash") != commit["source_hash"]
                    or config.get("execution_contract", {}).get("schema") != "ac_triplet_v2"
                    or (config.get("exp"), config.get("stage"), config.get("seed")) != (group, stage, seed)):
                raise ValueError(f"Triplet config mismatch: {run}")
            if (latest != commit["groups"][group] or latest.get("update") != budget
                    or state.get("status") != "completed" or state.get("update") != budget):
                raise RuntimeError(f"Uncommitted or incomplete triplet group: {run}")
            checkpoint = Path(latest["path"])
            if checkpoint.resolve().parent != (run / "checkpoints").resolve():
                raise ValueError("Triplet checkpoint outside run")
            if sha256_file(checkpoint) != latest["sha256"]:
                raise ValueError("Triplet checkpoint checksum mismatch")
            flags = ("encoder_parameters_unchanged", "encoder_gradients_absent",
                     "last_checkpoint_sha256_verified", "exact_reference_pre_post_every_update",
                     "shared_rng_unchanged_every_model_step", "complete")
            if (any(checks.get(key) is not True for key in flags)
                    or checks.get("triplet_commit_update") != budget or checks.get("update") != budget):
                raise RuntimeError(f"Triplet training checks failed: {run}")
            sequence = []
            with (run / "train.jsonl").open(encoding="utf-8") as stream:
                for expected, line in enumerate(stream, 1):
                    row = json.loads(line)
                    if (row.get("update") != expected
                            or row.get("exact_reference_and_unchanged") is not True
                            or row.get("shared_rng_unchanged") is not True):
                        raise RuntimeError(f"Missing exact per-step training proof: {run}")
                    for tensor in ("input", "target"):
                        hashes = [row.get(f"{prefix}{tensor}_sha256")
                                  for prefix in ("", "reference_", "pre_", "post_")]
                        if len(set(hashes)) != 1 or not isinstance(hashes[0], str) or len(hashes[0]) != 64:
                            raise RuntimeError(f"Triplet input/target mutation: {run}, step {expected}")
                    sequence.append((row["input_sha256"], row["target_sha256"]))
            if len(sequence) != budget or (reference is not None and sequence != reference):
                raise RuntimeError(f"Triplet paired sequence mismatch: {run}")
            reference = sequence
            evidence.append({"method": group, "seed": seed, "updates": budget,
                             "checkpoint_sha256": latest["sha256"],
                             "log_sha256": sha256_file(run / "train.jsonl"),
                             "commit_sha256": sha256_file(coordinator / "commit.json")})
    return {"status": "passed", "schema": "ac_triplet_gate_v1", "runs": evidence}
