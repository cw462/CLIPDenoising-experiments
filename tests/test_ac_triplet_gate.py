"""The recovery benchmark must reject stale commits or unproved training pairs."""
import json
from pathlib import Path

import pytest

from tools.ac_triplet_gate import GROUPS, validate_triplet_stage
from tools.experiment_common import atomic_json, digest, read_json, sha256_file


def fixture(tmp_path):
    paths = {"output_root": str(tmp_path / "runs"), "budgets": {"smoke": 2}}
    root = Path(paths["output_root"]) / "smoke"
    coordinator = root / "_triplet/seed100"
    commit = {"schema": "ac_triplet_commit_v1", "update": 2, "source_hash": "frozen",
              "groups": {}, "config_hashes": {}, "paired_protocol_hash": "paired"}
    for group in GROUPS:
        run = root / group / "seed100"
        checkpoint = run / "checkpoints/update_000002.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(group.encode())
        config = {"exp": group, "seed": 100, "stage": "smoke", "source_hash": "frozen",
                  "execution_contract": {"schema": "ac_triplet_v2"}, "paired_protocol_hash": "paired"}
        config["config_hash"] = digest(config)
        atomic_json(run / "effective_config.json", config)
        commit["config_hashes"][group] = config["config_hash"]
        ref = {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "update": 2}
        commit["groups"][group] = ref
        atomic_json(run / "checkpoints/latest.json", ref)
        atomic_json(run / "status.json", {"status": "completed", "update": 2})
        checks = {key: True for key in ("encoder_parameters_unchanged", "encoder_gradients_absent",
                  "last_checkpoint_sha256_verified", "exact_reference_pre_post_every_update",
                  "shared_rng_unchanged_every_model_step", "complete")}
        atomic_json(run / "training_checks.json", dict(checks, update=2, triplet_commit_update=2))
        rows = []
        for update in (1, 2):
            row = {"update": update, "exact_reference_and_unchanged": True, "shared_rng_unchanged": True}
            row.update({f"{prefix}{tensor}_sha256": digest((tensor, update))
                        for prefix in ("", "reference_", "pre_", "post_") for tensor in ("input", "target")})
            rows.append(row)
        (run / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    atomic_json(coordinator / "commit.json", commit)
    atomic_json(coordinator / "status.json", {"status": "completed", "update": 2})
    return paths


def test_gate_accepts_only_complete_equal_sequences(tmp_path):
    paths = fixture(tmp_path)
    assert len(validate_triplet_stage(paths, "smoke", [100], "frozen")["runs"]) == 3
    log = Path(paths["output_root"]) / "smoke/B1/seed100/train.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    # A self-consistent but different pair is rejected against the common reference.
    for prefix in ("", "reference_", "pre_", "post_"):
        rows[1][f"{prefix}input_sha256"] = "1" * 64
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RuntimeError, match="sequence mismatch"):
        validate_triplet_stage(paths, "smoke", [100], "frozen")


@pytest.mark.parametrize("mutation", ["step", "post", "rng", "commit", "checkpoint"])
def test_gate_rejects_incomplete_or_changed_proof(tmp_path, mutation):
    paths = fixture(tmp_path)
    run = Path(paths["output_root"]) / "smoke/B3/seed100"
    if mutation == "commit":
        latest = read_json(run / "checkpoints/latest.json")
        latest["update"] = 1
        atomic_json(run / "checkpoints/latest.json", latest)
    elif mutation == "checkpoint":
        (run / "checkpoints/update_000002.pth").write_bytes(b"modified")
    else:
        log = run / "train.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        if mutation == "step":
            rows.pop()
        elif mutation == "post":
            rows[0]["post_target_sha256"] = "2" * 64
        else:
            rows[0]["shared_rng_unchanged"] = False
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises((RuntimeError, ValueError)):
        validate_triplet_stage(paths, "smoke", [100], "frozen")
