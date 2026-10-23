"""Recovery counts never waive full input protection or independent evidence."""
import copy
import json

import pytest

from tools import evaluate_c_only as evaluation
from tools.experiment_common import sha256_file


@pytest.fixture
def recovered_run(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "UPDATES", 3)
    config = {"source_hash": "source", "config_hash": "config",
              "execution_contract": {"readback_recovery_policy": "independent_stable_v1"},
              "runtime": {"readback_recovery_policy": "independent_stable_v1"}}
    events = []
    for index, update in enumerate((0, 2, 2), 1):
        events.append({"schema": "independent_stable_readback_recovery_v1", "policy": "independent_stable_v1",
                       "status": "recovered", "event_index": index, "update": update,
                       "source_hash": "source", "config_hash": "config", "phase": "snapshot",
                       "timestamp": "2026-10-21T00:00:00Z", "device": "cpu", "dtype": "torch.float32",
                       "shape": [16, 3, 128, 128], "stride": [49152, 16384, 128, 1],
                       "first_sha256": "a" * 64, "second_sha256": "b" * 64, "stable_sha256": "b" * 64,
                       "different_byte_count": 1, "first_differences": [{"offset": 3, "first": 61, "second": 63}],
                       "confirmation_rounds": 3, "independent_reads_agree": True,
                       "source_clone_exact": True, "source_unchanged": True,
                       "confirmation_sha256": [{key: "b" * 64 for key in ("source_before", "clone", "source_after")} for _ in range(3)]})
    rows = [{"update": update, "dataset_replay_count": 16,
             **{field: True for field in ("dataset_replay_exact", "dataset_rng_exact", "feed_replay_exact",
                "raw_unchanged", "feed_rng_exact", "optimizer_inputs_unchanged", "snapshot_copies_exact")},
             "input_sha256": "c" * 64, "post_input_sha256": "c" * 64,
             "target_sha256": "d" * 64, "post_target_sha256": "d" * 64,
             "readback_recovery_policy": "independent_stable_v1", "readback_recovery_events": 2 if update == 2 else 0}
            for update in range(1, 4)]
    path, log, recovery = (tmp_path / name for name in ("input_protection.jsonl", "train.jsonl", "readback_recovery.jsonl"))
    def save():
        for file, values in ((path, rows), (recovery, events),
                             (log, [{**row, "loss": .1} for row in rows])):
            file.write_text("".join(json.dumps(row) + "\n" for row in values))
        return {"schema": "c_only_input_protection_v1", "status": "passed", "update": 3, "verified_updates": 3,
                "source_hash": "source", "config_hash": "config", "proof_sha256": sha256_file(path),
                "readback_recovery_policy": "independent_stable_v1", "readback_recovery_events_total": 3,
                "readback_recovery_initialization_events": 1, "readback_recovery_sha256": sha256_file(recovery)}
    return dict(path=path, log=log, recovery=recovery, rows=rows, events=events, config=config, save=save)


def test_confirmed_recoveries_count_initialization_separately_and_bind_log(recovered_run):
    x = recovered_run
    proof = x["save"]()
    assert evaluation.validate_input_protection(x["path"], proof, x["config"], x["log"]) == 3
    disclosure, counts = evaluation.validate_readback_recovery(x["recovery"], x["config"])
    assert counts == {0: 1, 2: 2}
    assert disclosure["events_total"] == 3 and disclosure["initialization_events"] == 1
    assert disclosure["affected_training_updates"] == 1
    assert disclosure["affected_training_update_numbers"] == [2]
    assert disclosure["root_cause_established"] is False
    contract = {"readback_recovery": disclosure, "input_protection": proof, "training_checks": dict(proof),
                "proof_sha256": {"readback_recovery.jsonl": sha256_file(x["recovery"])}}
    assert evaluation.validate_recovery_disclosure(contract) == disclosure
    contract["proof_sha256"]["readback_recovery.jsonl"] = "e" * 64
    with pytest.raises(ValueError, match="contract SHA256"):
        evaluation.validate_recovery_disclosure(contract)


@pytest.mark.parametrize("mutation", ["confirmation", "stable", "source", "count", "original_guard", "log_hash", "root_claim"])
def test_recovery_never_masks_invalid_evidence_or_original_protection(recovered_run, mutation):
    x = recovered_run
    if mutation == "confirmation":
        x["events"][1]["confirmation_sha256"][1]["clone"] = "a" * 64
    elif mutation == "stable":
        x["events"][1]["stable_sha256"] = "e" * 64
    elif mutation == "source":
        x["events"][1]["source_hash"] = "different"
    elif mutation == "count":
        x["rows"][1]["readback_recovery_events"] = 1
    elif mutation == "original_guard":
        x["rows"][1]["raw_unchanged"] = False
    proof = x["save"]()
    if mutation == "log_hash":
        proof["readback_recovery_sha256"] = "e" * 64
    if mutation == "root_claim":
        disclosure, _ = evaluation.validate_readback_recovery(x["recovery"], x["config"])
        disclosure["root_cause_established"] = True
        with pytest.raises(ValueError, match="root cause"):
            evaluation.validate_recovery_disclosure({"readback_recovery": disclosure})
    else:
        with pytest.raises(ValueError):
            evaluation.validate_input_protection(x["path"], proof, x["config"], x["log"])


def test_strict_policy_cannot_hide_recovery_log_and_policy_must_agree(recovered_run):
    x = recovered_run
    x["save"]()
    strict = copy.deepcopy(x["config"])
    strict["execution_contract"]["readback_recovery_policy"] = "strict"
    strict["runtime"]["readback_recovery_policy"] = "strict"
    with pytest.raises(ValueError, match="unexpected readback"):
        evaluation.validate_readback_recovery(x["recovery"], strict)
    strict["runtime"]["readback_recovery_policy"] = "independent_stable_v1"
    with pytest.raises(ValueError, match="Runtime"):
        evaluation.validate_readback_recovery(x["recovery"], strict)


def test_resume_initialization_events_do_not_reset_positive_update_order(recovered_run):
    x = recovered_run
    for event, update in zip(x["events"], (2, 0, 2)):
        event["update"] = update
    assert evaluation.validate_input_protection(x["path"], x["save"](), x["config"], x["log"]) == 3
    x["events"][-1]["update"] = 1
    with pytest.raises(ValueError, match="order/update"):
        evaluation.validate_input_protection(x["path"], x["save"](), x["config"], x["log"])
