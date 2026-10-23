"""Bounded observation recovery: never retry or change actual training inputs."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_c_only_experiment as runner
from tools.experiment_common import read_json, rng_state, seed_all, sha256_file, torch_load
from tests.test_c_only_contract import full_paths, RandomDataset, ToyModel, raw_batch
from tests.test_ac_training_contract import continuation_paths


def guard(path, policy=runner.RECOVERY_POLICY):
    return runner.Protection(path, {'config_hash': 'fixture', 'source_hash': 'fixture',
        'runtime': {'readback_recovery_policy': policy}})


def flip(data, bit=1):
    result = bytearray(data)
    result[0] ^= bit
    return bytes(result)


def one_bad_observation(monkeypatch):
    original = runner._bytes_once
    calls = []
    def read(value):
        result = original(value)
        calls.append(value)
        return flip(result) if len(calls) == 1 else result
    monkeypatch.setattr(runner, '_bytes_once', read)
    return calls


def test_single_observation_recovers_without_tensor_or_rng_change(tmp_path, monkeypatch):
    seed_all(31)
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    before = runner._source_bytes(value)
    rng_before = copy.deepcopy(rng_state())
    protection = guard(tmp_path)
    protection.begin_update(7)
    calls = one_bad_observation(monkeypatch)
    with protection.scope('feed_train_data'):
        actual = runner.tensor_bytes(value)
    assert actual == before == runner._source_bytes(value)
    assert len(calls) == 2  # No repeated ordinary observer and no training retry.
    assert runner.exact_equal(rng_before, rng_state())
    rows = runner.read_recovery_rows(tmp_path / 'readback_recovery.jsonl')
    assert len(rows) == 1 and rows[0]['update'] == 7 and rows[0]['confirmation_rounds'] == 3
    assert rows[0]['first_sha256'] != rows[0]['second_sha256'] == rows[0]['stable_sha256']
    assert protection.recovery_metadata()['readback_recovery_events_total'] == 1


def test_default_strict_policy_still_fails(tmp_path, monkeypatch):
    protection = runner.Protection(tmp_path, {'config_hash': 'fixture', 'source_hash': 'fixture'})
    one_bad_observation(monkeypatch)
    with protection.scope('dataset'), pytest.raises(runner.InputProtectionError, match='Repeated actual'):
        runner.tensor_bytes(torch.ones(4))
    assert not (tmp_path / 'readback_recovery.jsonl').exists()


@pytest.mark.parametrize('fault', ['persistent', 'third_value', 'clone', 'source_mutation'])
def test_confirmation_faults_are_fatal(tmp_path, monkeypatch, fault):
    value = torch.arange(4, dtype=torch.float32)
    protection = guard(tmp_path)
    one_bad_observation(monkeypatch)
    independent = runner._independent_bytes
    if fault == 'persistent':
        monkeypatch.setattr(runner, '_independent_bytes', lambda tensor: flip(independent(tensor)))
    elif fault == 'third_value':
        source = runner._source_bytes
        monkeypatch.setattr(runner, '_source_bytes', lambda tensor: flip(source(tensor), 2))
        monkeypatch.setattr(runner, '_independent_bytes', lambda tensor: flip(independent(tensor), 2))
    elif fault == 'clone':
        original = torch.Tensor.clone
        def bad_clone(tensor, *args, **kwargs):
            result = original(tensor, *args, **kwargs)
            result.view(torch.int32)[0] ^= 1
            return result
        monkeypatch.setattr(torch.Tensor, 'clone', bad_clone)
    else:
        source = runner._source_bytes
        count = 0
        def mutating_read(tensor):
            nonlocal count
            count += 1
            if count == 2: tensor.view(torch.int32)[0] ^= 2
            return source(tensor)
        monkeypatch.setattr(runner, '_source_bytes', mutating_read)
    with protection.scope('dataset'), pytest.raises(runner.InputProtectionError):
        runner.tensor_bytes(value)
    assert (tmp_path / 'readback_recovery.jsonl').read_text() == ''


def test_recovery_record_must_be_durable(tmp_path, monkeypatch):
    protection = guard(tmp_path)
    one_bad_observation(monkeypatch)
    monkeypatch.setattr(runner.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('disk write failed')))
    with protection.scope('dataset'), pytest.raises(OSError, match='disk write failed'):
        runner.tensor_bytes(torch.ones(4))
    assert protection.recovery_total == 0


def test_recovered_observation_does_not_hide_dataset_difference(tmp_path, monkeypatch):
    class BadDataset(RandomDataset):
        calls = 0
        def __getitem__(self, index):
            result = super().__getitem__(index)
            self.calls += 1
            if self.calls == 2: result['gt'].view(torch.int32)[0, 0, 0] ^= 1
            return result
    protection = guard(tmp_path)
    one_bad_observation(monkeypatch)
    with pytest.raises(runner.InputProtectionError, match='Dataset replay differs'):
        runner.ProtectedDataset(BadDataset(), protection)[0]
    assert (tmp_path / 'input_failure/failure.json').exists()
    assert protection.recovery_total == 1


def test_normal_dataset_and_feed_recovery_preserve_rng(tmp_path, monkeypatch):
    seed_all(91)
    initial = copy.deepcopy(rng_state())
    expected = RandomDataset()[0]
    expected_after = copy.deepcopy(rng_state())
    runner.restore_rng(initial)
    protection = guard(tmp_path)
    one_bad_observation(monkeypatch)
    actual = runner.ProtectedDataset(RandomDataset(), protection)[0]
    assert runner.exact_equal(actual, expected)
    assert runner.exact_equal(rng_state(), expected_after)
    assert protection.recovery_total == 1


def test_enabled_training_checkpoint_and_resume_event_binding(full_paths, monkeypatch):
    paths = dict(full_paths, readback_recovery=runner.RECOVERY_POLICY)
    original = runner._bytes_once
    injected = False
    def read(value):
        nonlocal injected
        result = original(value)
        context = runner._READBACK_CONTEXT.get()
        if context is not None and context.phase == 'dataset' and context.update == 1 and not injected:
            injected = True
            return flip(result)
        return result
    monkeypatch.setattr(runner, '_bytes_once', read)
    run = runner.run('G1C', 'full', 0, paths, stop_after=2, tag='recovery_resume')
    checkpoint = torch_load(read_json(run / 'checkpoints/latest.json')['path'])
    assert checkpoint['readback_recovery']['total'] == 1
    assert checkpoint['readback_recovery']['sha256'] == sha256_file(run / 'readback_recovery.jsonl')
    with (run / 'readback_recovery.jsonl').open('a') as stream:
        stream.write('{"event_index":2,"update":3')
    runner.run('G1C', 'full', 0, paths, resume=True, stop_after=4, tag='recovery_resume')
    summary = read_json(run / 'summary.json')
    assert summary['readback_recovery_events_total'] == 1
    assert summary['readback_recovery_sha256'] == sha256_file(run / 'readback_recovery.jsonl')
    assert any('readback_recovery' in p.name for p in (run / 'interrupted_tails').iterdir())
    rows = [json.loads(line) for line in (run / 'input_protection.jsonl').read_text().splitlines()]
    assert [row['readback_recovery_events'] for row in rows] == [1, 0, 0, 0]
    assert all(row['readback_recovery_policy'] == runner.RECOVERY_POLICY and runner.validate_proof_row(row) for row in rows)
    checkpoint = torch_load(read_json(run / 'checkpoints/latest.json')['path'])
    assert checkpoint['input_protection_sha256'] == sha256_file(run / 'input_protection.jsonl')


def test_invalid_policy_rejected(full_paths):
    with pytest.raises(ValueError, match='Unsupported readback'):
        runner.configuration('G1C', 'full', 0, dict(full_paths, readback_recovery='ignore'))
