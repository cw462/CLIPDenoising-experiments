"""Scheduling safety: external GPU workloads and persisted child completion."""
import json
from types import SimpleNamespace

from tools import run_ac_pipeline as pipeline


def test_scheduler_excludes_external_gpu_and_recognizes_own_training_child(monkeypatch):
    queue = pipeline.Queue.__new__(pipeline.Queue)
    queue.a = {'gpu_indices': [0, 1], 'jobs_per_gpu': 4}
    queue.state = {'jobs': {'ours': {'status': 'running', 'gpu': 0,
                                   'process': {'pid': 90001}}}}

    def query(command, text):
        if '--query-gpu=index,uuid,memory.free' in command:
            return '0, GPU-a, 16000\n1, GPU-b, 22000\n'
        return 'GPU-a, 90002\nGPU-b, 90003\n'

    class Process:
        def __init__(self, pid):
            assert pid == 90001
            self.pid = pid

        def children(self, recursive):
            assert recursive
            return [SimpleNamespace(pid=90002)]

    monkeypatch.setattr(pipeline.subprocess, 'check_output', query)
    monkeypatch.setattr(pipeline.psutil, 'Process', Process)
    assert queue.gpu_options('train') == [(1, -16000.0, 0)]
    assert queue.gpu_options('check') == []  # check jobs must have an otherwise empty card


def test_resumed_queue_consumes_existing_exit_record_without_duplicate_launch(tmp_path, monkeypatch):
    queue = pipeline.Queue.__new__(pipeline.Queue)
    exit_file = tmp_path / 'job.exit.json'
    exit_file.write_text(json.dumps({'returncode': 0}), encoding='utf-8')
    queue.state = {'jobs': {'task': {'status': 'interrupted', 'exit_file': str(exit_file),
                                   'process': {'pid': 90001}}}}
    monkeypatch.setattr(pipeline, 'process_alive', lambda process: True)
    monkeypatch.setattr(queue, 'record', lambda: None)

    def no_launch(*args, **kwargs):
        raise AssertionError('Existing live wrapper completion must not be launched twice')

    monkeypatch.setattr(pipeline.subprocess, 'Popen', no_launch)
    queue.execute(['task'])
    assert queue.state['jobs']['task']['status'] == 'completed'
    assert queue.state['jobs']['task']['returncode'] == 0
