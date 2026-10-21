"""Run preregistered A/C continuation experiments, paired evaluation, and reporting."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from tools.experiment_common import (RunLock, atomic_json, digest, load_paths, process_alive,
                                     process_info, read_json, sha256_file, source_hash, utc_now)


class Queue:
    def __init__(self, config_a, config_c):
        self.config_a, self.config_c = str(config_a), str(config_c)
        self.a, self.c = load_paths(config_a), load_paths(config_c)
        shared = ('output_root', 'reports_dir', 'budgets', 'micro_batch', 'accumulation', 'amp',
                  'train_dir', 'test_datasets', 'data_manifest', 'pretrained_rn50', 'c0',
                  'learning_rate', 'minimum_learning_rate', 'weight_decay', 'cache_decoded_images')
        if any(self.a.get(key) != self.c.get(key) for key in shared):
            raise ValueError('A/C common training, data, output and budget settings must match')
        if self.a['micro_batch'] != 16 or self.a['accumulation'] != 1 or self.a['amp']:
            raise ValueError('This server protocol requires real batch16 and FP32')
        if any(self.a['budgets'].get(stage) != budget for stage, budget in
               {'smoke': 200, 'screen': 5000, 'confirm': 20000}.items()):
            raise ValueError('This queue requires the preregistered 200/5000/20000 budgets')
        self.output = Path(self.a['output_root'])
        self.reports = Path(self.a['reports_dir'])
        self.reports.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / 'pipeline_state.json'
        identity = {'source_hash': source_hash(), 'config_a_sha256': sha256_file(config_a),
                    'config_c_sha256': sha256_file(config_c)}
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state['identity'] != identity:
                raise RuntimeError('Source/config changed: preserve this queue and use a fresh output root')
        else:
            self.state = {'identity': identity, 'status': 'created', 'phase': 'preflight',
                          'started_at': utc_now(), 'jobs': {}, 'protocol': {
                              'groups': {'B0': 'matched L1 continuation', 'B1': 'proposal A rho=.5',
                                         'B3': 'proposal C structure_weight=.05'},
                              'screen_seed': 100, 'confirmation_seeds': [101, 102],
                              'budgets': {'smoke': 200, 'screen': 5000, 'confirm': 20000},
                              'formal_updates': 135000, 'batch': 16, 'common_c0': self.a['c0'],
                              'selection': 'Both A and C fixed in advance; no hyperparameter changes after screening',
                              'holdout_boundary': 'CBSD68 48 images unused in this round screening; all were evaluated in the earlier study',
                              'noise_seeds': [0, 1, 2], 'final_datasets': ['CBSD68_holdout48', 'Kodak24'],
                              'expected_final_per_image_rows': 10584}}
        self.python = sys.executable

    def record(self):
        self.state['updated_at'] = utc_now()
        atomic_json(self.state_path, self.state)
        lines = ['# A/C 续训实验进度', '', f"状态：{self.state['status']}；阶段：{self.state['phase']}",
                 f"更新时间（UTC）：{self.state['updated_at']}", '',
                 '| 任务 | 状态 | GPU | 更新步数 |', '|---|---|---:|---:|']
        for key, job in self.state['jobs'].items():
            progress = ''
            if job.get('run_dir'):
                status = Path(job['run_dir']) / 'status.json'
                if status.exists():
                    info = read_json(status)
                    progress = f"{info.get('update', 0)}/{info.get('budget', '')}"
                    job['update'] = info.get('update', 0)
            lines.append(f"| {key} | {job['status']} | {job.get('gpu', '')} | {progress} |")
        if self.state.get('reason'):
            lines += ['', self.state['reason']]
        (self.reports / 'progress.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def add(self, key, command, kind='train', run_dir=None):
        job = {'command': command, 'kind': kind, 'status': 'pending',
               'exit_file': str(self.output / 'jobs' / (key + '.exit.json')),
               'log': str(self.reports / 'jobs' / (key + '.log'))}
        if run_dir:
            job['run_dir'] = str(run_dir)
        if key in self.state['jobs']:
            if self.state['jobs'][key]['command'] != command:
                raise RuntimeError('Job command changed: ' + key)
        else:
            self.state['jobs'][key] = job
        return key

    def gpu_options(self, kind):
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                          '--format=csv,noheader,nounits'], text=True)
        devices = {}
        for fields in csv.reader(result.splitlines()):
            index, uuid, free = [field.strip() for field in fields]
            if int(index) in self.a.get('gpu_indices', [0, 1]):
                devices[uuid] = {'index': int(index), 'free_mib': float(free), 'external': []}
        own_pids = {os.getpid()}
        running = [j for j in self.state['jobs'].values() if j['status'] == 'running']
        for job in running:
            try:
                process = psutil.Process(job['process']['pid'])
                own_pids.add(process.pid)
                own_pids.update(child.pid for child in process.children(recursive=True))
            except psutil.Error:
                pass
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
        for fields in csv.reader(apps.splitlines()):
            if len(fields) == 2:
                uuid, pid = [field.strip() for field in fields]
                if uuid in devices and int(pid) not in own_pids:
                    devices[uuid]['external'].append(int(pid))
        maximum = 1 if kind == 'check' else (2 if kind == 'eval' else int(self.a.get('jobs_per_gpu', 4)))
        required = 8000 if kind == 'check' else (3500 if kind == 'train' else 4000)
        choices = []
        for device in devices.values():
            count = sum(j.get('gpu') == device['index'] for j in running)
            # Respect pre-existing compute workloads. Reconsider cards on every poll.
            if not device['external'] and count < maximum and device['free_mib'] >= required:
                choices.append((count, -device['free_mib'], device['index']))
        return sorted(choices)

    def execute(self, keys):
        for key in keys:
            job = self.state['jobs'][key]
            if job['status'] in ('failed', 'interrupted'):
                if process_alive(job.get('process')):
                    job['status'] = 'running'
                else:
                    job['status'] = 'pending'
                    Path(job['exit_file']).unlink(missing_ok=True)
        while True:
            for key in keys:
                job = self.state['jobs'][key]
                if job['status'] != 'running':
                    continue
                exit_file = Path(job['exit_file'])
                if exit_file.exists():
                    result = read_json(exit_file)
                    job.update(status='completed' if result['returncode'] == 0 else 'failed',
                               returncode=result['returncode'], ended_at=utc_now())
                elif not process_alive(job.get('process')):
                    job.update(status='interrupted', ended_at=utc_now())
            failed = [key for key in keys if self.state['jobs'][key]['status'] in ('failed', 'interrupted')]
            if failed:
                self.record()
                raise RuntimeError('Tasks failed/interrupted; preserve logs and resume after diagnosis: ' + ', '.join(failed))
            if all(self.state['jobs'][key]['status'] == 'completed' for key in keys):
                self.record()
                return
            for key in keys:
                job = self.state['jobs'][key]
                if job['status'] != 'pending':
                    continue
                if job['kind'] == 'cpu':
                    gpu = None
                else:
                    choices = self.gpu_options(job['kind'])
                    if not choices:
                        continue
                    gpu = choices[0][2]
                if shutil.disk_usage(self.output).free < float(self.a.get('minimum_free_gb', 2)) * 2**30:
                    raise RuntimeError('Insufficient free disk; preserve checkpoints and address storage')
                for target in (job['log'], job['exit_file']):
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
                                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
                environment['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
                wrapped = [self.python, 'tools/ac_job_runner.py', '--exit-file', job['exit_file'], '--', *job['command']]
                with Path(job['log']).open('a', encoding='utf-8') as stream:
                    process = subprocess.Popen(wrapped, cwd=ROOT, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                child = psutil.Process(process.pid)
                job.update(status='running', gpu=gpu, started_at=utc_now(),
                           process={'pid': child.pid, 'create_time': child.create_time(),
                                    'host': process_info()['host']})
                self.record()
                # Give CUDA context allocation time to appear before allocating another slot.
                time.sleep(3)
            self.record()
            time.sleep(10)

    def training(self, stage, seeds):
        keys = []
        for seed in seeds:
            for method in ('B0', 'B1', 'B3'):
                config = self.config_c if method == 'B3' else self.config_a
                script = 'tools/run_structure_experiment.py' if method == 'B3' else 'tools/run_pfa_experiment.py'
                command = [self.python, script, '--group', method, '--stage', stage,
                           '--seed', str(seed), '--paths', config, '--resume']
                keys.append(self.add(f'{stage}_{method}_seed{seed}', command,
                                     run_dir=self.output / stage / method / f'seed{seed}'))
        self.execute(keys)
        # Full step-by-step input and target hashes must match each paired training seed.
        checks = []
        for seed in seeds:
            reference = None
            for method in ('B0', 'B1', 'B3'):
                run = self.output / stage / method / f'seed{seed}'
                status, latest = read_json(run / 'status.json'), read_json(run / 'checkpoints/latest.json')
                budget = int(self.a['budgets'][stage])
                if status['status'] != 'completed' or status['update'] != budget or latest['update'] != budget:
                    raise RuntimeError('Incomplete training: ' + str(run))
                if sha256_file(latest['path']) != latest['sha256']:
                    raise RuntimeError('Checkpoint SHA256 mismatch: ' + str(run))
                sequence = [(r['update'], r['input_sha256'], r['target_sha256'])
                            for r in map(json.loads, (run / 'train.jsonl').read_text().splitlines())]
                if len(sequence) != budget or [r[0] for r in sequence] != list(range(1, budget + 1)):
                    raise RuntimeError('Missing/duplicate training rows: ' + str(run))
                if reference is not None and sequence != reference:
                    raise RuntimeError('Paired input/target sequences differ: ' + str(run))
                reference = sequence
                checks.append({'method': method, 'seed': seed, 'updates': budget, 'checkpoint_sha256': latest['sha256']})
        atomic_json(self.reports / f'{stage}_paired_training_check.json', {'status': 'passed', 'runs': checks})

    def evaluation(self, stage, seeds):
        keys = []
        pairs = [('C0', 0)] + [(method, seed) for seed in seeds for method in ('B0', 'B1', 'B3')]
        for method, seed in pairs:
            config = self.config_c if method == 'B3' else self.config_a
            command = [self.python, 'tools/evaluate_pfa.py', '--paths', config, '--stage', stage,
                       '--method', method, '--training-seed', str(seed),
                       '--output', str(self.reports / 'evaluations'), '--splits']
            command += ['dev'] if stage == 'screen' else ['holdout', 'kodak']
            keys.append(self.add(f'eval_{stage}_{method}_seed{seed}', command, kind='eval'))
        self.execute(keys)
        command = [self.python, 'tools/summarize_ac.py', '--stage', stage,
                   '--root', str(self.reports / 'evaluations'), '--output', str(self.reports / stage)]
        self.execute([self.add(f'summarize_{stage}', command, kind='cpu')])

    def run(self):
        with RunLock(self.output / 'pipeline.lock'):
            if self.state['status'] == 'completed':
                print('A/C queue already completed', flush=True)
                return
            self.state.update(status='running', process=process_info())
            self.state.pop('reason', None)
            try:
                self.state['phase'] = 'contract_tests'
                command = [self.python, '-m', 'pytest', '-q', 'tests/test_spatial_pfa.py',
                           'tests/test_structure_loss.py', 'tests/test_ac_training_contract.py',
                           'tests/test_ac_evaluation.py', 'tests/test_ac_pipeline.py',
                           '--basetemp=/dev/shm/clip_ac_contracts_20261021']
                self.execute([self.add('contract_tests', command, kind='check')])
                command = [self.python, '-m', 'tests.test_ac_training_contract', '--paths', self.config_a,
                           '--structure-paths', self.config_c, '--device', 'cuda', '--batch', '2',
                           '--report', str(self.reports / 'actual_model_contract.json')]
                self.execute([self.add('actual_model_contract', command, kind='check')])
                if not read_json(self.reports / 'actual_model_contract.json').get('accepted'):
                    raise RuntimeError('Actual RN50/C0 paired training and recovery contract failed')
                for phase, seeds in [('smoke', [100]), ('screen', [100]), ('confirm', [101, 102])]:
                    self.state['phase'] = phase
                    self.record()
                    self.training(phase, seeds)
                    if phase != 'smoke':
                        self.state['phase'] = 'evaluation_' + phase
                        self.record()
                        self.evaluation(phase, seeds)
                self.state.update(status='completed', phase='completed', ended_at=utc_now())
                self.record()
            except BaseException as error:
                self.state.update(status='failed', reason=f'{type(error).__name__}: {error}')
                self.record()
                (self.reports / 'pipeline_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-a', default='configs/plan_a_server.yaml')
    parser.add_argument('--plan-c', default='configs/plan_c_server.yaml')
    args = parser.parse_args()
    Queue(args.plan_a, args.plan_c).run()


if __name__ == '__main__':
    main()
