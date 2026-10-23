"""One authorized evaluation-only recovery; preserve the failed attempt unchanged."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
sys.path.insert(0, str(ROOT))
from tools.experiment_common import (RunLock, atomic_json, digest, git_commit, load_paths,
    process_alive, process_info, read_json, sha256_file, source_hash, utc_now)
from tools.run_g1c_standard_pipeline import G1CStandardQueue
from tools.evaluate_c_only import (STANDARD_MODE, validated_gate, load_historical,
    historical_baseline, protocol_for_seed)
from tools.summarize_g1c_multiseed import (validate_completion_contract, validate_rows,
    read_csv, cell_key, TABLE1_CONDITIONS)

CONFIG_REL = 'configs/plan_g1c_standard_server.yaml'
CONFIG = ROOT / CONFIG_REL
OUTPUT = ROOT / 'runs/g1c_standard'
STATE = OUTPUT / 'pipeline_state.json'
REPORTS = ROOT / 'reports/g1c_standard'
EVIDENCE = ROOT / 'reproducibility/g1c_standard'
AUDIT = EVIDENCE / 'evaluation_recovery_audit.json'
RECHECK = EVIDENCE / 'evaluation_failed_case_recheck.json'
PINS = {'scientific_commit': 'c02ebedd103af716afc3ad29491b3886e6ba37ff',
        'source_hash': '86d821ff44713da6918b73f92ed7b9e2a7b3d83f53082eef9c82f36a654a534d',
        'config_sha256': '9ba8dcdea71c06d9591c865bd85644b81a2c04770fcc0aabfc6bc820bdc22eb8'}
KEYS = [f'evaluation_G1C_seed{i}' for i in (0, 1, 2)]
ERROR = 'ValueError: Historical noisy-input mismatch: Kodak24/kodim18.png/speckle/0.04'


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def safe_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_relative_to(ROOT), 'Path is outside the owned worktree')
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Resolved path escaped worktree')
    cursor = path
    while cursor != ROOT:
        require(not cursor.is_symlink(), 'Symlink is forbidden for recovery artifacts')
        cursor = cursor.parent
    return path


def file_proof(path):
    path = safe_path(path)
    require(path.is_file(), 'Required evidence file missing: ' + str(path))
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': sha256_file(path)}


def tree_proofs(path):
    path = safe_path(path)
    require(path.is_dir(), 'Partial evaluation directory missing')
    files = []
    for item in sorted(path.rglob('*')):
        safe_path(item)
        require(item.is_dir() or item.is_file(), 'Special file in evaluation output')
        if item.is_file():
            files.append(file_proof(item))
    return files


def no_live_processes(state):
    candidates = [state.get('process')]
    candidates.extend(job.get('process') for job in state['jobs'].values())
    for seed in (0, 1, 2):
        run = OUTPUT / 'full/G1C' / f'seed{seed}'
        candidates.append(read_json(run / 'status.json').get('process'))
        for lock in (run / 'process.lock', REPORTS / 'evaluation/G1C' / f'seed{seed}/process.lock'):
            if lock.exists():
                candidates.append(read_json(lock))
    require(not any(process_alive(info) for info in candidates), 'An existing owned job is still alive')


def verify_seed1(paths, contract, identity):
    directory = REPORTS / 'evaluation/G1C/seed1'
    item = read_json(directory / 'summary.json')
    require(item.get('completed') is True and all(item.get(k) == 3150 for k in
            ('rows', 'baseline_rows', 'matched_historical_inputs')), 'Seed1 evaluation incomplete')
    require(item['identity'] == identity and item['contract'] == contract
            and item['contract_hash'] == digest(contract), 'Seed1 training identity changed')
    validate_completion_contract(contract, identity, 1)
    protocol = protocol_for_seed(1, STANDARD_MODE)
    require(item['protocol_hash'] == digest(protocol) == digest(item['protocol']), 'Seed1 protocol changed')
    historical_csv, reference, historical = load_historical(paths)
    require(item['historical_csv']['sha256'] == historical['sha256'], 'Historical CSV changed')
    artifacts = item['artifact_sha256']
    require({'per_image_results.csv', 'baseline_per_image_results.csv', 'access_gate.json'}.issubset(artifacts),
            'Seed1 mandatory artifacts absent')
    proofs = [file_proof(directory / 'summary.json')]
    for name, expected in artifacts.items():
        target = safe_path(directory / name)
        require(target.resolve().is_relative_to(directory.resolve()), 'Seed1 artifact path escaped')
        proof = file_proof(target)
        require(proof['sha256'] == expected, 'Seed1 artifact hash mismatch')
        proofs.append(proof)
    gate = read_json(directory / 'access_gate.json')
    require((gate.get('variant'), gate.get('training_seed', 0), gate.get('source_hash'), gate.get('contract_hash'))
            == ('G1C', 1, identity['source_hash'], item['contract_hash']), 'Seed1 access gate mismatch')
    manifest_hash = hashlib.sha256(json.dumps(item['dataset_manifest'], sort_keys=True).encode()).hexdigest()
    require(manifest_hash == item['dataset_manifest_hash'] == historical['dataset_manifest_hash'], 'Dataset changed')
    old_rows, old_identity = historical_baseline(historical_csv, 'G1', reference, seed=1)
    require(old_identity == item['baseline_identity'], 'Historical same-seed identity changed')
    baseline_proof = item['baseline_checkpoint_proof']
    require(baseline_proof.get('identity_verified') is True and
            baseline_proof['checkpoint_sha256'] == old_identity['checkpoint_sha256'], 'Baseline proof invalid')
    new = validate_rows(read_csv(directory / 'per_image_results.csv'), reference, identity,
                        item['dataset_manifest'], manifest_hash)
    old = validate_rows(read_csv(directory / 'baseline_per_image_results.csv'), reference, old_identity,
                        item['dataset_manifest'], manifest_hash)
    original = {cell_key(row): row for row in old_rows}
    for row in old:
        require(all(row[field] == original[cell_key(row)][field] for field in
                    ('psnr', 'ssim', 'checkpoint_sha256', 'config_hash', 'source_hash')), 'Baseline scores changed')
    for dataset, family, level in TABLE1_CONDITIONS:
        require(f'panels/{dataset}_{family}_{level:g}_panel.png' in artifacts, 'Seed1 fixed panel missing')
    return {'seed': 1, 'rows': len(new), 'baseline_rows': len(old), 'files': proofs,
            'summary_sha256': sha256_file(directory / 'summary.json'), 'identity': identity}


def verify_recovery():
    require(not AUDIT.exists(), 'This one-attempt recovery has already been used')
    require(git_commit() == PINS['scientific_commit'] and source_hash() == PINS['source_hash']
            and sha256_file(CONFIG) == PINS['config_sha256'], 'Frozen scientific identity changed')
    state = read_json(STATE)
    require(state['identity'] == {k: PINS[k] for k in ('source_hash', 'config_sha256')}, 'Queue identity changed')
    require(state.get('status') == 'failed' and state.get('phase') == 'evaluation'
            and 'Task failed; preserve exit record without retry: evaluation_G1C_seed0' in state.get('reason', ''),
            'This is not the authorized evaluation failure')
    no_live_processes(state)
    paths = load_paths(CONFIG)
    training, seed1_contract, seed1_identity = [], None, None
    for seed in (0, 1, 2):
        opt, checkpoint_path, checkpoint, contract = validated_gate(paths, 'G1C', seed)
        identity = {'method': 'G1C', 'training_seed': seed, 'stage': 'full', 'update': 300000,
                    'checkpoint_sha256': sha256_file(checkpoint_path), 'config_hash': opt['config_hash'],
                    'source_hash': opt['source_hash'], 'git_commit': opt['git_commit'],
                    'measurement_origin': 'new_model_inference'}
        require(opt['source_hash'] == PINS['source_hash'] and opt['git_commit'] == PINS['scientific_commit']
                and opt.get('training_integrity_mode') == STANDARD_MODE, 'Training identity changed')
        training.append(identity)
        if seed == 1:
            seed1_contract, seed1_identity = contract, identity
        del checkpoint
        gc.collect()
    jobs = [state['jobs'][key] for key in KEYS]
    files = [file_proof(STATE), file_proof(RECHECK)]
    for seed in (0, 1):
        job = jobs[seed]
        record = read_json(safe_path(job['exit_file']))
        require(record['command'] == job['command'] and record['returncode'] == (1 if seed == 0 else 0),
                'Evaluation exit record changed')
        files.extend([file_proof(job['exit_file']), file_proof(job['log'])])
    require(ERROR in safe_path(jobs[0]['log']).read_text(encoding='utf-8') and
            safe_path(jobs[0]['log']).read_text(encoding='utf-8').rstrip().endswith(ERROR), 'Unexpected failed evaluation')
    require(jobs[0]['status'] == 'failed' and jobs[1]['status'] in ('running', 'completed')
            and jobs[2]['status'] == 'pending' and not safe_path(jobs[2]['exit_file']).exists(), 'Unexpected evaluation state')
    partial = REPORTS / 'evaluation/G1C/seed0'
    require(not (partial / 'summary.json').exists(), 'Seed0 is already complete or has an unexpected summary')
    require(not (REPORTS / 'evaluation/G1C/seed2').exists(), 'Seed2 evaluation was already started')
    files.extend(tree_proofs(partial))
    recheck = read_json(RECHECK)
    require(all(recheck.get(k) == v for k, v in PINS.items()) and recheck.get('all_three_matched') is True
            and recheck.get('checks_bypassed') is False
            and recheck.get('case') == ['Kodak24', 'kodim18.png', 'speckle', 0.04], 'Failed-case recheck proof invalid')
    command = jobs[0]['command']
    pos = command.index('tools/evaluate_c_only.py')
    original_python = command[pos - 1]
    require(command[command.index('--paths') + 1] == CONFIG_REL, 'Original relative config changed')
    for seed, job in enumerate(jobs):
        require(job['command'][pos - 1] == original_python and job['command'][pos] == 'tools/evaluate_c_only.py'
                and job['command'][job['command'].index('--seed') + 1] == str(seed)
                and job['command'][job['command'].index('--paths') + 1] == CONFIG_REL, 'Evaluation command identity changed')
    reused = verify_seed1(paths, seed1_contract, seed1_identity)
    files.extend(reused['files'])
    return {'state': state, 'files': files, 'training': training, 'reused_seed1': reused,
            'python': original_python, 'config_argument': CONFIG_REL, 'recheck': file_proof(RECHECK)}


def prepare_recovery(evidence):
    require(not AUDIT.exists(), 'Recovery marker already exists')
    for proof in evidence['files']:
        require(file_proof(proof['path']) == proof, 'Evidence changed since verification')
    no_live_processes(evidence['state'])
    state = copy.deepcopy(evidence['state'])
    archive = safe_path(EVIDENCE / 'evaluation_recovery_attempts' / str(time.time_ns()))
    require(not archive.exists(), 'Archive already exists')
    archive.mkdir(parents=True)
    audit = dict(PINS, schema='g1c_evaluation_recovery_v1', status='prepared', attempt=1,
                 created_at=utc_now(), checks_bypassed=False, input_checks_unchanged=True,
                 scientific_source_unchanged=True, seed0_reexecuted_from_start=True, seed1_reused=True,
                 training_restarted=False, archive_directory=str(archive.relative_to(ROOT)),
                 failed_attempt={'reason': state['reason'], 'files': evidence['files']},
                 reused_seed1=evidence['reused_seed1'], training=evidence['training'],
                 failed_case_recheck=evidence['recheck'], original_python=evidence['python'],
                 original_config_argument=evidence['config_argument'],
                 evaluation_commands={key: state['jobs'][key]['command'] for key in KEYS},
                 evaluation_gpu_assignment={'0': 0, '1': 1, '2': 1},
                 placement_change={'job': KEYS[2], 'old': state['jobs'][KEYS[2]].get('preferred_gpu'), 'new': 1},
                 script_sha256=sha256_file(Path(__file__)), archived_files=[])
    atomic_json(AUDIT, audit)  # durable one-shot marker precedes every mutation
    shutil.copyfile(STATE, archive / 'failed_pipeline_state.json')
    job = state['jobs'][KEYS[0]]
    for field, name in (('exit_file', 'seed0_failed.exit.json'), ('log', 'seed0_failed.log')):
        safe_path(job[field]).rename(archive / name)
    failure = REPORTS / 'pipeline_failure.txt'
    if failure.exists():
        safe_path(failure).rename(archive / 'failed_pipeline_trace.txt')
    partial = safe_path(REPORTS / 'evaluation/G1C/seed0')
    tree_proofs(partial)  # reject any symlink introduced since verification
    partial.rename(archive / 'seed0_partial')
    for key in ('process', 'gpu', 'returncode', 'started_at', 'ended_at', 'start', 'end'):
        job.pop(key, None)
    job.update(status='pending', preferred_gpu=0)
    state['jobs'][KEYS[1]].update(status='completed', returncode=0)
    state['jobs'][KEYS[2]].update(preferred_gpu=1)
    state.update(status='running', phase='evaluation', process=process_info(),
                 evaluation_recovery_audit=str(AUDIT.relative_to(ROOT)))
    state.pop('reason', None)
    state.pop('ended_at', None)
    audit['archived_files'] = tree_proofs(archive)
    atomic_json(AUDIT, audit)
    atomic_json(STATE, state)
    return state


class RecoveryQueue(G1CStandardQueue):
    def __init__(self, evidence):
        super().__init__(evidence['config_argument'])
        self.python = evidence['python']
        self.evidence = evidence

    def record(self):
        audit = read_json(AUDIT)
        audit.update(updated_at=utc_now(), status=self.state['status'], process=process_info())
        if self.state['status'] == 'completed':
            audit['completed_at'] = utc_now()
            audit['final_summary'] = file_proof(self.reports / 'final/summary.json')
        if self.state['status'] == 'failed':
            audit['reason'] = self.state['reason']
        atomic_json(AUDIT, audit)
        super().record()  # completed pipeline state is the final publication barrier

    def run(self):
        with RunLock(self.output / 'pipeline.lock'):
            try:
                self.state = prepare_recovery(self.evidence)
                self.record()
                self.evaluate()  # frozen evaluation and summarization; never train/preflight
                result = read_json(self.reports / 'final/summary.json')
                require(result.get('completed') is True and result.get('new_measurement_rows') == 9450
                        and result.get('reused_baseline_rows') == 9450 and result.get('paired_rows') == 9450
                        and result.get('training_seeds') == [0, 1, 2], 'Final summary incomplete')
                self.state.update(status='completed', phase='completed', ended_at=utc_now())
                self.record()
            except BaseException as error:
                if AUDIT.exists():
                    self.state.update(status='failed', phase='evaluation', reason=f'{type(error).__name__}: {error}',
                                      ended_at=utc_now())
                    self.record()
                    (EVIDENCE / 'evaluation_recovery_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    evidence = verify_recovery()
    if args.check_only:
        print(json.dumps({'accepted': True, **PINS, 'training': evidence['training'],
                          'reused_seed1': evidence['reused_seed1']['summary_sha256'], 'checks_bypassed': False}))
        return
    RecoveryQueue(evidence).run()


if __name__ == '__main__':
    main()
