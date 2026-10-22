"""One authorized seed2 GPU1 -> GPU0 checkpoint migration; frozen trainer unchanged."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
sys.path.insert(0, str(ROOT))
import psutil
from tools.experiment_common import (RunLock, atomic_json, git_commit, load_paths,
    process_alive, process_info, read_json, sha256_file, source_hash, utc_now)
from tools.run_g1c_standard import configuration
from tools import run_experiment as legacy
from tools.run_g1c_standard_pipeline import G1CStandardQueue

CONFIG = ROOT / 'configs/plan_g1c_standard_server.yaml'
OUTPUT = ROOT / 'runs/g1c_standard'
STATE = OUTPUT / 'pipeline_state.json'
EVIDENCE = ROOT / 'reproducibility/g1c_standard'
AUDIT = EVIDENCE / 'migration_audit.json'
HEARTBEAT = EVIDENCE / 'migration_status.json'
PLAN = EVIDENCE / 'migration_plan.json'
KEY = 'full_G1C_seed2'
PINS = {'scientific_commit': 'c02ebedd103af716afc3ad29491b3886e6ba37ff',
        'source_hash': '86d821ff44713da6918b73f92ed7b9e2a7b3d83f53082eef9c82f36a654a534d',
        'config_sha256': '9ba8dcdea71c06d9591c865bd85644b81a2c04770fcc0aabfc6bc820bdc22eb8'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity_check():
    require(git_commit() == PINS['scientific_commit'], 'Frozen scientific commit changed')
    require(source_hash() == PINS['source_hash'], 'Frozen scientific source changed')
    require(sha256_file(CONFIG) == PINS['config_sha256'], 'Frozen config changed')
    state = read_json(STATE)
    require(state['identity'] == {k: PINS[k] for k in ('source_hash', 'config_sha256')}, 'Pipeline identity changed')
    require(state['protocol']['training_integrity_mode'] == 'standard_user_requested_v1'
            and state['protocol']['per_update_input_consistency_checks'] is False, 'Wrong training mode')
    return state


def seed_run(seed):
    return OUTPUT / 'full/G1C' / f'seed{seed}'


def status(seed):
    return read_json(seed_run(seed) / 'status.json')


def verified_process(info, script, seed=None):
    require(process_alive(info), 'Expected owned process is no longer alive')
    proc = psutil.Process(info['pid'])
    require(abs(proc.create_time() - info['create_time']) < 0.1, 'PID identity changed')
    require(Path(proc.cwd()).resolve() == ROOT, 'Process belongs to another worktree')
    command = proc.cmdline()
    require(script in command, 'Process command differs from expected role')
    if seed is not None:
        pos = command.index('--seed')
        require(command[pos + 1] == str(seed), 'Wrong training seed process')
        require(proc.environ().get('CUDA_VISIBLE_DEVICES') == '1', 'Seed2 is not on its original GPU1')
    require(proc.pid != os.getpid(), 'Cannot signal self')
    return proc


def stop_verified(proc):
    report = {'pid': proc.pid, 'created_at': utc_now(), 'observations': [], 'status': 'checking'}
    path = EVIDENCE / f'child_stop_observations_{proc.pid}_{time.time_ns()}.json'
    try:
        first = proc_observation(proc)
        report['observations'].append(first)
        proc.send_signal(signal.SIGSTOP)
        report['status'] = 'stop_requested'
        end = time.monotonic() + 10
        while time.monotonic() < end:
            current = proc_observation(proc)
            report['observations'].append(current)
            require(current['starttime_ticks'] == first['starttime_ticks'], 'PID reused during child STOP')
            if current['stat_state'] == current['status_state'] == 'T' and current['tracer_pid'] == 0:
                report['status'] = 'stopped'
                return
            time.sleep(0.05)
        raise RuntimeError('Owned child process did not enter stopped state')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(path, report)


def proc_observation(proc):
    """Fresh kernel observations; never accept tracing-stop or a reused PID."""
    fresh = psutil.Process(proc.pid)
    require(abs(fresh.create_time() - proc.create_time()) < 0.1, 'PID identity changed')
    stat = Path(f'/proc/{proc.pid}/stat').read_text()
    close = stat.rfind(')')
    require(close >= 0 and int(stat.split(' ', 1)[0]) == proc.pid, 'Malformed /proc stat identity')
    fields = stat[close + 1:].split()
    require(len(fields) > 19, 'Incomplete /proc stat')
    status_fields = {}
    for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            status_fields[key] = value.strip()
    require(int(status_fields['Pid']) == proc.pid, 'Wrong /proc status PID')
    return {'pid': proc.pid, 'create_time': fresh.create_time(), 'stat_state': fields[0],
            'status_state': status_fields['State'].split()[0], 'starttime_ticks': int(fields[19]),
            'tracer_pid': int(status_fields['TracerPid']), 'psutil_status': fresh.status(),
            'observed_at': utc_now()}


def continue_if_same_process(proc):
    """Clear our STOP even if it remained pending or status reads were wrong."""
    try:
        fresh = psutil.Process(proc.pid)
        if abs(fresh.create_time() - proc.create_time()) >= 0.1 or not fresh.is_running():
            return False
        proc.send_signal(signal.SIGCONT)
        return True
    except psutil.NoSuchProcess:
        return False


def wait_not_alive(info, timeout):
    """Non-child zombies have exited; their parent need not have reaped them."""
    end = time.monotonic() + timeout
    while process_alive(info):
        require(time.monotonic() < end, f'Owned process did not exit: {info["pid"]}')
        time.sleep(0.05)


def terminate_coordinator(info):
    proc = verified_process(info, 'tools/run_g1c_standard_pipeline.py')
    proc.send_signal(signal.SIGTERM)
    try:
        wait_not_alive(info, 15)
    except RuntimeError:
        if process_alive(info):
            # Recheck ownership before escalation; never signal a process group.
            verified_process(info, 'tools/run_g1c_standard_pipeline.py').send_signal(signal.SIGKILL)
            wait_not_alive(info, 15)


def gpu0_available():
    rows = list(csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()))
    zero = next(row for row in rows if int(row[0].strip()) == 0)
    if float(zero[2].strip()) < 10240:
        return False
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    return all(row[0].strip() != zero[1].strip() for row in csv.reader(applications.splitlines()) if row)


def last_logged_update(run):
    path = run / 'train.jsonl'
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        data = stream.read()
    lines = data.split(b'\n')[:-1]  # A partial line is not a completed log record.
    for line in reversed(lines):
        try:
            row = json.loads(line)
            return int(row['update'])
        except (ValueError, KeyError):
            continue
    raise RuntimeError('No complete training update found in the tail')


def seed0_complete(state, validate_file=False):
    job = state['jobs']['full_G1C_seed0']
    s = status(0)
    if s.get('status') != 'completed' or s.get('update') != 300000:
        return False
    exit_path = Path(job['exit_file'])
    if not exit_path.exists() or process_alive(job.get('process')):
        return False
    record = read_json(exit_path)
    require(record.get('returncode') == 0 and record.get('command') == job['command'], 'Seed0 did not exit normally')
    summary = read_json(seed_run(0) / 'summary.json')
    latest = read_json(seed_run(0) / 'checkpoints/latest.json')
    require(summary.get('completed') is True and summary.get('updates') == latest.get('update') == 300000,
            'Seed0 completion metadata is incomplete')
    require(summary['checkpoint_sha256'] == latest['sha256'], 'Seed0 final identity mismatch')
    if validate_file:
        require(sha256_file(seed_run(0) / 'checkpoints/final.pth') == latest['sha256'], 'Seed0 final file differs')
    return True


def ensure_active(state):
    require(state.get('status') == 'running' and state.get('phase') == 'full300k', 'Training queue is not active')
    for seed in (0, 1, 2):
        s = status(seed)
        require(s.get('status') not in ('failed', 'blocked', 'interrupted'), f'Seed{seed} has a real training failure')


def heartbeat(phase, **fields):
    atomic_json(HEARTBEAT, dict(phase=phase, updated_at=utc_now(), process=process_info(), **fields))


def base_audit():
    return dict(schema='g1c_seed_gpu_migration_v1', status='waiting', job_key=KEY,
                seed=2, from_gpu=1, to_gpu=0, **PINS,
                authorization='User requested checkpoint migration of seed2 after seed0 completes',
                controller_sha256=sha256_file(__file__), armed_at=utc_now())


def check_expected_interrupt(run, exit_record, child_pid):
    s = read_json(run / 'status.json')
    require(exit_record.get('child_pid') == child_pid, 'Exit record belongs to another child')
    require(exit_record.get('returncode') in (-signal.SIGINT, 128 + signal.SIGINT), 'Child did not exit by expected SIGINT')
    require(s.get('status') == 'failed' and str(s.get('reason', '')).startswith('KeyboardInterrupt:'),
            'A real training error cannot be relabelled as a planned migration')
    return s


class MigrationQueue(G1CStandardQueue):
    def __init__(self):
        # Queue job identity includes the exact argv, including a relative
        # --paths argument and the original interpreter spelling.
        previous = read_json(STATE)['jobs'][KEY]['command']
        script_index = previous.index('tools/run_g1c_standard.py')
        require(script_index > 0 and previous.count('--paths') == 1,
                'Cannot recover the original training command')
        original_python = previous[script_index - 1]
        original_config = previous[previous.index('--paths') + 1]
        require(Path(original_config).resolve() == CONFIG.resolve(),
                'Original training command references another config')
        super().__init__(original_config)
        self.python = original_python
        self.state['protocol'].update(
            gpu_assignment_after_migration={'0': 0, '1': 1, '2': 0},
            concurrency='After seed0 completion: independent seed1 on GPU1 and resumed seed2 on GPU0',
            migration_audit=str(AUDIT.relative_to(ROOT)),
        )

    def add_seed_job(self, key, command, seed, kind='train', run_dir=None):
        result = self.add(key, command, kind, run_dir)
        preferred = 1 if seed == 1 else 0
        job = self.state['jobs'][key]
        require(job.get('preferred_gpu', preferred) == preferred, 'Unexpected GPU reassignment beyond authorized seed2')
        job['preferred_gpu'] = preferred
        return result

    def record(self):
        audit = read_json(AUDIT)
        if audit.get('status') == 'in_progress':
            s = status(2)
            manifest_path = seed_run(2) / 'manifest.json'
            if (s.get('status') in ('running', 'completed') and s.get('update', 0) > audit['resume_from_update']
                    and s.get('process', {}).get('pid') != audit['old_child_process']['pid']
                    and manifest_path.exists()):
                manifest = read_json(manifest_path)
                require(manifest.get('cuda_visible_devices') == '0', 'Resumed seed2 is not on GPU0')
                require(s.get('config_hash') == audit['training_config_hash']
                        and s.get('source_hash') == PINS['source_hash'], 'Resumed training identity changed')
                require(s.get('recovery') == 'complete params/optimizer/scheduler/RNG/sampler state restored',
                        'Full optimizer/RNG/sampler recovery not recorded')
                audit.update(status='completed', completed_at=utc_now(), resume_verified_at=utc_now(),
                             resumed_process=s['process'], first_observed_resumed_update=s['update'])
                atomic_json(AUDIT, audit)
        heartbeat('continued_training_and_evaluation', pipeline_status=self.state['status'],
                  pipeline_phase=self.state['phase'], migration_status=audit['status'])
        # The completion publisher may observe pipeline_state immediately:
        # commit migration evidence first, then publish pipeline completion.
        super().record()


def migrate(audit):
    state = identity_check()
    ensure_active(state)
    require(seed0_complete(state, validate_file=True), 'Seed0 must finish before migration')
    require(gpu0_available(), 'GPU0 is not yet free')
    run = seed_run(2)
    initial_latest = read_json(run / 'checkpoints/latest.json')
    audit.update(status='waiting', phase='waiting_next_seed2_checkpoint', seed0_completed_at=status(0)['ended_at'],
                 previous_checkpoint_update=initial_latest['update'])
    atomic_json(AUDIT, audit)
    # Keep all training and the old scheduler running until a fresh regular
    # checkpoint is fully committed. No live-trainer source modification.
    deadline = time.monotonic() + 3600
    while True:
        state = read_json(STATE)
        ensure_active(state)
        if status(2).get('status') == 'completed':
            audit.update(status='cancelled', reason='Seed2 completed before migration was needed', completed_at=utc_now())
            atomic_json(AUDIT, audit)
            return
        latest = read_json(run / 'checkpoints/latest.json')
        if latest['update'] > initial_latest['update']:
            break
        require(time.monotonic() < deadline, 'No new seed2 checkpoint within one hour')
        heartbeat('waiting_next_seed2_checkpoint', checkpoint_update=latest['update'])
        time.sleep(1)

    coordinator = child = None
    child_stopped = old_coordinator_terminated = False
    old_coordinator_info = None
    try:
        state = identity_check()
        ensure_active(state)
        require(gpu0_available(), 'GPU0 must still be free before scheduler handoff')
        coordinator = verified_process(state['process'], 'tools/run_g1c_standard_pipeline.py')
        old_coordinator_info = dict(state['process'])
        require(seed0_complete(state), 'Seed0 completion changed')
        before = status(2)
        require(before['status'] == 'running', 'Seed2 is no longer running')
        old_job = dict(state['jobs'][KEY])
        wrapper = verified_process(old_job['process'], 'tools/ac_job_runner.py')
        child = verified_process(before['process'], 'tools/run_g1c_standard.py', 2)
        require(child.ppid() == wrapper.pid, 'Seed2 is not the child of its recorded wrapper')
        archive = EVIDENCE / 'migration_attempts' / str(time.time_ns())
        archive.mkdir(parents=True)
        for name in ('status.json', 'manifest.json', 'effective_config.json'):
            shutil.copyfile(run / name, archive / ('before_' + name))
        atomic_json(archive / 'before_pipeline_state.json', state)
        audit.update(status='in_progress', phase='terminating_old_coordinator', started_at=utc_now(),
                     old_child_process=before['process'], old_wrapper_process=old_job['process'],
                     old_coordinator_process=old_coordinator_info,
                     archive_directory=str(archive.relative_to(ROOT)),
                     coordinator_handoff='TERM owned coordinator; tmux automatically resumes STOPped pane children')
        atomic_json(AUDIT, audit)
        # tmux resumes a stopped pane child, so STOP cannot quiesce this
        # coordinator. Its separately-sessioned training wrappers stay alive.
        terminate_coordinator(old_coordinator_info)
        old_coordinator_terminated = True
        state = read_json(STATE)
        ensure_active(state)
        require(state['jobs'][KEY]['command'] == old_job['command']
                and state['jobs'][KEY]['process'] == old_job['process'], 'Seed2 job changed during handoff')
        child = verified_process(before['process'], 'tools/run_g1c_standard.py', 2)
        child_stopped = True
        stop_verified(child)
        latest = read_json(run / 'checkpoints/latest.json')
        checkpoint = Path(latest['path'])
        require(checkpoint.resolve().parent == (run / 'checkpoints').resolve(), 'Checkpoint escaped seed2 directory')
        require(0 < latest['update'] < 300000 and latest['update'] % 5000 == 0, 'Not a normal complete checkpoint')
        require(sha256_file(checkpoint) == latest['sha256'], 'Migration checkpoint digest mismatch')
        opt, _ = configuration(2, load_paths(CONFIG))
        payload = legacy.validate_checkpoint(checkpoint, opt, latest['update'])
        require(payload.get('pfa_rng') is None, 'Wrong legacy PFA state')
        del payload
        os.link(checkpoint, archive / checkpoint.name)
        audit.update(status='in_progress', phase='stopping_seed2_after_checkpoint', started_at=utc_now(),
                     checkpoint=latest, resume_from_update=latest['update'], training_config_hash=opt['config_hash'],
                     old_child_process=before['process'], old_wrapper_process=old_job['process'],
                     old_coordinator_process=state['process'], archive_directory=str(archive.relative_to(ROOT)))
        atomic_json(AUDIT, audit)
        # The only expected failure here is our planned KeyboardInterrupt.
        child.send_signal(signal.SIGINT)
        child.send_signal(signal.SIGCONT)
        child_stopped = False
        wait_not_alive(before['process'], 60)
        wait_not_alive(old_job['process'], 60)
        exit_path = Path(old_job['exit_file'])
        require(exit_path.exists(), 'Interrupted wrapper has no durable exit record')
        exit_record = read_json(exit_path)
        require(exit_record.get('command') == old_job['command'], 'Interrupted command identity mismatch')
        stopped = check_expected_interrupt(run, exit_record, child.pid)
        last_update = last_logged_update(run)
        tail = last_update - latest['update']
        require(0 <= tail <= 5000, 'Unexpected checkpoint rollback interval')
        require(status(1).get('status') in ('running', 'completed'), 'Seed1 failed during migration')
        shutil.copyfile(run / 'status.json', archive / 'interrupted_status.json')
        if (run / 'failure.txt').exists():
            shutil.move(str(run / 'failure.txt'), str(archive / 'planned_interrupt.txt'))
        shutil.move(str(exit_path), str(archive / 'original_exit.json'))
        audit.update(last_successful_update_before_stop=last_update, recomputed_tail_updates=tail,
                     tail_update_note='Count of complete logged updates discarded after the saved checkpoint; the trainer archives the original log when resuming',
                     interrupted_status_sha256=sha256_file(archive / 'interrupted_status.json'),
                     original_exit_sha256=sha256_file(archive / 'original_exit.json'))
        atomic_json(AUDIT, audit)

        require(not process_alive(old_coordinator_info), 'Old scheduler is still alive')
        state['jobs'][KEY] = {**old_job, 'status': 'pending', 'preferred_gpu': 0, 'gpu': None,
                              'process': None, 'update': latest['update'],
                              'migration_history': [{'archive': str(archive.relative_to(ROOT)),
                                                     'from_gpu': 1, 'to_gpu': 0, 'checkpoint_update': latest['update']} ]}
        for field in ('returncode', 'ended_at', 'started_at'):
            state['jobs'][KEY].pop(field, None)
        state.update(status='running', phase='full300k', process=process_info())
        state['protocol'].update(gpu_assignment_after_migration={'0': 0, '1': 1, '2': 0},
                                 migration_audit=str(AUDIT.relative_to(ROOT)))
        paused = dict(stopped, status='paused_at_checkpoint', update=latest['update'],
                      checkpoint_update=latest['update'], checkpoint=str(checkpoint),
                      reason='Planned GPU1 to GPU0 migration; full checkpoint will be restored')
        atomic_json(run / 'status.json', paused)
        atomic_json(STATE, state)
        audit.update(phase='restarting_seed2_on_gpu0', scheduler_taken_over_at=utc_now())
        atomic_json(AUDIT, audit)
        MigrationQueue().run()
    except BaseException as error:
        # MigrationQueue.record() owns the durable completion transition;
        # this function's in-memory audit may still say in_progress afterwards.
        audit = read_json(AUDIT)
        reason = f'{type(error).__name__}: {error}'
        if audit.get('status') == 'completed':
            audit.update(adopted_pipeline_status='failed', adopted_pipeline_failed_at=utc_now(),
                         pipeline_failure_reason=reason)
        else:
            audit.update(status='failed', phase='migration_or_adopted_pipeline_failed', failed_at=utc_now(),
                         reason=reason)
        atomic_json(AUDIT, audit)
        if old_coordinator_terminated or (old_coordinator_info is not None and not process_alive(old_coordinator_info)):
            failed = read_json(STATE)
            failed.update(status='failed', reason=reason, ended_at=utc_now())
            atomic_json(STATE, failed)
        raise
    finally:
        active_error = sys.exc_info()[1]
        for proc, stopped in ((child, child_stopped),):
            if proc is not None and stopped:
                try:
                    continue_if_same_process(proc)
                except Exception as cleanup_error:
                    current = read_json(AUDIT)
                    current['cleanup_failure'] = {'pid': proc.pid, 'recorded_at': utc_now(),
                        'reason': f'{type(cleanup_error).__name__}: {cleanup_error}',
                        'resumed': False}
                    atomic_json(AUDIT, current)
                    if active_error is None:
                        raise


def termination_requested(signum, frame):
    # SIGKILL/host failure cannot run cleanup; ordinary termination must release
    # any processes temporarily stopped by this controller.
    raise InterruptedError(f'Migration controller received signal {signum}')


def prepare_single_retry(check_only=False):
    """One repair retry, only for the failed pre-child coordinator STOP."""
    marker = EVIDENCE / 'migration_retry_once.json'
    require(not marker.exists(), 'The one authorized coordinator-handoff retry was already used')
    previous = read_json(AUDIT)
    previous_status = read_json(HEARTBEAT)
    require(previous.get('status') == 'failed' and previous.get('attempt', 1) == 1,
            'Only the original failed attempt may be retried')
    require(previous.get('reason') == 'RuntimeError: Owned process did not enter stopped state',
            'Retry is restricted to the original coordinator STOP timeout')
    require(previous.get('job_key') == KEY and previous.get('seed') == 2,
            'Failed audit belongs to a different job')
    require(all(previous.get(key) == value for key, value in PINS.items()), 'Failed attempt scientific pins differ')
    require(not any(key in previous for key in ('old_child_process', 'old_wrapper_process',
            'started_at', 'resume_from_update', 'archive_directory', 'scheduler_taken_over_at')),
            'Prior attempt may have touched seed2 or taken over its coordinator')
    require(previous_status.get('phase') == 'failed'
            and 'stop_verified(coordinator)' in previous_status.get('traceback', '')
            and 'stop_verified(child)' not in previous_status.get('traceback', ''),
            'Failure traceback does not establish a pre-child coordinator STOP failure')
    require(previous_status.get('process') and not process_alive(previous_status['process']),
            'Previous migration controller is still alive or has no identity')
    state = identity_check()
    ensure_active(state)
    job = state['jobs'][KEY]
    current = status(2)
    require(job.get('status') == current.get('status') == 'running'
            and job.get('gpu') == job.get('preferred_gpu') == 1,
            'Seed2 is no longer the original running GPU1 job')
    coordinator = verified_process(state['process'], 'tools/run_g1c_standard_pipeline.py')
    wrapper = verified_process(job['process'], 'tools/ac_job_runner.py')
    child = verified_process(current['process'], 'tools/run_g1c_standard.py', 2)
    require(child.ppid() == wrapper.pid, 'Seed2 wrapper identity changed')
    armed = datetime.fromisoformat(previous['armed_at'].replace('Z', '+00:00')).timestamp()
    require(all(proc.create_time() < armed for proc in (coordinator, wrapper, child)),
            'An original training/coordinator process was replaced after migration was armed')
    require(not (seed_run(2) / 'failure.txt').exists(), 'Seed2 has an unarchived training failure')
    latest = read_json(seed_run(2) / 'checkpoints/latest.json')
    checkpoint = Path(latest['path'])
    require(checkpoint.resolve().parent == (seed_run(2) / 'checkpoints').resolve(), 'Wrong seed2 checkpoint path')
    require(sha256_file(checkpoint) == latest['sha256'], 'Seed2 checkpoint digest differs')
    opt, _ = configuration(2, load_paths(CONFIG))
    require(current.get('config_hash') == opt['config_hash']
            and current.get('source_hash') == PINS['source_hash'], 'Seed2 scientific identity changed')
    legacy.validate_checkpoint(checkpoint, opt, latest['update'])
    result = {'reason': 'One repaired retry: the original coordinator STOP timed out before seed2 was touched; '
                       'tmux resumes stopped pane children, so this attempt terminates only the owned coordinator.',
              'previous_audit_sha256': sha256_file(AUDIT), 'seed2_touched_by_previous_attempt': False,
              'verified_original_child_process': current['process'],
              'verified_original_wrapper_process': job['process'],
              'verified_original_coordinator_process': state['process'], 'verified_at': utc_now(), 'files': []}
    if check_only:
        return result
    archive = EVIDENCE / 'migration_controller_attempts' / str(time.time_ns())
    archive.mkdir(parents=True)
    for original, name in ((AUDIT, 'failed_attempt1_audit.json'), (PLAN, 'failed_attempt1_plan.json'),
                           (HEARTBEAT, 'failed_attempt1_status.json')):
        target = archive / name
        shutil.copyfile(original, target)
        result['files'].append({'path': str(target.relative_to(ROOT)), 'sha256': sha256_file(target)})
    result['archive_directory'] = str(archive.relative_to(ROOT))
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(dict(attempt=2, retry_of=result, controller_sha256=sha256_file(__file__)), stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--retry-coordinator-stop-once', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    state = identity_check()
    if args.check_only:
        ensure_active(state)
        retry = prepare_single_retry(check_only=True) if args.retry_coordinator_stop_once else None
        print(json.dumps({'ready': True, 'pins': PINS, 'pipeline_status': state['status'],
                          'source_unchanged': True, 'retry_validation': retry,
                          'trigger': 'seed0 completes full300k; next seed2 checkpoint'}))
        return
    signal.signal(signal.SIGTERM, termination_requested)
    with RunLock(EVIDENCE / 'migration_controller.lock'):
        retry = prepare_single_retry() if args.retry_coordinator_stop_once else None
        if retry is None:
            require(not AUDIT.exists(), 'Migration already armed; inspect its existing audit instead of duplicating it')
        audit = base_audit()
        audit['attempt'] = 2 if retry is not None else 1
        if retry is not None:
            audit['retry_of'] = retry
        atomic_json(PLAN, {**audit, 'trigger': 'seed0 completes full300k and GPU0 is free',
                           'checkpoint_policy': 'next committed regular seed2 checkpoint; preserve and resume complete state',
                           'ordinary_training_checks_unchanged': True,
                           'followup': 'same frozen evaluation and three-seed report pipeline'})
        atomic_json(AUDIT, audit)
        deadline = time.monotonic() + 48 * 3600
        try:
            while True:
                state = read_json(STATE)
                ensure_active(state)
                heartbeat('waiting_seed0_completion', seed0_update=status(0)['update'], seed2_update=status(2)['update'])
                if seed0_complete(state) and gpu0_available():
                    migrate(audit)
                    return
                require(time.monotonic() < deadline, 'Migration trigger deadline exceeded')
                time.sleep(15)
        except BaseException as error:
            current = read_json(AUDIT)
            if current.get('status') not in ('completed', 'failed'):
                current.update(status='failed', failed_at=utc_now(), reason=f'{type(error).__name__}: {error}')
                atomic_json(AUDIT, current)
            heartbeat('failed', reason=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
