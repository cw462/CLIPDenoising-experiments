"""One attempt in a new NumPy environment; never starts formal training."""
import os
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_VISIBLE_DEVICES="0")
from pathlib import Path
import hashlib, json, subprocess, sys, traceback

ROOT = Path('/home/imss/cw/CLIPDenoising_G1C_Seeds_20261021')
ENV = Path('/home/imss/cw/CLIPDenoising_G1C_numpy_isolation_20261021')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
import torch
import yaml
from tools.experiment_common import atomic_json, source_hash, sha256_file, process_info, utc_now, RunLock

PINS = {'scientific_commit':'af56111ad0fe5d9355ff89b16db7568b0a918682',
        'source_hash':'51c704dfa85f9cbe938e13984cf063e15c92eaebdf358c5b76cdaad994e2ce54',
        'base_config_sha256':'28b5eb5506852ea40cceb5b111a42bbf83c7d31699b9431d04a75545f6d3f950',
        'environment_manifest_sha256':'49694f24631539ba926a75876e948827e8d070339057b1c97872bd8c7155c25b'}
base = ROOT / 'configs/plan_g1c_multiseed_server.yaml'
envmanifest = ENV / 'isolation_evidence/environment_manifest_r2.json'
folder = ROOT / 'reproducibility'
folder.mkdir(exist_ok=True)
state_path = folder / 'isolated_validation_r2_state.json'
launch_path = folder / 'isolated_validation_r2_launch.json'
candidate = folder / 'candidate_numpy_isolation_r2.yaml'
reports = ROOT / 'reports/g1c_numpy_candidate_r2'
reports.mkdir(parents=True, exist_ok=True)

def main():
    with RunLock(folder / 'isolated_validation_r2.lock'):
        if state_path.exists() or launch_path.exists() or candidate.exists():
            raise RuntimeError('Existing attempt must be preserved; no automatic retry')
        assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == PINS['scientific_commit']
        assert source_hash() == PINS['source_hash']
        assert sha256_file(base) == PINS['base_config_sha256']
        assert sha256_file(envmanifest) == PINS['environment_manifest_sha256']
        assert np.__version__ == '1.26.4' and Path(np.__file__).resolve().is_relative_to(ENV.resolve())
        assert torch.__version__.split('+')[0] == '2.2.1' and torch.version.cuda == '12.1'
        config = yaml.safe_load(base.read_text())
        config.update(output_root='runs/g1c_numpy_candidate_r2', reports_dir='reports/g1c_numpy_candidate_r2')
        candidate.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        commands=[]
        for seed in (0,1,2):
            commands.append({'phase':'contracts','seed':seed,'command':[str(ENV/'bin/python'),
                'tools/check_c_full_contract.py','--variant','G1C','--seed',str(seed),'--paths',str(candidate),
                '--legacy-root','/home/imss/cw/CLIPDenoising_20261020',
                '--report',str(reports/f'actual_G1C_seed{seed}_contract.json')]})
        commands.append({'phase':'validation','seed':0,'command':[str(ENV/'bin/python'),
            'tools/run_c_only_experiment.py','--variant','G1C','--stage','full','--seed','0',
            '--paths',str(candidate),'--tag','numpy_isolation_validation10000','--stop-after','10000']})
        identity={**PINS,'candidate_config_sha256':sha256_file(candidate),
            'candidate_config_path':str(candidate),'environment_manifest_path':str(envmanifest),
            'starter_sha256':sha256_file(__file__),'python':str(ENV/'bin/python')}
        atomic_json(launch_path,{'identity':identity,'created_at':utc_now(),'commands':commands,
            'validation_updates':10000,'formal_training_started':False,
            'interpretation':'Controlled NumPy build-isolation candidate, not an established root-cause repair'})
        state={'identity':identity,'process':process_info(),'status':'created','phase':'created',
               'started_at':utc_now(),'exitcodes':[],'formal_training_started':False}
        atomic_json(state_path,state)
        try:
            for item in commands:
                state.update(status=item['phase'],phase=item['phase'],seed=item['seed'],updated_at=utc_now())
                atomic_json(state_path,state)
                log=reports/f"{item['phase']}_seed{item['seed']}.log"
                with log.open('a') as stream:
                    code=subprocess.call(item['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
                state['exitcodes'].append({'phase':item['phase'],'seed':item['seed'],'returncode':code})
                if code:
                    raise RuntimeError(f"{item['phase']} seed{item['seed']} failed with exit {code}")
                if item['phase']=='contracts':
                    report=json.loads((reports/f"actual_G1C_seed{item['seed']}_contract.json").read_text())
                    if report.get('accepted') is not True:
                        raise RuntimeError('GPU contract did not accept the candidate')
            state.update(status='passed',phase='validation_processes_finished',ended_at=utc_now(),
                         completion_audit_required=True)
            atomic_json(state_path,state)
        except BaseException as error:
            state.update(status='failed',reason=f'{type(error).__name__}: {error}',ended_at=utc_now())
            atomic_json(state_path,state)
            (reports/'validation_failure.txt').write_text(traceback.format_exc())
            raise

if __name__ == '__main__':
    main()
