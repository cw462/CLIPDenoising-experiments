import os,sys,time,shutil,subprocess,traceback
from pathlib import Path
ROOT=Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from tools.experiment_common import RunLock,atomic_json,read_json,process_info,utc_now
from tools.run_g1c_standard_pipeline import G1CStandardQueue
q=G1CStandardQueue('configs/plan_g1c_standard_server.yaml')
archive=ROOT/'reproducibility/g1c_standard'/('direct_no_noise_check_'+str(time.time_ns()))
def reset(seed,gpu):
 job=q.state['jobs'][f'evaluation_G1C_seed{seed}']
 for field in ('exit_file','log'):
  p=Path(job[field])
  if p.exists():p.rename(archive/p.name)
 p=q.reports/'evaluation/G1C'/f'seed{seed}'
 if p.exists():p.rename(archive/f'seed{seed}_partial')
 cmd=job['command'];cmd[cmd.index('tools/evaluate_c_only.py')]='reproducibility/g1c_standard/evaluate_direct.py'
 for key in ('process','gpu','returncode','started_at','ended_at','start','end'):job.pop(key,None)
 job.update(status='pending',preferred_gpu=gpu)
with RunLock(q.output/'pipeline.lock'):
 archive.mkdir()
 shutil.copy2(q.state_path,archive/'pipeline_state_before.json')
 atomic_json(archive/'authorization.json',{'authorization':'User requested evaluation without checks','noise_identity_check':False,'training_restarted':False,'at':utc_now()})
 reset(2,1)
 q.state.update(status='running',phase='evaluation',process=process_info(),direct_evaluation_audit=str(archive/'authorization.json'))
 q.state.pop('reason',None);q.record()
 try:
  q.execute(['evaluation_G1C_seed2'])
  j=q.state['jobs']['evaluation_G1C_seed0']
  while not Path(j['exit_file']).exists():time.sleep(5)
  if read_json(j['exit_file'])['returncode']!=0:reset(0,0)
  q.execute(['evaluation_G1C_seed0'])
  q.state['phase']='summary';q.record()
  subprocess.run([sys.executable,'reproducibility/g1c_standard/summarize_direct.py'],check=True)
  q.state.update(status='completed',phase='completed',ended_at=utc_now())
 except BaseException as e:
  q.state.update(status='failed',reason=str(e),ended_at=utc_now())
  (archive/'failure.txt').write_text(traceback.format_exc())
  raise
 finally:q.record()
