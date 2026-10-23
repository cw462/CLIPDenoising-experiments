import os,sys,time,shutil,subprocess,traceback
from pathlib import Path
ROOT=Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from tools.experiment_common import RunLock,atomic_json,read_json,process_info,utc_now
from tools.run_g1c_standard_pipeline import G1CStandardQueue
q=G1CStandardQueue('configs/plan_g1c_standard_server.yaml')
with RunLock(q.output/'pipeline.lock'):
 archive=ROOT/'reproducibility/g1c_standard'/('seed0_direct_'+str(time.time_ns()))
 archive.mkdir();shutil.copy2(q.state_path,archive/'pipeline_state_before.json')
 job=q.state['jobs']['evaluation_G1C_seed0']
 if read_json(job['exit_file'])['returncode']!=1:raise RuntimeError('Unexpected seed0 exit')
 for field in ('exit_file','log'):
  p=Path(job[field]);p.rename(archive/p.name)
 p=q.reports/'evaluation/G1C/seed0'
 if p.exists():p.rename(archive/'seed0_partial')
 cmd=job['command'];cmd[cmd.index('reproducibility/g1c_standard/evaluate_direct.py')]='reproducibility/g1c_standard/evaluate_direct.py'
 for key in ('process','gpu','returncode','started_at','ended_at','start','end'):job.pop(key,None)
 job.update(status='pending',preferred_gpu=0)
 q.state.update(status='running',phase='evaluation',process=process_info())
 q.state.pop('reason',None);q.record()
 try:
  q.execute(['evaluation_G1C_seed0','evaluation_G1C_seed2'])
  q.state['phase']='summary';q.record()
  subprocess.run([sys.executable,'reproducibility/g1c_standard/summarize_direct.py'],check=True)
  q.state.update(status='completed',phase='completed',ended_at=utc_now())
 except BaseException as e:
  q.state.update(status='failed',reason=str(e),ended_at=utc_now())
  (archive/'failure.txt').write_text(traceback.format_exc())
  raise
 finally:q.record()
