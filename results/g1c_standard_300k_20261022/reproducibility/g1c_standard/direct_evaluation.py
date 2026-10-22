import os,sys,json,time,shutil,traceback
from pathlib import Path
ROOT=Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from tools.experiment_common import RunLock,atomic_json,read_json,process_alive,process_info,utc_now
from tools.run_g1c_standard_pipeline import G1CStandardQueue
q=G1CStandardQueue('configs/plan_g1c_standard_server.yaml')
with RunLock(q.output/'pipeline.lock'):
    if q.state['status']!='failed': raise RuntimeError('Queue is not in the expected stopped state')
    if any(process_alive(j.get('process')) for j in q.state['jobs'].values()): raise RuntimeError('An existing job is still live')
    archive=ROOT/'reproducibility/g1c_standard'/('direct_evaluation_'+str(time.time_ns()))
    archive.mkdir()
    shutil.copy2(q.state_path,archive/'pipeline_state_before.json')
    audit={'authorization':'User requested direct evaluation without added recovery checks','at':utc_now(),'recovery_prechecks_performed':False,'evaluation_program':'original unchanged tools/evaluate_c_only.py','seed1_reused':True,'training_restarted':False,'archive':str(archive)}
    atomic_json(archive/'direct_evaluation_audit.json',audit)
    job=q.state['jobs']['evaluation_G1C_seed0']
    for field in ('exit_file','log'):
        p=Path(job[field])
        if p.exists(): p.rename(archive/p.name)
    p=q.reports/'evaluation/G1C/seed0'
    if p.exists(): p.rename(archive/'seed0_partial')
    p=q.reports/'pipeline_failure.txt'
    if p.exists(): p.rename(archive/'pipeline_failure.txt')
    for seed,gpu in ((0,0),(2,1)):
        job=q.state['jobs'][f'evaluation_G1C_seed{seed}']
        for key in ('process','gpu','returncode','started_at','ended_at','start','end'): job.pop(key,None)
        job.update(status='pending',preferred_gpu=gpu)
    q.state['jobs']['evaluation_G1C_seed1'].update(status='completed',returncode=0)
    cmd=q.state['jobs']['evaluation_G1C_seed0']['command']
    q.python=cmd[cmd.index('tools/evaluate_c_only.py')-1]
    q.state.update(status='running',phase='evaluation',process=process_info(),direct_evaluation_audit=str(archive/'direct_evaluation_audit.json'))
    q.state.pop('reason',None)
    q.state.pop('ended_at',None)
    q.record()
    try:
        q.evaluate()
        q.state.update(status='completed',phase='completed',ended_at=utc_now())
    except BaseException as error:
        q.state.update(status='failed',reason=str(error),ended_at=utc_now())
        (archive/'failure.txt').write_text(traceback.format_exc())
        raise
    finally: q.record()
