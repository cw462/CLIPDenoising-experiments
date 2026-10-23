import concurrent.futures,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
OUT=ROOT/'reports/table1_channel_salt_3seed_20261022'
os.chdir(ROOT)
OUT.mkdir(parents=True,exist_ok=True)
def save(value):
    tmp=OUT/'pipeline_status.json.tmp'
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(OUT/'pipeline_status.json')
info={'status':'running','started_at':time.time(),'pid':os.getpid(),'seeds':[0,1,2]}
save(info)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
available=[int(row.split(',')[0]) for row in gpu.splitlines() if int(row.split(',')[1])<500 and int(row.split(',')[2])<5]
if len(available)<2:raise RuntimeError('Two idle GPUs unavailable')
def cohort(gpu,seeds):
    for seed in seeds:
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log=OUT/f'seed{seed}.log'
        with log.open('x') as stream:
            result=subprocess.run([sys.executable,'-u',str(OUT/'scripts/evaluate_channel_salt.py'),'--seed',str(seed)],env=env,stdout=stream,stderr=subprocess.STDOUT)
        (OUT/f'seed{seed}.exit.json').write_text(json.dumps({'returncode':result.returncode,'gpu':gpu,'seed':seed}))
        if result.returncode:raise RuntimeError(f'seed{seed} exited {result.returncode}')
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(cohort,available[0],[0,2]),pool.submit(cohort,available[1],[1])]
        for future in futures:future.result()
    info.update(status='completed',new_measurement_rows=1260,ended_at=time.time())
except BaseException as error:
    info.update(status='failed',reason=str(error),traceback=traceback.format_exc(),ended_at=time.time())
    raise
finally:save(info)
