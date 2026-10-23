from __future__ import annotations
import argparse,csv,gc,json,os,sys,time
from pathlib import Path
ROOT=Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
OLD=Path('/home/imss/cw/CLIPDenoising_20261020')
OUT=ROOT/'reports/table1_channel_salt_3seed_20261022'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from tools.experiment_common import load_paths,read_json,torch_load,atomic_json,utc_now
from tools.evaluation_utils import read_rgb,image_files,calculate_metrics,stable_seed
from tools.evaluate_experiment import infer
import hashlib
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    paths=load_paths('configs/plan_g1c_standard_server.yaml')
    dest=OUT/'evaluation'/f'seed{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():raise FileExistsError('Completed result already exists')
    status={'status':'running','training_seed':args.seed,'started_at':utc_now(),'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'new_measurements':0}
    atomic_json(dest/'status.json',status)
    device=torch.device('cuda')
    models={};identities={}
    for method in ('G1','G1C'):
        run=(OLD/'runs/full/G1' if method=='G1' else ROOT/'runs/g1c_standard/full/G1C')/f'seed{args.seed}'
        config=read_json(run/'effective_config.json')
        latest=read_json(run/'checkpoints/latest.json')
        checkpoint_path=Path(latest['path'])
        if not checkpoint_path.is_absolute():checkpoint_path=ROOT/checkpoint_path
        checkpoint=torch_load(checkpoint_path)
        network=dict(config['network_g']);network.pop('type',None)
        network['model_path']=paths['pretrained_rn50']
        model=CLIPDenoising(**network)
        model.load_state_dict(checkpoint['params'],strict=True)
        del checkpoint;gc.collect()
        model.to(device).eval()
        adapter=getattr(model.up_path[4],'rgb_calibration',None)
        if adapter is not None:adapter.capture_diagnostics=False
        models[method]=model
        identities[method]={'checkpoint':str(checkpoint_path),'recorded_checkpoint_sha256':latest.get('sha256'),'training_seed':args.seed,'recorded_training_update':latest.get('update'),'config_path':str(run/'effective_config.json')}
    fields=('experiment','train_seed','dataset','image_id','noise_type','noise_level','psnr','ssim','noise_seed','noise_sha256','checkpoint')
    rows=[]
    with (dest/'per_image_results.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for dataset,expected_count in (('CBSD68',68),('McMaster',18),('Kodak24',24),('Urban100',100)):
            base=Path(paths['test_datasets'][dataset]);files=image_files(base)
            if len(files)!=expected_count:raise RuntimeError('Unexpected image count for '+dataset)
            for index,file in enumerate(files):
                image_id=file.relative_to(base).as_posix();clean=read_rgb(file)
                noise_seed=stable_seed(dataset,image_id,'salt_pepper',.02)
                draw=np.random.RandomState(noise_seed).uniform(size=clean.shape)
                noisy=clean.copy();noisy[draw<.01]=0;noisy[(draw>=.01)&(draw<.02)]=1
                noisy=np.ascontiguousarray(noisy,dtype=np.float32)
                noise_hash=hashlib.sha256(noisy.tobytes()).hexdigest()
                for method,model in models.items():
                    restored,elapsed,_=infer(model,noisy,device)
                    scores=calculate_metrics(clean,restored)
                    row={'experiment':method,'train_seed':args.seed,'dataset':dataset,'image_id':image_id,'noise_type':'salt_pepper','noise_level':.02,**scores,'noise_seed':noise_seed,'noise_sha256':noise_hash,'checkpoint':identities[method]['checkpoint']}
                    writer.writerow(row);rows.append(row)
                if (index+1)%20==0 or index+1==len(files):
                    stream.flush()
                    status.update(new_measurements=len(rows),dataset=dataset,dataset_images=index+1,updated_at=utc_now())
                    atomic_json(dest/'status.json',status)
                    print(f'seed{args.seed} {dataset} {index+1}/{len(files)}; {len(rows)}/420 measurements',flush=True)
        stream.flush()
    result={'completed':True,'training_seed':args.seed,'completed_at':utc_now(),'rows':len(rows),'images':210,'methods':['G1','G1C'],'model_identities':identities,'noise_mode':'independent_RGB_channel_elements','density':.02,'noise_sampling':'H x W x 3 uniform; u<d/2 ->0; d/2<=u<d ->1; otherwise unchanged','same_input_object_used_for_both_models':True,'noise_seed_independent_of_training_seed':True,'matlab_bitwise_rng_reproduction':False,'additional_historical_identity_gates':False,'training_restarted':False}
    atomic_json(dest/'summary.json',result)
    status.update(status='completed',new_measurements=len(rows),ended_at=utc_now())
    atomic_json(dest/'status.json',status)
if __name__=='__main__':main()
