import os,sys,json,shutil
from pathlib import Path
ROOT=Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from tools.summarize_g1c_multiseed import aggregate,across_seeds,hierarchical_ood,TABLE1_CONDITIONS,cell_key
from tools.evaluation_utils import read_csv,write_csv
from tools.experiment_common import atomic_json,utc_now
base=ROOT/'reports/g1c_standard/evaluation'
out=ROOT/'reports/g1c_standard/final';out.mkdir(exist_ok=True)
rows=[];pairs=[];summaries=[]
for seed in range(3):
 d=base/'G1C'/f'seed{seed}'
 summaries.append(json.loads((d/'summary.json').read_text()))
 new=read_csv(d/'per_image_results.csv');old=read_csv(d/'baseline_per_image_results.csv')
 if len(new)!=3150 or len(old)!=3150: raise RuntimeError('Incomplete measurement files')
 for row in new+old:
  for k in ('psnr','ssim','noise_level'): row[k]=float(row[k])
  row['train_seed']=int(row['train_seed'])
 rows.extend(old+new)
 index={cell_key(r):r for r in old}
 for row in new:
  b=index[cell_key(row)]
  pairs.append({'train_seed':seed,**{k:row[k] for k in ('dataset','image_id','noise_type','noise_level','protocol_group','domain')},'psnr':row['psnr'],'baseline_psnr':b['psnr'],'delta_psnr':row['psnr']-b['psnr'],'ssim':row['ssim'],'baseline_ssim':b['ssim'],'delta_ssim':row['ssim']-b['ssim'],'noise_sha256':row['noise_sha256'],'baseline_noise_sha256':b['noise_sha256'],'input_identity_verified':False})
keys=('dataset','noise_type','noise_level','protocol_group','domain')
metrics=('psnr','baseline_psnr','delta_psnr','ssim','baseline_ssim','delta_ssim')
conditions=aggregate(rows,('experiment','train_seed',*keys),('psnr','ssim'))
cs=[{**{k:r[k] for k in ('experiment','train_seed',*keys)},'psnr':r['psnr_mean'],'ssim':r['ssim_mean']} for r in conditions]
ps=[{**{k:r[k] for k in ('train_seed',*keys)},'n_images':r['n'],**{k:r[k+'_mean'] for k in metrics}} for r in aggregate(pairs,('train_seed',*keys),metrics)]
summary=across_seeds(ps,keys,metrics)
selected=lambda r:(r['dataset'],r['noise_type'],float(r['noise_level'])) in TABLE1_CONDITIONS
table=[r for r in summary if selected(r)]
ood=hierarchical_ood(conditions)
oi={(r['experiment'],r['train_seed'],r['table']):r for r in ood}
op=[]
for seed in range(3):
 for name in ('official_three_families','five_families_with_extensions'):
  a,b=oi['G1C',seed,name],oi['G1',seed,name]
  op.append({'train_seed':seed,'table':name,'delta_ood_psnr':a['ood_psnr']-b['ood_psnr'],'delta_ood_ssim':a['ood_ssim']-b['ood_ssim']})
artifacts={'per_image_results.csv':rows,'same_seed_comparisons.csv':pairs,'condition_seedwise.csv':conditions,'condition_summary.csv':across_seeds(cs,('experiment',*keys),('psnr','ssim')),'same_seed_condition_summary.csv':summary,'table1_conditions.csv':table,'table1_seedwise.csv':[r for r in ps if selected(r)],'ood_seedwise.csv':ood,'ood_summary.csv':across_seeds(ood,('experiment','table'),('ood_psnr','ood_ssim')),'ood_difference_seedwise.csv':op,'ood_difference_summary.csv':across_seeds(op,('table',),('delta_ood_psnr','delta_ood_ssim')),'id_summary.csv':[r for r in summary if r['domain']=='ID']}
for name,data in artifacts.items():write_csv(out/name,data)
note='Evaluation completed with the user-requested direct mode. Additional recovery and seed0/seed2 noisy-input consistency checks were skipped. Seed0 also skipped training-asset and historical-checkpoint identity checks. Same-seed differences below are descriptive comparisons; identical historical noise is not established for the combined results. Actual new and baseline noise hashes are retained separately. Training was not restarted. Original failures remain archived.'
lines=['# G1C vs historical G1: three seeds, 300k updates','',note,'','Mean and sample SD across three training seeds. OOD uses the original hierarchical aggregation.','','| Dataset | Noise | G1 PSNR | G1C PSNR | Difference dB |','|---|---|---:|---:|---:|']
for r in table:lines.append('| {} | {} {} | {:.4f} +/- {:.4f} | {:.4f} +/- {:.4f} | {:+.4f} +/- {:.4f} |'.format(r['dataset'],r['noise_type'],r['noise_level'],r['baseline_psnr_mean'],r['baseline_psnr_std'],r['psnr_mean'],r['psnr_std'],r['delta_psnr_mean'],r['delta_psnr_std']))
(out/'experiment_report.md').write_text('\n'.join(lines)+'\n')
atomic_json(out/'summary.json',{'completed':True,'completed_at':utc_now(),'training_seeds':[0,1,2],'new_measurement_rows':9450,'reused_baseline_rows':9450,'rows':18900,'same_seed_comparison_rows':len(pairs),'paired_rows':None,'historical_noise_identity_verified':False,'evaluation_mode':'direct_user_requested','additional_recovery_prechecks':False,'independent_publication_reaudit':False,'limitation':note})
print('Completed: 9450 new + 9450 historical rows',flush=True)
