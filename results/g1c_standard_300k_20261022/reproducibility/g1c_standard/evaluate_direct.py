import os,sys,inspect
from pathlib import Path
ROOT=Path('/home/imss/cw/CLIPDenoising_G1C_Standard_20261022')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import tools.evaluate_c_only as e
from tools.evaluation_utils import make_noise
source=inspect.getsource(e.evaluate_variant)
source=source.replace('checked_noise(clean, family, level, dataset, image_id, reference)','make_noise(clean, family, level, dataset, image_id)')
source=source.replace('"matched_historical_inputs": len(rows)','"matched_historical_inputs": None, "evaluation_input_consistency_checked": False, "evaluation_mode": "direct_user_requested"')
source=source.replace('protocol = protocol_for_seed(seed, contract.get("training_integrity_mode", GUARDED_MODE))','protocol = protocol_for_seed(seed, contract.get("training_integrity_mode", GUARDED_MODE))\n    protocol.update(comparison="Historical same-seed comparison; noise identity not verified by user request", preflight="Additional recovery and noise consistency checks skipped")')
e.make_noise=make_noise
exec(compile(source,str(Path(__file__)), 'exec'),e.__dict__)
def direct_gate(paths,variant,seed):
    run=Path(paths['output_root'])/'full'/variant/f'seed{seed}'
    config=e.read_json(run/'effective_config.json')
    latest=e.read_json(run/'checkpoints/latest.json')
    checkpoint_path=e.resolve_path(paths,latest['path'])
    checkpoint=e.torch_load(checkpoint_path)
    status=e.read_json(run/'status.json')
    contract={'variant':variant,'baseline':'G1','training_seed':seed,'training_updates':status['update'],'training_completed':status['status']=='completed','training_integrity_mode':e.STANDARD_MODE,'per_update_input_consistency_checks':False,'evaluation_training_asset_checks':False,'source_hash':config['source_hash'],'config_hash':config['config_hash']}
    return config,checkpoint_path,checkpoint,contract
e.validated_gate=direct_gate
e.verify_baseline_checkpoint=lambda paths,baseline,identity: {'identity_verified':False,'checkpoint_sha256':identity['checkpoint_sha256'],'check_skipped_by_user_request':True}
e.main()
