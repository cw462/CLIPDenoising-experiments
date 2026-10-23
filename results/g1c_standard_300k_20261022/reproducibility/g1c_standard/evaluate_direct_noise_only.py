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
e.main()
