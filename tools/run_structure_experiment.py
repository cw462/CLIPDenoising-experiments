"""Proposal C: paired B0/B3 continuation, original model/PFA, different loss only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.experiment_common import digest, load_paths
from tools.run_pfa_experiment import configuration as base_configuration, run


def configuration(group, stage, seed, paths):
    if group not in ('B0', 'B3'):
        raise ValueError('Proposal C supports B0 and B3 only')
    if paths.get('rho', 0) != 0:
        raise ValueError('Proposal C must use the original independent PFA (rho=0)')
    opt, _ = base_configuration('B0', stage, seed, dict(paths, rho=0))
    directory = Path(paths['output_root']) / stage / group / f'seed{seed}'
    opt.update(exp=group, experiment=group, name=f'plan_c_{stage}_{group}_seed{seed}')
    opt['path'].update(experiments_root=str(directory), models=str(directory / 'checkpoints'),
                       training_states=str(directory / 'checkpoints'), log=str(directory),
                       visualization=str(directory / 'visuals'))
    if group == 'B3':
        opt['train']['pixel_opt'] = dict(paths['pixel_opt'])
        if opt['train']['pixel_opt'].get('type') != 'L1StructureLoss':
            raise ValueError('B3 requires L1StructureLoss')
    opt['plan_c'] = {'c0_sha256': opt.pop('plan_a')['c0_sha256'],
                     'change': 'Loss only; original model, original PFA, matched paired Mixup targets',
                     'training_noise': 'Gaussian sigma15 only', 'new_optimizer': True,
                     'pixel_opt': dict(opt['train']['pixel_opt'])}
    opt.pop('config_hash')
    opt['config_hash'] = digest(opt)
    return opt, directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('B0', 'B3'), required=True)
    parser.add_argument('--stage', choices=('smoke', 'screen', 'confirm', 'confirm_short'), default='smoke')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--paths', default='configs/plan_c.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    run(args.group, args.stage, args.seed, load_paths(args.paths), args.resume, args.stop_after,
        configuration_builder=configuration)


if __name__ == '__main__':
    main()
