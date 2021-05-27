#!/usr/bin/env python

import numpy as np
import argparse
import csky as cy
import pickle

import sys
sys.path.append('/home/apizzuto/Nova/scripts/stacking/')
from source_config import *
from stacking_config import *

def run_fitting_trials(args):

    delta_t = args.deltaT
    delta_t_days = delta_t / 86400.

    ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
    greco, conf = get_stacking_objs(minLogE=args.minLogE, allflavor=args.allflavor)
    greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)

    only_gamma = ~args.all_nova
    weighting_scheme = args.weighting
    src, sample_str = get_sources(only_gamma, weighting_scheme, delta_t_days)

    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    tr = cy.get_trial_runner(
        conf, ana=greco_ana, src=src, 
        inj_conf={'flux': cy.hyp.PowerLawFlux(args.index)}
        )

    n_sigs = np.r_[:201:10]
    trials = [tr.get_many_fits(int(args.ntrials_sig), 
        n_sig=n_sig, logging=False, seed=args.seed) for n_sig in n_sigs]
    for (n_sig, t) in zip(n_sigs, trials):
        t['ntrue'] = np.repeat(n_sig, len(t))
    allt = cy.utils.Arrays.concatenate(trials)

    add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
    filename = '/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/fits/' \
        + '{}_delta_t_{:.2e}_gamma_{}{}_allflavor_{}_seed_{}.npy'.format(
        sample_str, delta_t, args.index, add_str, args.allflavor, args.seed)
    np.save(filename, allt.as_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Novae stacked analysis bias trials')
    parser.add_argument('--deltaT', type=float, default = 86400.,
                        help='Time window in seconds')
    parser.add_argument('--index', type=float, default=1, help='Spectral Index')
    parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
    parser.add_argument('--allflavor', action='store_true', default=False, help="All neutrino flavors in MC")
    parser.add_argument('--ntrials_sig', type=float, default=250, help="Number of signal trials")
    parser.add_argument('--seed', type=int, default=123, help="Random number seed")
    parser.add_argument(
        '--all_nova', default=False, action='store_true',
        help = 'Raise this flag if you want to include the optically detected novae'
    )
    parser.add_argument(
        '--weighting', default="optical", 
        help="Weighting scheme. Choose between 'optical' and 'gamma'"
    )
    args = parser.parse_args()

    run_fitting_trials(args)
