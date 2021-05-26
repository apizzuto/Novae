#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import csky as cy
import pickle

import sys
sys.path.append('/home/apizzuto/Nova/scripts/')
from source_config import *
from stacking_config import *

def run_background_trials(args):
    """Run the background trials for the stacked nova analysis

    :type args: dict-like
    :param args: Command line arguments or dict with keys needed for analysis
    """
    delta_t = args.deltaT
    delta_t_days = delta_t / 86400.

    ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
    greco, conf = get_stacking_objs(minLogE=args.minLogE)
    greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)

    only_gamma = ~args.all_nova
    weighting_scheme = args.weighting
    src, sample_str = get_sources(only_gamma, weighting_scheme, delta_t_days)

    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    tr = cy.get_trial_runner(conf, ana=greco_ana, src=src)
    n_trials = args.ntrials
    
    bg_trials = tr.get_many_fits(n_trials, seed=args.seed, logging=False)
    
    add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
    filename = '/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/bg/' \
        + '{}_delta_t_{:.2e}_{}seed_{}.npy'.format(
        sample_str, delta_t, add_str, args.seed)
    np.save(filename, bg_trials.as_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Novae stacked analysis background'
        )
    parser.add_argument(
        '--deltaT', type=float, default = 86400.,
        help='Time window in seconds'
        )
    parser.add_argument(
        '--minLogE', type=float, default=None, 
        help='Cut on the minimum reco energy'
        )
    parser.add_argument(
        '--ntrials', type=float, default=5000, 
        help="Number of backgound trials"
        )
    parser.add_argument(
        '--seed', type=int, default=123, help="Random number seed"
        )
    parser.add_argument(
        '--all_nova', default=False, action='store_true',
        help = 'Only stack gamma ray detected novae if False, else, stack all novae'
    )
    parser.add_argument(
        '--weighting', default="optical", 
        help="Weighting scheme. Choose between 'optical' and 'gamma'"
    )
    args = parser.parse_args()

    run_background_trials(args)
