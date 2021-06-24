#!/usr/bin/env python

import numpy as np
import argparse
import csky as cy
import pickle

import sys
sys.path.append('/home/apizzuto/Nova/scripts/stacking/')
from source_config import *
from stacking_config import *

def run_all_signal_trials(args):
    r"""Look at output background trials and run the signal
    trials for a stacked nova analysis
    """
    delta_t = args.deltaT
    delta_t_days = delta_t / 86400.

    ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
    greco, conf = get_stacking_objs(minLogE=args.minLogE, allflavor=args.allflavor)
    greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)

    only_gamma = not args.all_nova
    print('only', only_gamma)
    print('all', args.all_nova)
    print('~all', ~args.all_nova)
    weighting_scheme = args.weighting
    src, sample_str = get_sources(only_gamma, weighting_scheme, delta_t_days)

    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    def ndarray_to_Chi2TSD(trials):
        return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

    add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
    bg = cy.bk.get_all(
        '/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/bg/',
        '{}_delta_t_{:.2e}_{}seed_*.npy'.format(
        sample_str, delta_t, add_str), 
        merge=np.concatenate, 
        post_convert=ndarray_to_Chi2TSD
        )

    tr = cy.get_trial_runner(
        conf, ana=greco_ana, src=src, 
        inj_conf={'flux': cy.hyp.PowerLawFlux(args.index)}
        )

    result = {}

    if delta_t < 1e3:
        n_sig_step = 5
    elif delta_t <= 86400. and args.index <= 2.5:
        n_sig_step = 7
    elif delta_t <= 86400:
        n_sig_step = 15
    elif args.index <= 2.5:
        n_sig_step = 15
    else:
        n_sig_step = 25

    ########################################################################
    ################ SENSITIVITY CALCULATION ###############################
    ########################################################################
    beta = 0.9
    sensitivity = tr.find_n_sig(bg.median(), beta, 
                        batch_size=args.ntrials_sig,
                        n_sig_step=n_sig_step,
                        max_batch_size=0, 
                        logging=True, 
                        n_bootstrap=1)

    sensitivity['E2dNdE'] = tr.to_E2dNdE(sensitivity, E0=1., unit=1e3)

    ########################################################################
    ################ DISCOVERY POTENTIAL CALC ##############################
    ########################################################################
    thresh_ts = bg.isf_nsigma(5.)
    beta = 0.5 # beta = 0.5
    discovery = tr.find_n_sig(thresh_ts, beta,
                        batch_size=args.ntrials_sig,
                        n_sig_step=n_sig_step,
                        max_batch_size=0,
                        logging=True,
                        n_bootstrap=1)
    discovery['E2dNdE'] = tr.to_E2dNdE(discovery, E0=1., unit=1e3)
    discovery['nsigma'] = 5.
    discovery['CL'] = beta

    ########################################################################
    ######################## FIT BIAS TRIALS ###############################
    ########################################################################
    n_sigs = np.r_[:201:10]
    trials = [tr.get_many_fits(int(args.ntrials_sig/2), n_sig=n_sig, logging=False, seed=n_sig) for n_sig in n_sigs]
    for (n_sig, t) in zip(n_sigs, trials):
        t['ntrue'] = np.repeat(n_sig, len(t))
    allt = cy.utils.Arrays.concatenate(trials)

    result['bg'] = bg
    result['sensitivity'] = sensitivity
    result['discovery'] = discovery
    result['fit'] = allt
    result['settings'] = args
    result['source_info'] = {'ra': src.ra, 'dec': src.dec, 'mjd': src.mjd}

    with open('/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/signal_results/' +
        '{}_delta_t_{:.2e}_gamma_{}_{}allflavor_{}.pkl'.format(
        sample_str, delta_t, args.index, add_str, args.allflavor), 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Novae stacked analysis')
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

    run_all_signal_trials(args)
