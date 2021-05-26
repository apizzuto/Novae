#!/usr/bin/env python

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
import argparse
import matplotlib as mpl
mpl.use('Agg')
import histlite as hl
import csky as cy
import pickle
from glob import glob

def run_all_signal_trials(args):
    r"""Look at output background trials and run the signal
    trials for a stacked nova analysis
    """
    delta_t = args.deltaT
    delta_t_days = delta_t / 86400.

    greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.5/'

    data_fs = sorted(glob(greco_base + 'IC86_20*data_with_angErr.npy'))
    exp = [np.load(data) for data in data_fs]
    exp = np.hstack(exp)
    if args.allflavor:
        mcfiles = glob(greco_base + 'IC86_2012.nu*_with_angErr.npy')
        mc = np.load(mcfiles[0])
        for flav in mcfiles[1:]:
            mc = np.concatenate((mc, np.load(flav)))
    else:
        mcfile = glob(greco_base + 'IC86_2012.numu_merged_with_angErr.npy')[0]
        mc = np.load(mcfile)
    grls = sorted(glob(greco_base + 'GRL/IC86_20*data.npy'))
    grl = [np.load(g) for g in grls]
    grl = np.hstack(grl)

    if args.minLogE is not None:
        exp_msk = exp['logE'] > args.minLogE
        exp = exp[exp_msk]
        mc_msk = mc['logE'] > args.minLogE
        mc = mc[mc_msk]

    greco = cy.selections.CustomDataSpecs.CustomDataSpec(
        exp, mc, np.sum(grl['livetime']), 
        np.linspace(-1., 1., 31),
        np.linspace(0., 4., 31), 
        grl=grl, key='GRECOv2.5', cascades=True
        )

    ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
    greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)

    master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
    ras = master_df['RA']
    decs = master_df['Dec']
    names = master_df['Name']
    mjds = np.array([t.mjd for t in master_df['Date']])
    delta_ts = np.ones_like(ras) * delta_t_days

    conf = {'extended': True,
        'space': "ps",
        'time': "transient",
        'sig': 'transient',
        }

    src = cy.utils.Sources(
        ra=np.radians(ras), 
        dec=np.radians(decs), 
        mjd=mjds, 
        sigma_t=np.zeros_like(delta_ts), 
        t_100=delta_ts
        )

    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    def ndarray_to_Chi2TSD(trials):
        return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

    add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
    bg = cy.bk.get_all(
        '/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/bg/',
        'delta_t_{:.2e}_{}seed_*.npy'.format(
        delta_t, add_str), 
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
    beta = 0.9 # beta = 0.5
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
    result['source_info'] = {'ra': ras, 'dec': decs, 'name': names, 'mjd': mjds}

    with open('/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/signal_results/' +
        'delta_t_{:.2e}_gamma_{}_{}allflavor_{}.pkl'.format(
        delta_t, args.index, add_str, args.allflavor), 'wb') as f:
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
    args = parser.parse_args()

    run_all_signal_trials(args)