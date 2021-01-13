#!/usr/bin/env python

import numpy as np
import pandas as pd
from astropy.time import Time
import argparse
import matplotlib as mpl
mpl.use('Agg')
import histlite as hl
import csky as cy
import pickle
from glob import glob


def run_sensitivity_trials(args):
    """Run the sensitivity trials for the stacked nova analysis

    :type args: dict-like
    :param args: Command line arguments or dict with keys needed for analysis
    """
    delta_t = args.deltaT
    delta_t_days = delta_t / 86400.

    greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.4/'

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
        grl=grl, key='GRECOv2.4', cascades=True
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

    tr = cy.get_trial_runner(
        conf, ana=greco_ana, src=src, 
        inj_conf={'flux': cy.hyp.PowerLawFlux(args.index)}
        )

    trials = tr.get_many_fits(args.ntrials_sig, args.nsig, poisson=True, 
        seed=args.seed, logging=False)

    mydir = cy.utils.ensure_dir('/data/user/apizzuto/Nova/csky_trials/' \
        + 'stacking_sens_res/sensitivity/nsig/{:05.1f}'.format(args.nsig)
        )
    add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
    filename = mydir \
        + '/delta_t_{:.2e}_gamma_{}{}_allflavor_{}_seed_{}.npy'.format(
        delta_t, args.index, add_str, args.allflavor, args.seed)
    np.save(filename, trials.as_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Novae stacked analysis sensitivity trials')
    parser.add_argument('--deltaT', type=float, default = 86400.,
                        help='Time window in seconds')
    parser.add_argument('--index', type=float, default=1, help='Spectral Index')
    parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
    parser.add_argument('--allflavor', action='store_true', default=False, help="All neutrino flavors in MC")
    parser.add_argument('--ntrials_sig', type=float, default=250, help="Number of signal trials")
    parser.add_argument('--nsig', type=float, default=1., help="Number of signal events to inject")
    parser.add_argument('--seed', type=int, default=123, help="Random number seed")
    args = parser.parse_args()

    run_sensitivity_trials(args)
