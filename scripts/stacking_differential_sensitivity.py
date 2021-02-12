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

def run_differential_sens(args):
    r"""Look at output background trials and run the signal
    trials for a stacked nova analysis
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

    Ebins = 10**np.r_[0.:4.1:1.0]
    trs = [
        cy.get_trial_runner(conf, 
            ana=greco_ana, src=src,
            flux=cy.hyp.PowerLawFlux(2.0, energy_range=(Emin, Emax)))
        for (Emin, Emax) in zip(Ebins[:-1], Ebins[1:])
    ]

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
    
    result = {}

    for ii, tr in enumerate(trs):
        try:
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

            sensitivity['E2dNdE'] = tr.to_E2dNdE(sensitivity, E0=Ebins[ii]/1e3, unit=1e3)
            result[f'sensitivity_{Ebins[ii]:.1f}_{Ebins[ii+1]:.1f}'] = sensitivity
        except:
            print(f"Could not do energy range {Ebins[ii]} to {Ebins[ii+1]}")

    result['bg'] = bg
    result['settings'] = args
    result['source_info'] = {'ra': ras, 'dec': decs, 'name': names, 'mjd': mjds}

    with open('/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/differential_sens/' +
        'delta_t_{:.2e}_{}allflavor_{}.pkl'.format(
        delta_t, add_str, args.allflavor), 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Novae stacked analysis')
    parser.add_argument('--deltaT', type=float, default = 86400.,
                        help='Time window in seconds')
    parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
    parser.add_argument('--allflavor', action='store_true', default=False, help="All neutrino flavors in MC")
    parser.add_argument('--ntrials_sig', type=float, default=250, help="Number of signal trials")
    parser.add_argument('--seed', type=int, default=123, help="Random number seed")
    args = parser.parse_args()

    run_differential_sens(args)