#!/usr/bin/env python

import numpy as np
from scipy import stats
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from datetime import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg')
import histlite as hl
import csky as cy
import pickle
from glob import glob
import sys

parser = argparse.ArgumentParser(description='Sensitivity vs declination for a fixed time window')
parser.add_argument('--deltaT', type=float, default = 1000.,
                    help='Time window in seconds')
parser.add_argument('--index', type=float, default=1, help='Spectral Index')
parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
parser.add_argument('--ntrials_bg', type=int, default=10000, help="Number of background trials")
parser.add_argument('--ntrials_sig', type=int, default=500, help='Number of trials per signal strength')
parser.add_argument('--allflavor', action='store_true', default=False, help="All neutrino flavors in MC")
args = parser.parse_args()

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
    low_en_bin = args.minLogE
else:
    low_en_bin = 0.

greco = cy.selections.CustomDataSpecs.CustomDataSpec(exp, mc, np.sum(grl['livetime']), 
                                                     np.linspace(-1., 1., 31),
                                                     np.linspace(low_en_bin, 4., 31), 
                                                     grl=grl, key='GRECOv2.5', cascades=True)

ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)

conf = {'ana': greco_ana,
       'extended': True,
       'space': "ps",
        'time': "transient",
        'sig': 'transient',
       }

result_dict = {'dec': [], 
    'sens': [],
    'sens_nsig': []}

ra = 0.
mjd_start = 57200.
sin_decs = np.linspace(-0.95, 0.95, 20)
for dec in np.arcsin(sin_decs):
    src = cy.utils.Sources(ra=ra, 
                        dec=dec, 
                        mjd=mjd_start, 
                        sigma_t=0., 
                        t_100=delta_t_days)
    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    tr = cy.get_trial_runner(conf, ana=greco_ana, src=src)
    bg = cy.dists.Chi2TSD(tr.get_many_fits(args.ntrials_bg))

    tr = cy.get_trial_runner(conf, ana=greco_ana, src=src, 
        inj_conf={'flux': cy.hyp.PowerLawFlux(args.index)})

    ########################################################################
    ################ SENSITIVITY CALCULATION ###############################
    ########################################################################
    nsig_step = 1. if delta_t < 864000. else 20.

    sensitivity = tr.find_n_sig(bg.median(), 0.9, 
                        batch_size=args.ntrials_sig,
                        n_sig_step=nsig_step,
                        max_batch_size=0, 
                        logging=True, 
                        n_bootstrap=1)
    sensitivity['E2dNdE'] = tr.to_E2dNdE(sensitivity, E0=1., unit=1e3)

    result_dict['dec'].append(dec)
    result_dict['sens'].append(sensitivity['E2dNdE'])
    result_dict['sens_nsig'].append(sensitivity['n_sig'])


add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
delta_t_str = f"{delta_t:.2e}"
output_loc = f"/data/user/apizzuto/Nova/csky_trials/sens_vs_dec/sens_vs_dec_delta_t_{delta_t_str}_{add_str}gamma_{args.index}_allflavor_{args.allflavor}_trials.pkl"

with open(output_loc, 'wb') as f:
    pickle.dump(result_dict, f)
