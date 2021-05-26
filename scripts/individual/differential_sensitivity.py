#!/usr/bin/env python

import numpy as np
from scipy import stats
import pandas as pd
import astropy as ap
from astropy.table import Table
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

parser = argparse.ArgumentParser(description='Sensitivity for individual gamma-ray detected novae')
parser.add_argument('--deltaT', type=float, default = 1000.,
                    help='Time window in seconds')
parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
parser.add_argument('--ntrials_bg', type=int, default=10000, help="Number of background trials")
parser.add_argument('--ntrials_sig', type=int, default=500, help='Number of trials per signal strength')
parser.add_argument('--allflavor', action='store_true', default=False, help="All neutrino flavors in MC")
parser.add_argument('--dec', type=float, required=True, help='declination in degrees')
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

ra = 0.0
dec = np.radians(args.dec)
mjd_start = 58200. #This is just kinda random in 2017
mjd_stop = mjd_start + delta_t_days

conf = {'ana': greco_ana,
       'extended': True,
       'space': "ps",
        'time': "transient",
        'sig': 'transient',
       }

src = cy.utils.Sources(ra=ra, 
                       dec=dec, 
                       mjd=mjd_start, 
                       sigma_t=0., 
                       t_100=delta_t_days)
cy.CONF['src'] = src
cy.CONF['mp_cpus'] = 5

tr = cy.get_trial_runner(conf, ana=greco_ana, src=src)
bg = cy.dists.Chi2TSD(tr.get_many_fits(args.ntrials_bg))

Ebins = 10**np.r_[0.:4.1:1.0]
trs = [
    cy.get_trial_runner(conf, 
        ana=greco_ana, src=src,
        flux=cy.hyp.PowerLawFlux(2.0, energy_range=(Emin, Emax)))
    for (Emin, Emax) in zip(Ebins[:-1], Ebins[1:])
]

result = {}

for ii, tr in enumerate(trs):
    try:
        ########################################################################
        ################ SENSITIVITY CALCULATION ###############################
        ########################################################################
        sensitivity = tr.find_n_sig(bg.median(), 0.9, 
                            batch_size=args.ntrials_sig,
                            n_sig_step=3,
                            max_batch_size=0, 
                            logging=True, 
                            n_bootstrap=1)
        sensitivity['E2dNdE'] = tr.to_E2dNdE(sensitivity, E0=Ebins[ii]/1e3, unit=1e3)
        result[f'sensitivity_{Ebins[ii]:.1f}_{Ebins[ii+1]:.1f}'] = sensitivity
    except:
        print(f"Could not do energy range {Ebins[ii]} to {Ebins[ii+1]}")

result['bg'] = bg
result['settings'] = args
result['source_info'] = {'ra': ra, 'dec': dec, 
    'mjd_start': mjd_start, 'mjd_stop': mjd_stop}

add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
delta_t_str = f"{delta_t:.2e}"
output_loc = f"/data/user/apizzuto/Nova/csky_trials/differential_sens/dec_{dec:.1f}_delta_t_{delta_t_str}_{add_str}_allflavor_{args.allflavor}_trials.pkl"

with open(output_loc, 'wb') as f:
    pickle.dump(result, f)