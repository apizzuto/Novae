#!/usr/bin/env python

import numpy as np
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

sys.path.append('/home/apizzuto/Nova/scripts/')
from greco_config import *

parser = argparse.ArgumentParser(description='Sensitivity for individual gamma-ray detected novae')
parser.add_argument('--deltaT', type=float, default = 1000.,
                    help='Time window in seconds')
parser.add_argument('--index', type=float, default=1, help='Spectral Index')
parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
parser.add_argument('--nova_num', type=int, help="Which nova from the list to use")
parser.add_argument('--ntrials_bg', type=int, default=10000, help="Number of background trials")
parser.add_argument('--ntrials_sig', type=int, default=500, help='Number of trials per signal strength')
parser.add_argument('--full_gamma_time', action='store_true', default=False,
                        help="Raise if you want to use the full gamma ray time window")
parser.add_argument('--disc_n_sigma', type=float, default=5., help="Number of sigma for disc. pot")
parser.add_argument('--disc_CL', type=float, default=0.5, help="Confidence level for disc. pot")
parser.add_argument('--allflavor', action='store_true', default=False, help="All neutrino flavors in MC")
args = parser.parse_args()

delta_t = args.deltaT
delta_t_days = delta_t / 86400.
nova_ind = args.nova_num

greco, conf = get_greco_conf(minLogE=args.minLogE, allflavor=args.allflavor)
ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)

master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
gamma_df = master_df[master_df['gamma']==True]
gamma_df = gamma_df.reset_index()

try:
    ra = np.radians(gamma_df['RA'][nova_ind])
    dec = np.radians(gamma_df['Dec'][nova_ind])
    mjd_start = gamma_df['gamma_start'][nova_ind].mjd
    mjd_stop = gamma_df['gamma_stop'][nova_ind].mjd
    name = gamma_df['Name'][nova_ind]
except:
    print(f"Only {len(gamma_df)} novae to choose from, index {nova_ind} not available")
    sys.exit()

if not args.full_gamma_time:
    mjd_stop = mjd_start + delta_t_days

during_greco = (mjd_start > greco_ana.mjd_min) & (mjd_start < greco_ana.mjd_max)
if not during_greco:
    print(f"Nova {name} did not occur during GRECO livetime. Exiting . . . ")
    sys.exit()

delta_t_days = delta_t_days if not args.full_gamma_time else mjd_stop - mjd_start

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
                       t_100=delta_t_days, 
                       name=name)
cy.CONF['src'] = src
cy.CONF['mp_cpus'] = 5

tr = cy.get_trial_runner(conf, ana=greco_ana, src=src)
bg = cy.dists.Chi2TSD(tr.get_many_fits(args.ntrials_bg))

tr = cy.get_trial_runner(conf, ana=greco_ana, src=src, inj_conf={'flux': cy.hyp.PowerLawFlux(args.index)})

result = {}

########################################################################
################ SENSITIVITY CALCULATION ###############################
########################################################################
sensitivity = tr.find_n_sig(bg.median(), 0.9, 
                       batch_size=args.ntrials_sig,
                       n_sig_step=3,
                       max_batch_size=0, 
                       logging=True, 
                       n_bootstrap=1)
sensitivity['E2dNdE'] = tr.to_E2dNdE(sensitivity, E0=1., unit=1e3)

########################################################################
################ DISCOVERY POTENTIAL CALC ##############################
########################################################################
thresh_ts = bg.isf_nsigma(args.disc_n_sigma)
discovery = tr.find_n_sig(thresh_ts, args.disc_CL, 
                       batch_size=args.ntrials_sig,
                       n_sig_step=3,
                       max_batch_size=0,
                       logging=True,
                       n_bootstrap=1)
discovery['E2dNdE'] = tr.to_E2dNdE(discovery, E0=1., unit=1e3)
discovery['nsigma'] = args.disc_n_sigma
discovery['CL'] = args.disc_CL

########################################################################
######################## FIT BIAS TRIALS ###############################
########################################################################
n_sigs = np.r_[:51:10]
trials = [tr.get_many_fits(args.ntrials_sig, n_sig=n_sig, logging=False, seed=n_sig) for n_sig in n_sigs]
for (n_sig, t) in zip(n_sigs, trials):
    t['ntrue'] = np.repeat(n_sig, len(t))
allt = cy.utils.Arrays.concatenate(trials)

result['bg'] = bg
result['sensitivity'] = sensitivity
result['discovery'] = discovery
result['fit'] = allt
result['settings'] = args
result['source_info'] = {'ra': ra, 'dec': dec, 'name': name, 'mjd_start': mjd_start, 'mjd_stop': mjd_stop}

name = name.replace(' ', '_')
add_str = 'minLogE_{:.1f}_'.format(args.minLogE) if args.minLogE is not None else ''
delta_t_str = f"{delta_t:.2e}" if not args.full_gamma_time else "full_gamma_time"
output_loc = f"/data/user/apizzuto/Nova/csky_trials/nova_{nova_ind}_{name}_delta_t_{delta_t_str}_{add_str}gamma_{args.index}_allflavor_{args.allflavor}_trials.pkl"

with open(output_loc, 'wb') as f:
    pickle.dump(result, f)
