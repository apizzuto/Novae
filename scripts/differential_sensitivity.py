#!/usr/bin/env python

import numpy as np
import pandas as pd
import os, sys, argparse
from astropy.time import Time
from astropy.time import TimeDelta
from numpy.lib.recfunctions import append_fields
base_path = os.path.join('/home/apizzuto/Nova/scripts/','')
sys.path.append(base_path)
from config import *
from Novae import Nova

parser = argparse.ArgumentParser(description='Nova Analysis Sensitivity')
parser.add_argument('--deltaT', type=float, default = 1000.,
                    help='Time window in seconds')
parser.add_argument('--n', type=int, default=1000,
                    help='Number of trials per mean signal')
parser.add_argument('--index', type=int, default=1, help='Index of nova list')
parser.add_argument('--spec', type=str, default='SPL', help='Spectrum, either single' \
                            +' power law (SPL) or power law with cutoff (EPL)')
parser.add_argument('--lowE', type=float, default=0., help='Log10 of minimum energy')
parser.add_argument('--deltaLogE', type=float, default=1., help='Width in log space of' \
                        +' energy interval')
parser.add_argument('--display', default=False, action='store_true', help='Print results '\
                        +'for debugging')
parser.add_argument('--allflavor', default=False, action='store_true', help='Include nue nutau')
args = parser.parse_args()

df = pd.read_pickle('/home/apizzuto/Nova/Novae_details_with_seasons.csv')
df = df.sort_values('Start Time (gamma ray)') 

deltaT = args.deltaT

novae = []
seasons = []
if args.spec == 'EPL':
    for index, row in df.iterrows():
        a = Nova(row['Name'], row['EPL gamma'], np.radians(row['RA (Deg)']), np.radians(row['Dec. (Deg)']), 
                row['Start Time (gamma ray)'], deltaT / 86400., #row['Duration (gamma ray)'], 
                flux_norm=row['EPL flux']*1e-7, cutoff = row['EPL cutoff '])
        novae.append(a)
        seasons.append(row['GRECO season'])
elif args.spec == 'SPL':
    for index, row in df.iterrows():
        a = Nova(row['Name'], row['SPL gamma'], np.radians(row['RA (Deg)']), np.radians(row['Dec. (Deg)']), 
                row['Start Time (gamma ray)'], deltaT / 86400., #row['Duration (gamma ray)'], 
                flux_norm=row['SPL flux']*1e-7, cutoff = None)
        novae.append(a)
        seasons.append(row['GRECO season'])
else:
    raise ValueError('Spectrum type must either be single power law (SPL) or '\
                        + 'power law with cutoff (EPL)')

nova = novae[args.index]
season = seasons[args.index]
if season == None:
    print("No valid GRECO data for nova {}".format(nova.name))
    exit()

inj_low_E = 10.**args.lowE
inj_high_E = inj_low_E * (10.**args.deltaLogE)
#FIGURE OUT SEASON?
llh = initialize_llh(nova, season=season, all_flavor=args.allflavor)
inj = initialize_injector(nova, llh, inj_e_range=(inj_low_E, inj_high_E), fixed_inj_gamma=2.0)


TS, ns, gamma, mean_ninj, ninj, flux = [], [], [], [], [], []
nsigs = np.unique(np.append(np.linspace(1, 20, 20), np.linspace(20., 100., 9)))
for nsig in nsigs:
    for jjj in range(args.n):
        ni, sample = inj.sample(nova.ra, nsig, poisson=True)
        val =  llh.scan(nova.ra, nova.dec, scramble = True, seed=jjj, inject = sample,
                        time_mask = [deltaT / 2. / 86400., nova.center_time + (nova.time_sigma / 2.)])
        TS.append(val['TS'][0]), ns.append(val['nsignal'][0]), gamma.append(val['gamma'][0])
        mean_ninj.append(nsig), ninj.append(val['n_inj'][0]), flux.append(inj.mu2flux(nsig))

sig_trials = {'TS': TS, 'ns': ns, 'gamma': gamma, 'mean_ninj': mean_ninj,
                'ninj': ninj, 'flux': flux}
sig_trials = pd.DataFrame(sig_trials)

if args.display:
    from tabulate import tabulate
    headers = sig_trials.columns
    ttable = tabulate(sig_trials, headers, tablefmt = 'fancy_grid')
    print(ttable)

flavor_str = 'all_flavor/' if args.allflavor else ''
sig_trials.to_pickle('/data/user/apizzuto/Nova/analysis_trials/differential_sensitivity/{}deltaT_{:.1e}_index_{}_spec_{}_lowE_{:.1e}_highE_{:.1e}.pkl'.format(flavor_str, deltaT, args.index, args.spec, inj_low_E, inj_high_E))
