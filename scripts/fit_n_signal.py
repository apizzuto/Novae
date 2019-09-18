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
parser.add_argument('--sigma', type=str, default = '30',
                    help='Assigned angular uncertainty in degrees, or "RandomForest"')
parser.add_argument('--n', type=int, default=1000,
                    help='Number of trials per mean signal')
parser.add_argument('--index', type=int, default=1, help='Index of nova list')
parser.add_argument('--spec', type=str, default='SPL', help='Spectrum, either single' \
                            +' power law (SPL) or power law with cutoff (EPL)')
parser.add_argument('--display', default=False, action='store_true', help='Print results '\
                        +'for debugging')
args = parser.parse_args()

df = pd.read_pickle('/home/apizzuto/Nova/Novae_details_with_seasons.csv')
df = df.sort_values('Start Time (gamma ray)')

novae = []
seasons = []
if args.spec == 'EPL':
    for index, row in df.iterrows():
        a = Nova(row['Name'], -1*row['EPL gamma'], np.radians(row['RA (Deg)']), np.radians(row['Dec. (Deg)']), 
                row['Start Time (gamma ray)'], row['Duration (gamma ray)'], flux_norm=row['EPL flux']*1e-7, cutoff = row['EPL cutoff '])
        novae.append(a)
        seasons.append(row['GRECO season'])
elif args.spec == 'SPL':
    for index, row in df.iterrows():
        a = Nova(row['Name'], -1.*row['SPL gamma'], np.radians(row['RA (Deg)']), np.radians(row['Dec. (Deg)']), 
                row['Start Time (gamma ray)'], row['Duration (gamma ray)'], flux_norm=row['SPL flux']*1e-7, cutoff = None)
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

if args.sigma == 'RandomForest':
    raise ValueError("I still haven't implemented this option")
else:
    try:
        sigma = float(args.sigma) * np.pi / 180.
    except:
        print('Bad sigma argument, using 30 degrees')
        sigma = 30. * np.pi / 180.

#FIGURE OUT SEASON?
llh = initialize_llh(nova, sigma = sigma, season=season)
inj = initialize_injector(nova, llh)

results = None
nsigs = np.linspace(1, 20, 20)
for nsig in nsigs:
    result = llh.do_trials(args.n, src_ra = nova.ra, src_dec = nova.dec, injector = inj, mean_signal=int(nsig), poisson=False)
    result = append_fields(result, 'mean_ninj', [nsig]*len(result), usemask=False)
    result = append_fields(result, 'flux', [inj.mu2flux(int(nsig))]*len(result), usemask=False)
    names = result.dtype.names
    names = list(names)
    names.remove('spectrum')
    result = result[names]
    if results == None:
        results = result
    else:
        results = np.append(results, result)

if args.display:
    from tabulate import tabulate
    headers = results.dtype.names
    ttable = tabulate(results, headers, tablefmt = 'fancy_grid')
    print(ttable)
np.save('/data/user/apizzuto/Nova/analysis_trials/fits/index_{}_spec_{}_sigma_{}.npy'.format(args.index, args.spec, args.sigma), results)
