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

parser = argparse.ArgumentParser(description='Novae background')
parser.add_argument('--deltaT', type=float, default = 1000.,
                    help='Time window in seconds')
parser.add_argument('--n', type=int, default=100000,
                    help='Number of trials')
parser.add_argument('--index', type=int, default=1, help='Index of nova list')
parser.add_argument('--spec', type=str, default='SPL', help='Spectrum, either single' \
                            +' power law (SPL) or power law with cutoff (EPL)')
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

#FIGURE OUT SEASON?
llh = initialize_llh(nova, season=season)
TS, ns, gamma, = [], [], []
for jjj in range(args.n):
    val =  llh.scan(nova.ra,nova.dec, scramble = True, seed=jjj, 
                    time_mask = [deltaT / 2. / 86400., nova.center_time + (nova.time_sigma / 2.)])
    TS.append(val['TS'][0]), ns.append(val['nsignal'][0]), gamma.append(val['gamma'][0])
bg_trials = {'TS': TS, 'ns': ns, 'gamma': gamma}
np.save('/data/user/apizzuto/Nova/analysis_trials/bg/kent/deltaT_{:.1e}_index_{}_spec_{}.npy'.format(deltaT, args.index, args.spec), bg_trials)
