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
parser.add_argument('--deltaT', type=float, default = '1000.',
                    help='Time window in seconds')
parser.add_argument('--n', type=int, default=1000,
                    help='Number of trials per mean signal')
parser.add_argument('--index', type=int, default=1, help='Index of nova list')
parser.add_argument('--spec', type=str, default='SPL', help='Spectrum, either single' \
                            +' power law (SPL) or power law with cutoff (EPL)')
parser.add_argument('--display', default=False, action='store_true', help='Print results '\
                        +'for debugging')
parser.add_argument('--scale', default=1., type=float, help='Scale errors '\
                        +'down to see if it helps the fit')
parser.add_argument('--lowE', default=None, help='logE cut')
parser.add_argument('--maxSigma', default=None, help='maximum assigned error')
parser.add_argument('--pull', action='store_true', default=False, help='Pull correct')
parser.add_argument('--fixed', action='store_true', default=False, help='Fix gamma in likelihood')
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

fit_gamma = not args.fixed
#FIGURE OUT SEASON?
scale = args.scale if args.scale != 1. else None
llh = initialize_llh(nova, season=season, scale=scale, only_low_en=args.lowE, 
                    only_small_sigma=args.maxSigma, pull_corr=args.pull, fit_gamma = fit_gamma,
                    all_flavor = args.allflavor)
inj = initialize_injector(nova, llh)
print('Declination: {:.2f} \t Index: {}'.format(nova.dec*180. / np.pi, nova.gamma))

TS, ns, gamma, mean_ninj, ninj, flux = [], [], [], [], [], []
nsigs = np.unique(np.append(np.linspace(1, 20, 20), np.linspace(20., 100., 9)))
for nsig in nsigs:
    for jjj in range(args.n):
        ni, sample = inj.sample(nova.ra, nsig, poisson=False)
        val =  llh.scan(nova.ra, nova.dec, scramble = True, seed=jjj, inject = sample,
                        time_mask = [deltaT / 2. / 86400., nova.center_time + (nova.time_sigma / 2.)])
        TS.append(val['TS'][0]), ns.append(val['nsignal'][0])
        mean_ninj.append(nsig), ninj.append(val['n_inj'][0]), flux.append(inj.mu2flux(nsig))
        if fit_gamma:
            gamma.append(val['gamma'][0])
        else:
            gamma.append(nova.gamma)

sig_trials = {'TS': TS, 'ns': ns, 'gamma': gamma, 'mean_ninj': mean_ninj, 
                'ninj': ninj, 'flux': flux}
sig_trials = pd.DataFrame(sig_trials)

if args.display:
    from tabulate import tabulate
    headers = sig_trials.columns
    ttable = tabulate(sig_trials, headers, tablefmt = 'fancy_grid')
    print(ttable)

flavor_str = 'all_flavor/' if args.allflavor else ''
if scale is None and args.maxSigma is None and args.lowE is None and not args.pull and fit_gamma:
    sig_trials.to_pickle('/data/user/apizzuto/Nova/analysis_trials/fits/{}deltaT_{:.1e}_index_{}_spec_{}.pkl'.format(flavor_str, deltaT, args.index, args.spec))
elif args.pull:
    print('pull corrcted')
    add_str = 'scale_{:.2f}_'.format(scale) if scale is not None else ''
    add_str += 'lowE_{:.2f}_'.format(args.lowE) if args.lowE is not None else ''
    add_str += 'sigma_{:.2f}_'.format(args.maxSigma) if args.maxSigma is not None else ''
    add_str += 'fit_gamma_{}_'.format(fit_gamma)
    sig_trials.to_pickle('/data/user/apizzuto/Nova/analysis_trials/fits/pull/{}deltaT_{:.1e}_index_{}_spec_{}.pkl'.format(add_str, deltaT, args.index, args.spec))
elif scale is not None:
    print('scaled option')
    sig_trials.to_pickle('/data/user/apizzuto/Nova/analysis_trials/fits/scaled/fit_gamma_{}_scale_{:.2f}_deltaT_{:.1e}_index_{}_spec_{}.pkl'.format(fit_gamma, scale, deltaT, args.index, args.spec))
elif args.maxSigma is not None:
    print('maxSigma option')
    sig_trials.to_pickle('/data/user/apizzuto/Nova/analysis_trials/fits/cuts/fit_gamma_{}_sigma_{:.2f}_deltaT_{:.1e}_index_{}_spec_{}.pkl'.format(fit_gamma, float(args.maxSigma), deltaT, args.index, args.spec))
else:
    print('Energy cut option')
    sig_trials.to_pickle('/data/user/apizzuto/Nova/analysis_trials/fits/cuts/fit_gamma_{}_lowE_{:.2f}_deltaT_{:.1e}_index_{}_spec_{}.pkl'.format(fit_gamma, float(args.lowE), deltaT, args.index, args.spec))
