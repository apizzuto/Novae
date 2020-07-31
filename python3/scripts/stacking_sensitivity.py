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


parser = argparse.ArgumentParser(description='Novae background')
parser.add_argument('--deltaT', type=float, default = 1000.,
                    help='Time window in seconds')
parser.add_argument('--index', type=float, default=1, help='Spectral Index')
parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
args = parser.parse_args()

delta_t = args.deltaT
delta_t_days = delta_t / 86400.

greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.2/'

data_fs = sorted(glob(greco_base + 'IC86_20*data_with_angErr.npy'))
exp = [np.load(data) for data in data_fs]
exp = np.hstack(exp)
mc = np.load(greco_base + 'IC86_2012.numu_with_angErr.npy')
grls = sorted(glob(greco_base + 'GRL/IC86_20*data.npy'))
grl = [np.load(g) for g in grls]
grl = np.hstack(grl)

if args.minLogE is not None:
    exp_msk = exp['logE'] > args.minLogE
    exp = exp[exp_msk]
    mc_msk = mc['logE'] > args.minLogE
    mc = mc[mc_msk]

greco = cy.selections.CustomDataSpecs.CustomDataSpec(exp, mc, np.sum(grl['livetime']), 
                                                     np.linspace(-1., 1., 31),
                                                     np.linspace(0., 4., 31), 
                                                     grl=grl, key='GRECOv2.2', cascades=True)

ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)

tab = Table.read('/home/apizzuto/Nova/source_list/appendix.tex')
df = tab.to_pandas()
coords = SkyCoord(frame="galactic", l=df['$l$']*u.degree, b=df['$b$']*u.degree)
equatorial = coords.icrs
df['ra'] = equatorial.ra.deg
df['dec'] = equatorial.dec.deg
df = df.replace(['-'], np.nan)

#df = df[~df['gamma']]
df['mjd'] = np.array([Time(pt, '%Y-%m-%d').mjd for pt in df['Peak Time']])
during_greco = (df['mjd'] > greco_ana.mjd_min) & (df['mjd'] + delta_t_days < greco_ana.mjd_max)
df = df[during_greco]

ras = df['ra']
decs = df['dec']
delta_ts = np.ones_like(ras) * delta_t_days

conf = {'extended': True,
       'space': "ps",
        'time': "transient",
        'sig': 'transient',
       }

src = cy.utils.Sources(ra=np.radians(ras), 
                       dec=np.radians(decs), 
                       mjd=df['mjd'], 
                       sigma_t=np.zeros_like(delta_ts), 
                       t_100=delta_ts)

cy.CONF['src'] = src
cy.CONF['mp_cpus'] = 5

tr = cy.get_trial_runner(conf, ana=greco_ana, src=src)
n_trials = 5000
bg = cy.dists.Chi2TSD(tr.get_many_fits(n_trials))

tr = cy.get_trial_runner(conf, ana=greco_ana, src=src, inj_conf={'flux': cy.hyp.PowerLawFlux(args.index)})

beta = 0.9

result = tr.find_n_sig(bg.median(), beta, 
                       batch_size=250,
                       n_sig_step=5,
                       max_batch_size=0, 
                       logging=True, 
                       n_bootstrap=1)

result['E2dNdE'] = tr.to_E2dNdE(result, E0=1., unit=1e3)
result['bg'] = bg

add_str = 'minLogE_{:.1f}'.format(args.minLogE) if args.minLogE is not None else ''
with open('/home/apizzuto/Nova/python3/scripts/stacking_sens_res/delta_t_{:.2e}_gamma_{}{}.pkl'.format(delta_t, args.index, add_str), 'wb') as f:
    pickle.dump(result, f)



