#!/usr/bin/env python

import numpy as np
import argparse
import pickle
from glob import glob

import csky as cy

parser = argparse.ArgumentParser(
    description='Sensitivity vs declination for a fixed time window')
parser.add_argument(
    '--deltaT', type=float, default=1000.,
    help='Time window in seconds')
parser.add_argument('--index', type=float, default=1, help='Spectral Index')
parser.add_argument(
    '--ntrials_bg', type=int, default=10000,
    help="Number of background trials")
parser.add_argument(
    '--ntrials_sig', type=int, default=500,
    help='Number of trials per signal strength')
args = parser.parse_args()

delta_t = args.deltaT
delta_t_days = delta_t / 86400.

ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
greco_ana = cy.get_analysis(
    cy.selections.repo,
    'version-002-p10',
    cy.selections.GRECOOnlineDataSpecs.GRECO_IC86_2012_2019,
    dir=ana_dir)

conf = {
    'ana': greco_ana,
    'extended': True,
    'space': "ps",
    'time': "transient",
    'sig': 'transient'}

result_dict = {
    'dec': [],
    'sens': [],
    'sens_nsig': []}

ra = 0.
mjd_start = 57200.
sin_decs = np.linspace(-0.95, 0.95, 20)
for dec in np.arcsin(sin_decs):
    src = cy.utils.Sources(
        ra=ra, dec=dec, mjd=mjd_start,
        sigma_t=0., t_100=delta_t_days)
    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    tr = cy.get_trial_runner(conf, ana=greco_ana, src=src)
    bg = cy.dists.Chi2TSD(tr.get_many_fits(args.ntrials_bg))

    tr = cy.get_trial_runner(
        conf, ana=greco_ana, src=src,
        inj_conf={'flux': cy.hyp.PowerLawFlux(args.index)})

    ########################################################################
    # SENSITIVITY CALCULATION
    ########################################################################
    nsig_step = 1. if delta_t < 864000. else 20.

    sensitivity = tr.find_n_sig(
        bg.median(), 0.9, batch_size=args.ntrials_sig,
        n_sig_step=nsig_step,
        max_batch_size=0, logging=True,
        n_bootstrap=1)
    sensitivity['E2dNdE'] = tr.to_E2dNdE(sensitivity, E0=1., unit=1e3)

    result_dict['dec'].append(dec)
    result_dict['sens'].append(sensitivity['E2dNdE'])
    result_dict['sens_nsig'].append(sensitivity['n_sig'])

delta_t_str = f"{delta_t:.2e}"
output_loc = "/data/user/apizzuto/Nova/csky_trials/sens_vs_dec/" \
    + f"sens_vs_dec_delta_t_{delta_t_str}_gamma_{args.index}_trials.pkl"

with open(output_loc, 'wb') as f:
    pickle.dump(result_dict, f)
