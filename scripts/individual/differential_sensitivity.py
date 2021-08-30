#!/usr/bin/env python

import numpy as np
import argparse
import csky as cy
import pickle
from glob import glob

parser = argparse.ArgumentParser(
    description='Sensitivity for individual gamma-ray detected novae')
parser.add_argument(
    '--deltaT', type=float, default=1000.,
    help='Time window in seconds')
parser.add_argument(
    '--ntrials_bg', type=int, default=10000,
    help="Number of background trials")
parser.add_argument(
    '--ntrials_sig', type=int, default=500,
    help='Number of trials per signal strength')
parser.add_argument(
    '--dec', type=float, required=True,
    help='declination in degrees')
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

ra = 0.0
dec = np.radians(args.dec)
mjd_start = 58200.  # This is just kinda random in 2017
mjd_stop = mjd_start + delta_t_days

conf = {
    'ana': greco_ana,
    'extended': True,
    'space': "ps",
    'time': "transient",
    'sig': 'transient'}

src = cy.utils.Sources(
    ra=ra, dec=dec, mjd=mjd_start,
    sigma_t=0.,
    t_100=delta_t_days)
cy.CONF['src'] = src
cy.CONF['mp_cpus'] = 5

tr = cy.get_trial_runner(conf, ana=greco_ana, src=src)
bg = cy.dists.Chi2TSD(tr.get_many_fits(args.ntrials_bg))

Ebins = 10**np.r_[0.:4.1:1.0]
trs = [
    cy.get_trial_runner(
        conf,
        ana=greco_ana, src=src,
        flux=cy.hyp.PowerLawFlux(2.0, energy_range=(Emin, Emax)))
    for (Emin, Emax) in zip(Ebins[:-1], Ebins[1:])]

result = {}

for ii, tr in enumerate(trs):
    try:
        ######################################################################
        # SENSITIVITY CALCULATION
        ######################################################################
        sensitivity = tr.find_n_sig(
            bg.median(), 0.9, batch_size=args.ntrials_sig,
            n_sig_step=3, max_batch_size=0, logging=True,
            n_bootstrap=1)
        sensitivity['E2dNdE'] = tr.to_E2dNdE(
            sensitivity, E0=Ebins[ii]/1e3, unit=1e3)
        result[f'sensitivity_{Ebins[ii]:.1f}_{Ebins[ii+1]:.1f}'] = sensitivity
    except Exception as e:
        print(f"Could not do energy range {Ebins[ii]} to {Ebins[ii+1]}")

result['bg'] = bg
result['settings'] = args
result['source_info'] = {
    'ra': ra, 'dec': dec,
    'mjd_start': mjd_start, 'mjd_stop': mjd_stop}

delta_t_str = f"{delta_t:.2e}"
output_loc = "/data/user/apizzuto/Nova/csky_trials/differential_sens/" \
    + f"dec_{dec:.1f}_delta_t_{delta_t_str}_trials.pkl"

with open(output_loc, 'wb') as f:
    pickle.dump(result, f)
