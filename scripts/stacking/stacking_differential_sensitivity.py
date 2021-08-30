#!/usr/bin/env python

import numpy as np
import argparse
import csky as cy
import pickle

import sys
sys.path.append('/home/apizzuto/Nova/scripts/stacking/')
from source_config import *


def run_differential_sens(args):
    r"""Look at output background trials and run the signal
    trials for a stacked nova analysis
    """
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

    only_gamma = not args.all_nova
    weighting_scheme = args.weighting
    src, sample_str = get_sources(only_gamma, weighting_scheme, delta_t_days)

    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    def ndarray_to_Chi2TSD(trials):
        return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

    bg = cy.bk.get_all(
        '/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/bg/',
        '{}_delta_t_{:.2e}_seed_*.npy'.format(
            sample_str, delta_t),
        merge=np.concatenate,
        post_convert=ndarray_to_Chi2TSD
        )

    Ebins = 10**np.r_[0.:4.1:1.0]
    trs = [
        cy.get_trial_runner(
            conf,
            ana=greco_ana, src=src,
            flux=cy.hyp.PowerLawFlux(2.0, energy_range=(Emin, Emax)))
        for (Emin, Emax) in zip(Ebins[:-1], Ebins[1:])
    ]

    if delta_t < 1e3:
        n_sig_step = 5
    elif delta_t <= 86400.:
        n_sig_step = 7
    elif delta_t <= 86400:
        n_sig_step = 15
    else:
        n_sig_step = 15

    result = {}

    for ii, tr in enumerate(trs):
        try:
            #################################################################
            # SENSITIVITY CALCULATION
            #################################################################
            beta = 0.9
            sensitivity = tr.find_n_sig(
                bg.median(), beta, batch_size=args.ntrials_sig,
                n_sig_step=n_sig_step, max_batch_size=0,
                logging=True, n_bootstrap=1)

            sensitivity['E2dNdE'] = tr.to_E2dNdE(
                sensitivity, E0=Ebins[ii]/1e3, unit=1e3)
            tmp_key = f'sensitivity_{Ebins[ii]:.1f}_{Ebins[ii+1]:.1f}'
            result[tmp_key] = sensitivity
        except Exception as e:
            print(f"Could not do energy range {Ebins[ii]} to {Ebins[ii+1]}")

    result['bg'] = bg
    result['settings'] = args
    result['source_info'] = {
        'ra': list(src.ra), 'dec': list(src.dec), 'mjd': list(src.mjds)}

    with open(
        '/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/'
        + 'differential_sens/' +
        '{}_delta_t_{:.2e}.pkl'.format(
            sample_str, delta_t), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Novae stacked analysis')
    parser.add_argument(
        '--deltaT', type=float, default=86400.,
        help='Time window in seconds')
    parser.add_argument(
        '--ntrials_sig', type=float, default=250,
        help="Number of signal trials")
    parser.add_argument(
        '--seed', type=int, default=123,
        help="Random number seed")
    parser.add_argument(
        '--all_nova', default=False, action='store_true',
        help='Raise this flag if you want to include '
             + 'the optically detected novae'
    )
    parser.add_argument(
        '--weighting', default="optical",
        help="Weighting scheme. Choose between 'optical' and 'gamma'"
    )
    args = parser.parse_args()

    run_differential_sens(args)
