r'''
Script to initialize likelihood using Skylab
for Nova analysis
Author: Alex Pizzuto
Date: August, 2019
'''

import os
import sys
import time
import subprocess, pickle
import shutil
import dateutil.parser
import healpy as hp
import matplotlib as mpl
import numpy as np
from helper_functions import *
from astropy.time import Time
from scipy.special import erfinv
from skylab.datasets import Datasets
from skylab.llh_models import EnergyLLH
from skylab.ps_injector import PointSourceInjector
from skylab.ps_llh import PointSourceLLH
from skylab.spectral_models import PowerLaw
from skylab.temporal_models import BoxProfile, TemporalModel

import skylab
print skylab.__file__

mlarson_path = '/home/mlarson/GRECO/version-001-p00/'
GeV = 1.
TeV = 1000. * GeV

def initialize_llh(nova, scramble=True, dataset = mlarson_path, 
                        season = "IC86_2012", ncpu = 1, sigma = 30. * np.pi / 180., perfect_scale = 1.):
    r''' Initializes a point source llh

    Parameters:
    -----------
    nova: Nova object
        Nova which we wish to follow up
    Returns:
    --------
    llh: Skylab ps_llh object
    '''
    print("Initializing Point Source LLH in Skylab")
    start, stop = nova.center_time, nova.center_time + nova.time_sigma

    ###################### BEGIN DATASET ######################
    t0 = time.time()
    
    if dataset != mlarson_path:
        sample = dataset
        name = season
        exp, mc, livetime = Datasets[sample].season(name, floor=np.radians(0.2))
        sinDec_bins = Datasets[sample].sinDec_bins(name)
        energy_bins = Datasets[sample].energy_bins(name)
        grl         = Datasets[sample].grl(name)
    else:
        exp = np.load(mlarson_path + season + '_data.npy')
        mc = np.load(mlarson_path + 'IC86_2012_mc.npy')
        if sigma == 'perfect':
            deltapsi = deltaPsi(mc['dec'], mc['ra'], mc['trueDec'], mc['trueRa'])
            #exp['angErr'] = 30. * np.pi / 180.
            mc['angErr'] = deltapsi / 1.177 * perfect_scale #conversion factor from sigma to median
            exp['angErr'] = np.random.choice(mc['angErr'], size = len(exp))
        else:
            exp['angErr'] = sigma
            mc['angErr'] = sigma
        grl = np.load(mlarson_path + 'GRL/' + season + '_data.npy')
        livetime = np.sum(grl['livetime'])
        sinDec_bins = np.linspace(-1., 1., 21)
        energy_bins = np.logspace(0., 4., 21)
    ###################### END DATASET   ######################

    ##################### BEGIN LIKELIHOOD MODELS #####################
    llh_model = EnergyLLH(twodim_bins=[energy_bins, sinDec_bins],   # energy and sin(dec) binnings
                        allow_empty=True,                           # allow empty bins.
                        spectrum = PowerLaw(A=1, gamma=nova.gamma, E0=1., Ecut = nova.cutoff))

    box = TemporalModel(grl=grl,
                        poisson_llh=True,   # set to True for GRBLLH style likelihood with poisson term
                        days=10,            # use 10 days on either side of ontime for background calc
                        signal=BoxProfile(start, stop))

    llh = PointSourceLLH(exp,                   # array with data events
                            mc,                    # array with Monte Carlo events
                            livetime,              # total livetime of the data events
                            ncpu=ncpu,               # use 10 CPUs when computing trials
                            scramble=scramble,        # use scrambled data, set to False for unblinding
                            timescramble=scramble,     # use full time scrambling, not just RA scrambling
                            llh_model=llh_model,   # likelihood model
                            temporal_model=box,    # use box profile for temporal model
                            nsource_bounds=(0., 1e3),  # bounds on fitted number of signal events
                            nsource=1.)            # seed for nsignal fit

    print("Initializing Point Source COMPLETE")
    print("LLH Initialization took {} seconds\n\n".format(time.time() - t0))
    return llh

def initialize_injector(nova, llh, seed=123):
    r'''Method to make relevant injector in Skylab, done for analysis
    checks as well as for calculating sensitivities

    Parameters:
    -----------
    nova: Nova object
    llh: initialized Skylab PointSource Likelihood object
    Returns:
    --------
    inj: Skylab injector object'''
    print("Initializing Point Source Injector")
    inj = PointSourceInjector(gamma = nova.gamma, E0 = 1., Ecut = nova.cutoff) 
    inj.fill(nova.dec, llh.exp, llh.mc, llh.livetime, temporal_model=llh.temporal_model)
    return inj
