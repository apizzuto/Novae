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
from glob import glob
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

#mlarson_path = '/home/mlarson/GRECO/version-001-p00/'
mlarson_path = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/'

GeV = 1.
TeV = 1000. * GeV

def fields_view(array, fields):
    return array.getfield(np.dtype(
        {name: array.dtype.fields[name] for name in fields}
    ))

exp_fields = ('run', 'event', 'subevent', 'time', 'azi', 'zen', 'ra', 'dec', 'angErr', 'logE')
mc_fields = ('run', 'event', 'subevent', 'time', 'azi', 'zen', 'ra', 'dec', 'angErr', 'logE', 'trueRa', 'trueDec', 'trueE', 'ow')

def initialize_llh(nova, scramble=True, dataset = mlarson_path, fit_gamma = True, 
                        season = "IC86_2012", ncpu = 1, sigma = 30. * np.pi / 180., perfect_scale = 1.
                        verbose=True):
    r''' Initializes a point source llh

    Parameters:
    -----------
    nova: Nova object
        Nova which we wish to follow up
    Returns:
    --------
    llh: Skylab ps_llh object
    '''
    if verbose:
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
        exps = sorted(glob(mlarson_path + '*.data_with_angErr.npy'))
        exp = np.load(exps[0])
        for e in exps[1:]:
            exp = np.append(exp, np.load(e))
        #exp = np.load(mlarson_path + season + '.data_with_angErr.npy')
        mc = np.load(mlarson_path + 'IC86_2012.numu_with_angErr.npy')
        #if sigma == 'perfect':
        #    deltapsi = deltaPsi(mc['dec'], mc['ra'], mc['trueDec'], mc['trueRa'])
            #exp['angErr'] = 30. * np.pi / 180.
        #    mc['angErr'] = deltapsi / 1.177 * perfect_scale #conversion factor from sigma to median
        #    mc['angErr'] *= np.power(10., np.random.randn(len(mc['angErr'])) / 3. )
        #    exp['angErr'] = np.random.choice(mc['angErr'], size = len(exp))
        #else:
        #    exp['angErr'] = sigma
        #    mc['angErr'] = sigma
        grls = sorted(glob(mlarson_path + 'GRL/*.data.npy'))
        grl = np.load(grls[0])
        for g in grls[1:]:
            grl = np.append(grl, np.load(g))
        #grl = np.load(mlarson_path + 'GRL/' + season + '.data.npy')
        livetime = np.sum(grl['livetime'])
        sinDec_bins = np.linspace(-1., 1., 26)
        energy_bins = np.linspace(0., 5., 26)
    ###################### END DATASET   ######################
    exp.dtype.names = [item.encode('utf-8') for item in exp.dtype.names]
    mc.dtype.names = [item.encode('utf-8') for item in mc.dtype.names]
    exp = fields_view(exp, exp_fields)
    mc = fields_view(mc, mc_fields)
    mc = np.rec.fromrecords(mc, formats = ['<i8','<i8','<i8','<f8','<f4','<f4','<f4','<f4','<f4','<f4','<f4','<f4','<f4','<f4'], names=mc.dtype.names)
    exp = np.rec.fromrecords(exp, formats = ['<i8','<i8','<i8','<f8','<f4','<f4','<f4','<f4','<f4','<f4'], names=exp.dtype.names)
    #print(exp['angErr'][:50])
    #print(mc['angErr'][:50])

    factor = 1.0
    exp['angErr'] *= factor
    mc['angErr'] *= factor

    #exp['angErr'] = np.radians(90.)
    #mc['angErr'] = np.radians(90.)
    ##################### BEGIN LIKELIHOOD MODELS #####################
    if fit_gamma:
        if verbose:
            print "Fitting gamma"
        llh_model = EnergyLLH(twodim_bins=[energy_bins, sinDec_bins],   # energy and sin(dec) binnings
                            allow_empty=True,                           # allow empty bins.
                            seed=2.5,
                            bounds = [1., 5.])
    else:
        if verbose:
            print "not fitting gamma"
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

    if verbose:
        print("Initializing Point Source COMPLETE")
        print("LLH Initialization took {} seconds\n\n".format(time.time() - t0))
    return llh

def initialize_injector(nova, llh, seed=123, verbose=True):
    r'''Method to make relevant injector in Skylab, done for analysis
    checks as well as for calculating sensitivities

    Parameters:
    -----------
    nova: Nova object
    llh: initialized Skylab PointSource Likelihood object
    Returns:
    --------
    inj: Skylab injector object'''
    if verbose:
        print("Initializing Point Source Injector")
    inj = PointSourceInjector(gamma = nova.gamma, E0 = 1., Ecut = nova.cutoff) 
    inj.fill(nova.dec, llh.exp, llh.mc, llh.livetime, temporal_model=llh.temporal_model)
    return inj
