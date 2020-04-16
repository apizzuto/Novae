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
mlarson_path = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.2/'

GeV = 1.
TeV = 1000. * GeV

def rename_fields(data, subs):
    r"""Substitute data field names for new names

    Parameters
    ----------
    data : structured array
        Structured data array with original field names
    subs : list
        List of (name, new name) tuples

    Returns
    -------
    data : structured array
        Structured data array with new field names
    """
    dnames = list(data.dtype.names)
    for name, rename in subs:
        dnames[dnames.index(name)] = rename
    data.dtype.names = tuple(dnames)
    return data



def fields_view(array, fields):
    return array.getfield(np.dtype(
        {name: array.dtype.fields[name] for name in fields}
    ))

exp_fields = ('run', 'event', 'subevent', 'time', 'azi', 'zen', 'ra', 'dec', 'angErr', 'logE')
mc_fields = ('run', 'event', 'subevent', 'time', 'azi', 'zen', 'ra', 'dec', 'angErr', 'logE', 'trueRa', 'trueDec', 'trueE', 'ow')

def initialize_llh(nova, scramble=True, dataset = mlarson_path, fit_gamma = True, 
                        season = "IC86_2012", ncpu = 1, scale=None,
                        verbose=True, only_low_en = None, only_small_sigma = None, pull_corr=False):
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
        pc_str = 'pull_corr_numu/' if pull_corr else ''
        exps = sorted(glob(mlarson_path + pc_str  + '*.data_with_angErr.npy'))
        exp = np.load(exps[0])
        for e in exps[1:]:
            exp = np.append(exp, np.load(e))
        mcfile = glob(mlarson_path + pc_str + 'IC86_2012.numu*.npy')[0]
        mc = np.load(mcfile)
        
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

    if "angErr" in exp.dtype.names:
        subs = [('angErr', 'sigma'),
                ('azi', 'azimuth'),
                ('zen', 'zenith')]
        exp = rename_fields(exp, subs)
        mc = rename_fields(mc, subs)

    #I PROBABLY NEED TO FIX THINGS ABOUT RATES IN THE GRL IF I'M CUTTING OUT EVENTS
    if only_low_en is not None:
        start_size = float(len(exp['logE']))
        exp = exp[exp['logE'] > float(only_low_en)]
        mc = mc[mc['logE'] > float(only_low_en)]
        grl['events'] = (grl['events']*len(exp['logE']) / start_size).astype(int)

    if only_small_sigma is not None:
        start_size = float(len(exp['logE']))
        exp = exp[exp['sigma'] < float(only_small_sigma)]
        mc = mc[mc['sigma'] < float(only_small_sigma)]
        grl['events'] = (grl['events']*len(exp['logE']) / start_size).astype(int)


    if scale is not None:
        exp['sigma'] *= scale
        mc['sigma'] *= scale

    ##################### BEGIN LIKELIHOOD MODELS #####################
    print fit_gamma
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

def initialize_injector(nova, llh, seed=123, verbose=True, inj_e_range=(0., np.inf),
        fixed_inj_gamma = None):
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
    inj_gamma = nova.gamma if fixed_inj_gamma is None else fixed_inj_gamma
    inj_cut = nova.cutoff if fixed_inj_gamma is None else None
    inj = PointSourceInjector(gamma = inj_gamma, E0 = 1., Ecut = inj_cut, e_range=inj_e_range) 
    inj.fill(nova.dec, llh.exp, llh.mc, llh.livetime, temporal_model=llh.temporal_model)
    return inj
