import numpy as np
from glob import glob 
import pickle
import h5py
import inspect
import astropy
#import seaborn as sns
import pandas as pd
import healpy as hp
from icecube import astro

from numpy.lib.recfunctions import append_fields

neutrinos = np.load('/home/mlarson/IC86_2018_MC.npy')
true_dpsi = hp.rotator.angdist([np.rad2deg(neutrinos['ra']), 
                                    np.rad2deg(neutrinos['dec'])],
                                   [np.rad2deg(neutrinos['trueRa']), 
                                    np.rad2deg(neutrinos['trueDec'])],
                                   lonlat = True
                                  )

#Next few lines because monopod reco angles are swapped in original MC file
tmp_azi = neutrinos['monopod_zen'].copy()
tmp_zen = neutrinos['monopod_azi'].copy()
neutrinos['monopod_zen'] = tmp_zen
neutrinos['monopod_azi'] = tmp_azi

monopod_ra, monopod_dec = astro.dir_to_equa(neutrinos['monopod_zen'], neutrinos['monopod_azi'], neutrinos['time'])
monopod_pegleg_dpsi = hp.rotator.angdist([np.rad2deg(neutrinos['ra']), 
                                    np.rad2deg(neutrinos['dec'])],
                                   [np.rad2deg(monopod_ra), 
                                    np.rad2deg(monopod_dec)],
                                   lonlat = True
                                  )

newsavefile = append_fields(neutrinos, 'monopod_ra', monopod_ra, usemask=False)
newsavefile = append_fields(newsavefile, 'monopod_dec', monopod_dec, usemask=False)
newsavefile = append_fields(newsavefile, 'monopod_pegleg_dpsi', monopod_pegleg_dpsi, usemask=False)
newsavefile = append_fields(newsavefile, 'true_dpsi', true_dpsi, usemask=False)

np.save('/data/user/apizzuto/Nova/RandomForests/IC86_2012-2018MC_with_dpsi_with_Nstring.npy', newsavefile)
