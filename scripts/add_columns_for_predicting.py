import numpy as np
from glob import glob 
import pickle
import h5py
import inspect
import astropy
import pandas as pd
import healpy as hp
from icecube import astro
import argparse

parser = argparse.ArgumentParser(description='Calculate opening angle between monopod and pegleg')
parser.add_argument('--infile', type=str, required=True, help='Numpy file to read in')
args = parser.parse_args()

from numpy.lib.recfunctions import append_fields

#neutrinos = np.load('/home/mlarson/IC86_2018_MC.npy')

neutrinos = np.load(args.infile)
#Next few lines because monopod reco angles are swapped in original MC file
tmp_azi = neutrinos['monopod_zen'].copy()
tmp_zen = neutrinos['monopod_azi'].copy()
neutrinos['monopod_zen'] = tmp_zen
neutrinos['monopod_azi'] = tmp_azi

mzens = neutrinos['monopod_zen'].astype(float)
mazis = neutrinos['monopod_azi'].astype(float)
times = neutrinos['time'].astype(float)
#monopod_ra, monopod_dec = astro.dir_to_equa(neutrinos['monopod_zen'], neutrinos['monopod_azi'], neutrinos['time'])
monopod_ra, monopod_dec = astro.dir_to_equa(mzens, mazis, times)

monopod_pegleg_dpsi = hp.rotator.angdist([np.rad2deg(neutrinos['ra']), 
                                    np.rad2deg(neutrinos['dec'])],
                                   [np.rad2deg(monopod_ra), 
                                    np.rad2deg(monopod_dec)],
                                   lonlat = True
                                  )

newsavefile = append_fields(neutrinos, 'monopod_ra', monopod_ra, usemask=False)
newsavefile = append_fields(newsavefile, 'monopod_dec', monopod_dec, usemask=False)
newsavefile = append_fields(newsavefile, 'monopod_pegleg_dpsi', monopod_pegleg_dpsi, usemask=False)

np.save(args.infile[:-4] + '_with_delta_psi.npy', newsavefile)
