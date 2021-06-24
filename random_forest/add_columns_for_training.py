import numpy as np
import healpy as hp
from icecube import astro
import argparse

parser = argparse.ArgumentParser(description='Calculate opening angle between monopod and pegleg')
parser.add_argument('--infile', type=str, required=True, help='Numpy file to read in')
args = parser.parse_args()

from numpy.lib.recfunctions import append_fields

neutrinos = np.load(args.infile)
true_dpsi = hp.rotator.angdist([np.rad2deg(neutrinos['ra']), 
                                    np.rad2deg(neutrinos['dec'])],
                                   [np.rad2deg(neutrinos['trueRa']), 
                                    np.rad2deg(neutrinos['trueDec'])],
                                   lonlat = True
                                  )

monopod_ra, monopod_dec = astro.dir_to_equa(neutrinos['monopod_zen'].astype(float),
                                neutrinos['monopod_azi'].astype(float), neutrinos['time'].astype(float))

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

np.save(args.infile[:-4] + '_with_delta_psi.npy', newsavefile)
