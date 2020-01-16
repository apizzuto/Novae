import numpy as np
from glob import glob
import pickle, h5py, inspect, astropy, argparse, sys
import pandas as pd
import healpy as hp
from numpy.lib.recfunctions import append_fields
from icecube import astro
sys.path.append('/home/apizzuto/Nova/scripts/')
from load_model import *

parser = argparse.ArgumentParser(description='Calculate opening angle between monopod and pegleg')
parser.add_argument('--infiles', type=str, required=True, help='Numpy file to read in')
args = parser.parse_args()

infiles = glob(args.infiles)
for infile in infiles:
    events = np.load(infile)

    if np.max(events['monopod_zen']) > np.max(events['monopod_azi']):
        #Next few lines because monopod reco angles are swapped in original MC file
        tmp_azi = events['monopod_zen'].copy()
        tmp_zen = events['monopod_azi'].copy()
        events['monopod_zen'] = tmp_zen
        events['monopod_azi'] = tmp_azi

    monopod_ra, monopod_dec = astro.dir_to_equa(events['monopod_zen'].astype(float), 
                        events['monopod_azi'].astype(float), events['time'].astype(float))

    monopod_pegleg_dpsi = hp.rotator.angdist([np.rad2deg(events['ra']),
                                        np.rad2deg(events['dec'])],
                                       [np.rad2deg(monopod_ra),
                                        np.rad2deg(monopod_dec)],
                                       lonlat = True
                                      )

    #Add some fields to the numpy recarray
    appended_events = append_fields(events, 'monopod_ra', monopod_ra, usemask=False)
    appended_events = append_fields(appended_events, 'monopod_dec', monopod_dec, usemask=False)
    appended_events = append_fields(appended_events, 'monopod_pegleg_dpsi', monopod_pegleg_dpsi, usemask=False)
    df = pd.DataFrame(appended_events)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    appended_events = df.to_records()

    forest = load_model() #no arguments defaults to optimal model
    X = clean_data(appended_events)
    predictions = predict_from_model(X, model = forest)

    #df['angErr'] = predictions
    appended_events['angErr'] = predictions
    np.save(infile[:-4] + '_with_angErr.npy', appended_events)
