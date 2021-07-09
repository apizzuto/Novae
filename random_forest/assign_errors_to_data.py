r'''Run this script on GRECO data files after the model has been
trained. This will add the extra columns as well as load the 
model and perform the regression'''

import numpy as np
from glob import glob
import argparse, sys
import pandas as pd
import healpy as hp
from numpy.lib.recfunctions import append_fields
from icecube import astro
from load_model import *

parser = argparse.ArgumentParser(description='Calculate opening angle between monopod and pegleg')
parser.add_argument('--infiles', type=str, required=True, help='Numpy file to read in')
args = parser.parse_args()

infiles = glob(args.infiles)
for infile in infiles:
    events = np.load(infile)

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

    appended_events['angErr'] = predictions
    np.save(infile[:-4] + '_with_angErr.npy', appended_events)
