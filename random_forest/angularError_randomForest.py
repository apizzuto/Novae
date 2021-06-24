#!/usr/bin/env python

r'''This script trains a random forest on numu MC
given a set of hyperparameters taken as command
line arguments'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import argparse
import pickle

#First, parse options to fix a time window. This script will calculate the 
parser = argparse.ArgumentParser(description = 'Nova Random Forest Grid Search')
parser.add_argument('--infile', type=str, required=True, help='Numpy file to read in')
parser.add_argument('--log', default=False, action='store_true',
                        help='Fit for log of separation')
parser.add_argument('--boot', default=False, action='store_true',
                        help='Bootsrap')
parser.add_argument('--minsamp', type=int, required=True, help='Min samples split for random forest')
args = parser.parse_args()

logSeparation = args.log

neutrinos = np.load(args.infile)
neutrinos_df = pd.DataFrame.from_dict(neutrinos)
# We train on numu NC and CC:
neutrinos_df = neutrinos_df[np.abs(neutrinos_df['ptype']) == 14]

neutrinos_df['int_type'] = np.where(neutrinos_df['iscc'] == False, 'NC', 'CC')
neutrinos_df = neutrinos_df.drop([
    'run', 'event', 'subevent', 'angErr', 'trueE', 'azi', 'monopod_azi',
    'trueRa', 'trueDec', 'time', 'ptype', 'iscc',
    'ra', 'dec', 'monopod_ra', 'monopod_dec', 'ow',
    'conv', 'prompt', 'astro', 'genie_gen_r', 
    'genie_gen_z', 'uncorrected_ow'], axis = 'columns')
old_names = neutrinos_df.columns
new_names = [on.replace('_', ' ') for on in old_names]
neutrinos_df.columns = new_names

neutrinos_df = neutrinos_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
scaled_nus = neutrinos_df.copy()
if logSeparation:
    scaled_nus['true dpsi'] = np.log10(neutrinos_df['true dpsi'])
else:
    scaled_nus['true dpsi'] = neutrinos_df['true dpsi']
scaled_nus['monopod zen'] = np.cos(neutrinos_df['monopod zen'])
scaled_nus['zen'] = np.cos(neutrinos_df['zen'])
scaled_nus['pidMonopodLLH'] = np.log10(neutrinos_df['pidMonopodLLH'])
scaled_nus = scaled_nus.drop(['int type'], axis = 'columns')

neutrinos_df = scaled_nus.copy()

feature_cols = list(neutrinos_df.keys())
feature_cols.remove('true dpsi')

X,y = neutrinos_df[feature_cols].values, neutrinos_df['true dpsi'].values

X_train, X_test, y_train, y_test =  train_test_split(X, y, 
                                                     test_size=0.5, 
                                                     random_state=1)

param_grid = {
'n_estimators': [int(x) for x in np.unique(np.append(np.linspace(20,100,5), np.linspace(200, 300, 2)))],
'max_features': [int(x) for x in np.linspace(1, len(feature_cols), 5)],
'max_depth': [4, 6, 8, 10, 12, 14, 16, 32],
'min_samples_split': [args.minsamp],
'bootstrap': [args.boot]
}

forest = RandomForestRegressor()

rf_search = GridSearchCV(estimator = forest,
                        param_grid = param_grid,
                        cv = 3,
                        n_jobs = 5,
                        verbose = 3)

rf_search.fit(X_train, y_train)

# Version 2.4 model is the same as v2.5 model because the MC did not change
version = 'v2.4'
outfile = '/data/user/apizzuto/Nova/RandomForests/{}/GridSearchResults_logSeparation_{}_bootstrap_{}_minsamples_{}'.format(version, logSeparation, args.boot, args.minsamp)

pickle.dump(rf_search, open(outfile, 'wb'))