#!/usr/bin/env python

import numpy as np
from glob import glob as glob
import h5py
import inspect
import astropy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
import argparse
import pickle

#First, parse options to fix a time window. This script will calculate the 
#passing fraction for all levels of injected flux/fluence for this window
parser = argparse.ArgumentParser(description = 'Nova Random Forest Grid Search')
parser.add_argument('--a', type=bool, required=True, help='include azimuth information')
parser.add_argument('--s', type=bool, required=True, help='Standardize Data')
parser.add_argument('--log', type=bool, required=True, help='Fit for log of separation')
args = parser.parse_args()

standardize = args.s
includeAzi = args.a
logSeparation = args.log



FileNames = ['/home/apizzuto/Nova/GRECO_nugen_numu_LE_with_PeglegCasc.h5', '/home/apizzuto/Nova/GRECO_nugen_numu_ME_with_PeglegCasc.h5', '/home/apizzuto/Nova/GRECO_genie_numu_with_PeglegCascade.h5']
NFiles = [4879, 644, 735]

#Some helpful functions to read in from h5 files and mask arrays

def getTruth(neutrinoKey, weightKey):
    trueDict = {}
    for key in ['energy', 'zenith', 'azimuth', 'type', 
                'OneWeight', 'NEvents', 'MaxEnergyLog', 
                'MinEnergyLog', 'OneWeightbyNGen']:
        trueDict[key] = np.array([])
        
    for i in range(3):
        f = h5py.File(FileNames[i], 'r')
        for item in ['energy', 'zenith', 'azimuth', 'type']:
            trueDict[item] = np.append(trueDict[item], f[neutrinoKey][item])
        for item in ['OneWeight', 'NEvents', 'MaxEnergyLog', 'MinEnergyLog']:
            trueDict[item] = np.append(trueDict[item], f[weightKey][item])
        trueDict['OneWeightbyNGen'] = np.append(trueDict['OneWeightbyNGen'], 
                                                f[weightKey]['OneWeight'] / (f[weightKey]['NEvents'] * NFiles[i]) )
    return trueDict

def getRecos(recoName, recoFits):
    recoDict = {}
    for key in ['energy', 'zenith', 'azimuth',
               'logl', 'rlogl']:
        recoDict[key] = np.array([])
        
    recoVars = ['energy', 'zenith', 'azimuth']
    recoParams = ['logl', 'rlogl']
    
    for i in range(3):
        f = h5py.File(FileNames[i], 'r')
        for item in recoVars:
            recoDict[item] = np.append(recoDict[item], f[recoName][item])
        for item in recoParams:
            recoDict[item] = np.append(recoDict[item], f[recoFits][item])
            
    angularSeparation(recoDict, true)
    return recoDict

def angularSeparation(reco, truth):
    reco['angErr'] = np.arccos(np.sin(reco['zenith'])*np.sin(truth['zenith']) +
                        (np.cos(reco['zenith']) *np.cos(truth['zenith']) * np.cos(reco['azimuth'] - truth['azimuth'])) ) 

def maskAllRecos():
    mask = np.zeros_like(Pegleg_track['energy'])
    for reco in [Pegleg_track, Pegleg_cascade, Monopod, true]:
        mask += np.isnan(reco['energy'])
        mask += np.isnan(reco['zenith'])
        mask += np.isinf(reco['zenith'])
        mask += np.isinf(reco['energy']) #get rid of this if energy = 0 is helpful
    print np.unique(mask)
    print mask
    mask = map(bool, mask)
    mask = [not x for x in mask]
    for reco in [Pegleg_track, Pegleg_cascade, Monopod, true]:
        for key in reco.keys():
            reco[key] = reco[key][mask]


true = getTruth('MCNeutrino', 'I3MCWeightDict')
Pegleg_track = getRecos('Pegleg_Fit_MN_tol10Track', 'Pegleg_Fit_MN_tol10FitParams')
Pegleg_cascade = getRecos('Pegleg_Fit_MN_tol10HDCasc', 'Monopod_bestFitParams')
Monopod = getRecos('Monopod_best', 'Monopod_bestFitParams')
maskAllRecos()


allRecoDict = {}
for key in Monopod.keys():
    if "Monopod" in key:
        continue
    if "Pegleg" in key:
        continue
    if "delta" in key:
        continue
    newKey = r'Monopod_' + key
    allRecoDict[newKey] = Monopod[key]
for key in Pegleg_track.keys():
    newKey = r'Pegleg_' + key
    allRecoDict[newKey] = Pegleg_track[key]
allRecoDict['trueEnergy'] = true['energy']
allRecoDict['weight'] = true['OneWeightbyNGen']
allRecoDict[r'deltaLLH'] = Monopod['logl'] - Pegleg_track['logl']
df = pd.DataFrame.from_dict(allRecoDict)


for key in df.keys():
    if "energy" in key:
        df[key] = np.log10(df[key])
    if "zenith" in key:
        df[key] = np.cos(df[key])
    if "angErr" in key and logSeparation:
        df[key] = np.log10(df[key])
        
df = df.replace([np.inf, -np.inf], np.nan) #Some of these result in log(0) -> -inf, so remask
df = df.dropna(axis=0) 


feature_cols = list(df.keys())
feature_cols.remove('Monopod_angErr')
feature_cols.remove('Pegleg_angErr')
if not includeAzi:
    feature_cols.remove('Monopod_azimuth')
    feature_cols.remove('Pegleg_azimuth')
feature_cols.remove('trueEnergy')
feature_cols.remove('weight')

X,y = df[feature_cols].values, df['Pegleg_angErr'].values

X_train, X_test, y_train, y_test =  train_test_split(X, y, 
                                                     test_size=0.5, 
                                                     random_state=1) 

if standardize:
    stdsc = StandardScaler()
    X_train = stdsc.fit(X_train).transform(X_train)
    X_test = stdsc.transform(X_test)


param_grid = {
'n_estimators': [int(x) for x in np.linspace(10,100,10)],
'max_features': [int(x) for x in np.linspace(1, len(feature_cols), 5)],
'max_depth': [int(x) for x in np.linspace(3,20,10)],
'min_samples_split': [100, 1000, 10000, 10000],
'bootstrap': [True, False]
}

forest = RandomForestRegressor()

#weight = df['weight'] * np.power(df['trueEnergy'], -2)

rf_search = GridSearchCV(estimator = forest,
                        param_grid = param_grid,
                        cv = 3,
                        n_jobs = 5,
                        verbose = 3)

rf_search.fit(X_train, y_train)

outfile = '/data/user/apizzuto/Nova/RandomForests/GridSearchResults_azimuth_{}_logSeparation_{}_standardize_{}'.format(includeAzi, logSeparation, standardize)

pickle.dump(rf_search, open(outfile, 'wb'))

