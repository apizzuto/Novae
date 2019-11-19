import numpy        as np
import pandas       as pd
import scipy        as sp
import os
import astropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pickle

GeV = 1.
TeV = 1000. * GeV

model_path = '/data/user/apizzuto/Nova/RandomForests/'

def load_model(minsamp = 100, logSep = True, standardize = True):
    r'''
    Load already trained cross validated model. Default kwargs are
    those found to result in best model previously
    Parameters:
    -----------
    minsamp: int
        min_samples_split argument in sklearn.ensemble.RandomForestRegressor
    logSep: bool
        Train on log of the opening angle or on the opening angle
    standardize: bool
        Standardize data with sklearn.preprocessing.StandardScaler
    Returns:
    --------
    model: sklearn.ensemble.RandomForestRegressor
        Best model from a cross validated hyperparameter grid search
    '''
    cv = pickle.load(open(model_path + \ 
        'GridSearchResults_logSeparation_{}'.format(logSep) + \
        '_standardize_{}'.format(standardize) + \
        '_bootstrap_True_minsamples_{}'.format(minsamp)))
    best_model = cv.best_estimator_
    return best_model

def clean_data(path, logSep = True, standardize = True, 
                pid = None, CC_only = False):
    r'''
    Loads data directly from .i3 -> .npy output and prepares
    it for predictions
    Parameters:
    -----------
    path: str
        Path to .npy file
    logSep: bool
        Train on log of opening angle or opening angle
    standardize: bool
        Standardize data with sklearn.preprocessing.StandardScaler
    Returns:
    --------
    data: array-like (pd.dataframe?)
        Vectors which can be passed directly to `predict`
    '''
    neutrinos = np.load(path)
    neutrinos_df = pd.DataFrame.from_dict(neutrinos)
    if pid is not None:
        neutrinos_df = neutrinos_df[np.abs(neutrinos_df['ptype']) == 14]
    if CC_only:
        neutrinos_df['int_type'] = np.where(neutrinos_df['iscc'] == False, 'NC', 'CC')
    neutrinos_df = neutrinos_df.drop(['run', 'event', 'subevent', 'angErr', 'trueE', 'azi', 'monopod_azi',
                                  'trueRa', 'trueDec', 'time', 'ptype', 'iscc',
                                  'trueDeltaLLH', 'ra', 'dec', 'monopod_ra', 'monopod_dec', 'ow'], axis = 'columns')
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
    #scaled_nus['monopod pegleg dpsi'] = np.power(neutrinos_df['monopod pegleg dpsi'], 0.5)
    #scaled_nus['pidLength'] = np.where(neutrinos_df['pidLength'] > 0., np.log10(neutrinos_df['pidLength']), 0.)
    scaled_nus = scaled_nus.drop(['int type'], axis = 'columns')

    neutrinos_df = scaled_nus.copy()

    feature_cols = list(neutrinos_df.keys())
    feature_cols.remove('true dpsi')

    X, y = neutrinos_df[feature_cols].values, neutrinos_df['true dpsi'].values

    if standardize:
        stdsc = StandardScaler()
        X_train = stdsc.fit(X_train).transform(X_train)
        X_test = stdsc.transform(X_test)



def predict(data, model = None, logSep = True, standardize = True):
    r'''
    [Load and] make prediction with a trained model
    Parameters:
    -----------
    data: np.ndarray
        Data directly from .npy MC file
    model: None or sklearn.ensemble.RandomForestRegressor
        Best model from a cross validated hyperparameter grid search 
        loads if None
    logSep: bool
        Train on log of the opening angle or on the opening angle
    standardize: bool
        Standardize data with sklearn.preprocessing.StandardScaler
    Returns:
    --------
    predictions for opening angle (IN RADIANS)
    '''
