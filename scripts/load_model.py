import numpy        as np
import pandas       as pd
import astropy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

df_feature_cols = ['nstring', 'nchannel', 'zen', 'logE', 'cascade energy',
         'monopod zen', 'pidDeltaLLH', 'pidPeglegLLH', 'pidMonopodLLH',
         'pidLength', 'monopod pegleg dpsi']

model_path = '/data/user/apizzuto/Nova/RandomForests/'

def load_model(minsamp = 100, logSep = True, standardize = False):
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
    cv = pickle.load(open(model_path +  
        'GridSearchResults_logSeparation_{}'.format(logSep) + \
        '_standardize_{}'.format(standardize) + \
        '_bootstrap_True_minsamples_{}'.format(minsamp)))
    best_model = cv.best_estimator_
    return best_model

def clean_monte_carlo(path, logSep = True, standardize = False, 
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
        neutrinos_df = neutrinos_df[np.abs(neutrinos_df['ptype']) == pid]
    #if CC_only:
    neutrinos_df['int_type'] = np.where(neutrinos_df['iscc'] == False, 'NC', 'CC')
    neutrinos_df = neutrinos_df.drop(['run', 'event', 'subevent', 'angErr', 'trueE', 'azi', 'monopod_azi',
                                  'trueRa', 'trueDec', 'time', 'ptype', 'iscc',
                                  'trueDeltaLLH', 'ra', 'dec', 'monopod_ra', 'monopod_dec', 'ow'], axis = 'columns')
    old_names = neutrinos_df.columns
    new_names = [on.replace('_', ' ') for on in old_names]
    neutrinos_df.columns = new_names

    neutrinos_df = neutrinos_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    scaled_nus = neutrinos_df.copy()
    if logSep:
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

    X, y = neutrinos_df[feature_cols].values, neutrinos_df['true dpsi'].values

    if standardize:
        stdsc = StandardScaler()
        X = stdsc.fit(X).transform(X)

    return X, y

def clean_data(data, logSep = True, standardize = False):
    r'''
    Given data as a .npy array with all relevant columns except
    circularized sigma, prepare data to be passed to model to 
    calculate circularized sigma
    Parameters:
    -----------
    data: np.ndarray
        formatted numpy array (skylab format-like)
    logSep: bool
        Train on log of the opening angle or on the opening angle
    standardize: bool
        Standardize data with sklearn.preprocessing.StandardScaler
    Returns:
    --------
    X: matrix
        Array to pass directly to model to make predictions
    '''
    events_df = pd.DataFrame.from_dict(data)
    old_names = events_df.columns
    new_names = [on.replace('_', ' ') for on in old_names]
    events_df.columns = new_names
    events_df = events_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    scaled_events = events_df.copy()
    scaled_events['monopod zen'] = np.cos(events_df['monopod zen'])
    scaled_events['zen'] = np.cos(events_df['zen'])
    scaled_events['pidMonopodLLH'] = np.log10(events_df['pidMonopodLLH'])
    events_df = scaled_events.copy()
    X = events_df[df_feature_cols].values
    if standardize:
        stdsc = StandardScaler()
        X = stdsc.fit(X).transform(X)

    return X

def predict_from_model(X, model = None, logSep = True, standardize = False):
    r'''
    [Load and] make prediction with a trained model
    Parameters:
    -----------
    X: np.ndarray
        Data cleaned using functions above
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
    #X = clean_data(data, logSep = logSep, standardize = standardize)
    y_pred = model.predict(X)
    if logSep:
        y_pred = np.power(10., y_pred)
    return y_pred



