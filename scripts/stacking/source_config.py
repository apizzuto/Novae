import numpy as np
import pandas as pd
import csky as cy

def get_sources(only_gamma, weights, delta_t_days):
    """
    Based on analysis type, get the right csky Sources Array

    Args:
        only_gamma (bool): If True, only use gamma-ray detected novae
        weights (str): Weighting scheme. Choose between 'optical' and 'gamma'
        delta_t_days (float): Analysis time window in units of days

    Returns:
        cy.utils.Sources: Array object of Sources
        sample_str: Bookkeeping name for output files
    """
    if weights not in ['optical', 'gamma']:
        raise ValueError("Not an allowed weighting scheme")
    master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
    if only_gamma:
        master_df = master_df[master_df['gamma']]
        sample_str = 'only_gamma'
    else:
        sample_str = 'all_novae'
    if weights == 'optical':
        weight = 10.**(master_df['Peak'] / 2.5)
        weight /= weight.sum()
        sample_str += 'weight_optical'
    else:
        if not only_gamma:
            raise ValueError("Not able to use gamma derived weights " \
                + "for non-gamma-detected novae")
        weight = master_df['gamma_norm']
        weight /= weight.sum()
        sample_str += 'weight_gamma'
    
    ras = master_df['RA']
    decs = master_df['Dec']
    names = master_df['Name']
    mjds = np.array([t.mjd for t in master_df['Date']])
    delta_ts = np.ones_like(ras) * delta_t_days
    mjds -= delta_ts / 2.

    src = cy.utils.Sources(
        ra=np.radians(ras), 
        dec=np.radians(decs), 
        mjd=mjds, 
        sigma_t=np.zeros_like(delta_ts), 
        t_100=delta_ts,
        weight=weight
        )
    
    return src, sample_str