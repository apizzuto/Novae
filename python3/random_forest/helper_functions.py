r'''
Set of helper functions for IceCube Novae analysis. 
Author: Alex Pizzuto
Date: (original commit) August 28, 2019
'''
import numpy as np

def mids(arr):
    r'''
    Given an array of bin edges, calculate the central values
    '''
    return arr[:-1] + (np.diff(arr) / 2.)

def find_nearest(array, values):
    r'''
    Map array of one values to closest values in 
    another array. Modified version of function
    found in stackexchange
    '''
    array = np.asarray(array)
    idx = [(np.abs(array - value)).argmin() for value in values]
    return array[idx]

def deltaPsi(dec1, ra1, dec2, ra2):
    """
    Calculate angular distance.
    
    Args:
        dec1: Declination of first direction in radian
        ra1: Right ascension of first direction in radian
        dec2: Declination of second direction in radian
        ra2: Right ascension of second direction in radian
        
    Returns angular distance in radian
    """
    return deltaPsi2(np.sin(dec1), np.cos(dec1), np.sin(ra1), np.cos(ra1), np.sin(dec2), np.cos(dec2), np.sin(ra2), np.cos(ra2))

def deltaPsi2(sDec1, cDec1, sRa1, cRa1, sDec2, cDec2, sRa2, cRa2):
    """
    Calculate angular distance.
    
    Args:
        sDec1: sin(Declination of first direction)
        cDec1: cos(Declination of first direction)
        sRa1: sin(Right ascension of first direction)
        cRa1: cos(Right ascension of first direction)
        sDec2: sin(Declination of second direction)
        cDec2: cos(Declination of second direction)
        sRa2: sin(Right ascension of second direction)
        cRa2: cos(Right ascension of second direction)
        
    Returns angular distance in radian
    """
    tmp = cDec1*cRa1*cDec2*cRa2 + cDec1*sRa1*cDec2*sRa2 + sDec1*sDec2
    tmp[tmp>1.] = 1.
    tmp[tmp<-1.] = -1.
    return np.arccos(tmp)

def append_rows(arrayIN, NewRows):
    r'''
    Concatenate numpy recarrays. Taken from stackexchange
    '''
    # Calculate the number of old and new rows
    len_arrayIN = arrayIN.shape[0]
    len_NewRows = len(NewRows)
    # Resize the old recarray
    arrayIN.resize(len_arrayIN + len_NewRows, refcheck=False)
    # Write to the end of recarray
    arrayIN[-len_NewRows:] = NewRows
    return arrayIN
