import numpy        as np
import pandas       as pd
import scipy        as sp
import os

GeV = 1.
TeV = 1000. * GeV
n_bins = 50
delta_cos_theta = 0.1

mlarson_path = '/data/user/mlarson/combo_r129072/skylab/scripts/testing/GRECO/version-001-p00/IC86_2012_mc.npy'
def mids(arr):
    return arr[:-1] + (np.diff(arr) / 2.)

class Nova(object):
    r''' A class to handle relevant 
        details of Fermi-LAT detected 
        novae for IceCube followup. 

    Parameters
    ----------
    Parameters of a nova, including lightcurve and spectral fits

    Attributes
    ----------
    . . . 
    '''

    def __init__(self, name, gamma, ra, dec, center_time, time_sigma, cutoff=None,
                    lightcurve=None, flux_norm = 1., ref = 1.*GeV, dataset = mlarson_path):
        r"""Constructor
        
        Parameters
        __________
        Information about the Nova
        \*\*kwargs :
            Additional keyword args
        """
        self.name = name
        self.gamma = gamma
        if ra < 0. or ra > 2 * np.pi:
            raise Exception('RA is in units of radians. Should be contained between 0 and 2pi\n' +
                'Value of RA was {}'.format(ra))
        elif dec < -np.pi / 2. or dec > np.pi / 2.:
            raise Exception('Declination is in units of radians. Should be contained between -pi/2 and pi/2\n' +
                'Value of Dec was {}'.format(dec))
        else:
            self.ra = ra
            self.dec = dec
        self.center_time = center_time
        try:
            from icecube import astro
            self.zenith, self.azimuth = astro.equa_to_dir(self.ra, self.dec, 
                                            self.center_time)
        except:
            print("Was not able to import icecube.astro"),
            print("cannot use detector coordinates")
            self.zenith = self.dec + (np.pi/2.)
        self.time_sigma = time_sigma
        self.cutoff = cutoff
        self.lightcurve = lightcurve
        self.flux_norm = flux_norm
        self.ref = ref
        self.mc = None
        self.aeff_dict = None
        self.initialize_mc(dataset)
        self.initialize_aeff()


    def spectrum(self, energy, cutoff = True):
        r'''
        Returns the value of dN/dEdAdt at a given energy
        Parameters:
        -----------
        energy  (float or array-like)
            energy(-ies) at which to evaluate the flux
        cutoff  (bool, optional)
            if the nova has a cutoff, option to include it. If false,
            same normalization and slope are assumed and extrapolated
        Returns:
        --------
        flux (float or array-like)
            Value(s) of flux for given energies
        '''
        if cutoff and self.cutoff is not None:
            return self.flux_norm * np.power(energy/self.ref, self.gamma) \
                 * np.exp(-1. * energy / self.cutoff)
        else:
            return self.flux_norm * np.power(energy / self.ref, self.gamma)

    def initialize_mc(self, dataset):
        r'''
        Initialize a Monte Carlo numpy array from either path or Skylab dataset
        Parameters:
        -----------
        dataset (string, ndarray, or skylab.dataset object)
            Way to initialize monte-carlo: either
            (1) skylab.dataset object
                - Initialize from approved skylab dataset, extracts ndarray
            (2) path to numpy ndarray
            (3) numpy ndarray
        '''
        if str(type(dataset)).startswith("<class 'skylab.data"):
            exp, mc, livetime = dataset.season('IC86, 2015') #MC is all the same
            zen_msk = np.cos(mc['zenith']) > np.cos(self.zenith) - (delta_cos_theta/2.)
            zen_msk *= np.cos(mc['zenith']) < np.cos(self.zenith) + (delta_cos_theta/2.)
        elif type(dataset) == str:
            if os.path.isfile(dataset):
                mc = np.load(dataset)
                zen_msk = np.cos(mc['zen']) > np.cos(self.zenith) - (delta_cos_theta/2.)
                zen_msk *= np.cos(mc['zen']) < np.cos(self.zenith) + (delta_cos_theta/2.)
            else:
                raise Exception("Dataset not valid format. Must be one of:" +
                "\n(1) Skylab Dataset\n(2) Path to MC npy file\n(3) np.ndarray")
        else:
            if issubclass(np.ndarray, type(dataset)):
                try:
                    mc = dataset
                    zen_msk = np.cos(mc['zen']) > np.cos(self.zenith) - (delta_cos_theta/2.)
                    zen_msk *= np.cos(mc['zen']) < np.cos(self.zenith) + (delta_cos_theta/2.)
                except:
                    mc = dataset
                    zen_msk = np.cos(mc['zenith']) > np.cos(self.zenith) - (delta_cos_theta/2.)
                    zen_msk *= np.cos(mc['zenith']) < np.cos(self.zenith) + (delta_cos_theta/2.)
            else:
                raise Exception("Dataset not valid format. Must be one of:" +
                "\n(1) Skylab Dataset\n(2) Path to MC npy file\n(3) np.ndarray")
        #zen_msk = np.cos(mc['zen']) > np.cos(self.zenith) - (delta_cos_theta/2.)
        #zen_msk *= np.cos(mc['zen']) < np.cos(self.zenith) + (delta_cos_theta/2.)
        mc = mc[zen_msk] #only look at events in zenith angle band
        self.mc = mc

    def initialize_aeff(self):
        r'''
        Calculates effective area from Monte-Carlo events with weights, stores
        in the class
        '''
        d_omega = 2. * np.pi * delta_cos_theta
        if np.max(self.mc['trueE']) < 1e3:
            E_bins = np.logspace(0., 3., 31)
        else:
            E_bins = np.logspace(0., 9., 31)
        logE_bins = np.log10(E_bins)
        dlog_E = np.diff(logE_bins)
        weights = self.mc['ow'] / (1e4 * self.mc['trueE'] * dlog_E[np.digitize(np.log10(self.mc['trueE']), bins = logE_bins) -1] * d_omega * np.log(10.))
        h, b = np.histogram(self.mc['trueE'], weights = weights, bins = E_bins)
        self.aeff_dict = {'bins': b, 'vals': h}

    def aeff(self, en):
        r'''
        Parameters:
        -----------
        en (float or array-like)
            Energy at which you wish to evaluate effective area
        Returns:
        --------
        aeff (float or array-like) 
            effective area in units of m^2
        '''
        if self.aeff_dict == None:
            self.initialize_aeff()
        bin_centers = mids(self.aeff_dict['bins'])
        aeffs = np.interp(en, bin_centers, self.aeff_dict['vals'])
        return aeffs

    def aeff_binned(self, en):
        r'''
        Older implementation of effective area that I don't mind keeping around
        for cross-checks
        Parameters:
        -----------
        en (float or array-like)
            Energy at which you wish to evaluate effective area
        Returns:
        --------
        aeff (float or array-like) 
            effective area in units of m^2
        '''
        if type(en) == float or type(en) == int:
            ind = np.digitize(en, bins = self.aeff_dict['bins']) - 1
            if ind < len(h[0]):
                return self.aeff_dict['vals'][ind] 
            else:
                return self.aeff_dict['vals'][-1]
        else:
            inds = np.digitize(en, bins = self.aeff_dict['bins']) - 1
            aeffs = [self.aeff_dict['vals'][ind] if ind < len(self.aeff_dict['vals']) else self.aeff_dict['vals'][-1] for ind in inds]
            return aeffs

    def calc_dNdE(self, energies, time_integrated = False):
        r'''
        Given effective area and flux, calculate spectrum of expected
        events at IceCube. Note: this curve is NOT integrated over energy, 
        this must be done to get an actual signal
        Parameters:
        ----------
        energies (float or array-like)
            energies at which to calculate dN/dE
        time_integrated (bool, optional, default=False)
            integrate over time of the nova
        '''
        flux = self.spectrum(energies)
        if time_integrated:
            tot_flux = flux * self.time_sigma * 86400.
        else:
            tot_flux = flux
        aeffs = self.aeff(energies) * 1e4 #Effective area in m^2, convert to cm^2
        dNdE = aeffs * tot_flux 
        return dNdE

    def calc_expected_signal_binned(self, energy_bins, time_integrated = False):
        r'''
        Calculates number of signal events expected per energy bin
        Parameters:
        -----------
        energy_bins (array)
            bins in true neutrino energy
        time_integrated (bool, optional, default=False)
            integrate over time of the nova
        '''
        energies = mids(energy_bins)
        flux = self.spectrum(energies)
        if time_integrated:
            tot_flux = flux * self.time_sigma * 86400.
        else:
            tot_flux = flux
        aeffs = self.aeff(energies) * 1e4 #Effective area in m^2, convert to cm^2
        signal = aeffs * tot_flux * np.diff(energy_bins) #Integrate in energy space
        return signal, np.sum(signal)
