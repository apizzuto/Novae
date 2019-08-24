import numpy        as np
import pandas       as pd
import os

GeV = 1.
TeV = 1000. * GeV
n_bins = 50
delta_cos_theta = 0.05

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
                    lightcurve=None, flux_norm = 1., ref = 1.*GeV, dataset = ):
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


    def spectrum(self, energy, cutoff = True):
        if cutoff and self.cutoff is not None:
            return self.flux_norm * np.power(energy/self.ref, self.gamma) \
                 * np.exp(-1. * energy / self.cutoff)
        else:
            return self.flux_norm * np.power(energy / self.ref, self.gamma)

    '''def calc_expected_signal(self, dataset, cutoff = True):
        if str(type(dataset)).startswith("<class 'skylab.data"):
            exp, mc, livetime = dataset.season('IC86, 2015') #MC is all the same
        elif type(dataset) == str:
            if os.path.isfile(dataset):
                mc = np.load(dataset)
            else:
                raise Exception("Dataset not valid format. Must be one of:" + 
                "\n(1) Skylab Dataset\n(2) Path to MC npy file\n(3) np.ndarray")
        else:
            if issubclass(np.ndarray, type(dataset)):
                mc = dataset
            else:
                raise Exception("Dataset not valid format. Must be one of:" + 
                "\n(1) Skylab Dataset\n(2) Path to MC npy file\n(3) np.ndarray")
        energy_bins = np.logspace(0., 4., n_bins + 1)
        energies = energy_bins[:-1] + (np.diff(energy_bins) / 2.)
        flux = self.spectrum(energies, cutoff=cutoff)
        time_int_flux = flux * self.time_sigma * 86400.
        zen_msk = np.cos(mc['zen']) > np.cos(self.zenith) - (delta_cos_theta/2.)
        zen_msk *= np.cos(mc['zen']) < np.cos(self.zenith) + (delta_cos_theta/2.)
        mc_msk = mc[zen_msk] #only look at events in zenith angle band
        aeff = np.histogram(mc_msk['trueE'], bins = energy_bins, weights = mc_msk['ow'])[0]
        aeff = aeff / (2. * np.pi * delta_cos_theta * 1e4 * np.diff(energy_bins)) #scale for solid angle and m^2
        signal = aeff * time_int_flux * np.diff(energy_bins) #Integrate in energy space
        return energies, signal'''

    def initialize_mc(self):
        self.mc = None

    def aeff(self):
        if self.aeff is None:
            self.initialize_aeff()
        return 2.

    def calc_expected_signal(self, dataset, cutoff = True):
        energy_bins = np.logspace(0., 6., n_bins + 1)
        energies = energy_bins[:-1] + (np.diff(energy_bins) / 2.)
        flux = self.spectrum(energies, cutoff=cutoff)
        time_int_flux = flux * self.time_sigma * 86400.
        #print energies
        #print np.digitize(energies, bins = energy_bins)
        aeff = self.calc_aeff(dataset, energies)
        signal = aeff * time_int_flux * np.diff(energy_bins) #Integrate in energy space
        return energies, signal

    #SOMETHING IS VERY VERY WRONG HERE< I SHOULD DEFINITELY FIX THIS
    def calc_aeff(self, dataset, energy):
        if str(type(dataset)).startswith("<class 'skylab.data"):
            exp, mc, livetime = dataset.season('IC86, 2015') #MC is all the same
        elif type(dataset) == str:
            if os.path.isfile(dataset):
                mc = np.load(dataset)
            else:
                raise Exception("Dataset not valid format. Must be one of:" + 
                "\n(1) Skylab Dataset\n(2) Path to MC npy file\n(3) np.ndarray")
        else:
            if issubclass(np.ndarray, type(dataset)):
                mc = dataset
            else:
                raise Exception("Dataset not valid format. Must be one of:" + 
                "\n(1) Skylab Dataset\n(2) Path to MC npy file\n(3) np.ndarray")
        energy_bins = np.logspace(0., 3., n_bins + 1)
        energies = energy_bins[:-1] + (np.diff(energy_bins) / 2.)
        zen_msk = np.cos(mc['zen']) > np.cos(self.zenith) - (delta_cos_theta/2.)
        zen_msk *= np.cos(mc['zen']) < np.cos(self.zenith) + (delta_cos_theta/2.)
        mc_msk = mc[zen_msk] #only look at events in zenith angle band
        aeff = np.histogram(mc_msk['trueE'], bins = energy_bins, weights = mc_msk['ow'])[0]
        aeff = aeff / (2. * np.pi * delta_cos_theta * 1e4 * np.diff(energy_bins)) #scale for solid angle and m^2
        #print aeff
        #print np.digitize(energy, bins = energy_bins)
        #print np.where(energy < 1e3, np.digitize(energy, bins = energy_bins) - 1, aeff[-1])
        #print np.digitize(energy, bins = energy_bins)
        inds = np.digitize(energy, bins = energy_bins) - 1
        return [aeff[-1] if ind == 50 else aeff[ind] for ind in inds]



