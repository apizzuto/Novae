import numpy as np
import scipy as sp
#from lmfit                import Model
from scipy.optimize       import curve_fit
from scipy.stats          import chi2

palette = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f']


def erfunc(x, a, b):
    return 0.5 + 0.5*sp.special.erf(a*x + b)

def chi2cdf(x,df1,loc,scale):
    func = chi2.cdf(x,df1,loc,scale)
    return func

def incomplete_gamma(x, a, scale):
    return sp.special.gammaincc( scale*x, a)

def poissoncdf(x, mu, loc):
    func = sp.stats.poisson.cdf(x, mu, loc)
    return func

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def pass_vs_inj(index, spectra='SPL', deltaT=1000., threshold = 0.5, in_ns = True, with_err = True, trim=-1):
    print("YO WHAT UP")
    #bg_trials = np.load('/data/user/apizzuto/Nova/analysis_trials/bg/kent/deltaT_{:.1e}_index_{}_spec_{}.npy'.format(deltaT, index, spectra), allow_pickle=True).item()
    #bg_trials = bg_trials['TS']
    signal_trials = np.load('/data/user/apizzuto/Nova/analysis_trials/sensitivity/deltaT_{:.1e}_index_{}_spec_{}.npy'.format(deltaT, index, spectra))
    #signal_trials = signal_trials[signal_trials['gamma'] == gamma]
    bg_thresh = 0.0#np.percentile(bg_trials, threshold * 100.)
    signal_fluxes, signal_indices = np.unique(signal_trials['mean_ninj'], return_index=True)
    signal_indices = np.append(signal_indices, len(signal_trials))
    if trim != -1 and trim < 0:
        signal_indices = signal_indices[:trim]
        signal_fluxes = signal_fluxes[:trim]
    elif trim > 0:
        signal_indices = signal_indices[:trim + 1]
        signal_fluxes = signal_fluxes[:trim]
    passing = np.array([np.count_nonzero(signal_trials['TS'][li:ri] > bg_thresh) / float(ri - li) for li, ri in zip(signal_indices[:-1], signal_indices[1:])])
    if not with_err:
        return signal_fluxes, passing
    else:
        errs = np.array([np.sqrt(p*(1.-p) / float(ri - li)) for p, li, ri in zip(passing, signal_indices[:-1], signal_indices[1:])])
        ngen = np.array([float(ri - li) for li, ri in zip(signal_indices[:-1], signal_indices[1:])])
        ntrig = passing * ngen
        bound_case_pass = (ntrig + (1./3.)) / (ngen + (2./3.))
        bound_case_sigma = np.sqrt(bound_case_pass*(1. - bound_case_pass) / (ngen + 2))
        errs = np.maximum(errs, bound_case_sigma)
        return signal_fluxes, passing, errs
    
def sensitivity_curve(index, spectra='SPL', deltaT=1000., threshold = 0.5, in_ns = True, with_err = True, trim=-1, ax = None, p0 = None, fontsize = 16):
    signal_fluxes, passing, errs = pass_vs_inj(index, spectra=spectra, deltaT=deltaT, threshold=threshold, in_ns=in_ns, with_err=with_err, trim=trim)
    fits, plist = [], []
    try:
        fits.append(sensitivity_fit(signal_fluxes, passing, errs, chi2cdf, p0=p0))
        plist.append(fits[-1]['pval'])
        fits.append(sensitivity_fit(signal_fluxes, passing, errs, erfunc, p0=p0))
        plist.append(fits[-1]['pval'])
        fits.append(sensitivity_fit(signal_fluxes, passing, errs, incomplete_gamma, p0=p0))
        plist.append(fits[-1]['pval'])
    except:
        pass
        #print("at least one fit failed")
    #Find best fit of the three, make it look different in plot
    plist = np.array(plist)
    best_fit_ind= np.argmax(plist)
    fits[best_fit_ind]['ls'] = '-'
    
    if ax==None:
        fig, ax = plt.subplots()
    
    for fit_dict in fits:
        ax.plot(fit_dict['xfit'], fit_dict['yfit'], 
                 label = r'{}: $\chi^2$ = {:.2f}, d.o.f. = {}'.format(fit_dict['name'], fit_dict['chi2'], fit_dict['dof']),
                ls = fit_dict['ls'])
        if fit_dict['ls'] == '-':
            ax.axhline(0.9, color = palette[-1], linewidth = 0.3, linestyle = '-.')
            ax.axvline(fit_dict['sens'], color = palette[-1], linewidth = 0.3, linestyle = '-.')
            ax.text(5, 0.5, r'Sens. = {:.2f}'.format(fit_dict['sens']))
    ax.errorbar(signal_fluxes, passing, yerr=errs, capsize = 3, linestyle='', marker = 's', markersize = 2)
    ax.legend(loc=4, fontsize = fontsize)
    ax.set_ylim(0.0, 1.05)
    
def calc_sensitivity(index, spectra='SPL', deltaT=1000., threshold = 0.5, in_ns = True, with_err = True, trim=-1, p0=None):
    signal_fluxes, passing, errs = pass_vs_inj(index, spectra, deltaT=deltaT, threshold=threshold, in_ns=in_ns, with_err=with_err, trim=trim)
    fits, plist = [], []
    try:
        fits.append(sensitivity_fit(signal_fluxes, passing, errs, chi2cdf, p0=p0))
        plist.append(fits[-1]['pval'])
        fits.append(sensitivity_fit(signal_fluxes, passing, errs, erfunc, p0=p0))
        plist.append(fits[-1]['pval'])
        fits.append(sensitivity_fit(signal_fluxes, passing, errs, incomplete_gamma, p0=p0))
        plist.append(fits[-1]['pval'])
    except:
        pass
    #Find best fit of the three, make it look different in plot
    plist = np.array(plist)
    best_fit_ind= np.argmax(plist)
    return fits[best_fit_ind]
    
def sensitivity_fit(signal_fluxes, passing, errs, fit_func, p0 = None, conf_lev = 0.9):
    try:
        name = fit_func.__name__
        name = name.replace("_", " ")
    except:
        name = 'fit'
    popt, pcov = curve_fit(fit_func, signal_fluxes, passing, sigma = errs, p0 = p0, maxfev=10000)
    #print popt
    fit_points = fit_func(signal_fluxes, *popt)
    chi2 = np.sum((fit_points - passing)**2. / errs**2.)
    dof = len(fit_points) - len(popt)
    xfit = np.linspace(np.min(signal_fluxes) - 0.5, np.max(signal_fluxes), 100)
    yfit = fit_func(xfit, *popt)
    pval = sp.stats.chi2.sf(chi2, dof)
    sens = xfit[find_nearest_idx(yfit, 0.9)]
    return {'popt': popt, 'pcov': pcov, 'chi2': chi2, 
            'dof': dof, 'xfit': xfit, 'yfit': yfit, 
            'name': name, 'pval':pval, 'ls':'--', 'sens': sens}

def pvals_for_signal(index, spectra='SPL', deltaT=1000., ns = 1, sigma_units = False):
    bg_trials = np.load('/data/user/apizzuto/Nova/analysis_trials/bg/kent/deltaT_{:.1e}_index_{}_spec_{}.npy'.format(deltaT, index, spectra), allow_pickle=True).item()
    bg_trials = bg_trials['TS']
    signal_trials = np.load('/data/user/apizzuto/Nova/analysis_trials/sensitivity/deltaT_{:.1e}_index_{}_spec_{}.npy'.format(deltaT, index, spectra))
    signal_trials = signal_trials[signal_trials['n_inj'] == ns]
    #print(len(bg_trials['TS']))
    pvals = [100. - sp.stats.percentileofscore(bg_trials, ts, kind='strict') for ts in signal_trials['TS']]
    pvals = np.array(pvals)*0.01
    pvals = np.where(pvals==0, 1e-6, pvals)
    if not sigma_units:
        return pvals
    else:
        return sp.stats.norm.ppf(1. - (pvals / 2.))
