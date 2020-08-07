import numpy as np
import matplotlib as mpl
try:
    mpl.use('agg')
except:
    pass
import matplotlib.pyplot as plt
import pandas as pd
import astropy as ap
import pickle
import scipy as sp
import mpmath
import seaborn as sns
from matplotlib.lines import Line2D

import histlite as hl
import csky as cy

from glob import glob
mpl.style.use('/home/apizzuto/Nova/python3/scripts/novae_plots.mplstyle')
gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')


class StackingPlots():
    r'''Helper class to make analysis plots for the
    stacking analysis'''
    
    def __init__(self, **kwargs):
        pass

class GammaCatalog():
    r'''Helper class to make analysis plots for the
    individually gamma-ray novae
    Warning: initializing this class will eat up like
    0.5 GB of memory
    '''

    def __init__(self, **kwargs):
        self.all_flavor = kwargs.pop('allflavor', True)
        self.spec_ind = kwargs.pop('index', [2., 2.5, 3.0])
        if type(self.spec_ind) is float:
            self.spec_ind = [self.spec_ind]
        self.min_log_e = kwargs.pop('min_log_e', 0.)
        self.verbose = kwargs.pop('verbose', False)
        self.fontsize = kwargs.pop
        self.fontsize = kwargs.pop('fontsize', 16)
        self.gamma_colors = {2.0: 'C0', 2.5: 'C1', 3.0: 'C2'}
        self.central_90 = {2.0: (23.26, 4430.12), 2.5: (8.69, 1594.00), 
                           3.0: (4.76, 419.32)}
        
        gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')
        names = gamma_df['Name']
        delta_ts = np.logspace(1., 6.5, 12)
        self.delta_ts = delta_ts
        all_novae = {name: {delta_t: GammaRayNova(name, delta_t=delta_t, **kwargs) for delta_t in delta_ts}
                             for name in gamma_df['Name']}
        self.all_novae = all_novae
        self.full_time_novae = {name: GammaRayNova(name, **kwargs) for name in gamma_df['Name']}
        self.names = names
        self.dpi = kwargs.pop('dpi', 150)
        self.sens_col = kwargs.pop('sens_col', sns.xkcd_rgb['light navy blue'])
        self.savefigs = kwargs.pop('save', False)
        self.savepath = kwargs.pop('output', '/data/user/apizzuto/Nova/plots/')
        
    def sensitivity_vs_time(self, **kwargs):
        fig, ax = plt.subplots(dpi=self.dpi)
        spec = kwargs.pop('gamma', 2.)
        discovery = kwargs.pop('discovery', False)
        sens = {}; good_ts = {};
        disc = {}
        for name in self.names:
            sens[name] = []; good_ts[name] = []
            disc[name] = []
            for delta_t in self.delta_ts:
                try:
                    if not discovery:
                        sens[name].append(self.all_novae[name][delta_t].sensitivity_trials[spec]['E2dNdE'])
                    good_ts[name].append(delta_t)
                    if discovery:
                        disc[name].append(self.all_novae[name][delta_t].discovery_trials[spec]['E2dNdE'])
                except Exception as e:
                    if self.verbose:
                        print(e)
            good_ts[name] = np.array(good_ts[name])
            if discovery:
                disc[name] = np.array(disc[name])
            else:
                sens[name] = np.array(sens[name])

        for name in sens.keys():
            try:
                if discovery:
                    ax.plot(good_ts[name], disc[name], color = self.sens_col,
                           ls = '--')
                else:
                    ax.plot(good_ts[name], sens[name], color = self.sens_col)
                if 'annotate' in kwargs.keys():
                    if kwargs['annotate'] and len(good_ts[name] > 0):
                        if discovery:
                            plt.annotate(name, (good_ts[name][-1], disc[name][-1]))
                        else:
                            plt.annotate(name, (good_ts[name][-1], sens[name][-1]))
                if discovery:
                    ax.scatter(self.full_time_novae[name].delta_t, 
                          self.full_time_novae[name].discovery_trials[spec]['E2dNdE'],
                          marker='^', s=20, color=sns.xkcd_rgb['almost black'],
                          zorder=20)
                else:
                    ax.scatter(self.full_time_novae[name].delta_t, 
                          self.full_time_novae[name].sensitivity_trials[spec]['E2dNdE'],
                          marker='*', s=20, color=sns.xkcd_rgb['almost black'],
                          zorder=20)
            except Exception as e:
                if self.verbose:
                    print(e)

        ax.loglog()
        ax.set_xlim(8e0, 1e8)
        ax.set_xlabel(r"$\Delta T$ (s)")
        ax.set_ylabel(r"$E^2 \frac{dN}{dE} @ 1$ TeV (TeV cm$^{-2}$)")
        if spec == 2.:
            title = r"$\frac{dN}{dE} \propto E^{-2}$"
        elif spec == 2.5:
            title = r"$\frac{dN}{dE} \propto E^{-2.5}$"
        elif spec == 3.0:
            title = r"$\frac{dN}{dE} \propto E^{-3}$"
        else:
            title = ''
        ax.set_title(title)
        if self.savefigs:
            sens_str = 'sensitivity' if not discovery else 'discovery'
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'{sens_str}_vs_time_gamma_{spec:.1f}_allflavor_{self.all_flavor}_minloge_{self.min_log_e:.1f}.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')
    
    def background_ts_panel(self, **kwargs):
        if not 'fig' in kwargs.keys() and not 'axs' in kwargs.keys():
            fig, aaxs = plt.subplots(nrows=3, ncols=5, dpi=self.dpi,
                                    figsize=(16,10), sharey=True, sharex=True)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        axs = np.ravel(aaxs)
        used_axs = []
        if 'delta_t' in kwargs.keys():
            delta_t = kwargs['delta_t']
            ii = 0
            for name in self.names:
                ax = axs[ii]
                try:
                    self.all_novae[name][delta_t].background_ts_plot(ax=ax, label_axes=False)
                    used_axs.append(ax)
                    ii += 1
                except:
                    pass
            title = r"$\Delta T_{\nu} = $" + f"{delta_t:.2e} s"
        else:
            ii = 0
            for name in self.names:
                ax = axs[ii]
                try:
                    self.full_time_novae[name].background_ts_plot(ax=ax, label_axes=False)
                    used_axs.append(ax)
                    ii += 1
                except:
                    pass
            title = r"$\Delta T_{\nu} = \Delta T_{\gamma}$"
        for ax in axs:
            if ax not in used_axs:
                ax.set_visible(False)
        for ax in aaxs[-1]:
            if ax in used_axs:
                ax.set_xlabel('TS', fontsize=self.fontsize)
        for ax in aaxs[:,0]:
            if ax in used_axs:
                ax.set_ylabel('Number of Trials', fontsize=self.fontsize)
        if not 'omit_title' in kwargs.keys():
            fig.suptitle(title, y=0.92, fontsize=self.fontsize) 
        elif not kwargs['omit_title']:
            fig.suptitle(title, y=0.92, fontsize=self.fontsize) 
        if self.savefigs:
            delta_t_str = f"delta_t_{delta_t:.2e}" if 'delta_t' in kwargs.keys() else 'full_gamma_time'
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'background_ts_panel_{delta_t_str}_allflavor_{self.all_flavor}_minloge_{self.min_log_e:.1f}.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')

    def ns_fitting_panel(self, **kwargs):
        if not 'fig' in kwargs.keys() and not 'axs' in kwargs.keys():
            fig, aaxs = plt.subplots(nrows=3, ncols=5, dpi=self.dpi,
                                    figsize=(16,10), sharey=True, sharex=True)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        spec = kwargs.pop('gamma', 2.)
        axs = np.ravel(aaxs)
        used_axs = []
        if 'delta_t' in kwargs.keys():
            delta_t = kwargs['delta_t']
            ii = 0
            for name in self.names:
                ax = axs[ii]
                try:
                    self.all_novae[name][delta_t].ns_fit_bias_plot(ax=ax, label_axes=False, gamma=spec)
                    used_axs.append(ax)
                    ii += 1
                except:
                    pass
            title = r"$\Delta T_{\nu} = $" + f"{delta_t:.2e} s"
        else:
            ii = 0
            for name in self.names:
                ax = axs[ii]
                try:
                    self.full_time_novae[name].ns_fit_bias_plot(ax=ax, label_axes=False, gamma=spec)
                    used_axs.append(ax)
                    ii += 1
                except:
                    pass
            title = r"$\Delta T_{\nu} = \Delta T_{\gamma}$"
        for ax in axs:
            if ax not in used_axs:
                ax.set_visible(False)
        for ax in aaxs[-1]:
            if ax in used_axs:
                ax.set_xlabel(r'$n_\mathrm{inj}$', fontsize=self.fontsize)
        for ax in aaxs[:,0]:
            if ax in used_axs:
                ax.set_ylabel(r'$\hat{n_s}$', fontsize=self.fontsize)
        if not 'omit_title' in kwargs.keys():
            fig.suptitle(title, y=0.92, fontsize=self.fontsize) 
        elif not kwargs['omit_title']:
            fig.suptitle(title, y=0.92, fontsize=self.fontsize) 
        if self.savefigs:
            delta_t_str = f"delta_t_{delta_t:.2e}" if 'delta_t' in kwargs.keys() else 'full_gamma_time'
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'ns_fitting_panel_{delta_t_str}_gamma_{spec:.1f}_allflavor_{self.all_flavor}_minloge_{self.min_log_e:.1f}.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')
    
    def gamma_fitting_panel(self, **kwargs):
        if not 'fig' in kwargs.keys() and not 'axs' in kwargs.keys():
            fig, aaxs = plt.subplots(nrows=3, ncols=5, dpi=self.dpi,
                                    figsize=(16,10), sharey=True, sharex=True)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        axs = np.ravel(aaxs)
        used_axs = []
        if 'delta_t' in kwargs.keys():
            delta_t = kwargs['delta_t']
            ii = 0
            for name in self.names:
                ax = axs[ii]
                try:
                    self.all_novae[name][delta_t].gamma_fit_bias_plot(ax=ax, label_axes=False)
                    used_axs.append(ax)
                    ii += 1
                except:
                    pass
            title = r"$\Delta T_{\nu} = $" + f"{delta_t:.2e} s"
        else:
            ii = 0
            for name in self.names:
                ax = axs[ii]
                try:
                    self.full_time_novae[name].gamma_fit_bias_plot(ax=ax, label_axes=False)
                    used_axs.append(ax)
                    ii += 1
                except:
                    pass
            title = r"$\Delta T_{\nu} = \Delta T_{\gamma}$"
        for ax in axs:
            if ax not in used_axs:
                ax.set_visible(False)
        for ax in aaxs[-1]:
            if ax in used_axs:
                ax.set_xlabel(r'$n_\mathrm{inj}$', fontsize=self.fontsize)
        for ax in aaxs[:,0]:
            if ax in used_axs:
                ax.set_ylabel(r'$\hat{\gamma}$', fontsize=self.fontsize)
        if not 'omit_title' in kwargs.keys():
            fig.suptitle(title, y=0.92, fontsize=self.fontsize) 
        elif not kwargs['omit_title']:
            fig.suptitle(title, y=0.92, fontsize=self.fontsize) 
        if self.savefigs:
            delta_t_str = f"delta_t_{delta_t:.2e}" if 'delta_t' in kwargs.keys() else 'full_gamma_time'
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'gamma_fitting_panel_{delta_t_str}_allflavor_{self.all_flavor}_minloge_{self.min_log_e:.1f}.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')
    
    def compare_photon_to_nus(self, **kwargs):
        if not 'fig' in kwargs.keys() and not 'ax' in kwargs.keys():
            fig, ax = plt.subplots(dpi=self.dpi)
        else:
            fig = kwargs['fig']; ax = kwargs['ax']
        if not 'delta_t' in kwargs:
            for name in self.names:
                try:
                    self.full_time_novae[name].compare_sens_to_photons(ax=ax)
                except:
                    if self.verbose:
                        print(f"No sensitivity for nova {name}")
            title = r"$\Delta T_{\nu} = \Delta T_{\gamma}$"
        else:
            delta_t = kwargs['delta_t']
            for name in self.names:
                try:
                    self.all_novae[name][delta_t].compare_sens_to_photons(ax=ax)
                except:
                    if self.verbose:
                        print(f"No sensitivity for nova {name}")
            title = r"$\Delta T_{\nu} = $" + f"{delta_t:.2e} s"
        if not 'omit_title' in kwargs.keys():
            ax.set_title(title, fontsize = self.fontsize)
        elif not kwargs['omit_title']:
            ax.set_title(title, fontsize = self.fontsize)
        if self.savefigs:
            delta_t_str = f"delta_t_{delta_t:.2e}" if 'delta_t' in kwargs.keys() else 'full_gamma_time'
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'photon_and_nu_SED_{delta_t_str}_allflavor_{self.all_flavor}_minloge_{self.min_log_e:.1f}.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')
        if 'return_fig' in kwargs.keys():
            return fig, ax


class GammaRayNova():
    r'''Holds information about analysis for 
    individual gamma-ray novae'''
    def __init__(self, name, **kwargs):
        if 'delta_t' in kwargs.keys():
            self.full_time = False
        else:
            self.full_time = True
        if self.full_time:
            try:
                delta_t = gamma_df[gamma_df['Name'] == name]['Stop Time'] - \
                            gamma_df[gamma_df['Name'] == name]['Start Time']
                delta_t = delta_t.values[0] * 86400.
            except NameError:
                gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')
                delta_t = gamma_df[gamma_df['Name'] == name]['Stop Time'] - \
                            gamma_df[gamma_df['Name'] == name]['Start Time']
                delta_t = delta_t.values[0] * 86400.
            self.delta_t = delta_t
            self.delta_t_str = "full_gamma_time"
        else:
            try:
                self.delta_t = kwargs['delta_t']
                self.delta_t_str = f"{self.delta_t:.2e}"
            except:
                print('Must either enter a time window or use default full gamma-ray duration')
        self.name = name
        self.all_flavor = kwargs.pop('allflavor', True)
        self.spec_ind = kwargs.pop('index', [2., 2.5, 3.0])
        if type(self.spec_ind) is float:
            self.spec_ind = [self.spec_ind]
        self.min_log_e = kwargs.pop('min_log_e', 0.)
        self.verbose = kwargs.pop('verbose', False)
        self.trials_base = '/data/user/apizzuto/Nova/csky_trials/'
        try:
            self.sensitivity_trials = {}
            self.discovery_trials = {}
            self.fitting_trials = {}
            for ind in self.spec_ind:
                trials_path = self.trials_base + f'nova_*_{self.name}_delta_t_{self.delta_t_str}_minLogE_{self.min_log_e:.1f}_gamma_{ind:.1f}_allflavor_{self.all_flavor}_trials.pkl'
                trials_f = glob(trials_path)[0]
                with open(trials_f, 'rb') as f:
                    nova_trials = pickle.load(f)
                self.sensitivity_trials[ind] = nova_trials['sensitivity']
                self.discovery_trials[ind] = nova_trials['discovery']
                self.fitting_trials[ind] = nova_trials['fit']
            self.background = nova_trials['bg']
            self.ra = nova_trials['source_info']['ra']
            self.dec = nova_trials['source_info']['dec']
        except Exception as e:
            if self.verbose:
                print(f"Could not find trials for Nova {self.name} with analysis details:")
                print(f"\t All flavor: {self.all_flavor}\t Gamma: {self.spec_ind}\t Min log10(E): {self.min_log_e}")
                print(f"\t Duration: {self.delta_t:.2e} s")
        self.fontsize = kwargs.pop('fontsize', 16)
        self.units_ref_str = 'TeV cm^-2 @ 1 TeV'
        self.gamma_colors = {2.0: 'C0', 2.5: 'C1', 3.0: 'C2'}
        self.central_90 = {2.0: (23.26, 4430.12), 2.5: (8.69, 1594.00), 
                           3.0: (4.76, 419.32)}
                
    def background_ts_plot(self, ax=None, label_axes=True, **kwargs):
        r'''Make plot showing chi2 distribution and observed TS
        for background trials'''
        if ax is None:
            fig, ax = plt.subplots()
        bg = self.background
        h = bg.get_hist(bins=40, range=(0, 20))
        hl.plot1d(ax, h, crosses=True)
        norm = h.integrate().values
        ts = np.linspace(.1, h.range[0][1], 100)
        ax.plot(ts, norm * bg.pdf(ts))
        ax.semilogy(nonposy='clip')
        ax.set_ylim(1e-1, bg.n_total*1.5)
        ax.text(20, 6e2, self.name.replace('_', ' ') + '\n' + r'$\delta={:.0f}$'.format(self.dec*180./np.pi), 
                ha='right', va='center', fontsize=self.fontsize)
        if label_axes:
            ax.set_xlabel('TS', fontsize=self.fontsize)
            ax.set_ylabel('Number of Trials', fontsize=self.fontsize)
            
    def ns_fit_bias_plot(self, ax=None, label_axes=True, **kwargs):
        r'''Compare fit and injected number of events'''
        gamma = kwargs.pop('gamma', self.spec_ind[0])
        if ax is None:
            fig, ax = plt.subplots()
        n_sigs = np.unique(self.fitting_trials[gamma].ntrue)
        dns = np.mean(np.diff(n_sigs))
        ns_bins = np.r_[n_sigs - 0.5*dns, n_sigs[-1] + 0.5*dns]
        h = hl.hist((self.fitting_trials[gamma].ntrue, 
                     self.fitting_trials[gamma].ns), 
                    bins=(ns_bins, 100))
        hl.plot1d(ax, h.contain_project(1), 
                  errorbands=True, drawstyle='default', color=self.gamma_colors[gamma])
        lim = [0., max(n_sigs)+1]
        ax.set_xlim(ax.set_ylim(lim))
        expect_kw = dict(color=self.gamma_colors[gamma], 
                         ls='--', lw=1, zorder=-10)
        ax.plot(lim, lim, **expect_kw)
        if label_axes:
            ax.set_xlabel(r'$n_\mathrm{inj}$', fontsize=self.fontsize)
            ax.set_ylabel(r'$\hat{n_s}$', fontsize=self.fontsize)

    def gamma_fit_bias_plot(self, ax=None, label_axes=True, **kwargs):
        r'''Compare fit spectral indices as a function of signal strength'''
        if ax is None:
            fig, ax = plt.subplots()
        n_sigs = np.unique(self.fitting_trials[self.spec_ind[0]].ntrue)
        dns = np.mean(np.diff(n_sigs))
        ns_bins = np.r_[n_sigs - 0.5*dns, n_sigs[-1] + 0.5*dns]
        for gamma in self.spec_ind:
            h = hl.hist((self.fitting_trials[gamma].ntrue, 
                         self.fitting_trials[gamma].gamma), 
                        bins=(ns_bins, 100))
            hl.plot1d(ax, h.contain_project(1), errorbands=True, 
                      drawstyle='default', color=self.gamma_colors[gamma])
            expect_kw = dict(color=self.gamma_colors[gamma], 
                         ls='--', lw=1, zorder=-10)
            ax.axhline(gamma, **expect_kw)
        if label_axes:
            ax.set_xlabel(r'$n_\mathrm{inj}$', fontsize=self.fontsize)
            ax.set_ylabel(r'$\hat{\gamma}$', fontsize=self.fontsize)

    def compare_sens_to_photons(self, ax=None, label_axes=True, **kwargs):
        r'''Plot time integrated sensitivities and compare them to 
        the best-fit gamma-ray spectra'''
        gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')
        try:
            gamma_info = gamma_df.loc[gamma_df['Name'] == self.name]
            gamma = float(gamma_info['Index'])
            integrated_norm = float(gamma_info['Flux'])*1e-7
            cutoff = float(gamma_info['Cutoff'])
        except:
            print(f"Could not find Nova {self.name} in dataframe")
            return
        if ax is None:
            fig, ax = plt.subplots()
        ens = np.logspace(-1., 3., 500)
        
        handles = []
        for spec in self.spec_ind:
            nu_sens_spec = np.power(ens, -spec)*self.sensitivity_trials[spec]['E2dNdE']*1e3*(1e-3)**(2.-spec)
            en_msk = (ens > self.central_90[spec][0]) & (ens < self.central_90[spec][1])
            ax.plot(ens[en_msk], nu_sens_spec[en_msk]*ens[en_msk]**2,
                   color=self.gamma_colors[spec], lw = 1.5, zorder=5*(5.-spec))
            handles.append(Line2D([0], [0], color=self.gamma_colors[spec], 
                                  lw=1.5, label=r"$\gamma = $" + f" {spec:.1f}"))
            
        photon_norm = self.calc_flux_norm(gamma, cutoff, integrated_norm)
        if np.isinf(cutoff):
            photon_flux = photon_norm * np.power(ens, -gamma)
        else:
            photon_flux = photon_norm * np.power(ens, -gamma) * \
                            np.exp(-ens / cutoff)
        ax.plot(ens, photon_flux* ens**2. * self.delta_t, 
                color=sns.xkcd_rgb['battleship grey'], alpha=0.7)
        
        ax.loglog()
        ax.set_ylim(1e-4, 3e4)
        ax.set_xlim(1e-1, 1e3)
        if label_axes:
            ax.set_xlabel(r"$E$ (GeV)", fontsize=self.fontsize)
            ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (GeV cm$^{-2}$)", fontsize = self.fontsize)
            
        ax.legend(handles=handles, frameon=False, loc=2)
        
    def calc_flux_norm(self, gamma, cutoff, norm):
        r'''Fermi fluxes are reported in terms of integral fluxes
        above 100 MeV. Convert them here to the same units used here
        Parameters:
        -----------
            -norm: float, integrated flux above 100 MeV in units of photons per cm^-2 s^-1
        Returns:
        --------
            Flux normalization at 1 GeV in units of GeV^-1 cm^-2 s^1
        '''
        if np.isinf(cutoff):
            denom = 0.1*np.exp(2.30259*gamma) / (-1. + gamma)
        else:
            denom = cutoff ** (1.-gamma)*float(mpmath.gammainc(1.-gamma, 0.1/cutoff))
        return norm / denom

class GRECOPlots():
    r'''Helper class to make plots that describe the 
    overall dataset and its performance'''

    def __init__(self, **kwargs):
        pass
