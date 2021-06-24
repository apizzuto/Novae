import numpy as np
import matplotlib as mpl
#try:
#    mpl.use('agg')
#except:
#    pass 
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import pickle
import scipy as sp
from scipy.interpolate import UnivariateSpline
import scipy.stats as st
import mpmath
import seaborn as sns
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore") # histlite throws a bunch of UserWarnings

import histlite as hl
import csky as cy

from glob import glob
mpl.style.use('/home/apizzuto/Nova/scripts/novae_plots.mplstyle')
import sys
sys.path.append('/home/apizzuto/Nova/scripts/stacking/')
from source_config import *
from stacking_config import *

master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
gamma_df = master_df[master_df['gamma']==True]
gamma_df = gamma_df.reset_index()

class StackingPlots():
    r'''Helper class to make analysis plots for the
    stacking analysis'''
    
    def __init__(self, delta_t, catalog, weighting, **kwargs):
        self.delta_t = delta_t
        assert catalog in ['all_novae', 'gamma'], 'Catalog not recognized'
        assert weighting in ['optical', 'gamma'], 'Weighting scheme not recognized'
        self.catalog = catalog
        self.weights = weighting
        self.min_log_e = kwargs.pop('min_log_e', 0.)
        self.all_flavor = kwargs.pop('allflavor', False)
        self.spec_ind = kwargs.pop('index', [2., 2.5, 3.])
        self.verbose = kwargs.pop('verbose', False)
        self.rng_seed = kwargs.pop('seed', None)
        self.df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
        self.names = self.df['Name']
        only_gamma = True if self.catalog == 'gamma' else False
        src, sample_str = get_sources(only_gamma, self.weights, self.delta_t / 86400.)
        self.src = src
        self.sample_str = sample_str
        self.dpi = kwargs.pop('dpi', 150)
        self.savefigs = kwargs.pop('save', False)
        self.savepath = kwargs.pop('output', '/data/user/apizzuto/Nova/plots/')
        self.fontsize = kwargs.pop('fontsize', 16)
        self.show = kwargs.pop('show', True)
        self.trials_base = '/data/user/apizzuto/Nova/csky_trials/stacking_sens_res/'
        self.all_delta_ts = np.sort(np.append(np.logspace(-1.5, 1., 6)[:]*86400.,
            np.array([86400.*5.])))
        self.all_results = None
        self.ana = None
        self.gam_cols = {2.0: 'C0', 2.5: 'C1', 3.0: 'C3'}
        self.min_log_cols = {0.0: 'C0', 0.5: 'C1', 1.0: 'C3', None: 'C0'}
        self.diff_sens = None
        self.get_all_sens()

    def initialize_analysis(self):
        """Set up a csky analysis object"""
        if self.verbose:
            print("Initializing csky analysis")
        greco, conf = get_stacking_objs(
            minLogE=self.min_log_e, 
            allflavor=self.all_flavor
            )
        ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
        greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)
        cy.CONF['src'] = self.src
        cy.CONF['mp_cpus'] = 10
        self.conf = conf
        self.greco = greco
        self.ana = greco_ana
    
    def likelihood_scan(self, n_inj=0., inj_gamma=2.0, truth=False):
        """
        Perform a single trial, with or without signal injected,
        and calculate the likelihood landscape
        
        :type n_inj: float
        :param n_inj: Number of injected signal events

        :type inj_gamma: float
        :param inj_gamma: Injected spectral index

        :type truth: bool
        :param truth: True llh scan (only permitted after unblinding)
        """
        if self.ana is None:
            self.initialize_analysis()
        if truth and n_inj != 0.:
            print("Can't look at truth with signal")
        elif truth:
            tr = cy.get_trial_runner(self.conf, ana=self.ana, src=self.src)
            print("ONLY LOOK AT TRUTH IF YOU HAVE PERMISSION TO UNBLIND")
        else:
            tr = cy.get_trial_runner(self.conf, ana=self.ana, src=self.src, 
                    inj_conf={'flux': cy.hyp.PowerLawFlux(inj_gamma)})
        trial = tr.get_one_trial(n_sig=n_inj, TRUTH=truth, seed=self.rng_seed,
                    poisson=False)
        llh = tr.get_one_llh_from_trial(trial)
        nss = np.linspace(0., 120., 80)
        gammas = np.linspace(1., 4., 80)
        llh_space = llh.scan_ts(ns=nss, gamma=gammas)
        mesh_ns, mesh_gam = np.meshgrid(nss, gammas)
        best_fit_ts = np.max(llh_space[0])
        best_fit_ns = mesh_ns.flatten()[np.argmax(llh_space[0].T)]
        best_fit_gamma = mesh_gam.flatten()[np.argmax(llh_space[0].T)]
        # Calculate critical values
        chi2 = st.chi2(2., loc=0., scale=1)
        crit_med = chi2.isf(0.5)
        crit_90 = chi2.isf(1.0-0.9)
        crit_99 = chi2.isf(1.0-0.99)

        fig, ax1 = plt.subplots(dpi=150)
        ts_space = llh_space[0]

        im = ax1.pcolor(mesh_ns, mesh_gam, -1.*(ts_space.T - best_fit_ts), 
                            cmap="Blues_r", vmin=0, vmax=50)#, norm=LogNorm(vmin=0.01, vmax=20))
        ax_c = plt.colorbar(im, ax=ax1, format='$%.1f$')
        ax_c.ax.tick_params(direction='out')
        ax_c.set_label(r"$-2\Delta\,\mathrm{LLH}$")

        ax1.contour(mesh_ns, mesh_gam, -1.*(ts_space.T - best_fit_ts), 
                    #levels=[1,4, 9, 16, 25], 
                    levels=[crit_med, crit_90, crit_99], colors="w")
        if not truth:
            ax1.plot(n_inj, inj_gamma, marker="^", color="w", markersize=10, label='Truth' )
        ax1.plot(best_fit_ns, best_fit_gamma, marker="*", color="w", 
                        markersize=10, label='Best-fit')
        print(best_fit_ts, best_fit_ns, best_fit_gamma)
        ax1.set_ylabel(r"$\gamma$")
        ax1.set_xlabel(r"$n_\mathrm{s}$")
        ax1.legend(loc=4, facecolor=sns.xkcd_rgb['light grey'])
        fig.tight_layout()
        if self.savefigs:
            add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'stacking_llh_scan_{self.sample_str}_delta_t_{self.delta_t:.2e}{add_str}_truth_{truth}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def background_distribution(self, ax=None, show=True):
        """
        Plot the background TS distribution for a given time 
        window

        :type ax: matplotlib axes object
        :param ax: pass if you are already working with a set of axes

        :type show: bool
        :param show: call plt.show if true
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        bg = self.results[self.spec_ind[0]]['bg']
        h = bg.get_hist(bins=50)
        hl.plot1d(ax, h, crosses=True,
                label='{} bg trials'.format(bg.n_total))

        x = h.centers[0]
        norm = h.integrate().values
        ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
                    label=r'$\chi^2[{:.2f}\mathrm{{dof}},\ \eta={:.3f}]$'.format(bg.ndof, bg.eta))

        if show:
            ax.set_xlabel(r'TS')
            ax.set_ylabel(r'number of trials')
            ax.legend()
            plt.tight_layout()
        if self.savefigs:
            cut = self.min_log_e
            add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'stacking_bg_ts_distribution_{self.sample_str}_delta_t_{self.delta_t:.2e}{add_str}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def background_vs_time(self):
        """
        Make a 3x3 panel plot for the background trials for all of 
        the different time windows, not just the one that we are
        focusing on
        """
        fig, aaxs = plt.subplots(
            nrows=3, ncols=3, sharex=True, sharey=True, dpi=200,
            figsize=(13,11))
        axs = aaxs.ravel()
        plt.subplots_adjust(hspace=0.03, wspace=0.03)
        for ii, del_t in enumerate(self.all_delta_ts[1:]):
            bg = self.all_results[self.spec_ind[0]][del_t]['bg']
            h = bg.get_hist(bins=np.linspace(0., 12., 51))
            hl.plot1d(axs[ii], h, crosses=True,
                    label=r'BG, $\Delta T = {:.1e}$ s'.format(del_t))

            x = h.centers[0]
            norm = h.integrate().values
            axs[ii].semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
                        label=r'$\chi^2[{:.2f}\mathrm{{dof}},\ \eta={:.3f}]$'.format(bg.ndof, bg.eta))
            if ii // 3 == 2:
                axs[ii].set_xlabel(r'TS')
            if ii % 3 == 0:
                axs[ii].set_ylabel(r'$N$')
            axs[ii].set_ylim(3e-1, 3e5)
            axs[ii].legend(loc=1)
        plt.tight_layout()
        if self.savefigs:
            cut = self.min_log_e
            add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'stacking_bg_vs_time_{self.sample_str}{add_str}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def sensitivity_plot(self, gamma=2.0):
        """
        Plot the TS distributions (bg and signal) for the strengths
        that give us sensitivity and discovery. 
        Note, we round to the nearest trial that we have because
        we don't often inject exactly the right amount of signal

        :type gamma: float
        :param gamma: spectral index for the signal trials
        """
        fig, ax = plt.subplots(dpi=200)
        self.background_distribution(ax=ax, show=False)
        bins = np.linspace(0., 20., 51)

        sens_trials = self.sensitivity[gamma]['tss']
        sens_n = self.sensitivity[gamma]['n_sig']
        ninjs = np.unique(np.asarray(list(sens_trials.keys())))
        close_nsig_sens = self.find_nearest(ninjs, sens_n)
        plt.hist(
            sens_trials[close_nsig_sens], bins=bins, histtype='step',
            lw=1.5, label=r'$\langle n_{\mathrm{inj}} \rangle = $' +
            f'{close_nsig_sens}', zorder=2
            )

        disc_trials = self.discovery[gamma]['tss']
        disc_n = self.discovery[gamma]['n_sig']
        ninjs = np.unique(np.asarray(list(disc_trials.keys())))
        close_nsig_disc = self.find_nearest(ninjs, disc_n)
        plt.hist(
            disc_trials[close_nsig_disc], bins=bins, histtype='step',
            lw=1.5, label=r'$\langle n_{\mathrm{inj}} \rangle = $' +
            f'{close_nsig_sens}', zorder=2
            )

        ax.set_xlabel('TS')
        ax.set_ylabel(r'$N$')
        ax.legend()
        ax.set_ylim(3e-1, 1e4)
        if self.savefigs:
            cut = self.min_log_e
            add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'stacking_signal_and_bg_ts_dists_{self.sample_str}_delta_t_{self.delta_t:.2e}{add_str}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def plot_sensitivity_vs_time(
        self, ax=None, gamma=2.0, in_flux=True, show=True, 
        with_discovery=False):
        r"""Make a plot of sensitivity vs. time

        :type ax: matplotlib axes instance
        :param ax: pass if you are already working on a set of axes

        :type gamma: float
        :param gamma: spectral index for signal injection

        :type in_flux: bool
        :param in_flux: return in flux units if true (else n_events)

        :type show: bool
        :param show: if true, label axes and call plt.show

        :type with_discovery: bool
        :param with_discovery: Also plot discovery potential
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        if self.min_log_e is None:
            lab = 'None' 
        else:
            lab = "$\log_{10}\Big(E_{\mathrm{min}} / \mathrm{GeV} \Big) = $" \
                     + f" {self.min_log_e}"
        if in_flux:
            sens_dict = self.all_sensitivity
            disc_dict = self.all_discovery
        else:
            sens_dict = self.all_event_sensitivity
            disc_dict = self.all_event_discovery
        ax.plot(
            self.all_delta_ts, sens_dict[gamma], label=lab,
            ls='-', lw=2., color=self.min_log_cols[self.min_log_e])
        if with_discovery:
            disc_lab = ''
            ax.plot(
                self.all_delta_ts, disc_dict[gamma], 
                label=disc_lab, ls='--', lw=2.,
                color=self.min_log_cols[self.min_log_e]
                )
        if show:
            ax.set_title('$\gamma = $' + f"{gamma:.1f}")
            ax.set_xlabel('$\Delta T$ (s)')
            if in_flux:
                ax.set_ylabel(r'$E^2 \frac{dN}{dEdA} @ $' +
                    r'$1 \mathrm{TeV}$ (TeV cm${-2}$)')
            else:
                ax.set_ylabel(r'$\langle n_{\mathrm{inj}} \rangle$')
            ax.loglog()
            legend1 = ax.legend(fontsize=12, loc=4)
            if with_discovery:
                handles = []
                handles.append(Line2D([0], [0], color='grey', 
                                  lw=1.5, label=r"Sensitivity"))
                handles.append(
                    Line2D([0], [0], color='grey', ls='--', lw=1.5, 
                    label=f"{self.discovery_nsigma} sigma discovery pot."))
                legend2 = ax.legend(
                    handles=handles, loc=2, fontsize=12, frameon=False
                    )
                plt.gca().add_artist(legend1)
            ylims = ax.set_ylim()
            ax.set_ylim(ylims[0]*0.6, ylims[1])
        if self.savefigs:
            cut = self.min_log_e
            add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'stacking_sensitivity_vs_time_{self.sample_str}{add_str}_allflavor_{self.all_flavor}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def fitting_plot(self, gamma = 2.0, no_labels=False):
        """
        Include something to compare systematics (ie does bias go away
        with an energy cut)

        :type gamma: float or array
        :param gamma: Spectral index (or list thereof) to inject and fit

        :type no_labels: bool
        :param no_labels: raise this flag if iterating over this 
                          function multiple times for a subplot
        """
        if type(gamma) == float:
            gamma = [gamma]

        fig, axs = plt.subplots(1, 2, figsize=(9,5), dpi=200)
        plt.subplots_adjust(wspace=0.04, hspace=0.04)

        for gam in gamma:
            fit_trials = self.results[gam]['fit']
            n_inj = np.unique(fit_trials.ntrue)
            dns = np.mean(np.diff(n_inj))
            ns_bins = np.r_[n_inj - 0.5*dns, n_inj[-1] + 0.5*dns]
            expect_kw = dict(color=self.gam_cols[gam], ls='--', lw=1, zorder=-10)

            ax = axs[0]
            h = hl.hist(
                (fit_trials.ntrue, fit_trials.ns), 
                bins=(ns_bins, 100))
            hl.plot1d(
                ax, h.contain_project(1), errorbands=True, 
                drawstyle='default', label=r'$\gamma = {:.1f}$'.format(gam))

            lim = ns_bins[[0, -1]]
            ax.set_xlim(ax.set_ylim(lim))
            ax.plot(lim, lim, **expect_kw)
            ax.set_aspect('equal')

            ax = axs[1]
            h = hl.hist(
                (fit_trials.ntrue, fit_trials.gamma), 
                bins=(ns_bins, 100))
            hl.plot1d(
                ax, h.contain_project(1), errorbands=True, 
                drawstyle='default')
            ax.axhline(gam, **expect_kw)
            ax.set_xlim(axs[0].get_xlim())

            for ax in axs:
                ax.set_xlabel(r'$n_\mathrm{inj}$')
                ax.grid()
            axs[0].set_ylabel(r'$n_s$')
            axs[1].set_ylabel(r'$\gamma$')

        axs[0].legend(loc=2)
        plt.tight_layout()
        if self.savefigs:
            cut = self.min_log_e
            add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'stacking_fitting_bias_{self.sample_str}_delta_t_{self.delta_t:.2e}{add_str}_allflavor_{self.all_flavor}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def fitting_plot_panel(self):
        """bla"""
        pass

    def find_sens_vs_time(self):
        """
        Obtain a dictionary of the analysis sensitivity versus time for
        a given set of systematics
        """
        if self.all_results is None:
            self.get_all_sens()
        self.sens_vs_time = {gamma: [self.all_sensitivity[gamma][t]
            for t in self.all_delta_ts]
            for gamma in self.spec_ind}
        self.ev_sens_vs_time = {gamma: [self.all_event_sensitivity[gamma][t]
            for t in self.all_delta_ts]
            for gamma in self.spec_ind}
        self.disc_vs_time = {gamma: [self.all_discovery[gamma][t]
            for t in self.all_delta_ts]
            for gamma in self.spec_ind}
        self.ev_disc_vs_time = {gamma: [self.all_event_discovery[gamma][t]
            for t in self.all_delta_ts]
            for gamma in self.spec_ind}

    def get_all_sens(self):
        """Fetch all of the relevant analysis trials"""
        cut = self.min_log_e
        add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''

        results = {gamma: {} for gamma in self.spec_ind}
        for gamma in self.spec_ind:
            for t in self.all_delta_ts:
                try:
                    results[gamma][t] = np.load(self.trials_base + 'signal_results/' \
                        + f'{self.sample_str}_delta_t_{t:.2e}_gamma_{gamma}{add_str}_' \
                        + f'allflavor_{self.all_flavor}.pkl',
                        allow_pickle=True)
                except:
                    pass
        # {gamma: {t: 
        #     np.load(self.trials_base + 'signal_results/' \
        #         + f'delta_t_{t:.2e}_gamma_{gamma}{add_str}_' \
        #         + f'allflavor_{self.all_flavor}.pkl',
        #     allow_pickle=True) for t in self.all_delta_ts} 
        #     for gamma in self.spec_ind}
        event_sensitivity = {gamma: 
            np.array([results[gamma][ii]['sensitivity']['n_sig'] 
            for ii in self.all_delta_ts]) 
            for gamma in self.spec_ind}
        sensitivity = {gamma: 
            np.array([results[gamma][ii]['sensitivity']['E2dNdE'] 
            for ii in self.all_delta_ts]) 
            for gamma in self.spec_ind}
        event_discovery = {gamma: 
            np.array([results[gamma][ii]['discovery']['n_sig'] 
            for ii in self.all_delta_ts]) 
            for gamma in self.spec_ind}
        discovery = {gamma: 
            np.array([results[gamma][ii]['discovery']['E2dNdE'] 
            for ii in self.all_delta_ts]) 
            for gamma in self.spec_ind}

        self.all_results = results
        self.all_sensitivity = sensitivity
        self.all_event_sensitivity = event_sensitivity
        self.all_discovery = discovery
        self.all_event_discovery = event_discovery
        disc_dict = results[self.spec_ind[0]][self.all_delta_ts[0]]['discovery']
        self.disc_cl = disc_dict['CL']
        self.discovery_nsigma = disc_dict['nsigma']
        self.get_this_sens()

    def get_this_sens(self):
        """From all of the trials, extract the relevant time window"""
        if self.all_results is None:
            self.get_all_sens()
        self.results = {gamma: self.all_results[gamma][self.delta_t]
            for gamma in self.spec_ind}
        self.sensitivity = {gamma: self.results[gamma]['sensitivity']
            for gamma in self.spec_ind}
        self.discovery = {gamma: self.results[gamma]['discovery']
            for gamma in self.spec_ind}

    def get_differential_sens(self):
        """Load in the differential sensitivity results"""
        cut = self.min_log_e
        add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
        results =  np.load(self.trials_base + 'differential_sens/' \
            + f'{self.sample_str}_delta_t_{self.delta_t:.2e}{add_str}_' \
            + f'allflavor_{self.all_flavor}.pkl',
            allow_pickle=True)
        self.diff_sens = results

    def plot_differential_sens(self, ax=None, color=None, label=None,
        show_label=False):
        """Plot the differential sensitivity"""
        if self.diff_sens is None:
            self.get_differential_sens()
        log_es = np.linspace(0., 4., 5)
        mids = 10.**(log_es[:-1] + np.diff(log_es) / 2.)
        e_bins = 10.**log_es
        diff_sens = []
        for low_en, high_en in zip(e_bins[:-1], e_bins[1:]):
            diff_sens.append(
                self.diff_sens[f'sensitivity_{low_en:.1f}_' \
                    + f'{high_en:.1f}']['E2dNdE'])
        diff_sens = np.asarray(diff_sens)

        if ax is None:
            fig, ax = plt.subplots()
        bin_width = np.sqrt(10.)
        if color is None:
            color = self.min_log_cols[self.min_log_e]
        ax.errorbar(mids, diff_sens,  
            xerr=[mids-mids/bin_width, mids*bin_width-mids],
            marker='^', ls=':', #label=r'$\sin \delta = $' +'{:.1f}'.format(np.sin(np.radians(dec))),
            color=color, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$E_{\nu}$ (GeV)')
        ax.set_ylabel(r'$E^2 \frac{dN}{dEdA} @ $' +
                    r'$1 \mathrm{TeV}$ (TeV cm${-2}$)')
        if show_label:
            ax.legend(loc=1)

        if self.savefigs:
            cut = self.min_log_e
            add_str = f'_minLogE_{cut:.1f}' if cut is not None else ''
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'differential_sens_stacking_nova_{self.sample_str}_delta_t_{self.delta_t:.2e}{add_str}_allflavor_{self.all_flavor}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
    def set_seed(self, seed):
        """
        Reset the seed to a new value to get unique llh trials

        :type seed: int
        :param seed: new random seed for llh trials
        """
        self.rng_seed = seed

    def find_nearest(self, arr, val, in_val=True):
        """Find an element in an array closest to certain value

        :type arr: array-like
        :param arr: array to compare against

        :type val: float
        :param val: value to find nearby element for

        :type in_val: bool
        :param in_val: Return value if true, index if not

        :rtype: float or int
        :return: element in array nearby val or the index thereof
        """
        arr = np.asarray(arr)
        idx = (np.abs(arr - val)).argmin()
        if in_val:
            return arr[idx]
        else:
            return idx
    
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
        self.fontsize = kwargs.pop('fontsize', 16)
        self.gamma_colors = {2.0: 'C0', 2.5: 'C1', 3.0: 'C2'}
        self.central_90 = {2.0: (23.26, 4430.12), 2.5: (8.69, 1594.00), 
                           3.0: (4.76, 419.32)}
        
        #gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')
        master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
        gamma_df = master_df[master_df['gamma']==True]
        gamma_df = gamma_df.reset_index()
        names = gamma_df['Name']
        # delta_ts = np.logspace(2., 6.5, 10)
        delta_ts = np.logspace(-3.5, 1., 10)*86400.
        self.delta_ts = delta_ts
        all_novae = {name: {delta_t: GammaRayNova(name, delta_t=delta_t, **kwargs) for delta_t in delta_ts}
                             for name in gamma_df['Name']}
        self.all_novae = all_novae
        # Put a cap on the maximum allowed time window of 10^6 seconds
        self.full_time_novae = {}
        for name in gamma_df['Name']:
            tmp_del_t = gamma_df[gamma_df['Name'] == name]['gamma_stop'] - \
                            gamma_df[gamma_df['Name'] == name]['gamma_start']
            tmp_del_t = tmp_del_t.values[0].sec
            if tmp_del_t > 10.*86400:
                self.full_time_novae[name] = GammaRayNova(name, delta_t=86400.*10., **kwargs)
            else:
                self.full_time_novae[name] = GammaRayNova(name, **kwargs)
        # self.full_time_novae = {name: GammaRayNova(name, **kwargs) for name in gamma_df['Name']}
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
            fig, aaxs = plt.subplots(nrows=4, ncols=4, dpi=self.dpi,
                                    figsize=(14,14), sharey=True, sharex=True)
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
                except Exception as e:
                    print(e)
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
                except Exception as e:
                    print(e)
                    pass
            title = r"$\Delta T_{\nu} = \min(\Delta T_{\gamma}, 10 \;\mathrm{days})$"
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
            fig, aaxs = plt.subplots(nrows=4, ncols=4, dpi=self.dpi,
                                    figsize=(14,14), sharey=True, sharex=True)
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
            title = r"$\Delta T_{\nu} = \min(\Delta T_{\gamma}, 10 \;\mathrm{days})$"
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
            fig, aaxs = plt.subplots(nrows=4, ncols=4, dpi=self.dpi,
                                    figsize=(14,14), sharey=True, sharex=True)
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
            title = r"$\Delta T_{\nu} = \min(\Delta T_{\gamma}, 10 \;\mathrm{days})$"
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
            title = r"$\Delta T_{\nu} = \min(\Delta T_{\gamma}, 10 \;\mathrm{days})$"
        else:
            delta_t = kwargs['delta_t']
            for name in self.names:
                try:
                    self.all_novae[name][delta_t].compare_sens_to_photons(ax=ax, **kwargs)
                except Exception as e:
                    if self.verbose:
                        print(e, f"No sensitivity for nova {name}")
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
                delta_t = gamma_df[gamma_df['Name'] == name]['gamma_stop'] - \
                            gamma_df[gamma_df['Name'] == name]['gamma_start']
                delta_t = delta_t.values[0].sec
            except NameError:
                #gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')
                master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
                gamma_df = master_df[master_df['gamma']==True]
                gamma_df = gamma_df.reset_index()
                delta_t = gamma_df[gamma_df['Name'] == name]['gamma_stop'] - \
                            gamma_df[gamma_df['Name'] == name]['gamma_start']
                delta_t = delta_t.values[0].sec
            self.delta_t = delta_t
            self.delta_t_str = "full_gamma_time"
            if self.delta_t > 1e6:
                print("LOOKING AT NOVA WITH TIME WINDOW LONGER THAN 1e6 SECONDS")
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
        self.min_log_e = kwargs.pop('min_log_e', None)
        self.min_log_e_str = f'_minLogE_{self.min_log_e:.1f}' if self.min_log_e is not None else ''
        self.verbose = kwargs.pop('verbose', True)
        self.trials_base = '/data/user/apizzuto/Nova/csky_trials/'
        try:
            self.sensitivity_trials = {}
            self.discovery_trials = {}
            self.fitting_trials = {}
            for ind in self.spec_ind:
                name = self.name.replace(' ', '_')
                tmp_trials_base = self.trials_base + f'nova_*_{name}_delta_t_{self.delta_t_str}{self.min_log_e_str}_gamma_{ind:.1f}_allflavor_{self.all_flavor}_trials.pkl'
                trials_f = glob(tmp_trials_base)[0]
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
        ax.semilogy(nonpositive='clip')
        ax.set_ylim(1e-1, bg.n_total*1.5)
        ax.text(20, 6e2, self.name.replace('_', ' ') + '\n' + r'$\delta={:.0f}$'.format(self.dec*180./np.pi) \
            + r'$^{\circ}$' + '\n' + r'$\Delta T = {:.1f}$ days'.format(self.delta_t / 86400.), 
            ha='right', va='center', fontsize=self.fontsize)
        # ax.text(20, 1.0e2, r'$\Delta T = {:.1e}$ s'.format(self.delta_t),
        #     ha='right', va='center', fontsize=self.fontsize)
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
        ax.text(2, 42, self.name.replace('_', ' ') + '\n' + r'$\delta={:.0f}$'.format(self.dec*180./np.pi) \
            + r'$^{\circ}$' + '\n' + r'$\Delta T = {:.1f}$ days'.format(self.delta_t / 86400.), 
            ha='left', va='center', fontsize=self.fontsize)
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
        #gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')
        master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
        gamma_df = master_df[master_df['gamma']==True]
        gamma_df = gamma_df.reset_index()
        try:
            gamma_info = gamma_df.loc[gamma_df['Name'] == self.name]
            gamma = float(gamma_info['gamma_ind'])
            integrated_norm = float(gamma_info['gamma_norm'])*1e-7
            cutoff = float(gamma_info['gamma_cutoff'])
        except:
            print(f"Could not find Nova {self.name} in dataframe")
            return
        if ax is None:
            fig, ax = plt.subplots()
        ens = np.logspace(-1., 3., 500)
        
        handles = []
        alpha = kwargs.pop('alpha', 1.0)
        for spec in self.spec_ind:
            nu_sens_spec = np.power(ens, -spec)*self.sensitivity_trials[spec]['E2dNdE']*1e3*(1e-3)**(2.-spec)
            en_msk = (ens > self.central_90[spec][0]) & (ens < self.central_90[spec][1])
            ax.plot(ens[en_msk], nu_sens_spec[en_msk]*ens[en_msk]**2,
                   color=self.gamma_colors[spec], lw = 1.5, zorder=5*(5.-spec), alpha=alpha)
            handles.append(Line2D([0], [0], color=self.gamma_colors[spec], 
                                  lw=1.5, label=r"$\gamma = $" + f" {spec:.1f}"))
            
        photon_norm = self.calc_flux_norm(gamma, cutoff, integrated_norm)
        if np.isinf(cutoff):
            photon_flux = photon_norm * np.power(ens, -gamma)
        else:
            photon_flux = photon_norm * np.power(ens, -gamma) * \
                            np.exp(-ens / cutoff)
            photon_flux = np.where(photon_flux > 1e-100, photon_flux, 1e-100)
        ph_alpha = 0.7 if alpha > 0.7 else alpha*2.
        ax.plot(ens, photon_flux* ens**2. * self.delta_t,
                color=sns.xkcd_rgb['battleship grey'], alpha=ph_alpha)
        
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
        self.all_flavor = kwargs.pop('allflavor', True)
        self.spec_ind = kwargs.pop('index', [2., 2.5, 3.0])
        if type(self.spec_ind) is float:
            self.spec_ind = [self.spec_ind]
        self.min_log_e = kwargs.pop('min_log_e', 0.)
        self.verbose = kwargs.pop('verbose', False)
        if self.min_log_e is not None:
            self.low_en_bin = self.min_log_e
        else:
            self.low_en_bin = 0.0
        self.log_energy_bins = kwargs.pop('log_e_bins', np.linspace(0.0, 4., 31))
        self.sin_dec_bins = kwargs.pop('sin_dec_bins', np.linspace(-1., 1., 31))
        self.initialize_analysis()
        self.dpi = kwargs.pop('dpi', 150)
        self.savefigs = kwargs.pop('save', False)
        self.savepath = kwargs.pop('output', '/data/user/apizzuto/Nova/plots/')
        self.fontsize = kwargs.pop('fontsize', 16)

    def initialize_analysis(self, **kwargs):
        greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.5/'

        data_fs = sorted(glob(greco_base + 'IC86_20*data_with_angErr.npy'))
        exp = [np.load(data) for data in data_fs]
        exp = np.hstack(exp)
        if self.all_flavor:
            mcfiles = glob(greco_base + 'IC86_2012.nu*_with_angErr.npy')
            mc = np.load(mcfiles[0])
            for flav in mcfiles[1:]:
                mc = np.concatenate((mc, np.load(flav)))
        else:
            mcfile = glob(greco_base + 'IC86_2012.numu_with_angErr.npy')[0]
            mc = np.load(mcfile)
        grls = sorted(glob(greco_base + 'GRL/IC86_20*data.npy'))
        grl = [np.load(g) for g in grls]
        grl = np.hstack(grl)
        if self.min_log_e is not None:
            exp_msk = exp['logE'] > self.min_log_e
            exp = exp[exp_msk]
            mc_msk = mc['logE'] > self.min_log_e
            mc = mc[mc_msk]

            # If you cut out events, you need to fix your grl
            bin_lefts = grl['start']
            bin_rights = grl['stop']
            # Some bins out of order because of test runs
            sorted_bins = np.argsort(bin_lefts)
            bin_lefts_sorted = bin_lefts[sorted_bins]
            bin_rights_sorted = bin_rights[sorted_bins]
            bins = [None] * (bin_lefts_sorted.size * 2)
            bins[::2] = bin_lefts_sorted
            bins[1::2] = bin_rights_sorted
            count_hist = np.histogram(exp['time'], bins=bins)
            new_grl_counts = count_hist[0][::2]
            #Undo the bin sorting that we did
            undo_argsort = np.argsort(sorted_bins)
            new_grl_counts = new_grl_counts[undo_argsort]
            grl['events'] = new_grl_counts

        greco = cy.selections.CustomDataSpecs.CustomDataSpec(exp, mc, np.sum(grl['livetime']),
                                                             self.sin_dec_bins,
                                                             self.log_energy_bins,
                                                             grl=grl, key='GRECOv2.5', cascades=True)
        cy.CONF['mp_cpus'] = 5

        ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
        greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)
        self.ana = greco_ana

    def declination_pdf(self, **kwargs):
        if 'ax' not in kwargs:
            fig, ax = plt.subplots()
        else:
            ax = kwargs['ax']
        hl.plot1d (ax, self.ana[0].bg_space_param.h, crosses=True, color='k', label='histogram')
        sd = np.linspace (-1, 1, 300)
        ax.plot (sd, self.ana[0].bg_space_param(sindec=sd), label='spline')
        ax.set_ylim(0)
        ax.set_title(self.ana[0].plot_key)
        ax.set_xlabel(r'$\sin(\delta)$', fontsize=self.fontsize)
        ax.set_ylabel(r'probability density', fontsize=self.fontsize)
        ax.legend(loc='lower left')
        if self.savefigs:
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'GRECO_declination_pdf.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')

    def energy_pdf(self, **kwargs):
        for gamma in self.spec_ind:
            fig, ax = plt.subplots()
            eprm = self.ana[0].energy_pdf_ratio_model
            ss = dict(zip(eprm.gammas, eprm.ss_hl))
            things = hl.plot2d(ax, ss[gamma].eval(bins=100),
                               vmin=1e-2, vmax=1e2, log=True, cbar=True, cmap='RdBu_r')
            ax.set_title(self.ana[0].plot_key + r', $\gamma = $' + f" {gamma:.1f}")
            things['colorbar'].set_label(r'$S/B$')
            things['colorbar'].ax.tick_params(which='both', direction='out')
            ax.set_xlabel(r'$\sin(\delta)$', fontsize=self.fontsize)
            ax.set_ylabel(r'$\log_{10}(E/\mathrm{GeV})$', fontsize=self.fontsize)
            plt.tight_layout()
            if self.savefigs:
                for ftype in ['pdf', 'png']:
                    plt.savefig(self.savepath + \
                                f'GRECO_energy_pdf_gamma_{gamma:.1f}.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')

    def angular_error_plot(self, **kwargs):
        r'''Bla'''
        true_energy = kwargs.pop('true_energy', True)
        true_error = kwargs.pop('true_error', True)
        a = self.ana[-1]
        data, sig = a.data, a.sig
        if true_energy:
            energy_array = sig.true_energy
            xlab = r'$E_\mathrm{true}$ [GeV]'
        else:
            energy_array = sig.energy
            xlab = r'$E_\mathrm{reco}$ [GeV]'
        if true_error:
            error_array = sig.dpsi_deg
            ylab = r'$\Delta\Psi[\mathrm{true,reco}]~[^\circ]$'
        else:
            error_array = sig.sigma_deg
            ylab = r'$\sigma~[^\circ]$'

        h = hl.hist_slide(
            (5,1),
            (energy_array, error_array),
            # E^-2 weighting
            sig.oneweight*sig.true_energy**-2,
            bins=(10**np.r_[0.0:4.0:.25], np.r_[0:180.01:1])
        )

        # normalize along the angular error axis
        h = h.normalize(1)
        # get 20%, 50%, and 80% quantiles
        h2 = h.contain(1, .2); h5 = h.contain(1, .5); h8 = h.contain(1, .8)
        soft_colors = cy.plotting.soft_colors
        fig, ax = plt.subplots()
        # plot quantiles, emphasize median
        color = soft_colors[0]
        hl.fill_between(ax, 0, h2, color=color, alpha=.3, drawstyle='line')
        hl.fill_between(ax, 0, h5, color=color, alpha=.3, drawstyle='line')
        hl.fill_between(ax, 0, h8, color=color, alpha=.3, drawstyle='line')
        hl.plot1d (ax, h5, color=color, lw=2, drawstyle='default')

        # trick to get the legend handles colored right
        # try testing what happens if you just do hl.fill_between(..., label='...')
        nans = [np.nan, np.nan]
        ax.plot (nans, nans, color=color, lw=5, alpha=1 - (1-0.3)**1, label='80\%')
        ax.plot (nans, nans, color=color, lw=5, alpha=1 - (1-0.3)**2, label='50\%')
        ax.plot (nans, nans, color=color, lw=5, alpha=1 - (1-0.3)**3, label='20\%')

        # labels etc
        ax.semilogx(); ax.set_xlabel(xlab, fontsize=self.fontsize)
        ax.set_ylabel(ylab, fontsize=self.fontsize)
        ax.set_xlim(h.bins[0][1], h.bins[0][-2]); ax.set_ylim(0)
        ax.legend(loc='upper right', fontsize=self.fontsize); plt.tight_layout()
        if self.savefigs:
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'GRECO_energy_vs_error_true_energy_{true_energy}_true_errors_{true_error}.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')

    def errors_vs_declination(self, **kwargs):
        r'''Bla'''
        a = self.ana[-1]
        data, sig = a.data, a.sig
        h = hl.hist_slide(
            (5,1),
            (np.sin(data.dec), data.sigma_deg),
            # E^-2 weighting
            #sig.oneweight*sig.true_energy**-2,
            bins=(np.r_[-1.0:1.0:30j], np.r_[0:180.01:1])
        )

        # normalize along the angular error axis
        h = h.normalize(1)
        # get 20%, 50%, and 80% quantiles
        h2 = h.contain(1, .2); h5 = h.contain(1, .5); h8 = h.contain(1, .8)
        soft_colors = cy.plotting.soft_colors
        fig, ax = plt.subplots()
        # plot quantiles, emphasize median
        color = soft_colors[0]
        hl.fill_between(ax, 0, h2, color=color, alpha=.3, drawstyle='line')
        hl.fill_between(ax, 0, h5, color=color, alpha=.3, drawstyle='line')
        hl.fill_between(ax, 0, h8, color=color, alpha=.3, drawstyle='line')
        hl.plot1d (ax, h5, color=color, lw=2, drawstyle='default')

        # trick to get the legend handles colored right
        # try testing what happens if you just do hl.fill_between(..., label='...')
        nans = [np.nan, np.nan]
        ax.plot (nans, nans, color=color, lw=5, alpha=1 - (1-0.3)**1, label='80\%')
        ax.plot (nans, nans, color=color, lw=5, alpha=1 - (1-0.3)**2, label='50\%')
        ax.plot (nans, nans, color=color, lw=5, alpha=1 - (1-0.3)**3, label='20\%')

        # labels etc
        ax.set_xlabel(r'$\sin(\delta)$', fontsize=self.fontsize)
        ax.set_ylabel(r'$\sigma~[^\circ]$', fontsize=self.fontsize)
        ax.set_xlim(h.bins[0][1], h.bins[0][-2]); ax.set_ylim(0)
        ax.legend(loc='upper right', fontsize=self.fontsize); plt.tight_layout()
        if self.savefigs:
            for ftype in ['pdf', 'png']:
                plt.savefig(self.savepath + \
                            f'GRECO_error_vs_sindec.{ftype}', 
                            dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def effective_area(self, **kwargs):
        r'''Bla'''
        if 'ax' in kwargs:
            ax = kwargs['ax']
        else:
            fig, ax = plt.subplots()

        mc = self.ana.anas[0].sig
        decs = np.radians(np.asarray([[-90., -30.], [-30., 30.], [30., 90.]]))
        sin_decs = np.sin(decs)
        E_bins = np.logspace(0., 9., 61)
        logE_bins = np.log10(E_bins)
        dlog_E = np.diff(logE_bins)

        for ii, (low_dec, high_dec) in enumerate(sin_decs):
            d_omega = 2.*np.pi*np.abs(high_dec - low_dec)
            dec_msk = np.sin(mc['true_dec']) > low_dec
            dec_msk *= np.sin(mc['true_dec']) < high_dec
            mc_cut = mc[dec_msk]
            weights = mc_cut['oneweight'] / (1e4 * mc_cut['true_energy'] * dlog_E[np.digitize(
                np.log10(mc_cut['true_energy']), bins = logE_bins) -1] * d_omega * np.log(10.))
            lab = f"{np.degrees(np.arcsin(low_dec)):.1f}" + r'$^{\circ} <  \delta < \; $' \
                            + f"{np.degrees(np.arcsin(high_dec)):.1f}" + r'$^{\circ}$'
            ax.hist(mc_cut['true_energy'], bins = E_bins, 
                    weights = weights,
                    histtype = 'step', linewidth = 2., 
                    label=lab)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(4e-1, 3e3)
        ax.set_xlabel(r'$E_{\nu}$ (GeV)', fontsize = 20)
        ax.set_ylabel('$A_{\mathrm{eff}}$ (m$^{2}$)', fontsize=20)
        ax.legend(loc=2, fontsize=14, frameon=False)

class SynthesisPlots():
    r'''Combines GRECO information with general nova information'''
    def __init__(self, **kwargs):
        self.nova_info = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
        self.greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.5/'
        grls = sorted(glob(self.greco_base + 'GRL/IC86_20*data.npy'))
        self.grls = [np.load(g) for g in grls]
        grl = np.hstack(grls)
        self.master_grl = grl
        datafs = sorted(glob(self.greco_base + 'IC86_20*data.npy'))
        self.datafs = datafs
        datas = [np.load(dataf) for dataf in datafs]
        self.exps = datas
        self.palette = sns.color_palette('colorblind')
        self.save = kwargs.pop('save', False)
        if self.save:
            self.savedir = kwargs.pop('savedir', './')
            self.dpi = kwargs.pop('dpi', 150)

    def __mids(self, arr):
        return arr[:-1] + (np.diff(arr) / 2.)

    def __find_nearest(self, array, values):
            array = np.asarray(array)
            idx = [(np.abs(array - value)).argmin() for value in values]
            return array[idx]

    def gamma_lightcurve_with_greco_rate(self, show_opt=False):
        vals, ts = [], []
        uptime = []
        col_ind = []

        for grl, exp in zip(self.grls, self.exps):
            time_bins = np.arange(grl['start'][0], grl['stop'][-1], 14.)
            time_bins = np.append(time_bins, grl['stop'][-1])
            time_bins = self.__find_nearest(grl['stop'], time_bins)
            h, b = np.histogram(exp['time'], bins = time_bins)
            b = self.__mids(b)
            vals.append(h)
            ts.append(b)
            ut, b = np.histogram(grl['start'], bins = time_bins, weights = grl['livetime'])
            uptime.append(ut)

        rates = [np.where(up != 0., v / (up * 86400.), 0.) for up, v in zip(uptime, vals)]
        val_err = [np.sqrt(v) for v in vals]
        rate_err = [np.where(up != 0., v_err / (up * 86400.), 0.) for up, v_err in zip(uptime, val_err)]

        fig, ax = plt.subplots(figsize = (12, 4), dpi=200)

        for i in range(len(ts)):
            plt.errorbar(ts[i], rates[i], yerr = rate_err[i], 
                mec = self.palette[i], mfc = self.palette[i], 
                ecolor = self.palette[i], ls = '', 
                label = self.datafs[i][-18:-9].replace('_', ' '))

        has_labeled_gam = False
        for i, nova in self.nova_info.iterrows():
            if nova['gamma']:
                if not has_labeled_gam:
                    plt.axvspan(nova['gamma_start'].mjd, nova['gamma_stop'].mjd,
                        color = sns.xkcd_rgb['light navy'], alpha = 0.2, 
                        label = r"$\gamma$ detected nova")
                    has_labeled_gam = True
                else:
                    plt.axvspan(nova['gamma_start'].mjd, nova['gamma_stop'].mjd,
                        color = sns.xkcd_rgb['light navy'], alpha = 0.2)
            if show_opt:
                if i == 0:
                    plt.axvline(nova['Date'].mjd, ls='--', alpha = 0.4,
                        color = sns.xkcd_rgb['light navy'], label='Optical peak')
                else:
                    plt.axvline(nova['Date'].mjd, ls='--', alpha = 0.4,
                        color = sns.xkcd_rgb['light navy'])
            
        plt.legend(loc = (1.01, 0.03), frameon=False, fontsize = 16)
        plt.xlim(56000, 59000)
        plt.ylim(0.0039, 0.0052)
        plt.ylabel('Rate (Hz)')
        plt.xlabel('Time (MJD)')
        # plt.text(56200, 0.0041, 'IceCube Preliminary', color=sns.xkcd_rgb['tomato red'], fontsize=20)
        if self.save:
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savedir + \
                                f'lightcurve_with_greco_rate.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def all_sky_scatter_plot(self, fig = None, ax = None, 
        show_legend=True, **kwargs):
        if fig is None and ax is None:
            fig = plt.figure(figsize=(8,4), dpi=200, facecolor='w')
            ax = fig.add_subplot(111, projection='mollweide')
        ax.grid(True, alpha = 0.35, zorder=1, ls = '--')

        gplane = SkyCoord(frame='galactic', b = np.zeros(5000)*u.degree, 
            l = np.linspace(0.0, 360., 5000)*u.degree)
        gplane_icrs = gplane.icrs
        gcent = SkyCoord(frame='galactic', b = [0.0]*u.degree, 
            l = [0.0]*u.degree)
        gcent_icrs = gcent.icrs
        cols = [sns.xkcd_rgb['orange pink'] if k is True 
            else sns.xkcd_rgb['grey'] for k in self.nova_info['gamma']]
        s = np.array([14 if k is True else 10 for k in self.nova_info['gamma']])

        legend_els = [ 
            Line2D([0], [0], marker='^', ls = '', 
                color=sns.xkcd_rgb['orange pink'], 
                label=r'$\gamma$ detected'),
            Line2D([0], [0], marker='o', ls = '', 
                color=sns.xkcd_rgb['grey'], 
                label='Optical only')
            ]

        gamma_msk = self.nova_info['gamma']
        equatorial = SkyCoord(ra=self.nova_info['RA'][~gamma_msk]*u.deg, 
            dec=self.nova_info['Dec'][~gamma_msk]*u.deg)
        gamma_coords = SkyCoord(ra=self.nova_info['RA'][gamma_msk]*u.deg, 
            dec=self.nova_info['Dec'][gamma_msk]*u.deg)

        ax.scatter(-1*equatorial.ra.wrap_at('360d').radian + np.pi, 
            equatorial.dec.radian,
            zorder=20, s = 10, c = sns.xkcd_rgb['grey'])
        ax.scatter(-1*gamma_coords.ra.wrap_at('360d').radian + np.pi, 
            gamma_coords.dec.radian,
            zorder=20, s = 14, marker='^',
            c = sns.xkcd_rgb['orange pink'])

        ax.scatter(-1.*gplane_icrs.ra.wrap_at('360d').radian + np.pi, 
            gplane_icrs.dec.radian,
            zorder=10, c = 'k', s = 0.5)

        ax.set_xticklabels(["{:.0f}".format(v) + r'$^{\circ}$' 
            for v in np.linspace(330., 30., 11)], fontsize = 14)
        ax.set_yticklabels(["{:+.0f}".format(v) + r'$^{\circ}$' 
            for v in np.linspace(-75., 75., 11)], fontsize = 14)
        plt.text(110.*np.pi / 180., -45 * np.pi / 180, 'Equatorial\n(J2000)')
        if show_legend:
            ax.legend(loc=(0.2, -0.18), handles=legend_els, ncol = 2, 
                frameon=False)
        if self.save:
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savedir + \
                                f'all_sky_nova_scatter_plot.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def mollview_with_sensitivity(self, **kwargs):
        fig = plt.figure(figsize=(8,4), dpi=200, facecolor='w')
        ax = fig.add_subplot(111, projection='mollweide')
        self.all_sky_scatter_plot(fig=fig, ax=ax, show_legend=False)
        sens_vs_dec = np.load('/data/user/apizzuto/Nova/csky_trials/sens_vs_dec/' 
            + 'sens_vs_dec_delta_t_8.64e+04_gamma_2.0_allflavor_True_trials.pkl',
            allow_pickle=True)
        spl = UnivariateSpline(np.sin(np.array(sens_vs_dec['dec'])), 
            np.array(sens_vs_dec['sens']))

        nlats, nlons = (73, 145)
        lats = np.linspace(-np.pi / 2, np.pi / 2, nlats)
        lons = np.linspace(-np.pi, np.pi, nlons)
        lons, lats = np.meshgrid(lons, lats)

        lat_deg = np.rad2deg(lats)
        lon_deg = np.rad2deg(lons)

        sens = spl(np.sin(lats))

        cmap = sns.color_palette("crest", 200)
        cmap = mpl.colors.ListedColormap(cmap)

        cc = ax.pcolormesh(lons, lats, sens,
            shading='nearest',
            cmap = cmap
            #transform=ccrs.PlateCarree()
            )

        cbaxes = fig.add_axes([0.18, 0.0, 0.4, 0.06]) 
        cbaxes.axis('off')
        cbar = fig.colorbar(cc, ax=cbaxes, orientation='horizontal',
            fraction=1.0, pad=0.0,
            label = r'$E^2 \frac{dN}{dEdA} @ $' +
                    r'$1 \mathrm{TeV}$ (TeV cm$^{-2}$)')
        
        cbar.ax.tick_params(direction='out')
        ax.grid(True, alpha = 0.35, zorder=2, ls = '--')

        legend_els = [ 
            Line2D([0], [0], marker='^', ls = '', 
                color=sns.xkcd_rgb['orange pink'], 
                label=r'$\gamma$ detected'),
            Line2D([0], [0], marker='o', ls = '', 
                color=sns.xkcd_rgb['grey'], 
                label='Optical only')
            ]

        ax.legend(loc=(0.62, -0.22), handles=legend_els, ncol = 1, 
            frameon=False)

        if self.save:
            for ftype in ['pdf', 'png']:
                    plt.savefig(self.savedir + \
                                f'all_sky_nova_scatter_plot_with_sens.{ftype}', 
                                dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()