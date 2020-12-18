import numpy as np
import matplotlib as mpl
#try:
#    mpl.use('agg')
#except:
#    pass 
import matplotlib.pyplot as plt
import pandas as pd
import astropy as ap
import pickle
import scipy as sp
import scipy.stats as st
import mpmath
import seaborn as sns
from matplotlib.lines import Line2D

import histlite as hl
import csky as cy

from glob import glob
mpl.style.use('/home/apizzuto/Nova/scripts/novae_plots.mplstyle')
master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
gamma_df = master_df[master_df['gamma']==True]
gamma_df = gamma_df.reset_index()
#gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')

class StackingPlots():
    r'''Helper class to make analysis plots for the
    stacking analysis'''
    
    def __init__(self, delta_t, **kwargs):
        self.delta_t = delta_t
        self.min_log_e = kwargs.pop('min_log_e', 0.)
        self.all_flavor = kwargs.pop('allflavor', False)
        self.spec_ind = kwargs.pop('index', [2., 2.5, 3.])
        self.verbose = kwargs.pop('verbose', False)
        self.df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
        self.names = self.df['Name']
        #self.initialize_analysis()
        self.dpi = kwargs.pop('dpi', 150)
        self.savefigs = kwargs.pop('save', False)
        self.savepath = kwargs.pop('output', '/data/user/apizzuto/Nova/plots/')
        self.fontsize = kwargs.pop('fontsize', 16)
        self.show = kwargs.pop('show', True)
        self.trials_base = '/home/apizzuto/Nova/scripts/stacking_sens_res/'
        self.all_delta_ts = np.logspace(-3., 1., 9)[:-1]*86400.
        self.all_results = None
        self.ana = None

    def initialize_analysis(self):
        if self.verbose:
            print("Initializing csky analysis")
        greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.4/'

        data_fs = sorted(glob(greco_base + 'IC86_20*data_with_angErr.npy'))
        exp = [np.load(data) for data in data_fs]; exp = np.hstack(exp)
        mc = np.load(greco_base + 'IC86_2012.numu_merged_with_angErr.npy')
        grls = sorted(glob(greco_base + 'GRL/IC86_20*data.npy'))
        grl = [np.load(g) for g in grls]; grl = np.hstack(grl)

        greco = cy.selections.CustomDataSpecs.CustomDataSpec(exp, mc, np.sum(grl['livetime']), 
                                                            np.linspace(-1., 1., 31),
                                                            np.linspace(0., 4., 31), 
                                                            grl=grl, key='GRECOv2.4', cascades=True)

        ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
        greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)
        ras = self.df['RA']; decs = self.df['Dec']
        delta_t = np.ones_like(ras)*self.delta_t/86400.
        mjds = np.array([t.mjd for t in self.df['Date']])
        conf = {'extended': True, 'space': "ps", 'time': "transient", 'sig': 'transient'}
        src = cy.utils.Sources(ra=np.radians(ras), 
                       dec=np.radians(decs), 
                       mjd=mjds, 
                       sigma_t=np.zeros_like(delta_t), 
                       t_100=delta_t)

        cy.CONF['src'] = src
        cy.CONF['mp_cpus'] = 10
        self.conf = conf
        self.src = src
        self.greco = greco
        self.ana = greco_ana
    
    def likelihood_scan(self, n_inj=0., inj_gamma =2.0, truth=False):
        r'''Perform a single trial, with or without signal injected,
        and calculate the likelihood landscape
        
        Parameters:
        -----------
         - n_inj: float
            Number of injected signal events
         - inj_gamma: float
            Injected spectral index
         - truth: bool
            True llh scan (only permitted after unblinding)
        '''
        if self.ana is None:
            self.initialize_analysis()
        if truth and n_inj != 0.:
            print("Can't look at truth with signal")
        elif truth:
            print("ONLY LOOK AT TRUTH IF YOU HAVE PERMISSION TO UNBLIND")
        else:
            tr = cy.get_trial_runner(self.conf, ana=self.ana, src=self.src, 
                    inj_conf={'flux': cy.hyp.PowerLawFlux(inj_gamma)})
        trial = tr.get_one_trial(n_sig=n_inj, TRUTH=truth)
        llh = tr.get_one_llh_from_trial(trial)
        nss = np.linspace(0., 120., 100)
        gammas = np.linspace(1., 4., 100)
        llh_space = llh.scan_ts(ns=nss, gamma=gammas)
        mesh_ns, mesh_gam = np.meshgrid(nss, gammas)
        best_fit_ts = np.max(llh_space[0])
        best_fit_ns = mesh_ns.flatten()[np.argmax(llh_space[0])]
        best_fit_gamma = mesh_gam.flatten()[np.argmax(llh_space[0])]
        # Calculate critical values
        chi2 = st.chi2(2., loc=0., scale=1)
        crit_med = chi2.isf(0.5)
        crit_90 = chi2.isf(1.0-0.9)
        crit_99 = chi2.isf(1.0-0.99)

        fig, ax1 = plt.subplots(dpi=150)
        ts_space = llh_space[0]

        im = ax1.pcolor(mesh_ns, mesh_gam, -1.*(ts_space - best_fit_ts), 
                            cmap="Blues_r", vmin=0, vmax=50)#, norm=LogNorm(vmin=0.01, vmax=20))
        ax_c = plt.colorbar(im, ax=ax1, format='$%.1f$')
        ax_c.ax.tick_params(direction='out')
        ax_c.set_label(r"$-2\Delta\,\mathrm{LLH}$")

        ax1.contour(mesh_ns, mesh_gam, -1.*(ts_space - best_fit_ts), 
                    #levels=[1,4, 9, 16, 25], 
                    levels=[crit_med, crit_90, crit_99], colors="w")
        if not truth:
            ax1.plot(n_inj, inj_gamma, marker="^", color="w", markersize=10, label='Truth' )
        ax1.plot(best_fit_ns, best_fit_gamma, marker="*", color="w", 
                        markersize=10, label='Best-fit')
        ax1.set_ylabel(r"$\gamma$")
        ax1.set_xlabel(r"$n_\mathrm{s}$")
        ax1.legend(loc=4, facecolor=sns.xkcd_rgb['light grey'])
        fig.tight_layout()

    def background_distribution(self):
        """
        Plot the background TS distribution for a given time 
        window
        """
        fig, ax = plt.subplots(dpi=200)
        bg = self.results[self.spec_ind[0]]['bg']
        h = bg.get_hist(bins=50)
        hl.plot1d(ax, h, crosses=True,
                label='{} bg trials'.format(bg.n_total))

        x = h.centers[0]
        norm = h.integrate().values
        ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
                    label=r'$\chi^2[{:.2f}\mathrm{{dof}},\ \eta={:.3f}]$'.format(bg.ndof, bg.eta))

        ax.set_xlabel(r'TS')
        ax.set_ylabel(r'number of trials')
        ax.legend()
        plt.tight_layout()

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
        for ii, del_t in enumerate(self.all_delta_ts):
            bg = self.all_results[self.spec_ind[0]][del_t]['bg']
            h = bg.get_hist(bins=np.linspace(0., 10., 41))
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
            axs[ii].set_ylim(3e-1, 8e3)
            axs[ii].legend(loc=1)
        plt.tight_layout()

    def sensitivity_plot(self):
        """
        Plot the TS distributions (bg and signal) for the strengths
        that give us sensitivity and discovery. 
        Note, we round to the nearest trial that we have because
        we don't often inject exactly the right amount of signal
        """
        pass

    def plot_sensitivity_vs_time(self):
        pass

    def fitting_plot(self, gamma = 2.0, no_labels=False):
        """
        Include something to compare systematics (ie does bias go away
        with an energy cut)

        Parameters:
        -----------
        - gamma: float or array
            Spectral index (or list thereof) to inject and fit
        - no_labels: bool (default False)
            raise this flag if iterating over this function multiple 
            times for a subplot
        """
        if type(gamma) == float:
            gamma = [gamma]

        fig, axs = plt.subplots(1, 2, figsize=(9,5), dpi=200)
        plt.subplots_adjust(wspace=0.04, hspace=0.04)
        gam_cols = {2.0: 'C0', 2.5: 'C1', 3.0: 'C3'}

        for gam in gamma:
            fit_trials = self.results[gam]['fit']
            n_inj = np.unique(fit_trials.ntrue)
            dns = np.mean(np.diff(n_inj))
            ns_bins = np.r_[n_inj - 0.5*dns, n_inj[-1] + 0.5*dns]
            expect_kw = dict(color=gam_cols[gam], ls='--', lw=1, zorder=-10)

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
        add_str = f'minLogE_{cut:.1f}' if cut is not None else ''
        results = {gamma: {t: 
            np.load(self.trials_base + f'delta_t_{t:.2e}_gamma_{gamma}{add_str}.pkl',
            allow_pickle=True) for t in self.all_delta_ts} 
            for gamma in self.spec_ind}
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
        delta_ts = np.logspace(2., 6.5, 10)
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
                self.trials_base = self.trials_base + f'nova_*_{self.name}_delta_t_{self.delta_t_str}_minLogE_{self.min_log_e:.1f}_gamma_{ind:.1f}_allflavor_{self.all_flavor}_trials.pkl'
                trials_f = glob(self.trials_base)[0]
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
            photon_flux = np.where(photon_flux > 1e-100, photon_flux, 1e-100)
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
        self.log_energy_bins = kwargs.pop('log_e_bins', np.linspace(self.low_en_bin, 4., 31))
        self.sin_dec_bins = kwargs.pop('sin_dec_bins', np.linspace(-1., 1., 31))
        self.initialize_analysis()
        self.dpi = kwargs.pop('dpi', 150)
        self.savefigs = kwargs.pop('save', False)
        self.savepath = kwargs.pop('output', '/data/user/apizzuto/Nova/plots/')
        self.fontsize = kwargs.pop('fontsize', 16)

    def initialize_analysis(self, **kwargs):
        greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.4/'

        data_fs = sorted(glob(greco_base + 'IC86_20*data_with_angErr.npy'))
        exp = [np.load(data) for data in data_fs]
        exp = np.hstack(exp)
        if self.all_flavor:
            mcfiles = glob(greco_base + 'IC86_2012.nu*_with_angErr.npy')
            mc = np.load(mcfiles[0])
            for flav in mcfiles[1:]:
                mc = np.concatenate((mc, np.load(flav)))
        else:
            mcfile = glob(greco_base + 'IC86_2012.numu_with_angErr.npy')
            mc = np.load(mcfile)
        grls = sorted(glob(greco_base + 'GRL/IC86_20*data.npy'))
        grl = [np.load(g) for g in grls]
        grl = np.hstack(grl)
        if self.min_log_e is not None:
            exp_msk = exp['logE'] > self.min_log_e
            exp = exp[exp_msk]
            mc_msk = mc['logE'] > self.min_log_e
            mc = mc[mc_msk]

        greco = cy.selections.CustomDataSpecs.CustomDataSpec(exp, mc, np.sum(grl['livetime']),
                                                             self.sin_dec_bins,
                                                             self.log_energy_bins,
                                                             grl=grl, key='GRECOv2.4', cascades=True)
        cy.CONF['mp_cpus'] = 5

        ana_dir = cy.utils.ensure_dir('/data/user/apizzuto/csky_cache/greco_ana')
        greco_ana = cy.get_analysis(cy.selections.repo, greco, dir=ana_dir)
        self.ana = greco_ana

    def declination_pdf(self, **kwargs):
        fig, ax = plt.subplots()
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

    def effective_area(self, **kwargs):
        r'''Bla'''
        pass
