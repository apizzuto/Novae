import matplotlib.pyplot as plt

class StackingPlots():
    r'''Helper class to make analysis plots for the
    stacking analysis'''
    
    def __init__(self, **kwargs):
        pass

class CatalogPlot():
    r'''Helper class to make analysis plots for the
    individually gamma-ray novae'''

    def __init__(self, **kwargs):
        pass

class GammaRayNova():
    r'''Holds information about analysis for 
    individual gamma-ray novae'''

    def __init__(self, name, delta_t, **kwargs):
        self.delta_t = delta_t
        self.name = name
        self.all_flavor = kwargs.pop('allflavor', True)
        self.spec_ind = kwargs.pop('index', 2.)
        self.min_log_e = kwargs.pop('min_log_e', 0.)
        self.verbose = kwargs.pop('verbose', False)
        self.trials_base = '/data/user/apizzuto/Nova/csky_trials/'
        try:
            trials_f = glob(self.trials_base + f'nova_*_{self.name}_delta_t_{self.delta_t:.2e}_minLogE_{self.min_log_e:.1f}_gamma_{self.spec_ind:.1f}_allflavor_{self.all_flavor}_trials.pkl')[0]
            with open(trials_f, 'rb') as f:
                nova_trials = pickle.load(f)
            self.sensitivity_trials = nova_trials['sensitivity']
            self.discovery_trials = nova_trials['discovery']
            self.fitting_trials = nova_trials['fit']
            self.background = nova_trials['bg']
            self.ra = nova_trials['source_info']['ra']
            self.dec = nova_trials['source_info']['dec']
        except Exception as e:
            if self.verbose:
                print(f"Could not find trials for Nova {self.name} with analysis details:")
                print(f"\t All flavor: {self.all_flavor}\t Gamma: {self.spec_ind}\t Min E: log10({self.min_log_e})")
                print(f"\t Duration: {self.delta_t:.2e} s")
        self.fontsize = kwargs.pop('fontsize', 16)
        self.units_ref_str = 'TeV cm^-2 @ 1 TeV'
                
    def background_ts_plot(self, ax=None, label_axes=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        bg = self.background
        h = bg.get_hist(bins=40, range=(0, 20))
        hl.plot1d(ax, h, crosses=True)
        norm = h.integrate().values
        ts = np.linspace(.1, h.range[0][1], 100)
        ax.plot(ts, norm * bg.pdf(ts))
        ax.semilogy(nonposy='clip')
        ax.set_ylim(3e-3, bg.n_total*1.5)
        ax.text(20, 6e2, self.name.replace('_', ' ') + '\n' + r'$\delta={:.0f}$'.format(self.dec*180./np.pi), 
                ha='right', va='center', fontsize=self.fontsize)
        if label_axes:
            ax.set_xlabel('TS', fontsize=self.fontsize)
            ax.set_ylabel('Number of Trials', fontsize=self.fontsize)

class GRECOPlots():
    r'''Helper class to make plots that describe the 
    overall dataset and its performance'''

    def __init__(self, **kwargs):
        pass
