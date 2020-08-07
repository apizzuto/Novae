import argparse
from plotting import StackingPlots, GammaCatalog, GammaRayNova, GRECOPlots

def make_gamma_correlation_plots(cat):
    cat.background_ts_panel()
    cat.gamma_fitting_panel()
    cat.compare_photon_to_nus()
    for gamma in [2.0, 2.5, 3.0]:
        cat.ns_fitting_panel(gamma=gamma)
        for disc in [True, False]:
            cat.sensitivity_vs_time(discovery=disc, annotate=True, gamma=gamma)
    for delta_t in cat.delta_ts:
        cat.background_ts_panel(delta_t=delta_t)
        cat.gamma_fitting_panel(delta_t=delta_t)
        cat.compare_photon_to_nus(delta_t=delta_t)
        for gamma in [2.0, 2.5, 3.0]:
            cat.ns_fitting_panel(gamma=gamma, delta_t=delta_t)
            for disc in [True, False]:
                cat.sensitivity_vs_time(discovery=disc, annotate=True, gamma=gamma)

def make_stacking_plots(stack):
    pass

def make_GRECO_plots():
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot creation for GRECO-Nova analysis')
    parser.add_argument('--minLogE', type=float, default=None, help='Cut on the minimum reco energy')
    parser.add_argument('--allflavor', action='store_true', default=False, help="All neutrino flavors in MC")
    parser.add_argument('--output_path', type=str, default='/data/user/apizzuto/Nova/plots/')
    args = parser.parse_args()

    print("Makiing bla bla bla with parameters bla bla")
    cat = GammaCatalog(allflavor=args.allflavor, min_log_e=args.minLogE, save=True, 
            output=args.output_path)
    make_gamma_correlation_plots(cat)

    #print("Makiing bla bla bla with parameters bla bla")
    #cat = GammaCatalog()
    #make_gamma_correlation_plots(cat)

    #print("Makiing bla bla bla with parameters bla bla")
    #cat = GammaCatalog()
    #make_gamma_correlation_plots(cat)