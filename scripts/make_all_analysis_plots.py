import argparse
import os
from plotting import StackingPlots, GammaCatalog, GRECOPlots, SynthesisPlots
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')


def make_gamma_correlation_plots(cat):
    print("\n\t - Beginning with full gamma ray duration")
    cat.background_ts_panel()
    cat.gamma_fitting_panel()
    cat.compare_photon_to_nus()
    for gamma in [2.0, 2.5, 3.0]:
        cat.ns_fitting_panel(gamma=gamma)
        for disc in [True, False]:
            cat.sensitivity_vs_time(discovery=disc, annotate=True, gamma=gamma)
    plt.close('all')
    print("\t - Finished plots with full gamma ray duration")

    # print("\t - Beginning plots with fixed time windows")
    # for delta_t in cat.delta_ts[::2]:
    #     # Note that we are skipping some windows because otherwise it's
    #     # just too many plots
    #     cat.background_ts_panel(delta_t=delta_t)
    #     cat.gamma_fitting_panel(delta_t=delta_t)
    #     cat.compare_photon_to_nus(delta_t=delta_t)
    #     for gamma in [2.0, 2.5, 3.0]:
    #         cat.ns_fitting_panel(gamma=gamma, delta_t=delta_t)
    #     plt.close('all')
    # print("\t - Finished plots with fixed time windows")


def make_stacking_plots(stack):
    print("\n\t Beginning stacking plots")
    stack.get_all_sens()
    stack.sensitivity_plot()
    stack.background_distribution()
    stack.background_vs_time()
    # stack.likelihood_scan(truth=True)
    stack.fitting_plot(gamma=[2.0, 2.5, 3.])
    # for gam in [2.0, 2.5, 3.0]:
    #     for disc in [True, False]:
    #         stack.sensitivity_efficiency_curve(
    #             gamma=gam, discovery=disc
    #             )
    print("\t Finished stacking plots")


def make_GRECO_plots(gplots):
    gplots.declination_pdf()
    gplots.energy_pdf()
    for true_en in [True, False]:
        for true_err in [True, False]:
            gplots.angular_error_plot(true_energy=true_en, true_error=true_err)
    gplots.errors_vs_declination()


def make_synthesis_plots(syn_plots):
    syn_plots.gamma_lightcurve_with_greco_rate()
    syn_plots.all_sky_scatter_plot()
    syn_plots.mollview_with_sensitivity()
    syn_plots.make_magnitude_hist()
    syn_plots.make_weights_scatter_plot()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot creation for GRECO-Nova analysis')
    parser.add_argument(
        '--output_path', type=str,
        default='/data/user/apizzuto/Nova/plots/')
    parser.add_argument(
        '--stacking_time', type=float, default=86400.,
        help="Stacking time window")
    args = parser.parse_args()

    print("Making figures for the unstacked analysis")
    print(f"\t Figures will be saved to {args.output_path}")
    cat = GammaCatalog(save=True, output=args.output_path)
    make_gamma_correlation_plots(cat)

    print("Making figures for the stacked analysis with parameters:")
    print(f"\t Stacking time window: {args.stacking_time:.2e} s")
    print(f"\t Figures will be saved to {args.output_path}")
    stack_plots = StackingPlots(
        args.stacking_time, min_log_e=None, allflavor=args.allflavor,
        save=True, output=args.output_path)
    make_stacking_plots(stack_plots)

    print("Making figures for the GRECO dataset")
    print(f"\t Figures will be saved to {args.output_path}")
    gplots = GRECOPlots(save=True, output=args.output_path)
    make_GRECO_plots(gplots)

    print("Making figures that synthesize top level nova info and GRECO info:")
    print(f"\t Figures will be saved to {args.output_path}")
    syn_plots = SynthesisPlots(save=True, savedir=args.output_path)
    make_synthesis_plots(syn_plots)
