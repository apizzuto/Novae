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
    print("\t - Finished plots with full gamma ray duration\n")


def make_stacking_plots(stack):
    print("\n\t Beginning stacking plots")
    stack.get_all_sens()
    stack.sensitivity_plot()
    stack.background_distribution()
    stack.set_seed(2)
    stack.likelihood_scan(n_inj=50., inj_gamma=2.0)
    # stack.likelihood_scan(truth=True)
    stack.fitting_plot(gamma=[2.0, 2.5, 3.])
    for discovery in [True, False]:
        stack.sensitivity_efficiency_curve(discovery=discovery)
    for in_flux in [True, False]:
        fig, ax = plt.subplots()
        for discovery in [True, False]:
            label = 'Discovery potential' if discovery else 'Sensitivity'
            stack.sensitivity_vs_gamma(
                in_flux=in_flux, discovery=discovery,
                ax=ax, label=label)
        fl_label = 'flux' if in_flux else 'events'

        ax.legend(loc=2)
        for ftype in ['pdf', 'png']:
            plt.savefig(
                stack.savepath
                + f'sensitivity_vs_gamma_{stack.sample_str}'
                + f'_{fl_label}.{ftype}',
                bbox_inches='tight')
        plt.close()
    print("\t Finished stacking plots")


def make_GRECO_plots(gplots):
    gplots.declination_pdf()
    gplots.energy_pdf()
    plt.close('all')


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
    print(f"\t Catalog - only gamma ray novae")
    print(f"\t Figures will be saved to {args.output_path}")
    stack_plots = StackingPlots(
        args.stacking_time, 'gamma', 'gamma', index=[2.0, 2.5, 3.0],
        save=True, output=args.output_path)
    make_stacking_plots(stack_plots)

    print("Making figures for the stacked analysis with parameters:")
    print(f"\t Stacking time window: {args.stacking_time:.2e} s")
    print(f"\t Catalog - all novae")
    print(f"\t Figures will be saved to {args.output_path}")
    stack_plots = StackingPlots(
        args.stacking_time, 'all_novae', 'optical', index=[2.0, 2.5, 3.0],
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
