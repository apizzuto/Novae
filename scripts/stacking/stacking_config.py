import numpy as np
from glob import glob
import csky as cy

def get_stacking_objs(minLogE=None, allflavor=True):
    """
    Format the GRECO data into csky dataset objects

    Args:
        minLogE (None or float): Cut out events with reconstructed energy
            less than minLogE
        allflavor (bool): Include all flavors for signal. If false, only
            use numu simulation
    """
    greco_base = '/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.5/'

    data_fs = sorted(glob(greco_base + 'IC86_20*data_with_angErr.npy'))
    exp = [np.load(data) for data in data_fs]
    exp = np.hstack(exp)
    if allflavor:
        mcfiles = glob(greco_base + 'IC86_2012.nu*_with_angErr.npy')
        mc = np.load(mcfiles[0])
        for flav in mcfiles[1:]:
            mc = np.concatenate((mc, np.load(flav)))
    else:
        mcfile = glob(greco_base + 'IC86_2012.numu_merged_with_angErr.npy')[0]
        mc = np.load(mcfile)
    grls = sorted(glob(greco_base + 'GRL/IC86_20*data.npy'))
    grl = [np.load(g) for g in grls]
    grl = np.hstack(grl)

    if minLogE is not None:
        exp_msk = exp['logE'] > minLogE
        exp = exp[exp_msk]
        mc_msk = mc['logE'] > minLogE
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

    greco = cy.selections.CustomDataSpecs.CustomDataSpec(
        exp, mc, np.sum(grl['livetime']), 
        np.linspace(-1., 1., 31),
        np.linspace(0., 4., 31), 
        grl=grl, key='GRECOv2.5', cascades=True
        )

    conf = {'extended': True,
            'space': "ps",
            'time': "transient",
            'sig': 'transient',
        }

    return greco, conf