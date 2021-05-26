import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

dagman = pycondor.Dagman(
    'Skylab_Novae_stacking_jobs', submit=submit, verbose=2
    )

low_mem_signal = pycondor.Job(
    'sensitivity_signal_novae_greco_low_mem',
    '/home/apizzuto/Nova/scripts/stacking_signal_trials.py',
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2, 
    request_memory=8000,
    request_cpus=5,
    extra_lines= ['should_transfer_files = YES', 
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman
	)

high_mem_signal = pycondor.Job(
    'sensitivity_stacking_novae_greco_high_mem',
    '/home/apizzuto/Nova/scripts/stacking_signal_trials.py',
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2, 
    request_memory=12000,
    request_cpus=5,
    extra_lines= ['should_transfer_files = YES', 
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman
	)

for allflavor_str in [' --allflavor', '']:
    for deltaT in np.append(np.logspace(-3., 1., 9)[:]*86400., np.array([86400.*5.])):
        # for deltaT in [86400.*10.]:
        if deltaT > 86400.*5.:
            ntrials_sig = 50
        elif deltaT > 86400.:
            ntrials_sig = 100
        else:
            ntrials_sig = 250
        for gamma in [2.0, 2.5, 3.0]:
            # for cut in [0.0, 0.5, 1.0]:
            if deltaT > 86400:
                high_mem_signal.add_arg(
                f'--deltaT={deltaT} --index={gamma}' \
                    + f' --ntrials_sig={ntrials_sig}' \
                    + f'{allflavor_str}'
                    )
            else:
                low_mem_signal.add_arg(
                    f'--deltaT={deltaT} --index={gamma}' \
                    + f' --ntrials_sig={ntrials_sig}' \
                    + f'{allflavor_str}'
                    ) 

dagman.build_submit()