import pycondor
from glob import glob
import numpy as np

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

dagman = pycondor.Dagman(
    'csky_Novae_stacking_jobs', submit=submit, verbose=2
    )

signal = pycondor.Job(
    'sensitivity_signal_novae_greco_low_mem',
    '/home/apizzuto/Nova/scripts/stacking/stacking_signal_trials.py',
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2,
    request_memory=8000,
    request_cpus=5,
    extra_lines=[
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman)

sample_schemes = [(' --all_nova', 'optical'), ('', 'gamma')]
for sample, weighting in sample_schemes[:]:
    if sample == ' --all_nova':
        delta_ts = [86400.]
    else:
        delta_ts = [86400.]
    for deltaT in delta_ts:
        if deltaT > 86400.:
            ntrials_sig = 100
        else:
            ntrials_sig = 500
        for gamma in np.linspace(2.0, 3.0, 11):
            if gamma in [2.0, 2.5, 3.0]:
                continue # we've already done these ones
            else:
                signal.add_arg(
                    f'--deltaT={deltaT} --index={gamma}'
                    + f' --ntrials_sig={ntrials_sig}'
                    + f' --weighting={weighting}{sample}'
                    )

dagman.build_submit()
