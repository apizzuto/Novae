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

background_jobs = pycondor.Job(
    'sensitivity_stacking_novae_greco',
    '/home/apizzuto/Nova/scripts/stacking/stacking_background.py',
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

low_mem_signal = pycondor.Job(
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

high_mem_signal = pycondor.Job(
    'sensitivity_stacking_novae_greco_high_mem',
    '/home/apizzuto/Nova/scripts/stacking/stacking_signal_trials.py',
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2,
    request_memory=12000,
    request_cpus=5,
    extra_lines=[
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman)

sample_schemes = [(' --all_nova', 'optical'), ('', 'gamma')][:1]
for sample, weighting in sample_schemes:
    if sample == ' --all_nova':
        delta_ts = np.append(
            np.logspace(-1.5, 0.5, 5)[:]*86400.,
            np.array([86400.*5.]))
        # delta_ts = [1. * 86400.]
    else:
        delta_ts = np.append(
            np.logspace(-1.5, 1., 6)[:]*86400.,
            np.array([86400.*5.]))
        # delta_ts = [1. * 86400.]
    for deltaT in delta_ts:
        for seed in range(10):
            if deltaT > 86400.:
                ntrials_bg = 2000
            else:
                ntrials_bg = 5000
            background_jobs.add_arg(
                f"--deltaT={deltaT} --ntrials={ntrials_bg} "
                + f"--seed={seed} --weighting={weighting}{sample}")

for sample, weighting in sample_schemes[:]:
    if sample == ' --all_nova':
        delta_ts = np.append(
            np.logspace(-1.5, 0.5, 5)[:]*86400.,
            np.array([86400.*5.]))
        # delta_ts = [1. * 86400.]
    else:
        delta_ts = np.append(
            np.logspace(-1.5, 1., 6)[:]*86400.,
            np.array([86400.*5.]))
        # delta_ts = [1. * 86400.]
    for deltaT in delta_ts:
        if deltaT > 86400.:
            ntrials_sig = 100
        else:
            ntrials_sig = 250
        for gamma in [2.0, 2.5, 3.0]:
            if deltaT > 86400:
                high_mem_signal.add_arg(
                    f'--deltaT={deltaT} --index={gamma}'
                    + f' --ntrials_sig={ntrials_sig}'
                    + f' --weighting={weighting}{sample}'
                    )
            else:
                low_mem_signal.add_arg(
                    f'--deltaT={deltaT} --index={gamma}'
                    + f' --ntrials_sig={ntrials_sig}'
                    + f' --weighting={weighting}{sample}'
                    )


background_jobs.add_child(low_mem_signal)
background_jobs.add_child(high_mem_signal)

dagman.build_submit()
