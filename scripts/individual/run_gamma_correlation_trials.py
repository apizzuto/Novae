import pycondor
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

job = pycondor.Job(
    'sensitivity_stacking_novae_greco',
    '/home/apizzuto/Nova/scripts/individual/gamma_nova_sensitivity.py',
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2,
    request_memory=6000,
    extra_lines=[
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT',
        'Requirements =  (Machine != "node128.icecube.wisc.edu")']
    )

master_df = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
gamma_df = master_df[master_df['gamma']]
gamma_df = gamma_df.reset_index()

#for nova_num in range(len(gamma_df)):
#    for deltaT in np.append(np.logspace(-1.5, 1., 6)[:]*86400.,
#                            np.array([86400.*5.])):
#        for gamma in [2.0, 2.5, 3.0]:
#            job.add_arg(
#                '--nova_num={} --deltaT={} --index={}'.format(
#                    nova_num, deltaT, gamma))

for nova_num in range(len(gamma_df)):
    for gamma in [2.0, 2.5, 3.0]:
        job.add_arg(
            f'--nova_num={nova_num} --index={gamma} '
            + '--full_gamma_time')

dagman = pycondor.Dagman(
    'Skylab_Novae_sensitivity_trials',
    submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
