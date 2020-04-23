import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

job = pycondor.Job('backgroundtrials_novae_greco','/home/apizzuto/Nova/scripts/bg_trials.py',
			error=error,
			output=output,
			log=log,
			submit=submit,
            getenv=True,
            universe='vanilla',
			verbose=2, 
			request_memory=3000,
			extra_lines= ['should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'Requirements =  (Machine != "node128.icecube.wisc.edu")']
			)
for deltaT in np.logspace(-1, 7., 17):
    for index in range(16):
        for spec in ['SPL', 'EPL']:
    		job.add_arg('--deltaT={} --index={} --spec={} --n=100000'.format(deltaT, index, spec))

dagman = pycondor.Dagman('Skylab_Novae_background_trials', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
