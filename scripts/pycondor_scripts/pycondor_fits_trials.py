import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

job = pycondor.Job('sensitivitytrials_novae_greco','/home/apizzuto/Nova/scripts/fit_n_signal.py',
			error=error,
			output=output,
			log=log,
			submit=submit,
            getenv=True,
            universe='vanilla',
			verbose=2, 
			request_memory=8000,
			extra_lines= ['should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'Requirements =  (Machine != "node128.icecube.wisc.edu")']
			)
for sigma in ['1', '10', '20', '30', '45', '90']:
    for index in range(15):
        for spec in ['SPL', 'EPL']:
    		job.add_arg('--sigma={} --index={} --spec={} --n=1000'.format(sigma, index, spec))

dagman = pycondor.Dagman('Skylab_Novae_fitting_trials', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
