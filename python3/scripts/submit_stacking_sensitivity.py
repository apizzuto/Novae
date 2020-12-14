import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

job = pycondor.Job('sensitivity_stacking_novae_greco','/home/apizzuto/Nova/python3/scripts/stacking_sensitivity.py',
			error=error,
			output=output,
			log=log,
			submit=submit,
            getenv=True,
            universe='vanilla',
			verbose=2, 
			request_memory=12000,
            request_cpus=5,
			extra_lines= ['should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'Requirements =  (Machine != "node128.icecube.wisc.edu")']
			)
#for sigma in ['1', '10', '20', '30', '45', '90']:
for deltaT in [86400.*0.1, 86400.*0.2, 86400.*0.5, 86400.*1.0, 86400.*2.0, 86400.*5., 86400.*10., 86400.*14.]:
    for gamma in [2.0, 2.5, 3.0]:
        for cut in [0.0, 1.0]:
            job.add_arg('--deltaT={} --index={} --minLogE={}'.format(deltaT, gamma, cut))

dagman = pycondor.Dagman('Skylab_Novae_sensitivity_trials', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
