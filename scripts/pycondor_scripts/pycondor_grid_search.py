import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

job = pycondor.Job('Novae_random_forests','/home/apizzuto/Nova/scripts/angularError_randomForest.py',
			error=error,
			output=output,
			log=log,
			submit=submit,
            getenv=True,
            universe='vanilla',
			verbose=2, 
			request_memory=4000,
            request_cpus=5,
			extra_lines= ['should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'Requirements =  (Machine != "node128.icecube.wisc.edu")']
			)
for sep in [True, False]:
    for logS in [True, False]:
        for boot in [True]:
            for minsamp in [2, 10, 100, 1000, 10000, 100000][:1]:
                mystr = '--boot'
                if sep:
                    mystr += ' --s'
                if logS:
                    mystr += ' --log'
                job.add_arg(mystr + ' --minsamp={}'.format(minsamp))

dagman = pycondor.Dagman('Novae_Grid_Search', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
