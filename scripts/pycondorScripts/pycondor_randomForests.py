import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np

error = 'condor/error'
output = 'condor/output'
log = 'condor/log'
submit = 'condor/submit'

job = pycondor.Job('Nova_Random_Forest_Training','/home/apizzuto/Nova/scripts/angularError_randomForest.py',
			error=error,
			output=output,
			log=log,
			submit=submit,
			verbose=2, 
			request_memory=8000
			)

for a in [True, False]:
    for s in [True, False]:
        for log in [True, False]:
            job.add_arg('--a={} --s={} --log={}'.format(a,s,log))

dagman = pycondor.Dagman('Nova_Random_Forest', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
