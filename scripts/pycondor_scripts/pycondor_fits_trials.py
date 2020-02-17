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
			request_memory=3000,
			extra_lines= ['should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'Requirements =  (Machine != "node128.icecube.wisc.edu")']
			)
for deltaT in np.logspace(1., 7., 7):
    for index in range(15):
        for spec in ['SPL', 'EPL']:
            #for fixed in [True, False]:
            #    add_str = ' --fixed' if fixed else ''
            for maxsig in [10., 20., 30., 40., 50., 60., 70.]:
                maxsig *= np.pi / 180.
                job.add_arg('--maxSigma={} --deltaT={} --index={} --spec={} --n=1000'.format(maxsig, deltaT, index, spec))
            #for lowE in [0.1, 0.5, 1., 1.5]:
            #    job.add_arg('--lowE={} --deltaT={} --index={} --spec={} --n=1000'.format(lowE, deltaT, index, spec))
            for scale in [1.1, 1.2]:
                job.add_arg('--scale={} --deltaT={} --index={} --spec={} --n=1000'.format(scale, deltaT, index, spec))
            for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                job.add_arg('--scale={} --deltaT={} --index={} --spec={} --n=1000'.format(scale, deltaT, index, spec))
            #for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            #    job.add_arg('--fixed --scale={} --pull --deltaT={} --index={} --spec={} --n=1000'.format(scale, deltaT, index, spec))

dagman = pycondor.Dagman('Skylab_Novae_fitting_trials', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
