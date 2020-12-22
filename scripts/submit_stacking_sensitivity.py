import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/novae/condor/error'
output = '/scratch/apizzuto/novae/condor/output'
log = '/scratch/apizzuto/novae/condor/log'
submit = '/scratch/apizzuto/novae/condor/submit'

low_mem_job = pycondor.Job('sensitivity_stacking_novae_greco','/home/apizzuto/Nova/scripts/stacking_sensitivity.py',
			error=error,
			output=output,
			log=log,
			submit=submit,
            getenv=True,
            universe='vanilla',
			verbose=2, 
			request_memory=8000,
            request_cpus=5,
			extra_lines= ['should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'Requirements =  (Machine != "node128.icecube.wisc.edu")']
			)

high_mem_job = pycondor.Job('sensitivity_stacking_novae_greco','/home/apizzuto/Nova/scripts/stacking_sensitivity.py',
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
for allflavor_str in [' --allflavor', '']:
	for deltaT in np.append(np.logspace(-3., 1., 9)[:]*86400., np.array([86400.*5.])):
		if deltaT > 86400.:
			ntrials_sig = 100
			ntrials_bg = 1000
		else:
			ntrials_sig = 250
			ntrials_bg = 5000
		for gamma in [2.0, 2.5, 3.0]:
			for cut in [0.0, 0.5, 1.0]:
				if deltaT > 86400:
					high_mem_job.add_arg(f'--deltaT={deltaT} --index={gamma} --minLogE={cut} --ntrials_sig={ntrials_sig} --ntrials_bg={ntrials_bg}{allflavor_str}')
				else:
					low_mem_job.add_arg(f'--deltaT={deltaT} --index={gamma} --minLogE={cut} --ntrials_sig={ntrials_sig} --ntrials_bg={ntrials_bg}{allflavor_str}')

dagman = pycondor.Dagman('Skylab_Novae_sensitivity_trials', submit=submit, verbose=2)
dagman.add_job(low_mem_job)
dagman.add_job(high_mem_job)
dagman.build_submit()
