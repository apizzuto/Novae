import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np

error = 'condor/error'
output = 'condor/output'
log = 'condor/log'
submit = 'condor/submit'

job = pycondor.Job('GRECO_Simulation_Processing','/home/apizzuto/Nova/scripts/convert_l7mc_hdf5.py',
			error=error,
			output=output,
			log=log,
			submit=submit,
            getenv = True,
            universe='vanilla',
			verbose=2, 
			request_memory=4000
			)


base_path = '/data/ana/LE/NBI_nutau_appearance/level7_5July2017/'

##First deal with GENIE
for fl, ind in zip(['e', 'mu', 'tau'], ['12', '14', '16']):
    i3List = glob(base_path + 'genie/{}640/'.format(ind) + 'Level7*.bz2')
    outfile = '/data/user/apizzuto/Nova/simulation/genie_nu{}'.format(fl)
    #job.add_arg('{} {}'.format(outfile, base_path + 'genie/{}640/'.format(ind) + 'Level7*bz2'))
    job.add_arg('{} {}'.format(outfile, ' '.join(i3List)))


##Then do Nugen
for fl in ['e', 'mu', 'tau']:
    for en in ['LE', 'ME']:
        i3List = glob(base_path + 'nugen/nu{}_{}/'.format(fl, en) + 'Level7*.bz2')
        outfile = '/data/user/apizzuto/Nova/simulation/nugen_nu{}_{}'.format(fl, en)
        #job.add_arg('{} {}'.format(outfile, base_path + 'nugen/nu{}_{}/'.format(fl, en) + 'Level7*.bz2'))
        job.add_arg('{} {}'.format(outfile, ' '.join(i3List)))

dagman = pycondor.Dagman('Nova_GRECO_processing', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
