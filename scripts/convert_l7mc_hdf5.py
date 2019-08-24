#!/usr/bin/env python

import icecube
from icecube import dataclasses, dataio, icetray, hdfwriter, millipede
from I3Tray import *
from icecube.tableio import I3TableWriter

import os
import sys

files=[]

myFileList = sys.argv[2:]
#myFileList.remove('/data/user/sfahey/Nova/mc/GRECO_nugen_numu_LE_Level7/Level7_NuGen_NuMu_low_energy.00000429.0000.i3.bz2')

tray=I3Tray()

tray.AddModule('I3Reader', 'reader', FilenameList = sys.argv[2:])

tray.AddSegment(icecube.hdfwriter.I3HDFWriter, 'writer', 
                Output=sys.argv[1]+'.h5',
                Keys=['MCNeutrino', 'MPEFitMuEX', 'MPEFitFitParams', 'Pegleg_Fit_MN_tol10Track', 'Pegleg_Fit_MN_tol10FitParams', 'Monopod_bestFitParams', 'Monopod_best', 'I3MCWeightDict', 'Pegleg_Fit_MN_tol10HDCasc', 'InIcePulses', 'SRTTWOfflinePulsesDC'],
                SubEventStreams=['InIceSplit']
                )

tray.AddModule('TrashCan', 'YesWeCan')
tray.Execute()
tray.Finish()
