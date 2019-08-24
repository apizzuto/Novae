#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v2/icetray-start
#METAPROJECT /data/user/mlarson/combo/build/
##!/usr/bin/env python
import sys, numpy, glob, os
import cPickle as pickle
from optparse import OptionParser

import icecube
from icecube import dataio, dataclasses, icetray, gulliver, millipede, recclasses, fill_ratio
from icecube.dataclasses import I3Particle
from icecube.dataio import I3File
from icecube.icetray import I3Frame, I3Int, OMKey

from flaring_doms import pyFlaringDOMFilter

from optparse import OptionParser
"""Make the argument parser"""
parser = OptionParser()
parser.add_option("--input", action="store",
                  type="string", default=None, dest="input")
parser.add_option("--type", action="store",
                  type="string", default=None, dest="ptype") 
parser.add_option("--output", action="store",
                  type="string", default=None, dest="output")
parser.add_option("--testFiles", action="store_true",
                  default=False, dest="test")
parser.add_option("--skipNch", action="store_true",
                  default=False, dest="skipNch")
parser.add_option("--nfiles", type=int, action="store",
                  default=-1, dest="nfiles")
parser.add_option("--verbose", action='store_true',
                  default=False, dest='verbose')
(options, args) = parser.parse_args()

if "*" in options.input:
    options.input = glob.glob(os.path.abspath(options.input))
if type(options.input) != list:
    options.input = [options.input,]
options.input.sort()
num_files = len(options.input)
print "Found %i files" % num_files
print "Given nfiles from options:", options.nfiles

# If desired, test the integrity of the input files 
if options.test:
    import subprocess
    new_input = []
    for f in options.input:
        print "Testing: ", os.path.basename(f)
        try:
            subprocess.check_call( ['bunzip2', '-t', f], shell=False )
            new_input.append(f)
        except: 
            print '\t... Bad file. Skipping'
            pass
    print "Found %i good files of %i" % (len(new_input), len(options.input))
    options.input = new_input

# Variables that ALL events get
results = {}
results["was_split"] = []

results["nchannel"] = []
results["nstring"] = []
results["charge"] = []

results['charge_fraction'] = []
results["charge_rms_normalized"] = []

results["reco_zenith"] = []
results["reco_azimuth"] = []
results["reco_energy"] = []
results["reco_track_energy"] = []
results["reco_cascade_energy"] = []
results["tracklength"] = []
results["reco_x"] = []
results["reco_y"] = []
results["reco_z"] = []
results["reco_rho"] = []

results["reco_x_stop"] = []
results["reco_y_stop"] = []
results["reco_z_stop"] = []
results["reco_rho_stop"] = []

results["rLLH_Pegleg"] = []
results["rLLH_Monopod"] = []
results["deltaLLH"] = []

results['CalibrationErrataLength'] = []
results['CE_contains_baddoms'] = []

results["event_length"] = []
results["fr_from_nch"] = []

results['GeV_per_channel'] = []

results["nearest_om"] = []
results["nearest_string"] = []
results["dist_nearest_om"] = []
results["dist_nearest_string"] = []

results['t_rms'] = []
results['z_firsthit'] = []

results['StartingContainmentBool'] = []

results['ContainsFlaringDoms'] = []
results['nFlaringDoms'] = []

# Prepare the additional objects for specialized work
specialized = None
if 'genie' in options.ptype.lower() or 'coincident' in options.ptype.lower() or 'bulk' in options.ptype.lower(): 
    import neutrinos
    specialized = neutrinos.Neutrinos(results, options.input, options.ptype, options.nfiles)
    results = specialized.prepare(results, options.ptype)
elif 'nugen' in options.ptype.lower(): 
    import neutrinos
    specialized = neutrinos.Neutrinos(results, options.input, options.ptype, options.nfiles)
    results = specialized.prepare(results, options.ptype)
elif 'muongun' in options.ptype.lower(): 
    import muons
    specialized = muons.Muons(results, options.input, options.nfiles)
    results = specialized.prepare(results, options.ptype)
elif 'corsika' in options.ptype.lower(): 
    import muons
    specialized = muons.Muons(results, options.input, options.nfiles)
    results = specialized.prepare(results, options.ptype)
elif 'noise' in options.ptype.lower(): 
    import noise
    specialized = noise.Noise(results, options.input, options.nfiles)
    results = specialized.prepare(results, options.ptype)
elif any([x in options.ptype.lower() for x in ['data', 'exp', 'pass2']]):
    import data
    specialized = data.Data(results, options.input, options.nfiles)
    results = specialized.prepare(results, options.ptype)
else:
    print "-"*20
    print "Unknown data type. "
    print "You specified ptype=", options.ptype
    print "Set the --ptype to genie, nugen, muongun, corsika, noise, data, or pass2"
    print "-"*20
    sys.exit(10)

##########################################################################
# Grab the geometry information
##########################################################################
geo = numpy.loadtxt("/data/user/mlarson/combo/scripts/extract_pickle_files/scripts/rereco/Icecube_geometry.20110414.complete.txt",)
geo = geo[geo[:,1] < 60]
string, om, xcoord, ycoord, zcoord = geo.T

# Set up the flaring dom code with the geometry
fmod = pyFlaringDOMFilter.pyFlaringDOMFilter(pulsesName = 'SRTTWOfflinePulsesDC',
                                             geo = geo)


######################################################
# Start looping
######################################################
for filename in options.input:
    #print filename
    infile = I3File(filename, 'r')
    nsplit = 0
    nframes = 0
    npassed = 0

    while infile.more():
        frame = infile.pop_physics()

        if frame == None: continue
        nframes += 1
        #print nframes

            
        eid = frame["I3EventHeader"].event_id
        seid = frame["I3EventHeader"].sub_event_id

        k = frame.keys()
        if not 'I3SuperDST' in k: 
            print '\tEvent is somehow missing the SuperDST pulses...?! Skipping.'
            continue

        if not 'SplitInIcePulses' in frame:
            print '\tNo split in ice'
            continue

        #----------------------------
        # Cuts before filling!
        #----------------------------
        # Apply specialized cuts
        if not specialized.cuts(frame, options.ptype, filename):
            continue
        
        #print 'event:', (eid, seid)
        
        try:us_hits = frame['InIcePulses'].apply(frame)
        except:us_hits = frame['InIcePulses']
        s_hits = frame['SplitInIcePulses'].apply(frame)
        split = False
        for dom in us_hits.keys():
            if dom.om > 60: continue
            if not dom in s_hits.keys(): 
                split = True
                break
            for hit in us_hits[dom]:
                if not hit in s_hits[dom]: 
                    split = True
                    break

        # Find and apply an nchannel cut
        if not frame.Has("SRTTWOfflinePulsesDC"): 
            if options.verbose: print '\tNo srttwofflinepulsesdc'
            continue

        hitmap = frame["SRTTWOfflinePulsesDC"].apply(frame)
        nch = 0
        charge = 0
        charge_sqr = 0
        max_charge = 0
        times = []
	strings = []
        for dom in hitmap.keys():
            nch += len(hitmap[dom]) > 0
            current_charge = sum([hit.charge for hit in hitmap[dom]])
            charge += current_charge
            charge_sqr += current_charge**2

            times.append(hitmap[dom][0].time)
	    strings.append(dom.string)

            if current_charge > max_charge: max_charge = current_charge

	nstring = len(set(strings))

        # Standard L6 cuts
        #keys = frame.keys()
        #keys.sort()
        #print keys
        if not frame["TauL6_bool"].value:
            #print frame["TauL6_FillRatio"].fillratio_from_mean
            if options.verbose: print '\tFailed tau L6'
            continue
        if nch < 8 and not options.skipNch: 
            if options.verbose: print '\tfailed nch'
            continue

        # Observables for the analyses
        if not frame.Has('Pegleg_Fit_MN_tol10HDCasc'): 
            if options.verbose: print '\tno pegleg'
            continue
        reco_cascade = frame['Pegleg_Fit_MN_tol10HDCasc']
        reco_track = frame['Pegleg_Fit_MN_tol10Track']

        #----------------------------
        # Filling after cuts
        #----------------------------
        # CalibrationErrata
        errata_length = 0
        errata_baddoms = False
        if frame.Has("CalibrationErrata"): 
            bad_oms = [OMKey(83, 33),
                       OMKey(83, 58),
                       ]
            
            # Kill any event that has an 83/33 in the errata?
            if any([dom in bad_oms for dom in frame["CalibrationErrata"].keys()]):
                errata_baddoms = True

            errata_length = len(frame["CalibrationErrata"].keys())

        results['CE_contains_baddoms'].append(errata_baddoms)
        results['CalibrationErrataLength'].append( errata_length )

        # Check for flaring doms in the event using the script from Jim Braun?
        fmod.Physics(frame)
        if 'LIDErrata' in frame:
            results['ContainsFlaringDoms'].append(True)
            results['nFlaringDoms'].append(len(frame['LIDErrata']))
        else:
            results['ContainsFlaringDoms'].append(False)
            results['nFlaringDoms'].append(0)


        # Find the nearest dom/string info
        dx = reco_cascade.pos.x - xcoord
        dy = reco_cascade.pos.y - ycoord
        dz = reco_cascade.pos.z - zcoord
        rho = numpy.sqrt(dx**2 + dy**2)
        r = numpy.sqrt(dx**2 + dy**2 + dz**2)
        
        rho_i = numpy.argmin(rho)
        r_i = numpy.argmin(r)

        results["was_split"].append(split)

        results["nearest_om"].append(om[r_i])
        results["nearest_string"].append(string[rho_i])
        results["dist_nearest_om"].append(r[r_i])
        results["dist_nearest_string"].append(rho[rho_i])
        
        results["reco_energy"].append(reco_cascade.energy + reco_track.energy)
        results["reco_zenith"].append(reco_track.dir.zenith)
        results["tracklength"].append(reco_track.length)

        # PID for Martin
        results["rLLH_Pegleg"].append(frame['Pegleg_Fit_MN_tol10FitParams'].rlogl)
        results["rLLH_Monopod"].append(frame['Monopod_bestFitParams'].rlogl)
        results["deltaLLH"].append(frame['Pegleg_Fit_MN_tol10FitParams'].logl - frame['Monopod_bestFitParams'].logl)

        # Other reco quantities
        results["reco_track_energy"].append(reco_track.energy)
        results["reco_cascade_energy"].append(reco_cascade.energy)

        results["reco_x"].append(reco_cascade.pos.x)
        results["reco_y"].append(reco_cascade.pos.y)
        results["reco_z"].append(reco_cascade.pos.z)
        results["reco_azimuth"].append(reco_track.dir.azimuth)

        results["reco_rho"].append( numpy.sqrt( ( reco_cascade.pos.x-46.29 ) ** 2 +
                                                ( reco_cascade.pos.y+34.88) ** 2 ) )

        # Containment cut
        z = reco_cascade.pos.z
        rho = numpy.sqrt( ( reco_cascade.pos.x-46.29 ) ** 2 + ( reco_cascade.pos.y+34.88) ** 2 )
        containment = z < -200
        containment *= rho < 100
        containment *= numpy.logical_or((z+100)/(rho-50) < -2.5, rho < 50)
        
        results['StartingContainmentBool'].append(containment)


        # Stopping variables
        results['reco_x_stop'].append( reco_cascade.pos.x - 
                                       reco_track.length * numpy.sin(reco_track.dir.zenith) * numpy.cos(reco_track.dir.azimuth) )
        results['reco_y_stop'].append( reco_cascade.pos.y - 
                                       reco_track.length * numpy.sin(reco_track.dir.zenith) * numpy.sin(reco_track.dir.azimuth) )
        results['reco_z_stop'].append( reco_cascade.pos.z - 
                                       reco_track.length * numpy.cos(reco_track.dir.zenith) )
                                       
        results['reco_rho_stop'].append( numpy.sqrt( (results['reco_x_stop'][-1]-46.29)**2 + 
                                                     (results['reco_y_stop'][-1]+34.88)**2 ) )
        

        # Base-level quantities that may be useful
        results["charge"].append( charge )
        results["nchannel"].append( nch )
	results["nstring"].append( nstring )
        results['GeV_per_channel'].append( (reco_track.energy + reco_cascade.energy)/nch )

        results['charge_fraction'].append( max_charge / charge )        
        results["charge_rms_normalized"].append( numpy.sqrt(charge_sqr)/charge )

        header = frame['I3EventHeader']
        results["event_length"].append( header.end_time - header.start_time )
            
        results["fr_from_nch"].append(frame['TauL6_FillRatio'].fillratio_from_nch)

        results['t_rms'].append(numpy.std(times))
        results['z_firsthit'].append(frame['VertexGuess_TauL4'].pos.z)

        spec_results = specialized.getvars(frame, options.ptype, filename)

        for key in spec_results:
            results[key].extend( spec_results[key] )

        npassed += 1

        sys.stdout.flush()

    infile.close()

    print filename, npassed, 'events'#, nframes

# Numpyify it all
for key in results:
    results[key] = numpy.array(results[key])

print "Found %i events in %i files." % (len(results['nchannel']), num_files)

pickle.dump(results, open(options.output + '.pckl', 'w'), protocol=-1)

cut = numpy.array(results['StartingContainmentBool'], dtype=bool)

if 'genie' in options.ptype.lower() or 'coincident' in options.ptype.lower():
    print numpy.sum(results['weight'][cut]), "Hz"
    isCC = (results['interaction'] == 1) & cut
    print 'CC:', numpy.sum(results['weight_e'][isCC]), numpy.sum(results['weight_mu'][isCC])
    print 'NC:', numpy.sum(results['weight_e'][~isCC]), numpy.sum(results['weight_mu'][~isCC])
elif 'nugen' in options.ptype.lower():
    print numpy.sum(results['weight']), "Hz"
    isCC = (results['interaction'] == 1) & cut
    print 'CC:', numpy.sum(results['weight_e'][isCC]), numpy.sum(results['weight_mu'][isCC])
    print 'NC:', numpy.sum(results['weight_e'][~isCC]), numpy.sum(results['weight_mu'][~isCC])
elif 'muongnun' in options.ptype.lower():
    print numpy.sum(results['weight'][cut]), "Hz"
elif 'corsika' in options.ptype.lower():
    print numpy.sum(results['weight'][cut]), "Hz"
elif 'noise' in options.ptype.lower():
    print numpy.sum(results['weight'][cut]), "Hz"
