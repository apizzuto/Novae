#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/mlarson/combo_r129072/build_v3
##!/usr/bin/env python
import os, sys, glob, icecube, healpy, scipy
import numpy as np
from icecube import recclasses, dataclasses, dataio, icetray, millipede, gulliver
from icecube.recclasses import *
from icecube.astro import I3GetEquatorialFromDirection
from scipy import misc

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-o", "--outputfile",
                  type      = "string",
                  default   = None,
                  help      = "Name of the output file(s), i.e. .root and .i3.gz names",)
(options, args) = parser.parse_args()

filelist = args
print args

# Set up the basic stuff
nside = 32
npix = healpy.nside2npix(nside)
hp_xyz = np.array(healpy.pix2vec(nside, np.arange(npix)))
hpdec, hpra = healpy.pix2ang(nside, np.arange(npix))
zenith, azimuth = np.pi-np.copy(hpdec), np.copy(hpra)
hpdec = np.pi/2 - hpdec

# Convert these into ellipse coordinates
points = np.zeros((len(zenith), 2), dtype=float)
iz, ia = 0, 1
def to_xyz(ra, lat):
    return np.array([np.cos(ra)*np.sin(lat),
                     np.sin(ra)*np.sin(lat),
                     np.cos(lat)])
points[:,iz] = 0.5*(np.cos(zenith))+0.5
points[:,ia] = azimuth / (2*np.pi)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def vonmises_llh(x, norm, kappa, ra, dec):
    if kappa < 0: return 1000
    xyz_center = to_xyz(ra, dec)
    llh = norm 
    llh += kappa * np.dot(xyz_center, x)
    return llh
    
def kent_llh(x, norm, kappa, ra_rad, dec_rad, beta, major_theta):
    if np.any([kappa<0, beta<0]): return 1000
    if beta > kappa: return 1000

    # We have the center and the orientation of the 
    # major axis. Convert this into a usable form for the
    # kent distribution
    xyz_center = to_xyz(ra_rad, dec_rad)
    initial_major = np.array([1, 0, 0])
    initial_major = np.cross(xyz_center, initial_major)
    initial_major /= np.sqrt(np.dot(initial_major, initial_major))
    
    major = np.dot(rotation_matrix(xyz_center, major_theta), initial_major)
    minor = np.cross(xyz_center, major)
    
    llh = vonmises_llh(x, norm, kappa, ra_rad, dec_rad)
    llh += beta * np.dot(major, x)**2 
    llh -= beta * np.dot(minor, x)**2
    
    return llh

def fb8_llh(x, norm, kappa, ra, dec, 
            beta, major_theta,
            nu_ra, nu_dec):
    if kappa < 0: return 1000
    if beta < 0: return 1000
    if beta > kappa: return 10000
    
    xyz_center = to_xyz(ra, dec)
    xyz_nu = to_xyz(nu_ra, nu_dec)
    
    initial_major = np.array([1, 0, 0])
    initial_major = np.cross(xyz_center, initial_major)
    initial_major /= np.sqrt(np.dot(initial_major, initial_major))
    
    major = np.dot(rotation_matrix(xyz_center, major_theta), initial_major)
    minor = np.cross(xyz_center, major)
    
    gamma = np.array([xyz_center, major, minor]).T
    
    #llh = norm
    #llh += kappa * np.dot(np.dot(xyz_nu, gamma), x)
    #llh += kappa * np.dot(xyz_nu, np.dot(gamma.T, x))
    #llh += beta * np.dot(major, x)**2
    #llh -= eta * beta * np.dot(minor, x)**2    
    #llh -= beta * np.dot(minor, x)**2    
    
    llh = kent_llh(x, norm, kappa, ra, dec, beta, major_theta)
    llh += kappa * (gamma.dot(xyz_nu) - xyz_center).dot(x)
    
    return llh


def get_map(ellipsoids, bestfit, bf_llh=0, ia=1, iz=0):
    def vonmises_bf(x, norm, kappa):
        return vonmises_llh(x, norm, kappa, bf_long, bf_lat)
    def kent_bf(x, norm, kappa, beta, major_theta):
        return kent_llh(x, norm, kappa, bf_long, bf_lat, beta, major_theta)
    def fb8_bf(x, norm, kappa, beta, major_theta,
               nu_ra, nu_dec):
        return fb8_llh(x, norm, kappa, bf_long, bf_lat, 
                       beta, major_theta, nu_ra, nu_dec)

    # Plot the best fit as well?
    bf_lat = np.pi - bestfit.dir.zenith
    bf_long = bestfit.dir.azimuth
    
    kent_sum = np.ones(npix,dtype=float) * np.inf
    llh = np.ones(npix, dtype=float) * np.inf
    points = np.zeros((len(zenith), 2), dtype=float)
    iz, ia = 0, 1
    points[:,iz] = 0.5*(np.cos(zenith))+0.5
    points[:,ia] = azimuth / (2*np.pi)

    for i in range(len(ellipsoids)):
        ell = ellipsoids[i]
        center = ell.center
        dllh = (ell.nllh - bf_llh)
    
        indicies = np.arange(npix)[dllh < llh]
        for j in indicies:
            if ell.contains(points[j]):
                if llh[j] == np.inf: 
                    llh[j] = dllh
                else:
                    llh[j] = np.minimum(llh[j], dllh)
        
        # make sure the ellipse isn't completely contained in the healpix bin
        center[iz] = np.pi-np.arccos(2*center[iz]-1)
        center[ia] *= 2*np.pi
        b = healpy.ang2pix(nside, center[iz], center[ia])

    isinf = llh == np.inf
    llh[isinf] = np.max(llh[~isinf])
    # Find the edges for each ellipse
    # find all pixels that have at least one neighbor lower in llh
    contours = np.ones(npix)*np.nan
    contours[llh == np.max(llh)] = -np.max(llh)

    for px in np.arange(npix):
        theta, phi = healpy.pix2ang(nside, px)
        neighbors = healpy.pixelfunc.get_all_neighbours(nside, theta, phi)

        current = llh[px]
        neighbors = llh[neighbors]
        if np.any(neighbors > current):
            contours[px] = -llh[px]

    # Put the values into a simpler format for fitting
    i = np.isfinite(contours)
    x = healpy.pix2vec(nside, np.arange(npix)[i])
    y = contours[i]
    sigma = np.sqrt(-2*y) + 1e-6

    xyz = healpy.pix2vec(nside, np.arange(npix))
            
    # First step: get the von-mises fit
    f = vonmises_bf;
    param_names = ['norm', 'kappa']
    bounds = [[-np.inf, 0], 
              [np.inf, np.inf]]

    popt, pcov = curve_fit(f, x, y, sigma=sigma, maxfev=100000,
                           absolute_sigma=True, bounds = bounds)
    
    # Try fitting the kent distribution to the points?
    # Get the points and llh values
    f = kent_bf; 
    param_names += ['beta', 'axis_rotation',]
    bounds[0].extend([0, 0])
    bounds[1].extend([np.inf, 2*np.pi])

    p0 = np.copy(popt).tolist() 
    p0 += [0, 0]
    popt, pcov = curve_fit(f, x, y, sigma=sigma, maxfev=100000,
                           absolute_sigma=True,  p0=p0, bounds = bounds)

    '''
    f = fb8_bf; 
    param_names += ['nu_ra', 'nu_dec']
    bounds[0].extend([0, 0])
    bounds[1].extend([2*np.pi, np.pi])
    p0 = np.copy(popt).tolist() 
    p0 += [1, 2]

    popt, pcov = curve_fit(f, x, y, sigma=sigma, maxfev=100000,
                            absolute_sigma=True, p0=p0, bounds = bounds)
    '''
    
    #healpy.mollview(-llh)
    #healpy.mollview(contours)
    
    result = f(xyz, *popt)
    #healpy.mollview(result,)#min=result.max()-10)
    #healpy.projscatter(bf_lat, bf_long, lonlat=False, color='r', marker='*')

    return result, popt, param_names

# In[ ]:


# Build the structures to hold the information I want
run = []
event = []
subevent = []
time = []

reco_energy = []
reco_zenith = []
reco_azimuth = []
reco_ra = []
reco_dec = []

monopod_zenith = []
monopod_azimuth = []

pid_dllh = []
pid_llh_pl = []
pid_llh_mo = []
pid_length = []
llh_map = []

#Alex added these ones
cascade_energy = []
nstring = []
nchannel = []

kent_parameters = []
kent_names = []

# MC Only
is_mc = False
oneweight = []
ptype = []
iscc = []
true_energy = []
true_ra = []
true_dec = []
dllh_truth = []

for filenum, infile in enumerate(filelist):
    i3file = dataio.I3File(infile, 'r')        
    print filenum,'of',len(filelist), ':', infile
    eventnum = 0
    while i3file.more():
        try: frame = i3file.pop_physics()
        except: 
            print("Something broke")
            break
        
        eventnum += 1
        
        # Event header information
        header = frame['I3EventHeader']
        run.append(header.run_id)
        event.append(header.event_id)
        subevent.append(header.sub_event_id)
        
        t = header.start_time
        time.append(t.mod_julian_day_double)

        if 'I3MCTree' in frame.keys():
            is_mc = True

            # Get the information
            truth = dataclasses.get_most_energetic_neutrino(frame['I3MCTree'])
            mcwd = frame['I3MCWeightDict']

            # Scale by the number of events
            ow = mcwd['OneWeight']/mcwd['NEvents']
            if 'genie' in infile:
                if truth.pdg_encoding > 0: 
                    ow/=0.7
                else: 
                    ow/=0.3
            else:
                ow /= 0.5            

            # Intentionally not going to scale by
            # nfiles here, since I want to do it when
            # I merge files. That'll make my life easier
            # when I add more files to these.
            oneweight.append(ow)

	    ptype.append(truth.pdg_encoding)
	    iscc.append(mcwd['InteractionType']==1)
            
            x = I3GetEquatorialFromDirection(truth.dir, t)
            true_ra.append(x.ra)
            true_dec.append(x.dec)
            true_energy.append(truth.energy)
        
        # Reconstruction values    
        bestfit = frame['Pegleg_Fit_NestleTrack']
        hdcasc = frame['Pegleg_Fit_NestleHDCasc']
        
        cascade_energy.append(hdcasc.energy)
        reco_energy.append(bestfit.energy + hdcasc.energy)
        reco_zenith.append(bestfit.dir.zenith)
        reco_azimuth.append(bestfit.dir.azimuth)
	pid_length.append(bestfit.length)

        i3dst = frame['I3DST']
        nchannel.append(i3dst.ndom)
        nstring.append(i3dst.n_string)

        x = I3GetEquatorialFromDirection(bestfit.dir, t)
        reco_ra.append(x.ra)
        reco_dec.append(x.dec)
        
        bf_llh = frame['Pegleg_Fit_NestleFitParams'].logl
        mo_llh = frame['Monopod_bestFitParams'].logl
        pid_dllh.append(bf_llh-mo_llh)
	pid_llh_pl.append(bf_llh)
	pid_llh_mo.append(mo_llh)

	monopod = frame['Monopod_best']
	monopod_zenith.append(monopod.dir.azimuth)
	monopod_azimuth.append(monopod.dir.zenith)

        ellipses = frame['Pegleg_Fit_Nestle_NestleMinimizer']

        # Convert the truth into something usable with the ellipsoids
        ellipses = ellipses.prune(50)
        names = list(ellipses.axis_names)

        iz, ia = names.index('Zenith'), names.index('Azimuth')
        remove = np.ones(len(names), dtype=bool)
        remove[iz] = False
        remove[ia] = False
        ellipses = ellipses.profile(np.arange(len(names))[remove])
        names = list(ellipses.axis_names)
        iz, ia = names.index('Zenith'), names.index('Azimuth')
        ellipses = [ellipses[i] for i in range(len(ellipses))]
        
        # Make a map for this event?
        kent_sum, kent_params, kent_names = get_map(ellipses, bf_llh, ia, iz)
        llh_map.append(kent_sum)
        kent_parameters.append(kent_params)

        # Where is the truth in this map?
        if is_mc:
            truth_bin = healpy.ang2pix(nside, np.pi-truth.dir.zenith, truth.dir.azimuth)
            dllh = np.max(kent_sum) - kent_sum[truth_bin]
            dllh_truth.append(dllh)
	    
	    # What's the true angular distance?
	    #truth = to_xyz(truth.dir.azimuth, truth.dir.zenith)
	    #reco = to_xyz(bestfit.dir.azimuth, bestfit.dir.zenith)
	    #dist = healpy.rotator.angdist(truth, reco)
	    #print np.rad2deg(dist), dllh
	    
        del ellipses        


run      = np.array(run, dtype=int)
event    = np.array(event, dtype=int)
subevent = np.array(subevent, dtype=int)
time     = np.array(time, dtype=float)

reco_energy  = np.array(reco_energy, dtype=float)
reco_zenith  = np.array(reco_zenith, dtype=float)
reco_azimuth = np.array(reco_azimuth, dtype=float)
reco_ra  = np.array(reco_ra, dtype=float)
reco_dec = np.array(reco_dec, dtype=float)

monopod_zenith = np.array(monopod_zenith, dtype=float)
monopod_azimuth = np.array(monopod_azimuth, dtype=float)

llh_map = np.array(llh_map, dtype=float)
pid_dllh = np.array(pid_dllh, dtype=float)
pid_llh_pl = np.array(pid_llh_pl, dtype=float)
pid_llh_mo = np.array(pid_llh_mo, dtype=float)
pid_length = np.array(pid_length, dtype=float)
monopod_energy = np.array(cascade_energy, dtype=float)
n_string = np.array(nstring, dtype=int)
n_channel = np.array(nchannel, dtype=int)

kent_parameters = np.array(kent_parameters, dtype=float)

if is_mc:
    oneweight    = np.array(oneweight, dtype=float)
    true_energy  = np.array(true_energy, dtype=float)
    true_ra      = np.array(true_ra, dtype=float)
    true_dec     = np.array(true_dec, dtype=float)
    dllh_truth   = np.array(dllh_truth, dtype=float)


    ptype = np.array(ptype, dtype=int)
    iscc = np.array(iscc, dtype=bool)

    datatypes = np.dtype( [ 
        ('run', np.int64), ('event', np.int64),
        ('subevent', np.int64), ('time', np.float64),
        ('ra', np.float64), ('dec', np.float64),
        ('azi', np.float64), ('zen', np.float64),
        ('angErr', np.float64), ('logE', np.float64),
	('monopod_azi', np.float64), ('monopod_zen', np.float64),
	('pidDeltaLLH', np.float64),
	('pidPeglegLLH', np.float64), ('pidMonopodLLH', np.float64),
	('pidLength', np.float64),
        ('trueE', np.float64), ('trueRa', np.float64),
        ('trueDec', np.float64), ('ow', np.float64),
	('ptype', np.float64), ('iscc', np.bool),
        ('trueDeltaLLH', np.float64),
        ('monopod_energy', np.float64),
        ('nstring', np.int64),
        ('nchannel', np.int64)] )
    data = np.array([run, event, subevent, time,
                    reco_ra, reco_dec, reco_azimuth, reco_zenith,
                    np.zeros_like(reco_zenith), 
                    np.log10(reco_energy), 
		    monopod_azimuth, monopod_zenith, 
		    pid_dllh, 
		    pid_llh_pl, pid_llh_mo,
		    pid_length,
		    true_energy,
                    true_ra, true_dec, oneweight,
		    ptype, iscc,
                    dllh_truth]).T
else:
    datatypes = np.dtype( [ 
        ('run', np.int64), ('event', np.int64),
        ('subevent', np.int64), ('time', np.float64),
        ('ra', np.float64), ('dec', np.float64),
        ('azi', np.float64), ('zen', np.float64),
        ('angErr', np.float64), ('logE', np.float64),	
	('monopod_azi', np.float64), ('monopod_zen', np.float64),
	('pidDeltaLLH', np.float64),
	('pidPeglegLLH', np.float64), ('pidMonopodLLH', np.float64),
	('pidLength', np.float64),
 ] )
    
    data = np.array([run, event, subevent, time,
                    reco_ra, reco_dec, reco_azimuth, reco_zenith,
                    np.zeros_like(reco_zenith), 
                    np.log10(reco_energy), 
		    monopod_azimuth, monopod_zenith, 
		    pid_dllh,
		    pid_llh_pl, pid_llh_mo,
		    pid_length,]).T
    

tuple_output = [tuple(data[i]) for i in range(data.shape[0])]
data = np.rec.array(tuple_output, dtype=datatypes)

np.save(options.outputfile, data)
np.save(options.outputfile+"_maps", llh_map)

