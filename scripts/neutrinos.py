import sys, numpy, glob, os, pickle
from optparse import OptionParser
import icecube
from icecube import dataio, dataclasses, icetray, gulliver, millipede, genie_icetray, NuFlux, recclasses
from icecube.dataclasses import I3Particle, I3Position
from icecube.icetray import I3Units, I3Frame, I3Int
from icecube.dataio import I3File

from numpy.polynomial.polynomial import polyfit

class Neutrinos:
    ##############################################################
    # Prepare the weighting code for corsika or muongun
    ##############################################################
    def __init__(self, results, filenames, ptype, nfiles):
        self.lowe_flux_service = NuFlux.makeFlux("IPhonda2014_spl_solmin")
        #self.highe_flux_service = NuFlux.makeFlux("IPhonda2006_sno_solmin")
        self.highe_flux_service = NuFlux.makeFlux("honda2006")

        # try adding in the knee stuff?
        self.highe_flux_service.knee_reweighting_model = 'gaisserH3a_elbert'

        # get the nfiles
        if nfiles < 0:
            self.num_files = len(filenames)
        else: self.num_files = nfiles

        return

    ##############################################################
    # Prepare the lists we'll need
    ##############################################################
    def prepare(self, results, ptype):
        results['energy']       = []
        results['zenith']       = []
        results['azimuth']      = []
        results['ptype']        = []
        results['x']            = []
        results['y']            = []
        results['z']            = []
        results['nprimaries']   = []

        results['interaction']  = []
        results['oneweight']    = []
        results['weight']       = []
        results['weight_e']     = []
        results['weight_mu']    = []

        results['depo_energy']  = []
        
        results["scattering"]   = []
        results['ma_qe']        = []
        results['ma_res']       = []
                
        results["GENIE_x"]      = []
        results["GENIE_y"]      = []
        results["GENIE_Q2"]     = []

        results["MaNCRES"]      = []
        results["MaNCEL"]       = []

        results['cross_section'] = []

        return results
        

    ##############################################################
    # Additional cuts to apply for this file
    ##############################################################
    def cuts(self, frame, ptype, filename):
        return True

    ##############################################################
    # Get the specialized keys for neutrino events
    ##############################################################
    def getvars(self, frame, ptype, filename):
        results = self.prepare({}, ptype)

        # Handle the weighting first
        mc_weights = frame['I3MCWeightDict']
        true_neutrino = dataclasses.get_most_energetic_neutrino(frame['I3MCTree'])#frame['MCNeutrino']
        
        true_energy = mc_weights['PrimaryNeutrinoEnergy']
        true_zenith = true_neutrino.dir.zenith
        true_azimuth = true_neutrino.dir.azimuth

        if true_neutrino.energy < 10000: flux_service = self.lowe_flux_service
        else: flux_service = self.highe_flux_service

        if 'genie' in ptype.lower() or 'coincident' in ptype.lower() or 'bulk' in ptype.lower():
            nu_nubar_genratio = 0.7 # fraction of neutrinos. 0.3 = fraction of antinus
        else:
            nu_nubar_genratio = 0.5

        if true_neutrino.pdg_encoding > 0:
            flux_nue = flux_service.getFlux(I3Particle.ParticleType.NuE, 
                                            true_energy,
                                            true_azimuth, 
                                            numpy.cos(true_zenith))
            flux_numu = flux_service.getFlux(I3Particle.ParticleType.NuMu, 
                                             true_energy, 
                                             true_azimuth, 
                                             numpy.cos(true_zenith))
        else:
            nu_nubar_genratio = 1-nu_nubar_genratio
            flux_nue = flux_service.getFlux(I3Particle.ParticleType.NuEBar, 
                                            true_energy, 
                                            true_azimuth, 
                                            numpy.cos(true_zenith))
            flux_numu = flux_service.getFlux(I3Particle.ParticleType.NuMuBar, 
                                             true_energy, 
                                             true_azimuth, 
                                             numpy.cos(true_zenith))

        one_weight = mc_weights['OneWeight']
        n_events = mc_weights['NEvents']
        norm = (1.0 / (n_events * self.num_files * nu_nubar_genratio))
        
        if "InteractionCrosssectionCGS" in mc_weights.keys():
            results['cross_section'].append( mc_weights['InteractionCrosssectionCGS'] / I3Units.cm**2 )
        else:
            results['cross_section'].append( mc_weights['Crosssection'] * (1e11))

        results["oneweight"].append(one_weight * norm)
        results["interaction"].append(int(mc_weights['InteractionType']))
        results["weight"].append( norm * one_weight *(flux_nue * (true_neutrino.pdg_encoding in [-12, 12]) + 
                                                      flux_numu * (true_neutrino.pdg_encoding in [-14, 14])) )

        results["weight_e"].append(norm * one_weight * flux_nue)
        results["weight_mu"].append(norm * one_weight * flux_numu)

        # MC truth information
        p = dataclasses.get_most_energetic_neutrino(frame["I3MCTree"])
        daughters = frame["I3MCTree"].get_daughters(p)
        daughter = daughters[0]
        results["x"].append(daughter.pos.x)
        results["y"].append(daughter.pos.y)
        results["z"].append(daughter.pos.z)
        results["zenith"].append(true_neutrino.dir.zenith)
        results["energy"].append(true_neutrino.energy)
        results["azimuth"].append(true_neutrino.dir.azimuth)
        results["ptype"].append(true_neutrino.pdg_encoding)

        # Get the number of primaries
        primaries = frame['I3MCTree'].get_primaries()
        results["nprimaries"].append(len(primaries))

        # Find the deposited energy
        deposited = 0
            
        def getEnergy(p):
            daughters = frame['I3MCTree'].get_daughters(p)            
            energies = 0
            if len(daughters) >0:
                energies = 0
                for d in daughters:
                    energies += getEnergy(d)
                return energies
            else:
                if p.is_neutrino:
                    return 0
                if p.energy == None:
                    return 0
                return p.energy

        results['depo_energy'].append(getEnergy(p))

        # Get the genie stuff and the axial mass information
        if 'genie' in ptype.lower() or 'coincident' in ptype.lower():
            res = frame["I3GENIEResultDict"]["res"]
            qe = frame["I3GENIEResultDict"]["qel"]
            dis = frame["I3GENIEResultDict"]["dis"]
            coh = frame["I3GENIEResultDict"]["coh"]

            results["scattering"].append( dis + 2*res + 3*qe + 4*coh )

            print frame["I3GENIEResultDict"].keys()
            sys.exit()

            results["GENIE_x"].append( frame["I3GENIEResultDict"]["x"] ) 
            results["GENIE_y"].append( frame["I3GENIEResultDict"]["y"] ) 
            results["GENIE_Q2"].append( frame["I3GENIEResultDict"]["Q2"] )                 

            gw = frame['I3MCWeightDict']['GENIEWeight']
            if not 'rw_MaCCQE' in frame["I3GENIEResultDict"].keys():
                y_values = numpy.ones(4) * gw
                results["ma_qe"].append(self.fitMa(y_values, gw))
                results["ma_res"].append(self.fitMa(y_values, gw))
                results["MaNCRES"].append(self.fitMa(y_values, gw))
                results["MaNCEL"].append(self.fitMa(y_values, gw))
            else:
                results["ma_qe"].append(self.fitMa(frame['I3GENIEResultDict']['rw_MaCCQE'],gw))
                results["ma_res"].append(self.fitMa(frame['I3GENIEResultDict']['rw_MaCCRES'],gw))
                results["MaNCRES"].append(self.fitMa(frame['I3GENIEResultDict']['rw_MaNCRES'],gw))
                results["MaNCEL"].append(self.fitMa(frame['I3GENIEResultDict']['rw_MaNCEL'],gw))

            
        else:

            # More work needed to get the x/y/Q2 for nugen, since its not nicely saved...
            lepton = None
            for daughter in daughters:
                if numpy.abs( daughter.pdg_encoding ) in [11,12,13,14,15,16]:
                    lepton = daughter
                    break

            if lepton == None:
                results["GENIE_x"].append( 1.0 )
                results["GENIE_y"].append( 1.0 )
                results["GENIE_Q2"].append( 1.0 )
            else:
                y = 1 - lepton.energy/true_neutrino.energy
                dotprod = (true_neutrino.dir.x*lepton.dir.x) + (true_neutrino.dir.y*lepton.dir.y) + (true_neutrino.dir.z*lepton.dir.z)
                Q2 = -(0.105)**2 + 2*true_neutrino.energy*lepton.energy*(1-dotprod)
                
                x = Q2/(2*1*true_neutrino.energy*y)
                results["GENIE_x"].append( x )
                results["GENIE_y"].append( y )
                results["GENIE_Q2"].append( Q2 )
              
            results["scattering"].append( 1 )
            results["ma_qe"].append(numpy.array([0,0]))
            results["ma_res"].append(numpy.array([0,0]))
            results["MaNCRES"].append(numpy.array([0,0]))
            results["MaNCEL"].append(numpy.array([0,0]))
            
        return results

    ##############################################################
    # a wrapper to get the axial mass junk from GENIE files
    ##############################################################
    def fitMa(self, in_yvalues, genie_weight):
        rw_xvalues  = numpy.array([-2,-1,0,1,2])
        if sum(in_yvalues) == 4.0: return numpy.array([0,0])
        in_yvalues = numpy.array(in_yvalues)
        yvalues = numpy.concatenate((in_yvalues[:2]/genie_weight, [1.], in_yvalues[2:]/genie_weight))
        fitcoeff = polyfit(rw_xvalues, yvalues, deg = 2)[::-1]
        return fitcoeff[:2]



