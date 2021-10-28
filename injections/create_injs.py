"""
Simple script that loads an xml injection file, generates all the WFs in there and adds them to real noise. This is useful for training a ML model
The injections can be easily created with lvc_rates_injections command provided in: https://git.ligo.org/RatesAndPopulations/lvc-rates-and-pop/-/blob/master/bin/lvc_rates_injections
"""

__author__ = 'Stefano Schmidt'

import numpy as np
import matplotlib.pyplot as plt
import pandas

import lal
import lalsimulation

from tqdm import tqdm

from ligo.lw import utils
from ligo.lw import ligolw
from ligo.lw import table
from ligo.lw import lsctables
from ligo.lw.utils import process as ligolw_process

def get_WF_and_snr(inj, PSD, sample_rate, instrument = "H1", plot_dir = None):
	"Given an injection row, it computes the WF and the SNR"
	#https://git.ligo.org/lscsoft/gstlal/-/blob/precession_hm-0.2/gstlal-inspiral/bin/gstlal_inspiral_injection_snr
	#PSD should be a lal PSD obj
	assert instrument in ["H1", "L1", "V1"]

	injtime = inj.time_geocent


	sample_rate = 16384.0

	approximant = lalsimulation.GetApproximantFromString(str(inj.waveform))
	f_min = inj.f_lower

	h_plus, h_cross = lalsimulation.SimInspiralTD(
				m1 = inj.mass1*lal.MSUN_SI,
				m2 = inj.mass2*lal.MSUN_SI,
				S1x = inj.spin1x,
				S1y = inj.spin1y,
				S1z = inj.spin1z,
				S2x = inj.spin2x,
				S2y = inj.spin2y,
				S2z = inj.spin2z,
				distance = inj.distance*1e6*lal.PC_SI,
				inclination = inj.inclination,
				phiRef = inj.coa_phase,
				longAscNodes = 0.0,
				eccentricity = 0.0,
				meanPerAno = 0.0,
				deltaT = 1.0 / sample_rate,
				f_min = f_min,
				f_ref = 0.0,
				LALparams = None,
				approximant = approximant
		)

	h_plus.epoch += injtime
	h_cross.epoch += injtime

		# Compute strain in the chosen detector.
	h = lalsimulation.SimDetectorStrainREAL8TimeSeries(h_plus, h_cross, inj.longitude, inj.latitude, inj.polarization, lalsimulation.DetectorPrefixToLALDetector(instrument))
		#Compute the SNR
	if PSD is not None:
		snr = lalsimulation.MeasureSNR(h, PSD, options.flow, options.fmax)
	else: snr = 0.

	if isinstance(plot_dir, str):
		plt.figure()
		plt.title("(m1, m2, s1 (x,y,z), s2 (x,y,z), d_L) =\n {0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8:.2f} ".format(inj.mass1, inj.mass2, inj.spin1x,
				inj.spin1y, inj.spin1z, inj.spin2x, inj.spin2y, inj.spin2z, inj.distance))
		plt.plot(np.linspace(0, len(h_plus.data.data)/sample_rate, len(h_plus.data.data)), h_plus.data.data)
		plt.savefig(plot_dir+'/inj_{}.png'.format(injtime))
		#plt.show()
		plt.close('all')

	return (h.data.data, snr)


def create_inj_WFs(filename, PSD, sample_rate, instrument, N_inj = None, plot_dir = None):
	#https://git.ligo.org/RatesAndPopulations/lvc-rates-and-pop/-/blob/master/bin/lvc_rates_injections
	
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass
	lsctables.use_in(LIGOLWContentHandler)

	xmldoc = utils.load_filename(filename, verbose = True, contenthandler = LIGOLWContentHandler)
	sim_inspiral_table = lsctables.SimInspiralTable.get_table(xmldoc)
		
	WFs = [] #a list of tuples
	if N_inj is None: N_inj = len(sim_inspiral_table)
			
	for row in tqdm(sim_inspiral_table):
		WFs.append(get_WF_and_snr(row, PSD, sample_rate, instrument, plot_dir))
		if len(WFs)>=N_inj: break
			
	return WFs

def get_PSD_from_xml(PSD_filename, df, f_min = 15., f_max = 1024.):
	"Gets the PSD from lal from an ASD file"
	PSD = lal.CreateREAL8FrequencySeries(
		'PSD',
		lal.LIGOTimeGPS(0),
		0.0,
		df, lal.SecondUnit,
		int(round(f_max /df)) + 1)

	#FIXME: how shall I read the PSD??
	lalsimulation.SimNoisePSDFromFile(PSD, f_min, filename)
	#see documentation here:
		#https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_noise_p_s_d__c.html#ga67d250556e4e8647c50d379364eb7911
	
	# SimNoisePSDFromFile expects ASD in the file, but this one
	# contains the PSD, so take the square root
	PSD.data.data = PSD.data.data ** 0.5

	return PSD


#############################

if __name__=='__main__':
	
	sample_rate = 4096.
	filename = 'test_inj_file.xml'
	ASD_filename =  'H1L1V1-REFERENCE_PSD-1241800311-1732.xml.gz' #this is a PSD! WHERE THE FUCK I FIND A ASD FILE???
	noise_filename ='H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt.gz'
	instrument = 'H1'
	f_min = 15.
	
		#should do a lal PSD eventually
	PSD = None #get_PSD_from_xml(ASD_filename, df = 1/sample_rate, f_min = f_min, f_max = 1024.)
	
	inj_WFs = create_inj_WFs(filename, PSD, sample_rate, instrument, N_inj = 5, plot_dir = './out_pics')

		#loading noise
		#no check on the sampling rate is performed
	raw_data = np.squeeze(pandas.read_csv(noise_filename, skiprows = 3).to_numpy())

	id_data = 0
	data_list = []
	
	for inj, snr in inj_WFs:
		len_wf = len(inj)
		data_with_WF = raw_data[id_data:id_data+len_wf] + inj
		data_list.append( data_with_WF )
		id_data +=len_wf +100 #updating id_data (adding 100 to separate injections a little bit)

	#Congrats!! Now you have a data_list with a lot of data segments with noise + signal
	#You may also want to pad them

	#TODO: 1) Create injection file: lalapps_inspinj or https://git.ligo.org/RatesAndPopulations/lvc-rates-and-pop/-/blob/master/bin/lvc_rates_injections
	#			See run_inj_cmd.sh for a test
	#TODO: 2) Deal properly with the PSD
	#TODO: 3) Do whatever you need with the injections




































	
