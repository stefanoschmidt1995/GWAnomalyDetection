import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
from memspectrum import MESA
import scipy.signal as sig

import numpy as np
import matplotlib.pyplot as plt
from pipeline import *
import scipy.signal as sig
import pandas as pd

import mlgw.GW_generator as gen
g = gen.GW_generator()

import emd #nice package for emd: pip install emd

#############
#	This is a small snippet to illustrate how Empirical Mode Decomposition (EMD) can be used to dump lower frequencies, acting effectively as a whithener (see the PSD in the plot below)
#	It decomposes the signal in different components, each with a signal composed by increasing frequencies
#	The highest frequencies component is two orders of magnitude lower than the actual strain and has a "flat" PSD
#	Anomalies can be detected there!!
#	By looking at the PSD of the high frequency mode, it seems that EMB acts both as a pass band filter and as a whithener: it seems more robust than standard approach to whithening (now windows or segments length choices)
#	An injected signal can be "easily" recovered in amplitude in the highest frequency mode
#	The high frequency mode can be used as the input for the LSTM and/or memspectrum based pipeline: seems promising :D
#############

#TODO: add the pipeline at the end of this, using as input data imf[:,0]


	#loading strain data
#datafile = "data/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz"
datafile = "data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz"
srate = 4096.#*4.
#data = np.loadtxt(datafile)
data = np.squeeze(pd.read_csv(datafile, skiprows =3))
#np.savetxt("H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz", data[:int(100*srate)]) #for saving a short version of data
print("Loaded {}s of data sampled @ {}Hz".format(len(data)/srate,srate))
times = np.linspace(0,len(data)/srate, len(data))

	#setting up injection
t_merger = 25.
WF = g.get_WF([36,29,-0.1,0.2, 41.0*10., 0.3, 2.435], times - t_merger)[0]
#WF = np.max(WF)*np.exp(-np.square(times - t_merger)) #for gaussian injection

	#scaling things
data = data*1e18
WF = WF*1e18

#data = data+WF

	#For bandpassing: useful?
if False:
	(B,A) = sig.butter(5,[20/(.5*srate), 1024/(.5*srate)], btype='band')#, fs = srate)
	data_pass = sig.filtfilt(B, A, data)
	data = data_pass

	#For downsampling... It is a good idea to do so: all the high frequency are gathered in a single mode
if True:
	downsampling_factor = 4
	data = sig.decimate(data, downsampling_factor)
	times = sig.decimate(times, downsampling_factor)
	WF = sig.decimate(WF, downsampling_factor)
	srate = srate/float(downsampling_factor)
	dt = 1./srate

	#doing Empirical Mode Decomposition
	#It will output K times series, each of which keeps increasingly higher frequency modes (imf[:,0] has the fastest changing component)
#help(emd.sift.sift) #for the help page of the function
imf = emd.sift.sift(data, sift_thresh = 1e-8)

	#removing borders of the data: edge effects can be quite heavy in the high frequecies
T = 30
offset = 500
data = data[offset:int(T*srate)+offset]
times = times[offset:int(T*srate)+offset]
WF = WF[offset:int(T*srate)+offset]
imf = imf[offset:int(T*srate)+offset,:]

print("Data are {}s long".format(len(data)/srate))

	#computing PSD of the high frequency residuals
	#computing the PSD of the signal
M = MESA()
M.solve(imf[:,0], method = 'Standard', optimisation_method = 'CAT')
freq = np.linspace(1./times[-1], 0.5*srate,1000) #vector for evaluating the spectrum
spec_imf = M.spectrum(1/srate,freq)
M.solve(data, method = 'Standard')
spec_data = M.spectrum(1/srate,freq)
M.solve(WF, method = "standard")
spec_WF = M.spectrum(1/srate,freq)

	#running pipeline
data_pipeline = imf[:,0]
AnomalyDetection_pipeline(data_pipeline, srate, T_train = 10., N_step = 20000, outfile = None, plot = True, injection = WF)

	#plotting everything
	#In the order 0 mode, you can see the chirp of the injected signal!! 
fig = plt.figure( figsize=(8,4) )
ax = fig.gca()
plt.title("PSD of the EMD mode with highest frequency vs PSD of data")
ax.loglog(freq, spec_data, label= "data")
ax.loglog(freq, spec_imf, label= "1st EMD")
ax.loglog(freq, spec_WF, label= "WF")
plt.legend()
fig.savefig("PSD.pdf")

fig, ax = plt.subplots(imf.shape[1]+1,1, sharex = True, figsize=(16,8) )
fig.suptitle("Data + the {} components of Empirical Mode Decomposition".format(imf.shape[1]))

for i, a_ in enumerate(ax):
	if i < 20:
		pass
		#a_.plot(times, WF, c = 'k')
	if i ==0:
		#a_.set_title("Data")
		res = np.sum(imf, axis =1)
		a_.plot(times, data, c = 'r')
		#a_.plot(times, res, c = 'r')
	else:
		#a_.set_title("EMB #{}".format(i-1))
		a_.plot(times, imf[:,i-1])

fig.tight_layout()
fig.savefig("EMD_modes.pdf")

	#plotting correlation
if False:
	plt.figure()
	plt.title("Correlation between signal and data")
	corr  = sig.correlate(data, WF[int(10*srate):int(16*srate)],mode = 'same')
	plt.plot(times, corr)
	plt.plot(times, data)
plt.show()










