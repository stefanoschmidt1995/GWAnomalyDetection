import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
from memspectrum import MESA
import scipy.signal as sig

import numpy as np
import matplotlib.pyplot as plt

import mlgw.GW_generator as gen
g = gen.GW_generator()

import emd #nice package for emd: pip install emd

#############
#	This is a small snippet to illustrate how Empirical Mode Decomposition (EMB) can be used as pass band filter for the low frequencies.
#	It seems reasonably robust and it allows to cut lower frequencies by "whitening" them (see the PSD in the plot below)
#	It decomposes the signal in different components, each with a signal composed by increasing frequencies
#	The highest frequencies component is two orders of magnitude lower than the actual strain and has a "flat" PSD
#	Anomalies can be detected there!!
#	By looking at the PSD of the high frequency mode, it seems that EMB acts both as a pass band filter and as a whithener: it seems more robust than standard approach to whithening (now windows or segments length choices)
#	An injected signal can be "easily" recovered in amplitude in the highest frequency mode
#	The high frequency mode can be used as the input for the LSTM and/or memspectrum based pipeline: seems promising :D
#############

#TODO: add the pipeline at the end of this, using as input data imf[:,0]


	#loading strain data
datafile = "H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz"
srate = 4096.#*4.
data = np.loadtxt(datafile)
times = np.linspace(0,len(data)/srate, len(data))

	#setting up injection
t_merger = 5.
WF = g.get_WF([36,29,-0.1,0.2, 41.0*5., 0.3, 2.435], times - t_merger)[0]

data = data+WF
data = data*1e18

	#For downsampling... It is a good idea to do so: all the high frequency are gathered in a single mode
if True:
	data = sig.decimate(data, 2)
	times = sig.decimate(times, 2)
	srate = srate/2.
	dt = 1./srate

	#doing Empirical Mode Decomposition
	#It will output K times series, each of which keeps increasingly higher frequency modes (imf[:,0] has the fastest changing component)
#help(emd.sift.sift) #for the help page of the function
imf = emd.sift.sift(data, sift_thresh = 1e-8)

	#removing borders of the data: edge effects can be quite heavy in the high frequecies
T = 20
offset = 5000
data = data[offset:int(T*srate)+offset]
times = times[offset:int(T*srate)+offset]
imf = imf[offset:int(T*srate)+offset,:]

	#computing PSD of the high frequency residuals
M = MESA()
M.solve(imf[:,0], method = 'Standard')
freq = np.linspace(1./times[-1], 0.5*srate,1000) #vector for evaluating the spectrum
spec_imf = M.spectrum(1/srate,freq)
M.solve(data, method = 'Standard')
spec_data = M.spectrum(1/srate,freq)

	#plotting everything
	#In the order 0 mode, you can see the chirp of the injected signal!! 
fig = plt.figure( figsize=(8,4) )
ax = fig.gca()
plt.title("PSD of the EMD mode with highest frequency vs PSD of data")
ax.loglog(freq, spec_data, label= "data")
ax.loglog(freq, spec_imf, label= "1st EMD")
plt.legend()

fig, ax = plt.subplots(imf.shape[1]+1,1, sharex = True, figsize=(16,8) )
fig.suptitle("Data + the {} components of Empirical Mode Decomposition".format(imf.shape[1]))

for i, a_ in enumerate(ax):
	if i ==0:
		a_.set_title("Data")
		res = np.sum(imf, axis =1)
		a_.plot(times, data, c = 'k')
		a_.plot(times, res, c = 'r')
		continue
	a_.set_title("EMB #{}".format(i-1))
	a_.plot(times, imf[:,i-1])

fig.tight_layout()
plt.show()










