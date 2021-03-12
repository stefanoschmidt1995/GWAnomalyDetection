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
from PyEMD import EMD #This package is way better than the other!!

from pipeline_helper import *
#####################################

	#loading strain data
datafile = "data/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz"
#datafile = "data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz"
srate = 4096.#*4.
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

data = data+WF

	#For bandpassing: useful?
if False:
	data = band_pass(20, 1024, srate, data)

	#For downsampling... It is a good idea to do so: all the high frequency are gathered in a single mode
if True:
	srate, data, times, WF = downsample_data(4, srate, data, times, WF)

	#doing Empirical Mode Decomposition
imf = do_emd(data, 'emd')
imf_bis = do_emd(data, 'PyEMD')

print("Data are {}s long".format(len(data)/srate))

plot_PSD_imf(imf,data, srate, None, 'std', 'plots')
plot_PSD_imf(imf_bis,data, srate, None,'bis', 'plots')

plot_imf(imf,data, 'std', 'plots')
plot_imf(imf_bis, data,'bis', 'plots')

plt.show()








