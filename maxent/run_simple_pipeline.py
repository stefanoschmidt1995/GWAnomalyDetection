import numpy as np
from pipeline import *
from pipeline_helper import *

import mlgw.GW_generator as gen

###########################

#TODO: the weak part is a slow emd decomposition: data should be divided in batches (~1000s) and analysed separately. This should be pretty natural...

#datafile = "data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz"
datafile = "data/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz"
srate = 4096.#*4.
#data = np.loadtxt(datafile)
data = np.squeeze(pd.read_csv(datafile, skiprows =3))
#np.savetxt("H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz", data[:int(100*srate)]) #for saving a short version of data
print("Loaded {}s of data sampled @ {}Hz".format(len(data)/srate,srate))
times = np.linspace(0,len(data)/srate, len(data))

	#setting up injection
t_merger = 45.
g = gen.GW_generator(0)
WF = g.get_WF([36,29,-0.1,0.2, 41.0*10., 0.3, 2.435], times - t_merger)[0]

	#mandatory to scale data!!!!
WF *= 1e18
data *= 1e18
#data = data+WF

if True: #downsampling
	srate, data, times, WF = downsample_data(4, srate, data, times, WF)

imf, data, times, WF = do_emd(data, emd_type = 'PyEMD', outfile = 'data/H-H1_GWOSC_4KHZ_R1-1126259447-32.emd.dat', trim_borders = 30, times = times, WF = WF)
#imf, data, times, WF = load_emd('data/H-H1_GWOSC_4KHZ_R1-1126257415-100.emd.dat', data, times, WF)

AnomalyDetection_pipeline(imf[:,0], srate, T_train = 10., N_step = 250, outfile = 'data/LL-1126259447-32.pkl', plot = True, injection = None)

#plot_PSD_imf(imf, data, srate, WF = None, title = None, folder = None)
#plot_imf(imf, data, title = None, folder = None, times = times)

detect_outliers('data/LL-1126259447-32.pkl')

plt.show()



