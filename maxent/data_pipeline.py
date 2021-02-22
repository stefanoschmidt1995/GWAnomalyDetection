#TODO: analyse better noise properties and find a statistics for discerning noise from signal
#TODO: perhaps is better to compare every bin with its previous...

import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
from memspectrum import MESA

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import wavfile


import mlgw.GW_generator as gen
g = gen.GW_generator()

def data_LL_gauss(data, prediction):
	mu = np.mean(prediction, axis = 0)
	std_dev = np.std(prediction, axis = 0)
	LL = -0.5*np.square(np.divide(data - mu, std_dev)) #(D,)
	#LL = LL - 0.5*np.log(2*np.pi) - np.log(std_dev) #(D,)

	return LL 

def data_LL(data, prediction):
	"Using cumulative distribution"
	cum = np.sum(predictions-data>=0, axis = 0) #(D,)

	return cum.astype(np.float32) / float(predictions.shape[0])

	#loading data and solving mesa with train data

datafile = "H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz"
srate = 4096.#*4.
data = np.loadtxt(datafile)

#srate, data = wavfile.read("test.wav")
#data = data[:35*srate]

#data = np.loadtxt('test.txt').T
#srate = 1./(data[0,1]-data[0,0])
#data = data[1,:]


print("Data length: ", len(data)/srate, srate)
#data = np.loadtxt(datafile)[::4]
dt = 1./srate
times = np.linspace(0, len(data)*dt, len(data))

	#adding WF
t_start = 0.
t_merger = 26#1126259462.4-1126259447
WF = np.zeros(times.shape)
WF = g.get_WF([36,29,-0.1,0.2, 41.0*2, 0.3, 2.435], times - t_merger)[0]
#WF = 1e-20*np.cos(2*np.pi*700*times) #* np.exp(-np.square(times - t_merger))
#WF = 1e-4*np.exp(-np.square(times - t_merger))

#adding WF to the data
data =  data + WF #DEBUG: removed WF
#plt.plot(times,data)
#plt.show()

#bandpassing the data
(B,A) = sig.butter(5,[20/(.5*srate), 1024/(.5*srate)], btype='band')#, fs = srate)
#(B,A) = sig.butter(4, 1500/(.5*srate), btype='lowpass')
#data_pass = sig.lfilter(B, A, data)
data_pass = sig.filtfilt(B, A, data)
#data_pass = data + WF #DEBUG: removed WF

if False: #plot bandpassing
	plt.figure()
	plt.title("Data vs filtered data")

	plt.plot(times,data)
	plt.plot(times,data_pass)
	#plt.axvline(1126259462.4-1126259447, c = 'r')
	plt.axvline(t_merger, c = 'r')


	fig_PSDseries = plt.figure(2)
	ax_PSDseries  = fig_PSDseries.add_subplot(111)
	ax_PSDseries.set_ylabel("PSD")
	ax_PSDseries.set_yscale('log')
	M = MESA()
	M.solve(data_pass, early_stop = True, method = 'Standard')

	freq = np.linspace(1./times[-1], 0.5*srate,1000)
	spec = M.spectrum(1/srate,freq)
	ax_PSDseries.loglog(freq, np.abs(spec), c = 'b', label= "bandpass")
	M.solve(data, method = 'Standard')
	freq = np.linspace(1./times[-1], 0.5*srate,1000)
	spec = M.spectrum(1/srate,freq)
	ax_PSDseries.loglog(freq, spec, c = 'r', label= "standard data")
	plt.legend()
	plt.show()

data = data_pass
#data = data + WF

#downsampling
if True:
	plt.figure(100)
	plt.plot(times, data)
	data = sig.decimate(data, 2)
	print(times)
	times = sig.decimate(times, 2)
	WF = sig.decimate(WF, 2)
	print(times)
	srate = srate/2.
	dt = 1./srate
	plt.plot(times, data)
	#plt.show()

T  = 10 #T used for training

id_start = int(T*srate) #start id for forecasting
train_data = data[100:id_start]

N = data.shape[0]
M = MESA()
start = time.perf_counter()
P, ak, _ = M.solve(train_data, method = "Standard", optimisation_method = "FPE", m = int(2*N/(2*np.log(N))))
print(ak, P)

	#starting "pipeline"
time_step = 1500/srate # 1 seconds of data
id_step = int(time_step* srate)
Np = 100 #number of predictions

fig_timeseries = plt.figure(0)
ax_timeseries  = fig_timeseries.add_subplot(111)
ax_timeseries.set_ylabel("Strain")

fig_LLseries = plt.figure(1)
ax_LLseries  = fig_LLseries.add_subplot(211)
ax_WF  = fig_LLseries.add_subplot(212)
ax_LLseries.set_ylabel("LL")
ax_LLseries.set_yscale('log')
ax_WF.set_ylabel("Strain")

fig_PSDseries = plt.figure(2)
ax_PSDseries  = fig_PSDseries.add_subplot(111)
ax_PSDseries.set_ylabel("PSD")
ax_PSDseries.set_yscale('log')
freq = np.linspace(1./T, 0.5*srate,1000)
spec = M.spectrum(1/srate,freq)
ax_PSDseries.loglog(freq, spec)

ax_timeseries.plot(times[id_start-id_step:id_start], train_data[-id_step:], linewidth=1., color='r', zorder = 3)

	#PSD init
f_min = 1./time_step
f_grid = np.logspace(np.log10(f_min),np.log10(0.5*srate), 1000)
PSD_baseline = M.spectrum(1./srate, f_grid)


#ax_timeseries.plot(times, data, linewidth=1.5, color='g', zorder = 3, label = "Data")
ax_timeseries.axvline(15.615353510692678)
#ax_WF.axvline(15.615353510692678)
ax_WF.axvline(t_merger)
ax_timeseries.axvline(t_merger)

T_max = len(data)*dt

for i, id_ in enumerate(range(id_start-id_step, int(T_max * srate-id_step) , id_step)): #problem here with missing gaps (or something weird related...)
	sys.stderr.write("\rAnalysing batch {} of {}: t in [{},{}]".format(i+1, 
			int((int(T_max * srate-id_step)- (id_start-id_step))/id_step +0.5),
			times[id_],times[id_+id_step]))

		#forecasting predictions
	times_batch = times[id_:id_+id_step]
	WF_batch = WF[id_:id_+id_step]
	data_batch = data[id_:id_+id_step] #data to try to predict
	
	forecast_basis = data[id_-M.get_p():id_] #batch of data that are the basis for forecasting
	#predictions = np.zeros(( Np,int(id_step)+1))
	predictions = M.forecast(forecast_basis, int(id_step), Np) #(Np, D)
	
	#LL = np.zeros((int(id_step)+1,))
	LL = data_LL_gauss(data_batch, predictions)
	
	l, m, h = np.percentile(predictions, [5,50,95],axis=0)
	
	
	ax_LLseries.plot(times_batch, np.cumsum(LL)+1e5)
	
	#ax_WF.plot(times_batch,(LL))
	#ax_LLseries.axvline(times[id_],linestyle='dashed',color='blue')
	
	#ax_timeseries.axvline(times[id_],linestyle='dashed',color='blue')
	ax_timeseries.fill_between(times_batch,l,h,facecolor='turquoise',alpha=0.8, zorder = 0)
	ax_timeseries.plot(times_batch, m, ':', linewidth=1., color='b', zorder = 2, label = "Median prediction")
	ax_timeseries.plot(times_batch, data_batch, linewidth=1.5, color='r', zorder = 3, label = "Data")
	ax_timeseries.axvline(t_merger, linestyle='dashed',color='k')
	
	if id_ == id_start-id_step:
		ax_timeseries.legend()
	
	ax_WF.plot(times_batch, WF_batch, linewidth=1., color='k', zorder = 3)
	#ax_WF.axvline(times[id_],linestyle='dashed',color='blue')
	ax_WF.axvline(t_merger, linestyle='dashed',color='red')
	
	
	continue
	
		#PSD analysis (rubbish)
	temp_M = MESA()
	temp_M.solve(data_batch, method = "Fast", optimisation_method = "FPE")
	temp_spec = temp_M.spectrum(1./srate, f_grid)
	
	if t_merger >= times_batch[0] and t_merger <= times_batch[-1]:
		ax_PSDseries.axvline(i*len(f_grid),linestyle='dashed',color='red', linewidth=1.5, zorder = 2)
		ax_PSDseries.axvline((i+1)*len(f_grid),linestyle='dashed',color='red',linewidth=1.5, zorder = 2)
	else:
		ax_PSDseries.axvline(i*len(f_grid),linestyle='dashed',color='blue', zorder = 1)		
	ax_PSDseries.plot(range(i*len(f_grid),(i+1)*len(f_grid) ), np.sqrt(temp_spec))#- PSD_baseline)

	
sys.stderr.write('\n')	
plt.show()
	




















