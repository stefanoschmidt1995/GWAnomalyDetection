import sys
sys.path.insert(0,'../maximum_entropy/Maximum-Entropy-Spectrum/')
from memspectrum import MESA

import numpy as np
import matplotlib.pyplot as plt

import time
import matplotlib.pyplot as plt
from scipy.signal import decimate

import mlgw.GW_generator as gen
g = gen.GW_generator()

def data_LL(data, prediction):
	mu = np.mean(prediction, axis = 0)
	std_dev = np.std(prediction, axis = 0)
	LL = -0.5*np.square(np.divide(data - mu, std_dev)) #(D,)
	LL = LL - 0.5*np.log(2*np.pi) - np.log(std_dev) #(D,)

	return np.sum(LL), LL 

srate = 4096.
dt = 1./srate
bandpassing = 0
f_min_bp = 20.0
f_max_bp = (srate-20)/2.0
T  = 6
t_forecast = 4
datafile = "V-V1_GWOSC_4KHZ_R1-1186741846-32.txt"
data = np.loadtxt(datafile)[:int(T*4096)]
if srate != 4096.:
	data = decimate(data, int(4096/srate), zero_phase=True)
if bandpassing == 1:
	from scipy.signal  import butter, filtfilt
	bb, ab = butter(4, [f_min_bp/(0.5*srate), f_max_bp/(0.5*srate)], btype='band')
	data = filtfilt(bb, ab, data)

N = data.shape[0]
f = np.fft.fftfreq(N, d=dt)
t = np.arange(0,T,step=dt)
M = MESA(data)
start = time.perf_counter()
P, ak, _ = M.solve(method = "Fast", optimisation_method = "FPE", m = int(2*N/(2*np.log(N))))
print("p = {}".format(len(ak)))
elapsed = time.perf_counter()
elapsed = elapsed - start
print ("Time spent MESA: {0} s".format(elapsed))
start = time.perf_counter()
PSD= M.spectrum(dt,f)
elapsed = time.perf_counter()
elapsed = elapsed - start
print ("Time spent PSD: {0} s".format(elapsed))

fig = plt.figure(1)
ax  = fig.add_subplot(111)
ax.loglog(f[:N//2], M.spectrum(dt,f)[:N//2],'-k')
ax.set_xlim(1,srate/2.)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (Hz$^{-1}$)")

M = MESA(data[:int(t_forecast*srate)])
P, ak, _ = M.solve(method = "Fast", optimisation_method = "FPE", m = int(2*N/(2*np.log(N))))
Np = 100
prediction = M.forecast(int(N-int(t_forecast*srate)), Np) #(Np, D)
l, h = np.percentile(prediction,[5,95],axis=0)

	#computing LL of the noise model & noise + signal model
print("Data LL: ", data_LL(data[int(t_forecast*srate):], prediction)[0])

t_WF = np.linspace(t_forecast, T, N- int(t_forecast*srate))
WF = g.get_WF([10,20,-0.3,0.5, 3, 0.3, 2.435], t_WF-T)[0]
signal_data = data[int(t_forecast*srate):] #+ WF

print("Data LL signal: ", data_LL(signal_data, prediction)[0])
print("LL ratio P(signal = noise)/P(benchmark = noise): ", data_LL(signal_data, prediction)[0]-data_LL(data[int(t_forecast*srate):], prediction)[0])

plt.figure(10)
plt.title("LL difference")
plt.plot(np.linspace(t_forecast, T, N- int(t_forecast*srate)), data_LL(signal_data, prediction)[1]-data_LL(data[int(t_forecast*srate):], prediction)[1])

fig = plt.figure(2)
ax  = fig.add_subplot(111)
ax.plot(t,data,linewidth=1.5,color='r',zorder=2)
ax.axvline(t_forecast,linestyle='dashed',color='blue')
ax.plot(t[int(t_forecast*srate):],np.mean(prediction, axis =0),linewidth=1., color='k', zorder = 1)
#ax.plot(t[int(t_forecast*srate):],prediction.T,linewidth=0.5, color='k', zorder = 0)
ax.fill_between(t[int(t_forecast*srate):],l,h,facecolor='turquoise',alpha=0.8, zorder = 0)
ax.set_xlabel("time (s)")
ax.set_ylabel("strain")


fig = plt.figure(3)
ax  = fig.add_subplot(111)
#ax.plot(t[:int(t_forecast*srate)],data[:int(t_forecast*srate)],linewidth=1.5,color='r',zorder=2)
ax.plot(t,data,linewidth=1.5,color='r',zorder=2)
ax.plot(t[int(t_forecast*srate):],signal_data,linewidth=1., color='b', zorder = 3)
ax.axvline(t_forecast,linestyle='dashed',color='blue')
ax.plot(t[int(t_forecast*srate):],np.mean(prediction, axis =0),linewidth=1., color='k', zorder = 1)
ax.fill_between(t[int(t_forecast*srate):],l,h,facecolor='turquoise',alpha=0.8, zorder = 0)
ax.set_xlabel("time (s)")
ax.set_ylabel("strain")
plt.show()
