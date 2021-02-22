"""
Given a time series, it computes the PSD and start forecasting and measuring the LL of the data, given the forecast.
It plots a number of quantieties
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
from memspectrum import MESA

def data_LL_gauss(data, prediction):
	mu = np.mean(prediction, axis = 0)
	std_dev = np.std(prediction, axis = 0)
	LL = -0.5*np.square(np.divide(data - mu, std_dev)) #(D,)
	return LL
	
	
def AnomalyDetection_pipeline(data, srate, T_train, N_step, plot = True, injection = None,):
	"Runs the Anomaly Detection Pipeline on the data given. Trains a memspectrum model with data from 0 to T_train and forecast the rest, with a step of N_step points. If an injection is give, it is plot together with LL series"

	id_start = int(T_train*srate) #start id for forecasting
	train_data = data[0:id_start]
	times = np.linspace(0, len(data)/srate, len(data))

	print("Computing PSD on {}s of data".format(T_train))
	M = MESA()
	P, ak, _ = M.solve(train_data, method = "Standard", optimisation_method = "FPE", early_stop = True)

	#starting "pipeline"
	time_step = N_step/srate # step in seconds seconds of data
	Np = 100 #number of predictions

	if plot:
			#timeseries plot
		fig_timeseries = plt.figure(0)
		ax_timeseries  = fig_timeseries.add_subplot(111)
		ax_timeseries.set_ylabel("Strain")
		ax_timeseries.plot(times[:id_start], train_data[:id_start], linewidth=1., color='r', zorder = 3) #plot train data

			#LL series plot
		fig_LLseries = plt.figure(1)
		if injection is not None:
			ax_LLseries  = fig_LLseries.add_subplot(211)
			ax_WF  = fig_LLseries.add_subplot(212)
			ax_WF.set_ylabel("Strain")
		else:
			ax_LLseries  = fig_LLseries.add_subplot(111)

		ax_LLseries.set_ylabel("LL")
		ax_LLseries.set_yscale('log')

			#PSD plot
		fig_PSDseries = plt.figure(2)
		ax_PSDseries  = fig_PSDseries.add_subplot(111)
		ax_PSDseries.set_ylabel("PSD")
		ax_PSDseries.set_yscale('log')
		freq = np.linspace(1./T_train, 0.5*srate,1000)
		spec = M.spectrum(1/srate,freq)
		ax_PSDseries.loglog(freq, spec)

	ids_iterator = range(id_start,len(data)-N_step, N_step)
	for i, id_ in enumerate(ids_iterator): #problem here with missing gaps (or something weird related...)
		sys.stderr.write("\rAnalysing batch {} of {}: t in [{},{}]".format(i+1, len(ids_iterator),
			times[id_],times[id_+N_step]))

			#forecasting predictions
		times_batch = times[id_:id_+N_step+1]
		data_batch = data[id_:id_+N_step+1] #data to predict
	
		forecast_basis = data[id_-M.get_p():id_] #batch of data that are the basis for forecasting
		predictions = M.forecast(forecast_basis, N_step+1, Np) #(Np, N_step)
		
			#computing LL	
		#LL = np.zeros((int(id_step)+1,))
		LL = data_LL_gauss(data_batch, predictions)
		l, m, h = np.percentile(predictions, [5,50,95],axis=0)
	
		if plot:
			#ax_LLseries.axvline(times[id_],linestyle='dashed',color='blue') #plotting vertical lines at division
				#plot LL, predictions and data_batch
			ax_LLseries.plot(times_batch, np.cumsum(LL)+1e5)
			ax_timeseries.fill_between(times_batch,l,h,facecolor='turquoise',alpha=0.8, zorder = 0)
			ax_timeseries.plot(times_batch, m, ':', linewidth=.7, color='b', zorder = 2, label = "Median prediction")
			ax_timeseries.plot(times_batch, data_batch, linewidth=1., color='r', zorder = 3, label = "Data")
	
		if i == 0:
			ax_timeseries.legend()
	
		if plot and injection is not None:
			WF_batch = injection[id_:id_+N_step+1]
			ax_WF.plot(times_batch, WF_batch, linewidth=1., color='k', zorder = 3)
	
	sys.stderr.write('\n')	
	
	return	
	
