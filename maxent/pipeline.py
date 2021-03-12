"""
Given a time series, it computes the PSD and start forecasting and measuring the LL of the data, given the forecast.
It plots a number of quantieties
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
from memspectrum import MESA
import pickle
import os

from mlgw.ML_routines import PCA_model

def data_LL_gauss(data, prediction, std_dev = None):
	mu = np.mean(prediction, axis = 0)
	if std_dev is None:
		std_dev = np.std(prediction, axis = 0)
	LL = -0.5*np.square(np.divide(data - mu, std_dev)) #(D,)
	return LL
	
	
def AnomalyDetection_pipeline(data, srate, T_train, N_step, outfile = None, plot = True, GPS_time = 0.,injection = None, method = "FPE"):
	"Runs the Anomaly Detection Pipeline on the data given. Trains a memspectrum model with data from 0 to T_train and forecast the rest, with a step of N_step points. If an injection is give, it is plot together with LL series"
	print("Running pipeline @ GPS time {}.\n\tT_train, N_step = {},{}".format(GPS_time,T_train, N_step))

	#TODO: save LL in pickle, so you can have a GPS time as int and the of the LL in float

	LL_list = []
	t_start_list = []

	id_start = int(T_train*srate) #start id for forecasting
	train_data = data[0:id_start]
	times = np.linspace(0, len(data)/srate, len(data))

	#print("Computing PSD on {}s of data".format(T_train))
	M = MESA()
	P, ak, _ = M.solve(train_data, method = "Standard", optimisation_method = method, early_stop = True)

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
		plt.title("PSD of {}s of training data".format(T_train))
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
	
		if i == 0 and plot:
			ax_timeseries.legend()
	
		if plot and injection is not None:
			WF_batch = injection[id_:id_+N_step+1]
			ax_WF.plot(times_batch, WF_batch, linewidth=1., color='k', zorder = 3)
	
		if isinstance(outfile, str):
			LL_list.append(LL) #(N_step,)
			t_start_list.append(times_batch[0])
	sys.stderr.write('\n')

		#creating output dict
	LL_dict = {}
	LL_dict['GPS'] = GPS_time
	LL_dict['srate'] = srate
	LL_dict['N_batches'] = len(LL_list)
	LL_dict['t_start'] = np.array(t_start_list) #(N_batches,)
	LL_dict['LL'] = np.array(LL_list) #(N_batches,D)
	#print(LL_dict['LL'].shape,LL_dict['t_start'].shape)

	if isinstance(outfile, str):
		if not outfile.endswith('.pkl'): outfile +='.pkl'
		with open(outfile, 'wb') as f:
			pickle.dump(LL_dict, f, pickle.HIGHEST_PROTOCOL)

	return LL_dict


def gather_LLs(infolder, injs):
	"Gathers together a number of LL series and state whether in each of them an injection is present. It performs PCA and plots"

	if not infolder.endswith('/'): infolder +='/'
	LL_files = os.listdir(infolder) #(N_batches,D)
	LL_list = []
	flag_list = []
	theta_list = []
	
	for LL_file in LL_files:
		with open(infolder+LL_file, 'rb') as f:
			LL_dict = pickle.load(f)

		flags = np.zeros((LL_dict['LL'].shape[0],))
		theta = np.zeros((LL_dict['LL'].shape[0],7))
		
			#assigning triggers
		t_start = LL_dict['t_start']
		inj_time_list = np.array([inj['time'] +float(inj['GPS'] -LL_dict['GPS']) for inj in injs])
		for i, t in enumerate(inj_time_list):
			t_diff = t - t_start
			ids_ = np.where(np.logical_and(t_diff < np.abs(t_diff[0]-t_diff[1]), t_diff >0))[0]
			print(t, t_start[ids_])
			flags[ids_] = 1.
			theta[ids_] = injs[i]['theta']
			try:
				if np.abs(t_diff[ids_+1]) > np.abs(t_diff[ids_-1]) and ids_!=0:
					flags[ids_-1] = .5
					theta[ids_-1] = injs[i]['theta']
				else:
					flags[ids_+1] = .5
					theta[ids_+1] = injs[i]['theta']
			except:
				pass
		
		flag_list.append(flags)
		LL_list.append(LL_dict['LL'])
		theta_list.append(theta)
		
	LLs = np.concatenate(LL_list, axis = 0) #(N,D)
	flags = np.concatenate(flag_list)#(N,)
	thetas = np.concatenate(theta_list,axis =0)
	
	print(LLs.shape, flags.shape)
	
		#PCA part
	model = PCA_model()
	model.fit_model(LLs, K =2)
	red_data = model.reduce_data(LLs) #ugly, you should divide things between train and test...
	
	plt.figure()
	injs_ids = np.where(flags ==1)[0]
	n_injs_ids = np.where(flags ==0.5)[0]
	other_ids = np.where(flags ==0)[0]
	plt.scatter(red_data[injs_ids,0], red_data[injs_ids,1], c = 'r', cmap = 'cool', zorder =10)
	plt.scatter(red_data[n_injs_ids,0], red_data[n_injs_ids,1], c = 'g', cmap = 'cool', zorder =2)
	plt.scatter(red_data[other_ids,0], red_data[other_ids,1], c = 'b', cmap = 'cool', zorder = 0)

	np.savetxt('data/red_data.dat', np.concatenate([red_data, flags[:,None]] , axis = 1))

		#creating a histogram
	import scipy.spatial
	ok_ids = np.concatenate([injs_ids])#,n_injs_ids])
	dist = scipy.spatial.distance_matrix(red_data[ok_ids,:],red_data) #(M,N)
	dist = np.mean( dist, axis = 1) #(M,) #distance of each injection from the rest
	
	plt.figure()
	plt.scatter((thetas[ok_ids,0]+thetas[ok_ids,1])/thetas[ok_ids,4], dist)
	plt.xlabel("M/R")
	plt.ylabel("dist")

	
	return
	
def detect_outliers(infile, injs = None):
	"Reduce the dimensionality of the LL series and try outliers detection with clustering"

	with open(infile, 'rb') as f:
		LL_dict = pickle.load(f)

	LL_timeseries = LL_dict['LL'] #(N_batches, N_step)
	t_start = LL_dict['t_start']

	print("Print loaded {} batches with {} time steps each".format(LL_timeseries.shape[0],LL_timeseries.shape[1]))
	
	model = PCA_model()
	model.fit_model(LL_timeseries, K =2)
	
	red_data = model.reduce_data(LL_timeseries) #ugly, you should divide things between train and test...

		#plotting injections if available
	
	if injs is not None:
		inj_time_list = np.array([inj['time'] +float(inj['GPS'] -LL_dict['GPS']) for inj in injs])
		inj_time_list = inj_time_list[np.where(inj_time_list>0)]

		print('inj_time_list',inj_time_list)

		colors = np.zeros(LL_timeseries.shape[0])
		
		for t in inj_time_list:
			t_diff = t - t_start
			ids_ = np.where(np.logical_and(t_diff < np.abs(t_diff[0]-t_diff[1]), t_diff >0))[0]
			print(t, t_start[ids_])
			colors[ids_] = 1
			try:
				if np.abs(t_diff[ids_+1]) > np.abs(t_diff[ids_-1]) and ids_!=0:
					colors[ids_-1] = .5
				else:
					colors[ids_+1] = .5
			except:
				pass

		plt.figure()
		injs_ids = np.where(colors ==1)[0]
		n_injs_ids = np.where(colors ==0.5)[0]
		other_ids = np.where(colors ==0)[0]
		plt.scatter(red_data[injs_ids,0], red_data[injs_ids,1], c = 'r', cmap = 'cool', zorder =10)
		plt.scatter(red_data[n_injs_ids,0], red_data[injs_ids,1], c = 'g', cmap = 'cool', zorder =2)
		plt.scatter(red_data[other_ids,0], red_data[other_ids,1], c = 'b', cmap = 'cool', zorder = 0)
		
		#plt.figure()
		#plt.scatter(t_start[injs_ids], red_data[injs_ids,0], c = 'r', cmap = 'cool', zorder =1)
		#plt.scatter(t_start[other_ids], red_data[other_ids,0], c = 'b', cmap = 'cool', zorder = 0)
	else:
		colors = t_start
		plt.figure()
		plt.scatter(red_data[:,0], red_data[:,1], c = colors, cmap = 'cool', zorder = 0)
		plt.colorbar()
		
		plt.figure()
		plt.scatter(t_start, red_data[:,0], c = colors, cmap = 'cool', zorder = 0)
		plt.xlabel("1st PCA")

		plt.figure()
		plt.scatter(t_start, red_data[:,1], c = colors, cmap = 'cool', zorder = 0)	
		plt.xlabel("2nd PCA")
	
	
	
	
