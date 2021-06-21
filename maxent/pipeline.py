"""
Given a time series, it computes the PSD and start forecasting and measuring the LL of the data, given the forecast.
It plots a number of quantieties
"""

#TODO: realistic SNR computation
#TODO: improve forecasting method: that's probably key to improving performances

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
from memspectrum import MESA
import pickle
import os
import scipy.spatial

from pipeline_helper import *

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
		fig_timeseries = plt.figure()
		ax_timeseries  = fig_timeseries.add_subplot(111)
		ax_timeseries.set_ylabel("Strain")
		ax_timeseries.plot(times[:id_start], train_data[:id_start], linewidth=1., color='r', zorder = 3) #plot train data

			#LL series plot
		fig_LLseries = plt.figure()
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


def gather_LLs(infolder, injs, LL_treshold = -4):
	"Gathers together a number of LL series and state whether in each of them an injection is present. It performs PCA and plots"

	if not infolder.endswith('/'): infolder +='/'
	LL_files = os.listdir(infolder) 
	LL_list = []
	times_list = []
	
	for LL_file in LL_files:
		with open(infolder+LL_file, 'rb') as f:
			LL_dict = pickle.load(f)

			#saving LLs_times and  LLs_timeseries
		LLs_times = np.zeros((LL_dict['LL'].shape[0],), dtype = np.float128)
		LLs_times = LL_dict['GPS'] + LL_dict['t_start']
		times_list.append(LLs_times)
		LL_list.append(LL_dict['LL'])
		
		len_batch = LL_dict['LL'].shape[1] / LL_dict['srate'] 
	
	LLs = np.concatenate(LL_list, axis = 0) #(N,D)
	LLs_times = np.concatenate(times_list)#(N,)

	trigger_times, triggers_ids, red_data = detect_outliers(LLs, LLs_times, threshold = LL_treshold, K_PCA = 2)
	print(triggers_ids, trigger_times)
	
	if injs is None: #this is just for GW 150914... as an example
		plt.figure()
		plt.scatter(red_data[:,0], red_data[:,1], c= 'b')
		deltaT = LLs_times[1]-LLs_times[0]
		id_event = np.where(np.logical_and((1126259462.4-LLs_times)>0, (1126259462.4-LLs_times)<deltaT))
		plt.scatter(red_data[id_event,0], red_data[id_event,1], c= 'r')
		plt.xlabel('PCA 1')
		plt.ylabel('PCA 2')
		plt.show()
	
	injected_triggers_ids, detected_injs_ids = check_triggers(trigger_times, injs, len_batch *2. ) #here we find the injected WFs among the triggers
	print(detected_injs_ids)

	print("Injections statistics")
	print("\tDetection/triggers: ", len(injected_triggers_ids)/len(triggers_ids))
	print("\tDetected injections (%): ",len(np.unique(detected_injs_ids)),len(np.unique(detected_injs_ids))/len(injs))
	print("\t# of injections: ", len(injs))
	
	injected_ids, _ = check_triggers(LLs_times, injs, len_batch *2. ) #here we find the ids of the injections performed

		#plotting: the injections are in red
	plt.figure()
	injs_bool_vector = np.zeros((red_data.shape[0],)).astype(bool)
	injs_bool_vector.fill(False)
	injs_bool_vector[injected_ids] = True
		#plotting data and injections
	plt.scatter(red_data[injs_bool_vector,0], red_data[injs_bool_vector,1], c = 'r', facecolors='none', cmap = 'cool', zorder =10)
	plt.scatter(red_data[~injs_bool_vector,0], red_data[~injs_bool_vector,1], c = 'b', cmap = 'cool', zorder = 0)
		#plotting triggers
	plt.scatter(red_data[triggers_ids,0], red_data[triggers_ids,1], c = 'y', marker = 'x', cmap = 'cool', zorder =10)
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')

		#plotting parameters
	theta_injs = [inj['theta'] for inj in injs]
	theta_injs = np.column_stack(theta_injs).T #(N_injs, 7)
	SNR_injs = np.array([inj['SNR'] for inj in injs])
	D_eff = np.array([inj['D_eff'] for inj in injs])
	
	plt.figure()
	plt.hist(SNR_injs)
	plt.hist(SNR_injs[detected_injs_ids], label = 'detected')
	plt.legend()

	plt.figure()
	plt.title("Injections performed")
	plt.scatter(mchirp(theta_injs[:,0],theta_injs[:,1]), D_eff, c= 'b')
	plt.scatter(mchirp(theta_injs[detected_injs_ids,0],theta_injs[detected_injs_ids,1]), D_eff[detected_injs_ids], marker = 'x', c = 'r')
	plt.xlabel(r'$\mathcal{M}_{c} (M_\odot)$')
	plt.ylabel(r'$D_{eff} (Mpc)$')
	
	plt.figure()
	plt.title("Injections performed")
	plt.scatter(SNR_injs, D_eff, c= 'b')
	plt.scatter(SNR_injs[detected_injs_ids], D_eff[detected_injs_ids], marker = 'x', c = 'r')
	plt.xlabel(r'SNR')
	plt.ylabel(r'$D_{eff} (Mpc)$')


	#np.savetxt('data/red_data.dat', np.concatenate([red_data, injs_bool_vector[:,None]] , axis = 1))

	return

def detect_outliers(LLs_timeseries, GPS_timeseries, threshold = 4, K_PCA = 2):
	"Detect eventual outliers with PCA and launch triggers at the outliers found."
	model = PCA_model()
	model.fit_model(LLs_timeseries, K = K_PCA) #(N,D)
	
		#ugly, you should divide things between train and test...
	red_data = model.reduce_data(LLs_timeseries) #(N,K)
	
		#distance between others
		#computing 25-75 percentile
	percentile = np.percentile(red_data, [15,85], axis = 0) #(2,K_PCA)
	normal = np.logical_and(red_data>percentile[0,:], red_data<percentile[1,:])# (N,K_PCA)
	normal = np.prod(normal, axis = 1).astype(bool)
	
	print("Len normal data", len(red_data[normal]))
	
	scores = scipy.spatial.distance_matrix(red_data, red_data[normal]) #(N,N)
	scores = np.log(np.mean(np.square(scores), axis = 1))
	
	ids_scores = np.where(scores>threshold)
	
	if False:
		plt.scatter(red_data[:,0],red_data[:,1], c= scores, zorder = 1)
		plt.colorbar()
		plt.scatter(red_data[ids_scores,0],red_data[ids_scores,1], c = 'r', zorder = 10)
		#plt.scatter(*red_data[normal].T	, c ='r', zorder = 0)
		
		plt.show()
	
	return GPS_timeseries[ids_scores], np.array(range(red_data.shape[0]))[ids_scores], red_data #trigger times and their ids
	

def detect_outliers_old(infile, injs = None):
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

def check_triggers(trigger_times, injs, threshold):
	"Given a list of GPS times for some triggers, it computes whether each trigger time matches any injection time within a threshold"
	injs_times = np.array([inj['time'] +np.array(inj['GPS'], dtype = np.float128) for inj in injs], dtype = np.float128) #(N_injs,)

	print(trigger_times.shape, injs_times.shape)

	deltaTs = scipy.spatial.distance_matrix(trigger_times[:,None], injs_times[:,None]) #(N,N_injs)
	
	ids_triggers, ids_injs = np.where(np.abs(deltaTs)<threshold)
	
	print(deltaTs.shape, ids_triggers, ids_injs)
	
	return ids_triggers, ids_injs	
	
	
	
