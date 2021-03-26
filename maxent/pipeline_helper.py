import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
from memspectrum import MESA
import scipy.signal as sig

import numpy as np
import matplotlib.pyplot as plt
from pipeline import *
import scipy.signal as sig
import pandas as pd
import pickle

import mlgw.GW_generator as gen

import emd #nice package for emd: pip install emd
from PyEMD import EMD #This package is way better than the other!!

################
def load_emd(infile, data = None, times = None, WF = None):
	imf = np.loadtxt(infile)
	if data is None and times is None and WF is None:
		return imf
	trim_borders = (len(data)-imf.shape[0])//2
	ret_list = [imf]
	if trim_borders >0: ret_list.append(data[trim_borders:data.shape[0]-trim_borders])
	if trim_borders >0 and times is not None: ret_list.append(times[trim_borders:times.shape[0]-trim_borders])
	if trim_borders >0 and WF is not None: ret_list.append(WF[trim_borders:WF.shape[0]-trim_borders])

	if len(ret_list) == 1:
		return ret_list[0]
	if len(ret_list) == 2:
		return ret_list[0],ret_list[1]
	elif len(ret_list) == 3:
		return ret_list[0],ret_list[1], ret_list[2]
	elif len(ret_list) == 4:
		return ret_list[0],ret_list[1], ret_list[2], ret_list[3]
	

def do_emd(data, emd_type = 'PyEMD', outfile = None, trim_borders = 0, times = None, WF = None, header = None):
	if emd_type == 'emd':
		imf = emd.sift.sift(data, sift_thresh = 1e-8)
	if emd_type == 'PyEMD':
		emd_model = EMD()
		imf = emd_model(data).T #(N,N_emd)

	if isinstance(trim_borders,int):
		trim_borders = (trim_borders,trim_borders)
	imf = imf[trim_borders[0]:imf.shape[0]-trim_borders[1],:]
	ret_list = [imf]
	if trim_borders != (0,0): ret_list.append(data[trim_borders[0]:data.shape[0]-trim_borders[1]])
	if trim_borders != (0,0) and times is not None: ret_list.append(times[trim_borders[0]:times.shape[0]-trim_borders[1]])
	if trim_borders != (0,0) and WF is not None: ret_list.append(WF[trim_borders[0]:WF.shape[0]-trim_borders[1]])
	
	if isinstance(outfile, str):
		if not isinstance(header, str): header = 'IMf modes: shape {}'.format(imf.shape)
		np.savetxt(outfile, imf, header = header)

		#returning values (ugly)
	if len(ret_list) ==1:
		return ret_list[0]
	elif len(ret_list) == 2:
		return ret_list[0],ret_list[1]
	elif len(ret_list) == 3:
		return ret_list[0],ret_list[1], ret_list[2]
	elif len(ret_list) == 4:
		return ret_list[0],ret_list[1], ret_list[2], ret_list[3]

def downsample_data(downsampling_factor, srate, data, times = None, WF = None):
	downsampling_factor = int(downsampling_factor)
	data = sig.decimate(data, downsampling_factor)
	if times is not None: times = sig.decimate(times, downsampling_factor)
	if WF is not None: WF = sig.decimate(WF, downsampling_factor)
	srate = srate/float(downsampling_factor)
	dt = 1./srate
		#return part
	ret_list = [srate, data]
	if times is not None: ret_list.append(times)
	if WF is not None: ret_list.append(WF)
	
		#returning values (ugly)
	if len(ret_list) ==2:
		return ret_list[0],ret_list[1]
	elif len(ret_list) == 3:
		return ret_list[0],ret_list[1], ret_list[2]
	elif len(ret_list) == 4:
		return ret_list[0],ret_list[1], ret_list[2], ret_list[3]

def create_injection_list(N_inj, srate, start_GPS, end_GPS, theta_range = [[10,100],[10,100],[-0.8,0.8],[-0.8,0.8],[100,500],[0.,np.pi],[0.,2*np.pi]]):
	"""
	Creates a list of injections.
	Each injection is a dictionary with entries:
		"WF": a WF array plus of variable length
		"srate": a sampling rate for the WF
		"GPS": GPS time for the merger
		"theta": a 7 dimensional array holding params of the WF [m1,m2,s1,s2,d_L, iota, phi]
	"""
	g = gen.GW_generator(0)
	inj_list = []
	theta_range = np.array(theta_range)
	for i in range(N_inj):
		inj = {}
		theta = np.random.uniform(*theta_range.T,size = (7,))
		t_min = np.random.uniform(-10,-6)
		t_grid = np.linspace(t_min, 0.1, int((0.1-t_min)*srate))
		WF = g.get_WF(theta, t_grid)[int(np.random.rand()+0.5)] #picking at random +/x polarization(D,)
		inj['theta'] = theta
		inj['srate'] = srate
		inj['GPS'] = int(start_GPS)
		inj['time'] = np.random.uniform(.3, float(end_GPS-start_GPS))
		inj['WF'] = WF
		inj_list.append(inj)
	return inj_list
		
def save_inj_list(filename, inj_list):
	if not filename.endswith('.pkl'): filename +='.pkl'
	with open(filename, 'wb') as f:
		pickle.dump(inj_list, f, pickle.HIGHEST_PROTOCOL)

def load_inj_list(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)
		
def split_data(indata, outfolder, srate, start_GPS, T_batch, T_overlap, injection_list = None, downsampling_factor = 1, prefix = 'H'):
	"""
	Split the raw data of infile in overlapping batches each of T_batch seconds and perform the emd.
	Save each batch to a separate file in outfolder. It can also perform injection and downsample the data
	"""
	if not outfolder.endswith('/'): outfolder +='/'
	if isinstance(indata,str):
		data = np.squeeze(pd.read_csv(indata, skiprows =3))
		print("Loaded data")
	elif isinstance(indata,np.ndarray):
		data = np.squeeze(indata)
	else:
		raise RuntimeError("Wrong type of data")
	
		#performing injections
	if isinstance(injection_list, list):
			#a bit messy with the indices here... but it seems allright
		for inj in injection_list:
			if inj['srate'] != srate: raise ValueError("The injection sampling rate is not the data sampling rate: unable to proceed.")
			id_data = int(((inj['GPS']-start_GPS) + inj['time'])*srate) #id at which the merger should happen
			id_WF = np.argmax(np.square(inj['WF'])) #id of the WF at which the merger happens
			len_end = len(inj['WF'])-id_WF #number of steps from merger to the end
			id_start = max(id_data+len_end-len(inj['WF']),0)
			id_end = min(id_data+len_end, len(data))
			data[id_start:id_end] += inj['WF'][id_start-id_data+id_WF:id_end-id_data+id_WF]

	if int(downsampling_factor) >1:
		srate, data = downsample_data(int(downsampling_factor), srate, data, times = None, WF = None)

	srate = float(srate)	
	N_batch = int(T_batch*srate)
	overlap = int(T_overlap*srate)

	data *=1e18 #crucial for emd
	
	n =0 #id for the current start batch
	file_list = []
	while n + N_batch <= len(data) :
		data_batch = data[n:n+N_batch]
		
		filename = outfolder+"{}-{}Hz-{}-{}.emd.dat".format(prefix, srate, start_GPS + n/srate, N_batch/srate)
		header = "#emd data\n#GPS start time: {}\n#srate: {}\n#lenght: {}s".format(start_GPS + n/srate,srate, N_batch/srate)
		
		#perform EMD & save to file
		do_emd(data_batch, emd_type = 'PyEMD', outfile = filename, trim_borders = 0, times = None, WF = None, header = header)

		file_list.append(filename)
		n += N_batch - overlap
		print("Saving batch @ {}".format(filename))
	
	return file_list
	

def band_pass(fmin, fmax, srate, data):
	(B,A) = sig.butter(5,[fmin/(.5*srate), fmax/(.5*srate)], btype='band')#, fs = srate)
	data_pass = sig.filtfilt(B, A, data)
	return data_pass

def plot_PSD_imf(imf, data, srate, WF = None, title = None, folder = '.'):
	M = MESA()
	fig = plt.figure( figsize=(8,4) )
	ax = fig.gca()
	title_str = "PSD of the EMD modes vs PSD of data"
	if title is not None:
		title_str = title_str+ ' - '+title
	plt.title(title_str)
	
		#spectrum of data
	M.solve(data, method = 'Fast', optimisation_method = 'CAT')
	spec_data, f = M.spectrum(1/srate)
	ax.loglog(f[:len(data)//2], spec_data[:len(data)//2], label= "data")

	if WF is not None:
		M.solve(WF, method = "standard")
		spec_WF, f = M.spectrum(1/srate)
		ax.loglog(f[:len(data)//2], spec_WF[:len(data)//2], label= "WF")

	for i in range(imf.shape[1]):
		print("EMD component {}/{}".format(i+1 ,imf.shape[1]))
		M.solve(imf[:,i], method = 'Standard', optimisation_method = 'CAT')
		spec, f= M.spectrum(1/srate)
		ax.loglog(f[:len(data)//2], spec[:len(data)//2], label= "EMD {}".format(i+1))

	plt.legend()
	if folder is None: return
	if title is None:
		fig.savefig(folder+"/PSD_imf.pdf")
	else:
		fig.savefig(folder+"/PSD_imf-{}.pdf".format(title))


def plot_imf(imf, data, title = None, folder = '.', times = None):
	if times is None: times = np.arange(0, imf.shape[0])
	fig, ax = plt.subplots(imf.shape[1]+1,1, sharex = True, figsize=(16,8) )
	title_str = "Data + the {} components of Empirical Mode Decomposition".format(imf.shape[1])
	if title is not None:
		title_str = title_str+ ' - '+title
	fig.suptitle(title_str)

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
	if folder is None: return
	if title is None:
		fig.savefig(folder+"/EMD_modes.pdf")
	else:
		fig.savefig(folder+"/EMD_modes-{}.pdf".format(title))


def plot_cluster_data(infile):
	indata = np.loadtxt(infile)
	
	red_data, flags = indata[:,[0,1]],indata[:,2]

	print(indata.shape)

	plt.figure()
	injs_ids = np.where(flags ==1)[0]
	n_injs_ids = np.where(flags ==0.5)[0]
	other_ids = np.where(flags ==0)[0]
	print(len(other_ids))
	plt.scatter(red_data[injs_ids,0], red_data[injs_ids,1], c = 'r', cmap = 'cool', zorder =10)
	plt.scatter(red_data[n_injs_ids,0], red_data[n_injs_ids,1], c = 'g', cmap = 'cool', zorder =2)
	plt.scatter(red_data[other_ids,0], red_data[other_ids,1], c = 'b', cmap = 'cool', zorder = 0)
	
	
	where_inside = np.logical_and(np.logical_and(red_data[:,0]>0.000124589058,red_data[:,0]<0.000124595325),
						np.logical_and(red_data[:,1]>-0.0001095050,red_data[:,1]<-0.0001094850))

	inside = np.zeros(flags.shape)
	
	plt.figure()
	plt.scatter(red_data[where_inside,0], red_data[where_inside,1], c = 'r', cmap = 'cool', zorder =10)
	plt.scatter(red_data[~where_inside,0], red_data[~where_inside,1], c = 'b', cmap = 'cool', zorder =0)
	
	true_pos = np.sum(flags[~where_inside] >0)
	false_pos = np.sum(flags[~where_inside] ==0)
	
	true_neg = np.sum(flags[where_inside] ==0)
	false_neg = np.sum(flags[where_inside] >0)
	print(true_pos, true_neg)
	print(false_pos, false_neg)
	print(true_pos+false_neg)
	print(true_neg+true_pos+false_pos+true_pos)
	
	
	
	
	
	
	
