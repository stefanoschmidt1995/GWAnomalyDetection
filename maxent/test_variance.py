import sys
sys.path.insert(0,'../../Maximum-Entropy-Spectrum/')
sys.path.insert(0,'../Maximum-Entropy-Spectrum/')
from memspectrum import MESA
import scipy.signal as sig

import numpy as np
import matplotlib.pyplot as plt
from pipeline import *
import scipy.signal as sig
import pandas as pd

import mlgw.GW_generator as gen
g = gen.GW_generator()

import pipeline

import emd #nice package for emd: pip install emd

datafile = "H-H1_GWOSC_4KHZ_R1-1126257415-1000.txt.gz"
print("loaded data")
srate = 4096.#*4.
data = np.squeeze(pd.read_csv(datafile, skiprows =3))
data = data*1e18

#imf = emd.sift.sift(data, sift_thresh = 1e-8)
#imf = imf[1000:,:] #removing start (edge effect?)
#data_good = imf[1000:,3]
data_good = data


	#computing variance of data
#var_emd = np.var(imf,axis =0)[[0,1]]
#var_data = np.var(data)
#var_data = np.concatenate([[var_data],var_emd])
#np.savetx('var_data.dat', var_data)

T_train = int(200*srate) #100 s of training data

print("#training data: ",T_train)
p_list = [100, 1000, 10000]#, 25000, 50000]
var_list = []
LL_list = []
L_max = 5000 #max forecast distance

M = MESA()
if True: #compute
	for p_ in p_list:
		print("Solving for p = {}".format(p_))
		M.solve(data_good[:T_train], method = 'Fast', m = p_,optimisation_method = 'Fixed', verbose = True)
		predictions = M.forecast(data_good[T_train-p_:T_train], L_max, number_of_simulations = 100, seed = 0, verbose = True)
		var = np.var(predictions, axis = 0)
		var_list.append(var)
		pred_LL = pipeline.data_LL_gauss(data_good[T_train:T_train+L_max], predictions,1.)
		LL_list.append(pred_LL)

		if True:
			plt.figure()
			spec,f = M.spectrum(1./srate)
			plt.loglog(f,spec)
			plt.figure()
			plt.plot(range(L_max),data_good[T_train:T_train+L_max])
			plt.plot(range(L_max),np.median(predictions, axis = 0))
			plt.show()

	var_list = np.array(var_list).T
	LL_list = np.array(LL_list).T
	np.savetxt('var.dat', var_list, header = 'Raw data\np = {}'.format(p_list))
	np.savetxt('LL.dat', var_list, header = 'Raw data\np = {}'.format(p_list))
else:
	var_list = np.loadtxt('var_rawdata.dat')
	LL_list = np.loadtxt('LL.dat')
	try:
		var_data = np.loadtxt('var_data.dat')
	except:
		var_data = np.ones((3,))
	var_list /= var_data[0]

print(var_list.shape)

	#plotting
plt.figure()
for i, p_ in enumerate(p_list):
	plt.plot(range(L_max),var_list[:,i], label = 'p = {}'.format(p_))
plt.legend()
plt.figure()
for i, p_ in enumerate(p_list):
	plt.plot(range(L_max),LL_list[:,i], label = 'p = {}'.format(p_))
plt.legend()
plt.show()






	

