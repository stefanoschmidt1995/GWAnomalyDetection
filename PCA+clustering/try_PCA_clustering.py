import numpy as np
import matplotlib.pyplot as plt

import h5py
from sklearn.decomposition import PCA
from mlgw.ML_routines import PCA_model
from itertools import combinations
from pathlib import Path
import os
import sys

########################

#The aux files are obtained with:
	#ln -s  /home/robin.vanderlaag/wp5/fractal_dims/* ./data

#To run everything that's in data in a single shot:
	#for a in $(ls data); do python try_PCA_clustering.py data/$a; done

#######################

if len(sys.argv)>1:
	filename = sys.argv[1]
else:
	#filename = 'data/aux_clean-event.hdf5'
	filename = 'data/aux_dq-issues-3.hdf5'
plotfolder = 'plot_{}'.format(Path(filename).stem)
if not os.path.isdir(plotfolder): os.mkdir(plotfolder)

print("Loading file {}".format(filename))

f = h5py.File(filename, 'r')

fd_array = [] #fractal dimension array

for k in f['data'].keys():
	fd_array.append(np.array(f['data'][k]))

fd_array = np.column_stack(fd_array) #(N_timebins, N_channels)

print("Created feature array with shape {}".format(fd_array.shape))

	#Separating train and test
train_frac = 0.8
N_train = int(fd_array.shape[0]*train_frac)
fd_array_train = fd_array[:N_train,:]
fd_array_test = fd_array[N_train:,:]

	#Doing PCA (with mlgw for my ease, that's bad)

for K_PCA in [3]:#, 5, 10, 20, 50, 100]:
	PCA_fd = PCA_model()
	PCA_fd.fit_model(fd_array_train, K_PCA, scale_PC = True)
	fd_array_test_rec = PCA_fd.reconstruct_data(PCA_fd.reduce_data(fd_array_test))
	mse = np.mean(np.square(fd_array_test_rec-fd_array_test).flatten())

	print("components | mse: {} \t {}".format(K_PCA, mse))

	#Doing some nice plots :)
K_PCA = 10
PCA_fd = PCA_model()
PCA_fd.fit_model(fd_array_train, K_PCA, scale_PC = True)
fd_array_red = PCA_fd.reduce_data(fd_array_test) #(N, K_PCA)

for k1, k2 in combinations(range(K_PCA),2):
	plt.figure()
	plt.scatter(*fd_array_red[:,[k1,k2]].T)

	plt.savefig('{}/scatterplot_PCA_K{}{}.png'.format(plotfolder, k1,k2))
	plt.close('all')






















