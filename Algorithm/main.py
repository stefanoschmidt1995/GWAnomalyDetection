#Import standard libraries

import torch
import torch.nn as nn
from glue import datafind
from gwpy.timeseries import TimeSeries
from gwpy.signal.filter_design import bandpass
import numpy as np
from scipy.stats import wasserstein_distance
from  sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')

#Initializing Matlab

import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('/home/melissa.lopez/Anomaly_Detection/Matlab_functions')

import sklearn.preprocessing as pp
from gwpy.signal.filter_design import bandpass
import pipeline_helper
from pipeline_helper import do_emd, downsample_data
#Import built functions
import useful_funcs as uf
import network as nw

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Alex's data sets from O2
n=1; #number of data samples. max 32
path='/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.16k/frame.v1/L1/'
IDs=[
'/1164967936/L-L1_GWOSC_O2_16KHZ_R1-1165631488-4096.gwf',
'/1167065088/L-L1_GWOSC_O2_16KHZ_R1-1167601664-4096.gwf',
'/1168113664/L-L1_GWOSC_O2_16KHZ_R1-1168973824-4096.gwf',
'/1169162240/L-L1_GWOSC_O2_16KHZ_R1-1169715200-4096.gwf',
'/1169162240/L-L1_GWOSC_O2_16KHZ_R1-1169723392-4096.gwf',
'/1171259392/L-L1_GWOSC_O2_16KHZ_R1-1172148224-4096.gwf',
'/1172307968/L-L1_GWOSC_O2_16KHZ_R1-1172307968-4096.gwf',
'/1172307968/L-L1_GWOSC_O2_16KHZ_R1-1172525056-4096.gwf',
'/1174405120/L-L1_GWOSC_O2_16KHZ_R1-1174970368-4096.gwf',
'/1174405120/L-L1_GWOSC_O2_16KHZ_R1-1174978560-4096.gwf',
'/1174405120/L-L1_GWOSC_O2_16KHZ_R1-1174982656-4096.gwf',
'/1175453696/L-L1_GWOSC_O2_16KHZ_R1-1175470080-4096.gwf',
'/1175453696/L-L1_GWOSC_O2_16KHZ_R1-1175601152-4096.gwf',
'/1175453696/L-L1_GWOSC_O2_16KHZ_R1-1175986176-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176522752-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176526848-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176612864-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176616960-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176633344-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176645632-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176907776-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1176915968-4096.gwf',
'/1176502272/L-L1_GWOSC_O2_16KHZ_R1-1177387008-4096.gwf',
'/1183842304/L-L1_GWOSC_O2_16KHZ_R1-1184522240-4096.gwf',
'/1184890880/L-L1_GWOSC_O2_16KHZ_R1-1185259520-4096.gwf',
'/1184890880/L-L1_GWOSC_O2_16KHZ_R1-1185275904-4096.gwf',
'/1185939456/L-L1_GWOSC_O2_16KHZ_R1-1186291712-4096.gwf',
'/1185939456/L-L1_GWOSC_O2_16KHZ_R1-1186295808-4096.gwf',
'/1185939456/L-L1_GWOSC_O2_16KHZ_R1-1186299904-4096.gwf',
'/1185939456/L-L1_GWOSC_O2_16KHZ_R1-1186467840-4096.gwf',
'/1185939456/L-L1_GWOSC_O2_16KHZ_R1-1186480128-4096.gwf',
'/1185939456/L-L1_GWOSC_O2_16KHZ_R1-1186955264-4096.gwf',
'/1186988032/L-L1_GWOSC_O2_16KHZ_R1-1187651584-4096.gwf']

#Call a dummy data set
data = TimeSeries.read(source=path+IDs[0], channel='L1:GWOSC-16KHZ_R1_STRAIN'); 
data = pp.MinMaxScaler(feature_range=(-1,1)).fit_transform(np.array(data).reshape(-1,1)).flatten();
#The data is normalized to downsample it.
srate, data_down =pipeline_helper.downsample_data(downsampling_factor=16, srate=16384, data=data, times = None, WF = None)

#We normalize the downsampled data to decompose it (?)
data_down = pp.MinMaxScaler(feature_range=(-1,1)).fit_transform(np.array(data_down).reshape(-1,1)).flatten(); print(data_down.shape)

# Decomposition methods: SSD & EMD (choose your fighter)
#data=eng.SSD(matlab.single(data_down[0:200000].tolist()), eng.single(srate),eng.single(0.005))[0]; 
data=pipeline_helper.do_emd(data_down[0:200000], emd_type = 'PyEMD',trim_borders = 20)[0][:,0]

#Rescaling the data to feed into the neural network
data = pp.MinMaxScaler(feature_range=(-1,1)).fit_transform(np.array(data).reshape(-1,1)).flatten()

# Parameters
epochs=100; batch=50; kernel=20
params = {'shuffle': False,'num_workers': 1}
window=1000 #size measured in time steps (around 1s)
channel= 1 #number of strains

#Split data
#We split the data in time windows. CNN accept the data as [batch_size, channels, time_steps] 
train=data[:180000]; valid=data[180000:190000]; test=data[190000:]; 

#We slit the data to load in batches (uf.slit_sequence) and we load the in dataloader (uf.dataset)
X, y = uf.split_sequence(train, window); 
training = uf.dataset(X.reshape(X.shape[0],1, X.shape[1]),y)

X_valid, y_valid= uf.split_sequence(valid, window); 
validation = uf.dataset(X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1]),y_valid) 

X_test, y_test= uf.split_sequence(test, window); print(X_test.shape, y_test.shape); 
y_test= np.asarray(y_test, dtype=np.float32); 

#Call model and predict
#model=nw.CNN(input_channel=1, kernel_size=kernel, batch_size=batch, window=window).to(device) #send to GPU
model=nw.Wavenet(input_channel=1, dilation=2).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
uf.learning_process(epochs,batch, training, validation, optimizer,model) #training and validation
#Note: at the end of the training process we plot the stats, so change the path of the plot.

pred = uf.predicting(local_batch=X_test[0], local_labels=y_test[0], timesteps=window, model=model) #test

print(pred.flatten().shape, X_test[window].shape, wasserstein_distance(pred.flatten(), np.array(X_test[window])))

#Plot predictions and reality. We use the 1st window (i.e. input is 1000 time steps), so we start predicting at time step 1000. Thus, we need to compare with the next 1000 points (input[window])

plt.plot(np.arange(0, 0+window), X_test[0], label='Input')
plt.plot(np.arange(window, window+window), pred, label='Prediction')
plt.plot(np.arange(window, window+window), X_test[window],label='Real', alpha=0.5)
plt.legend(loc='best'); plt.title('Real data vs predicted'); plt.xlabel('Time steps'); plt.ylabel('Amplitude')
plt.savefig('/home/melissa.lopez/Anomaly_Detection/Algorithm/dummy.png');plt.show()
