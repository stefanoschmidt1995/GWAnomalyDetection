#Import standard libraries

import torch
import torch.nn as nn
from glue import datafind
from gwpy.timeseries import TimeSeries
import numpy as np
from  sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')

#Import built functions
import useful_funcs as uf
import network as nw

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Data sets
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

#Call a dummy data set and re-scales it between (-1,1)
data=np.array(TimeSeries.read(source=path+IDs[0], channel='L1:GWOSC-16KHZ_R1_STRAIN'))[0:130000]; data=data.reshape(-1, 1)
data=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(data)

#We split the data in time windows. CNN accept the data as [batch_size, channels, time_steps] 

window=900 #size measured in time steps
channel= 1 #number of detectors

train=data[:110000]; valid=data[110000:120000]; test=data[120000:]

X, y = uf.split_sequence(train, window); training = uf.dataset(X.reshape(X.shape[0],1, X.shape[1]),y)
X_valid, y_valid= uf.split_sequence(valid, window); 
validation = uf.dataset(X_valid.reshape(X_valid.shape[0],1,X_valid.shape[1]),y_valid) #to be fed into the data 
X_test, y_test= uf.split_sequence(test, window)

print(X_test.shape, y_test.shape)



# Parameters
epochs=50; batch=200; kernel=100
params = {'shuffle': False,'num_workers': 6}

model=nw.CNN(input_channel=1, kernel_size=kernel, batch_size=batch, window=window).to(device) #send to GPU
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

uf.learning_process(epochs,batch, training, validation, optimizer,model)
pred = uf.predicting(local_batch=X_test[0], local_labels= y_test[0], timesteps=900, model=model)

#Plot predictions and reality. We use the 1st window (i.e. input is 1000 time steps), so we start predicting at time step 1000. Thus, we need to compare with the next 1000 points (input[window])
plt.plot(np.arange(0, 0+900), X_test[0], label='Input')
plt.plot(np.arange(window, window+900), pred, label='Prediction')
plt.plot(np.arange(window, window+900), X_test[window],label='Real')
plt.plot(np.arange(window, window+900), (pred-X_test[window])**2, label='MSE')
plt.legend(loc='best'); plt.title('Real data vs predicted'); plt.xlabel('Time steps'); plt.ylabel('Amplitude')
plt.savefig('/home/melissa.lopez/Anomaly_Detection/Algorithm/dummy.png');plt.show()