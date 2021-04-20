import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from scipy.stats import wasserstein_distance
import network as nw

from gwpy.signal import filter_design
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gwpy.plot import Plot
from gwpy.signal.filter_design import bandpass

from pycbc.filter import resample_to_delta_t

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def torch_wasserstein_distance(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(_torch_cdf_distance(tensor_a,tensor_b))


def _torch_cdf_distance(tensor_a,tensor_b):
    """
        Torch implementation of _cdf_distance for Wasserstein distance

    input: tensor_a, tensor_b
    output: cdf_loss which the computed distance between the tensors.
    
    #Note: this function yields an difference of \approx 10^-9
    """
    
    #It is necessary to reshape the tensors to match the dimensions of Scipy.
    tensor_a=torch.reshape(tensor_a, (1, tensor_a.shape[0]))
    tensor_b=torch.reshape(tensor_b, (1, tensor_b.shape[0]))
    
    # Creater sorters:
    sorter_a=torch.argsort(tensor_a);   sorter_b=torch.argsort(tensor_a)
    
    # We append both tensors and sort them
    all_values=torch.cat((tensor_a, tensor_b), dim=1)
    all_values, idx = torch.sort(all_values, dim=1); 
    
    # Calculate the n-th discrete difference along the given axis (equivalent to np.diff())
    deltas=  all_values[0, 1:] - all_values[0, :-1];
    
    sorted_a, idx = torch.sort(tensor_a, dim=1); 
    sorted_b, idx = torch.sort(tensor_b, dim=1); 
    
    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    a_cdf_index= torch.searchsorted(sorted_a.flatten(), all_values[0,:-1],right=True);
    b_cdf_index= torch.searchsorted(sorted_b.flatten(), all_values[0,:-1],right=True);
    
    #Compute the cdf
    a_cdf = a_cdf_index/tensor_a.shape[1]; 
    b_cdf = b_cdf_index/tensor_b.shape[1];
    
    #And the distance between them
    cdf_distance = torch.sum(torch.mul(torch.abs((a_cdf-b_cdf)), deltas),dim=-1)
    
    cdf_loss = cdf_distance.mean()
    
    return cdf_loss


def split_sequence(sequence, n_steps):
    """
        This function splits the data in time  windows.

    sequence: data
    n_steps: window size
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



class dataset(torch.utils.data.Dataset):
    """
        This is employed to call torch data.loader 
    """
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label
    
def learning_process(epochs, batch_size, train, valid, optimizer, model):
    """
        In this function we load the data (train, valid). Call the model (optimizer, model, batch_size) and iterate through # of epochs. Finally, we plot our results.
        
    epochs: # epochs
    batch_size: examples per batch
    train and valid: inputs for the train_generator and valid_generator
    optimizer: optimizer for the network
    model: model predefined in network.py
    """
    
    #Call data loader for (train, valid)
    train_generator = torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=False)
    valid_generator = torch.utils.data.DataLoader(valid,batch_size=batch_size,shuffle=False)
    
    #Here we define the loss function and the MAE
    MSE = nn.MSELoss()
    MAE=nn.L1Loss()
    
    
    #To store the loss and mae
    loss_train = []; mae_train=[];mse_train=[]
    loss_val=[]; mae_val=[];mse_val=[]
    # Loop over epochs

    for epoch in range(epochs):

        # Training
        t0 = time.time(); #to compute the time per training epoch

        #Define counters to store the loss and MSE
        mae_list=[]; loss_list=[]; mse_list=[]; wd_list=[]; WD_list=[]
        for idx, (local_batch,local_labels) in enumerate(train_generator):

            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            optimizer.zero_grad()

            # Run the forward pass
            outputs = model(local_batch.float()); 
            outputs = torch.reshape(outputs, (outputs.shape[0],1))
            local_labels= torch.reshape(local_labels.float(), (local_labels.float().shape[0],1))#to go from [batch] to [batch,1]
            loss = torch_wasserstein_distance(outputs, local_labels) 
            loss_list.append(loss.item()); #store loss
            
            #Backpropagation + optimizer step
          
            loss.backward(); 
            optimizer.step()

            mae= MAE(outputs, local_labels)
            mse = MSE(outputs, local_labels)

            WD =wasserstein_distance(outputs.data.cpu().numpy().flatten(), local_labels.data.cpu().numpy().flatten())
            
            mae_list.append(mae.item()); mse_list.append(mse.item());WD_list.append(WD.item());
        loss_train.append(np.mean(loss_list)); mae_train.append(np.mean(mae_list));mse_train.append(np.mean(mse_list))
        print('TRAIN: epoch '+str(epoch)+', loss= '+str(np.mean(loss_list))+' , mse ='+str(np.mean(mse_list))+', mae ='+str(np.mean(mae_list))+', WD='+str(np.mean(WD_list))+',  {} s/epoch'.format(time.time() - t0))


        # Validation
        t0 = time.time(); #to compute the time per training epoch 

        #Define counters to store the loss and MSE
        mae_list=[]; loss_list=[]; mse_list=[]
        for idx, (local_batch,local_labels) in enumerate(valid_generator):
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                # Run the forward pass
                outputs = model(local_batch.float())
                outputs = torch.reshape(outputs, (outputs.shape[0],1)) #to go from [batch] to [batch,1]
                local_labels= torch.reshape(local_labels.float(), (local_labels.float().shape[0],1))
                loss = torch_wasserstein_distance(outputs, local_labels) 
                
                loss_list.append(loss.item()); #store loss

                mae= MAE(outputs, local_labels)
                mse= MSE(outputs, local_labels)
                
                mae_list.append(mae.item()); mse_list.append(mse.item())

        loss_val.append(np.mean(loss_list)); mae_val.append(np.mean(mae_list));mse_val.append(np.mean(mse_list))
        print('Validate: epoch '+str(epoch)+', loss= '+str(np.mean(loss_list))+' , mse ='+str(np.mean(mse_list))+', mae='+str(np.mean(mae_list))+',   {} s/epoch'.format(time.time() - t0))
        print(' ')
        
    #We store the weights
    torch.save(model.state_dict(), '/home/melissa.lopez/Anomaly_Detection/Weights/network1.pth')

    #And we plot some results
    plt.plot(np.arange(epochs), loss_train, color='cornflowerblue',label='Loss (WD)')
    plt.plot(np.arange(epochs), loss_val, color='darkorange')
    
    plt.plot(np.arange(epochs), mae_train, color='cornflowerblue',linestyle='dashed', label='MAE')
    plt.plot(np.arange(epochs), mae_val,color='darkorange',linestyle='dashed')
    
    plt.plot(np.arange(epochs), mse_train, color='cornflowerblue',linestyle='dotted', label='MSE')
    plt.plot(np.arange(epochs), mse_val,color='darkorange', linestyle='dotted' )
    
    leg =  plt.legend(loc='best', handlelength=3)
    # get the lines and texts inside legend box
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=1)
    plt.title('Training and validation statistics'); plt.xlabel('Epochs'); 
    plt.savefig('/home/melissa.lopez/Anomaly_Detection/Algorithm/stats.png');plt.close()
        

def predicting(local_batch, local_labels, timesteps, model):
    
    """
        In this function we load the test data (local_batch, local_labels) and predict # of timesteps for a given model
        
    local_batch, local_labels: testing input and target
    timesteps: # of points to predict
    model: model predefined in network.py
    """
    
    #We send the input and the target from numpy --> torch --> gpu
    local_batch, local_labels = torch.from_numpy(local_batch).to(device), torch.tensor(local_labels).to(device)
    
    t0 = time.time(); #to measure the time for prediction
    x=local_batch; prediction=torch.empty(0,1).to(device)
    for j in range(timesteps): 
        
        out = model(torch.reshape(x, (1,1,x.shape[0])).float()) #predict
        x = torch.cat([x[1:],torch.flatten(out)]) #drop 1st value of the input and add at the end the prediction
        prediction=torch.cat([prediction,out]) #tensor of all predictions
    
    print('Test:  {} s/epoch'.format(time.time() - t0))  
    return prediction.data.cpu().numpy() #get the tensor data --> cpu --> numpy


def whiten(data, ffttime, window, low_f,  high_f,notch, rate):
    """
    This function whitens the data and band-pass it in the range [low_f,  high_f].
    
    Parameters
    ----------
    
    data: numpy array
        The signal to whiten as numpy array
        
    ffttime: int
        Portion of the strain to compute the psd
    
    window: str
        Type of function for the windowing
        
    low_f: int
        Lower bound of the band-pass filter
        
    high_f: int 
        Upper bound of the band-pass filter
    
    notch: list
        Frequencies of the notch filters. Depends on the detector
        
    rate: int
        Resampling rate. Represents the sampling frequency
        
    Returns
    -------
      
    whitened: numpy array
        The whitened and band-passed numpy array
  
    """
    
    # Band-pass filter in [35, 250]
    bp = bandpass(float(low_f), float(high_f), data.sample_rate)

    #Notches for the 1st three harminics of the 60 Hz AC
    notches = [filter_design.notch(line, data.sample_rate) for line in notch]

    #Concatenate both filters
    zpk = filter_design.concatenate_zpks(bp, *notches)

    #Whiten and band-pass filter
    white = data.whiten(ffttime,int(ffttime/2), window='hann') #whiten the data
    white_down = white.filter(zpk, filtfilt=True).resample(rate=rate, window='hann') #downsample to 2048Hz
    whitened = np.array(white_down) 
    
    #Plot version with and without notches
    plot = Plot(figsize=(15, 6)); ax = plot.gca()
    ax.plot(white_down , label='Downsampled', alpha=0.7) 
    ax.plot(white.filter(zpk, filtfilt=True), label='Not downsampled', alpha=0.7)
    ax.set_xscale('auto-gps')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('LIGO-Livingston strain data whitened, band-passed in range ['+str(low_f)+''+str(high_f)+'] $Hz$')
    plot.legend();plt.savefig('/home/melissa.lopez/Anomaly_Detection/Algorithm/dummy.png');plt.close()
    
    return whitened

