import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')

import network as nw

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

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
        In this function we load the data (train, valid). Call the model (optimizer, model, batch_size) and iterate through # of epochs.
        
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
    criterion = nn.MSELoss()
    MAE=nn.L1Loss()
    
    
    #To store the loss and mae
    loss_train = []; mae_train=[]
    loss_val=[]; mae_val=[]
    # Loop over epochs

    for epoch in range(epochs):

        # Training
        t0 = time.time(); #to compute the time per training epoch

        #Define counters to store the loss and MSE
        mae_list=[]; loss_list=[]
        for idx, (local_batch,local_labels) in enumerate(train_generator):

            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            optimizer.zero_grad()

            # Run the forward pass
            outputs = model(local_batch.float());
            outputs = torch.reshape(outputs, (outputs.shape[0],1)) #to go from [batch] to [batch,1]
       
            loss = criterion(outputs, local_labels.float())
            loss_list.append(loss.item()); #store loss

            #Backpropagation + optimizer step
            loss.backward()
            optimizer.step()

            mae= MAE(outputs, local_labels.float())
            mae_list.append(mae.item())
        loss_train.append(np.mean(loss_list)); mae_train.append(np.mean(mae_list));
        print('TRAIN: epoch '+str(epoch)+', loss= '+str(np.mean(loss_list))+' , mae ='+str(np.mean(mae_list))+',  {} s/epoch'.format(time.time() - t0))


        # Validation
        t0 = time.time(); #to compute the time per training epoch 

        #Define counters to store the loss and MSE
        mae_list=[]; loss_list=[]
        for idx, (local_batch,local_labels) in enumerate(valid_generator):
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                # Run the forward pass
                outputs = model(local_batch.float())
                outputs = torch.reshape(outputs, (outputs.shape[0],1)) #to go from [batch] to [batch,1]
                loss = criterion(outputs, local_labels.float())
                loss_list.append(loss.item()); #store loss

                mae= MAE(outputs, local_labels.float())
                mae_list.append(mae.item())

        loss_val.append(np.mean(loss_list)); mae_val.append(np.mean(mae_list))
        print('Validate: epoch '+str(epoch)+', loss= '+str(np.mean(loss_list))+' , mae ='+str(np.mean(mae_list))+',  {} s/epoch'.format(time.time() - t0))
        
    plt.plot(np.arange(epochs), loss_train, label='Training')
    plt.plot(np.arange(epochs), loss_val, label='Validation')
    plt.legend(loc='best'); plt.title('Losses'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.savefig('/home/melissa.lopez/Anomaly_Detection/Algorithm/losses.png');plt.close()
    
    plt.plot(np.arange(epochs), mae_train, label='Training')
    plt.plot(np.arange(epochs), mae_val, label='Validation')
    plt.legend(loc='best'); plt.title('Mean absolute error (MAE)'); plt.xlabel('Epochs'); plt.ylabel('MAE')
    plt.savefig('/home/melissa.lopez/Anomaly_Detection/Algorithm/mae.png');plt.close()
        

def predicting(local_batch, local_labels, timesteps, model):
    
    """
        In this function we load the test data (local_batch, local_labels) and predict # of timesteps for a given model
        
    local_batch, local_labels: testing input and target
    timesteps: # of points to predict
    model: model predefined in network.py
    """
    
    #We send the input and the target from numpy --> torch --> gpu
    local_batch, local_labels = torch.from_numpy(local_batch).to(device), torch.from_numpy(local_labels).to(device)
    
    t0 = time.time(); #to measure the time for prediction
    x=local_batch; prediction=torch.empty(0,1).to(device)
    for j in range(timesteps): 
        
        out = model(torch.reshape(x, (1,1,x.shape[0])).float()) #predict
        x = torch.cat([x[1:],out]) #drop 1st value of the input and add at the end the prediction
        prediction=torch.cat([prediction,out]) #tensor of all predictions
    
    print('Test:  {} s/epoch'.format(time.time() - t0))  
    return prediction.data.cpu().numpy() #get the tensor data --> cpu --> numpy
    