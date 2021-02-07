import torch
import torch.nn as nn

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True  

class CNN(nn.Module):
    def __init__(self,input_channel, kernel_size, batch_size,window):
        """
        This block is to convolve the input and extract the principal features of the time series.
        
        input_channel: depth of the previous block
        kernel_size: varying size of kernel
        
        """
        super(CNN, self).__init__()
        self.kernel_size=kernel_size
        self.batch_size=batch_size
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        
        #Define convolutions
        self.conv1 = torch.nn.Conv1d(in_channels=input_channel, out_channels=32, kernel_size=self.kernel_size, stride = 1,bias=True)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=self.kernel_size, stride = 1,bias=True)
        self.conv3 = torch.nn.Conv1d(in_channels=16, out_channels=8, kernel_size=self.kernel_size, stride = 1,bias=True)
        
        #Define LSTM
        self.lstm=nn.LSTM(input_size=window-(self.kernel_size-1)*3, hidden_size=50, batch_first=True,num_layers=2)
        
        # Define the output layer
        self.linear = nn.Linear(50, 1)
        
    def forward(self, x):
        """
        Convolve input with 3 layes
        """
        x = self.conv1(x) #input ?, output 10
        x = self.relu(x) 

        x = self.conv2(x) #input 10, output 10
        x = self.relu(x)

        x = self.conv3(x) #input 10, output 10
        x = self.relu(x) 
       
        x, (h_n, h_c) = self.lstm(x)
        x = self.linear(x[:, -1, :])
        x = self.tanh(x)
        
        return x
    
    
#NOTE: If you are using batch_first=True in your LSTM, the input shape should be [batch_size, seq_len, nb_features].