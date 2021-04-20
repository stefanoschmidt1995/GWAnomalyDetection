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
        self.conv1 = torch.nn.Conv1d(in_channels=input_channel, out_channels=32, dilation=2, kernel_size=self.kernel_size, stride = 1,bias=True)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=16, dilation=4, kernel_size=self.kernel_size, stride = 1,bias=True)
        self.conv3 = torch.nn.Conv1d(in_channels=16, out_channels=8, dilation=8, kernel_size=self.kernel_size, stride = 1,bias=True)
        self.conv4 = torch.nn.Conv1d(in_channels=8, out_channels=16, dilation=12, kernel_size=self.kernel_size, stride = 1,bias=True)
        
        #Define LSTM
        self.lstm=nn.LSTM(input_size=window-(self.kernel_size-1)*(2+4+8+12), hidden_size=50, batch_first=True,num_layers=2)
        
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
        
        x = self.conv4(x) #input 10, output 10
        x = self.relu(x)

        x, (h_n, h_c) = self.lstm(x)
        x = self.linear(x[:, -1, :])
        x = self.tanh(x)
    
        return x
    
    
#NOTE: If you are using batch_first=True in your LSTM, the input shape should be [batch_size, seq_len, nb_features].



class ModuleWavenet(nn.Module):
    def __init__(self,input_channel, dilation):
        """
        This block is the Wavenet Module, where the magic happens.
        For implementation details visit: 
        https://medium.com/@kion.kim/wavenet-a-network-good-to-know-7caaae735435
        
        Parameters
        ----------
        input_channel: int
            depth of the previous block
        dilation: int
            amount of dilation to convolve "with holes"
        
        Output after forward pass:
        --------------------------
            x_res: torch array
                Residual output to feed the next module
            x_skip: torch array
                Skip connection to compute the loss
        
        """
        super(ModuleWavenet, self).__init__()
        self.input=input_channel
        self.dilation=dilation
        
        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()
        
        #Define convolutions
        self.conv_dilated = torch.nn.Conv1d(in_channels=input_channel, out_channels=10, kernel_size=2, stride=1,
                                     dilation=dilation,padding=0,  bias=False)  
        self.conv_1x1 = torch.nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride = 1,bias=True)
        
    def forward(self, x):
       
        x_old=x
        x = self.conv_dilated(x) 
        #Pixel gate
        x = self.gate_tanh(x)*self.gate_sigmoid(x) 
        #1x1 convolution
        x_skip = self.conv_1x1(x)
        #Skip-connection
        x_res = x_skip+ x_old[:, :, -x_skip.size(2):] #we cut x_old to have the same shape as x_skip
        return x_res,x_skip
    
class Wavenet(nn.Module):
    def __init__(self,input_channel, dilation):
        """
        Wavenet inspired architecture to match our input size (1000 ts)
        
        Parameters
        ----------
        input_channel: int
            depth of the previous block
        dilation: int
            amount of dilation to convolve "with holes"
        
        Output after forward pass:
        --------------------------
            xs: float
                A single predicted data point
        """
        super(Wavenet, self).__init__()
        
        self.input_channel=input_channel
        self.dilation=dilation
        
        #Define convolutions
        self.Mod1= ModuleWavenet(self.input_channel, dilation)
        self.Mod2= ModuleWavenet(self.input_channel, dilation**2)
        self.Mod3= ModuleWavenet(self.input_channel, dilation**3)
        self.Mod4= ModuleWavenet(self.input_channel, dilation**4)
        self.Mod5= ModuleWavenet(self.input_channel, dilation**5)
        self.Mod6= ModuleWavenet(self.input_channel, dilation**6)
        self.Mod7= ModuleWavenet(self.input_channel, dilation**7)
        self.Mod8= ModuleWavenet(self.input_channel, dilation**8)
        
        
        self.conv_1x1 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride = 1,bias=True)

        self.linear = nn.Linear(42, 1)
        
    def forward(self, x):

        x,xs1 = self.Mod1(x)
        x,xs2 = self.Mod2(x)
        x,xs3 = self.Mod3(x)
        x,xs4 = self.Mod4(x)
        x,xs5 = self.Mod5(x)
        x,xs6_a = self.Mod6(x)
        x,xs6_b = self.Mod6(x)
        x,xs7_a = self.Mod7(x)
        x,xs7_b = self.Mod7(x)
        x,xs8_a = self.Mod8(x)
        x,xs8_b = self.Mod8(x)
        
        xs=xs8_b
        for i in [xs1, xs2, xs3, xs4, xs5, xs6_a, xs6_b, xs7_a,xs7_b,xs8_a]:
            xs= xs+ i[:, :, -xs.size(2):]
        
        xs=self.conv_1x1(xs)
        xs=self.linear(xs)
        xs = torch.reshape(xs, (xs.shape[0],1)).float()

        return xs
    
