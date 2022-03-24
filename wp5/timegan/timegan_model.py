import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import torch
import torch.nn as nn

# dict of loss functions that can be used by network, more can be added
_LOSS_FUNCS = {
    'mse': nn.MSELoss(),
    'mae': nn.L1Loss(),
    'bcewithlogits': nn.BCEWithLogitsLoss()
}

class Embedder(nn.Module):
    def __init__(self, parameters):
        super(Embedder, self).__init__()
        self.hidden_dim = parameters.get('hidden_dim')
        self.feature_dim = parameters.get('feature_dim')
        self.num_layers = parameters.get('num_layers')
        self.module_name = parameters.get('module', 'gru')
        self.max_seq_len = parameters.get('max_seq_len')
        self.padding_value = parameters.get('padding_value', -10)
        self.weights_like_tf = parameters.get('weights_like_tf', False)
        
        loss_fn = parameters.get('loss_fn_ers', 'mse').lower()
        try:
            self.loss = _LOSS_FUNCS[loss_fn]
        except KeyError:
            print(f'No loss function implimented for key: "{loss_fn}", supported loss functions are:\n\t{list(_LOSS_FUNCS.keys())}')
            print(f'Defaulting to MSE loss.')
            self.loss = _LOSS_FUNCS['mse']
        
        if self.module_name.lower() == 'gru':
            self.emb_rnn = nn.GRU(input_size=self.feature_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers,
                                  batch_first=True)
        elif self.module_name.lower() == 'lstm':
            self.emb_rnn = nn.LSTM(input_size=self.feature_dim,
                                   hidden_size=self.hidden_dim,
                                   num_layers=self.num_layers,
                                   batch_first=True)
        else:
            raise ValueError('"module" should be "gru" or "lstm"')
        
        self.emb_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = nn.Sigmoid()
        
        if self.weights_like_tf:
            # https://github.com/d9n13lt4n/timegan-pytorch/blob/main/models/timegan.py
            with torch.no_grad():
                for name, param in self.emb_rnn.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(1)
                    elif 'bias_hh' in name:
                        param.data.fill_(0)
                for name, param in self.emb_linear.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
    def forward(self, X, T):
        # ignore padding
        X_packed = nn.utils.rnn.pack_padded_sequence(input=X,
                                                     lengths=T,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        H_o, H_t = self.emb_rnn(X_packed)
        # repad
        H_o, T = nn.utils.rnn.pad_packed_sequence(sequence=H_o, 
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.max_seq_len)
        
        logits = self.emb_linear(H_o)
        H = self.emb_sigmoid(logits)
        return H

class Recovery(nn.Module):
    def __init__(self, parameters):
        super(Recovery, self).__init__()
        self.hidden_dim = parameters.get('hidden_dim')
        self.feature_dim = parameters.get('feature_dim')
        self.num_layers = parameters.get('num_layers')
        self.module_name = parameters.get('module', 'gru')
        self.max_seq_len = parameters.get('max_seq_len')
        self.padding_value = parameters.get('padding_value', -10)
        self.weights_like_tf = parameters.get('weights_like_tf', False)
        
        loss_fn = parameters.get('loss_fn_ers', 'mse').lower()
        try:
            self.loss = _LOSS_FUNCS[loss_fn]
        except KeyError:
            print(f'No loss function implimented for key: "{loss_fn}", supported loss functions are:\n\t{list(_LOSS_FUNCS.keys())}')
            print(f'Defaulting to MSE loss.')
            self.loss = _LOSS_FUNCS['mse']
        
        if self.module_name.lower() == 'gru':
            self.rec_rnn = nn.GRU(input_size=self.hidden_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers,
                                  batch_first=True)
        elif self.module_name.lower() == 'lstm':
            self.rec_rnn = nn.LSTM(input_size=self.hidden_dim,
                                   hidden_size=self.hidden_dim,
                                   num_layers=self.num_layers,
                                   batch_first=True)
        else:
            raise ValueError('"module" should be "gru" or "lstm"')
        
        self.rec_linear = nn.Linear(self.hidden_dim, self.feature_dim)
        self.rec_sigmoid = nn.Sigmoid()
        
        if self.weights_like_tf:
            # https://github.com/d9n13lt4n/timegan-pytorch/blob/main/models/timegan.py
            with torch.no_grad():
                for name, param in self.rec_rnn.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(1)
                    elif 'bias_hh' in name:
                        param.data.fill_(0)
                for name, param in self.rec_linear.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
    def forward(self, H, T):
        # ignore padding
        H_packed = nn.utils.rnn.pack_padded_sequence(input=H,
                                                     lengths=T,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        
        H_o, H_t = self.rec_rnn(H_packed)
        
        # repad
        H_o, T = nn.utils.rnn.pad_packed_sequence(sequence=H_o, 
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.max_seq_len)
        
        X_tilde = self.rec_linear(H_o)
        
        # Original TimeGAN code uses sigmoid here, is this required?
        return self.rec_sigmoid(X_tilde) # X_tilde
    
class Generator(nn.Module):
    def __init__(self, parameters):
        super(Generator, self).__init__()
        self.z_dim = parameters.get('z_dim')
        self.hidden_dim = parameters.get('hidden_dim')
        self.num_layers = parameters.get('num_layers')
        self.module_name = parameters.get('module', 'gru')
        self.max_seq_len = parameters.get('max_seq_len')
        self.padding_value = parameters.get('padding_value', -10)
        self.weights_like_tf = parameters.get('weights_like_tf', False)
        
        loss_fn = parameters.get('loss_fn_gd', 'bcewithlogits').lower()
        try:
            self.loss = _LOSS_FUNCS[loss_fn]
        except KeyError:
            print(f'No loss function implimented for key: "{loss_fn}", supported loss functions are:\n\t{list(_LOSS_FUNCS.keys())}')
            print(f'Defaulting to BCEWithLogits Loss.')
            self.loss = _LOSS_FUNCS['bcewithlogits']
        
        if self.module_name.lower() == 'gru':
            self.gen_rnn = nn.GRU(input_size=self.z_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers,
                                  batch_first=True)
        elif self.module_name.lower() == 'lstm':
            self.gen_rnn = nn.LSTM(input_size=self.z_dim,
                                   hidden_size=self.hidden_dim,
                                   num_layers=self.num_layers,
                                   batch_first=True)
        else:
            raise ValueError('"module" should be "gru" or "lstm"')
        
        self.gen_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = nn.Sigmoid() # why sigmoid ?
        
        if self.weights_like_tf:
            # https://github.com/d9n13lt4n/timegan-pytorch/blob/main/models/timegan.py
            with torch.no_grad():
                for name, param in self.gen_rnn.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(1)
                    elif 'bias_hh' in name:
                        param.data.fill_(0)
                for name, param in self.gen_linear.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
    def forward(self, Z, T):
        # ignore padding
        Z_packed = nn.utils.rnn.pack_padded_sequence(input=Z, 
                                                     lengths=T, 
                                                     batch_first=True, 
                                                     enforce_sorted=False)
        
        H_o, H_t = self.gen_rnn(Z_packed)
        # repad
        H_o, T = nn.utils.rnn.pad_packed_sequence(sequence=H_o, 
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.max_seq_len)
        
        logits = self.gen_linear(H_o)
        H = self.gen_sigmoid(logits)
        
        return H
    
class Supervisor(nn.Module):
    def __init__(self, parameters):
        super(Supervisor, self).__init__()
        self.hidden_dim = parameters.get('hidden_dim')
        self.num_layers = parameters.get('num_layers')
        self.module_name = parameters.get('module', 'gru')
        self.max_seq_len = parameters.get('max_seq_len')
        self.padding_value = parameters.get('padding_value', -10)
        self.weights_like_tf = parameters.get('weights_like_tf', False)
        
        loss_fn = parameters.get('loss_fn_ers', 'mse').lower()
        try:
            self.loss = _LOSS_FUNCS[loss_fn]
        except KeyError:
            print(f'No loss function implimented for key: "{loss_fn}", supported loss functions are:\n\t{list(_LOSS_FUNCS.keys())}')
            print(f'Defaulting to MSE loss.')
            self.loss = _LOSS_FUNCS['mse']
                                                           
        if self.module_name.lower() == 'gru':
            self.sup_rnn = nn.GRU(input_size=self.hidden_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers-1,
                                  batch_first=True)
        elif self.module_name.lower() == 'lstm':
            self.sup_rnn = nn.LSTM(input_size=self.hidden_dim,
                                   hidden_size=self.hidden_dim,
                                   num_layers=self.num_layers-1,
                                   batch_first=True)
        else:
            raise ValueError('"module" should be "gru" or "lstm"')
        
        self.sup_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = nn.Sigmoid() # why sigmoid ?
        
        if self.weights_like_tf:
            # https://github.com/d9n13lt4n/timegan-pytorch/blob/main/models/timegan.py
            with torch.no_grad():
                for name, param in self.sup_rnn.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(1)
                    elif 'bias_hh' in name:
                        param.data.fill_(0)
                for name, param in self.sup_linear.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
    def forward(self, H, T):
        # ignore padding
        H_packed = nn.utils.rnn.pack_padded_sequence(input=H,
                                                     lengths=T, 
                                                     batch_first=True, 
                                                     enforce_sorted=False)
        
        H_o, H_t = self.sup_rnn(H_packed)
        
        H_o, T = nn.utils.rnn.pad_packed_sequence(sequence=H_o, 
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.max_seq_len)

        logits = self.sup_linear(H_o)
        H_hat = self.sup_sigmoid(logits)
                                                           
        return H_hat
    
class Discriminator(nn.Module):
    def __init__(self, parameters):
        super(Discriminator, self).__init__()
        self.hidden_dim = parameters.get('hidden_dim')
        self.num_layers = parameters.get('num_layers')
        self.module_name = parameters.get('module', 'gru')
        self.max_seq_len = parameters.get('max_seq_len')
        self.padding_value = parameters.get('padding_value', -10)
        self.weights_like_tf = parameters.get('weights_like_tf', False)
        
        loss_fn = parameters.get('loss_fn_gd', 'bcewithlogits').lower()
        try:
            self.loss = _LOSS_FUNCS[loss_fn]
        except KeyError:
            print(f'No loss function implimented for key: "{loss_fn}", supported loss functions are:\n\t{list(_LOSS_FUNCS.keys())}')
            print(f'Defaulting to BCEWithLogits Loss.')
            self.loss = _LOSS_FUNCS['bcewithlogits']
        
        if self.module_name.lower() == 'gru':
            self.dis_rnn = nn.GRU(input_size=self.hidden_dim, # change? to?
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers,
                                  batch_first=True)
        elif self.module_name.lower() == 'lstm':
            self.dis_rnn = nn.LSTM(input_size=self.hidden_dim, # change? to?
                                   hidden_size=self.hidden_dim,
                                   num_layers=self.num_layers,
                                   batch_first=True)
        else:
            raise ValueError('"module" should be "gru" or "lstm"')
        
        self.dis_linear = nn.Linear(self.hidden_dim, 1)
        
        if self.weights_like_tf:
            # https://github.com/d9n13lt4n/timegan-pytorch/blob/main/models/timegan.py
            with torch.no_grad():
                for name, param in self.dis_rnn.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(1)
                    elif 'bias_hh' in name:
                        param.data.fill_(0)
                for name, param in self.dis_linear.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
    def forward(self, H, T):
        # ignore padding
        H_packed = nn.utils.rnn.pack_padded_sequence(input=H,
                                                     lengths=T,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        
        H_o, H_t = self.dis_rnn(H_packed)
        
        # repad
        H_o, T = nn.utils.rnn.pad_packed_sequence(sequence=H_o, 
                                                  batch_first=True,
                                                  padding_value=self.padding_value,
                                                  total_length=self.max_seq_len)
        
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits
    
class TimeGAN(nn.Module):
    def __init__(self, parameters):
        super(TimeGAN, self).__init__()
         
        self.hidden_dim = parameters.get('hidden_dim')
        self.num_layers = parameters.get('num_layers')
        #self.iterations = parameters.get('iterations') # UNUSED
        self.batch_size = parameters.get('batch_size')
        self.module_name = parameters.get('module', 'gru')
        self.z_dim = parameters.get('z_dim')
        self.gamma = parameters.get('gamma', 1)
        
        
        self.embedder = Embedder(parameters)
        self.recovery = Recovery(parameters)
        self.generator = Generator(parameters)
        self.supervisor = Supervisor(parameters)
        self.discriminator = Discriminator(parameters)
        
        
    def _recovery_forward(self, X, T):
        
        H = self.embedder(X, T)
        X_tilde = self.recovery(H, T)
        
        H_hat_supervise = self.supervisor(H, T)
        
        G_loss_S = self.supervisor.loss(H, H_hat_supervise)  
        
        E_loss_T0 = self.recovery.loss(X, X_tilde)
        E_loss0 = 10*torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1*G_loss_S
        
        return E_loss, E_loss0, E_loss_T0
    
    def _supervisor_forward(self, X, T):
        
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)
        
        G_loss_S = self.supervisor.loss(H, H_hat_supervise) 
        
        return G_loss_S
    
    def _generator_forward(self, X, T, Z):
        
        # Supervisor
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)
        
        # Generator
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)
        
        # Synthetic data
        X_hat = self.recovery(H_hat, T)
        
        # Discriminator
        Y_fake = self.discriminator(H_hat, T)
        Y_fake_e = self.discriminator(E_hat, T)
        
        # Loss
        G_loss_U = self.generator.loss(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = self.generator.loss(Y_fake_e, torch.ones_like(Y_fake_e))
        
        G_loss_S = self.supervisor.loss(H, H_hat_supervise)  
        
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False)+1e-6) - torch.sqrt(X.var(dim=0, unbiased=False)+1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))
        
        G_loss_V = G_loss_V1 + G_loss_V2
        
        G_loss = G_loss_U + self.gamma*G_loss_U_e + 100*torch.sqrt(G_loss_S) + 100*G_loss_V
        
        return G_loss
    
    def _discriminator_forward(self, X, T, Z):
        
        H = self.embedder(X, T).detach()
        
        E_hat = self.generator(Z, T).detach()
        H_hat = self.supervisor(E_hat, T).detach()
        #detach() because these models don't have to be trained anymore
        
        Y_real = self.discriminator(H, T)
        Y_fake = self.discriminator(H_hat, T)
        Y_fake_e = self.discriminator(E_hat, T)  
        
        D_loss_real = self.discriminator.loss(Y_real, torch.ones_like(Y_real))
        D_loss_fake = self.discriminator.loss(Y_fake, torch.ones_like(Y_fake))
        D_loss_fake_e = self.discriminator.loss(Y_fake_e, torch.ones_like(Y_fake_e))
        
        
        D_loss = D_loss_real + D_loss_fake + self.gamma*D_loss_fake_e
        
        return D_loss
    
    def _inference(self, Z, T):
        
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)
        
        X_hat = self.recovery(H_hat, T)
        
        return X_hat
    
    def _anomaly_score(self, X, T, Z):
        mse = nn.MSELoss(reduction='none')
        H = self.embedder(X, T).detach()
        
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)
        
        Y_real = self.discriminator(H, T)
        Y_fake = self.discriminator(H_hat, T)
        Y_fake_e = self.discriminator(E_hat, T)
        
        # anomaly score(s) based on the mse of the latent representations and the output of the discriminator
        A_h = mse(H, H_hat).mean(dim=(1,2))
        A_d = mse(Y_real, Y_fake).mean(dim=1)
        A_de = mse(Y_real, Y_fake_e).mean(dim=1)
        
        return A_h.cpu().detach(), A_d.cpu().detach(), A_de.cpu().detach()
        
    def forward(self, X, T, Z, obj):
        if obj == 'autoencoder':
            loss = self._recovery_forward(X, T)
        elif obj == 'supervisor':
            loss = self._supervisor_forward(X, T)
        elif obj == 'generator':
            loss = self._generator_forward(X, T, Z)
        elif obj == 'discriminator':
            loss = self._discriminator_forward(X, T, Z)
        elif obj == 'inference':
            X_hat = self._inference(Z, T)
            return X_hat.cpu().detach().numpy()
        elif obj == 'score':
            A_h, A_d, A_de = self._anomaly_score(X, T, Z)
            return A_h.numpy(), A_d.numpy(), A_de.numpy()
        
        return loss
