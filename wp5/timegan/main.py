import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from timegan_model import TimeGAN

import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import glob
import pathlib
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from tqdm.auto import tqdm, trange

class TimeGAN_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, time=None, padding_value=-10):
        self.padding_value = padding_value
        
        if isinstance(time, type(None)):
            time = [len(x) for x in data]
        else:
            assert len(data) == len(time), 'data and time not of same length'
        
        self.X = torch.from_numpy(data).float() # TODO: cast data to float or keep as double and cast model params to double?
        self.T = torch.from_numpy(time).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.T[i]
    
    def collate_fn(self, batch):
        X_mb = nn.utils.rnn.pad_sequence(batch[0], batch_first=True, padding_value=self.padding_value)
        T_mb = batch[1]
        return X_mb, T_mb

    
def _sliding_window_view(x, window_shape):
    """
    Alternative implementation of numpy.lib.stride_tricks.sliding_window_view
    from numpy version 1.20.0
    """
    
    axis = tuple(range(x.ndim))
    
    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    x_shape_trimmed = list(x.shape)
    
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError('window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    
    out_shape = tuple(x_shape_trimmed) + window_shape
    
    return as_strided(x, strides=out_strides, shape=out_shape)[:-1]

# preprocessing function | data reading from files should be changed if different file structures are used.
def preprocess_data(file, max_seq_len, norm, min_max = None):#, padding_value=-10):
    # TODO: implement padding, yes or no?
    # TODO: implement multi-feature support (_sliding_window_view(data, (max_seq_len, n)), n: number of features)
    if isinstance(file, (list,tuple,np.ndarray)):
        if len(file) == 1:
            return preprocess_data(file[0], max_seq_len, norm, min_max=min_max)
        
        data_list, time_list, GPSTimes_list = np.empty(shape=(0, max_seq_len, 1)), np.array([]), np.array([])
        if min_max == None:
            d_min = np.inf
            d_max = np.NINF
            for fi in file:
                with h5py.File(file, 'r') as f:
                    data = f['data']['FD-0_5-0_0'][:]
                d_min = np.min([d_min, np.min(data)])
                d_max = np.max([d_max, np.max(data)])
        else:
            d_min = min_max[0]
            d_max = min_max[1]
        
        # Recurrently loop through files
        for fi in file:
            data, time, _, _, GPSTimes = preprocess_data(fi, max_seq_len, norm, min_max=(d_min,d_max))
            data_list = np.append(data_list, data, axis=0)
            time_list = np.append(time_list, time)
            GPSTimes_list = np.append(GPSTimes_list, GPSTimes)
            #print(data_list.shape, time_list.shape, GPSTimes_list.shape)
        return data_list, time_list, d_min, d_max, GPSTimes_list
    
    
    with h5py.File(file, 'r') as f:
        data = f['data']['FD-0_5-0_0'][:]
        start = f['meta']['Start GPS'][()]
        dt = f['meta']['Delta T'][()]
        num_segs = f['meta']['Num Segments'][()]
        
    
    if isinstance(norm, (list,tuple,np.ndarray)):
        if min_max == None:
            d_min = np.min(data)
            d_max = np.max(data)
        else:
            d_min = min_max[0]
            d_max = min_max[1]
        data = (data-d_min)/(d_max*d_min)
        data = data*(norm[1]-norm[0]) + norm[0]
    else: 
        d_min = None
        d_max = None
        
    #data = data[max_seq_len:]
    data = _sliding_window_view(data, (max_seq_len,))
    if data.shape[-1] == max_seq_len:
        # unsqueeze
        data = np.expand_dims(data, axis=-1)
    
    # time same as example keras implementation
    time = np.ones(data.shape[0]) * data.shape[1] 
    
    #GPSTimes = (np.arange(num_segs)*dt + start)[max_seq_len:data.shape[0]+max_seq_len]
    GPSTimes = (np.arange(num_segs)*dt + start)[:data.shape[0]]
    
    return data.copy(), time, d_min, d_max, GPSTimes

# function to generate the random input vectors Z
def _random_generator(shape, norm, rand_like_tf, T=None):
    
    if rand_like_tf and T is not None:
        Z = list()
        for i in range(shape[0]):
            temp = np.zeros([shape[1], shape[2]])
            temp_Z = np.random.uniform(norm[0], norm[1],(int(T[i].item()), shape[2])).astype(np.float32)
            temp[:int(T[i].item()),:] = temp_Z
            Z.append(temp_Z)
        Z = torch.from_numpy(np.asarray(Z))
        return Z
    else:
        Z = torch.zeros(*shape)
        dist = torch.distributions.Uniform(*norm)
        for i in range(shape[0]):
            Z[i] = dist.sample(shape[1:])
        return Z
    
# rescaling function for saving the data after use
def rescale(data, d_min, d_max, norm):
    if not isinstance(data, np.ndarray):
        return None
    if isinstance(norm, tuple):
        data = (data - norm[0])/(norm[1]-norm[0])
        data = data*(d_max*d_min) + d_min
    return data


    
def embedding_training(model, dataloader, emb_opt, rec_opt, parameters, device):
    pbar = trange(parameters['emb_epochs'], desc=f"Epoch: 0, Loss: {0.0:.4f}")
    model.train()
    for epoch in pbar:
        for X, T in dataloader:
            X = X.to(device)
            # T = T.to(device)
            
            model.zero_grad()
            
            _, E_loss0, E_loss_T0 = model(X, T, None, "autoencoder")
            E_loss0.backward()
            
            emb_opt.step()
            rec_opt.step()
        
        pbar.set_description(f"Epoch: {epoch}, Loss: {np.sqrt(E_loss_T0.item()):.4f}")

def supervisor_trainer(model, dataloader, sup_opt, parameters, device):
    pbar = trange(parameters['sup_epochs'], desc=f"Epoch: 0, Loss: {0.0:.4f}")
    model.train()
    for epoch in pbar:
        for X, T in dataloader:
            X = X.to(device)
            # T = T.to(device)
            model.zero_grad()
            
            G_loss_S = model(X, T, None, "supervisor")
            G_loss_S.backward()
            
            sup_opt.step()
        
        pbar.set_description(f"Epoch: {epoch}, Loss: {np.sqrt(G_loss_S.item()):.4f}")

def joint_trainer(model, dataloader, emb_opt, rec_opt, gen_opt, sup_opt, dis_opt, parameters, device):
    pbar = trange(parameters['dis_epochs'], desc=f"Epoch: 0, E_loss: {0.0:.4f}, G_loss: {0.0:.4f}, D_loss: {0.0:.4f}")
    model.train()
    for epoch in pbar:
        # Generator training (twice more than discriminator training)
        for _ in range(2):
            for X, T in dataloader:
                X = X.to(device)
                # T = T.to(device)
                
                Z = _random_generator(shape=(len(T), parameters['max_seq_len'], parameters['z_dim']),
                                      norm=parameters['norm_data'], 
                                      rand_like_tf=parameters['rand_like_tf'],
                                      T=T)
                Z = Z.to(device)
                
                # Train generator
                model.zero_grad()
                
                G_loss = model(X, T, Z, "generator")
                G_loss.backward()
                
                gen_opt.step()
                sup_opt.step()
                
                # Train embedder
                model.zero_grad()
                
                E_loss, _, E_loss_T0 = model(X, T, Z, obj="autoencoder")
                E_loss.backward()
                
                emb_opt.step()
                rec_opt.step()
        
        # Discriminator Training
        for X, T in dataloader:
            X = X.to(device)
            # T = T.to(device)
            
            Z = _random_generator(shape=(len(T), parameters['max_seq_len'], parameters['z_dim']),
                                  norm=parameters['norm_data'], 
                                  rand_like_tf=parameters['rand_like_tf'],
                                  T=T)
            Z = Z.to(device)
            
            model.zero_grad()
            
            D_loss = model(X, T, Z, "discriminator")
            if D_loss > parameters['dis_threshold']:
                D_loss.backward()
                
                dis_opt.step()
    
        pbar.set_description(f"Epoch: {epoch}, E_loss: {np.sqrt(E_loss.item()):.4f}, G_loss: {G_loss.item():.4f}, D_loss: {D_loss.item():.4f}")
                
    

def train_timegan(model, data, time, parameters, device):
    
    dataset = TimeGAN_Dataset(data, time)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=parameters['batch_size'],
                                             shuffle=False,
                                             num_workers=3)
    
    # optimisors 
    '''
    # Original Keras
    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars) 
    '''
    emb_opt = torch.optim.Adam(model.embedder.parameters(), lr=parameters['learning_rate'])
    rec_opt = torch.optim.Adam(model.recovery.parameters(), lr=parameters['learning_rate'])
    gen_opt = torch.optim.Adam(model.generator.parameters(), lr=parameters['learning_rate'])
    sup_opt = torch.optim.Adam(model.supervisor.parameters(), lr=parameters['learning_rate'])
    dis_opt = torch.optim.Adam(model.discriminator.parameters(), lr=parameters['learning_rate'])
    
    # 1. Embedding network training
    print('Start Embedding Network Training')
    embedding_training(model, dataloader, emb_opt, rec_opt, parameters, device)
    
    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    supervisor_trainer(model, dataloader, sup_opt, parameters, device)
    
    # 3. Joint Training
    print('Start Joint Training')
    joint_trainer(model, dataloader, emb_opt, rec_opt, gen_opt, sup_opt, dis_opt, parameters, device)
    
    # Save Trained Model
    torch.save({'model':model.state_dict(), 'parameters':parameters}, f"{parameters['save_path']}/model.pt")
    print(f"Model saved at: {parameters['save_path']}/model.pt")


def generate_synthetic_data(model, T, device, file=None, parameters=None):
    if file:
        parameters = torch.load(file)['parameters']
        model.load_state_dict(torch.load(file)['model'])
    elif parameters == None:
        raise ValueError('One of `file` or `parameters` must be given.')
    
    T = torch.from_numpy(T)
    
    model.eval()
    with torch.no_grad():
        Z = _random_generator(shape=(len(T), parameters['max_seq_len'], parameters['z_dim']),
                              norm=parameters['norm_data'], 
                              rand_like_tf=parameters['rand_like_tf'],
                              T=T)
        
        Z = Z.to(device)
        generated_data = model(None, T, Z, "inference")
    
    return generated_data
    

def score_test(model, data, time, device, file=None, parameters=None):
    if file:
        parameters = torch.load(file)['parameters']
        model.load_state_dict(torch.load(file)['model'])
    elif parameters == None:
        raise ValueError('One of `file` or `parameters` must be given.')
    
    dataset = TimeGAN_Dataset(data, time)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=parameters['batch_size'],
                                             shuffle=False,
                                             num_workers=3)
    anom_scores = np.zeros(shape=(len(dataset),3))
    
    model.eval()
    with torch.no_grad():
        for i, (X, T) in enumerate(dataloader):
            X = X.to(device)
            # T = T.to(device)
            
            Z = _random_generator(shape=(len(T), parameters['max_seq_len'], parameters['z_dim']),
                                  norm=parameters['norm_data'], 
                                  rand_like_tf=parameters['rand_like_tf'],
                                  T=T)
            Z = Z.to(device)
            
            A_h, A_d, A_de = model(X, T, Z, 'score')
            anom_scores[i*parameters['batch_size']:i*parameters['batch_size']+len(A_h)] = np.array([A_h, A_d, A_de]).T
    
    #avg_anom_scores = np.array([np.mean(anom_scores[i:i+parameters['max_seq_len']], axis=0) for i in range(anom_scores.shape[0]-parameters['max_seq_len'])])
    return anom_scores
    




if __name__ == '__main__':
    parameters = {
        'save_path': "trained_models/testgan_synth",
        'seed': 832022, # random seed to ensuree determinism
        'norm_data': (0, 1), # orginal code uses sigmoid for output of X_tilde, => X\in[0,1] is assumed
        'test_split': 0, # TODO: !temp! no validition done
        'loss_fn_ers': "mse", # loss function for the embedder, recovery, and supervisor | see _LOSS_FUNCS in timegan_model.py
        'loss_fn_gd': "bcewithlogits", # loss function for the generator and discriminator | see _LOSS_FUNCS in timegan_model.py
        'learning_rate': 0.001,
        'batch_size': 128,
        'emb_epochs': 25, # number of epochs for the embedding network training
        'sup_epochs': 25, # number of epochs for the supervised loss training
        'dis_epochs': 25, # number of epochs for the joint
        'dis_threshold': 0.15, # discriminator loss threshhold for backpropagation [0,1]
        'max_seq_len': 12, # (max) length of each input sequence
        'module': "lstm", # type of module to use in the timegan | ['gru','lstm']
        'hidden_dim': 24, # hidden dimension for RNNs
        'num_layers': 3, # number of layers for RNNs
        'gamma': 1, # parameter used in discriminator and generator loss 
        'padding_value': -10, # padding value for sequences smaller than max_seq_len
        'weights_like_tf': True, # bool, initialize the network weights the same way as the original keras implemention
        'rand_like_tf': True # bool, generate random input vectors in the same way as the original keras implemention
    }
    pathlib.Path(parameters['save_path']+'/').mkdir(parents=True, exist_ok=True) 
    
    # list of training file locations (coloured guassian noise)
    # prefix for CIT: /home/robin.vanderlaag/wp5/strain_fractals/TimeGAN/
    train_files = glob.glob('./synth_data/FD/*/*Synthetic_Data*.hdf5')
    parameters['training_data'] = train_files
    
    # list of test file locations (real L1 data)
    # prefix for CIT: /home/robin.vanderlaag/wp5/strain_fractals/TimeGAN/
    test_files = ['./1262476261-1262481170/VAR-L1-multi.hdf5',
                  './1262491498-1262500158/VAR-L1-multi.hdf5',
                  './1262503380-1262508582/VAR-L1-multi.hdf5',
                  './1262517193-1262523196/VAR-L1-multi.hdf5',
                  './1262529883-1262534731/VAR-L1-multi.hdf5',
                  './1262547088-1262555061/VAR-L1-multi.hdf5']
    test_files = ['./1262547088-1262555061/VAR-L1-multi.hdf5']
    parameters['test_data'] = test_files
    
    # !! change appropriatly to free device | order is the same as nvidia-smi
    device = torch.device('cuda', 3)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parameters['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Cuda unavailable, switching to cpu.")
        device = torch.device('cpu')
    
    torch.manual_seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    
    data, times, d_min, d_max, GPSTimes = preprocess_data(train_files, 
                                                          parameters['max_seq_len'], 
                                                          parameters['norm_data'], 
                                                          min_max=(1,2)) # min=1, max=2 for the fractal dimension of curves in R^2
    
    
    # TODO: multi dimensional data support (use data.shape[-1])
    #       could unsqueeze data if 1-dimensional to make data.shape[-1] work
    parameters['feature_dim'] = 1# data.shape[-1]
    parameters['z_dim'] = 1# data.shape[-1]
    
    if parameters['test_split'] > 0:
        X_train, X_val, T_train, T_val = train_test_split(data, times, 
                                                            test_size=parameters['test_split'],
                                                            random_state=parameters['seed'])
    else:
        X_train, T_train = shuffle(data, times, random_state=parameters['seed'])
        X_val, T_val = None, None
        
    
    model = TimeGAN(parameters)
    model.to(device)
    
    train_timegan(model, X_train, T_train, parameters, device)
    
    parameters = torch.load(f'{parameters["save_path"]}/model.pt')['parameters']
    model.load_state_dict(torch.load(f'{parameters["save_path"]}/model.pt')['model'])
    
    # generate data using the trained timegan model
    T_gen = T_train
    generated_data = generate_synthetic_data(model, T_gen, device, parameters=parameters)
    
    rescale_pars = {'d_min':d_min, 
                    'd_max':d_max, 
                    'norm': parameters['norm_data']}
    np.savez_compressed(f"{parameters['save_path']}/data", 
                        train_data=rescale(X_train, **rescale_pars), train_times=T_train,
                        val_data=rescale(X_val, **rescale_pars), val_times=T_val,
                        fake_data=rescale(generated_data, **rescale_pars), fake_times=T_gen)
    
    # Determine anomaly scores for the test data
    data, times, d_min, d_max, GPSTimes = preprocess_data(test_files, 
                                                          parameters['max_seq_len'], 
                                                          parameters['norm_data'], 
                                                          min_max=(1,2)) # min=1, max=2 for the fractal dimension of curves in R^2
    anom_scores = score_test(model, data, times, device, parameters=parameters)
    
    rescale_pars = {'d_min':d_min, 
                    'd_max':d_max, 
                    'norm': parameters['norm_data']}
    np.savez_compressed(f"{parameters['save_path']}/anom_score", 
                        data=rescale(data, **rescale_pars),
                        GPSTimes=GPSTimes,
                        anom_scores=anom_scores,
                        seq_len=parameters['max_seq_len'])
    
