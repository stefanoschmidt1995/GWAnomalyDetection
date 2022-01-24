import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # ensures same order as nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # needed for numba to see all 4 gpus | change accordingly if less gpus are available on system

import glob
import numpy as np
import h5py
from numba import cuda
from tqdm import tqdm
import gc

import time

# Select GPU to use | same order as nvidia-smi
cuda.select_device(3)

# ANAM CUDA kernel
@cuda.jit("void(float32[:,:], float32[:,:], int16)")
def ANAM(f, A, dec):
    x, tau, i = cuda.grid(3)
    tau += 1
    n = f.shape[1]
    if x < A.shape[0] and tau < dec+1 and i < n-tau and i >= tau:
        out = float(0)
        denom = (tau+1)**2/(n-2*tau)
        for j in range(0, tau+1):
            for l in range(0, tau+1):
                out += abs(f[x,i+j]-f[x,i-l])
        A[x, i-tau + (n-1)*(tau-1)-(tau-1)**2] =  out/denom



# Files containing the segment data
files = glob.glob('segments/*.hdf5')
keys = ['256Hz', '512Hz', '1024Hz', '2048Hz', '4096Hz']

desired_segment_time = 0.2 # in seconds | should match desired_segment_time from aux_segments.py
secs_per_seg = np.ceil(256*desired_segment_time)/256 # time in seconds per segment 

for file in tqdm(files, desc='Files'):
    with h5py.File(f'./{file}', 'r') as f:
        start_t, end_t = f['meta']['Start GPS'][()], f['meta']['End GPS'][()]
    with h5py.File(f'./fractal_dims/{file.split("/")[1]}', 'w') as g:
        data_group = g.create_group('data')
        meta_group = g.create_group('meta')
        meta_group.create_dataset('Start GPS', data=int(start_t))
        meta_group.create_dataset('End GPS', data=int(end_t))
    
    
    # Loop over different sampling rates
    # The sampling rates determines the shape of the data segments and requires a different kernel size
    for key in tqdm(keys, desc='Sample Rates', leave=False):
        sample_rate = int(key[:-2])
        dec = min(32, int(sample_rate*secs_per_seg/2)-1) # decimate factor | minimum of 32 should be changed together with 'segs_per_run'
        
        # read data from segment filee
        with h5py.File(f'./{file}', 'r') as f:
            data_all =f['data'][key][:]
            names = f['data'][key].attrs['Channel names'][:]

        shape = data_all.shape
        
        # values used to determine the slope using linear regression later
        log_taus = np.log(np.arange(1,dec+1))
        lin_regress_denom = ((log_taus**2).mean() - (log_taus.mean())**2)
        
        # number of threads per block | product limited to 1024 (depends on gpu)
        threadsperblock = (4, 4, 64)
        
        # numbers of block per grid | should be less than 65535 in each dimension
        blockspergrid = ((shape[1] + (threadsperblock[0] - 1)) // threadsperblock[0],
                         (dec + (threadsperblock[1] - 1)) // threadsperblock[1],
                         (shape[2] + (threadsperblock[2] - 1)) // threadsperblock[2])
        
        mask = np.arange(1,dec+1)
        mask = ((shape[2]-1)*(mask-1)-(mask-1)**2).astype(int)
        
        outsums = []
        timer_start = time.perf_counter()
        for data, name in zip(data_all, names):
            data = cuda.to_device(data)
            out = np.zeros((shape[1], (shape[2]-1)*(dec)-(dec)**2), dtype=data_all.dtype)
            ANAM[blockspergrid,threadsperblock](data, out, dec)
            # synchronize gpu to ensure all tasks are completed
            #cuda.synchronize()
            outsums.append(np.add.reduceat(out, mask, axis=1))
        cuda.synchronize() # synchronize gpu to ensure all tasks are completed | not sure if necessary
        tqdm.write(f'{key} | {time.perf_counter()-timer_start} seconds')
        #print(key, ':', time.perf_counter()-timer_start, ' s')
        out = np.log(np.asarray(outsums)) 
        
        # vectorized fractal dimension calculation using slope linear regression
        frac_dims = (2-((log_taus*out).mean(axis=2)-log_taus.mean()*out.mean(axis=2))/lin_regress_denom) 
        
        with h5py.File(f'./fractal_dims/{file.split("/")[1]}', 'a') as g:
            for name, frac_dim in zip(names, frac_dims):
                g['data'].create_dataset(name, data=frac_dim)
                g['data'][name].attrs['Sample rate'] = int(key[:-2])
        
        del data_all, frac_dims, out, data # free memory of no longer used data
