import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # ensures same order as nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # needed for numba to see all 4 gpus | change accordingly if less gpus are available on system

import numpy as np
import pandas as pd
import gwdatafind
import gwpy.timeseries
import math
import h5py
import gc
from tqdm import tqdm

from numba import cuda, float32, int16
import time


# Changeable parameters

detector = 'L1' # Detector ifo

t_start = 1264294887+21*3600 # start GPS time
t_end = t_start+3600 # end GPS time

desired_segment_time = 1 # in seconds
overlap = 0.5 # fraction of overlap between consecutive segments

dec_fac_strain = 128 # decimate factor for fractal dim. calculation of the main strain | wrong implimentation atm FIX

strain_channel = 'L1:DCS-CALIB_STRAIN_C01'

# Select GPU to use | same order as nvidia-smi
cuda.select_device(3)
gpu = cuda.get_current_device()
print(gpu.name)
str_overlap = '-overlap' if overlap > 0 else ''
save_file = f'./strain_fractals/{t_start}-{t_end}/{strain_channel.replace(":", "_")}{str_overlap}.hdf5' # path for save file

####################################
# blockspergrid is limited to 65535, however memory issues occur for lower values
# block_limit parameter should be tuned according to GPU memory, sampling rate, and length of segments
block_limit = 625 # number of blocks per grid (equivalant to number of segments it computes in parallel)

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

if __name__ == '__main__':
    t1 = time.perf_counter() # timing
    segment_time = math.ceil(desired_segment_time*256)/256 # this ensures GPS time of segment does not
    
    # number of threads per block for the GPU | product limited to 1024 (depends on gpu)
    threadsperblock = (4, 4, 64)
    
    '''
    Open data for main strain
    '''
    
    if overlap > 0:
        # comment out for first run of longer series
        #t_start -= overlap*segment_time
        pass
    
    # Fetch (open) strain data
    strain_data = gwpy.timeseries.TimeSeries.get(strain_channel, t_start, t_end)
    strain_sample_rate = strain_data.sample_rate.value
    strain_data = strain_data.value
    
    # Turn data into segments
    data_strides = strain_data.strides
    seg_length = int(segment_time * strain_sample_rate) # number of points per segment
    
    overlap_length = int(seg_length*overlap) # number of points that overlap with next segment
    num_segs = int( (strain_data.shape[0] - overlap_length)/(seg_length - overlap_length) ) # number of segments
    
    seg_shape = (num_segs, seg_length)
    seg_strides = (data_strides[0]*(seg_length-overlap_length), data_strides[0])
    
    # contiguous data is needed to transfer to gpu
    strain_segs = np.ascontiguousarray(np.lib.stride_tricks.as_strided(strain_data, shape=seg_shape, strides=seg_strides))
    
    t_cuda = time.perf_counter() # timing
    
    strain_ests = np.empty((num_segs,dec_fac_strain), dtype=strain_segs.dtype)
    
    # used for calculating slope
    log_taus = np.log(np.arange(1,dec_fac_strain+1))
    lin_regress_denom = ((log_taus**2).mean() - (log_taus.mean())**2)
    
    # indexes used for summing the output of ANAM()
    mask = np.arange(1,dec_fac_strain+1)
    mask = ((seg_length-1)*(mask-1)-(mask-1)**2).astype(int)
    if num_segs > block_limit:
        for i in range(int(num_segs/block_limit)+1):
            left_idx = i*block_limit
            right_idx = min((i+1)*(block_limit), num_segs)
            idx_length = right_idx-left_idx
            
            out = np.zeros((idx_length, (seg_length-1)*(dec_fac_strain)-(dec_fac_strain)**2), dtype=strain_segs.dtype)
            
            blockspergrid = ((idx_length + (threadsperblock[0] - 1)) // threadsperblock[0],
                             (dec_fac_strain + (threadsperblock[1] - 1)) // threadsperblock[1],
                             (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
            print(out.nbytes/2**20, strain_segs[left_idx:right_idx].nbytes/2**20)
            ANAM[blockspergrid,threadsperblock](strain_segs[left_idx:right_idx], out, dec_fac_strain)
            cuda.synchronize()
            strain_ests[left_idx:right_idx] = np.add.reduceat(out, mask, axis=1)
    else:
        blockspergrid = ((num_segs + (threadsperblock[0] - 1)) // threadsperblock[0],
                         (dec_fac_strain + (threadsperblock[1] - 1)) // threadsperblock[1],
                         (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
        
        out = np.zeros((num_segs, (seg_length-1)*(dec_fac_strain)-(dec_fac_strain)**2), dtype=strain_segs.dtype)
        print(out.nbytes/2**20, strain_segs.nbytes/2**20)
        ANAM[blockspergrid,threadsperblock](strain_segs, out, dec_fac_strain)
        strain_ests[:] = np.add.reduceat(out, mask, axis=1)
    cuda.synchronize()
    
    strain_ests = np.log(strain_ests)
    strain_frac_dims = (2-((log_taus*strain_ests).mean(axis=1)-log_taus.mean()*strain_ests.mean(axis=1))/lin_regress_denom)
    
    print('Calculation of Frac. Dim. in: ',time.perf_counter() - t_cuda, ' seconds.')
    print('Total: ',time.perf_counter() - t1, ' seconds.')
    # save data
    with h5py.File(save_file, 'w') as f:
        f.create_group('meta')
        f['meta'].create_dataset('Start GPS', data=t_start)
        f['meta'].create_dataset('End GPS', data=t_end)
        f['meta'].create_dataset('Segment Length', data=segment_time)
        f['meta'].create_dataset('Overlap', data=overlap)
        
        f.create_group('data')
        f['data'].create_dataset('Strain', data=strain_frac_dims)
        f['data']['Strain'].attrs['Sample Rate'] = strain_sample_rate
    
    
