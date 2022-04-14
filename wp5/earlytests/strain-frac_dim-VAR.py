import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # ensures same order as nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # needed for numba to see all 4 gpus | change accordingly if less gpus are available on system

import numpy as np
import gwpy.timeseries
import math
import h5py
import pathlib
from numba import cuda, float32, int16, float64


import time

t1 = time.perf_counter() # timing

##################################################################
# Changeable parameters

# Select GPU to use | same order as nvidia-smi
cuda.select_device(3)

strain_channel = 'L1:DCS-CALIB_STRAIN_C01'

t_start = 1262304018#1262547088 #1264294887+21*3600 # start GPS time
t_end =   1262307618#1262555061 #t_start+64 # end GPS time

desired_segment_time = 1 # in seconds
overlap = 0.0 # fraction of overlap between consecutive segments

dec = 64 # decimate factor for fractal dim. calculation

output_dir = 'FD/'
##################################################################

# VAR CUDA kernel
@cuda.jit('void(float64[:,:], float64[:,:], int16, float64[:])')
def VAR(f, out, NV, minmax):
    N = f.shape[1]
    dtype_min = minmax[0]
    dtype_max = minmax[1]
    x, k, l = cuda.grid(3)
    if x < out.shape[0] and k-1 < out.shape[1]:
        if k <= NV and k >= 1 and l < N-k and l>=k:
            local_min = dtype_max
            local_max = dtype_min
            for i in range(l-k, l+k+1):
                element = f[x,i]
                local_min = min(element, local_min)
                local_max = max(element, local_max)

            # Combine the (max-min) for each k [sum over l]
            cuda.atomic.add(out, (x, k-1), local_max - local_min)

context = cuda.current_context()
free_mem = context.get_memory_info().free - 10*2**20 # makes sure at least 10MiB of memory is left for safety

gpu = cuda.get_current_device()
MAX_THREADS_PER_BLOCK = gpu.MAX_THREADS_PER_BLOCK
        
# ensures GPS time of segments overlap nicely segments with differing (lower) sample rates
segment_time = math.ceil(desired_segment_time*256)/256 # change 256 to lowest sample rate used if different

# number of threads per block for the GPU | product limited to MAX_THREADS_PER_BLOCK
threadsperblock = (4, 4, 64)
assert np.prod(threadsperblock) <= MAX_THREADS_PER_BLOCK, f'Exceeded maximum threads per block on currect device: {gpu.name}'

str_overlap = f'-o{str(overlap).replace(".","_")}' if overlap > 0 else ''
output_dir = f'{output_dir}{t_start}-{t_end}/'
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
save_file = f'./{output_dir}/VAR-{strain_channel.replace(":", "_")}{str_overlap}.hdf5' # path for save file


##################################################################
strain_data = gwpy.timeseries.TimeSeries.get(strain_channel, t_start, t_end)
strain_sample_rate = strain_data.sample_rate.value
# TODO: change to float32 for speed increase (conversion takes time) at the cost of some precision
strain_data = strain_data.value #.astype(np.float32)



data_strides = strain_data.strides
seg_length = int(segment_time * strain_sample_rate) # number of points per segment
overlap_length = int(seg_length*overlap) # number of points that overlap with next segment
num_segs = int( (strain_data.shape[0] - overlap_length)/(seg_length - overlap_length) ) # number of segments

seg_shape = (num_segs, seg_length)
seg_strides = (data_strides[0]*(seg_length-overlap_length), data_strides[0])


# TODO: explore rewriting kernel to accept unsegmented strain and use slicing to avoid having to segment here
if overlap == 0:
    strain_segs = np.lib.stride_tricks.as_strided(strain_data, shape=seg_shape, strides=seg_strides)
else:
    # contiguous data is needed to transfer to gpu
    strain_segs = np.ascontiguousarray(np.lib.stride_tricks.as_strided(strain_data, shape=seg_shape, strides=seg_strides))


NV = seg_length//(2*dec)

# Array to store outputs of the estimators.
strain_ests = np.empty((num_segs,NV), dtype=strain_segs.dtype)

# Get the min and max of the dtype
finfo = np.finfo(strain_segs.dtype)
minmax_dtype = np.array([finfo.min, finfo.max])

# used for calculating slope
log_taus = np.log(np.arange(1,NV+1))
lin_regress_denom = ((log_taus**2).mean() - (log_taus.mean())**2)

# Number of elements to divide by for each tau
mean_denom = seg_length-2*np.arange(1,NV+1)

# Determine a max amount of segments to send per block without exceeding the available memory or the block limit
# determined by the size of the input and output arrays
block_limit = min(int(free_mem/(strain_data.itemsize*(seg_length + NV))), 65535) # max of 65535 as gpu variable
print(f'num segs: {num_segs} \tblock limit: {block_limit}')
t_frac = time.perf_counter() # timing

if num_segs > block_limit:
    for i in range(int(num_segs/block_limit)+1):
        left_idx = i*block_limit
        right_idx = min((i+1)*(block_limit), num_segs)
        idx_length = right_idx-left_idx
        
        # Create output array
        out = np.zeros((idx_length, NV), dtype=strain_segs.dtype)
        
        blockspergrid = ((idx_length + (threadsperblock[0] - 1)) // threadsperblock[0],
                         (NV+1 + (threadsperblock[1] - 1)) // threadsperblock[1],
                         (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
        
        VAR[blockspergrid,threadsperblock](strain_segs[left_idx:right_idx], out, NV, minmax_dtype)
        #print(out)
        strain_ests[left_idx:right_idx] = out/mean_denom
else:     
    # Create output array
    out = np.zeros((num_segs, NV), dtype=strain_segs.dtype)
    
    blockspergrid = ((num_segs + (threadsperblock[0] - 1)) // threadsperblock[0],
                     (NV+1 + (threadsperblock[1] - 1)) // threadsperblock[1],
                     (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
    
    VAR[blockspergrid,threadsperblock](strain_segs, out, NV, minmax_dtype)
    
    strain_ests[:] = out/mean_denom

cuda.synchronize()

# Compute fractal dimensions
strain_ests = np.log(strain_ests)
strain_frac_dims = 2 - ((log_taus*strain_ests).mean(axis=1) - log_taus.mean()*strain_ests.mean(axis=1))/lin_regress_denom

# timing
print('Calculation of Frac. Dim. in: ',time.perf_counter() - t_frac, ' seconds.')
print('Total: ',time.perf_counter() - t1, ' seconds.')

print(strain_frac_dims[:8]) # debug

# Save data to file
with h5py.File(save_file, 'w') as f:
    f.create_group('meta')
    f['meta'].create_dataset('Start GPS', data=t_start)
    f['meta'].create_dataset('End GPS', data=t_end)
    f['meta'].create_dataset('Segment Length', data=segment_time)
    f['meta'].create_dataset('Overlap', data=overlap)

    f.create_group('data')
    f['data'].create_dataset('Strain', data=strain_frac_dims)
    f['data']['Strain'].attrs['Sample Rate'] = strain_sample_rate
