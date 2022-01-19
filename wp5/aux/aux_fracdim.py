import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # ensures same order as nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # needed for numba to see all 4 gpus | change accordingly if less gpus are available on system

import glob
import numpy as np
import h5py
from numba import cuda
from tqdm import tqdm
import gc

# Select GPU to use | same order as nvidia-smi
cuda.select_device(0)

# ANAM CUDA kernel
@cuda.jit("void(float32[:,:], float32[:,:,:])")
def ANAM(f, A):
    x, tau, i = cuda.grid(3)
    tau += 1
    n = f.shape[1]
    if x < A.shape[0] and tau < A.shape[1]+1 and i < n-tau and i >= tau:
        out = float(0)
        denom = (tau+1)**2/(n-2*tau)
        for j in range(0, tau+1):
            for l in range(0, tau+1):
                out += abs(f[x,i+j]-f[x,i-l])
        A[x, tau-1, i] =  out/denom



# Files containing the segment data
files = glob.glob('segments/*.hdf5')
keys = ['256Hz', '512Hz', '1024Hz', '2048Hz', '4096Hz']

desired_segment_time = 0.2 # in seconds | should match desired_segment_time from aux_segments.py
secs_per_seg = np.ceil(256*desired_segment_time)/256 # time in seconds per segment 
segs_per_run = 36 # number of segments the kernel can handle per run | should be changed accordingly if number of channels changes

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
        dec = min(32, int(sample_rate*secs_per_seg/2)) # decimate factor | minimum of 32 should be changed together with 'segs_per_run'
        
        # values used to determine the slope using linear regression later
        log_taus = np.log(np.arange(1,dec))
        lin_regress_denom = ((log_taus**2).mean() - (log_taus.mean())**2)
        
        # number of threads per block | product limited to 1024 (depends on gpu)
        threadsperblock = (4, 4, 64)
        
        '''
        TODO: FIX blockpergrid
            Currently any higher values than these will result in the error: 
            CudaAPIError: [1] Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE
            However, each value is significantly smaller than the limit of 65535.
            
            Thread asking for help/clarification has been openened:
                https://forums.developer.nvidia.com/t/cudaapierror-1-call-to-culaunchkernel-results-in-cuda-error-invalid-value-with-numba/200699
            But no reply as of 2022/01/19
            
            Currently values are the highest they can be without causing the error
            If a different time bin length (secs_per_seg) is chosen the values of blockspergrid will
            have to be redetermined. 
            Easiest way would be to manually change 'segs_per_run' to be as high as possible without the error occuring
        '''
        # numbers of block per grid | should be less than 65535 in each dimension
        blockspergrid = ((segs_per_run + (threadsperblock[0] - 1)) // threadsperblock[0],
                         (dec + (threadsperblock[1] - 1)) // threadsperblock[1],
                         (int(sample_rate*secs_per_seg) + (threadsperblock[2] - 1)) // threadsperblock[2])
        
        # read data from segment filee
        with h5py.File(f'./{file}', 'r') as f:
            data_all =f['data'][key][:]
            names = f['data'][key].attrs['Channel names'][:]

        shape = data_all.shape
        
        '''
        TODO: transfer output array to gpu manually could improve performance
            initial attempts to do this ran into out of memory errors as the output array can be greater than 10Gb
            
        '''
        outsums = []
        for j, (data, name) in enumerate(zip(data_all, names)):
            data = cuda.to_device(data)
            out = np.zeros((shape[1], dec-1, shape[2]), dtype=data_all.dtype)
            for i in range(int(shape[1]/segs_per_run+1)):
                ANAM[threadsperblock, blockspergrid](data[i*segs_per_run:(i+1)*segs_per_run], 
                                                     out[i*segs_per_run:(i+1)*segs_per_run])
            outsums.append(np.sum(out, axis=2)) 
        cuda.synchronize() # synchronize gpu to ensure all tasks are completed | not sure if necessary
        
        
        out = np.log(np.asarray(outsums)) 
        # vectorized fractal dimension calculation using slope linear regression
        frac_dims = (2-((log_taus*out).mean(axis=2)-log_taus.mean()*out.mean(axis=2))/lin_regress_denom) 
        
        with h5py.File(f'./fractal_dims/{file.split("/")[1]}', 'a') as g:
            for name, frac_dim in zip(names, frac_dims):
                g['data'].create_dataset(name, data=frac_dim)
                g['data'][name].attrs['Sample rate'] = int(key[:-2])
        
        del data_all, frac_dims, out, data # free memory of no longer used data
        
    
        
            

