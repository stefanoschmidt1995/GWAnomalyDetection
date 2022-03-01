import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # ensures same order as nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # needed for numba to see all 4 gpus | change accordingly if less gpus are available on system

import numpy as np
import gwpy.timeseries
import math
import h5py
import pathlib
from numba import cuda, float32, int16

import argparse
    
def main(strain_channel, whiten, t_start, t_end, segment_time, overlap, estimator, decimate_factor, gpu, out_dir='FD/'):
    str_overlap = f'-o{str(overlap).replace(".","_")}' if overlap > 0 else ''
    whiten_overlap = f'-whitened' if whiten else ''
    out_dir = f'{out_dir}{t_start}-{t_end}/'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
    save_file = f'./{out_dir}/{estimator}-{strain_channel.split(":")[0]}-{str(segment_time).replace(".","_")}s{str_overlap}{whiten_overlap}.hdf5' # path for save file
    
    if estimator == 'VAR':
        # VAR CUDA kernel
        @cuda.jit('void(float64[:,:], float64[:,:], int16, float64[:])')
        def kernel(f, out, NV, minmax):
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
    elif estimator == 'ANAM':
        # ANAM CUDA kernel
        @cuda.jit("void(float64[:,:], float64[:,:], int16)")
        def kernel(f, A, NV):
            x, tau, i = cuda.grid(3)
            tau += 1
            n = f.shape[1]
            if x < A.shape[0] and tau-1 < A.shape[1]:
                if tau >= 1 and tau <= NV and i < n-tau and i >= tau:
                    out = float(0)
                    #denom = (tau+1)**2*(n-2*tau)
                    for j in range(0, tau+1):
                        for l in range(0, tau+1):
                            out += abs(f[x,i+j]-f[x,i-l])
                    cuda.atomic.add(A, (x, tau-1), out)#/denom)
    else:
        raise ValueError("Incorrect value for: estimator. Choose from {VAR, ANAM}.")
    
    cuda.select_device(gpu)
    context = cuda.current_context()
    free_mem = context.get_memory_info().free - 10*2**20
    
    gpu = cuda.get_current_device()
    MAX_THREADS_PER_BLOCK = gpu.MAX_THREADS_PER_BLOCK
    threadsperblock = (4, 4, 64)
    assert np.prod(threadsperblock) <= MAX_THREADS_PER_BLOCK, f'Exceeded maximum threads per block on currect device: {gpu.name}'
    
    segment_time = math.ceil(segment_time*256)/256
    
    #####
    strain_data = gwpy.timeseries.TimeSeries.get(strain_channel, t_start, t_end)
    strain_sample_rate = strain_data.sample_rate.value
    
    if whiten:
        fduration = 2 # default
        strain_data = strain_data.whiten(method='median', fduration=fduration) # which method?
        # remove corrupted samples
        t_start = strain_data.times[0].value+0.5*fduration
        t_end = strain_data.times[-1].value-0.5*fduration
        strain_data = strain_data.crop(t_start, t_end)
        
    
    # TODO: change to float32 for speed increase (conversion takes time) at the cost of some precision (worth it?)
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
    
    NV = seg_length//(2*decimate_factor)
    
    # Array to store outputs of the estimators.
    strain_ests = np.empty((num_segs,NV), dtype=strain_segs.dtype)
    
    # used for calculating slope
    log_taus = np.log(np.arange(1,NV+1))
    lin_regress_denom = ((log_taus**2).mean() - (log_taus.mean())**2)
    
    if estimator == 'VAR':
        finfo = np.finfo(strain_segs.dtype)
        minmax_dtype = np.array([finfo.min, finfo.max])
        denom = seg_length-2*np.arange(1,NV+1)
    elif estimator == 'ANAM':
        denom = np.arange(1,NV+1)
        denom = (out_denom+1)**2 * (seg_length - 2*out_denom)
    
    # TODO: max of 65535 as gpu variable
    block_limit = min(int(free_mem/(strain_data.itemsize*(seg_length + NV))), 65535) 
    
    if num_segs > block_limit:
        for i in range(int(num_segs/block_limit)+1):
            left_idx = i*block_limit
            right_idx = min((i+1)*(block_limit), num_segs)
            idx_length = right_idx-left_idx

            # Create output array
            out = np.zeros((idx_length, NV), dtype=strain_segs.dtype)
            
            if estimator == 'VAR':
                blockspergrid = ((idx_length + (threadsperblock[0] - 1)) // threadsperblock[0],
                                 (NV+1 + (threadsperblock[1] - 1)) // threadsperblock[1],
                                 (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
                kernel[blockspergrid,threadsperblock](strain_segs[left_idx:right_idx], out, NV, minmax_dtype)
            elif estimator == 'ANAM':
                blockspergrid = ((idx_length + (threadsperblock[0] - 1)) // threadsperblock[0],
                                 (NV + (threadsperblock[1] - 1)) // threadsperblock[1],
                                 (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
                kernel[blockspergrid,threadsperblock](strain_segs[left_idx:right_idx], out, NV)
            
            strain_ests[left_idx:right_idx] = out/denom
    else:     
        # Create output array
        out = np.zeros((num_segs, NV), dtype=strain_segs.dtype)
        
        if estimator == 'VAR':
            blockspergrid = ((num_segs + (threadsperblock[0] - 1)) // threadsperblock[0],
                             (NV+1 + (threadsperblock[1] - 1)) // threadsperblock[1],
                             (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])

            kernel[blockspergrid,threadsperblock](strain_segs, out, NV, minmax_dtype)
        elif estimator == 'ANAM':
            blockspergrid = ((num_segs + (threadsperblock[0] - 1)) // threadsperblock[0],
                             (NV + (threadsperblock[1] - 1)) // threadsperblock[1],
                             (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])

            kernel[blockspergrid,threadsperblock](strain_segs, out, NV)

        strain_ests[:] = out/denom
    
    cuda.synchronize()
    
    # Compute fractal dimensions
    strain_ests = np.log(strain_ests)
    strain_frac_dims = 2 - ((log_taus*strain_ests).mean(axis=1) - log_taus.mean()*strain_ests.mean(axis=1))/lin_regress_denom
    
    
    with h5py.File(save_file, 'w') as f:
        f.create_group('meta')
        f['meta'].create_dataset('Strain', data=strain_channel)
        f['meta'].create_dataset('Strain Sample Rate', data=strain_sample_rate)
        f['meta'].create_dataset('Start GPS', data=t_start)
        f['meta'].create_dataset('End GPS', data=t_end)
        f['meta'].create_dataset('Segment Length', data=segment_time)
        f['meta'].create_dataset('Overlap', data=overlap)
        f['meta'].create_dataset('Whitened', data=whiten)

        f.create_group('data')
        f['data'].create_dataset('FD', data=strain_frac_dims)

        
        
if __name__ == "__main__":
    assert cuda.is_available(), 'Cuda not available'
    
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('start', type=int, help='GPS Start time')
    parser.add_argument('end', type=int, help='GPS End time')
    parser.add_argument('segtime', type=float, help='Length of segments in seconds')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--var', action='store_true', help='Use the VAR estimator')
    group.add_argument('--anam', action='store_true', help='Use the ANAM estimator')
    
    # optional arguments (defaults are set)
    parser.add_argument('-c','--channel', type=str, help='Channel to get strain data from', default='L1:DCS-CALIB_STRAIN_C01')
    parser.add_argument('-w','--whiten', action='store_true', help='Whiten strain data')
    parser.add_argument('-o','--overlap', type=float, help='Fraction of segment time to overlap with previous segment', default=0.0)
    parser.add_argument('-d','--dec', type=int, help='Decimate factor for the fractal dimension estimator', default=32)
    parser.add_argument('-g','--gpu', type=int, help='GPU to use', choices=[i for i in range(len(cuda.gpus))], default=0)
    parser.add_argument('-f', '--outdir', type=str, help='Output directory', default='FD/')
    
    args = parser.parse_args()
    
    if args.start > args.end:
        parser.error('Invalid choice of start and end times: start time > end time')
    
    estimator = 'VAR' if args.var else 'ANAM'
    
    main(args.channel, args.whiten, args.start, args.end, args.segtime, args.overlap, estimator, args.dec, args.gpu, args.outdir)