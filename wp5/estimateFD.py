import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # ensures same order as nvidia-smi command
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # needed for numba to see all 4 gpus | change accordingly if less gpus are available on system

import numpy as np
from numba import cuda, float32, int16

'''
cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py38/lib/python3.8/site-packages/numba/cuda/cudadrv/devicearray.py:790: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
'''
# TODO: investigate for performance increase

# Ignore warning for now
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
##################################################################


def get_fd(segs, estimator, decimate=32, gpu=0):
    """
    Computes the Fractal Dimension for multiple segments using CUDA kernels.

    Parameters
    ----------
    segs : numpy.ndarray
        2D array containing segmented data to compute the fractal dimension of, shape=(num_segs, seg_length).
    estimator : {'VAR', 'ANAM'}
        Estimator to use for computation.
    decimate : int, optional
        Decimate factor. The default is 32.
    gpu : int, optional
        Index of gpu to use, same order as nvidia-smi. The default is 0.

    Raises
    ------
    ValueError
        If unsupported estimator is passed.

    Returns
    -------
    frac_dims : numpy.ndarray
        1D array containing the fractal dimension of each segment, shape=(num_segs,).

    """
    
    if not segs.data.contiguous:
        # segs has to be a contiguous array for the kernel to work
        segs = np.ascontiguousarray(segs) 
    
    # select gpu before kernel declaration
    cuda.select_device(gpu) 
    
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
                    for j in range(0, tau+1):
                        for l in range(0, tau+1):
                            out += abs(f[x,i+j]-f[x,i-l])
                    cuda.atomic.add(A, (x, tau-1), out)
    else:
        raise ValueError("Incorrect value for: estimator. Choose from {VAR, ANAM}.")
    
    # Check how much memory is available after kernel is compiled
    context = cuda.current_context()
    free_mem = context.get_memory_info().free - 10*2**20
    
    gpu = cuda.get_current_device()
    MAX_THREADS_PER_BLOCK = gpu.MAX_THREADS_PER_BLOCK # hardware limitation
    threadsperblock = (4, 4, 64)
    assert np.prod(threadsperblock) <= MAX_THREADS_PER_BLOCK, f'Exceeded maximum threads per block on currect device: {gpu.name}'
    
    
    num_segs = segs.shape[0] # number of segments 
    seg_length = segs.shape[1] # length of each segment
    
    NV = seg_length // (2 * decimate)
    ests = np.empty((num_segs,NV), dtype=segs.dtype) # array to store estimation values
    
    # used for calculating slope
    log_taus = np.log(np.arange(1,NV+1))
    lin_regress_denom = ((log_taus**2).mean() - (log_taus.mean())**2)
    
    # initialise denominator for estimator computation
    if estimator == 'VAR':
        finfo = np.finfo(segs.dtype)
        minmax_dtype = np.array([finfo.min, finfo.max])
        denom = seg_length-2*np.arange(1,NV+1)
    elif estimator == 'ANAM':
        denom = np.arange(1,NV+1)
        denom = (denom+1)**2 * (seg_length - 2*denom)
    
    #TODO: max of 65535 as gpu variable
    block_limit = min(int(free_mem/(segs.itemsize*(seg_length + NV))), 65535)  # hardware limitation
    
    if num_segs > block_limit:
        for i in range(int(num_segs/block_limit)+1):
            left_idx = i*block_limit
            right_idx = min((i+1)*(block_limit), num_segs)
            idx_length = right_idx-left_idx

            # Create output array
            out = np.zeros((idx_length, NV), dtype=segs.dtype)

            if estimator == 'VAR':
                blockspergrid = ((idx_length + (threadsperblock[0] - 1)) // threadsperblock[0],
                                 (NV+1 + (threadsperblock[1] - 1)) // threadsperblock[1],
                                 (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
                kernel[blockspergrid,threadsperblock](segs[left_idx:right_idx], out, NV, minmax_dtype)
            elif estimator == 'ANAM':
                blockspergrid = ((idx_length + (threadsperblock[0] - 1)) // threadsperblock[0],
                                 (NV + (threadsperblock[1] - 1)) // threadsperblock[1],
                                 (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])
                kernel[blockspergrid,threadsperblock](segs[left_idx:right_idx], out, NV)

            ests[left_idx:right_idx] = out/denom
    else:     
        # Create output array
        out = np.zeros((num_segs, NV), dtype=segs.dtype)

        if estimator == 'VAR':
            blockspergrid = ((num_segs + (threadsperblock[0] - 1)) // threadsperblock[0],
                             (NV+1 + (threadsperblock[1] - 1)) // threadsperblock[1],
                             (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])

            kernel[blockspergrid,threadsperblock](segs, out, NV, minmax_dtype)
        elif estimator == 'ANAM':
            blockspergrid = ((num_segs + (threadsperblock[0] - 1)) // threadsperblock[0],
                             (NV + (threadsperblock[1] - 1)) // threadsperblock[1],
                             (seg_length + (threadsperblock[2] - 1)) // threadsperblock[2])

            kernel[blockspergrid,threadsperblock](segs, out, NV)

        ests[:] = out/denom

        cuda.synchronize() # synchronize cuda
        
        # Compute fractal dimensions
        ests = np.log(ests)
        frac_dims = 2 - ((log_taus*ests).mean(axis=1) - log_taus.mean()*ests.mean(axis=1))/lin_regress_denom
        
        return frac_dims