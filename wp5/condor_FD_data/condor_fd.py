################## remove for CONDOR
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
#################

import os
from math import log2
import gc
import argparse
import traceback
import pathlib
from multiprocessing import Queue, Process
from queue import Empty
import time
import numpy as np
import pandas as pd
import itertools
import glob
import gwpy.timeseries
from numba import cuda, float32, int16

_MAX_RETRIES = 4 # number of retries to get data in case of errors (for each gpstime)

def segmentize(data, seg_length, overlap):
    """
    Segmentizes an array of data.

    Parameters
    ----------
    data : numpy.ndarray
        1D array containing data to segmentize.
    seg_length : int
        Desired length of each segment.
    overlap : int
        Number of elements that overlap with the next segment.

    Returns
    -------
    segs : numpy.ndarray
        2D array containing the segmentized data, shape=(num_segs, seg_length).

    """
    
    data_strides = data.strides
    num_segs = int((data.shape[1] - overlap)/(seg_length - overlap))
    
    seg_shape = (data.shape[0], num_segs, seg_length)
    seg_strides = (data_strides[0], data_strides[1]*(seg_length - overlap), data_strides[1])
    
    segs = np.lib.stride_tricks.as_strided(data, shape=seg_shape, strides=seg_strides)
    
    return segs


def get_fd(segs, kernel, estimator, decimate=32, gpu=0):
    """
    Computes the Fractal Dimension for multiple segments using CUDA kernels.

    Parameters
    ----------
    segs : numpy.ndarray
        2D array containing segmented data to compute the fractal dimension of, shape=(num_segs, seg_length).
    kernel : func
        Estimator function to use for computation.
    estimator : {'VAR', 'ANAM'}
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
    
    block_limit = min(int(free_mem/(segs.itemsize*(seg_length + NV))), gpu.MAX_GRID_DIM_X)  # hardware limitation | gpu.MAX_GRID_DIM_X
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


def compute_fd(data_queue, n_senders, times, args):
    cuda.select_device(0)
    # initialise estimator kernel
    estimator = 'VAR' if args.var else 'ANAM'
    if estimator == 'VAR':
        # VAR CUDA kernel
        @cuda.jit('void(float32[:,:], float32[:,:], int16, float32[:])')
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
        @cuda.jit("void(float32[:,:], float32[:,:], int16)")
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
    
    seglen_dict = {} # dict of all possible segment lengths | sampling_rate*seg_len_seconds
    for key in itertools.product(2**np.arange(10,14+1).astype(int), args.segs):
        if int(key[0]*key[1]) in seglen_dict.keys():
            seglen_dict[int(key[0]*key[1])].append(key)
        else:
            seglen_dict[int(key[0]*key[1])] = [key]
    
    n_sentinels = 0
    _job_finished = False # bool for breaking while True loop when finished
    while True:
        if _job_finished : # break while loop when done
            print(f'FD: Job done', flush=True)
            break
        # initialise dicts/list for current batch
        frac_dims = {}
        time_list = []
        file_dict = {}
        while len(file_dict) < args.timespergpu: # fetch `args.timespergpu` number of data files
            try: 
                # long timeout in case a data fetching process gets stuck and all others have finished
                gpstime, fname = data_queue.get(timeout=1800)
            except Empty as e:
                # queue should only be empty if a fetching process takes longer than 1800s
                if args.verbose: print(f'FD: Breaking - Empty{e} | {len(time_list)=}', flush=True)
                _job_finished = True
                break
            if gpstime is None:
                n_sentinels += 1
                if n_sentinels >= n_senders: # all fetching processes are finished -> break
                    _job_finished = True
                    if args.verbose: print(f'FD: Breaking - {n_sentinels=} |{len(time_list)=}', flush=True)
                    break
            else:
                print(f'FD: Adding: {gpstime}', flush=True)
                # add `gpstime` and data to dicts/list
                time_list.append(gpstime)
                file_dict[int(gpstime)] = fname
                frac_dims[int(gpstime)] = {}
        if len(time_list) == 0: # if no gpstimes have been added go to next loop and break
            continue
        if args.verbose: print(f'FD: Starting', flush=True) 
        start_time = time.time()
        
        # The kernel to compute the fractal dimensions can only do segments of equal size in parallel
        # Thus we loop over each segment length
        # TODO: explore rewriting kernel to accept varying segment lengths
        for length, ids_list in seglen_dict.items():
            segs = []
            segs_ids = []
            for gpstime in time_list: # loop over each gpstime in batch
                file = np.load(file_dict[int(gpstime)]) # load file associated with the gpstime
                for (sr, seg_length) in ids_list: # loop over each combination of sampling rate and segment length (in seconds)
                    # read in data and names for sampling rate
                    data = file[str(sr)]
                    chan_names = file[f'names_{sr}']
                    
                    # set segmentize parameters
                    stride = seg_length * (1-args.overlap)
                    assert args.width % seg_length == 0
                    if int(args.width // seg_length) & 1 == 0: # even
                        width = args.width - seg_length
                    else: # odd
                        width = args.width
                    
                    # see if data can be precisly segmented without remainder
                    rem = data.shape[1] % sr*seg_length
                    if int((data.shape[1]-rem) // sr*seg_length) & 1 == 0: # slice awaythe remaining end part
                        segments = segmentize(data[:,int(rem+sr*seg_length//2):-int(sr*seg_length//2 + (sr*seg_length%2>0))],
                                              int(sr*seg_length), int(sr*(seg_length-stride)))
                    else: # data can be precisly segmented
                        segments = segmentize(data, int(sr*seg_length), int(sr*(seg_length-stride)))
                    
                    points_per_seg = segments.shape[1] # number of slices (points) to compute to FD of within each segment
                    segments = segments.reshape(segments.shape[0]*segments.shape[1], segments.shape[2]) # reshape to be a 2d-array
                    segs.append(segments) # add segments to list
                    segs_ids.append(np.array([[gpstime, seg_length, chan, points_per_seg] for chan in chan_names])) # add info of data to list
            del segments
            
            #1 stack corresponds to each unique set of (sr, seg_length) per gpstime
            stack_shapes = [stack.shape[0] for stack in segs] # shapes of each stack, 
            segs = np.ascontiguousarray(np.vstack(segs), dtype=np.float32) # most of the data is float32, downscale rest to conserve memory
            
            # set decimate factor for current length
            # TODO: find better method
            decimate = int(32/16384 * length)
            decimate = 1 if decimate <= 1 else 1 << int(log2(decimate-1)) + 1 # round decimate to 1 or to nearest power of 2 greater than
            with np.errstate(all='ignore'): # ignoring warnings since intermediate steps might divide by zero
                fd = get_fd(segs, kernel=kernel, estimator=estimator, decimate=decimate) # compute FD of segments
            del segs, data
            gc.collect()
            
            fd = np.split(fd, np.cumsum(stack_shapes)[:-1]) # split FDs according to stacks
            for segs_fd, seg_ids in zip(fd, segs_ids): # loop over each stack and its info
                # split FDs based on number of slices per channel segment (`points per seg`)
                segs_fd = np.split(segs_fd, np.cumsum(seg_ids[:,-1].astype(int))[:-1]) 
                for seg_fd, [gpstime, seg_length, chan, points_per_seg] in zip(segs_fd, seg_ids):
                    try:
                        frac_dims[int(float(gpstime))][f'sl-{seg_length}'][chan_map[chan]] = seg_fd
                    except KeyError: # if array for `seg_length` is not made yet, make array first
                        frac_dims[int(float(gpstime))][f'sl-{seg_length}'] = np.zeros(shape=(len(channels), int(points_per_seg)))
                        frac_dims[int(float(gpstime))][f'sl-{seg_length}'][chan_map[chan]] = seg_fd
        
        # TODO: change to joint file? which format?
        for gpstime in time_list: # loop over each gpstime and save to file
            np.savez_compressed(args.path+f'/{int(gpstime)}-fd.npz', **frac_dims[int(gpstime)], channels=channels, overlap=args.overlap)
            times = times[times != gpstime] # remove completed time
            if args.verbose: print(f'FD: removing {file_dict[int(gpstime)]}')
            try:
                os.remove(file_dict[int(gpstime)]) # remove the data file to clear up space
            except OSError as e:
                print(f'Error: {e.filename} - {e.strerror}.', flush=True) 
                if os.path.isfile(file_dict[int(gpstime)]):
                    raise(e) # Raise error if file is still there
                else: 
                    print(f'Error: {e.filename} - {e.strerror}.', flush=True) # only print error if file doesn't exist anymore
                
        np.save(f'times/{args.times}', times) # save to folder that will be returned on condor job exit
        if args.verbose: print(f'FD: {gpstime} \t\t | \t\t in: {time.time()-start_time} s', flush=True)


def fetch_data(times_queue, data_queue, args, times, max_retries, pid):
    while True: # loop until broken by sentinel value `None`
        gpstime = times_queue.get()
        start_time = time.time()
        if gpstime is None: # no more times left, end process
            data_queue.put((None, None))
            break
        retry_count = 0
        while retry_count <= max_retries: # retry to get the same time in case of errors
            try:
                # get data via gwpy
                data = gwpy.timeseries.TimeSeriesDict.get(channels,
                                                          gpstime-args.width/2, 
                                                          gpstime+args.width/2,
                                                          allow_tape=True, verbose=True, nproc=2)
            except Exception as e:
                print(e)
                traceback.print_exc()
                retry_count += 1
                continue
            break # break while loop if succesful
        if retry_count > max_retries: # skip getting data for `gpstime` if it fails after `max_retries` attempts
            print(f'Fetch {pid}: Failed to get data for time {gpstime} after {retry_count+1} attempts', flush=True)
            continue
        
        # dict for data and channels names keyed by sampling rate
        chan_data = {str(key):[] for key in (2**np.arange(10,14+1)).astype(int)}
        chan_names = {'names_'+str(key):[] for key in (2**np.arange(10,14+1).astype(int))}
        for ts in data.items(): # put data and name of each channel in dicts
            sr = int(ts[1].sample_rate.value)
            chan_names['names_'+str(sr)].append(ts[0])
            chan_data[str(sr)].append(ts[1].value)
        fname = f'{int(gpstime)}_data.npz'
        np.savez(fname, **chan_names, **chan_data) # save data and names to .npz file
        del data, chan_data, chan_names
        gc.collect()
        if args.verbose: print(f'Fetch {pid}: {gpstime} \t\t | \t\t in: {time.time()-start_time} s', flush=True)
        data_queue.put((gpstime, fname)) # add data to data_queue. Will block indefintely if queue is full, add timeout?
        
def parse_command_line():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('-t', '--times', type=str, help='GPS times at center of interval in .npy file')
    parser.add_argument('-w', '--width', type=int, help='width of time interval in seconds')
    parser.add_argument('-s', '--segs', nargs='+', type=float, help='Length of segments in seconds (can pass multiple)')
    parser.add_argument('-c', '--channels', type=str, help='Path to .csv file containing list of channels', 
                        default='L1_use_channels.csv')
    parser.add_argument('-p', '--path', type=str, help='Output directory', 
                        default='./')
    parser.add_argument('-o', '--overlap', type=float, help='Fraction of overlap between consecutive segments', metavar='[0,1)',default=0.0)
    parser.add_argument('-d', '--dec', type=int, help='Decimate factor for the fractal dimension estimator for segments of length 16384, scales for different lengths automatically', default=32)
    parser.add_argument('-q', '--maxqueuesize', type=int, help='Maximum size of the FD computing queue to limit memory', default=5)
    parser.add_argument('-r', '--maxretries', type=int, help='Number of retries for each data fetch attempt', default=0)
    parser.add_argument('-n', '--nproc', type=int, help='Number of data fetching processes', default=5)
    parser.add_argument('-i', '--timespergpu', type=int, help='Number gpstimes to compute the fd of at the same time', default=10)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print additional status update')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--var', action='store_true', help='Use the VAR estimator')
    group.add_argument('--anam', action='store_true', help='Use the ANAM estimator')
    
    args = parser.parse_args()
    
    if args.overlap >= 1: 
        parser.error(f'Invalid choice of overlap: overlap >= 1. overlap={args.overlap}')
    return args
    
if __name__ == '__main__':
    args = parse_command_line()
    
    channels = pd.read_csv(args.channels) # ['channel'].tolist()
    # these channels don't seem to be available for most gpstimes, thus remove
    err_chans = ['L1:OAF-OPTIC_IMCL_OUT_DQ', 'L1:OAF-OPTIC_PRCL_OUT_DQ', 'L1:OAF-OPTIC_SRCL_OUT_DQ', 'L1:OAF-OPTIC_XARM_OUT_DQ',
                 'L1:OAF-OPTIC_YARM_OUT_DQ', 'L1:OAF-OPTIC_MICH_OUT_DQ', 'L1:OAF-OPTIC_CARM_OUT_DQ', 'L1:OAF-OPTIC_DARM_OUT_DQ']
    channels = channels[~channels['channel'].isin(err_chans)]
    
    channels = channels['channel'].tolist() # convert to list
    chan_map = {chan: i for i, chan in enumerate(channels)}
    
    # load times
    times = np.load(args.times)
    if args.verbose:
        print(f'{len(times)} gpstimes loaded.')
    # create times diretory to save/track not completed times, useful incase condor returns before process is finished
    pathlib.Path('times').mkdir(exist_ok=True)
    
    # initialize queues
    times_queue = Queue()
    data_queue = Queue(maxsize=args.maxqueuesize) # maxqueuesize set to prevent too much memory usage
    for gpstime in times:
        times_queue.put(gpstime)
    
    # initialize data fetching processes
    fetch_pool = []
    for i in range(args.nproc):
        times_queue.put(None)
        fetch_pool.append(Process(target=fetch_data, args=(times_queue,data_queue,args,times,args.maxretries, i+1)))
        fetch_pool[i].start()
        time.sleep(0.2) # short pause between starting processes, should be redundant
    
    # initialize FD computing process
    compute_proc = Process(target=compute_fd, args=(data_queue, args.nproc, times, args))
    compute_proc.start()
    
    # wait for all processes to finish
    compute_proc.join() # compute_proc first because fetch_pool processes can get stuck
    for fetch_proc in fetch_pool:
        fetch_proc.join(1) # 1 second timeout, since it should have already finished if compute_proc finished
        if fetch_proc.is_alive():
            # process didn't join, should only happen if it got stuck and data_queue timed out
            fetch_proc.kill()
    
    
    
    
    
    