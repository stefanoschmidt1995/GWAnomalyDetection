import os
import numpy as np
import pandas as pd
import numba
import itertools
import glob
import h5py
import argparse
import pathlib
import time
import gc
import gwdatafind
import pycbc.psd

import warnings
warnings.filterwarnings("ignore")

import gwpy.timeseries

from multiprocessing import Queue, Process, Value
from queue import Empty

#numba.set_num_threads(os.environ['OMP_NUM_THREADS']) # set number of threads to number of cpus given by HTCondor
numba.set_num_threads(10) # local running

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

@numba.njit("f4[:](f4[:,:], i8, i8)", parallel=True)
def compute_fd_parallel_f4(fs, dec, step):
    """
    Computes the Fractal Dimension using the VAR method 
    and `pyramid summing` for single-precision (float32) data.
    For double-precision (float64) data use compute_fd_parallel/compute_fd_serial.

    Parameters
    ----------
    fs : numpy.ndarray
        2D numpy array of the data, shape=(num, N).
    dec : int, optional
        Decimate value for standard k_func. The default is 64.
    k_func : numba.core.registry.CPUDispatcher, optional
        Function that takes N, dec and returns a numpy.ndarray for k_n, 
        must be decorated by @numba.jit(nopython=True).
        By default it is set to use numpy.arange(1, N//(2*dec)) for k_n.

    Returns
    -------
    FDs : numpy.ndarray
        Array of computed Fractal Dimension, shape=(num,).

    """
    FDs = np.empty(shape=fs.shape[0], dtype=np.float32)
    N = fs.shape[1]
    k_n = np.arange(1, N//(2*dec), step, np.int64)
    n_max = len(k_n)
    for ii in numba.prange(fs.shape[0]):
        f = fs[ii]
        V_i = np.empty(shape=(n_max), dtype=np.float32)
        
        ub = np.empty(shape=(N-2*k_n[0],2), dtype=np.float32) # current iteration
        for i in range(0, N-2*k_n[0]):
            ub[i,0] = np.max(f[i:i+2*k_n[0]+1])
            ub[i,1] = np.min(f[i:i+2*k_n[0]+1])
        V_i[0] = np.mean(ub[:,0]-ub[:,1])

        for n in range(1,n_max):
            d = k_n[n] - k_n[n-1]
            for i in range(0, N-2*k_n[n]):
                ub[i,0] = max(ub[i,0], ub[i+2*d,0])
                ub[i,1] = min(ub[i,1], ub[i+2*d,1])
            V_i[n] = np.mean(ub[:N-2*k_n[n],0]-ub[:N-2*k_n[n],1])

        X = np.log(k_n)
        X_m = X - np.mean(X)
        Y = np.log(V_i)
        Y_m = Y - np.mean(Y)
        FDs[ii] = 2 - np.sum((X_m)*(Y_m))/np.sum((X_m)**2)
    return FDs

def compute_fd(data_queue, n_senders, times, args, channels, chan_map):
    seglen_dict = {} # dict of all possible segment lengths | sampling_rate*seg_len_seconds
    for key in itertools.product(2**np.arange(10,14+1).astype(int), args.segs):
        if int(key[0]*key[1]) in seglen_dict.keys():
            seglen_dict[int(key[0]*key[1])].append(key)
        else:
            seglen_dict[int(key[0]*key[1])] = [key]
    
    _job_finished = False # bool for breaking while True loop when finished
    sentinels = 0
    while True:
        if _job_finished : # break while loop when done
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: Job done', flush=True)
            break
        
        # initialise dicts/list for current batch
        frac_dims = {}
        time_list = []
        file_dict = {}
        while len(file_dict) < args.timesparallel: # fetch `args.timespergpu` number of data files
            try: 
                # long timeout in case a data fetching process gets stuck and all others have finished
                gpstime, fname = data_queue.get(timeout=1800) # 1800 ?
            except Empty as e:
                # queue should only be empty if a fetching process takes longer than 1800s
                if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: Breaking - Empty{e} | {len(time_list)=}', flush=True)
                _job_finished = True
                break
            if gpstime is None:
                sentinels += 1
                if sentinels >= n_senders: # all fetching processes are finished -> break
                    _job_finished = True
                    if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: Breaking - {sentinels=} | {len(time_list)=}', flush=True)
                    break
            else:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: Adding: {gpstime}', flush=True)
                # add `gpstime` and data to dicts/list
                time_list.append(gpstime)
                file_dict[str(gpstime)] = fname
                frac_dims[str(gpstime)] = {}
        if len(time_list) == 0: # if no gpstimes have been added go to next loop and break
            continue
        if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: Starting', flush=True) 
        start_time = time.time()
        
        for length, ids_list in seglen_dict.items():
            #if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: Length={length}', flush=True) 
            segs = []
            segs_ids = []
            for gpstime in time_list: # loop over each gpstime in batch
                file = np.load(file_dict[str(gpstime)]) # load file associated with the gpstime
                for (sr, seg_length) in ids_list: # loop over each combination of sampling rate and segment length (in seconds)
                    # read in data and names for sampling rate
                    data = file[str(sr)]
                    chan_names = file[f'names_{sr}']
                    
                    width = int(args.width*sr)
                    seg_length_sr = int(seg_length*sr)
                    assert seg_length*sr*args.overlap == np.floor(seg_length*sr*args.overlap) # assert is whole number
                    # number of points to slice away from each side
                    rm_points = int((width//2 - seg_length_sr//2) % ((1-args.overlap)*seg_length_sr))
                    if rm_points == 0:
                        segments = segmentize(data, seg_length_sr, int(seg_length_sr*args.overlap))
                    else:
                        segments = segmentize(data[:,rm_points:-rm_points], seg_length_sr, int(seg_length_sr*args.overlap))
                    
                    points_per_seg = segments.shape[1] # number of slices (points) to compute to FD of within each segment
                    segments = segments.reshape(segments.shape[0]*segments.shape[1], segments.shape[2]) # reshape to be a 2d-array
                    segs.append(segments) # add segments to list
                    segs_ids.append(np.array([[gpstime, seg_length, chan, points_per_seg] for chan in chan_names])) # add info of data to list
            del segments
            # 1 stack corresponds to each unique set of (sr, seg_length) per gpstime
            stack_shapes = [stack.shape[0] for stack in segs] # shapes of each stack, 
            segs = np.ascontiguousarray(np.vstack(segs), dtype=np.float32) # most of the data is float32, downscale rest to conserve memory
            '''
            segs_white = np.vstack(segs_white) 
            n_segs = segs.shape[0]
            assert n_segs == segs_white.shape[0]
            segs_all = np.ascontiguousarray(np.vstack([segs,segs_white]), dtype=np.float32)
            '''
            fd = compute_fd_parallel_f4(segs, args.dec, args.kstep)
            del segs, data
            gc.collect()
            '''
            print(f'{len(fd)}, {n_segs=}')
            [fd, fd_white] = np.split(fd, [n_segs])
            print(f'{fd.shape=}, {fd_white.shape=}, ')
            '''
            fd = np.split(fd, np.cumsum(stack_shapes)[:-1]) # split FDs according to stacks
            for segs_fd, seg_ids in zip(fd, segs_ids): # loop over each stack and its info
                # split FDs based on number of slices per channel segment (`points per seg`)
                segs_fd = np.split(segs_fd, np.cumsum(seg_ids[:,-1].astype(int))[:-1]) 
                for seg_fd, [gpstime, seg_length, chan, points_per_seg] in zip(segs_fd, seg_ids):
                    try:
                        frac_dims[str(gpstime)][f'sl-{seg_length}'][chan_map[chan]] = seg_fd
                    except KeyError: # if array for `seg_length` is not made yet, make array first
                        frac_dims[str(gpstime)][f'sl-{seg_length}'] = np.zeros(shape=(len(channels), int(points_per_seg)), dtype=seg_fd.dtype)
                        frac_dims[str(gpstime)][f'sl-{seg_length}'][chan_map[chan]] = seg_fd
                    '''
                    try:
                        frac_dims_white[str(gpstime)][f'sl-{seg_length}'][chan_map[chan]] = seg_fd_white
                    except KeyError: # if array for `seg_length` is not made yet, make array first
                        frac_dims_white[str(gpstime)][f'sl-{seg_length}'] = np.zeros(shape=(len(channels), int(points_per_seg)), dtype=seg_fd_white.dtype)
                        frac_dims_white[str(gpstime)][f'sl-{seg_length}'][chan_map[chan]] = seg_fd_white
                    '''
            
        with h5py.File(args.filename, 'a') as f:
            for gpstime in time_list:
                label = str(gpstime)
                
                # save qscan data
                #qscan = np.load(file_dict[str(gpstime)])['qscan']
                #f['qscan'].create_dataset(label, data=qscan)
                
                # prepare to save fd data
                split_idx = np.cumsum([frac_dims[str(gpstime)][f'sl-{sl}'].shape[1] for sl in args.segs])
                split_idx = [0, *split_idx]
                # create dataset for gpstime
                # ds shape is (n_channels, fd_forall_seglens)
                # thus data for one channel will be [a_1, ..., a_n, b_1, ... b_m, ...] with a_i and b_j FDs for different seg_lens
                ds_shape = (len(channels), split_idx[-1])
                ds_dtype = frac_dims[str(gpstime)][f'sl-{args.segs[0]}'].dtype
                
                f['fd'].create_dataset(label, shape=ds_shape, dtype=ds_dtype)
                #f['fd'][label].attrs['gpstime'] = gpstime
                
                chans_failed_whiten = np.load(file_dict[str(gpstime)])['chans_failed_whiten']
                f['fd'][label].attrs.create('chans_failed_whiten', chans_failed_whiten, chans_failed_whiten.shape, dtype=h5py.special_dtype(vlen=str))
                
                for i, sl in enumerate(args.segs):
                    f['fd'][label][:, split_idx[i]:split_idx[i+1]] = frac_dims[str(gpstime)][f'sl-{sl}']
                    #f['fd_white'][label][:, split_idx[i]:split_idx[i+1]] = frac_dims_white[str(gpstime)][f'sl-{sl}']
                times = times[times != gpstime] # remove completed time
                
                if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: removing {file_dict[str(gpstime)]}')
                try:
                    os.remove(file_dict[str(gpstime)]) # remove the data file to clear up space
                except OSError as e:
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Error: {e.filename} - {e.strerror}.', flush=True) 
                    if os.path.isfile(file_dict[str(gpstime)]):
                        raise(e) # Raise error if file is still there
                    else: 
                        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Error: {e.filename} - {e.strerror}.', flush=True) # only print error if file doesn't exist anymore
            
            if 'seg_len_start_idx' not in f['metadata']: # TODO: add this at creation of file 
                f['metadata'].create_dataset('seg_len_start_idx', data=split_idx)
            else:
                assert np.all(split_idx == f['metadata']['seg_len_start_idx'][:])
        
        np.save(f'times/{args.times}', times) # save to folder that will be returned on condor job exit
        if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: {len(time_list)} times, finished in: {time.time()-start_time} s', flush=True)
        
    if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FD: Exiting', flush=True) 

def fetch_data(times_queue, data_queue, args, times, channels, max_retries, pid):
    while True: # loop until broken by sentinel value `None`
        gpstime = times_queue.get()
        start_time = time.time()
        if gpstime is None: # no more times left, end process
            data_queue.put((None, None))
            break
        retry_count = 0
        while retry_count <= max_retries: # retry to get the same time in case of errors
            try:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Fetch {pid}: fetching {gpstime}, attempt: {retry_count+1}', flush=True)
                urls = gwdatafind.find_urls("L", "L1_R", gpstime-(args.width+args.scratch)/2, gpstime+(args.width+args.scratch)/2)
                files = []
                files_start = np.inf
                for url in urls:
                    files_start = min(files_start, int(url.split('.')[0].split('-')[-2]))
                    frame_length = int(url.split('.')[0].split('-')[-1])
                    url_file = args.frame_prefix + url.split('/O3/')[1]
                    if pathlib.Path(url_file).is_file():
                        files.append(url_file)
                    else:
                        if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Fetch {pid}: Found unmatched frame for {gpstime=}, falling back to gwdatafind.find_urls()', flush=True)
                        files.append(url)
                
                files_end = files_start + len(urls)*frame_length
                dist_l, dist_r = gpstime-files_start, files_end-gpstime
                if dist_l >= dist_r:
                    psd_start, psd_end = files_start, files_start+args.psd_length
                else:
                    psd_start, psd_end = files_end-args.psd_length, files_end
                
                psd_eval = gwpy.timeseries.TimeSeriesDict.read(files, channels[1:], start=psd_start, end=psd_end)
                data = gwpy.timeseries.TimeSeriesDict.read(files, channels[1:], start=gpstime-(args.width+args.scratch)/2, end=gpstime+(args.width+args.scratch)/2)
                
                # add the strain, since its not in L1_R files
                psd_eval[channels[0]] = gwpy.timeseries.TimeSeries.get(channels[0], start=psd_start, end=psd_end)
                data[channels[0]] = gwpy.timeseries.TimeSeries.get(channels[0], start=gpstime-(args.width+args.scratch)/2, end=gpstime+(args.width+args.scratch)/2)
            except Exception as e:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Fetch {pid}: error fetching {gpstime}: {e}, attempt: {retry_count+1}')
                #traceback.print_exc()
                retry_count += 1
                continue
            break # break while loop if succesful
        if retry_count > max_retries: # skip getting data for `gpstime` if it fails after `max_retries` attempts
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Fetch {pid}: Failed to get data for time {gpstime} after {retry_count+1} attempts', flush=True)
            continue
        
        whitened_data = {}
        chans_failed_whiten = []
        for chan in psd_eval.keys():
            seg_len = 4.
            seg_overlap = seg_len/2
            
            ASD = psd_eval[chan].asd(seg_len, overlap=seg_overlap, method='welch', window='hanning')
            '''
            if chan == channels[0]:
                # compute qscan of main strain
                qscan = data[channels[0]].q_transform(gps=gpstime, whiten=ASD, fduration=args.scratch, **args.qscankwargs)
            '''
            if np.all(data[chan].value == 0.):
                white_data = data[chan]
                chans_failed_whiten.append([chan, 'data_all_zeros']) # data is all zeros
            else:
                white_data = data[chan].whiten(window='hanning', asd=ASD, fduration=args.scratch, highpass=None) # TODO: highpass?
                if np.any(np.isnan(white_data.value)): # try without ASD
                    white_data = data[chan].whiten(window='hanning', fduration=args.scratch, highpass=None)
                    if np.any(np.isnan(white_data.value)): # don't whiten
                        white_data = data[chan]
                        chans_failed_whiten.append([chan, 'failed_whiten_both']) # whiten failed with and without ASD
                    else:
                        chans_failed_whiten.append([chan, 'failed_whiten_asd']) # whiten failed with ASD, succeeded without
            
            # whitened data will have fduration/2 corrupted segments at beginning and end
            whitened_data[chan] = white_data.value[int(args.scratch*data[chan].sample_rate.value/2):-int(args.scratch*data[chan].sample_rate.value/2)]
        
        # dict for data and channels names keyed by sampling rate
        chan_data = {str(key):[] for key in (2**np.arange(10,14+1)).astype(int)}
        #chan_data_white = {'white_'+str(key):[] for key in (2**np.arange(10,14+1)).astype(int)}
        chan_names = {'names_'+str(key):[] for key in (2**np.arange(10,14+1).astype(int))}
        for chan in whitened_data.keys(): # put data and name of each channel in dicts
            sr = int(data[chan].sample_rate.value)
            chan_names['names_'+str(sr)].append(chan)
            #chan_data[str(sr)].append(data[chan].value[int(args.scratch*sr/2):-int(args.scratch*sr/2)])
            chan_data[str(sr)].append(whitened_data[chan])
        fname = f'{int(gpstime)}_data.npz'
        np.savez(fname, **chan_names, **chan_data, chans_failed_whiten=np.asarray(chans_failed_whiten)) # save data, names, and qscan to .npz file
        del data, chan_names, chan_data, chans_failed_whiten
        gc.collect()
        if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Fetch {pid}: {gpstime}, in: {time.time()-start_time} s', flush=True)
        data_queue.put((gpstime, fname)) # add data to data_queue. Will block indefintely if queue is full, add timeout?


def parse_command_line():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--times', type=str, help='GPS times at center of interval in .npy file')
    parser.add_argument('-w', '--width', type=int, help='width of time interval in seconds')
    parser.add_argument('-s', '--segs', nargs='+', type=float, help='Length of segments in seconds (can pass multiple)')
    parser.add_argument('-c', '--channels', type=str, help='Path to .csv file containing list of channels', 
                        default='L1_use_channels_v2.csv')
    parser.add_argument('-p', '--path', type=str, help='Output directory', 
                        default='./')
    parser.add_argument('--scratch', type=float, help='Padding (left+right) in seconds for dealing with corrupted data from whitening',default=4)
    parser.add_argument('--psd_length', type=float, help='Length of the data to estimate the PSD over for whitening',default=20)
    parser.add_argument('--frame_prefix', type=str, help='Directory of frame files on disk, up to /O3/, in the same format as returned by gwdatafind.find_urls', default='/scratch/robin.vanderlaag/temp.copy.of.O3.raw.frames/O3/')
    parser.add_argument('-o', '--overlap', type=float, help='Fraction of overlap between consecutive segments', metavar='[0,1)',default=0.0)
    parser.add_argument('-d', '--dec', type=int, help='Decimate factor for the fractal dimension estimator for segments of length 16384, scales for different lengths automatically', default=32)
    parser.add_argument('-k', '--kstep', type=int, help='Step size for the k_n used in the FD computation', default=1)
    parser.add_argument('-q', '--maxqueuesize', type=int, help='Maximum size of the FD computing queue to limit memory', default=5)
    parser.add_argument('-r', '--maxretries', type=int, help='Number of retries for each data fetch attempt', default=0)
    parser.add_argument('-n', '--nproc', type=int, help='Number of data fetching processes', default=5)
    parser.add_argument('-m', '--timesparallel', type=int, help='Number gpstimes to compute the fd of at the same time', default=10)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print additional status update')
    parser.add_argument('--cont', action='store_true', help='Continues from previous point, only if file exists')
    parser.add_argument('--whiten', action='store_true', help='Whiten the data')
    
    args = parser.parse_args()
    
    if args.overlap >= 1: 
        parser.error(f'Invalid choice of overlap: overlap >= 1. overlap={args.overlap}')
    return args

if __name__ == '__main__':
    args = parse_command_line()
    
    # create times diretory to save/track not completed times, useful incase condor returns before process is finished
    pathlib.Path('times').mkdir(exist_ok=True)
    pathlib.Path(args.path).mkdir(exist_ok=True)
    
    channels = pd.read_csv(args.channels)['channel'].tolist()
    '''
    # these channels don't seem to be available for most gpstimes, thus remove
    err_chans = ['L1:OAF-OPTIC_IMCL_OUT_DQ', 'L1:OAF-OPTIC_PRCL_OUT_DQ', 'L1:OAF-OPTIC_SRCL_OUT_DQ', 'L1:OAF-OPTIC_XARM_OUT_DQ',
                 'L1:OAF-OPTIC_YARM_OUT_DQ', 'L1:OAF-OPTIC_MICH_OUT_DQ', 'L1:OAF-OPTIC_CARM_OUT_DQ', 'L1:OAF-OPTIC_DARM_OUT_DQ']
    channels = channels[~channels['channel'].isin(err_chans)]
    
    channels = channels['channel'].tolist() # convert to list
    '''
    chan_map = {chan: i for i, chan in enumerate(channels)}
    
    #args.qscankwargs = {'qrange':[4,64], 'frange':[10, 2048], 'search':0.5, 'tres':0.002, 'fres':0.5}
    
    # load times
    times = np.load(args.times)
    if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {len(times)} gpstimes loaded.')
    
    # set output filename based on date and Job id
    job_id = args.times.split('_')[0]
    #today = datetime.datetime.now()
    prev_files = glob.glob(f"{args.path}/FD_{job_id}_n={len(times)}*.hdf5")
    print(prev_files)
    print(args.cont)
    if len(prev_files)>0 and args.cont:
        assert len(prev_files) == 1, 'Multiple files found!'
        args.filename = prev_files[0]
        with h5py.File(args.filename, 'r') as f:
            assert 'fd' in f
            assert 'metadata' in f
            for md in ['whitened', 'seg_lengths', 'total_width', 'scratch', 'psd_length', 
                       'decimate', 'kstep', 'overlap', 'channels', 'times']:
                assert md in f['metadata']
            assert f['metadata']['whitened'][...] == args.whiten
            assert np.all(f['metadata']['seg_lengths'][:] == args.segs)
            assert f['metadata']['total_width'][...] == args.width
            assert f['metadata']['scratch'][...] == args.scratch
            assert f['metadata']['psd_length'][...] == args.psd_length
            assert f['metadata']['decimate'][...] == args.dec
            assert f['metadata']['kstep'][...] == args.kstep
            assert f['metadata']['overlap'][...] == args.overlap
            assert np.all(f['metadata']['channels'][:].astype(str) == np.asarray(channels, dtype=str))
            
            assert f['metadata']['times'][:].shape == times.shape
            assert np.all(np.isin(f['metadata']['times'][:], times))
            try:
                times = np.load(f'times/{args.times}')
            except FileNotFoundError:
                for key in f['fd'].keys():
                    times = times[times != float(key)]
            
        if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Continueing previous file: {len(times)} gpstimes loaded.')
            
    else:
        if args.cont: print("No file to continue could be found, starting from beginning")
        args.filename = f"{args.path}/FD_{job_id}_n={len(times)}_{time.strftime('%Y-%m-%d')}.hdf5"
        # create output file 
        with h5py.File(args.filename, 'w') as f:
            f.create_group('fd')
            f.create_group('metadata')
            # set metadata values 
            f['metadata'].create_dataset('whitened', data=args.whiten)
            f['metadata'].create_dataset('seg_lengths', data=args.segs)
            f['metadata'].create_dataset('total_width', data=args.width)
            f['metadata'].create_dataset('scratch', data=args.scratch)
            f['metadata'].create_dataset('psd_length', data=args.psd_length)
            f['metadata'].create_dataset('decimate', data=args.dec)
            f['metadata'].create_dataset('kstep', data=args.kstep)
            f['metadata'].create_dataset('overlap', data=args.overlap)

            f['metadata'].create_dataset('channels', len(channels), dtype=h5py.special_dtype(vlen=str))
            f['metadata']['channels'][:] = channels
            f['metadata'].create_dataset('times', data=times)
    if args.verbose: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Output path set to: {args.filename}')
    
        
    
    # initialize queues
    times_queue = Queue()
    data_queue = Queue(maxsize=args.maxqueuesize) # maxqueuesize set to prevent too much memory usage
    
    for gpstime in times: # fill times_queue with gpstimes
        times_queue.put(gpstime)
    
    # initialize data fetching processes
    fetch_pool = []
    for i in range(args.nproc):
        times_queue.put(None)
        fetch_pool.append(Process(target=fetch_data, args=(times_queue,data_queue,args,times,channels,args.maxretries, i+1)))
        fetch_pool[i].start()
        time.sleep(0.1) # short pause between starting processes, should be redundant
    
    compute_proc = Process(target=compute_fd, args=(data_queue, args.nproc, times, args, channels, chan_map))
    compute_proc.start()
    
    compute_proc.join() # wait for compute proc to finish
    for i, fetch_proc in enumerate(fetch_pool):
        print(f"joining Fetch {i+1}, PID={fetch_proc.pid}")
        fetch_proc.join(1) # 1 second timeout, since it should have already finished if compute_proc finished
        
        if fetch_proc.is_alive():
            # process didn't join, should only happen if it got stuck and data_queue timed out
            fetch_proc.kill()
            print(f"Fetch {i+1} killed")
        else:
            print(f"Fetch {i+1} joined")