import numpy as np
import pandas as pd
import numba
import itertools
import pathlib
import gwdatafind
import gwpy.timeseries

################################
# Parameters

# random state/seed set for random sampling
# important since only a few frame files were transfered to disk
# and with this random state/seed it should select those exact files
# if you want to use more/different ones those tape files will have to be 
# transfered to disk first
RNG_STATE = 20220922
np.random.seed(RNG_STATE)

# number of parallel threads to use when computing the fractal dimension
# the computation has little to no downtime for hyperthreading to work well
# so there is no benefit to putting this higher than the number of cores 
# you want/can use
numba.set_num_threads(10) 

# path prefix for the frame files stored on disk
frame_prefix = '/scratch/robin.vanderlaag/temp.copy.of.O3.raw.frames/O3/'

################################
# Part 1: gather the sample times (and save to file)



# load file of clean segment times
clean_segments = np.load('/home/robin.vanderlaag/wp5/strain_fractals/condor_data/clean_segments_O3A.npy')

# load gravity spy triggers csv
gs_triggers = pd.read_csv('/home/robin.vanderlaag/wp5/strain_fractals/condor_data/gspyO3A.csv')

# select the triggers from the type of glitches we want and drop duplicates
# Whistle
df_w = gs_triggers[gs_triggers['label']=='Whistle']
df_w = df_w.drop_duplicates(subset='GPStime', keep='last')
# Tomte
df_t = gs_triggers[gs_triggers['label']=='Tomte']
df_t = df_t.drop_duplicates(subset='GPStime', keep='last')
# Scattered Light
df_s = gs_triggers[gs_triggers['label']=='Scattered_Light']
df_s = df_s.drop_duplicates(subset='GPStime', keep='last')


# remove segments shorter than 2 minute as they will be in regions surrounded by glitches
msk = np.where((clean_segments[:,1]-clean_segments[:,0] >= 2*60))[0]
clean_segments = clean_segments[msk]

num_samples = 1000 # number of samples per class

# maximum number for num_samples for a balanced dataset
# if each glitch type is its own class 
# i.e. (num_clean == num_whistle == num_tomte == num_scattered)
max_balanced_num_samples = min([clean_segments.shape[0], len(df_w), len(df_t), len(df_s)])
num_samples = min(num_samples, max_balanced_num_samples)

# randomly sample `num_sample` times from clean_segments
clean_ids = np.random.choice(np.arange(clean_segments.shape[0]), num_samples, replace=False)
# choose the random times to be in the middle of the chosen segments
clean_times = (clean_segments[clean_ids,0]+clean_segments[clean_ids,1])/2 # take the middle of each segment

# randomly sample `num_sample` times from the glitch dataframes
whistles = df_w.sample(n=num_samples, random_state=RNG_STATE)
tomte = df_t.sample(n=num_samples, random_state=RNG_STATE+1)
scattered = df_s.sample(n=num_samples, random_state=RNG_STATE+2)

# (unique ?) gravity spy indexes so it can be used to look up more information
# like SNR, confidence, ect. of a sampled glitch using the .csv file
# without us having to save every piece of information again
# clean segments are all given index `-1`.
gs_idx = np.concatenate([np.ones(len(clean_times), dtype=int)*-1,  
                         whistles.index.values, 
                         tomte.index.values, 
                         scattered.index.values])

# the times of all the clean/glitch samples
times = np.concatenate([clean_times, 
                        whistles['GPStime'].values, 
                        tomte['GPStime'].values, 
                        scattered['GPStime'].values])

# labels of all the clean/glitch samples
# where I have assigned the labels as follows:
# clean=0, whistle=1, tomte=2, scattered_light=3
labels = np.concatenate([np.zeros(len(clean_times)), 
                         np.ones(num_samples), 
                         np.ones(num_samples)*2, 
                         np.ones(num_samples)*3])

# save these 3 arrays in npz file
# not really needed if all is done in same file I guess
np.savez(f'ids_examples.npz', gs_idx=gs_idx, times=times, labels=labels)

################################
# Part 2: Fractal dimension computation functions!

# function to cut an array into segments of specified length and overlap
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

# main fractal dimension computation function
# uses single precision data (float32), which is a small speedup from float64
# if you want to use float64 you can change the dtypes in this function
# and the `f4` in the `@numbda.njit(...)` to `f8`
@numba.njit("f4[:](f4[:,:], i8, i8)", parallel=True)
def compute_fd_parallel_f4(fs, dec, step):
    """
    Computes the Fractal Dimension using the VAR method 
    and `pyramid summing` for single-precision (float32) data.
    
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

def fetch_data(gpstime, channels, width, scratch, psd_length):
    # first get the paths to the required frame files containing the auxiliary channels
    # (these will be on tape)
    urls = gwdatafind.find_urls("L", "L1_R", gpstime-(width+scratch)/2, gpstime+(width+scratch)/2)
    
    files = [] # empty list to store the paths to the files on disk
    files_start = np.inf # will contain the start of the earliest frame file used to determine total length
    
    for url in urls:
        files_start = min(files_start, int(url.split('.')[0].split('-')[-2])) # start gpstime of the frame file
        frame_length = int(url.split('.')[0].split('-')[-1]) # length of the frame file in seconds
        url_file = frame_prefix + url.split('/O3/')[1] # path to the file on disk
        if pathlib.Path(url_file).is_file():
            # check if the required file was actually in the list that were transfered to disk
            files.append(url_file)
        else: 
            # if the data was not transfered to file I had to default to getting it from tape
            # I've commented this out for now
            # files.append(url)
            pass # remove this if you uncomment the line above

    files_end = files_start + len(urls)*frame_length # end time of the last frame file used
    
    # determine the best time to sample the PSD from, such that it is as
    # far away from `gpstime` as possible within the same frame files
    dist_l, dist_r = gpstime-files_start, files_end-gpstime
    if dist_l >= dist_r:
        psd_start, psd_end = files_start, files_start+psd_length
    else:
        psd_start, psd_end = files_end-psd_length, files_end
    
    # fetch the data using gwpy and the files/channels specified
    # the strain data is not contained in the L1_R files so you only want to read the
    # auxiliary channels here, hence the `channels[1:]` as channels[0] was the strain channel for me
    psd_eval = gwpy.timeseries.TimeSeriesDict.read(files, channels[1:], start=psd_start, end=psd_end)
    data = gwpy.timeseries.TimeSeriesDict.read(files, channels[1:], start=gpstime-(width+scratch)/2, end=gpstime+(width+scratch)/2)
    
    # now add the strain to the TimeSeriesDict, 
    # the strain should always be available on disk (I think), so no fancy stuff needed here
    psd_eval[channels[0]] = gwpy.timeseries.TimeSeries.get(channels[0], start=psd_start, end=psd_end)
    data[channels[0]] = gwpy.timeseries.TimeSeries.get(channels[0], start=gpstime-(width+scratch)/2, end=gpstime+(width+scratch)/2)
    
    # a dict to store the whitened data in
    whitened_data = {}
    # loop over all channels and whiten them
    for chan in psd_eval.keys():
        # segment lengths and overlap used in the ASD approximation
        # can change this if there are better values
        # these settings might go wrong if psd_length < 4
        seg_len = 4.
        seg_overlap = seg_len/2
        ASD = psd_eval[chan].asd(seg_len, overlap=seg_overlap, method='welch', window='hanning')

        # can do checks on the data here to see if whitening is even possible
        if np.all(data[chan].value == 0.): # or perhaps `if np.all(data[chan].value == data[chan.value[0]])` to see if its constant
            # data is all zeros (channel probably offline)
            # default to just using the raw data (will be all zeros)
            white_data = data[chan] 
        else: # should be fine to whiten (? maybe other checks possible first)
            # can high/low-pass here as well (might be tricky to figure out the right frequencies for auxiliary channels)
            # `fduration=scratch`, as we will have some corrupted samples from whitening that we need to remove
            white_data = data[chan].whiten(window='hanning', asd=ASD, fduration=scratch, highpass=None)

            # check if whitening went okay
            if np.any(np.isnan(white_data.value)): # whitened data contained at least one NaN
                # default to whitening with just the data segment itself, no seperate ASD
                white_data = data[chan].whiten(window='hanning', fduration=scratch, highpass=None)
                if np.any(np.isnan(white_data.value)): # whitened data contained at least one NaN
                    # default to raw data (? probably a bad thing to do unless it always fails for this channel)
                    white_data = data[chan]
        
        # whitened data will have fduration/2 corrupted segments at beginning and end
        # according to the documentation of lalsuite
        # put white_data in dict for this channel
        whitened_data[chan] = white_data.value[int(scratch*data[chan].sample_rate.value/2):-int(scratch*data[chan].sample_rate.value/2)]
        
    # a dict for the data and channels names keyed by the sampling rate
    # this is needed because the FD computation can only parallelize arrays of the same size
    # and we need the sampling rate later to figure this out in the `compute_fd()` function
    chan_data = {str(key):[] for key in (2**np.arange(10,14+1)).astype(int)}
    chan_names = {'names_'+str(key):[] for key in (2**np.arange(10,14+1).astype(int))}
    for chan in whitened_data.keys(): # put data and name of each channel in dicts
        sr = int(data[chan].sample_rate.value)
        chan_names['names_'+str(sr)].append(chan)
        chan_data[str(sr)].append(whitened_data[chan])
    
    # in the original code i saved this data to a .npz file to then read it from
    # a different process, as sending lots of data between processes is very slow
    # in this example i will just return it from this function, as it's not a parallel running script
    return chan_data, chan_names

def compute_fd(channels, time_list, chan_data_list, chan_names_list, segs, overlap, dec, kstep):
    # we want to utilise the parallelization of `compute_fd_parallel_f4` to speedup the computation
    # so we will load in data from multiple gpstimes

    # map for index to channel. Used later for storing the FDs in the correct spot
    chan_map = {chan: i for i, chan in enumerate(channels)}

    seglen_dict = {} # dict of all possible segment lengths | sampling_rate*seg_len_seconds
    for key in itertools.product(2**np.arange(10,14+1).astype(int), segs):
        if int(key[0]*key[1]) in seglen_dict.keys():
            seglen_dict[int(key[0]*key[1])].append(key)
        else:
            seglen_dict[int(key[0]*key[1])] = [key]
    
    # output dict for fractal dimensions
    # will end up being nested dict like this:
    # frac_dims[gpstime][segment_length][channel] = np.ndarray of FD 
    frac_dims = {}

    # loop over all possible segment lengths
    # ids_list contains tuples of (sampling_rate, segment_length (seconds))
    for length, ids_list in seglen_dict.items(): # length is unused
        segs = []
        segs_ids = []
        for i, gpstime in enumerate(time_list):
            # chan_data = chan_data_list[i]
            for (sr, seg_length) in ids_list:
                data = chan_data_list[i][str(sr)]
                chan_names = chan_names_list[i][f'names_{sr}']

                # want to segmentize the array according to seg_length and overlap
                width = int(width*sr)
                seg_length_sr = int(seg_length*sr)
                assert seg_length*sr*overlap == np.floor(seg_length*sr*overlap) # assert is whole number
                # number of points to slice away from each side
                rm_points = int((width//2 - seg_length_sr//2) % ((1-overlap)*seg_length_sr))
                # to ensure that the gpstime (possible glitch) is centered in the center segment
                # we have to remove some points to make it fit
                # this part could be removed if you do not care about the centering
                if rm_points == 0:
                    segments = segmentize(data, seg_length_sr, int(seg_length_sr*overlap))
                else:
                    segments = segmentize(data[:,rm_points:-rm_points], seg_length_sr, int(seg_length_sr*args.overlap))
                
                points_per_seg = segments.shape[1] # number of slices (points) to compute to FD of within each segment
                segments = segments.reshape(segments.shape[0]*segments.shape[1], segments.shape[2]) # reshape to be a 2d-array
                segs.append(segments) # add segments to list
                segs_ids.append(np.array([[gpstime, seg_length, chan, points_per_seg] for chan in chan_names])) # add info of data to list
        del segments
        # stack the segments so that we can compute the FDs in parallel 

        # 1 stack corresponds to each unique set of (sr, seg_length) per gpstime
        stack_shapes = [stack.shape[0] for stack in segs] # shapes of each stack, 
        segs = np.ascontiguousarray(np.vstack(segs), dtype=np.float32) # most of the data is float32, downscale rest to conserve memory

        fd = compute_fd_parallel_f4(segs, dec, kstep)
        del segs, data
                
        # computation is done now
        # store them in `frac_dims` dict in a logical order
        # as now they are all intertwined
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
    return frac_dims


################################
# Part 3: Calling Fractal dimension computation functions
#         and main execution

# parameters:

batch_size = 10 # number of gpstimes to compute the FD for in parallel

# list of channels to use as strings
channels = [] # example, fetch_data() expects channels[0] to be the strain channel

width = 8 # width of the main data part to segmentize and compute FD off
segs = [0.25, 0.5, 1, 2] # Length of segments in seconds to compute FD over | can pass multiple

overlap = 0 # Fraction of overlap between consecutive segments
dec = 32 # decimation factor used in computation of FD (saves time at the loss of accuracy)
kstep = 1 # Step size for the k_n used in the FD computation

scratch = 4 # time in seconds to use as buffer for corruption at edges from whitening
psd_length = 20 # length of segment used for PSD estimation (seconds)

# main execution

data_list = []
chan_name_list = []

# loop over all the times in the `times` array from Part 1
# and fetch the data using `fetch_data()`
for gpstime in times:
    chan_data, chan_name = fetch_data(gpstime, channels, width, scratch, psd_length)
    data_list.append(chan_name)
    chan_name_list.append(chan_name_list)

frac_dims_all = {} # store frac_dims from batches in a dict

# loop over all times in batches and compute FD
for i in range(0, len(times), batch_size):
    fd = compute_fd(channels, times[i:i+batch_size], data_list[i:i+batch_size],
                    chan_name_list[i:i+batch_size], segs, overlap, dec)
    frac_dims_all.update(fd)

# frac_dims_all now contains the FD for all times, segment lengths, and channels specified
# and can be accessed using:
# frac_dims_all[gpstime][f'sl-{seg_length}'][chan_map[chan]],
# where `seg_length` is an element of `segs`, and `chan` a channel name in `channels`
