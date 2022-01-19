import numpy as np
import pandas as pd
import gwdatafind
import gwpy.timeseries
import math
import h5py
from tqdm import tqdm # for progress bar

desired_segment_time = 0.2 # in seconds
# time in seconds per segment
segment_time = math.ceil(desired_segment_time*256)/256 # this ensures GPS time of segment does not change with other sampling rates
overlap = 0.5 # fraction of overlap between consecutive segments

# start and end times for data in the O3b run
data_times = [[1262878818, 1262878818 + 30*60, 'clean-event'], # 30 minutes clean data, with one event
              [1263103218, 1263103218 + 30*60, 'clean-no_event'], # 30 minutes clean data, no events
              [1262698440, 1262698440 + 30*60, 'dq-issues-1'], # 30 minutes mixed data, including glitches
              [1263225858, 1263225858 + 10*60, 'dq-issues-2'], # 10 minutes mixed data, including glitches
              [1262704578, 1262704578 + 20*60, 'dq-issues-3']] # 20 minutes mixed data, including glitches


sampling_rates = [256, 512, 1024, 2048, 4096] # possible sampling rates of the aux channels
# Reads the save channels for L1_o3b | file found here: https://dcc.ligo.org/T2000277
safe_channels = pd.read_csv("L1_o3b_safe.csv", usecols=['Channel']).Channel.to_list()
safe_ISI_list = [chan for chan in safe_channels if 'L1:ISI' in chan] # only ISI channels (seismic)


for start, end, filetag in data_times:
    # create output file for the segments | will override existing files
    with h5py.File(f'./segments/aux_{filetag}.hdf5', 'w') as f:
        meta = f.create_group('meta')
        meta.create_dataset('Start GPS', data=int(start))
        # End GPS will be saved at the end of the script

    # uses gwdatafind to find the locations of the frame files (CVMFS) | https://gwdatafind.readthedocs.io/en/stable/index.html
    # extra info : https://gwpy.github.io/docs/v0.1/timeseries/gwf.html
    #              https://computing.docs.ligo.org/guide/data/#cvmfs-data-discovery
    conn = gwdatafind.connect()
    urls = conn.find_urls("L", "L1_R", start, end) # L1_R : All 'raw' data channels, stored at the native sampling rate
    conn.close() # not sure if needed
    
    prev_file_data = None # will hold left over data from previous file that did not fit into a segment
    for url in tqdm(urls):
        frame_start = int(url.split('-')[4])
        frame_size = int(url.split('-')[5].split('.')[0])
        # read data of all channeles in 'safe_ISI_list' from frame file located at 'url'
        data_dict = gwpy.timeseries.TimeSeriesDict.read(url, safe_ISI_list , start=frame_start, end=frame_start+frame_size) # start, end possibly not needed?
        
        # 2d lists to store data and channel names (tags) sorted by their sampling rates | sort is necessary for calculation of fractal dimension 
        data_all = [[], [], [], [], []]
        tags = [[], [], [], [], []]
        for key in data_dict.keys(): # loop over all aux channels and fill 2d lists
            idx = int(math.log(data_dict[key].sample_rate.value, 2))-8 # convert sampling rate to index for 2d lists
            data_all[idx].append(data_dict[key].value)
            tags[idx].append(key)
        
        # loop to convert all lists to np arrays
        for i in range(len(data_all)):
            if prev_file_data is not None: # add leftover data from previous frame file to beginning
                data_all[i] = np.concatenate((prev_file_data[i], data_all[i]), axis=1)
            else:
                data_all[i] = np.asarray(data_all[i])
        
       
        prev_file_data = []
        all_segs = [] # list to store segment data
        for data, sample_rate in zip(data_all, sampling_rates):
            data_strides = data.strides 
            seg_length = int(segment_time * sample_rate) # number of points per segment
            
            overlap_length = int(seg_length*overlap) # number of points that overlap with next segment
            num_segs = int( (data.shape[1] - overlap_length)/(seg_length - overlap_length) ) # number of segments
            
            shape = (data.shape[0], num_segs, seg_length)
            strides = (data_strides[0], data_strides[1]*(seg_length-overlap_length), data_strides[1])
            
            # makes segments | shape = (channels, number of segments, segment length)
            segs = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
            
            all_segs.append(segs)
            # if there are too feww data points at the end for a new segment we concatenate it with the data from the next file
            prev_file_data.append(data[:,int(num_segs*seg_length*overlap):])
        
        # save segments to file before going to next url
        with h5py.File(f'./segments/aux_{filetag}.hdf5', 'a') as f:
            if 'data' not in f: # create data group and dataset if this is the first url/iteration
                f.create_group('data')
                for segs, sample_rate, tag in zip(all_segs, sampling_rates, tags):
                    ds = f['data'].create_dataset(f'{int(sample_rate)}Hz',
                                                  data=segs,
                                                  maxshape=(segs.shape[0],None,segs.shape[2]))
                    # maxshape[1] = None, so that we can extend the dataset
                    ds.attrs['Channel names'] = tag
            else:
                for segs, sample_rate, tag in zip(all_segs, sampling_rates, tags):
                    ds = f['data'][f'{int(sample_rate)}Hz']
                    # asserts should always pass
                    # in case the input data is changed these might fail if there is something wrong/different
                    assert ds.shape[0] == segs.shape[0], 'Mismatched shape for axis=0'
                    assert ds.shape[2] == segs.shape[2], 'Mismatched shape for axis=2'
                    assert np.all(ds.attrs['Channel names'] == tag), f'Mismatched order of channels!\n{ds.attrs["Channel names"]}\n{tag}'
                    ds_size = ds.shape[1]
                    ds.resize(ds_size + segs.shape[1], axis=1) # extend dataset along the correct axis
                    ds[:,ds_size:] = segs # add new data to correct axis
                
                
    with h5py.File(f'./segments/aux_{filetag}.hdf5', 'a') as f:
        meta = f['meta']
        missed_points = prev_file_data[0].shape[1] # number of datapoints that could not be made into a new segment at the very end
        meta.create_dataset('End GPS', data=int(end)-missed_points/sampling_rates[0]) # change End GPS to exclude the missed points
                
        
                
                
                