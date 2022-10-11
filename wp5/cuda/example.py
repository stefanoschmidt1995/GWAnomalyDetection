import numpy as np
import gwpy.timeseries
from tqdm.auto import tqdm

from estimateFD import get_fd

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
    num_segs = int((data.shape[0] - overlap)/(seg_length - overlap))
    
    seg_shape = (num_segs, seg_length)
    seg_strides = (data_strides[0]*(seg_length - overlap), data_strides[0])
    
    segs = np.lib.stride_tricks.as_strided(data, shape=seg_shape, strides=seg_strides)
    if overlap != 0:
        segs = np.ascontiguousarray(segs) 
    
    return segs




if __name__ == '__main__':
    t_start, t_end = (1262476261, 1262481170)
    channel = 'L1:DCS-CALIB_STRAIN_C01' # can be any channel available on the cluster
    
    strain_data = gwpy.timeseries.TimeSeries.get(channel, t_start, t_end, verbose=True)
    sample_rate = strain_data.sample_rate.value
    strain_data = strain_data.value # numpy.ndarray
    
    overlap = 0
    segment_time = 1. # seconds
    # assures that segments have the same length in seconds regardless of the channels sampling rate (256 is the lowest sampling rate)
    seg_length = int(np.ceil(segment_time*256)/256 * sample_rate) 
    
    # check if seg_length is even
    if ~seg_length&1: 
        # want seg_length uneven because FD is calculated over ranges centered on the middle of the segment
        seg_length -= 1
    
    # segmentize the data
    segs = segmentize(strain_data, seg_length, overlap)
    
    # calculate the fractal dimension
    decimate = 32
    estimator = 'VAR'
    frac_dims = get_fd(segs, estimator=estimator, decimate=decimate, gpu=0)
    
    # save everything in .npz file | example
    np.savez_compressed('example_fd', 
                        frac_dims=frac_dims,
                        seg_length=seg_length,
                        overlap=overlap,
                        decimate=decimate,
                        estimator=estimator,
                        t_start=t_start,
                        t_end=t_end,
                        channel=channel)
    
        
        
