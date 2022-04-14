import numpy as np
import bisect
import json, urllib.request, urllib.error
from pathlib import Path
import lal.gpstime
import glob
import h5py
import pycbc.types, pycbc.filter
import scipy.fft
from scipy import stats
from numba import njit, prange


def get_GWTCs(start_time, end_time, time_ranges = [1126051217, 1137254417, 1164556817, 1187733618, 1238166018, 1253977218, 1256655618, 1269363618], GWTC_list = [None,'GWTC-1',None,'GWTC-1',None,'GWTC-2.1',None,'GWTC-3',None]):
    """
    Finds the GWTC tag(s) where the data for the given GPS time intervals can be found using https://www.gw-openscience.org/eventapi/html/.

    Parameters
    ----------
    start_time : int
        Start time of time interval (GPS time).
    end_time : int
        End time of time interval (GPS time).
    time_ranges : List of int, optional
        List of begin and end times of intervals of tags. 
        Fill in like: [begin_1, end_1, begin_2, end_2, ... , begin_n, end_n].
        If changed from default must change GWTC_list accordingly!
        The default is [1126051217, 1137254417, 1164556817, 1187733618, 1238166018, 1253977218, 1256655618, 1269363618].
    GWTC_list : TYPE, optional
        List of None and GWTC tags of time intervals in time_ranges. 
        Fill in like: [None, tag_1, None, tag_2, ... , None, tag_n]. 
        If changed from default must change time_ranges accordingly!
        The default is [None,'GWTC-1',None,'GWTC-1',None,'GWTC-2.1',None,'GWTC-3',None].

    Returns
    -------
    tags : List of str
        List of desired GWTC tag(s).

    """
    
    assert start_time <= end_time, "start_time has to be smaller than end_time"
    #assert len(time_ranges) == len(GWTC_list), "time_ranges and GWTC_list should have same length"
    
    start_idx = bisect.bisect_left(time_ranges, start_time)
    end_idx = bisect.bisect_left(time_ranges, end_time)
    
    tags = [gwtc for gwtc in GWTC_list[start_idx:end_idx+1] if gwtc is not None]
    
    return tags


def get_event_times(start_time, end_time, margin=300, observation_run = None, event_type = 'confident'):
    """
    Get the times of (confident) GW events within GPS time interval [start_time, end_time]

    Parameters
    ----------
    start_time : int
        Start GPS time of interval.
    end_time : int
        End GPS time of interval.
    margin : int, optional
        Margin around confirmed event. The default is 300.
    observation_run : str, optional
        Name of the desired observation run (not needed). The default is None.
    event_type : TYPE, optional
        Type of event catalog to retrieve data from (other options is <marginal>). The default is 'confident'.

    Returns
    -------
    event_times : List of int
        List of GPS times of events within given GPS time interval.
    margin : int
        Margin around confirmed event. The default is 300.

    """
    
    GWTCs = get_GWTCs(start_time, end_time)
    
    event_times = []
    for GWTC in GWTCs:
        url = f'https://www.gw-openscience.org/eventapi/json/{GWTC}-{event_type}/'

        req = urllib.request.Request(url)
        
        try:
            response = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            print(f'Error code: {e}')
            continue #return event_times
        except urllib.error.URLError as e:
            print(f'Reason: {e.reason}')
            continue #return event_times
        else:
            data = json.loads(response.read().decode('utf-8'))
        
        
        for event in data['events'].items():
            time = event[1]['GPS']
            if time + margin >= start_time and time - margin <= end_time:
                event_times.append(time)
    
    return event_times, margin


def create_datasets(start, end, data_prefix='L-L1_GWOSC_O3b_16KHZ_R1-', cache_dir='cache/', file_prefix='', output_dir='strain_files/', padding=0):
    """
    Creates dataset for the specified time range in (almost) identical fashion
    as GWOSC data.

    Parameters
    ----------
    start : int
        Start GPStime in seconds.
    end : int
        End GPStime in seconds.
    data_prefix : str, optional
        Prefix of strain file names (everything before the GPStime). 
        The default is 'L-L1_GWOSC_O3b_16KHZ_R1-'.
    cache_dir : str, optional
        Directory for the cached strain files. 
        The default is 'cache/'.
    file_prefix : str, optional
        Prefix of the output file name.
        The default is ''.
    output_dir : str, optional
        Directory for the output files. 
        The default is 'strain_files/'.
    padding : int, optional
        Padding for the time interval [start-padding, end+padding]
        to allow room for cropping in preprocessing. 
        The default is 0.

    Returns
    -------
    None.

    """
    if not isinstance(start, int) or not isinstance(end, int):
        start = int(float(start))
        end = int(float(end))
    
    start -= padding
    end += padding
    
    available_files = glob.glob(f'{cache_dir}{data_prefix}*.hdf5')

    files_start_times = []
    for file_name in available_files:
        files_start_times.append(int(file_name.strip().split('-')[2]))

    starts = start-np.asarray(files_start_times)
    ends = end-np.asarray(files_start_times)

    s_min_idx, e_min_idx = np.where(starts>0, starts, np.inf).argmin(), np.where(ends>0, ends, np.inf).argmin()
    assert s_min_idx <= e_min_idx
    assert starts[s_min_idx] < 4096 # TODO: make variable
    assert ends[e_min_idx] < 4096 # TODO: make variable
    
    # create output_dir directory (and parents) if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with h5py.File(f'./{output_dir}{file_prefix}{start}.hdf5', 'w') as f_dest:
        with h5py.File('./'+available_files[s_min_idx], 'r') as f_src:
            s_num = int(starts[s_min_idx])
            e_num = int(ends[e_min_idx])
            extra_e_num = 0 if s_min_idx == e_min_idx else 4096 # TODO: make variable
            hz = int(1/f_src['strain']['Strain'].attrs['Xspacing'])

            f_src.copy(f_src['/meta'], f_dest['/'])
            f_dest['meta']['Duration'][()] = int(end-start)
            f_dest['meta']['GPSstart'][()] = int(start)
            UTCstart = lal.gpstime.gps_to_utc(int(start))
            f_dest['meta']['UTCstart'][()] = f'{UTCstart.year}-{UTCstart.month:02d}-{UTCstart.day:02d}T{UTCstart.hour:02d}:{UTCstart.minute:02d}:{UTCstart.second:02d}'
            f_dest['meta']['Padding'] = padding
            
            f_src.copy(f_src['/quality'], f_dest['/'])
            if s_min_idx == e_min_idx: # for some reason the reesize and assignment have to be immediately after eachother
                temp_val = f_dest['quality']['injections']['Injmask'][s_num:e_num] # +1 at end? will give shape of 1201 instead of 1200 :/
            else:
                with h5py.File('./'+available_files[e_min_idx], 'r') as f_src2:
                    temp_val = np.concatenate((f_dest['quality']['injections']['Injmask'][s_num:], f_src2['quality']['injections']['Injmask'][:e_num]))
            f_dest['quality']['injections']['Injmask'].resize((extra_e_num+e_num-s_num,))
            f_dest['quality']['injections']['Injmask'][()] = temp_val
            f_dest['quality']['injections']['Injmask'].attrs['Npoints'] = extra_e_num+e_num-s_num
            f_dest['quality']['injections']['Injmask'].attrs['Xstart'] = int(start)
            if s_min_idx == e_min_idx: # for some reason the reesize and assignment have to be immediately after eachother
                temp_val = f_dest['quality']['simple']['DQmask'][s_num:e_num] # +1 at end? will give shape of 1201 instead of 1200 :/
            else:
                with h5py.File('./'+available_files[e_min_idx], 'r') as f_src2:
                    temp_val = np.concatenate((f_dest['quality']['simple']['DQmask'][s_num:], f_src2['quality']['simple']['DQmask'][:e_num])) # +1 at end? will give shape of 1201 instead of 1200 :/
            f_dest['quality']['simple']['DQmask'].resize((extra_e_num+e_num-s_num,))
            f_dest['quality']['simple']['DQmask'][()] = temp_val
            f_dest['quality']['simple']['DQmask'].attrs['Npoints'] = extra_e_num+e_num-s_num
            f_dest['quality']['simple']['DQmask'].attrs['Xstart'] = int(start)
            

            f_src.copy(f_src['/strain'], f_dest['/'])
            if s_min_idx == e_min_idx: # for some reason the reesize and assignment have to be immediately after eachother
                temp_val = f_dest['strain']['Strain'][hz*s_num:hz*e_num]
            else:
                with h5py.File('./'+available_files[e_min_idx], 'r') as f_src2:
                    temp_val = np.concatenate((f_dest['strain']['Strain'][hz*s_num:], f_src2['strain']['Strain'][:hz*e_num]))
            f_dest['strain']['Strain'].resize((hz*(extra_e_num+e_num)-hz*s_num,))
            f_dest['strain']['Strain'][()] = temp_val
            f_dest['strain']['Strain'].attrs['Npoints'] = hz*(extra_e_num+e_num)-hz*s_num
            f_dest['strain']['Strain'].attrs['Xstart'] = int(start)
    
    
def get_strain_files(dataset, detector, start_times, end_times, padding = 0, file_format = 'hdf5', save_dir = 'cache/', overwrite=False):
    """
    Downloads the strain data file(s) from gw-openscience.org for given dataset, detector, and GPS time interval.

    Parameters
    ----------
    dataset : str
        Name of the desired dataset (https://www.gw-openscience.org/data/).
    detector : str
        Acronym of the desired detector ('L1', 'V1', 'H1').
    start_times : int or list of int
        Start GPS time(s) of interval.
    end_times : int or list of int
        End GPS time(s) of interval.
    file_format : str, optional
        Format of file type ('hd5f' or 'gwf'). The default is 'hdf5'.
    save_dir : str, optional
        Directory to save files to (dir and parents will be made if needed). The default is 'cache/'.

    Returns
    -------
    downloaded_files : List of str
        List of paths to downloaded files.

    """
    assert type(start_times) == type(end_times)
    
    if isinstance(start_times,(list,np.ndarray)):
        downloaded_files = []
        for s, e in zip(start_times, end_times):
            downloaded_files.extend(get_strain_files(dataset, detector, s, e, padding, file_format, save_dir, overwrite))
        return downloaded_files
    
    if not isinstance(start_times, int):
        start_times = int(float(start_times))
        end_times = int(float(end_times))
    
    # expand the time range slightly to make sure that 1min on each side can be cropped away without losing desired data
    start_times -= padding
    end_times += padding
    
    url = 'https://gw-openscience.org/archive/links/{0}/{1}/{2}/{3}/json/'.format(dataset, detector, start_times, end_times)
    req = urllib.request.Request(url)
    #print(url)
    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        print(f'Error code: {e}')
        return []
    except urllib.error.URLError as e:
        print(f'Reason: {e.reason}')
        return []
    else:
        data = json.loads(response.read().decode('utf-8'))
    
    to_download = []
    for file in data['strain']:
        if file['format'] == file_format:
            if not Path(save_dir+file['url'].strip().split('/')[-1]).is_file() or overwrite:
                to_download.append(file['url'])
                
            
    print(f'Files to download: {to_download}')
    # create save_dir directory (and parents) if needed
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    for file_dl_link in to_download:
        filename = file_dl_link.strip().split('/')[-1]
        gpstime = int(filename.split('-')[2])
        # download file to location
        print(f'Downloading: {file_dl_link}')
        urllib.request.urlretrieve(file_dl_link, save_dir+filename)
        downloaded_files.append(save_dir+filename)
    
    print(f'Download of {dataset}-{detector}-{start_times}-{end_times} complete')
    return downloaded_files


def plot_dfs(dfs, xmin=None, xmax=None, save=False):
    """
    Plots list of gwpy.DataQualityFlag  within a given range

    Parameters
    ----------
    dfs : list of gwpy.DataQualityFlag
        DESCRIPTION.
    xmin : int, optional
        Minimum value of the x-axis (GPStime). 
        The default is None.
    xmax : int, optional
        Maximum value of the x-axis (GPStime). 
        The default is None.
    save : bool, optional
        Toggle saving of file.
        The defaullt is False.

    Returns
    -------
    None.

    """
    plot = dfs[0].plot()
    ax = plot.gca()
    for df in dfs[1:]:
        ax.plot(df)
    auto_xmin, auto_xmax = ax.get_xlim()
    xmin = xmin if xmin is not None else auto_xmin
    xmax = xmax if xmax is not None else auto_xmax
    ax.set_xlim(xmin, xmax)
    if save:
        name = f'{save}-{int(xmin)}' if isinstance(save, str) else str(int(xmin))
        plot.save(f'{name}.png', bbox_inches='tight')
    else:
        plot.show()


def process_strain(strain, epoch, dt, padding, sample_rate=4096, freq_range=None):
    """
    Does Preproccesing of the GW strain (whiten, downsample, bandpass)

    Parameters
    ----------
    strain : numpy.ndarray
        Numpy array of the raw strain data.
    epoch : float
        The epoch of the data (GPSTime).
    dt : float
        Delta t (time steps) of the data.
    padding : TYPE
        Available padding on each side of the data that can be removed (seconds).
    sample_rate : int, optional
        The desired sample rate of the output data. The default is 4096.
    freq_range : tuple of int
        The frequency range used for the band-pass. The default = (30, 500)

    Returns
    -------
    ts : pycbc.types.TimeSeries
        Preproccesed data.
    ts_raw : pycbc.types.TimeSeries
        Only band-passed data.

    """
    if freq_range == None:
        freq_range = (30, 500)
    
    nan_idx = np.argwhere(np.isnan(strain))
    
    ts = pycbc.types.TimeSeries(strain, delta_t=dt, epoch=lal.gpstime.LIGOTimeGPS(float(epoch)))
    
    if nan_idx.size:
        # remove nans
        l_crop, r_crop = 0, 0
        if nan_idx[nan_idx<strain.size-padding/dt].size:
            max_val = max(nan_idx[nan_idx<strain.size-padding/dt])
            assert max_val <= padding/dt, 'Too much to crop out'
            l_crop = max_val*dt
        if nan_idx[nan_idx>padding/dt].size:
            min_val = min(nan_idx[nan_idx>padding/dt])
            assert min_val >= strain.size-padding/dt, 'Too much to crop out'
            r_crop = (strain.size-min_val)*dt
        m_crop = max(l_crop, r_crop) # prefer symmetric cropping
        ts = ts.crop(m_crop, m_crop)
        padding -= m_crop
    
    # TODO: Bandpass before calculating PSD ??
    #psd = pycbc.psd.welch(ts, seg_len=4096*4*4, seg_stride=4096*2*4) # TODO: what values for seg_leb, seg_stride ??
    #psd = pycbc.psd.interpolate(psd, ts.delta_f)
    #psd = pycbc.psd.inverse_spectrum_truncation(psd, 
                                                #max_filter_len=2551809, 
                                                #low_frequency_cutoff=30)
    # TODO: Load in PSD from lal (or elsewhere) instead ??
    
    if 1/dt > sample_rate:
        ts = pycbc.filter.resample_to_delta_t(ts, 1.0/sample_rate)
    
    ts_raw = ts.copy()
    padding_raw = padding
    
    ts = ts.whiten(4, 4)
    padding -= 2
    
    
    #print(ts.delta_f, psd.delta_f)
    #print(len(ts), len(ts.to_frequencyseries()), len(psd))
    #ts = (ts.to_frequencyseries() / psd**0.5).to_timeseries()
    
    ts = ts.highpass_fir(freq_range[0], int(sample_rate/2)).lowpass_fir(freq_range[1], int(sample_rate/2)) 
    padding -= 1
    
    ts_raw = ts_raw.highpass_fir(freq_range[0], int(sample_rate/2)).lowpass_fir(freq_range[1], int(sample_rate/2)) 
    padding_raw -= 1
    
    ts = ts.crop(padding, padding)
    ts_raw = ts_raw.crop(padding_raw, padding_raw)
    
    return ts, ts_raw


def make_segments(ts, ts_raw, seg_length, overlap, chan_dict, padding):
    """
    Makes segments of specified length and overlap.

    Parameters
    ----------
    ts : pycbc.types.TimeSeries
        Total preprocessed TimeSeries that has to be sliced.
    ts_raw : pycbc.types.TimeSeries
        Total raw TimeSeries that has to be sliced.
    seg_length : int
        Desired length of the segments.
    overlap : float {[0,1]}
        Fraction of overlap between consecutive segments (0<=overlap<=1).
    chan_dict : dict of {str : numpy.ndarray}
        Dictionary of data quality flags, sampled at 1Hz.

    Returns
    -------
    s_segs : numpy.ndarray
        Array of preprocessed strain data in segments of shape (num_segs, seg_length).
    s_raw_segs : numpy.ndarray
        Array of raw strain data in segments of shape (num_segs, seg_length).
    epochs : numpy.ndarray
        Array of epochs for each segment of shape (num_segs, ).
    dq_flags : dict of {str : numpy.ndarray}
        Dictionary of data quality flags, one element per segment.
    sample_rate : int
        Sample rate of the segments.

    """
    sample_rate = int(ts.sample_rate)
    s = ts.numpy()
    s_strides = s.strides
    
    sample_rate_raw = int(ts_raw.sample_rate)
    s_raw = ts_raw.numpy()
    s_raw_strides = s_raw.strides
    
    assert sample_rate == sample_rate_raw
    assert s.size == s_raw.size
    
    overlap_length = int(seg_length*overlap)
    num_segs = int( (s.size - overlap_length)/(seg_length - overlap_length) )
    
    s_segs = np.lib.stride_tricks.as_strided(s, shape=(num_segs, seg_length), strides=((seg_length-overlap_length)*s_strides[0], s_strides[0])).copy()
    s_raw_segs = np.lib.stride_tricks.as_strided(s_raw, shape=(num_segs, seg_length), strides=((seg_length-overlap_length)*s_raw_strides[0], s_raw_strides[0])).copy()
    
    epochs = (np.full((num_segs,), (seg_length-overlap_length)*ts.delta_t)*np.arange(num_segs)+float(ts.start_time)) # epochs
    
    dq_flags = {}
    for key, val in chan_dict.items():
        dq = np.repeat(val[padding:-padding], sample_rate)[::seg_length-overlap_length][:num_segs].copy()
        dq_flags[key] = dq
    
    return s_segs, s_raw_segs, epochs, dq_flags, sample_rate


def save_segments(s_segs, s_raw_segs, epochs, dq_flags, sample_rate, meta_dict, file_name, output_dir):
    """
    Save segments to hdf5 file.

    Parameters
    ----------
    s_segs : numpy.ndarray
        Array of preprocessed strain data in segments.
    s_raw_segs : numpy.ndarray
        Array of raw strain data in segments.
    epochs : numpy.ndarray
        Array of epochs, one per segment.
    dq_flags : dict of {str : numpy.ndarray}
        Dictionary of data quality flags, one element per segment.
    sample_rate : int
        Sample rate of the segments..
    meta_dict : dict of {str : str}
        Dictonary of meta data inherited from GWOSC strain file.
    file_name : str
        File name without extension.
    output_dir : str
        Output directory.

    Returns
    -------
    None.

    """
    # create output_dir directory (and parents) if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with h5py.File(f'./{output_dir}{file_name}.hdf5', 'w') as f:
        meta = f.create_group('meta')
        for key, val in meta_dict.items():
            meta[key] = val
        meta['SampleRate'] = sample_rate
        meta['NumberOfSegments'] = s_segs.shape[0]
        meta['SegmentLength'] = s_segs.shape[1]
        
        dataquality = f.create_group('dataquality')
        for key, val in dq_flags.items():
            dataquality[key] = val
        
        data = f.create_group('data')
        data['Strain'] = s_segs
        data['RawStrain'] = s_raw_segs
        data['Epoch'] = epochs


@njit(parallel=True)
def ANAM(f, tau, n):
    """
    Discretized ANAM method used for calculating fractal dimension.
    ~5x slower than VAR method, but more accurate.

    Parameters
    ----------
    f : numpy.ndarray
        Array of data.
    tau : int

    Returns
    -------
    float
        K^tau(f,a,b).

    """
    out = 0
    for i in prange(tau+1, n-tau):
        for j in prange(0,tau+1):
            for l in prange(0, tau+1):
                out += np.abs(f[i+j]-f[i-l])
    return out/(tau+1)**2/(n-2*tau)


def get_fractal_dim(f):
    """
    Calculates the fractal dimension of the strain using ANAM

    Parameters
    ----------
    strain : numpy.ndarray
        Array of strain data.

    Returns
    -------
    dim : float
        Fractal dimension of the strain.

    """
    n = f.size
    taus = np.arange(1,int(n/20))
    ests = []
    
    for tau in taus:
        ests.append(ANAM(f, tau, n))
    
    slope, _, _, _, _ = stats.linregress(np.log(taus),np.log(ests))
    
    dim = 2-slope
    return dim


def generate_features(segs, segs_raw, segs_t, sample_rate=4096):
    """
    Generates features from strain, raw strain and epoch data
    
    Parameters
    ----------
    segs : numpy.ndarray
        Array of strain segments.
    segs_raw : numpy.ndarray
        Array of raw strain segments
    segs_t : numpy.ndarray
        Array of epochs for each segment
    sample_Rate : int, optional
        Sample rate of the strain data in Hz.
    
    Returns
    -------
    frac_dims : numpy.ndarray
        Array of fractal dimension of strain segments
    frac_dims_r : numpy.ndarray
        Array of fractal dimension of raw strain segments
    est_psds : numpy.ndarray
        Array of estimated PSDs of raw strain segments
    freq_bins : numpy.ndarray
        Frequency content of strain segments
    freq_bins_r : numpy.ndarray
        Frequency content of raw strain segments
    
    """
    
    frac_dims, frac_dims_r, est_psds, freq_bins, freq_bins_r = [], [], [], [], []
    
    for s, s_r, t in zip(segs, segs_raw, segs_t):
        frac_dims.append(get_fractal_dim(s))
        frac_dims_r.append(get_fractal_dim(s_r))
        
        # TODO: only PSD of 'raw' strain?
        ts = pycbc.types.TimeSeries(s_r, delta_t=1/sample_rate, epoch=lal.gpstime.LIGOTimeGPS(float(t)))
        psd_temp = pycbc.psd.welch(ts, seg_len=410, seg_stride=410//2)
        
        est_psds.append(psd_temp) # TODO: Change seg_len and seg_stride ??
        
        freq_bins.append(np.abs(scipy.fft.rfft(s)))
        freq_bins_r.append(np.abs(scipy.fft.rfft(s_r)))
    psd_df = psd_temp.delta_f
    
    return np.asarray(frac_dims), np.asarray(frac_dims_r), np.asarray(est_psds), psd_df, np.asarray(freq_bins), np.asarray(freq_bins_r)
