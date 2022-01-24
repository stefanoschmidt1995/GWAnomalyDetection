# aux_segments.py
This file reads in data of the specified auxiliary channels for the specified time ranges and turns it into segments with the specified length and overlap.\
It accesses the data using CVMFS and thus requires valid X.509 credentials. 
More info on this can be found at: https://computing.docs.ligo.org/guide/data/#cvmfs-data-discovery and https://computing.docs.ligo.org/guide/cvmfs/#proprietary-data.

The parameter ***desired_segment_time*** can be changed to specify how long each segment should be.\
The parameter ***data_times*** can be changed to whatever range of time you want. If you want a time outside of the O3b run the .csv file that ***safe_channels*** reads from must also be changed accordingly.\
The .csv file of safe channels for O3b that is required for the script to run can be found at https://dcc.ligo.org/T2000277.

The ouput files are relatively larger in memory (~23.6GB for 30 minutes of data with 0.2s segments and 0.5 overlap).\
When a definitive list of features has been determined this script should be merged with ***aux_fracdim.py*** (and scripts for subsequent features) to preserve disk space.

# aux_fracdim.py
This file reads in the segmented data made using ***aux_segments.py*** and computes the Fractal Dimension of each time bin for all the included channels, using a CUDA kernel writing with Numba.


## Timing
1021 auxiliary channels (Safe ISI channels)
With a decimate factor of **min(32, N/2)**
* aux_clean-no_event  (30 minutes of data) in 26 minutes, 19 seconds
* aux_clean-event     (30 minutes of data) in 26 minutes, 05 seconds
* aux_dq-issues-1     (30 minutes of data) in 25 minutes, 50 seconds
* aux_dq-issues-2     (10 minutes of data) in 08 minutes, 42 seconds
* aux_dq-issues-3     (20 minutes of data) in 16 minutes, 46 seconds
