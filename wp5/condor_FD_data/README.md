# Python scripts for the data fetching and FD computing process (HTCondor)

### `condor_setup.py`
Sets up `dag`, `sub`, and `sh` files for submitting to condor, as well as `.npy` files for the gpstimes of each job.
Note that `proxy_path` should be changed to your own X509 proxy certificate (see: https://computing.docs.ligo.org/guide/condor/data/)


### `condor_fd.py`
Fetches and computes the FD of specified (auxiliary) channels [`L1_use_channels.csv`, can be altered] in parallel on nvidia GPUs.
Can run locally (without HTCondor) through command prompt by setting parameters according to:
```
usage: condor_fd.py [-h] [-t TIMES] [-w WIDTH] [-s SEGS [SEGS ...]] [-c CHANNELS] [-p PATH] [-o [0,1)] [-d DEC] [-q MAXQUEUESIZE] [-r MAXRETRIES] [-n NPROC]
                    [-i TIMESPERGPU] [-v] (--var | --anam)

optional arguments:
  -h, --help            show this help message and exit
  -t TIMES, --times TIMES
                        GPS times at center of interval in .npy file
  -w WIDTH, --width WIDTH
                        width of time interval in seconds
  -s SEGS [SEGS ...], --segs SEGS [SEGS ...]
                        Length of segments in seconds (can pass multiple)
  -c CHANNELS, --channels CHANNELS
                        Path to .csv file containing list of channels
  -p PATH, --path PATH  Output directory
  -o [0,1), --overlap [0,1)
                        Fraction of overlap between consecutive segments
  -d DEC, --dec DEC     Decimate factor for the fractal dimension estimator for segments of length 16384, scales for different lengths automatically
  -q MAXQUEUESIZE, --maxqueuesize MAXQUEUESIZE
                        Maximum size of the FD computing queue to limit memory
  -r MAXRETRIES, --maxretries MAXRETRIES
                        Number of retries for each data fetch attempt
  -n NPROC, --nproc NPROC
                        Number of data fetching processes
  -i TIMESPERGPU, --timespergpu TIMESPERGPU
                        Number gpstimes to compute the fd of at the same time
  -v, --verbose         Print additional status update
  --var                 Use the VAR estimator
  --anam                Use the ANAM estimator
  ```
