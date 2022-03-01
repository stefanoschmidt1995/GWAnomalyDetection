### get_fd_data.py
Can be used from command line
```
usage: get_fd_data.py [-h] (--var | --anam) [-c CHANNEL] [-w] [-o OVERLAP]
                      [-d DEC] [-g {0,1,2,3}] [-f OUTDIR]
                      start end segtime

positional arguments:
  start                 GPS Start time
  end                   GPS End time
  segtime               Length of segments in seconds

optional arguments:
  -h, --help            show this help message and exit
  --var                 Use the VAR estimator
  --anam                Use the ANAM estimator
  -c CHANNEL, --channel CHANNEL
                        Channel to get strain data from, pass "open{ifo}" to
                        use open data
  -w, --whiten          Whiten strain data
  -o OVERLAP, --overlap OVERLAP
                        Fraction of segment time to overlap with previous
                        segment
  -d DEC, --dec DEC     Decimate factor for the fractal dimension estimator
  -g {0,1,2,3}, --gpu {0,1,2,3}
                        GPU to use
  -f OUTDIR, --outdir OUTDIR
                        Output directory
```
Alternatively by importing ***get_fd()*** from get_fd_data.py in another script:
```
from get_fd_data import get_fd

...

get_fdf(strain_channel, whiten, t_start, t_end, segment_time, overlap, estimator, decimate_factor, gpu, out_dir)
```


### Timing (1 hour of data)

| Algorithm     | Total Time (s) | Data Downloading Time (s) | Calculation Time (s) |
| :---          |     :----:     |            :---:          |         :---:        |
| VAR           |     121.2 s    |            110.1 s        |      ***11.1 s***    |
| ANAM          |     549.6 s    |             89.4 s        |     ***460.2 s***    |

| Parameter         | Value                   |
|-------------------|        :---:            |
| Strain            | L1:DCS-CALIB_STRAIN_C01 |
| Sampling Rate     | 16384                   |
| Decimate Factor   | 64                      |
| Total Data Length | 3600 s                  |
| Segment Length    | 1 s                     |
| Segment Overlap   | 0                       |
| GPU               | RTX 2080 Ti             |
