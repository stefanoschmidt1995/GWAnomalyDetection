# Relevant links

### Omicron

We can find the list of Omicron channels [here](https://git.ligo.org/detchar/ligo-channel-lists/-/tree/master/O3). Being `IFO` either H1 or L1, `IFO-O3-lldetchar.ini` contains all the channels by alphabetical order and `IFO-O3-deep.ini`	contains the channels grouped by detector module. Below we can find some modules and their most relevant channel(s).

[Input Mode Cleaner]\
Most relevant channels:

	IFO:IMC-F_OUT_DQ 16384 safe clean
	IFO:IMC-I_OUT_DQ 16384 safe clean
	IFO:IMC-L_OUT_DQ 2048 safe clean

[Output Mode Cleaner]\
Most relevant channels:

	IFO:OMC-DCPD_SUM_OUT_DQ 16384 unsafe clean
	IFO:OMC-LSC_I_OUT_DQ 16384 unsafe clean

[Length sensing and control]\
Most relevant channels:

	IFO:LSC-PRCL_IN1_DQ 16384 safe clean
	IFO:LSC-PRCL_OUT_DQ 16384 safe clean
    
### O3 data quality flags

In [this](https://wiki.ligo.org/DetChar/DataQuality/O3Flags) document we can find data quality flags and science segment definitions. From here we can get the witness channels of sever glitches:

- Sever whistle from L1: [here](https://ldas-jobs.ligo-la.caltech.edu/~detchar/hveto/day/20200106/1262304018-1262390418/) we can find the witness channel as ranked by Hveto.
-Sever scattering from L1: #FIXME


# Robin's data


`/home/robin.vanderlaag/wp5/strain_fractals/condor_data/labeled_anomalies.npy`: channels and times that whitening failed.

`/home/robin.vanderlaag/wp5/strain_fractals/condor_data/check_const_channels.py`: script to check failing whitening.

`/home/robin.vanderlaag/wp5/strain_fractals/condor_data/res_dir/FD_all_n=3584_2022-11-17.hdf5`: fractal data

`/scratch/robin.vanderlaag/temp.copy.of.O3.raw.frames/O3`: raw frame in disk. How do we delete?

