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

