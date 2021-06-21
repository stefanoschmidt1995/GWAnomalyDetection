import numpy as np
from pipeline import *
from pipeline_helper import *
import os

import mlgw.GW_generator as gen

N_injs = 30 #number of injections
N_batch = 10

injs = create_injection_list(N_injs, 4096, 1126257415, 1126257415+100, datafile = 'data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz')

split_data('data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz', outfolder = 'data_temp',
			 srate = 4096, start_GPS = 1126257415, T_batch = 100/N_batch, T_overlap = 0., 
			 injection_list = injs,
			 downsampling_factor = 4, #for some reason, the non downsampled data are not suitable for emd...
			 prefix = 'H',
			 do_EMD = False
			 )
