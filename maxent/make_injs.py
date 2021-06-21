import numpy as np
from pipeline import *
from pipeline_helper import *
import os

import mlgw.GW_generator as gen

	#creating 30 injs
injs = create_injection_list(30, 4096, 1126257415, 1126257415+100, datafile = 'data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz')

split_data('data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz', outfolder = 'data_temp',
			 srate = 4096, start_GPS = 1126257415, T_batch = 30, T_overlap = 10, 
			 injection_list = injs,
			 downsampling_factor = 4, #for some reason, the non downsampled data are not suitable for emd...
			 prefix = 'H',
			 do_EMD = False
			 )
