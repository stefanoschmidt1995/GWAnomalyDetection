import numpy as np
from pipeline import *
from pipeline_helper import *
import os

import mlgw.GW_generator as gen

N_injs = 5 #number of injections
N_batch = 1

injs = create_injection_list(N_injs, 4096, 1126257415, 1126257415+100,
	theta_range = [[10,100],[10,100],[-0.8,0.8],[-0.8,0.8],[40,400],[0.,np.pi],[0.,2*np.pi]],
	datafile = 'data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz')

time_list = [inj['time'] for inj in injs]
np.savetxt('data_temp/injs_time_6.dat', time_list)

split_data('data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz', outfolder = 'data_temp',
			 srate = 4096, start_GPS = 6000, T_batch = 100/N_batch, T_overlap = 0., 
			 injection_list = injs,
			 downsampling_factor = 2, #for some reason, the non downsampled data are not suitable for emd...
			 prefix = 'H',
			 do_EMD = False
			 )
