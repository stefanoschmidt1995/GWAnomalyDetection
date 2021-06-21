import numpy as np
from pipeline import *
from pipeline_helper import *
import os

import mlgw.GW_generator as gen

###########################

#TODO: the weak part is a slow emd decomposition: data should be divided in batches (~1000s) and analysed separately. This should be pretty natural...

if False:
	injs = create_injection_list(30, 4096, 1126257415, 1126257415+100, datafile = 'data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz')
	save_inj_list('data/injs.pkl', injs)
else:
	injs = load_inj_list('data/injs.pkl')
	gather_LLs('data/data_LL/', injs, LL_treshold = -11)
	plt.show()

print("Made injections")

if True:
	split_data('data/H-H1_GWOSC_4KHZ_R1-1126257415-100.txt.gz', outfolder = 'data/data_emd',
			 srate = 4096, start_GPS = 1126257415, T_batch = 30, T_overlap = 10, 
			 injection_list = injs,
			 downsampling_factor = 4, #for some reason, the non downsampled data are not suitable for emd...
			 prefix = 'H')

srate = 4096/4

emd_folder = 'data/data_emd/'
emd_files = os.listdir(emd_folder)

for f in emd_files:
	imf = load_emd(emd_folder+f)
	GPS_time = float(f[f.find('Hz')+3:f[f.find('Hz')+3:].find('-')+f.find('Hz')+3])
	outfile = 'data/data_LL/LL-{}-{}.pkl'.format(GPS_time,len(imf[:,0])/srate)
	AnomalyDetection_pipeline(imf[:,0], srate, T_train = 15., N_step = 250,
		outfile = outfile,
		plot = False,
		GPS_time = GPS_time)

	#detect_outliers(outfile, injs)

	#plt.show()

gather_LLs('data/data_LL/', injs, LL_treshold = -.1)
plt.show()


