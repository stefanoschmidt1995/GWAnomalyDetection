import os
import pathlib
import numpy as np
import pandas as pd

RNG_STATE = 20220922
np.random.seed(RNG_STATE)

n_samples = 8 #"max" # per category
width = 8 # total width of segments
segs = [str(s) for s in [0.25, 0.5, 1.0, 2.0]]
overlap = 0
maxretries = 4
nproc = 5
maxqueuesize = nproc
timespergpu = 10

clean_segments = np.load('/home/robin.vanderlaag/wp5/strain_fractals/condor_data/clean_segments_O3A.npy')
gs_triggers = pd.read_csv('/home/robin.vanderlaag/wp5/strain_fractals/condor_data/gspyO3A.csv')
df_w = gs_triggers[gs_triggers['label']=='Whistle']
df_t = gs_triggers[gs_triggers['label']=='Tomte']
df_s = gs_triggers[gs_triggers['label']=='Scattered_Light']

# remove segments shorter than 2 minute as they will be in regions surrounded by glitches
msk = np.where((clean_segments[:,1]-clean_segments[:,0] >= 2*60))[0]
clean_segments = clean_segments[msk]

if not isinstance(n_samples, int) and n_samples.lower()=='max':
    # maximum possible samples with a balanced dataset
    n_samples = min([clean_segments.shape[0], len(df_w), len(df_t), len(df_s)])


clean_ids = np.random.choice(np.arange(clean_segments.shape[0]), n_samples, replace=False)
clean_times = (clean_segments[clean_ids,0]+clean_segments[clean_ids,1])/2 # take the middle of each segment

whistles = df_w.sample(n=n_samples, random_state=RNG_STATE)
tomte = df_t.sample(n=n_samples, random_state=RNG_STATE+1)
scattered = df_s.sample(n=n_samples, random_state=RNG_STATE+2)

gs_idx = np.concatenate([np.ones(len(clean_times), dtype=int)*-1,  
                         whistles.index.values, 
                         tomte.index.values, 
                         scattered.index.values])
times = np.concatenate([clean_times, 
                        whistles['GPStime'].values, 
                        tomte['GPStime'].values, 
                        scattered['GPStime'].values])
labels = np.concatenate([np.zeros(len(clean_times)), 
                         np.ones(n_samples), 
                         np.ones(n_samples)*2, 
                         np.ones(n_samples)*3])

np.savez(f'Run_n={n_samples}_ids.npz', gs_idx=gs_idx, times=times,labels=labels)


n_jobs = 2
split_times = np.array_split(times, n_jobs)
for i in range(len(split_times)):
    print(f'Job {i}: {len(split_times[i])} times')

py_script = 'condor_fd.py'

# generate your own proxy | see: https://computing.docs.ligo.org/guide/condor/data/
proxy_path = '/home/robin.vanderlaag/wp5/strain_fractals/condor_data/proxy/x509up_u46130' 
log_dir = 'logs'
res_dir = 'res_dir'
times_dir = 'times'
channels_file = 'L1_use_channels.csv'
request_memory = 3000 # in MB | test on single system first
request_disk = 200 + (timespergpu+maxqueuesize+nproc)*290 + len(split_times)*0.5 # in MB | each temporary file on disk is ~280 MB and each resulst file is ~ 0.5 MB
accounting_group = 'ligo.dev.o4.burst.explore.test'

sh_name = 'fd_sh'
sub_name = 'fd_sub'
dag_name = 'fd_dag'

pathlib.Path(log_dir).mkdir(exist_ok=True)
pathlib.Path(res_dir).mkdir(exist_ok=True)
pathlib.Path(times_dir).mkdir(exist_ok=True)



sh_lines = ['#!/bin/bash', #"echo 'export PATH=/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/bin/:$PATH'",
            'export PATH=/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/bin/:$PATH',
            'source activate igwn-py38',
            f'python3 {py_script} -t ${{times}} -w ${{width}} -s ${{segs}} -o ${{overlap}} -p ${{path}} -q ${{maxqueuesize}} -r ${{maxretries}} -n ${{nproc}} -i ${{timespergpu}} --var']
with open(f'{sh_name}.sh', 'w') as f:
    f.write('\n'.join(sh_lines))
os.system(f'chmod +x {sh_name}.sh')

sub_lines = ['universe = vanilla',
             'getenv = true',
             'environment = ID=$(ID);times=$(times);path=$(path);width=$(width);segs=$(segs);overlap=$(overlap);maxqueuesize=$(maxqueuesize);maxretries=$(maxretries);nproc=$(nproc);timespergpu=$(timespergpu)',
             f'executable = {sh_name}.sh',
             f'request_memory = {request_memory} MB',
             f'request_disk = {request_disk} MB',
             f'log = {log_dir}/jobs.log',
             f'output = {log_dir}/J$(ID).out',
             f'error = {log_dir}/J$(ID).err',
             'use_x509userproxy = true',
             f'x509userproxy = {proxy_path}',
             'requirements = HAS_LIGO_FRAMES =?= True', # +DESIRED_Sites = "LIGO-CIT" # For testing only
             'request_gpus = 1',
             'should_transfer_files = YES',
             f'transfer_input_files = {times_dir}/$(times), {py_script}, {channels_file}, {res_dir}', 
             f'transfer_output_files = {res_dir}, {times_dir}' ,
             'when_to_transfer_output = ON_EXIT_OR_EVICT',
             f'accounting_group = {accounting_group}',
             'queue 1']
with open(f'{sub_name}.sub', 'w') as f:
    f.write('\n'.join(sub_lines))

dag_lines = []
for i in range(1,n_jobs+1):
    print(i)
    np.save(f'{times_dir}/J{i}_times.npy', split_times[i-1])
    dag_lines.append(f'JOB A{i} {sub_name}.sub')
    dag_lines.append(f'VARS A{i} ID="{i}" times="J{i}_times.npy" path="{res_dir}" width="{width}" segs="{" ".join(segs)}" overlap="{overlap}" maxqueuesize="{maxqueuesize}" maxretries="{maxretries}" nproc="{nproc}" timespergpu="{timespergpu}"')
    #dag_lines.append(f'RETRY A{i} 3000') # for final run
with open(f'{dag_name}.dag', 'w') as f:
    f.write('\n'.join(dag_lines))