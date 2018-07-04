# dataset path
run_path = 'data/Achi110119400_22000firstpca_all.mat'
sleep_path = 'data/Achi110124000_36700firstpca_all_sleep.mat'
event_path = 'data/postNREMevent.mat'

n_features = 10

# data selection options
run_speed = 0.15
min_run_duration = 0.25 # in sec
min_n_encoding_spikes = 10

# encoding - decoding options
bin_size_run = 0.25 # in seconds
bin_size_sleep = 0.02 # in seconds
grid_element_size_cm = 5 # will be used to generate a grid in case of no
behav_bw_cm = 8.5
spf_bw_mV = 0.14*0.001
compression_threshold = 1
offset = 1e-10 # Hz

# train set
# choose percentage of run data(bins) to use (later part)
r_data_not_to_use = 0.5
# set percentage of train/test bins
r_train = 0.7

# shuffle
n_time_bin = 40
n_max_spike = 100
n_shuffle = 1000

gpu_n_tt_use = 15
cpu_n_tt_use = 15
max_n_spikes_gpu_kde = 2**13

#fully_parallel = True

# for silicon:

max_error = 284/2
test_sleep = False
