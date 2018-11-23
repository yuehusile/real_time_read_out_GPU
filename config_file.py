# decoding and assessment options
# author: Sile Hu

compression_threshold = 1 # 0 or 1 in this demo
# encoding - decoding options
bin_size_run = 0.25 # in seconds
bin_size_sleep = 0.02 # in seconds
grid_element_size_cm = 5 # size of each spatial bin
behav_bw_cm = 8.5
spf_bw_mV = 0.15*0.001 # scaled to be consistant with the PCA data, the unit is not mV in this case
offset = 1e-10 # Hz
# shuffle
n_time_bin = 40
n_max_spike = 100
n_shuffle = 1001
max_n_spikes_gpu_kde = 2**13
# for silicon dataset:
max_error = 284/2
