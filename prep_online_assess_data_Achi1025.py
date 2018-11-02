# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:32:41 2016

@author: davide

This script validates and generates an encoding model from an input dataset

"""
from __future__ import division
import numpy as np
from scipy import interpolate
from os.path import dirname, join
import time
import threading

from fklab.segments import Segment as seg
from fklab.decoding import prepare_decoding as prep_dec
from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml

#from gmmcompression.fkmixture import GpuDecoderClass as GpuDecoder
from gmmcompression.fkmixture import SignificanceAnalyzerClass as SigAnalyzer
from kde_gpu import MergingCompressionDensity as mcd
from kde_gpu import save_mixture,load_mixture
from kde_gpu import Decoder, create_covariance_array

import config_file_Achi1025 as config
import pdb
import collections
import scipy.io as spio
import matplotlib.pyplot as plt
import random
import util
import pickle

prep_mixture = True
######################################################################################################
# load data from matlab file
######################################################################################################
config.event_path = 'data/event_for_online_Achi1025.mat'
behavior, ephys_run, ephys_sleep, event = util.loadData(config)

######################################################################################################
# preprocessing: select run data, split train/test, show dataset information, remove no spike bins
######################################################################################################
train, training_time = util.getTrainSet(behavior, ephys_run, config)
#test_run, _, n_spikes_run, true_behavior_run = util.getTestSet(behavior, ephys_run, config)
test_sleep, event_bins, n_spikes_sleep,_ = util.getTestSet(behavior, ephys_sleep, config, event, replay=True, rm_no_spk=False,count_bins_each_event=False)
bin_time = []
for i in range(len(test_sleep)):
    bin_time.append(test_sleep[i].asarray()[0][0]+0.01)
#pdb.set_trace()
spio.savemat('Achi1025_bin_time.mat',{'bin_time':bin_time})

grid = util.getGrid(behavior,config)
######################################################################################################
# encoding
######################################################################################################
config.compression_threshold = 0
if prep_mixture:
    print"encoding"
    mcd_spikebehav0, mcd_behav, tetrode_inclusion_mask, n_train_spike = util.encode(train, behavior, ephys_run, config, save=True)
    n_features = len(ephys_run[u'TT1']['spike_amplitudes'][0])
    n_shanks = len(ephys_run)
    save_mixture(mcd_behav,'mcd_behav.mixture')
    with open('data_online/info_Achi1025.dat',"wb") as f:
        pickle.dump(n_features,f)
        pickle.dump(n_shanks,f)
        pickle.dump(training_time,f)
        pickle.dump(grid,f)
        pickle.dump(tetrode_inclusion_mask,f)
        pickle.dump(n_train_spike,f)
    for i in range(len(mcd_spikebehav0)):
        save_mixture(mcd_spikebehav0[i], 'mcd_sb{}_0.mixture'.format(i))
    mcd_spikebehav, mcd_behav = util.load_mcd_from_file('mixture',n_features,n_shanks,config)
    pdb.set_trace()
######################################################################################################
# decoding
######################################################################################################
# load data information
with open('data_online/info_Achi1025.dat',"rb") as f:
    n_features = pickle.load(f)
    n_shanks = pickle.load(f)
    training_time = pickle.load(f)
    grid = pickle.load(f)
    tetrode_inclusion_mask = pickle.load(f)
    n_train_spike = pickle.load(f)


# prepare decoding spikes
spike_ampl_mask_list = np.array( [ np.ones( ephys_run[key]["spike_amplitudes"].shape[1], dtype='bool')\
    for key in ephys_run.keys() ] )[tetrode_inclusion_mask]

print "Preparing testing spikes..."
#test_spikes_run, n_spikes_run = util.getTestSpikes([ ephys_run[key] for key in ephys_run.keys() ],\
#        test_run, tetrode_inclusion_mask, spike_ampl_mask_list, config.bin_size_run,\
#        sf_keys=["spike_times", "spike_amplitudes"],shuffle=False, binned=False)

#test_spikes_sleep,n_spikes_sleep = util.getTestSpikes([ ephys_sleep[key] for key in ephys_sleep.keys() ],\
#        test_sleep, tetrode_inclusion_mask, spike_ampl_mask_list, config.bin_size_sleep,\
#        sf_keys=["spike_times", "spike_amplitudes"],shuffle=False, binned=False)

test_spikes_sleep_bin,n_spikes_sleep_bin = util.getTestSpikes([ ephys_sleep[key] for key in ephys_sleep.keys() ],\
        test_sleep, tetrode_inclusion_mask, spike_ampl_mask_list, config.bin_size_sleep,\
        sf_keys=["spike_times", "spike_amplitudes"],shuffle=True, binned=True)

#test_spikes_sleep_bin = test_spikes_sleep_bin[:500]

#with open('data/test_spikes_run.dat', "wb") as f:
#    pickle.dump(test_spikes_run,f)
#    pickle.dump(n_spikes_run,f)
#    pickle.dump(true_behavior_run,f)
#with open('data/test_spikes_sleep.dat', "wb") as f:
#    pickle.dump(test_spikes_sleep,f)
#    pickle.dump(n_spikes_sleep,f)
with open('test_spikes_sleep_bin_online_Achi1025.dat', "wb") as f:
    pickle.dump(test_spikes_sleep_bin,f)
    pickle.dump(n_spikes_sleep_bin,f)

#with open('data/test_spikes_run.dat', "rb") as f:
#    test_spikes_run = pickle.load(f)
#    n_spikes_run = pickle.load(f)
#with open('data/test_spikes_sleep.dat', "rb") as f:
#    test_spikes_sleep = pickle.load(f)
#    n_spikes_sleep = pickle.load(f)
#with open('data/test_spikes_sleep_bin.dat', "rb") as f:
#    test_spikes_sleep_bin = pickle.load(f)
#    n_spikes_sleep_bin = pickle.load(f)

pdb.set_trace()
# initialize decoder
decoder_cpu = Decoder( mcd_behav, mcd_spikebehav, training_time,\
        grid, config.offset )
decoder_gpu = Decoder( mcd_behav, mcd_spikebehav, training_time,\
        grid, config.offset, use_gpu = True, gpu_batch_size=1024)
# test decoding time
print "Test CPU vs GPU decoding time (RUN):"
posterior_cpu, logpos_cpu, n_spikes_cpu, n_tt, decode_time_cpu = \
        decoder_cpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)
posterior_gpu, logpos_gpu, n_spikes_gpu, n_tt, decode_time_gpu = \
        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)

print "\n\tCPU Decoding time: {0:.3f} ms/spike" .format(\
                              decode_time_cpu/np.sum(n_spikes_cpu) * 1e3)
print "\n\tGPU Decoding time: {0:.3f} ms/spike" .format(\
                              decode_time_gpu/np.sum(n_spikes_gpu) * 1e3)
print "------------------------------------------------------------------------"
print "Test CPU vs GPU decoding time (REPLAY):"
posterior_cpu, logpos_cpu, n_spikes_cpu, n_tt, decode_time_cpu = \
        decoder_cpu.decode_new( tetrode_inclusion_mask, config.bin_size_sleep, test_spikes_sleep, n_spikes_sleep, shuffle=False)
posterior_gpu, logpos_gpu, n_spikes_gpu, n_tt, decode_time_gpu = \
        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_sleep, test_spikes_sleep, n_spikes_sleep, shuffle=False)

print "\n\tCPU Decoding time: {0:.3f} ms/spike" .format(\
                              decode_time_cpu/np.sum(n_spikes_cpu) * 1e3)
print "\n\tGPU Decoding time: {0:.3f} ms/spike" .format(\
                              decode_time_gpu/np.sum(n_spikes_gpu) * 1e3)
print "------------------------------------------------------------------------"

print "Test shuffle decoding time (GPU):"

sigAna = SigAnalyzer(len(grid),len(ephys_sleep.keys()),config.bin_size_sleep,config.n_shuffle,config.n_time_bin,config.n_max_spike)
sigAna.uploadParam(decoder_gpu.pix(),decoder_gpu.lx())
all_shuffle_decode_time = []
for i in range(len(test_spikes_sleep_bin)):
    t1=time.time()
    decode_time_gpu = \
        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_sleep, test_spikes_sleep_bin[i], n_spikes_sleep_bin[i], shuffle=True)
    n_spikes_bin = np.squeeze(n_spikes_sleep_bin[i]).astype('int32')
    #print "n_spikes_bin={}".format(n_spikes_bin)
    prob,ready = sigAna.updateBin(decoder_gpu.cached_pax,n_spikes_bin,decoder_gpu.mu)
    t2=time.time()
    all_shuffle_decode_time.append((t2-t1) * 1e3)
    #print "bin {} decoded, time={} ms".format(i,(t2-t1) * 1e3)

max_decode_time = np.max(all_shuffle_decode_time)
mean_decode_time = np.mean(all_shuffle_decode_time)
print "Shuffling decode time for {} bin, max={} ms, mean={} ms".format(len(test_spikes_sleep_bin),max_decode_time,mean_decode_time)
