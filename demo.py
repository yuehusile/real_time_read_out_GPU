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

import config_file as config
import pdb
import collections
import scipy.io as spio
import util
import pickle

# load data information
with open('data/info.dat',"rb") as f:
    n_features = pickle.load(f)
    n_shanks = pickle.load(f)
    training_time = pickle.load(f)
    grid = pickle.load(f)
    tetrode_inclusion_mask = pickle.load(f)
    n_train_spike = pickle.load(f)

print "Data information:"
print "number of shanks: {}".format(n_shanks)
print "number of features: {}".format(n_features)
print "number of position grids: {}".format(len(grid))
print "number of training spikes: {}".format(n_train_spike)
print "compression threshold = {}".format(config.compression_threshold)

print "------------------------------------------------------------------------"
# load mixtures (encoded model)
mcd_spikebehav, mcd_behav = util.load_mcd_from_file('mixture',n_features,n_shanks,config)
######################################################################################################
# decoding
######################################################################################################

with open('data/test_spikes_run.dat', "rb") as f:
    test_spikes_run = pickle.load(f)
    n_spikes_run = pickle.load(f)
    true_behavior_run = pickle.load(f)
with open('data/test_spikes_sleep_bin.dat', "rb") as f:
    test_spikes_sleep_bin = pickle.load(f)
    n_spikes_sleep_bin = pickle.load(f)
print "testing spikes loaded"

print "------------------------------------------------------------------------"
# initialize decoder
decoder_cpu = Decoder( mcd_behav, mcd_spikebehav, training_time,\
        grid, config.offset )
decoder_gpu = Decoder( mcd_behav, mcd_spikebehav, training_time,\
        grid, config.offset, use_gpu = True, gpu_batch_size=1024)

# test decoding time
print "Test CPU vs GPU decoding time (RUN):"
print "number of testing spikes: {}".format(np.sum(n_spikes_run))
posterior_cpu, logpos_cpu, n_spikes_cpu, n_tt, decode_time_cpu = \
        decoder_cpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)
posterior_gpu, logpos_gpu, n_spikes_gpu, n_tt, decode_time_gpu = \
        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)

print "\n\tCPU Decoding time: {0:.3f} ms/spike" .format(\
                              decode_time_cpu/np.sum(n_spikes_cpu) * 1e3)
print "\n\tGPU Decoding time: {0:.3f} ms/spike" .format(\
                              decode_time_gpu/np.sum(n_spikes_gpu) * 1e3)
# compute decoding error
errors_cpu = util.getErrors(true_behavior_run,logpos_cpu,grid)
errors_gpu = util.getErrors(true_behavior_run,logpos_gpu,grid)
print "\n\tCPU Decoding median error: {0:.3f} cm".format(np.nanmedian(errors_cpu))
print "\n\tGPU Decoding median error: {0:.3f} cm".format(np.nanmedian(errors_gpu))
print "Position grid size = {} cm".format(config.grid_element_size_cm)
print "------------------------------------------------------------------------"

print "Test GPU shuffle decoding time (REPLAY):"

sigAna = SigAnalyzer(len(grid),n_shanks,config.bin_size_sleep,config.n_shuffle,config.n_time_bin,config.n_max_spike)
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
print "Shuffling decode for {} bin:".format(len(test_spikes_sleep_bin))
print "\n\ttime per bin: max={} ms, mean={} ms".format(max_decode_time,mean_decode_time)
