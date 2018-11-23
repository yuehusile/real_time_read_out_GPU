# -*- coding: utf-8 -*-
"""
Created on Nov 2018

@author: Sile Hu

This script uses pre-trained model to decode neural emsemble activities. 

"""
from __future__ import division
import numpy as np
from scipy import interpolate
from os.path import dirname, join
import time
import threading

from fklab.segments import Segment as seg
from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml

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
import matplotlib.pyplot as plt
# load data information
print "------------------------------------------------------------------------"
print " Loading data information..."
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
#print "number of position grids: {}".format(len(grid))
print "number of training spikes: {}".format(n_train_spike)
print "compression threshold = {}".format(config.compression_threshold)

print "------------------------------------------------------------------------"
# load mixtures (encoded model)
print " Loading pre-trained model and testing data..."
mcd_spikebehav, mcd_behav = util.load_mcd_from_file('data/mixture{}'.format(config.compression_threshold),n_features,n_shanks,config)
# load dist state matrix for online assessment
dist_state=spio.loadmat('data/Achi1101_dist_state_matrix_new.mat')
######################################################################################################
# decoding
######################################################################################################

with open('data/test_spikes_run.dat', "rb") as f:
    test_spikes_run = pickle.load(f)
    n_spikes_run = pickle.load(f)
    true_behavior_run = pickle.load(f)
with open('data/test_spikes_sleep_bin_online2000.dat', "rb") as f:
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
print "Position grid size = {} cm ({} position bins)".format(config.grid_element_size_cm, len(grid))
#print "number of testing spikes: {}".format(np.sum(n_spikes_run))
print "decoding {} spikes with CPU...".format(np.sum(n_spikes_run))
posterior_cpu, logpos_cpu, n_spikes_cpu, n_tt, decode_time_cpu = \
        decoder_cpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)
print "CPU decoding done, cost {0:.3f} ms ({1:.3f} ms/spike)".format(decode_time_cpu*1e3, decode_time_cpu/np.sum(n_spikes_cpu) * 1e3)

print "------------------------------------------------------------------------"
print "decoding {} spikes with GPU...".format(np.sum(n_spikes_run))
posterior_gpu, logpos_gpu, n_spikes_gpu, n_tt, decode_time_gpu = \
        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)
print "GPU decoding done, cost {0:.3f} ms ({1:.3f} ms/spike)".format(decode_time_gpu*1e3, decode_time_gpu/np.sum(n_spikes_gpu) * 1e3)

print "Speed up by {0:.3f}X".format(decode_time_cpu/decode_time_gpu)

# compute decoding error and plot cdf
errors_gpu = util.getErrors(true_behavior_run,logpos_gpu,grid)
num_bins = 100
counts, bin_edges = np.histogram(errors_gpu,bins=num_bins)
cdf = np.cumsum(counts)
plt.plot(bin_edges[1:],cdf/cdf[-1])
plt.title('Median error = {0:.3f} cm'.format(np.nanmedian(errors_gpu)))
plt.ylabel('CDF')
plt.xlabel('Decoding error (cm)')
plt.show()
print "\n\tGPU Decoding median error: {0:.3f} cm".format(np.nanmedian(errors_gpu))

