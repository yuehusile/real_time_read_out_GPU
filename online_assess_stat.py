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
import matplotlib.pyplot as plt
import random
import util
import pickle

# load transition matrix
#trans=spio.loadmat('data_online/trans_silicon.mat')
#dist_state=spio.loadmat('data_online/dist_state_matrix_Achi1101.mat')
dist_state=spio.loadmat('data_online/Achi1025_dist_state_matrix.mat')
event_idx=spio.loadmat('data_online/Achi1025_event_idx.mat')
pdb.set_trace()
# load data information
with open('data_online/info_Achi1025.dat',"rb") as f:
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
mcd_spikebehav, mcd_behav = util.load_mcd_from_file('mixture0',n_features,n_shanks,config)
######################################################################################################
# decoding
######################################################################################################

#with open('data/test_spikes_run.dat', "rb") as f:
#    test_spikes_run = pickle.load(f)
#    n_spikes_run = pickle.load(f)
#    true_behavior_run = pickle.load(f)
with open('data_online/test_spikes_sleep_bin_online_Achi1025.dat', "rb") as f:
    test_spikes_sleep_bin = pickle.load(f)
    n_spikes_sleep_bin = pickle.load(f)
print "testing spikes loaded"

#pdb.set_trace()
print "------------------------------------------------------------------------"
# initialize decoder
#decoder_cpu = Decoder( mcd_behav, mcd_spikebehav, training_time,\
#        grid, config.offset )
decoder_gpu = Decoder( mcd_behav, mcd_spikebehav, training_time,\
        grid, config.offset, use_gpu = True, gpu_batch_size=1024)

# test decoding time
#print "Test CPU vs GPU decoding time (RUN):"
#print "number of testing spikes: {}".format(np.sum(n_spikes_run))
#posterior_cpu, logpos_cpu, n_spikes_cpu, n_tt, decode_time_cpu = \
#        decoder_cpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)
#posterior_gpu, logpos_gpu, n_spikes_gpu, n_tt, decode_time_gpu = \
#        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_run, test_spikes_run, n_spikes_run, shuffle=False)

#print "\n\tCPU Decoding time: {0:.3f} ms/spike" .format(\
#                              decode_time_cpu/np.sum(n_spikes_cpu) * 1e3)
#print "\n\tGPU Decoding time: {0:.3f} ms/spike" .format(\
#                              decode_time_gpu/np.sum(n_spikes_gpu) * 1e3)
# compute decoding error
#errors_cpu = util.getErrors(true_behavior_run,logpos_cpu,grid)
#errors_gpu = util.getErrors(true_behavior_run,logpos_gpu,grid)
#print "\n\tCPU Decoding median error: {0:.3f} cm".format(np.nanmedian(errors_cpu))
#print "\n\tGPU Decoding median error: {0:.3f} cm".format(np.nanmedian(errors_gpu))
#print "Position grid size = {} cm".format(config.grid_element_size_cm)
#print "------------------------------------------------------------------------"

print "Test GPU shuffle decoding time (REPLAY):"
#sigAna = SigAnalyzer(len(grid),n_shanks,config.bin_size_sleep,dist_state['dmax'],config.n_shuffle,config.n_time_bin,config.n_max_spike)
sigAna = SigAnalyzer(len(grid),n_shanks,config.bin_size_sleep,2.8,config.n_shuffle,config.n_time_bin,config.n_max_spike)
sigAna.uploadParam(decoder_gpu.pix(),decoder_gpu.lx(),np.ascontiguousarray(dist_state['dist_states_matrix']))
#sigAna.uploadParam(decoder_gpu.pix(),decoder_gpu.lx(),np.ascontiguousarray(dist_state['dist_matrix_states']))
all_shuffle_decode_time = []
# check multi unit criterion
assessed_idx = []
spkCntThd=10
contThd=3
cnt = 0
n_assess = 10
assess_cnt = n_assess;
rwd_all = [];
p_value_all = [];
prob_all = [];
mua = [];
mua_assessed = [];
mua_all = [];
decode_bin_idx = 0;
assessed_db_idx = [];
decoded_bin_time = [];
event37_prob = [];

# get mua statistics
#pdb.set_trace()
mua_all_shanks = np.sum(n_spikes_sleep_bin,1)
#for i in range(10000):
#for i in range(len(test_spikes_sleep_bin)):

# buffer some initial spikes
for i in range(event_idx['event_idx'][0,0]-1):
    if sum(n_spikes_sleep_bin[i])<=2:
        all_shuffle_decode_time.append(-1)
        continue
    
    decode_time_gpu = \
        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_sleep, test_spikes_sleep_bin[i], n_spikes_sleep_bin[i], shuffle=True)
    n_spikes_bin = np.squeeze(n_spikes_sleep_bin[i]).astype('int32')
    #print "n_spikes_bin={}".format(n_spikes_bin)
    prob,ready = sigAna.updateBin(decoder_gpu.cached_pax,n_spikes_bin,decoder_gpu.mu)
   
event_p_values = []
for ii in range(event_idx['event_idx'].shape[0]):
    for i in range(event_idx['event_idx'][ii,0]-1,event_idx['event_idx'][ii,1]):
       
        if sum(n_spikes_sleep_bin[i])<=2:
            event_p_values.append(-1)
            continue
        
        t1=time.time()
        decode_time_gpu = \
            decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_sleep, test_spikes_sleep_bin[i], n_spikes_sleep_bin[i], shuffle=True)
        n_spikes_bin = np.squeeze(n_spikes_sleep_bin[i]).astype('int32')
        prob,ready = sigAna.updateBin(decoder_gpu.cached_pax,n_spikes_bin,decoder_gpu.mu)
       
          # assess all the time
        #do assessment
        p_value =1
        for bin_a in [5,6,7,8,9,10]:
            rwd = sigAna.assess(bin_a)
            larger_idx = np.where(rwd>rwd[0])
            tmp = len(larger_idx[0])/1000.0
            if tmp<p_value:
                p_value = tmp
        event_p_values.append(p_value)
        print"event[{}] bin[{}] p={}".format(ii,i,p_value)

spio.savemat("Achi1025_event_p_values.mat",{"event_p_values":event_p_values,"mua_all_shanks":mua_all_shanks})
