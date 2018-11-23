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

# show errors with figure
#print "\n\tCPU Decoding time: {0:.3f} ms/spike" .format(\
#                              decode_time_cpu/np.sum(n_spikes_cpu) * 1e3)
#print "\n\tGPU Decoding time: {0:.3f} ms/spike" .format(\
#                              decode_time_gpu/np.sum(n_spikes_gpu) * 1e3)
# compute decoding error
errors_cpu = util.getErrors(true_behavior_run,logpos_cpu,grid)
errors_gpu = util.getErrors(true_behavior_run,logpos_gpu,grid)
print "\n\tCPU Decoding median error: {0:.3f} cm".format(np.nanmedian(errors_cpu))
print "\n\tGPU Decoding median error: {0:.3f} cm".format(np.nanmedian(errors_gpu))
print "Position grid size = {} cm".format(config.grid_element_size_cm)

print "------------------------------------------------------------------------"
print "online assessment of replay events:"
# the example to plot
bins_to_plot = range(1227,171+1227)

sigAna = SigAnalyzer(len(grid),n_shanks,config.bin_size_sleep,2.8,config.n_shuffle,config.n_time_bin,config.n_max_spike)
sigAna.uploadParam(decoder_gpu.pix(),decoder_gpu.lx(),np.ascontiguousarray(dist_state['dist_matrix_states']))
mua_all_shanks = np.sum(n_spikes_sleep_bin,1)
mua_all = []
all_shuffle_decode_time=[]
rwd_all = []
prob_all = []
mua = []
p_value_all = []
decode_bin_idx = []

evn_cnt = 0
mua_thd = 4.1+4.0# mean + std, use this number for demostration, can be adjusted
cont_thd = 3
offset = 1
detected = False
skip_cnt = 0
score = []
score.append(0)
#for i in range(bins_to_plot[-1]+1):
for i in range(2000):
    mua.append(np.sum(n_spikes_sleep_bin[i]))
    mua_all.append(n_spikes_sleep_bin[i])

    # online event detection----------------------------
    if mua[i]>=mua_thd:
        if offset==1:
            evn_cnt = evn_cnt + 1
    else:
        evn_cnt = 0
        offset = 1
    if evn_cnt >= cont_thd:
        evn_cnt = 0
        offset = 0
        detected = True
        score[-1] = 0
    #print "detected={},mua[i]={},offset={},evn_cnt={}".format(detected,mua[i],offset,evn_cnt)
    # --------------------------------------------------
    # bins less than 3 spikes are skipped
    if sum(n_spikes_sleep_bin[i])<=2:
        score.append(score[-1])
        skip_cnt = skip_cnt + 1
        if skip_cnt >= 3:
            detected = False
            score[-1] = 0
        if len(rwd_all)==0 or len(prob_all)==0:
            prob_all = np.ones(len(grid))
        else:
            prob_all = np.vstack((prob_all,np.ones(len(grid))))

        p_value_all.append(1)
        all_shuffle_decode_time.append(0)
        continue
    skip_cnt = 0
    decode_bin_idx.append(i);
    
    t1=time.time()
    # decoding all the possible combination for shuffled samples------
    decode_time_gpu = \
        decoder_gpu.decode_new( tetrode_inclusion_mask, config.bin_size_sleep, test_spikes_sleep_bin[i], n_spikes_sleep_bin[i], shuffle=True)
    n_spikes_bin = np.squeeze(n_spikes_sleep_bin[i]).astype('int32')
    
    # ready = True when enough decoded bins are buffered
    prob,ready = sigAna.updateBin(decoder_gpu.cached_pax,n_spikes_bin,decoder_gpu.mu)
    
    # record decoding results 
    if len(rwd_all)==0 or len(prob_all)==0:
        prob_all = prob[0,:]
    else:
        prob_all = np.vstack((prob_all,prob[0,:]))
    # ---------------------------------------------------------------

    # calculate rwd and p-value with 5-10 latest bins---------------- 
    # the minimum p-value is used for assessment
    p_value =1
    rwd = np.zeros(1001)
    if detected:
	for bin_a in [5,6,7,8,9,10]:
	    rwd = sigAna.assess(bin_a)
    	    larger_idx = np.where(rwd>rwd[0])
	    tmp = len(larger_idx[0])/1000.0 + 0.001
	    if tmp<p_value:
	        p_value = tmp
    # --------------------------------------------------------------
    # calculate the cumulative score -------------------------------
    if p_value < 0.05:
        score.append(-np.log(p_value)+score[-1])
    else:
        score.append(score[-1])
    # --------------------------------------------------------------
    t2=time.time()
    #print "detected={},p_value={},score={}".format(detected,p_value,score[-1])

    # record rwds, p-values and scores
    if len(rwd_all)==0:
        rwd_all=rwd
        p_value_all.append( p_value)
    else:
        rwd_all=np.vstack((rwd_all,rwd))
        p_value_all.append(p_value)
    
    # total time used for the decoding and assessment of current bin
    all_shuffle_decode_time.append((t2-t1) * 1e3)
    #print"{} bins of {} done\r".format(i,len(test_spikes_sleep_bin)),
spio.savemat('demo_test.mat',{'errors_gpu':errors_gpu,'decode_bin_idx':decode_bin_idx,'mua_all':mua_all,'all_shuffle_decode_time':all_shuffle_decode_time,'prob':prob_all,'p_value':p_value_all,'score':score})
print "decoding + Assessment time of 2000 bins: mean={0:.3} ms, max={1:.3} ms".format(np.mean(all_shuffle_decode_time),np.max(all_shuffle_decode_time))
print "------------------------------------------------------------------------"
