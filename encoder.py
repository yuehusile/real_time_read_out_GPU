# -*- coding: utf-8 -*-
"""
Created on Wednesday June 27 2018

@author: Sile

This file defines a function that loads the silicon probe dataset and re-organize it for encoding and decoding

"""
from scipy import interpolate
import scipy.io as spio
import collections
import numpy as np
#import fklab.segments as seg
from fklab.segments import Segment as seg
from fklab.decoding import prepare_decoding as prep_dec
from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml
from kde_gpu import MergingCompressionDensity as mcd
from kde_gpu import Decoder, create_covariance_array
from kde_gpu import save_mixture
import pdb

def encode(train, behavior, ephys, config, save=False):
    # prepare encoding
    train_behav = prep_dec.extract_train_behav( train, behavior )
    train_spike, tetrode_inclusion_mask =\
        prep_dec.extract_train_spike( train, ephys, config.min_n_encoding_spikes )
    n_features = len(ephys[u'TT1']['spike_amplitudes'][0])
    n_shanks = len(ephys)
    covars = create_covariance_array( config.behav_bw_cm, config.spf_bw_mV, n_features)
    pdb.set_trace()
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "encoding"
    print "n_features = {}".format(n_features)
    # created encoding points
    encoding_points = [ prep_dec.attach( train_behav, tt ) for tt in train_spike ] 
    print "feature bandwidth = {}".format(config.spf_bw_mV*1000)
    print "compression threshold = {}".format(config.compression_threshold)
    print "number of shanks = {}".format(n_shanks)
    # create compressed (empty) joint density with all tetrodes (even if they have too few spikes)
    mcd_spikebehav = []
    for i in range(n_shanks):
        cov_tt = covars[:n_features+1]
        mcd_spikebehav.append( mcd( ndim=len(cov_tt), sample_covariance =\
            cov_tt, method='bandwidth', threshold=config.compression_threshold,\
            name=(ephys.keys()[i] ) ) )
    # fill joint compressed density with encoding points
    for i, dec in enumerate(mcd_spikebehav):
        if tetrode_inclusion_mask[i]:
            points = encoding_points[i]
            dec.addsamples( points )

    # uncompressed density of behavior data
    mcd_behav = mcd( train_behav[0], sample_covariance=covars[:1], threshold=0,\
        name='behavior' )
    training_time = np.sum( train.duration )
    print "Training time = {} s".format(training_time)
    if save:
        for i in range(len(mcd_spikebehav)):
            save_mixture(mcd_spikebehav[i], 'mixture/mcd_sb{}.mixture'.format(i))
        save_mixture(mcd_behav, 'mixture/mcd_behav.mixture')
        print "mixture saved"
    n_train_spike = 0
    for i in range(len(train_spike)):
        n_train_spike = n_train_spike + len(train_spike[i][0])
    return mcd_spikebehav, mcd_behav, tetrode_inclusion_mask, n_train_spike

