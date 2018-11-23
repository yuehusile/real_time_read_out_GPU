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
#from fklab.decoding import prepare_decoding as prep_dec
from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml
from kde_gpu import MergingCompressionDensity as mcd
from kde_gpu import Decoder, create_covariance_array
from kde_gpu import save_mixture
import pdb
######################################################################################################
# load data from matlab file
######################################################################################################
def loadData(config): 
    """
    load run and sleep epoch data based on input paths

    Args:
        config - configuration object
    returns:
        behavior - behavior data during run epoch
        ephys - spike data of run epoch
        ephys2 - spike data of sleep epoch
        event - replay events
    """
    # read matlab data set
    run_mat = spio.loadmat(config.run_path)
    # read replay events and data
    event = spio.loadmat(config.event_path)
    sleep_mat = spio.loadmat(config.sleep_path)
    # tmp data structures
    eph = run_mat['ephys']
    eph1 = sleep_mat['ephys']
    behav = run_mat['behavior']
    behavior = dict()
    # *100 is for transfering units from m to cm
    behavior['linear_position'] = behav[0][0][0].flatten()*100
    behavior['speed'] = behav[0][0][1].flatten()
    behavior['time'] = behav[0][0][2].flatten()

    # read run epoch data
    ephys = collections.OrderedDict()
    for i in range(len(eph[0])):
        v_shank_idx = i+1
        ephys[u'TT'+str(v_shank_idx)] = dict()        
        # reorder the data based on timestamps
        x = eph[0][i][0].flatten()
        sort_indices = np.argsort(x)
        x = x[sort_indices]
        ephys[u'TT'+str(v_shank_idx)]['spike_times'] = x
        ephys[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = eph[0][i][1][:,range(config.n_features)][sort_indices,:]/1000.0
        ephys[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = ephys[u'TT'+str(v_shank_idx)]['spike_amplitudes'].astype(np.float32)
    check_ephys_dataset( ephys )
    # read sleep epoch data
    ephys2 = collections.OrderedDict()
    for i in range(len(eph1[0])):
        v_shank_idx = i+1
        ephys2[u'TT'+str(v_shank_idx)] = dict()        
        # reorder the data based on timestamps
        x = eph1[0][i][0].flatten()
        sort_indices = np.argsort(x)
        x = x[sort_indices]
        ephys2[u'TT'+str(v_shank_idx)]['spike_times'] = x
        ephys2[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = eph1[0][i][1][:,range(config.n_features)][sort_indices,:]/1000.0
        ephys2[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = ephys2[u'TT'+str(v_shank_idx)]['spike_amplitudes'].astype(np.float32)
    check_ephys_dataset( ephys2 )
    return behavior, ephys, ephys2, event

def getTrainSet(behavior, ephys, config):
    """
    get train set from run epoch based on configuration

    Args:
        config - configuration object
        behavior - behavior data during run epoch
        ephys - spike data of run epoch
    returns:
        train - train segments
        training_time - total time of the training dataset
    """
    print "###################################################################################"
    print "Train set:"
    # select run segments
    print "run speed={}, min run duration={}, binsize={}".format(config.run_speed,config.min_run_duration,config.bin_size_run)
    run = seg.fromlogical( behavior["speed"] > config.run_speed, x=behavior["time"] )
    run = run[ run.duration > config.min_run_duration ]

    # split segments into independent bins
    bin_size_test = config.bin_size_run
    run_binned = run.split( size=bin_size_test )
    r_data_not_to_use = config.r_data_not_to_use
    first_bin_idx = int(r_data_not_to_use*len(run_binned))
    used_data = run_binned[first_bin_idx:]
    r_train = config.r_train
    n_train = int(r_train*len(used_data))
    train = used_data[0:n_train]

    total_time = ephys[u'TT1']['spike_times'][-1]-ephys[u'TT1']['spike_times'][0]    
    print "total recording time = {} min, start time = {} s, end time = {} s".format(total_time/60,ephys[u'TT1']['spike_times'][0],ephys[u'TT1']['spike_times'][-1])
    training_time = np.sum( train.duration )
    print "training_time={} s".format(training_time)

    # find number of spikes within train datasets
    n_spikes = []
    sum_tt = []
    mean_tt = []
    no_spike_bin_idx = []
    spike_bin_idx = []
    n_spikes_all = np.zeros(len(train))
    for i, key in enumerate( ephys ):
        tt = ephys[key]
        n_spikes.append( train.contains(tt['spike_times'])[1] )
        n_spikes_all = n_spikes_all + n_spikes[i]
        sum_tt.append(sum(n_spikes[i]))
    # remove bins without spike 
    for j,n in enumerate(n_spikes_all):
        if n==0:
            no_spike_bin_idx.append(j)
        else:
            spike_bin_idx.append(j)
    if len(no_spike_bin_idx)>0:
        train = train[spike_bin_idx]
    
    print "number spikes for training:"
    print np.sum(sum_tt)
    print ""
    return train, training_time

def getTestSet(behavior, ephys, config, event = [], replay=False, rm_no_spk=True, count_bins_each_event=True):
    """
    get train set from run epoch based on configuration

    Args:
        config - configuration object
        behavior - behavior data during run epoch
        ephys - spike data of run/sleep epoch
        event - replay events
        replay - whether test replay data or run data
                 run : False
                 replay: True
    returns:
        test_binned - test segments
        event_bins - number of spikes in each event
        n_spikes_all - number of spikes in each bin 
    """
    if replay:
        test = seg(event['postNREMevent'])
        bin_size_test = config.bin_size_sleep
    else:
        run = seg.fromlogical( behavior["speed"] > config.run_speed, x=behavior["time"] )
        run = run[ run.duration > config.min_run_duration ]
        # split segments into independent bins
        bin_size_test = config.bin_size_run
        run_binned = run.split( size=bin_size_test )
        r_data_not_to_use = config.r_data_not_to_use
        first_bin_idx = int(r_data_not_to_use*len(run_binned))
        used_data = run_binned[first_bin_idx:]
        r_train = config.r_train
        n_train = int(r_train*len(used_data))
        test = used_data[n_train:]
        bin_size_test = config.bin_size_run

    print "###################################################################################"
    print "Test set:"
    testing_time = np.sum( test.duration )
    print "testing_time={} s".format(testing_time)
    test_binned = test.split( size=bin_size_test )
    print "binsize={}, test bins = {}".format(bin_size_test,len(test_binned))

    # find number of spikes within test datasets
    n_spikes = []
    sum_tt = []
    mean_tt = []
    no_spike_bin_idx = []
    spike_bin_idx = []
    n_spikes_all = np.zeros(len(test_binned))
    max_bin = 0
    max_spike = 0

    for i, key in enumerate( ephys ):
        tt = ephys[key]
        n_spikes.append( test_binned.contains(tt['spike_times'])[1] )
        n_spikes_all = n_spikes_all + n_spikes[i]
        sum_tt.append(sum(n_spikes[i]))
        mean_tt.append(np.mean(n_spikes[i]))
    print"get spike count done"
    n_spikes_bin = []
    for i in range(len(n_spikes_all)):
        n_spikes_bin.append(np.asarray([n_spikes[j][i] for j in range(len(ephys))],dtype=np.int32))
        if n_spikes_all[i] > max_spike:
            max_spike = n_spikes_all[i]
            max_bin = i
    print"get spike count per bin done"
    if rm_no_spk:
        for j,n in enumerate(n_spikes_all):
            if n==0:
                no_spike_bin_idx.append(j)
            else:
                spike_bin_idx.append(j)
        if len(no_spike_bin_idx)>0:
            test_binned = test_binned[spike_bin_idx]
        print "{} no spike bins removed".format(len(no_spike_bin_idx))

    testing_time = np.sum( test_binned.duration )
    print "testing_time={} s".format(testing_time)
    print ""

    if count_bins_each_event:
        # get number of bins in each event
        if replay:
            event_bins = [[]]*len(test)
            for j,evnt_bin in enumerate(test_binned):
                for i,env in enumerate(test):
                    if env[0]<=evnt_bin[0] and env[1]>=evnt_bin[1]:
                        event_bins[i] = event_bins[i] + [j];
                        break
            #spio.savemat('event_bins.mat',{'event_bins':event_bins})
            true_behavior = []
        else:
            event_bins = []
            true_behavior = interpolate.interp1d( behavior["time"], behavior["linear_position"],\
                    kind='linear', axis=0 ) ( test_binned.center )
        print"get #bins in each event done"
    else:
        event_bins=len(test_binned)
        true_behavior = []
    
    return test_binned, event_bins, n_spikes_all, true_behavior

def load_mcd_from_file(file_path,n_features,n_shanks,config):
    # prepare encoding
    covars = create_covariance_array( config.behav_bw_cm, config.spf_bw_mV, n_features)

    #print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    #print "loading mixture"
    #print "n_features = {}".format(n_features)
    # created encoding points
    #print "feature bandwidth = {}".format(config.spf_bw_mV*1000)
    #print "compression threshold = {}".format(config.compression_threshold)
    #print "number of shanks = {}".format(n_shanks)
    # create compressed (empty) joint density with all tetrodes (even if they have too few spikes)
    mcd_spikebehav = []
    for i in range(n_shanks):
        cov_tt = covars[:n_features+1]
        mcd_spikebehav.append( mcd( ndim=len(cov_tt), sample_covariance =\
            cov_tt, method='bandwidth', threshold=config.compression_threshold,\
            name=('TT{}'.format(i) ) ) )
        mcd_spikebehav[i].load_from_mixturefile('{}/mcd_sb{}.mixture'.format(file_path,i))
    
    mcd_behav = mcd( ndim=1, sample_covariance=covars[:1], threshold=0,\
        name='behavior' )
    mcd_behav.load_from_mixturefile('{}/mcd_behav.mixture'.format(file_path))
    return mcd_spikebehav, mcd_behav

def getGrid(behavior,config):
    half_grid_size_cm = config.grid_element_size_cm / 2.0
    xgrid_vector = np.arange( np.nanmin( behavior["linear_position"] ) + half_grid_size_cm,\
        np.nanmax( behavior["linear_position"] ) - half_grid_size_cm, config.grid_element_size_cm )
    grid = xgrid_vector.reshape( (len(xgrid_vector),1) )
    return grid

def testSpikes(spike_features,bins,tt_included,spike_ampl_mask,sf_keys):
    test_spikes = []
    n_spikes = []
    j = 0
    for i, tt in enumerate( spike_features ):    
        if tt_included[i]:
            sel_test_spikes = bins.contains(tt[sf_keys[0]])[0]
            test_spikes.append(\
                np.vstack( ( tt[sf_keys[1]][sel_test_spikes].T[spike_ampl_mask[j]],\
                            tt[sf_keys[0]][sel_test_spikes] ) ) )  
            n_spikes.append( bins.contains(tt[sf_keys[0]])[1] )
            j += 1
    return test_spikes,np.array(n_spikes)

def getTestSpikes(spike_features, bins,  tt_included, spike_ampl_mask, bin_size, \
        shuffle=False, binned=False, sf_keys=["spike_times", "spike_amplitudes"]):
    
    bins = seg( bins )
    if bins.hasoverlap():
        raise ValueError("Bins have overlap")
        
    assert( len(spike_ampl_mask) <= len(tt_included) )
    assert( len(spike_features) == len(tt_included) )   
    assert( len(spike_ampl_mask) == sum(tt_included) )         
        
    # check bin size
    bin_sizes = np.unique( bins.duration[None,:] )
    if not np.all( np.isclose( bin_sizes, bin_size) ):
        raise ValueError("Not all bins have the requested bin size")
    
    if binned:
        test_spikes_binned = []
        n_spikes_binned = []
        for i in range(len(bins)):
            if i%1000==0:
                print"{} bins done".format(i)
            test_spikes,n_spikes = testSpikes(spike_features,bins[i],tt_included,spike_ampl_mask,sf_keys)
            if shuffle:
                test_spikes_tmp = []
                tmp=[]
                for j in range(len(test_spikes[0])):
                    nn = np.concatenate(([test_spikes[i][j] for i in range(len(test_spikes))]))
                    if j==0:
                        tmp=nn
                    else:
                        tmp = np.vstack((tmp,nn))
                for i in range(len(test_spikes)):
                    test_spikes_tmp.append(tmp)
                test_spikes = test_spikes_tmp

            test_spikes_binned.append(test_spikes)
            n_spikes_binned.append(n_spikes)
        return test_spikes_binned,n_spikes_binned
    else:
        test_spikes,n_spikes = testSpikes(spike_features,bins,tt_included,spike_ampl_mask,sf_keys)
        return test_spikes,np.array(n_spikes)

def getErrors(true_behavior,logpos,grid):
    decoded_behavior = grid[np.nanargmax(logpos, axis=1)].flatten()
    assert(len(true_behavior) == len(decoded_behavior))
    errors = np.array( [np.linalg.norm(pred_i - true_behav_i) \
                for pred_i, true_behav_i in zip(decoded_behavior, true_behavior)])
    return errors

