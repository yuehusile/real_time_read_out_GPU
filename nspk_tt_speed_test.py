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

from compressed_decoder.gmmcompression.fkmixture import GpuDecoderClass as GpuDecoder
from compressed_decoder.gmmcompression.fkmixture import SignificanceAnalyzerClass as SigAnalyzer
#from compressed_decoder.kde import MergingCompressionDensity as mcd
#from compressed_decoder.kde import Decoder, create_covariance_array
from compressed_decoder.kde_gpu import MergingCompressionDensity as mcd
from compressed_decoder.kde_gpu import Decoder, create_covariance_array

import config_file as config
import pdb
import collections
import scipy.io as spio
import matplotlib.pyplot as plt
import random

n_pos = 10000
n_time_bin = 10
n_max_spike = 100
n_spike_group = 10
n_shuffle = 1000
#sigAna = SigAnalyzer(n_pos,n_spike_group,n_shuffle,n_time_bin,n_max_spike)
nn = range(5000);
t00 = time.time()
for i in range(n_shuffle):
    random.shuffle(nn);
t01 = time.time()

print "shuffle time={}ms".format((t01-t00)*1e3);

#pdb.set_trace()

class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        self.output = None
        threading.Thread.__init__(self)
 
    def run(self):
        self.output = self._target(*self._args)

######################################################################################################
# load data from matlab file
######################################################################################################

# read matlab data set
prep_mat = spio.loadmat('../matfiles/Achi110119400_22000firstpca_all.mat')

# read replay events and data
event_mat = spio.loadmat('../matfiles/postNREMevent.mat')
sleep_mat = spio.loadmat('../matfiles/Achi110124000_36700firstpca_all_sleep.mat')

# tmp data structures
eph = prep_mat['ephys']
eph1 = sleep_mat['ephys']
behav = prep_mat['behavior']

behavior = dict()
# *100 is for transfering units from m to cm
behavior['linear_position'] = behav[0][0][0].flatten()*100
behavior['speed'] = behav[0][0][1].flatten()
behavior['time'] = behav[0][0][2].flatten()


config.test_sleep = True
# log ll for all replay bins
logpos_all = []
n_total_features = 10
#making virtual dataset:
# use 10 features, shanks are #8:#12 shanks 
n_features = 9
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
    ephys[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = eph[0][i][1][:,range(n_features+1)][sort_indices,:]/1000.0
    ephys[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = ephys[u'TT'+str(v_shank_idx)]['spike_amplitudes'].astype(np.float32)
check_ephys_dataset( ephys )

# read sleep epoch data
ephys1 = collections.OrderedDict()
ephys2 = collections.OrderedDict()

for i in range(len(eph1[0])):
    v_shank_idx = i+1
    ephys2[u'TT'+str(v_shank_idx)] = dict()        
    # reorder the data based on timestamps
    x = eph1[0][i][0].flatten()
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    
    ephys2[u'TT'+str(v_shank_idx)]['spike_times'] = x
    ephys2[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = eph1[0][i][1][:,range(n_features+1)][sort_indices,:]/1000.0 # first pca 10ch
    #ephys1[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = y[:,range(n_features+1)][sort_indices,:]/1000.0 # first pca 10ch
    ephys2[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = ephys2[u'TT'+str(v_shank_idx)]['spike_amplitudes'].astype(np.float32)


x = eph1[0][0][0]
y = eph1[0][0][1]
for i in range(len(eph1[0])-1):
    x = np.append(x,eph1[0][i+1][0].flatten())
    #pdb.set_trace()
    y = np.append(y,eph1[0][i+1][1],axis=0)
#pdb.set_trace()

for i in range(len(eph1[0])):
    v_shank_idx = i+1
    ephys1[u'TT'+str(v_shank_idx)] = dict()        
    # reorder the data based on timestamps
    #x = eph1[0][i][0].flatten()
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    
    ephys1[u'TT'+str(v_shank_idx)]['spike_times'] = x
    #ephys1[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = eph1[0][i][1][:,range(n_features+1)][sort_indices,:]/1000.0 # first pca 10ch
    ephys1[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = y[:,range(n_features+1)][sort_indices,:]/1000.0 # first pca 10ch
    ephys1[u'TT'+str(v_shank_idx)]['spike_amplitudes'] = ephys1[u'TT'+str(v_shank_idx)]['spike_amplitudes'].astype(np.float32)
#check_ephys_dataset( ephys1 )



# choose tt/shank to use
tmp = ephys
tmp1 = ephys2
cnt = 0;
#left = len(ephys.keys())/2; # number of tt/shanks want to use
left = len(ephys.keys()); # number of tt/shanks want to use
st = 0 # first tt/shanks starts from
for key1 in ephys.keys():
    cnt+=1
    if cnt>left+st or cnt<=st :
       del tmp[key1]
       del tmp1[key1]
ephys = tmp
ephys2 = tmp1

#pdb.set_trace()

# show active tt/shanks
n_tt_dataset = len(ephys)
n_active_channels = np.array( [ np.shape( ephys[key]["spike_amplitudes"][1] )[0]\
    for key in ephys.keys() ] )


######################################################################################################
# preprocessing: select run data, split train/test, show dataset information, remove no spike bins
######################################################################################################
# select run segments
print "run speed={}, min run duration={}, binsize={}".format(config.run_speed,config.min_run_duration,config.bin_size_run)
#flags_to_choose = [(aa > config.run_speed) & (bb>20661) for aa,bb in zip(behavior["speed"],behavior["time"])]
run = seg.fromlogical( behavior["speed"] > config.run_speed, x=behavior["time"] )
run = run[ run.duration > config.min_run_duration ]

# split segments into independent bins
bin_size_test = config.bin_size_run
run_binned = run.split( size=bin_size_test )

# choose percentage of run data(bins) to use (later part)
r_data_not_to_use = 0.95
#r_data_not_to_use = 0.5
first_bin_idx = int(r_data_not_to_use*len(run_binned))
used_data = run_binned[first_bin_idx:]

# set percentage of train/test bins
r_train = 0.1
#for r_train in [0.1,0.3,0.5,0.7,0.9]:
for r_train in [0.9]:
    n_train = int(r_train*len(used_data))
    train = used_data[0:n_train]
    test = used_data[n_train:]

    print "###################################################################################"
    total_time = ephys[u'TT1']['spike_times'][-1]-ephys[u'TT1']['spike_times'][0]    
    print "total recording time = {} min, start time = {}, end time = {}".format(total_time/60,ephys[u'TT1']['spike_times'][0],ephys[u'TT1']['spike_times'][-1])
    print ""
    print "run bins = {}, first run bin time = {}, last run bin time = {}".format(len(run_binned),run_binned[0][0],run_binned[-1][0])
    print ""
    print "ratio of used bins = {}, first used bin idx in run bins={}".format(1-r_data_not_to_use,first_bin_idx)
    print ""
    print "used bins = {}, first used bin time = {}, last used bin time = {}".format(len(used_data),used_data[0][0],used_data[-1][0])
    print ""
    print "ratio of train bins = {}".format(r_train)
    print ""
    print "train bins = {}, test bins = {} first train bin time = {}, first test bin time = {}".format(len(train),len(test),train[0][0],test[0][0])
    print ""
    training_time = np.sum( train.duration )
    testing_time = np.sum( test.duration )
    print "training_time={} s".format(training_time)
    print ""
    print "testing_time={} s".format(testing_time)
    print ""
    test_binned = test.split( size=bin_size_test )

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
        mean_tt.append(np.mean(n_spikes[i]))
    
    for j,n in enumerate(n_spikes_all):
        if n==0:
            no_spike_bin_idx.append(j)
        else:
            spike_bin_idx.append(j)
    if len(no_spike_bin_idx)>0:
        train = train[spike_bin_idx]
    
    print "number spikes for training:"
    print ""
    print sum_tt
    print "average spikes per bin:"
    print ""
    print mean_tt
    print "total spikes =  {}, total average spikes/bin = {}".format(sum(sum_tt),sum(mean_tt))
    print ""
    print "{} training bins have no spikes has been removed, now have {} bins".format(len(no_spike_bin_idx),len(train))
    print ""
   
    # set sleep status to decode replay
    #config.test_sleep = True
    if config.test_sleep:
        test = seg(event_mat['postNREMevent'])
        test_binned = test.split( size=config.bin_size_sleep )
        print "test sleep: binsize={}, test bins = {}".format(config.bin_size_sleep,len(test_binned))
        print ""
        bin_size_test = config.bin_size_sleep
        ephys1 = ephys2
    else:
        ephys1 = ephys

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
        #tt = ephys1[key]
        tt = ephys2[key]
        n_spikes.append( test_binned.contains(tt['spike_times'])[1] )
        n_spikes_all = n_spikes_all + n_spikes[i]
        sum_tt.append(sum(n_spikes[i]))
        mean_tt.append(np.mean(n_spikes[i]))
    n_spikes_bin = []
    for i in range(len(n_spikes_all)):
        n_spikes_bin.append(np.asarray([n_spikes[j][i] for j in range(len(ephys))],dtype=np.int32))
        if n_spikes_all[i] > max_spike:
        #if n_spikes_all[i] == 100:
            max_spike = n_spikes_all[i]
        #    print n_spikes_all[i]
            max_bin = i
        #    break;
    #pdb.set_trace()
    
    for j in range(len(n_spikes_all)):
        if n==0:
            no_spike_bin_idx.append(j)
        else:
            spike_bin_idx.append(j)
    #if len(no_spike_bin_idx)>0:
    #    test_binned = test_binned[spike_bin_idx]
    
    #print "number spikes for testing:"
    #print ""
    #print sum_tt
    #print "average spikes per bin:"
    #print ""
    #print mean_tt
    #print "total spikes =  {}, total average spikes/bin = {}".format(sum(sum_tt),sum(mean_tt))
    #print ""
    #print "{} testing bins have no spikes has been removed, now have {} bins".format(len(no_spike_bin_idx),len(test_binned))
    #print ""
    print "bin with max spikes:  {}".format(max_bin)
    print "max spike= {}".format(max_spike)
    #pdb.set_trace()
    #test_binned = test_binned[max_bin]
    #iii = 1
    #test_binned = test_binned[iii]
    #print "bin {} spike={}".format(iii,n_spikes_all[iii])    
    training_time = np.sum( train.duration )
    testing_time = np.sum( test_binned.duration )
    print "training_time={} s".format(training_time)
    print ""
    print "testing_time={} s".format(testing_time)
    print ""

    pdb.set_trace()
# get number of bins in each event
    event_bins = [[]]*len(test)
    for j,evnt_bin in enumerate(test_binned):
        for i,env in enumerate(test):
            if env[0]<=evnt_bin[0] and env[1]>=evnt_bin[1]:
                #event_bins[i] = event_bins[i] + 1;
                event_bins[i] = event_bins[i] + [j];
                break
    #spio.savemat('event_bins.mat',{'event_bins':event_bins})

######################################################################################################
# encoding
######################################################################################################

# prepare encoding
    train_behav = prep_dec.extract_train_behav( train, behavior )
    train_spike, tetrode_inclusion_mask =\
        prep_dec.extract_train_spike( train, ephys, config.min_n_encoding_spikes )
    covars = create_covariance_array( config.behav_bw_cm, config.spf_bw_mV )

    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "n_feature = {}".format(len(ephys[u'TT1']['spike_amplitudes'][0]))
    while len(covars)-1<len(ephys[u'TT1']['spike_amplitudes'][0]):
        cov = covars[-1]
        covars = np.append(covars, cov)

# created encoding points
    encoding_points = [ prep_dec.attach( train_behav, tt ) for tt in train_spike ] 
# create compressed (empty) joint density with all tetrodes (even if they have too few spikes)
# ################# hack: set covars manually
    cov_pos = [covars[0]]

    #base_cov = [0.15]*(len(covars)-1)
    base_cov = [0.0]*(len(covars)-1)
    #for step in np.arange(1,1.05,0.1):
    #for step,comp in zip(np.arange(0.03,0.16,0.02),np.arange(0.1,1,0.1)):
    #for step,comp in zip(np.arange(0.03,0.16,0.02),np.arange(1,6,1)):
    #for step,comp in zip(np.arange(0.04,0.16,0.01),np.arange(0,1,0.1)):
    #for comp in np.arange(0,2.1,0.2):
    for comp in np.arange(0,1,1):
        for step in np.arange(0.14,0.15,0.01):

            cov_pca4 = [(i+step)**2 for i in base_cov]

            covars = np.concatenate([cov_pos,cov_pca4])
            print "feature bd = {}".format(i+step)
            print "comp thresh = {}".format(comp)
            print " n shanks = {}".format(left)
            config.compression_threshold = comp

            mcd_spikebehav = []
            id_selected_sensors = np.array(ephys.keys())[tetrode_inclusion_mask].tolist()
            for i in range(n_tt_dataset):
                cov_tt = covars[:n_active_channels[i]+1]
                mcd_spikebehav.append( mcd( ndim=len(cov_tt), sample_covariance =\
                    cov_tt, method='bandwidth', threshold=config.compression_threshold,\
                    name=(ephys.keys()[i] ) ) )
# encoding part: addsamples function
# fill joint compressed density with encoding points
            for i, dec in enumerate(mcd_spikebehav):
                if tetrode_inclusion_mask[i]:
                    points = encoding_points[i]
                    dec.addsamples( points )

# uncompressed density of behavior data
            mcd_behav = mcd( train_behav[0], sample_covariance=covars[:1], threshold=0,\
                name='behavior' )
            training_time = np.sum( train.duration )
# define decoding grid
            with open( join( dirname( config.path_to_preprocessed_dataset ),\
            "preprocessing_info.yaml" ), 'r' ) as f_yaml:
                info_preprocess = yaml.load( f_yaml )
            pixel_to_cm = info_preprocess["ConversionFactorPixelsToCm"]
            if config.path_to_experimental_grid is None:
                half_grid_size_cm = config.grid_element_size_cm / 2
                xgrid_vector = np.arange( np.nanmin( behavior["linear_position"] ) + half_grid_size_cm,\
                    np.nanmax( behavior["linear_position"] ) - half_grid_size_cm, config.grid_element_size_cm )
                grid = xgrid_vector.reshape( (len(xgrid_vector),1) )
            else:
                grid_pixel = np.load( config.path_to_experimental_grid )
                grid_pixel = grid_pixel.reshape( (len(grid_pixel),1) )
                grid = grid_pixel / pixel_to_cm
            
            #pdb.set_trace()
######################################################################################################
# decoding
######################################################################################################
# decode during test run epochs
            print "cpu_decoder init"
            decoder = Decoder( mcd_behav, mcd_spikebehav[:config.cpu_n_tt_use], training_time,\
                    grid, config.offset )
            spike_ampl_mask_list = np.array( [ np.ones( ephys[key]["spike_amplitudes"].shape[1], dtype='bool')\
                for key in ephys.keys() ] )[tetrode_inclusion_mask]

            print "cpu_decoding"
            t0 = time.time()
            posterior_cpu, logpos_cpu, n_spikes_cpu, n_tt_cpu = decoder.decode(\
                [ ephys1[key] for key in ephys1.keys() ][:config.cpu_n_tt_use],\
                test_binned,\
                tetrode_inclusion_mask[:config.cpu_n_tt_use],\
                spike_ampl_mask_list[:config.cpu_n_tt_use],\
                #config.bin_size_run,\
                bin_size_test,\
                sf_keys=["spike_times", "spike_amplitudes"],
                mt=True)    
            decoded_behavior_cpu = grid[np.nanargmax(logpos_cpu, axis=1)].flatten()
            t1 = time.time()
            cpu_decoding_time = t1 - t0
            pax_cpu = decoder.cached_pax

######################################################################################################
# gpu decoding
######################################################################################################
# decode during test run epochs
            print "gpu_decoder init"
            decoder_gpu = Decoder( mcd_behav, mcd_spikebehav[:config.gpu_n_tt_use], training_time,\
                    grid, config.offset, use_gpu = True, gpu_batch_size=1024)
            
            sigAna = SigAnalyzer(len(grid),len(ephys.keys()),bin_size_test,n_shuffle,n_time_bin,n_max_spike)
            sigAna.uploadParam(decoder.pix(),decoder.lx())
            
            spike_ampl_mask_list = np.array( [ np.ones( ephys[key]["spike_amplitudes"].shape[1], dtype='bool')\
                for key in ephys.keys() ] )[tetrode_inclusion_mask]
            
            sum_nspk = np.sum(n_spikes_bin,axis=1)
            nz_idx = np.where(sum_nspk>0)
            #id2000 = np.where(nz_idx[0]>=2000)[0]
            id2000 = np.where(nz_idx[0]>=2962)[0]
            #id3000 = np.where(nz_idx[0]<3000)[-1]
            id3000 = np.where(nz_idx[0]<2983)[-1]
            print "gpu_decoding"
            t0 = time.time()
            max_bin_time = 0
            max_bin_idx = 0
            max_shf_time = 0
            max_shf_idx = 0
            pt = [];
            dt = [];
            shft = [];
            nspk = [];
            prob_bin_idx = [];
            all_prob = [];
            pdb.set_trace()
            #for bin_id in range(100):#range(len(test_binned)):
            #for bin_id in nz_idx[0][:100]:#range(len(test_binned)):
            for bin_id in nz_idx[0][id2000[0]:id3000[-1]]:#nz_idx[0]:#range(len(test_binned)):
                t0 = time.time()
                posterior_gpu, logpos_gpu, n_spikes_gpu, n_tt_gpu = decoder_gpu.decode(\
                    [ ephys1[key] for key in ephys1.keys() ][:config.gpu_n_tt_use],\
                    test_binned[bin_id],\
                    tetrode_inclusion_mask[:config.gpu_n_tt_use],\
                    spike_ampl_mask_list[:config.gpu_n_tt_use],\
                    #config.bin_size_run,\
                    bin_size_test,\
                    sf_keys=["spike_times", "spike_amplitudes"],\
                    mt=True,
                    shuffle=True,update_spike_only=True)    

                t1 = time.time()
                posterior_gpu, logpos_gpu, n_spikes_gpu, n_tt_gpu = decoder_gpu.decode(\
                    [ ephys1[key] for key in ephys1.keys() ][:config.gpu_n_tt_use],\
                    test_binned[bin_id],\
                    tetrode_inclusion_mask[:config.gpu_n_tt_use],\
                    spike_ampl_mask_list[:config.gpu_n_tt_use],\
                    #config.bin_size_run,\
                    bin_size_test,\
                    sf_keys=["spike_times", "spike_amplitudes"],\
                    mt=True,
                    shuffle=True,decode_only=True)    
                #decoded_behavior_gpu = grid[np.nanargmax(logpos_gpu, axis=1)].flatten()
                #for ii in range(25):#range(n_time_bin+1):
                #if bin_id==251:
                #print "bin_id={}".format(bin_id)
                pdb.set_trace()
                t2 = time.time()
                prob,ready = sigAna.updateBin(decoder_gpu.cached_pax,n_spikes_bin[bin_id],decoder_gpu.mu)
                if ready:
                    #pdb.set_trace()
                    prob_bin_idx.append(bin_id)
                    if len(all_prob)==0:
                        all_prob = prob
                    else:
                        all_prob = np.vstack((all_prob,prob))
                t3 = time.time()
                if max_bin_time<t3-t1:
                    max_bin_time = t3-t1
                    max_bin_idx = bin_id
                if max_shf_time<t3-t2:
                    max_shf_time = t3-t2
                    max_shf_idx = bin_id
                print "time for {0} bin: shf = {1:.3f} ms, dec = {2:.3f} ms, prep = {3:.3f} ms,  n_spikes={4}".format(bin_id,(t3-t2)*1e3,(t2-t1)*1e3,(t1-t0)*1e3,sum_nspk[bin_id])
                print "max_bin_time={0:.3f},id={1},max_shf_time={2:.3f},id={3}".format(max_bin_time*1e3,max_bin_idx,max_shf_time*1e3,max_shf_idx)
                prep_time = (t1-t0)*1e3
                decode_time = (t2-t1)*1e3
                shf_time = (t3-t2)*1e3
                n_spk = sum_nspk[bin_id]
                pt.append(prep_time)
                dt.append(decode_time)
                shft.append(shf_time)
                nspk.append(n_spk)
            #pdb.set_trace()
            spio.savemat('prob_silicon_sleep2000-3000.mat',{'all_prob':all_prob,'prob_bin_idx':np.array(prob_bin_idx),'n_shuffle':n_shuffle})
            spio.savemat('shf_silicon_sleep.mat',{'prep_time':np.array(pt),'decode_time':np.array(dt),'shf_time':np.array(shft),'n_spk':np.array(nspk)})
            #print "total time for {} bins = {} ms, average per bin = {} ms".format(len(nz_idx[0]),(t1-t0)*1e3,(t1-t0)*1e3/len(nz_idx[0]))
            gpu_decoding_time = t1 - t0
            pax_gpu = decoder_gpu.cached_pax
            decoder_gpu.clearGpuMem()

######################################################################################################
# print results
######################################################################################################
            gpu_n_tt_use = left
            max_n_spikes = config.max_n_spikes_gpu_kde*2
            max_n_components = max([ _mcd.ncomponents() for _mcd in mcd_spikebehav ])
            #if logpos_all==[]:
            #    logpos_all = [np.empty( (logpos_cpu.shape[0],logpos_cpu.shape[1])) for n_sf in range(n_shuffle)]
            #logpos_all[shuffle_idx] = logpos_cpu

            print "\n\tCPU Decoding time: {0:.3f} ms/spike, total={1:.3f}" .format(\
                                          cpu_decoding_time/np.sum(n_spikes_cpu) * 1e3,cpu_decoding_time*1e3\
                                          )#cpu_kde_time/np.sum(n_spikes_cpu) * 1e3)
            #true_behavior = interpolate.interp1d( behavior["time"], behavior["linear_position"],\
            #    kind='linear', axis=0 ) ( test_binned.center )       
            # compute cpu decoding error
            #assert(len(true_behavior) == len(decoded_behavior_cpu))
            n_test_bins = len(decoded_behavior_cpu)
            #errors_cpu = np.array( [np.linalg.norm(pred_i - true_behav_i) \
            #    for pred_i, true_behav_i in zip(decoded_behavior_cpu, true_behavior)])
            
            #print "\nCPU decoding completed with {0} tetrodes done with median error of:\t{1:.2f} cm"\
            #    .format(n_tt_cpu, np.nanmedian(errors_cpu))
            #print "\nCPU decoding completed with {0} tetrodes done with mean error of:\t{1:.2f} cm"\
            #    .format(n_tt_cpu, np.nanmean(errors_cpu))
            
            #for i, err in enumerate(errors_cpu):
            #    if err>config.max_error:
            #        decoded_behavior_cpu[i] = config.max_error*2-decoded_behavior_cpu[i]
            #        errors_cpu[i] = abs(decoded_behavior_cpu[i]-true_behavior[i])
            #print "\nCPU decoding corrected  median error of:\t{1:.2f} cm"\
            #    .format(n_tt_cpu, np.nanmedian(errors_cpu))
            #print "\nCPU decoding corrected mean error of:\t{1:.2f} cm"\
            #    .format(n_tt_cpu, np.nanmean(errors_cpu))


            print "\n\tGPU Decoding time: {0:.3f} ms/spike,total={1:.3f}" .format(\
                                          gpu_decoding_time/np.sum(n_spikes_gpu) * 1e3,gpu_decoding_time*1e3\
                                          )#cpu_kde_time/np.sum(n_spikes_cpu) * 1e3)
            # compute gpu decoding error
            #assert(len(true_behavior) == len(decoded_behavior_gpu))
            #n_test_bins = len(decoded_behavior_gpu)
            #errors_gpu = np.array( [np.linalg.norm(pred_i - true_behav_i) \
            #    for pred_i, true_behav_i in zip(decoded_behavior_gpu, true_behavior)])
            
            #print "\nGPU decoding completed with {0} tetrodes done with median error of:\t{1:.2f} cm"\
            #    .format(n_tt_gpu, np.nanmedian(errors_gpu))
            #print "\nGPU decoding completed with {0} tetrodes done with mean error of:\t{1:.2f} cm"\
            #    .format(n_tt_gpu, np.nanmean(errors_gpu))
            
            #for i, err in enumerate(errors_gpu):
            #    if err>config.max_error:
            #        decoded_behavior_gpu[i] = config.max_error*2-decoded_behavior_gpu[i]
            #        errors_gpu[i] = abs(decoded_behavior_gpu[i]-true_behavior[i])
            #print "\nGPU decoding corrected  median error of:\t{1:.2f} cm"\
            #    .format(n_tt_gpu, np.nanmedian(errors_gpu))
            #print "\nGPU decoding corrected mean error of:\t{1:.2f} cm"\
            #    .format(n_tt_gpu, np.nanmean(errors_gpu))
            
            print n_features
            print "\nThreshold used for compression : {0:.2f}" .format( config.compression_threshold )
            print "\nGrid element size : {0} cm" .format( config.grid_element_size_cm )
            #print "\ndketime:{}/nonkdetime:{}/total:{}".format(t3-t2,t4-t3,gpu_decoding_time)
            #print "\ncovars:{}".format(covars)
            print "==================================================================================="
            #spio.savemat('error_02468_45_5cm.mat',{'erro':errors_cpu,'pos':true_behavior,'train_pos':train_behav})
    #f = plt.figure()
#for i, err in enumerate(errors_cpu):
#    if err>config.max_error:
#        decoded_behavior_cpu[i] = config.max_error*2-decoded_behavior_cpu[i]
    #plt.scatter(range(len(decoded_behavior_cpu)),decoded_behavior_cpu,c="g")
    #plt.plot(range(len(true_behavior)),true_behavior)
#    for i, err in enumerate(errors_cpu):
#        if err>config.max_error:

#plt.plot(range(len(true_behavior)),true_behavior)

#spio.savemat('logpos_cpu_train_10_5cm_shuffle.mat',{'logpos_all':logpos_all})
# show cumulative error distribution
#f1 = plt.figure()
#plt.hist(errors_cpu[0:-1],100,normed=1, histtype='step',cumulative=True)
#plt.show()
#pdb.set_trace()
