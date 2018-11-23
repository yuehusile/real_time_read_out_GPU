# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:41:30 2016

@author: davide
"""
import numpy as np

def partition( segmented_data, n_partitions, partition_index=0, fold=0,\
method='block', coincident=False, keepremainder=False ):
    """Partition segmented data into a test and train set of segments
    
    Parameters
    ----------
    segmented_data : vector of Segments
        input data
    n_partitions : integer
        number of partitions in which the dataset should be divided
    partition_index : integer
        index of the partition set to be used
    fold : integer
        0 or 1, to selected which of partition will be used as test
    method : string
        partition method (see partition method of Segment)
    keepremainder : bool
        (see partition method of Segment)
        
    Returns
    -------
    train : vector of Segments
        train data partition
    test :  vector of Segments
        test data partition
        
    """
    partitions = [i for i in segmented_data.partition(nparts=n_partitions,\
        method=method, keepremainder=False)]
    if n_partitions == 1:
        test = train = partitions[0]
    elif n_partitions == 2:
            if fold > 1:
                raise ValueError("Incorrect fold index")
            test = partitions[fold]
            if coincident:
                train = test
            else:
                train = partitions[1-fold]
    elif n_partitions > 2:
        if partition_index > n_partitions/2:
            raise ValueError("Incorrect partition index")
        if fold > 1:
            raise ValueError("Incorrect fold index")
        train = partitions[partition_index + fold]  
        if coincident:
            test = train            
        else:
            test = partitions[partition_index + 1-fold]
            
    return train, test
    
    
def attach( behavior, spike_features, interpolation_method='linear' ):
    """Create an array of timestamped encoding points
    
    Parameters
    ----------
    behavior : 2d numpy array
        behavior values with timestamps (timestamps on 2nd row)
    spike_features : Nd numpy array
        contains the spike features and their timestamps on the last row
    
    """
    from scipy import interpolate    
    
    a_t = spike_features[-1, :]
    amp = spike_features[:-1, :]
    interpolated_x = interpolate.interp1d( behavior[-1, :], behavior[0:-1],\
        kind=interpolation_method, axis=1, bounds_error=False )(a_t)
    nan_idx = np.isnan( np.sum(interpolated_x, axis=0) )
    xa = np.vstack( (interpolated_x[:, ~nan_idx], amp[:, ~nan_idx]) )
    xa = np.ascontiguousarray( xa.T )
    
    return xa
    
    
def extract_train_behav( train_segs, dataset_behavior ):
    """
    
    
    """
    sel_train_behav = train_segs.contains( dataset_behavior["time"] )[0]
    train_behav_temp = dataset_behavior["linear_position"][sel_train_behav]

    return np.vstack( ( train_behav_temp, dataset_behavior["time"][sel_train_behav]) )   
    
    
def extract_train_spike( train_segs, dataset_ephys, min_n_encoding_spikes ):
    
    train_spike = []
    n_tt_dataset = len(dataset_ephys.keys())
    tetrode_inclusion_mask = np.zeros( n_tt_dataset, dtype='bool' )

    for i, _ in enumerate(dataset_ephys):
        
        key = dataset_ephys.keys()[i]
        sel_train_spike = train_segs.contains( dataset_ephys[key]["spike_times"] )[0]
        n_selected_spikes_tt = sum(sel_train_spike)
        
        train_spike.append( np.vstack( (\
                dataset_ephys[key]["spike_amplitudes"][sel_train_spike].T,\
                dataset_ephys[key]['spike_times'][sel_train_spike] ) ) )
        
        if n_selected_spikes_tt > min_n_encoding_spikes:
            tetrode_inclusion_mask[i] = True
        else:
            print(key + " was marked as excluded from the trainining dataset")
            
    return train_spike, tetrode_inclusion_mask