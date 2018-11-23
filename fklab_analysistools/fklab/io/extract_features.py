from __future__ import division
import numpy as np
from os import path

import collections
import fklab.io.neuralynx as nlx
import fklab.io.mwl as mwl
from fklab.utilities.general import blocks


def compute_features( filename ):
    
    invert = 1.
    
    try:
        fid = nlx.NlxOpen( filename )
        fid.correct_inversion = False
        if not fid.header['InputInverted']:
            invert = -1. #we want to invert, so we can use max to find spike peaks
    except nlx.NeuralynxIOError:
        fid = mwl.MwlOpen( filename )
    
    nrec = fid.nrecords
    
    dtype = [('id',np.int32),('time',np.float64)]
    for k in range(fid.nchannels):
        dtype.append( ('peak'+str(k),np.float32) )
    
    dtype = np.dtype(dtype)
    
    root, name = path.split( fid.fullpath )
    name, ext = path.splitext( name )
    
    outfile = mwl.MwlFileFeature.create( path.join(root,name), dtype=dtype)
    
    blocksize=1000
    data = np.zeros( blocksize, dtype=dtype )
    
    for start,n in blocks( nitems=nrec, blocksize=blocksize ):
        t = fid.data.time[start:(start+n)]
        w = fid.data.waveform[start:(start+n)]
        
        peak = np.amax(invert*w, axis=1)
        
        data['id'][0:n] = np.arange( start, start+n )
        data['time'][0:n] = t
        for k in range(fid.nchannels):
            data['peak'+str(k)][0:n] = peak[:,k]
        
        outfile.append_data( data[0:n] )
        

def extract_channel_mask( nlx_spike_file ):
    """Extract a boolean mask of active (non-disabled) channels from an opened
    Neuralynx spike file    
    
    Parameters
    ----------
    nlx_spike_file : NlxFileSpike
        opened Neuralynx spike file
        
    Returns
    -------
    channel_mask : 1d boolean array
        mask of enabled channels

    """    
    
    n_channels = nlx_spike_file.nchannels
    mask = np.ones( n_channels, dtype='bool' )
    if "DisabledSubChannels" in nlx_spike_file.header.keys():
        list_disabled = nlx_spike_file.header["DisabledSubChannels"]
        for c in list_disabled:
            if c.isdigit():
                mask[int(c)] = False
        
    return mask


def extract_spike_amplitudes( path_to_data, min_ampl=0, perc=50, max_n_tt=40,\
min_spike_rate=0.1, exclude=[], start=None, stop=None, print_messages=False ):
    """Extract spike amplitudes from a Neuralynx dataset, applying optional filters
    that discard low amplitudes spikes.
    
    Parameters
    ----------
    path_to_data : string
        the path to data directory containing the .ntt files of the dataset
    min_ampl : float
        the min value of the included spike amplitude
    perc : float in range [0, 100]
        percentile of the distribution of the sensor amplitudes to be used for comparing
        the spike amplitude to a give threshold; a value of 0 means that the all
        amplitudes must be above the given threshold min_ampl; a value of 100
        means that only the max amplitude must be above threshold 
    max_n_tt : integer
        maximum number of tetrodes expected in the dataset
    min_spike_rate : float
        minimum spiking rate for including the tetrode in the dataset
    start : float
        start time for reading spike data files
    stop : float
        end time for reading spike data files
    print_messages :  bool
        whether or not updates about the extraction of spike amplitudes should be printed or not
    
    Returns
    -------
        spike_features : dictionary
            containging all extracted spike features and related timestamps that passed
            the criteria
        channel_masks : list of 1d boolean arrays
            each array contains the boolean mask of the selected channel of each tetrode
        start_time : float
            Timestamp of the first spike
        stop_time : float
            Timestamp of the last spike
    
    """
    
    if print_messages:
        print ("I'm reading the spike files and extracting the spike feaatures ... ")
    spike_features = []
    channel_masks = []
    start_time = np.inf
    stop_time = 0
    
    event_file = nlx.NlxOpen( path.join( path_to_data, "Events.nev" ) )
    if start is None:
        start = event_file.starttime
    if stop is None:
        stop = event_file.stoptime
    
    for tt in range(max_n_tt):
        
        fullpath_spike_file = path.join( path_to_data, "TT" + str(tt+1) + ".ntt" )
        
        if path.isfile( fullpath_spike_file ) and (tt+1) not in exclude:
            
            spike_file = nlx.NlxOpen( fullpath_spike_file )
            channel_mask = extract_channel_mask( spike_file ) 
            channel_masks.append( channel_mask )
            n_active_channels = sum( channel_mask )
            
            spike_file.correct_inversion = False
            spikes = spike_file.readdata( start=start, stop=stop )
            n_spikes = np.shape(spikes.waveform)[0]
            spike_rate = n_spikes / (stop - start )
            if print_messages:
                print "spike rate = {0:.3f} spikes/s" .format( spike_rate )
            if spike_rate >= min_spike_rate:
                amplitudes_tt = collections.OrderedDict()
                amplitudes_tt["original_id"] = "TT" + str(tt+1)
                
                amplitudes_tt["value"] = np.zeros( ( n_spikes, n_active_channels ) )
                amplitudes_tt["time"] = np.zeros( n_spikes )
                m = 0
                for n in range(n_spikes):
                    sf_temp = np.max( spikes.waveform[n, :, :], axis=0 )[channel_mask]
                    # amplitudes could be better computed with quadratic interpolation,
                    # but probably not necessary at high stream rate
                    if np.percentile(sf_temp, perc) > min_ampl:
                        amplitudes_tt["value"][m] = sf_temp   
                        amplitudes_tt["time"][m] = spikes.time[n]
                        m += 1
                amplitudes_tt["value"] = amplitudes_tt["value"][:m]
                amplitudes_tt["time"] = amplitudes_tt["time"][:m]
                spike_features.append( amplitudes_tt )
                if print_messages:
                    print "spike features from TT{0} are loaded\n" .format(tt+1)
                start_time = min(spike_file.starttime, start_time)
                stop_time = max(stop_time, spike_file.endtime)
            else:
                print "spike features from TT{0} were excluded because spike rate was lower than {1}\n" .format(tt+1, min_spike_rate)
    
    return spike_features, channel_masks, start_time, stop_time


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract waveform features for xclust.')
    parser.add_argument('infile')
    #parser.add_argument( '-f', '--features', nargs='*' )
    args = parser.parse_args()
    
    compute_features( args.infile )
    
    
    
