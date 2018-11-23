"""
=====================================================
OpenEphys file utilities (:mod:`fklab.io.open_ephys`)
=====================================================

.. currentmodule:: fklab.io.open_ephys

Utilities to read Open Ephys data files.

.. autosummary::
    :toctree: generated/
    
    get_sample_rate
    get_event_times
    get_experiment_start_time
    check_synchronization
    
"""

import numpy as np
import h5py
import scipy.stats
import scipy.interpolate

import fklab.statistics.correlation

__all__ = ['get_sample_rate', 'get_event_times', 'get_experiment_start_time',
           'check_synchronization']

def get_sample_rate(f, recording=0):
    """Retrieve sampling rate from Open Ephys file.
    
    It assumes the Open Ephys file to be in HDF5 format.
    
    Parameters
    ----------
    f : str or HDF5 file
        Either the file name of a valid HDF5 file, or an opened HDF5 file.
    recording : int, optional
    
    Returns
    -------
    fs : float
        Sample rate
    """
    
    if isinstance(f, basestring):
        f = h5py.File(f, 'r')
    
    fs = f['recordings/{0}'.format(recording)].attrs['sample_rate']
        
    return fs

def get_event_times(f, TTLchan=0, rising=True):
    """Retrieve event times from Open Ephys file.
    
    The function assumes the Open Ephys file to be in HDF5 format, and
    looks for the path `event_files/TTL/events` in the file.
    
    Parameters
    ----------
    f : str or open HDF5 file
        File name or already opened Open Ephys file.
    TTLchan : int, optional
        The TTL channel for which the event times will be returned.
    rising : bool, optional
        Look for rising edges of the event.
    
    Returns
    -------
    event_times : float array
    
    """
    
    if isinstance(f, basestring):
        f = h5py.File(f, 'r')
    
    chan_mask = f['event_types/TTL/events/user_data/event_channels'][:] == TTLchan
    edge_mask = f['event_types/TTL/events/user_data/eventID'][:] == int(rising)
    
    event_times = f['event_types/TTL/events/time_samples'][ np.logical_and(chan_mask, edge_mask) ]
    
    event_times = event_times / get_sample_rate(f)
    
    return event_times

def get_experiment_start_time(f):
    """Retrieve start time of experiment.
    
    Parameters
    ----------
    f : str or open HDF5 file
        File name or already opened Open Ephys file.
    
    Returns
    -------
    start_time : float
    
    """
    
    if isinstance(f, basestring):
        f = h5py.File(f, 'r')
    
    return f['event_types/Messages/events/time_samples'][1]/ get_sample_rate(f)


def check_synchronization( t1, t2, extrapolate=False ):
    """Check synchronization of a clock signal.
    
    It is assumed that the clock signal consists of a set of events
    with (Poisson) random intervals that are timestamped on two different
    data acquisition systems.
    
    Parameters
    ----------
    t1, t2 : 1d array
        Clock event timestamps.
    extrapolate : bool
        Time interpolator will extrapolate beyond the overlapping (sync-ed)
        time window in `t1` and `t2`.
    
    Returns
    -------
    offset : float
        Average time difference between synchronized clock signals. `offset`
        will (on average) convert a time value from time base `t1` to 
        time base `t2`.
    drift : float
        The number of seconds per second drift in the offset value. Calculated
        by regressing the time difference between synchronized clock signals on 
        the `t1`. Ideally, `drift` should be zero (the internal clocks of 
        the two data acquisition systems run at the same rate). If there
        exists significant drift, then one may be better off using the
        `interpolator` function rather than a fixed time offset.
    interpolator : callable
        Function that transforms time values from `t1` time base to
        `t2` time base.
        
    
    """
    
    # compute clock event intervals
    intervals1 = np.diff( t1 )
    intervals2 = np.diff( t2 )
    
    # use cross correlation to identify best line up of both events
    cc, lags = fklab.statistics.correlation.xcorrn( intervals1, intervals2, scale='coeff', remove_mean=True )
    
    idx = np.argmax( cc )
    lag = lags[0][idx]
    
    
    if lag>0:
        n = np.minimum( len(t1) - lag, len(t2) )
        deltas = t2[:n] - t1[lag:lag+n]
        offset = np.mean( deltas )
        
        drift,_,_,_,_ = scipy.stats.linregress( t1[lag:lag+n], deltas )
        
        fcn = scipy.interpolate.interp1d( t1[lag:lag+n], t2[0:n], assume_sorted=True, fill_value='extrapolate' if extrapolate else np.nan)
        
    else:
        n = np.minimum( len(t1), len(t2) + lag )
        deltas = t2[-lag:n-lag] - t1[:n]
        offset = np.mean( deltas )
        
        drift,_,_,_,_ = scipy.stats.linregress( t1[0:n], deltas )
        
        fcn = scipy.interpolate.interp1d( t1[0:n], t2[-lag:n-lag], assume_sorted=True, fill_value='extrapolate' if extrapolate else np.nan)
        
        
    return offset, drift, fcn


