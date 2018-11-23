"""
=======================================================
Event algorithms (:mod:`fklab.events.basic_algorithms`)
=======================================================

.. currentmodule:: fklab.events.basic_algorithms

Provides basic algorithms for event time vectors.

.. autosummary::
    :toctree: generated/
    
    split_eventstrings
    event_rate
    event_bin
    event_bursts
    filter_bursts
    filter_bursts_length
    event_count
    event_intervals
    filter_intervals
    complex_spike_index
    peri_event_histogram

"""


from __future__ import division

import math

import numpy as np
import scipy as sp
import scipy.interpolate
import numba

import fklab.utilities as util
from fklab.segments.basic_algorithms import check_segments, segment_remove_overlap, segment_contains

from fklab.codetools import deprecated

__all__ = ['split_eventstrings','event_rate','event_bin','event_bursts','filter_bursts',
           'filter_bursts_length','event_count','event_intervals',
           'filter_intervals','complex_spike_index','peri_event_histogram',
           'check_events','check_events_list']

def split_eventstrings( timestamp, eventstrings ):
    """Split event strings.
    
    Converts a sequence of timestamps and corresponding event strings to
    a dictionary with for each unique event string, the timestamps at
    which the event happened.
    
    Parameters
    ----------
    timestamp : 1d array-like
    eventstrings : sequence of str
    
    Returns
    -------
    dict
    
    """
    
    events = np.unique( eventstrings )
    d = { e:timestamp[ eventstrings==e ] for e in events }
    
    return d

def check_events(x, copy=True):
    """Convert input to event vector.
    
    Parameters
    ----------
    x : array-like
    copy : bool
        the output vector will always be a copy of the input
    
    Returns
    -------
    1d array
    
    """
    
    return util.check_vector( x, copy=copy, real=True )
    
def check_events_list(x, copy=True):
    """Convert input to sequence of event vectors.
    
    Parameters
    ----------
    x : array-like or sequence of array-likes
    copy : bool
        the output vectors will always be copies of the inputs
    
    Returns
    -------
    tuple of 1d arrays
    
    """
    
    return util.check_vector_list( x, copy=copy, real=True )

def event_rate( events, segments=None, separate=False):
    """Return mean rate of events
    
    Parameters
    ----------
    events : 1d array or sequence of 1d arrays
        vector(s) of event times (in seconds)
    segments : (n,2) array or Segment, optional
        array of time segment start and end times
    separate : bool, optional
        compute event rates for all segments separately
    
    Returns
    -------
    rate : array
        Mean firing rate for each of the input event time vectors.
        If `separate`=True, then a 2d array is returned, where `rate[i,j]`
        represents the mean firing rate for event vector `i` and
        segment `j`.
    
    """
    
    events = check_events_list( events )
    n = len(events)
    
    if segments is None: #ignore separate
        fr = np.array( [len(x)/(np.max(x)-np.min(x)) for x in events], dtype=np.float64 )
    else:
        segments = check_segments(segments,copy=False)
        if not separate:
            #combine segments
            segments = segment_remove_overlap(segments)
            #find number of events in segments
            ne = [np.sum(segment_contains(segments,x)[1]) for x in events]
            ne = np.float64( ne )
            #convert to rate
            fr = ne / np.sum( np.diff(segments,axis=1) )
        else:
            #find number of events in each segment
            ne = [segment_contains(segments,x)[1] for x in events]
            ne = np.float64( np.vstack(ne) )
            #convert to rate
            fr = ne / np.diff(segments,axis=1).reshape( (1,len(segments)) )
    
    return fr

def event_bin( events, bins, kind='count' ):
    """Count number of events in bins.
    
    Parameters
    ----------
    events : 1d array or sequence of 1d arrays
        vector(s) of sorted event times (in seconds)
    bins : ndarray
        array of time bin start and end times (in seconds). Can be either
        a vector of sorted times, or a (n,2) array of bin start and
        end times.
    kind : {'count','binary','rate'}, optional
        determines what count to return for each bin. This can be the
        number of events (`count`), the presence or absence of events
        (`binary`) or the local event rate in the time bin (`rate`).
    
    Returns
    -------
    counts : 1d array
        event counts for each bin
        
    """
    
    events = check_events_list( events )
    
    bins = check_segments( bins )
    
    sortflag = False
    if not util.isascending(bins[:,0]):
        sort_idx = np.argsort( bins[:,0], axis=0, kind='mergesort' )
        bins = bins[sort_idx]
        sortflag = True
    
    m = np.zeros( (bins.shape[0], len(events)), dtype=np.uint64 )
    
    #for each event vector, compute histogram
    #and sort event vector if needed (will be slower)
    #note that histc cannot be used since it does not support overlapping
    #bins, something that fastbin does support
    for k,e in enumerate(events):
        if util.isascending(e):
            m[:,k] = fastbin( e, bins )
        else:
            m[:,k] = fastbin( np.sort(e), bins )
    
    #transpose output, such that rows represent events and columns represent bins
    #m = m.T
    
    if kind=='count':
        pass
    elif kind=='binary':
        m[m>0]=1
        m = np.uint8(m)
    elif kind=='rate':
        m = m / np.diff( bins, axis=1 )

    #if bins had to be sorted initially, unsort them here
    if sortflag:
        m = m[ np.argsort( sort_idx ) ]
    
    return m

def event_bursts( events, intervals=None, nevents=None, marks=None ):
    """Detect bursts of events.
    
    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    intervals : 2-element sequence, optional
        minimum and maximum inter-event time intervals to consider two
        events as part of a burst
    nevents : 2-element sequence, optional
        minimum and maximum number of events in a burst
    marks : 1d array, optional
        vector of event marks (e.g. spike amplitude)
    
    Returns
    -------
    1d array
        vector with burst indicators: 0=non-burst event, 1=first event
        in burst, 2=event in burst, 3=last event in burst
    
    """
    
    events = check_events( events, copy=False )
    n = len(events)
    
    if n==0:
        return np.array([])
    
    if intervals is None: intervals = [0.003,0.015]
    if nevents is None: nevents = [2,np.Inf]
    
    max_interval = intervals[-1]
    min_interval = intervals[0]
    
    if marks is not None:
        if len(marks)!=n:
            raise Error
    
    dpre = np.abs( event_intervals( events, kind='pre' )[0] )
    dpost = np.abs( event_intervals( events, kind='post' )[0] )
    
    inburst = np.zeros(n)
    
    if marks is not None:
        apre = event_intervals( marks, kind='pre' )[0]
        apost = event_intervals( marks, kind='post' )[0]
    else:
        apre = 0
        apost = 0
    
    #find first event in burst
    mask = np.logical_or.reduce( (dpre>max_interval, dpre<min_interval, np.isnan(dpre) ) )
    mask = np.logical_or( mask, apre<0 )
    mask = np.logical_and.reduce( (mask, dpost<=max_interval, dpost>=min_interval ) )
    mask = np.logical_and( mask, apost<=0 )
    inburst[mask] = 1
    
    mask = np.logical_and.reduce( (dpre<=max_interval, dpre>=min_interval, dpost<=max_interval, dpost>=min_interval) )
    mask = np.logical_and( mask, np.logical_and(apre>=0, apost<=0) )
    inburst[mask] = 2
    
    mask = np.logical_or.reduce( (dpost>max_interval, dpost<min_interval, np.isnan(dpost)) )
    mask = np.logical_and.reduce( (mask, dpre<=max_interval, dpre>=min_interval ) )
    mask = np.logical_and( mask, apre>=0 )
    inburst[mask] = 3
    
    burst_start = np.flatnonzero( inburst==1 )
    burst_end = np.flatnonzero( inburst==3 )
    
    if len(burst_start)==0 or len(burst_end)==0:
        return inburst
    
    # find and remove incomplete bursts at start and end
    if burst_start[0] > burst_end[0]:
        inburst[ burst_end[0] ]=0
        burst_end = np.delete( burst_end, 0 )
        
    if burst_end[-1] < burst_start[-1]:
        inburst[ burst_start[-1] ]=0
        burst_start = np.delete( burst_start, -1 )
    
    # determine number of events in every burst
    ne = burst_end - burst_start + 1;
    invalidbursts = np.logical_or(ne<nevents[0] , ne>nevents[-1])
    
    #to get rid of invalid burst for now do a loop, until we've figured out a
    #better way of doing it
    for bs, be, b in zip( burst_start, burst_end, invalidbursts ):
        if b: inburst[bs:(be+1)] = 0
    
    return inburst

def filter_bursts( events, bursts=None, method='none', **kwargs):
    """Filter events based on participation in bursts.
    
    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    bursts : 1d array, optional
        burst indicator vector as returned by `event_bursts` function.
        If not provided, it will be computed internally (parameters to
        the event_bursts function can be provided as extra keyword arguments)
    method : {'none', 'reduce', 'remove', 'isolate', 'isolatereduce'}
        filter method to be applied. 'none': remove all events,
        'reduce': only keep non-burst events and first event in bursts, 
        'remove': remove all burst events, 'isolate': remove all non-burst
        events, 'isolatereduce': only keep first event in bursts.
    
    Returns
    -------
    1d array
        filtered vector of event times
    1d bool array
        selection filter
    
    """
    
    if bursts is None:
        bursts = event_bursts( events, **kwargs )
    
    if method == 'reduce':
        idx = bursts <= 1
    elif method == 'remove':
        idx = bursts == 0
    elif method == 'isolate':
        idx = bursts != 0
    elif method == 'isolatereduce':
        idx = bursts == 1
    else:
        idx = bursts>=0
    
    events = events[idx]
    
    return events,idx

def filter_bursts_length( bursts, nevents=None ):
    """Filter bursts on number of events.
    
    Parameters
    ----------
    bursts : 1d array
        burst indicator vector as returned by `event_bursts` function.
    nevents : scalar or 2-element sequence
        range of burst lengths that will be filtered out. If `nevents`
        is a scalar, the range is [nevents, Inf].
    
    Returns
    -------
    bursts : 1d array
        filtered burst indicator vector
    
    """
    
    if nevents is None:
        nevents = 2
        
    # find first and last events in bursts
    burst_start = np.flatnonzero(bursts==1)
    burst_end = np.flatnonzero(bursts==3)
    
    # determine burst lengths
    burstlen = burstend-burststart+1;
    
    # find burst to remove
    nevents = np.array(nevents).ravel()
    if len(nevents)>1:
        burstremove = np.flatnonzero( np.logical_or( burstlen<nevents[0], burstlen>nevents[-1] ) ) 
    else:
        burstremove = np.flatnonzero( burstlen>=nevents[0] )
    
    # loop to do actual removal
    for k in burstremove:
        bursts[ burst_start[ k ]:(burst_end[ k ]+1) ] = 0
    
    return bursts

def event_count( events, x=None ):
    """Cumulative event count.
    
    Parameters
    ----------
    events : 1d array
        vector of event times (in seconds)
    x : 1d array
        times at which to evaluate cumulative event count
    
    Returns
    -------
    count : 1d array
        event counts
    
    """
    
    events = check_events( events, copy=False )
    ne = len(e);
    
    #make sure x is 1D numpy vector
    x = np.array(x).ravel()
    nx = len(x)
    
    #combine event and x vectors, label 0/1 and sort
    tmp = np.concatenate( (e, x) )
    q = np.concatenate( (np.zeros(ne), np.ones(nx)) )
    q = q[ tmp.argsort(), ]
    qi = np.nonzero( q )[0]
    
    #compute cumulative count
    cs = np.cumsum( q )
    c = qi - cs[qi] + 1
    
    return c

def event_intervals( events, other=None, kind='post'):
    """Return inter-event intervals.
    
    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    other : 1d array, optional
        vector of sorted event times (in seconds)
    kind : {'pre', '<', 'post', '>', 'smallest', 'largest'}
        type of interval to return. 'pre' or '<': interval to previous event, 
        'post' or '>': interval to next event, 'smallest' or 'largest':
        smallest/largest of the intervals to the previous and next events.
    
    Returns
    -------
    intervals : 1d array
        the requested interval for each event in the input vector `events`.
        Intervals to events in the past have a negative sign.
    index : 1d array
        index of the event to which the interval was determined
    
    """
    
    events = check_events( events, copy=False )
    n = len(events)
    
    if n==0:
        return np.array([]), np.array([])
    
    if not kind in ['pre','<','post','>','smallest','largest']:
        raise ValueError
    
    if other is None: #auto intervals
        
        d = np.diff( events )
        ipre = np.concatenate( ([np.nan],-d) )
        idxpre = np.int64( np.concatenate( ([-1],np.arange(n-1)) ) )
        ipost = np.concatenate( (d,[np.nan]) )
        idxpost = np.int64( np.concatenate( (np.arange(n-1)+1,[-1]) ) )
        
    else: #cross intervals
        other = np.array( other ).ravel()
        n2 = len(other)
        
        if n2==0:
            return np.zeros(n)+np.nan, np.zeros(n)+np.nan
        
        idxpre = np.asarray( np.floor( np.interp(events,other,np.arange(n2),left=-1) ), dtype=np.int64 )
        valids = np.flatnonzero( idxpre>=0 )
        ipre = np.zeros(n)+np.nan
        
        if len(valids)>0:
            ipre[valids] = other[idxpre[valids]] - events[valids]
            ipre[valids[-1]+1:] = other[-1] - events[valids[-1]+1:]
            idxpre[valids[-1]+1:] = n2
        
        idxpost = np.asarray( np.ceil( np.interp(events,other,np.arange(n2),right=-1) ), dtype=np.int64 )
        valids = np.flatnonzero( idxpost>=0 )
        ipost = np.zeros(n)+np.nan
        
        if len(valids)>0:
            ipost[valids] = other[idxpost[valids]] - events[valids]
            ipost[0:valids[0]] = other[0] - events[0:valids[0]]
            idxpost[0:valids[0]] = 0
    
    if kind in ['pre','<']:
        return ipre, idxpre
    elif kind in ['post','>']:
        return ipost, idxpost
    elif kind == 'smallest':
        ii = ipre
        tmp = np.flatnonzero( np.abs(ipost)<=np.abs(ipre) )
        ii[tmp] = ipost[tmp]
        ii[ np.logical_or( np.isnan(ipre), np.isnan(ipost) ) ] = np.nan
        
        idx = idxpre
        idx[tmp] = idxpost[tmp]
        idx[ np.logical_or( np.isnan(ipre), np.isnan(ipost) ) ] = -1
        
        return ii,idx
    elif kind == 'largest':
        ii = ipre
        tmp = np.flatnonzero( np.abs(ipost)>np.abs(ipre) )
        ii[tmp] = ipost[tmp]
        ii[ np.logical_or( np.isnan(ipre), np.isnan(ipost) ) ] = np.nan
        
        idx = idxpre
        idx[tmp] = idxpost[tmp]
        idx[ np.logical_or( np.isnan(ipre), np.isnan(ipost) ) ] = -1
        return ii,idx
    else:
        raise Error

def filter_intervals( events, mininterval=0.003):
    """Filter out events based on interval to previous event.
    
    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    mininterval : scalar, optional
    
    Returns
    -------
    events : 1d array
        filtered vector of sorted event times (in seconds)
    index : 1d array
        index into original vector of all removed events
    
    """
    
    events = check_events( events, copy=True )
    d = np.diff( events )
    idx = np.flatnonzero( d<mininterval ) + 1
    events = np.delete( events, idx )
    return events, idx

def complex_spike_index(spike_times, spike_amp=None, intervals=None):
    """Compute complex spike index.
    
    Parameters
    ----------
    spike_times : 1d array or sequence of 1d arrays
        vector(s) of sorted event times (in seconds)
    spike_amp : 1d array or sequence of 1d arrays
        vector(s) of spike amplitudes
    intervals : 2-element sequence
        minimum and maximum inter-spike time intervals to consider two
        spikes as part of a burst
    
    Returns
    -------
    2d array
        array of complex spike indices between all pairs of spike time
        vectors
    
    """
    
    if intervals is None: intervals = [0.003,0.015]
    
    spike_times = check_events_list( spike_times, copy=False )    
    nvectors = len(spike_times)
    
    if spike_amp is not None:
        spike_amp = util.check_vector_list( spike_amp, copy=False )
        if len(spike_amp)!=len(spike_times):
            raise ValueError
        if not all([len(x)==len(y) for x,y in zip(spike_times,spike_amp)]):
            raise ValueError
    
    min_interval = intervals[0]
    max_interval = intervals[-1]
    
    c = np.zeros( (nvectors, nvectors) )
    
    for k in xrange(nvectors):
        for j in xrange(nvectors):
            #find smallest ISI and corresponding delta amplitude for each spike
            if k==j:
                dt, idx = event_intervals( spike_times[k], kind='smallest' )
            else:
                dt, idx = event_intervals( spike_times[k], other=spike_times[j], kind='smallest' )
                
            ii = idx>=0
            dt = -dt[ii]
            if spike_amp is None:
                da = 0
            else:
                da = spike_amp[k][ii] - spike_amp[j][idx[ii]]
            
            c[k,j] = _calc_csi( dt, da, max_interval, min_interval ) // len(spike_times[k])
    
    return c
    
def _calc_csi( dt, da, max_int, min_int ):
    
    #find all valid intervals (i.e. interval smaller than or equal to max_int)
    valid = np.abs(dt) <= max_int
    
    #find intervals within refractory period
    refract = np.abs(dt) < min_int
    
    #find intervals for all quadrants
    q1 = np.logical_and(da<=0 , dt>0) # post intervals with smaller amplitude
    q2 = np.logical_and(da>0  , dt<0) # pre intervals with larger amplitude
    q3 = np.logical_and(da<=0 , dt<0) # pre intervals with smaller amplitude
    q4 = np.logical_and(da>0  , dt>0) # post intervals with larger amplitude
    
    #count the number of intervals that contribute positively to CSI
    #i.e. preceding intervals with da>0 and following intervals with da<0
    #(complex burst) which are both valid and not in the refractory priod
    pos = np.sum( np.logical_and.reduce( (np.logical_or(q1,q2), valid, ~refract) ) )
    
    #count the number of intervals that contribute negatively to CSI
    neg = np.sum( np.logical_and( np.logical_or.reduce( (q3, q4,refract) ), valid) )
    
    #calculate csi
    c = 100 * (pos - neg);
    
    return c


@numba.jit( 'f8(f8[:],f8)', nopython=True, nogil=True )
def _bsearchi( vector, key ):
    
    nmemb = len(vector)
    
    left = 0
    right = nmemb-1
    
    while left<=right:
        mid = int( math.floor( (left+right)/2 ) )
        
        if vector[mid] == key:
            return mid
        
        if vector[mid]>key:
            right = mid-1
        else:
            left = mid+1
    
    if (left>(nmemb-1)) or (right<0):
        return -1
    else:
        return right + (key-vector[right]) / (vector[left]-vector[right])

@numba.jit( 'uint64[:](f8[:],f8[:,:])', nopython=True, nogil=True )
def fastbin( events, bins ):
    """Count number of events in bins.
    
    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    bins : (n,2) array
        array of time bin start and end times (in seconds). The bins
        need to be sorted by start time.
    
    Returns
    -------
    counts : 1d array
        event counts for each bin
    
    """
    nevents = events.shape[0]
    nbins = bins.shape[0]
    
    counts = np.zeros( (nbins,), dtype=np.uint64 )
    
    next_idx = 0;
    
    for k in range(nbins):
        while next_idx<nevents and events[next_idx]<bins[k,0]:
            next_idx+=1
        
        idx = next_idx
        
        while idx<nevents and events[idx]<bins[k,1]:
            counts[k] += 1
            idx+=1
        
    return counts

@numba.jit('u8(f8[:],f8[:],f8,f8,f8[:,:],b1,i8[:,:])', nopython=True, nogil=True)
def _find_events_near_reference( ref, ev, minlag, maxlag, segs, unbiased, out ):
    
    nref = len(ref) #number of reference events
    nev = len(ev) #number of events
    nseg = len(segs) #number of segments
    
    event_i = 0 #index of first event to be processed next
    
    n = 0 #number of valid reference events (inside segments)
    
    #loop through all segments
    for k in range(nseg):
        
        #find index i1 of first reference event within segment
        if segs[k,0]<ref[0]: 
            i1 = 0
        else:
            tmp = segs[k,0]
            
            if unbiased:
                #adjust segment boundary
                tmp = tmp - minlag
                
            if tmp>ref[-1]:
                continue
            
            i1 = int( math.ceil( _bsearchi( ref, tmp ) ) )
        
        #find index i2 of last reference event within segment
        if segs[k,1]>ref[nref-1]:
            i2 = nref-1
        else:
            tmp = segs[k,1]
            
            if unbiased:
                #adjust segment boundary
                tmp = tmp-maxlag
            
            if tmp<ref[0]:
                continue
            
            i2 = int( math.floor( _bsearchi( ref, tmp ) ) )
        
        #print(k, i1,i2)
        
        if i1>i2 or i1<0 or i2<0:
            continue
        
        n += i2-i1+1
        
        #print(k,n)
        
        #loop through all reference events in segment
        for l in range(i1, i2+1):
            
            i = event_i
            event_i_set = 0
            while i<nev and ev[i]<=ref[l]+maxlag and ev[i]<=segs[k,1]:
                if ev[i]>=ref[l]+minlag and ev[i]>=segs[k,0]:
                    if event_i_set==0:
                        out[l,0] = i
                        event_i_set = 1
                        event_i = i
                i+=1
                
            if event_i_set:
                out[l,1] = i-out[l,0]
            
            #this code is slower
            #out[l,0] = math.ceil( bsearchi(ev,ref[l]+minlag) )
            #out[l,1] = math.floor( bsearchi(ev,ref[l]+maxlag) )


    return n

@numba.jit('f8[:](f8[:],f8[:],i8[:,:],f8[:])',nopython=True, nogil=True)
def _align_events( events, reference, idx, x ):
    n = len(reference)
    
    startidx = 0
    for k in range( n ):
        if idx[k,1]>0:
            for j in range( idx[k,1] ):
                x[startidx + j] = events[idx[k,0] + j] - reference[k]
            startidx = startidx + idx[k,1]
    
    return x
 
def _event_correlation( events, reference=None, lags=None, segments=None, unbiased=False ):
    
    events = check_events( events, copy=False )
    
    if reference is None:
        reference = events
    else:
        reference = check_events(reference, copy=False)
    
    if lags is None:
        minlag = -1.0
        maxlag = 1.0
    else:
        minlag = float(lags[0])
        maxlag = float(lags[-1])
    
    if minlag>maxlag:
        raise ValueError('minimum lag should be smaller than maximum lag')
    
    if segments is None:
        segments = np.array([[-1,1]],dtype=np.float64)*np.inf
    else:
        segments = check_segments(segments)
    
    #remove overlap between segments
    segments = segment_remove_overlap( segments, strict=False )
    
    idx = np.zeros( (len(reference),2), dtype=np.int64 )
    
    nvalidref = _find_events_near_reference( reference, events, minlag, maxlag, segments, unbiased, idx )
    
    n = np.sum( idx[:,1] ) #total number of events found near references
    
    x = np.zeros( (n,), dtype=np.float64 )
    
    x = _align_events( events, reference, idx, x )
    
    return x, nvalidref
   
def peri_event_histogram( events, reference=None, lags=None, segments=None, normalization='none', unbiased=False, remove_zero_lag=False ):
    """Compute peri-event time histogram.
    
    Parameters
    ----------
    events : 1d array or sequence of 1d arrays
        vector(s) of sorted event times (in seconds)
    reference : 1d array or sequence of 1d arrays, optional
        vector(s) of sorted reference event times (in seconds).
        If not provided, then `events` are used as a reference.
    lags : 1d array, optional
        vector of sorted lag times that specify the histogram time bins
    segments : (n,2) array or Segment, optional
        array of time segment start and end times
    normalization : {'none', 'coef', 'rate', 'conditional mean intensity',
                     'product density', 'cross covariance',
                     'cumulant density', 'zscore'}, optional
        type of normalization
    unbiased : bool, optional
        only include reference events for which data is available at all lags 
    remove_zero_lag : bool, optional
        remove zero lag event counts
    
    Returns
    -------
    3d array
        peri-event histogram of shape (lags, events, references)
    
    """
    
    events = check_events_list( events, copy=False )
    
    if reference is None:
        reference = events
    else:
        reference = check_events_list( reference, copy=False )
    
    if lags is None:
        lags = np.linspace(-1,1,51)
    
    if segments is None:
        segments = np.array( [ [x[0],x[-1]] for x in events+reference ] )
        segments = np.array( [ [np.min(segments[:,0]), np.max(segments[:,1])] ] )
    
    segments = check_segments( segments, copy=False )
    
    duration = np.sum( np.diff( segments, axis=1 ) )
    
    nev = len(events)
    nref = len(reference)
    nlags = len(lags) - 1
    minlag = lags[0]
    maxlag = lags[-1]
    
    p = np.zeros( (nlags, nev, nref), dtype=np.float64 )
    nvalid = np.zeros( (nref,) )
    
    for t in xrange(nref):
        for k in xrange(nev):
            tmp, nvalid[t] = _event_correlation( events[k], reference[t], lags=[minlag,maxlag], segments=segments, unbiased=unbiased )
            if remove_zero_lag:
                p[:,k,t] = np.histogram( tmp[tmp!=0], bins=lags )[0]
            else:
                p[:,k,t] = np.histogram( tmp, bins=lags )[0]
    
    if unbiased and normalization not in ['coef','rate','conditional mean intensity']:
        tmp = np.array( [np.sum(segment_contains(segments,x)[0]) for x in reference] )
        p = p * ( tmp[None,None,:] / nvalid[None,None,:] )
            
    if normalization in ['coef']:
        p = p / nvalid[None,None,:]
    elif normalization in ['rate','conditional mean intensity']:
        p = p / (nvalid[None,None,:] * np.diff( lags )[:,None,None])
    elif normalization in ['product density']:
        p = p / (np.diff(lags)[:,None,None]*duration)
    elif normalization in ['cross covariance','cumulant density']:
        refrate = np.array( [np.sum(segment_contains(segments,x)[0])/duration for x in reference] )
        evrate = np.array( [np.sum(segment_contains(segments,x)[0])/duration for x in events] )
        p = p / (np.diff(lags)[:,None,None]*duration) - evrate[None,:,None] * refrate[None,None,:]
    elif normalization in ['zscore']:
        refrate = np.array( [np.sum(segment_contains(segments,x)[0])/duration for x in reference] )
        evrate = np.array( [np.sum(segment_contains(segments,x)[0])/duration for x in events] )
        p1p0 = evrate[None,:,None] * refrate[None,None,:]
        p = p / (np.diff(lags)[:,None,None]*duration) - p1p0
        p = p / ( p1p0/ (np.diff(lags)[:,None,None]*duration) )
    
    bins = np.vstack( [ lags[0:-1], lags[1:] ] ).T
    
    return p, bins

#%b : binsize
#%T : total time

#%mean rate
#%P0 = N0/T 
#%P1 = N1/T

#%cross-correlation histogram
#%J10 = eventcorr( spike 0, spike 1 )

#%cross product density
#%P10 = J/bT

#%asymptotic distribution (independent)
#% sqrt(P10) -> sqrt(P1P0) +- norminv(1-alpha/2)/sqrt(4bT) 

#%conditional mean intensity
#%m10 = J/bN0

#%asymptotic distribution (independent)
#% sqrt(m10) -> sqrt(P1) +- norminv(1-alpha/2)/sqrt(4bN0)

#%cross-covariance / cumulant density
#%q10 = P10 - P1P0

#%asymptotic distribution (independent)
#% q10 -> 0 +- norminv(1-alpha/2)*sqrt(P1P0/Tb)

#% variance normalized q10 is approx normal only when normal approx to Poisson
#% distribution applies, i.e. when lamda > 20 (see Lubenov&Siapas,2005)
#% this gives condition bTP0P1>20

@deprecated("Please use fklab.signals.event_triggered_average instead.")
def event_average( events, t, data, lags=None, fs=None, interpolation='linear', method='fast', function=None ):
    """Event triggered average.
    
    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    t : 1d array
        vector of sample times for data array
    data : ndarray
        array of data samples. First dimension should be time.
    lags : 2-element sequence, optional
        minimum and maximum lags over which to compute average
    fs : float, optional
        sampling frequency of average. If not provided, will be calculated
        from time vector `t`
    interpolation : string or integer, optional
        kind of interpolation. See `scipy.interpolate.interp1d` for more
        information.
    method : {'fast'}, optional
        method for calculating event triggered average. Currently, only
        the 'fast' method is implemented.
    function : callable, optional
        function to apply to data samples (e.g. to compute something else
        than the average)
    
    Returns
    -------
    ndarray
        event triggered average of data
    
    """
    
    events = check_events(events,copy=False)
    
    if lags is None:
        lags = [-1,1]
    
    if fs is None:
        fs = 1/np.mean( np.diff( t ) )
    
    if function is None:
        function = np.nanmean
    
    lags = np.arange( lags[0], lags[-1], 1/fs )
    
    if method=='fast':
        b = sp.interpolate.interp1d( t, data, kind=interpolation, bounds_error=False, fill_value=np.nan, axis=0 )( events[:,None] + lags[None,:] )
        a = function( b, axis=0 )
    else:
        raise NotImplementedError
    
    return a

