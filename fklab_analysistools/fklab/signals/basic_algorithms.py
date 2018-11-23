"""
========================================================
Basic algorithms (:mod:`fklab.signals.basic_algorithms`)
========================================================

.. currentmodule:: fklab.signals.basic_algorithms

Basic algorithms for detecting extremes, zero crossing, etc. in data.

.. autosummary::
    :toctree: generated/
    
    localextrema
    localminima
    localmaxima
    zerocrossing
    detect_mountains
    remove_artefacts
    
    generate_windows
    extract_data_windows
    generate_trigger_windows
    extract_trigger_windows
    event_triggered_average

"""

from __future__ import division

import numpy as np
import scipy as sp
import scipy.interpolate

from numba import jit, int_, float64

from fklab.utilities import inrange
from fklab.events.basic_algorithms import check_events
import fklab.segments as seg

_USE_NUMBA = True

__all__ = ['detect_mountains', 'localextrema', 'localmaxima', 'localminima', 
           'zerocrossing', 'remove_artefacts', 'generate_windows',
           'extract_data_windows', 'generate_trigger_windows', 'extract_trigger_windows',
           'event_triggered_average']

def detect_mountains(y,x=None,low=None,high=None,segments=None):
    """Detects segments with above-threshold values 1D data array.
    
    Parameters
    ----------
    y : array
        array of data values
    x : array
        array of index values. Default is None.
    low : scalar number
        lower data value threshold
    high : scalar number
        upper data value threshold
    segments : Segment (optional)
        pre-defined segments in which to search for above-threshold values
    
    Returns
    -------
    seg : Segment
        list of segments that meet the threshold conditions
    
    """
    
    #define indices, if not given
    if x is None:
        x = np.arange(len(y))
    
    if low is None:
        if high is None:
            #compute 90% percentile if no thresholds are given
            if segments is None:
                low = np.percentile(y,90)
            else:
                low = segments.applyfcn(x,y, function = lambda z: np.percentile(z,90))
        else:
            #only one threshold is given
            low = high
            high = None
    
    #find all segments in which y is above lower threshold
    s = seg.Segment.fromlogical(y>low,x)
    
    if len(s)>0:
        #for each segment, test if maximum y-value is above upper threshold
        if high is not None:
            valid = s.applyfcn(x,y,function=lambda z: np.max(z)>high,separate=True,default=False)
            del s[~np.array(valid)]
        
        #combine with user-provided segments
        if segments is not None:
            s = s & segments
    
    return s

def localextrema(y,x=None,method='discrete',kind='extrema',yrange=None,interp='linear'):
    """Detects local extrema (maxima and/or minima) in 1D data array.
    
    Parameters
    ----------
    y : array
        array of data values
    x : array
        array of index values. Default is None.
    method : str
        method for computing extrema, one of ``discrete`` or ``gradient``.
        Default is ``discrete``.
    kind : str
        type of extrema to compute, one of ``extrema``, ``extremes``, 
        ``max``, ``maximum``, ``maxima``, ``min``, ``minimum``, ``minima``
        Default is ``extrema``
    yrange : 2-element sequence (optional)
        range of acceptable y-values for the absolute extrema. Default is None.
    interp : callable or str
        if callable, it should define an interpolation function that takes
        a 1D array of values and an array of indices at which to interpolate.
        If a string, it specifies the kind of interpolation for the
        scipy.interpolation.interp1d function. Default is ``linear``.
    
    Returns
    -------
    index : array
        if x is None, an array of (possibly fractional) indices at which
        the extrema where detected. If x is given, then the interpolated
        values in x are returned.
    value: array
        interpolated y-values at the detected extrema.
    
    """
    
    #compute the local extrema
    if method in ['gradient']:
        imax,imin = _localextrema_gradient(y)
    elif method in ['discrete']:
        imax,imin = _localextrema_discrete(y)
    else:
        raise Error
    
    #select the requested extrema
    if kind in ('extrema','extremes'):
        ii = np.sort( np.concatenate( (imax,imin) ) )
    elif kind in ('max','maximum','maxima'):
        ii = imax
    elif kind in ('min','minimum','minima'):
        ii = imin
    else:
        raise Error
    
    #compute signal amplitude at local extrema
    try:
        amp = interp( y, ii )
    except TypeError:
        amp = sp.interpolate.interp1d( np.arange(len(y)), y, kind=interp, copy=False)(ii) 
    
    #apply threshold on absolute amplitude
    if not yrange is None:
        valid = inrange(np.abs(amp),low=yrange[0],high=yrange[-1])
        ii = ii[valid]
        amp = amp[valid]
    
    #interpolate x values
    if x is not None:
        ii = sp.interp(ii, np.arange(len(y)), x)
    
    return ii, amp

def localmaxima(y,**kwargs):
    """Detects local maxima in 1D data array.
    
    See also
    --------
    localextrema
    
    """
    
    kwargs['kind']='max'
    return localextrema(y,**kwargs)

def localminima(y,**kwargs):
    """Detects local minima in 1D data array.
    
    See also
    --------
    localextrema
    
    """
    
    kwargs['kind']='min'
    return localextrema(y,**kwargs)

def zerocrossing(y, x=None):
    """Detects zero crossings in 1D data array.
    
    Parameters
    ----------
    y : array
        array of data values
    x : array
        array of index values. Default is None.
    
    Returns
    -------
    p2n : array
        index at positive-to-negative zero crossings
    n2p: array
        index at negative-to-positive zero crossings
    
    """
    p2n, n2p = _zerocrossing(y)
    
    if x is not None:
        p2n = sp.interp(p2n, np.arange(len(y)), x)
        n2p = sp.interp(n2p, np.arange(len(y)), x)
    
    return p2n,n2p

def remove_artefacts( signal, artefacts, time=None, axis=-1, window=1, interp='linear', fill_value=np.nan ):
    """Remove and interpolate signal around times of artefacts.
    
    Parameters
    ----------
    signal : array
    artefacts : 1D array-like
        time of the artefacts
    time : 1D array-like, optional
        time vector - should have the same size as signal.shape[axis]
    axis : scalar, optional
        axis of the time dimension in the signal array
    window : scalar or 2-element sequence, optional
        time window around the artefacts that should be removed.
        If `window` is a scalar, the window is symmetric around the
        artefact.
    interp : string, optional
        the kind interpolation to perform - valid values are 'fill'
        (set to `fill_value`), 'reflect' (weighted average of surrounding
        signal) or any valid interpolation kind accepted by
        `scipy.interpolate.interp1d`.
    fill_value : scalar, optional
        value to use for 'fill' interpolation and for extrapolated values.
    
    Returns
    -------
    signal : 1D array
    
    """
    
    # check arguments
    signal = np.array( signal, copy=True )
    artefacts = np.asarray( artefacts ).ravel()
    
    if time is None:
        time = np.arange( signal.shape[axis] )
    else:
        time = np.asarray( time ).ravel()
    
    if signal.shape[axis]!=len(time):
        raise ValueError('Signal and time vectors do not have the same size.')
    
    window = np.asarray( window ).ravel()
    if len(window) == 1:
        window = [-1,1] * window
            
    # from artefacts and window, create segments
    invalid_regions = seg.Segment( artefacts[:,None] + window[None,:] )
    
    # from segments and time, find all signal samples near artefact
    b,n,idx = invalid_regions.contains( time )
    
    k = [slice(None),]*signal.ndim
    
    # perform interpolation
    if interp=='fill':
        # set samples near artefact to interp
        k[axis] = b
        signal[k] = fill_value
    elif interp=='reflect':
        # perform weighted reflection interpolation
        # loop through invalid regions
        r = [slice(None),] * signal.ndim
        r[axis] = slice(None,None,-1)
        
        for region, regionsize in zip(idx,n):
            
            if region[0]<regionsize or region[1]>=len(signal)-regionsize:
                # do not interpolate anything too close to the ends
                k[axis] = slice(region[0], region[1]+1)
                signal[k] = fill_value
            else:
                # construct reflected signals and weights
                pre_reflect = 2 * np.take(signal,region[0],axis=axis) - np.take(signal, np.arange(region[0]-regionsize, region[0]), axis=axis)
                post_reflect = 2 * np.take(signal,region[1],axis=axis) - np.take(signal, np.arange(region[1]+1,region[1]+regionsize+1), axis=axis)
                
                weights = np.arange(1.,regionsize+1)/(regionsize+1)
                sz = np.ones( signal.ndim )
                sz[axis] = len(weights)
                weights = weights.reshape( sz )
                
                # compute weighted average and assign to signal
                k[axis] = slice(region[0],region[1]+1)
                signal[k] = pre_reflect[r] * (1-weights) + post_reflect[r] * weights
    else:
        # use scipy.interpolate.interp1d
        k[axis] = ~b
        interpolator = scipy.interpolate.interp1d( time[~b], signal[k], axis=axis, kind=interp, bounds_error=False, fill_value=fill_value, assume_sorted=True )
        k[axis]=b
        signal[k] = interpolator( time[b] )
    
    return signal


def _zerocrossing_python(y):
    #find all positive and negative values
    sy=np.sign(y)
    isy = np.flatnonzero(sy)
    sy = sy[isy]
    
    #compute difference: negative values indicate positive-to-negative
    #transitions and positive values indicate negative-to-positive
    #transitions
    dsy = np.diff(sy)
    
    p2n=np.flatnonzero(dsy<0)
    n2p=np.flatnonzero(dsy>0)
    
    #look up the start and end indices that span the transitions
    p2n_x1 = isy[p2n]+1
    p2n_x2 = isy[p2n+1]-1
    
    n2p_x1 = isy[n2p]+1
    n2p_x2 = isy[n2p+1]-1
    
    #compute the (fractional) index of the transition
    p2n = (p2n_x1+p2n_x2)/2.0
    n2p = (n2p_x1+n2p_x2)/2.0
    
    #todo: correct index
    mask = p2n_x2 < p2n_x1
    a = np.abs(y[p2n_x2[mask]]/y[p2n_x1[mask]])
    p2n[mask] = p2n[mask] + 0.5*(p2n_x1[mask]-p2n_x2[mask])*(a-1)/(a+1)
    
    mask = n2p_x2 < n2p_x1
    a = np.abs(y[n2p_x2[mask]]/y[n2p_x1[mask]])
    n2p[mask] = n2p[mask] + 0.5*(n2p_x1[mask]-n2p_x2[mask])*(a-1)/(a+1)
    
    return p2n, n2p
    
def _localextrema_discrete_python(y):
    
    dy = np.sign( np.diff(y) )
    inz = np.flatnonzero(dy)    
    dy = dy[inz]
    ddy = np.diff(dy)
    imax = np.flatnonzero(ddy<0)
    imin = np.flatnonzero(ddy>0)
    
    imax = (1+inz[imax]+inz[imax+1])/2.0
    imin = (1+inz[imin]+inz[imin+1])/2.0
    
    return imax, imin

@jit(nopython=True, nogil=True, locals={'nzc_p2n':int_,'nzc_n2p':int_,'p2n_start':int_,'n2p_start':int_,'k':int_})
def _zerocrossing_numba(y):
    
    #pre-allocate array
    zc_p2n_tmp = np.zeros(y.shape,dtype=np.float64)
    zc_n2p_tmp = np.zeros(y.shape,dtype=np.float64)
    #track number of crossings
    nzc_p2n = 0
    nzc_n2p = 0
    #initialize indices for the start of +- and -+ transitions
    p2n_start = -1
    n2p_start = -1
    
    #loop through samples
    for k in range(y.shape[0]-1):
        
        #bail if samples have same sign
        if y[k]*y[k+1]>0 or (y[k]==y[k+1] and y[k]==0):
            continue
        
        #detect start of transitions
        if y[k]>0:
            p2n_start = k
        elif y[k]<0:
            n2p_start = k
        
        #detect end of transitions and compute location of zero crossing
        if y[k+1]<0 and p2n_start>=0:
            if k==p2n_start:
                zc_p2n_tmp[nzc_p2n] = k+1 - y[k+1]/(y[k+1]-y[k])
            else:
                zc_p2n_tmp[nzc_p2n] = 0.5*(p2n_start+k+1)
            nzc_p2n += 1
            p2n_start = -1
        elif y[k+1]>0 and n2p_start>=0:
            if k==n2p_start:
                zc_n2p_tmp[nzc_n2p] = k+1 - y[k+1]/(y[k+1]-y[k])
            else:
                zc_n2p_tmp[nzc_n2p] = 0.5*(n2p_start+k+1)
            nzc_n2p += 1
            n2p_start = -1
            
    #zc_p2n = zc_p2n[0:nzc_p2n]
    #zc_n2p = zc_n2p[0:nzc_n2p]
    
    zc_p2n = np.empty( (nzc_p2n,), dtype=np.float64 )
    zc_n2p = np.empty( (nzc_n2p,), dtype=np.float64 )
    
    for k in range(nzc_p2n):
        zc_p2n[k] = zc_p2n_tmp[k]
    
    for k in range(nzc_n2p):
        zc_n2p[k] = zc_n2p_tmp[k]
    
    return zc_p2n , zc_n2p

@jit(nopython=True, nogil=True, locals={'npeaks':int_,'ntroughs':int_,'peakstart':int_,'troughstart':int_,'k':int_,'dprev':float64,'d':float64})
def _localextrema_discrete_numba(y):
    #pre-allocate arrays to hold peak and trough indices
    peaks_tmp = np.zeros( int(y.shape[0]/2) ,dtype=np.float64)
    troughs_tmp = np.zeros( int(y.shape[0]/2) ,dtype=np.float64)
    #track how many peaks/troughs we find
    npeaks = 0
    ntroughs = 0
    #compute first delta
    dprev = y[1]-y[0]
    #track start index of peaks and troughs
    peakstart = -1
    troughstart = -1
    
    #loop through samples
    for k in range(1,y.shape[0]-1):
        
        #compute next delta
        d = y[k+1]-y[k]
        
        #bail if previous and next deltas have identical sign
        if d*dprev>0:
            dprev=d
            continue
        
        #detect peak and trough starts
        if dprev>0 and d<=0:
            peakstart = k
        elif dprev<0 and d>=0:
            troughstart = k
        
        #detect peak and trough ends, store (fractional) indices
        if d<0 and dprev>=0 and peakstart>0:
            peaks_tmp[npeaks] = 0.5*(k+peakstart)
            peakstart = 0
            npeaks += 1
        elif d>0 and dprev<=0 and troughstart>0:
            troughs_tmp[ntroughs] = 0.5*(k+troughstart)
            troughstart = 0
            ntroughs += 1
            
        #next delta will be the previous delta in the following iteration
        dprev=d
    
    #only keep the number of peaks/troughs we found
    #peaks = peaks[0:npeaks]
    #troughs = troughs[0:ntroughs]
    
    peaks = np.zeros((npeaks,),dtype=np.float64)
    troughs = np.zeros((ntroughs,),dtype=np.float64)
    
    for k in range(npeaks):
        peaks[k] = peaks_tmp[k]
    
    for k in range(ntroughs):
        troughs[k] = troughs_tmp[k]
    
    return peaks, troughs

def _localextrema_gradient(y):
    return zerocrossing( np.gradient(y) )

_zerocrossing = _zerocrossing_numba if _USE_NUMBA else _zerocrossing_python
_localextrema_discrete = _localextrema_discrete_numba if _USE_NUMBA else _localextrema_discrete_python

def generate_windows( n, window_size, window_overlap=0., fs=1., start_time=0., epochs=None, center=False ):
    """Construct windows from regular sampled signal.
    
    Parameters
    ----------
    n : int
        Number of samples in signal
    window_size : float
        Size of the windows in time units
    window_overlap : float, optional
        Overlap fraction between neighbouring windows
    fs : float, optional
        Sampling frequency
    start_time : float, optional
    epochs : (n,2) array-like, optional
    center : bool, optional
        Center windows across signal (will be ignored if `epochs` is not None)
    
    Returns
    -------
    nwin : int
        Number of windows
    time : (nwin,2) array
        Start and end time of windows
    gen : generator
        Window sample index generator
    
    """
    
    window_shift = np.round( (1.-window_overlap) * float(window_size) * fs )
    window_size = np.round( float(window_size) * fs )
    
    if not epochs is None:
        epochs = seg.check_segments( epochs )
        
        #convert epochs to samples
        epochs = np.round( ( epochs - start_time ) * fs )
        
        #remove epochs with fewer samples than the window size
        epochs = epochs[ (epochs[:,1]-epochs[:,0]) >= window_size+1, : ]
        
        #remove overlap and restrict to data epoch
        epochs = seg.segment_intersection( seg.segment_remove_overlap(epochs), [0, n-1] )
    
        #divide each epoch into windows
        startidx = [ np.arange(start, end-window_size+1, window_shift, dtype=np.int) for start, end in epochs ]
        startidx = np.concatenate( startidx )
    else:
        startidx = np.arange( 0, n-window_size+1, window_shift, dtype=np.int)
    
    #adjust winstart to center windows
    if center and epochs is None:
        startidx = startidx + int( (n - (startidx[-1] + window_size)) * 0.5 )
    
    numwin = len(startidx)
    
    window_time = startidx[:,None] + np.array([0,window_size-1],dtype=int)[None,:]
    window_time = start_time + window_time / fs
    
    def _index_generator():
        for k in xrange(numwin):
            yield np.arange( startidx[k], startidx[k] + window_size, dtype=np.int )
    
    return numwin, window_time, _index_generator

def extract_data_windows( data, *args, **kwargs ):
    """Extract windows from regular sampled signal.
    
    Parameters
    ----------
    data : 1d array
    window_size : float
        Size of the windows in time units
    window_overlap : float, optional
        Overlap fraction between neighbouring windows
    fs : float, optional
        Sampling frequency
    start_time : float, optional
    epochs : (n,2) array-like, optional
    center : bool, optional
        Center windows across signal (will be ignored if `epochs` is not None)
    
    Returns
    -------
    time : (nwin,2) array
        Start and end time of windows
    data : 2d array
        Windowed data
    
    """
    
    n = len(data)
    
    nidx, t, idx = generate_windows( n, *args, **kwargs )
    
    idx = np.vstack( list(idx()) ).T

    data = data[ idx.ravel() ].reshape( idx.shape )
    
    return t, data

def generate_trigger_windows( n, triggers, window, fs=1., start_time=0., epochs=None ):
    """Construct windows around triggers in regular sampled signal.
    
    Parameters
    ----------
    n : int
        Number of samples in signal
    triggers : 1d array
        Trigger times
    window : float
        Size of window in time units. Either a float for a
        [-`window`, `window`] symmetrical window around the triggers,
        or [left, right] for an asymmetrical window.
    fs : float, optional
        Sampling frequency
    start_time : float, optional
    epochs : (n,2) array-like, optional
    
    Returns
    -------
    nwin : int
        Number of windows
    time : (nwin,2) array
        Start and end time of windows
    gen : generator
        Window sample index generator
    
    """
    
    triggers = np.array( triggers, dtype=np.float ).ravel()
    
    if not epochs is None:
        epochs = seg.check_segments( epochs )
        triggers = triggers[ seg.segment_contains( epochs, triggers )[0] ]
    
    window = np.array( window, dtype=np.float ).ravel()
    if len(window)==1:
        window = np.abs(window) * [-1,1]
    elif len(window)!=2 or np.diff(window)<=0.:
        raise ValueError('Invalid window')
    
    #center triggers in window
    triggers = triggers + np.mean(window)
    
    #convert to sample indices
    triggers = np.round( (triggers - start_time) * fs )
    window_size = int(np.round( 0.5*np.diff(window) * fs ))
    
    #remove triggers too close to start or end of data
    triggers = triggers[ inrange( triggers, low=window_size, high=n-1-window_size ) ]
    triggers = triggers.astype(np.int)
    
    window_indices = np.arange(-window_size, window_size+1, dtype=np.int)
    ntriggers = len(triggers)
    
    window_time = triggers[:,None] + np.array([[-window_size,window_size]],dtype=int)
    window_time = start_time + window_time / fs
    
    def _index_generator():
        for k in triggers:
            yield k + window_indices
    
    return ntriggers, window_time, _index_generator

def extract_trigger_windows( data, *args, **kwargs ):
    """Extract windows around triggers in regular sampled signal.
    
    Parameters
    ----------
    data : 1d array
    triggers : 1d array
        Trigger times
    window : float
        Size of window in time units. Either a float for a
        [-`window`, `window`] symmetrical window around the triggers,
        or [left, right] for an asymmetrical window.
    fs : float, optional
        Sampling frequency
    start_time : float, optional
    epochs : (n,2) array-like, optional
    
    Returns
    -------
    time : (nwin,2) array
        Start and end time of windows
    data : 2d array
        Windowed data
    
    """
    
    n = len(data)
    
    nidx, t, idx = generate_trigger_windows( n, *args, **kwargs )
    
    idx = np.vstack( list(idx()) ).T

    data = data[ idx.ravel() ].reshape( idx.shape )
    
    return t, data


def event_triggered_average( events, t, data, lags=None, fs=None, interpolation='linear', method='fast', function=None ):
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
    function : callable, optional
        function to apply to data samples (e.g. to compute something else
        than the average)
    
    Returns
    -------
    ndarray
        event triggered average of data
    ndarray
        vector of lags
    
    """
    
    events = check_events(events,copy=False)
    
    if lags is None:
        lags = [-1,1]
    
    if fs is None:
        fs = 1/np.mean( np.diff( t ) )
    
    if function is None:
        function = np.nanmean
        
    if interpolation is None:
        _, b = extract_trigger_windows( data, triggers = events, window = (lags[0],lags[-1]), fs=fs, start_time = t[0] )
        b = b.T
        lags = ( np.arange( b.shape[1] ) - (b.shape[1]-1)/2. ) / fs + (lags[0]+lags[-1])/2.
    else:
        lags = np.arange( lags[0], lags[-1], 1/fs ) # should we change this linspace?
        #lags = np.linspace( lags[0], lags[-1], (lags[-1]-lags[0]) * fs + 1 )
        b = sp.interpolate.interp1d( t, data, kind=interpolation, bounds_error=False, fill_value=np.nan, axis=0, copy=False, assume_sorted=True )( events[:,None] + lags[None,:] )
        #b = np.interp( events[:,None] + lags[None,:], t, data, left=0, right=0 )
        
    a = function( b, axis=0 )

    return a, lags
