"""
====================================
Ripple (:mod:`fklab.signals.ripple`)
====================================

.. currentmodule:: fklab.signals.ripple

Utilities for detection and analysis of hippocampal ripples

.. autosummary::
    :toctree: generated/
    
    ripple_envelope
    detect_ripples
    compute_threshold_zscore
    compute_threshold_median
    compute_threshold_percentile
    evaluate_online_ripple_detection

"""

import numpy as np
import scipy.signal
import fklab.signals
from fklab.signals.filter import compute_envelope
import fklab.segments
import fklab.events

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

__all__ = ['ripple_envelope', 'detect_ripples', 'compute_threshold_zscore', 
        'compute_threshold_median', 'compute_threshold_percentile', 
        'evaluate_online_ripple_detection']

def ripple_envelope( signals, **kwargs ):
    
    smooth_options = dict(kernel='gaussian', bandwidth=0.0075)
    smooth_options.update( kwargs.pop('smooth_options',{}) )
    
    filter_options = dict(transition_width="25%", attenuation=60)
    filter_options.update( kwargs.pop('filter_options',{}) )
    
    return compute_envelope( signals, 'ripple', filter_options=filter_options, smooth_options=smooth_options, **kwargs )

def detect_ripples( time, signals, axis=-1, isenvelope=False, isfiltered=False, segments=None, threshold=None, allowable_gap=0.02, minimum_duration=0.03, filter_options={}, smooth_options={} ):
    """Detect ripple events.
    
    Parameters
    ----------
    time : 1D array
    signal : array
        either an array of raw signals (`isenvelope`==False and
        `isfiltered`==False), or an array of already filtered signals
        (`isenvelope`==False, `isfiltered`==True) or a 1D array with
        pre-computed envelope (`isenvelope`==True)
    axis : scalar, optional
        axis of the time dimension in the signal array (not used if
        signal is already a pre-computed envelope)
    isenvelope : bool, optional
    isfiltered : bool, optional
    segments : segment-like
        restrict detection of ripples to segments
    threshold : scalar, 2-element sequence or callable, optional
        single upper threshold or [lower, upper] threshold for ripple
        detection. If a callable, the function should accept a 1D array
        and return a 1 or 2-element sequence of thresholds, computed from
        the input array
    allowable_gap : scalar, optional
        minimum gap between adjacent ripples. If the start of a ripple
        is within `allowable_gap` of a the end of previous ripple, then
        the two ripple events are merged.
    minimum_duration : scalar, optional
        minimum duration of a ripple event. Shorter duration events are
        dicarded.
    filter_options : dict, optional
        dictionary with options for filtering (if signal is not already filtered).
        See `apply_filter` and `construct_filter`.
    smooth_options : dict, optional
        dictionary with options for envelope smoothing (if envelope was
        not pre-computed). See `ripple_envelope`.
    
    Returns
    -------
    (peak_time, peak_amp) : 1D arrays
        time and amplitude of ripple peaks above upper threshold
    segments : Segment
        ripple start and end times
    (low,high) : scalars
        lower and upper threshold used for ripple detection
    
    """
    
    dt = np.median(np.diff(time))
    
    if not isenvelope:
        envelope = ripple_envelope( signals, axis=axis, fs=1./dt, isfiltered=isfiltered, filter_options=filter_options, smooth_options=smooth_options )
    else:
        envelope = signals
        if envelope.ndim!=1:
            raise ValueError('Envelope needs to be a vector.')
    
    # compute signal statistics in segments
    # and determine thresholds
    if segments is None:
        segments = [-np.inf, np.inf]
    
    segments = fklab.segments.check_segments( segments )
    
    b,_,_ = fklab.segments.segment_contains( segments, time )
    
    if threshold is None:
        threshold = [ np.mean( envelope[b] ), ]
    elif callable(threshold):
        threshold = threshold( envelope[b] )
    else:
        threshold = np.array( threshold, copy=False, dtype=np.float64 ).ravel()
    
    low = threshold[0]
    high = threshold[-1]
    
    # find ripple peaks
    ripple_peak_time, ripple_peak_amp = fklab.signals.localmaxima( envelope, x=time, method='gradient', yrange=[high, np.inf] )
    
    # find bumps
    ripple_segments = fklab.signals.detect_mountains( envelope, x=time, low=low, high=high, segments=segments)
    
    # join nearby bumps
    ripple_segments.ijoin( gap = allowable_gap )

    # eliminate short duration bumps
    del ripple_segments[ripple_segments.duration<minimum_duration]
    
    #only retain ripple peaks within segments
    selection = ripple_segments.contains( ripple_peak_time )[0]
    ripple_peak_time = ripple_peak_time[selection]
    ripple_peak_amp = ripple_peak_amp[selection]
    
    return (ripple_peak_time, ripple_peak_amp), ripple_segments, (low,high)


def compute_threshold_zscore( multipliers = 1 ):
    """Creates callable for computing thresholds based on zscore.
    
    Parameters
    ----------
    multipliers : scalar or sequence
    
    Returns
    -------
    callable object that takes a 1D array and returns
    mean(signal) + multipliers * standard_deviation(signal)
    
    """
    
    multipliers = np.array( multipliers, copy=True, dtype=np.float64 ).ravel()
    
    def inner( signal, multipliers=multipliers ):
        mu = np.mean( signal )
        std = np.std( signal )
        return mu + multipliers * std
    
    return inner

def compute_threshold_median( multipliers = 1 ):
    """Creates callable for computing thresholds based on median and
    upper quartile range.
    
    Parameters
    ----------
    multipliers : scalar or sequence
    
    Returns
    -------
    callable object that takes a 1D array and returns
    median(signal) + multipliers * ( percentile(signal,75) - median(signal) )
    
    """
    
    multipliers = np.array( multipliers, copy=True, dtype=np.float64 ).ravel()
    
    def inner( signal, multipliers=multipliers ):
        mu = np.median( signal )
        qr = np.percentile( signal, 75 ) - mu
        return mu + multipliers * qr
    
    return inner

def compute_threshold_percentile( percentiles = [50,90] ):
    """Creates callable for computing thresholds based on percentiles.
    
    Parameters
    ----------
    percentiles : scalar or sequence
    
    Returns
    -------
    callable object that takes an 1D array and returns the requested
    percentiles.
    
    """
    
    percentiles = np.array( percentiles, copy=True, dtype=np.float64 ). ravel()
    
    def inner( signal, percentiles=percentiles ):
        return np.percentile( signal, percentiles )
    
    return inner


def evaluate_online_ripple_detection( online_ripple, offline_ripple, window=0.25, lock_out=None ):
    """Evaluate online ripple detection accuracy.
    
    Parameters
    ----------
    online_ripple : 1d array
        Times of online detected ripples
    offline_ripple : Segment
        Time segments for offline detected ripples
    window : scalar
        Analysis window.
    lock_out : None or scalar
        Lock-out period after online detected ripple
        
    """
    
    n_online = len(online_ripple)
    n_offline = len(offline_ripple)
    
    lags = np.linspace( 0, window, np.ceil(window/0.005) )
    peth_offline,lags = fklab.events.peri_event_histogram( online_ripple, reference=offline_ripple.start, lags=lags)
    
    interval_to_offline,idx = fklab.events.event_intervals( online_ripple, offline_ripple.start, 'pre' )
    
    true_positive = np.sum(-interval_to_offline < offline_ripple.duration[idx])
    true_positive_rate = 100. * true_positive / n_offline
    
    false_negative = n_offline - true_positive
    false_negative_rate = 100. * false_negative / n_offline
    
    false_positive = n_online - true_positive
    false_positive_rate = 100. * false_positive / n_online
    
    if lock_out is not None:
        interval_to_online,_ = fklab.events.event_intervals( offline_ripple.start, online_ripple, 'pre' )
        n_lock_out = np.sum( -interval_to_online < lock_out )
        lock_out_rate = 100. * n_lock_out / n_offline
    
    # plotting
    fig = plt.figure()
    host = mpl_toolkits.axes_grid1.host_subplot(111)
    
    # plot distribution of online detction latencies
    bar = host.bar( left=1000*lags[:,0], bottom=0, width=1000*np.diff(lags,axis=1), height=peth_offline, label='delay of online detected events' )
    l = host.set_ylabel('number of online events')
    l.set_color( bar.patches[0].get_facecolor() )
    host.set_xlabel('time [ms] from offline ripple start')
    
    # plot offline ripple event durations
    par = host.twinx()
    xx = np.linspace(0,window,100)
    yy = fklab.segments.segment_count( offline_ripple - offline_ripple.start, xx)
    par.plot( 1000*xx, yy, color='black', label='# offline ripple events of certain duration' )
    par.grid(False)
    par.set_ylabel('number of offline events')
    
    # display statistics
    txt = "{tprate:.1f}% ({tp}/{n}) of ripple events detected online".format(tprate=true_positive_rate, tp=true_positive, n=n_offline)
    txt += "\n{fnrate:.1f}% ({fn}/{n}) ripple events missed".format(fnrate=false_negative_rate, fn=false_negative, n=n_offline)
    
    if lock_out is not None:
        txt += "\n{r:.1f}% ({nlock}/{n}) in lock out period ({p:.1f} ms)".format(r=lock_out_rate, nlock=n_lock_out, n=n_offline, p=1000*lock_out)
    
    txt += "\n{fprate:.1f}% ({fp}/{n}) spurious events detected online".format(fprate=false_positive_rate, fp=false_positive, n=n_online)
    
    txtbox = plt.text( 0.95, 0.95, txt, transform=host.transAxes, va='top', ha='right', linespacing=2)
    
    host.legend(loc='right')
    
    host.set_xlim(0,window*1000)

