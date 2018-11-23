"""
====================================
Filter (:mod:`fklab.signals.filter`)
====================================

.. currentmodule:: fklab.signals.filter

Utilities for the design, application and inspection of digital filters.

Filter utilities
================

.. autosummary::
    :toctree: generated/
    
    construct_filter
    construct_low_pass_filter
    construct_high_pass_filter
    apply_filter
    apply_low_pass_filter
    apply_high_pass_filter
    inspect_filter
    plot_filter_amplitude
    plot_filter_phase
    plot_filter_group_delay
    compute_envelope
    compute_sliding_rms

"""

import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.transforms
from matplotlib import gridspec

import fklab.signals.kernelsmoothing
from fklab.codetools import deprecated

__all__ = ['standard_frequency_bands', 'construct_filter', 'construct_low_pass_filter',
            'construct_high_pass_filter', 'apply_filter', 'apply_low_pass_filter',
            'apply_high_pass_filter', 'inspect_filter', 'plot_filter_amplitude',
            'plot_filter_phase', 'plot_filter_group_delay',
            'compute_envelope', 'compute_sliding_rms']

standard_frequency_bands = { 'slow' : [0.1, 1.],
                             'delta' : [1., 4.],
                             'theta' : [6., 12.],
                             'spindle' : [7.,14.],
                             'beta' : [15.,30.],
                             'gamma' : [30.,140.],
                             'gamma_low' : [30.,50.],
                             'gamma_high' : [60.,140.],
                             'ripple' : [140., 225.],
                             'mua' : [300., 2000.] }


def construct_filter( band, fs=1., transition_width="25%", attenuation=60 ):
    """Constructs FIR high/low/band-pass filter.
    
    Parameters
    ----------
    band : str, scalar or 2-element sequence
        either a valid key into the `default_frequency_bands` dictionary,
        a scalar for a low-pass filter, or a 2-element sequence with lower
        and upper pass-band frequencies. Use 0., None, Inf or NaN for the
        lower/upper cut-offs in the sequence to define a low/high-pass filter.
        If band[1]<band[0], then a stop-band filter is constructed.
    fs : scalar, optional
        sampling frequency of the signal to be filtered
    transition_width : str or scalar, optional
        size of the transition between pass and stop bands. Can be either
        a scalar frequency or a string that represents a transition width
        relative to the size of the pass band (e.g. "25%", the percentage
        sign is required).
    attenuation : scalar, optional
        stop-band attenuation in dB.
        
    Returns
    -------
    1D array
        filter coefficients
    
    """
    
    # look up pre-defined frequency band
    if isinstance(band, basestring):
        band = standard_frequency_bands[band]
    
    band = np.array( band, dtype=np.float64 ).ravel()
    
    if len(band)==1:
        # scalar -> low=pass filter
        band = np.array( [0., float(band)], dtype=np.float64 )
    elif len(band)!=2:
        raise ValueError('Invalid frequency band')
    
    if np.diff(band)==0.:
        raise ValueError('Identical frequencies not allowed.')
    
    lower, upper = np.logical_or.reduce( ( np.isnan(band), np.isinf(band), band<=0.) )
    
    if not lower and upper:
        #high pass filter
        band = band[0]
        pass_zero = False
        band_width = fs/2. - band
    elif not upper and lower:
        # low pass filter
        band = band[1]
        pass_zero = True
        band_width = band
    elif lower and upper:
        raise ValueError('Invalid frequency band')
    else:
        pass_zero = np.diff(band)<0
        if pass_zero:
            band = band[::-1]
        band_width = np.diff(band)
    
    
    if fs<=2*np.max( band ):
        raise ValueError('Frequency band too high for given sampling frequency')
    
    if isinstance(transition_width, basestring):
        transition_width = band_width * float(transition_width.rstrip('%')) / 100.
    
    N, beta = scipy.signal.kaiserord( attenuation, transition_width*2.0/fs)
    
    # always have odd N
    N = N + (N+1)%2
    
    h = scipy.signal.firwin( N, band, window=('kaiser',beta), pass_zero=pass_zero, scale=False, nyq=fs/2. )
    
    return h

@deprecated("Please use construct_filter instead.")
def construct_band_filter( *args, **kwargs ):
    
    return construct_filter( *args, **kwargs )

construct_band_filter.__doc__ = construct_filter.__doc__

def construct_low_pass_filter( cutoff, **kwargs ):
    """Constructs FIR low-pass filter.
    
    Parameters
    ----------
    cutoff : scalar
        Cut-off frequency for low-pass filter.
    fs : scalar, optional
        sampling frequency of the signal to be filtered
    transition_width : str or scalar, optional
        size of the transition between pass and stop bands. Can be either
        a scalar frequency or a string that represents a transition width
        relative to the size of the pass band (e.g. "25%", the percentage
        sign is required).
    attenuation : scalar, optional
        stop-band attenuation in dB.
        
    Returns
    -------
    1D array
        filter coefficients
    
    """
    
    return construct_filter( [None, float(cutoff)], **kwargs )
    
def construct_high_pass_filter( cutoff, **kwargs ):
    """Constructs FIR high-pass filter.
    
    Parameters
    ----------
    cutoff : scalar
        Cut-off frequency for high-pass filter.
    fs : scalar, optional
        sampling frequency of the signal to be filtered
    transition_width : str or scalar, optional
        size of the transition between pass and stop bands. Can be either
        a scalar frequency or a string that represents a transition width
        relative to the size of the pass band (e.g. "25%", the percentage
        sign is required).
    attenuation : scalar, optional
        stop-band attenuation in dB.
        
    Returns
    -------
    1D array
        filter coefficients
    
    """
    
    return construct_filter( [float(cutoff), None], **kwargs )


def apply_filter( signal, band, axis=-1, **kwargs ):
    """Applies low/high/band-pass filter to signal.
    
    Parameters
    ----------
    signal : array
    band : str, scalar or 2-element sequence
        frequency band, either as a string, a scalar or [low,high] sequence.
        See `construct_filter` for more details.
    axis : scalar, optional
        axis along which to filter
    fs : scalar
        sampling frequency
    transition_width : str or scalar
        size of teransition between stop and pass bands
    attenuation: scalar
        stop-band attenuation in dB
    
    Returns
    -------
    array
        filtered signal
    
    """
    
    b = construct_filter( band, **kwargs )
    
    if isinstance( signal, (tuple, list) ):
        signal = [ scipy.signal.filtfilt( b, 1., np.asarray(x), axis=axis ) for x in signal ]
    else:
        signal = np.asarray(signal)
        signal = scipy.signal.filtfilt( b, 1., signal, axis=axis )
    
    return signal

@deprecated("Please use apply_filter instead.")
def apply_band_filter( *args, **kwargs ):
    
    return apply_filter( *args, **kwargs )

apply_band_filter.__doc__ = apply_filter.__doc__

def apply_low_pass_filter( signal, cutoff, **kwargs ):
    """Applies low-pass filter to signal.
    
    Parameters
    ----------
    signal : array
    band : scalar
        cut-off frequency for low-pass filter.
    axis : scalar, optional
        axis along which to filter.
    fs : scalar
        sampling frequency
    transition_width : str or scalar
        size of teransition between stop and pass bands
    attenuation: scalar
        stop-band attenuation in dB
    
    Returns
    -------
    array
        filtered signal
    
    """
    
    return apply_filter( signal, [None, float(cutoff)], **kwargs )

def apply_high_pass_filter( signal, cutoff, **kwargs ):
    """Applies high-pass filter to signal.
    
    Parameters
    ----------
    signal : array
    band : scalar
        cut-off frequency for high-pass filter.
    axis : scalar, optional
        axis along which to filter.
    fs : scalar
        sampling frequency
    transition_width : str or scalar
        size of teransition between stop and pass bands
    attenuation: scalar
        stop-band attenuation in dB
    
    Returns
    -------
    array
        filtered signal
    
    """
    
    return apply_filter( signal, [float(cutoff), None], **kwargs )


def inspect_filter( b, a=1.0, fs=1., npoints=None, filtfilt=False, detail=None, grid=False ):
    """Plot filter characteristics.
    
    Parameters
    ----------
    b : filter coefficients (numerator)
    a : filter coefficients (denominator), optional
    fs : scalar
        sampling frequency
    npoints : scalar
        number of points to plot
    filtfilt : bool
        if True, will plot the filter's amplitude response assuming
        forward/backward filtering scheme. If False, will plot the
        filter's amplitude and phase responses and the group delay.
    detail : 2-element sequence
        plot an additional zoomed in digital filter response with the
        given frequency bounds.
    grid : bool
        if True, all plots will have both axes grids turned on.
    
    """    
    
    # prepare plot
    fig = plt.figure()
    
    ncols = 1 if detail is None else 2
    nrows = 3 if not filtfilt else 1
    
    g = gridspec.GridSpec( nrows=nrows, ncols=ncols, bottom=0.1, top=0.9, left=0.1, right=0.9, wspace = 0.5, hspace =0.5, width_ratios=[2,1] )
    
    host = mpl_toolkits.axes_grid1.host_subplot(g[0,0])
    plot_filter_amplitude( b, a, fs, npoints=npoints, filtfilt=filtfilt, axes=host, grid=grid ) 
    
    if not filtfilt:
        ax_phase = mpl_toolkits.axes_grid1.host_subplot(g[1,0],sharex=host)
        plot_filter_phase(b, a, fs, npoints=npoints, axes=ax_phase, grid=grid )
    
        ax_delay = mpl_toolkits.axes_grid1.host_subplot(g[2,0],sharex=host)
        plot_filter_group_delay(b, a, fs, npoints=npoints, axes=ax_delay, grid=grid )        
    
    if detail is not None:
        host_detail = mpl_toolkits.axes_grid1.host_subplot(g[0,1],sharey=host)
        plot_filter_amplitude( b, a, fs, npoints=npoints, filtfilt=filtfilt, freq_lim=detail, axes=host_detail, grid=grid )
        
        if not filtfilt:
            ax = mpl_toolkits.axes_grid1.host_subplot(g[1,1],sharex=host_detail,sharey=ax_phase)
            plot_filter_phase(b, a, fs, npoints=npoints, freq_lim=detail, axes=ax, grid=grid )
            
            ax = mpl_toolkits.axes_grid1.host_subplot(g[2,1],sharex=host_detail,sharey=ax_delay)
            plot_filter_group_delay(b, a, fs, npoints=npoints, freq_lim=detail, axes=ax, grid=grid )

def plot_filter_amplitude(b, a=1., fs=1., npoints=None, freq_lim=None, filtfilt=False, axes=None, grid=False):
    """Plot filter amplitude characteristics.
    
    Parameters
    ----------
    b, a : scalar or 1d array
        Filter coefficients.
    fs : scalar
        Sampling frequency.
    freq_lim : (min, max)
        Frequency limits for plotting.
    filtfilt : bool
        Perform bidirectional filtering using `filtfilt`
    axes : matplotlib axes
    grid : bool
        Display grid.
    
    Returns
    -------
    axes
    
    """
    
    if axes is None:
        fig = plt.figure()
        axes = mpl_toolkits.axes_grid1.host_subplot(111)
    
    w, h = scipy.signal.freqz( b, a, worN=npoints )
    
    freq = 0.5*w*fs / np.pi
    h = np.abs(h)
    if filtfilt:
        h = h**2
    
    axes.plot( freq, 20 * np.log10(h), 'k' )
    plt.setp(axes, xlabel='Frequency [Hz]', ylabel='Amplitude [dB]')
    axes.grid(grid)
    
    par = axes.twinx()
    par.plot( freq, h, 'b' )
    par.set_ylabel( 'Normalized amplitude' )
    par.yaxis.get_label().set_color( 'b' )
    par.grid(False)
    
    if freq_lim is not None:
        axes.set_xlim( freq_lim )
    
    return axes

def plot_filter_phase(b, a=1., fs=1., npoints=None, freq_lim=None, axes=None, grid=False):
    """Plot filter phase characteristics.
    
    Parameters
    ----------
    b, a : scalar or 1d array
        Filter coefficients.
    fs : scalar
        Sampling frequency.
    freq_lim : (min, max)
        Frequency limits for plotting.
    axes : matplotlib axes
    grid : bool
        Display grid.
    
    Returns
    -------
    axes
    
    """
    
    if axes is None:
        fig = plt.figure()
        axes = mpl_toolkits.axes_grid1.host_subplot(111)
    
    w, h = scipy.signal.freqz( b, a, worN=npoints )
    
    freq = 0.5*w*fs / np.pi
    phase = np.unwrap(np.angle(h))
    
    axes.plot( freq, phase, 'k' )
    plt.setp( axes, ylabel='Phase [radians]', xlabel='Frequency [Hz]' )
    axes.grid(grid)
    
    if freq_lim is not None:
        axes.set_xlim( freq_lim )
    
    return axes

def plot_filter_group_delay(b, a=1., fs=1., npoints=None, freq_lim=None, axes=None, grid=False):
    """Plot filter group delay characteristics.
    
    Parameters
    ----------
    b, a : scalar or 1d array
        Filter coefficients.
    fs : scalar
        Sampling frequency.
    freq_lim : (min, max)
        Frequency limits for plotting.
    axes : matplotlib axes
    grid : bool
        Display grid.
    
    Returns
    -------
    axes
    
    """
    if axes is None:
        fig = plt.figure()
        axes = mpl_toolkits.axes_grid1.host_subplot(111)
    
    w, delay = scipy.signal.group_delay( (b,a), w=npoints )
    
    freq = 0.5*w*fs / np.pi
    axes.plot( freq, delay, 'k' )
    plt.setp( axes, ylabel='Group delay [samples]', xlabel='Frequency [Hz]')
    axes.set_ylim( 0, np.max(delay)*1.2 )
    axes.grid(grid)
    
    t = matplotlib.transforms.Affine2D()
    t.scale( 1., fs/1000. )
    twin = axes.twin( t )
    twin.set_ylabel( 'Group delay [ms]' )
    twin.xaxis.set_visible(False)
    twin.grid(False)
    
    if freq_lim is not None:
        axes.set_xlim( freq_lim )
    
    return axes


def compute_envelope( signals, freq_band=None, axis=-1, fs=1., isfiltered=False, filter_options={}, smooth_options={}, pad=True ):
    """Computes average envelope of band-pass filtered signal.
    
    Parameters
    ----------
    signals : array
        either array with raw signals (`isfiltered`==False) or
        pre-filtered signals (`isfiltered`==True). Can also be a sequence
        of such signals.
    freq_band : str or 2-element sequence, optional
        frequency band (in case signal needs to filtered)
    axis : scalar, optional
        axis of the time dimension in the signals array
    fs : scalar, optional
        sampling frequency
    isfiltered : bool, optional
    filter_options : dict, optional
        dictionary with options for filtering (if signal is not already filtered).
        See `apply_filter` and `construct_filter`.
    smooth_options : dict, optional
        dictionary with optional kernel and bandwidth keys for envelope 
        smoothing (see `fklab.signals.kernelsmoothing.smooth1d`)
    pad : bool, optional
        allow zero-padding of signal to nearest power of 2 or 3 in order
        to speed up computation
    
    Returns
    -------
    envelope : 1D array
    
    """
    # filter
    if not isfiltered:
        if freq_band is None:
            raise ValueError('Please specify frequency band')
        filter_arg = dict(transition_width="25%", attenuation=60)
        filter_arg.update(filter_options)
        envelope = apply_filter( signals, freq_band, axis=axis, fs=fs, **filter_arg )
    else:
        envelope = signals
    
    # compute envelope
    if not isinstance( envelope, (tuple,list) ):
        envelope = [envelope,]
    
    if len(envelope)==0:
        raise ValueError('No signal provided.')
        
    # check that all arrays in the list have the same size along axis
    if not all( [ x.shape[axis]==envelope[0].shape[axis] for x in envelope ] ):
        raise ValueError('Signals in list do not have compatible shapes')
    
    N = envelope[0].shape[axis]
    if pad:
        Norig = N
        N = int(np.min( [2,3]**np.ceil( np.log(N)/np.log([2,3]) ) ))
    
    for k in range(len(envelope)):
        envelope[k] = np.abs( scipy.signal.hilbert(envelope[k], N=N, axis=axis) )
        if envelope[k].ndim>1:
            envelope[k] = np.mean( np.rollaxis( envelope[k], axis ).reshape( [ envelope[k].shape[axis], envelope[k].size / envelope[k].shape[axis] ] ), axis=1 )
    
    if len(envelope)>1:    
        envelope = reduce( np.add, envelope ) / len(envelope)
    else:
        envelope = envelope[0]
    
    if pad:
        envelope = envelope[:Norig]
    
    # (optional) smooth envelope
    smooth_arg = dict(kernel='gaussian', bandwidth=-1.)
    smooth_arg.update( smooth_options )
    if smooth_arg['bandwidth']>0:
        envelope = fklab.signals.kernelsmoothing.smooth1d( envelope, delta=1./fs, **smooth_arg)
    
    return envelope

def compute_sliding_rms( signals, freq_band=None, axis=-1, fs=1., isfiltered=False, filter_options={}, smooth_options={} ):
    """Computes root-mean-square of band-pass filtered signal.
    
    Parameters
    ----------
    signals : array
        either array with raw signals (`isfiltered`==False) or
        pre-filtered signals (`isfiltered`==True). Can also be a sequence
        of such signals.
    freq_band : str or 2-element sequence, optional
        frequency band (in case signal needs to filtered)
    axis : scalar, optional
        axis of the time dimension in the signals array
    fs : scalar, optional
        sampling frequency
    isfiltered : bool, optional
    filter_options : dict, optional
        dictionary with options for filtering (if signal is not already filtered).
        See `apply_filter` and `construct_filter`.
    smooth_options : dict, optional
        dictionary with optional kernel and bandwidth keys for smoothing
        (see `fklab.signals.kernelsmoothing.smooth1d`)
    
    Returns
    -------
    rms : 1D array
        root-mean-square of band-pass filtered signal. If multiple signals
        are provided, these are averaged after squaring. Time-weighted
        averaging (i.e. smoothing) is also performed on the squared signal.
    
    """
    
    # filter
    if not isfiltered:
        if freq_band is None:
            raise ValueError('Please specify frequency band')
        filter_arg = dict(transition_width="25%", attenuation=60)
        filter_arg.update(filter_options)
        envelope = apply_filter( signals, freq_band, axis=axis, fs=fs, **filter_arg )
    else:
        envelope = signals
    
    # compute envelope
    if not isinstance( envelope, (tuple,list) ):
        envelope = [envelope,]
    
    if len(envelope)==0:
        raise ValueError('No signal provided.')
    
    # check that all arrays in the list have the same size along axis
    if not all( [ x.shape[axis]==envelope[0].shape[axis] for x in envelope ] ):
        raise ValueError('Signals in list do not have compatible shapes')
    
    for k in range(len(envelope)):
        envelope[k] = envelope[k]**2
        if envelope[k].ndim>1:
            envelope[k] = np.mean( np.rollaxis( envelope[k], axis ).reshape( [ envelope[k].shape[axis], envelope[k].size / envelope[k].shape[axis] ] ), axis=1 )
    
    if len(envelope)>1:
        envelope = reduce( np.add, envelope ) / len(envelope)
    else:
        envelope = envelope[0]
    
    smooth_arg = dict(kernel='gaussian', bandwidth=-1.)
    smooth_arg.update( smooth_options )
    if smooth_arg['bandwidth']>0:
        envelope = fklab.signals.kernelsmoothing.smooth1d( envelope, axis=axis, delta=1./fs, **smooth_arg)
    
    envelope = np.sqrt(envelope)
    
    return envelope

