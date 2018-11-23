"""
=======================================================
Multi-rate functions (:mod:`fklab.signals.multirate`)
=======================================================

.. currentmodule:: fklab.signals.multirate

Functions to increase or decrease a signal's sampling rate.

.. autosummary::
    :toctree: generated/
    
    upsample
    interp
    interp_time_vector
    decimate
    resample

"""
__all__ = ['upsample', 'interp', 'interp_time_vector', 'decimate', 'resample']

import numpy as np
import scipy as sp
import scipy.signal
from scipy.signal import decimate, resample

def upsample( x, factor, axis=-1):
    """Upsample signal by inserting zeros.
    
    Parameters
    ----------
    x : ndarray
    factor : int
        Upscaling factor.
    axis : int
        Array axis along which to upscale.
    
    Returns
    -------
    ndarray
    
    """
    
    x = np.asarray(x)
    
    shape = list(x.shape)
    ndim = len(shape)
    
    factor = int(factor)
    
    if factor==1:
        return x
    elif factor<1:
        raise ValueError
    
    shape[axis] = shape[axis]*factor
    
    #create output array
    y = np.zeros( shape, dtype=x.dtype )
    
    indices = [ slice(None), ]*ndim
    indices[axis] = slice(None,None,factor)
    
    y[ indices ] = x
    
    return y
    
def interp( x, factor, axis=-1, L=4, alpha=0.5, window='blackman' ):
    """Increase sampling rate by factor using interpolation filter.
    
    Parameters
    ----------
    x : ndarray
    factor : int
        Upsampling factor.
    axis : int
        Array axis along which to upsample.
    L : int
        Filter length, calculated as 2*L*factor+1.
    alpha : float
        Normalized cut-off frequency for filter.
    window : str
        Filter window.
    
    Returns
    -------
    ndarray
    
    """
    
    #upsample data
    y = upsample( x, factor, axis=axis )
    
    #create low pass filter
    filter_length = 2*L*factor+1
    F, M = [0., 2.*alpha/factor, 2.*alpha/factor, 1.], [factor, factor, 0., 0.] #frequency and magnitude specification
    b = sp.signal.firwin2( filter_length, F, M, nfreqs= 2**(np.ceil(np.log2(filter_length))+2) + 1, window=window )
    a = 1.
    
    #to minimize edge effects, data at begin (end) of array is rotated 180 degrees
    #around first (last) point
    shape = list(x.shape)
    shape[axis] = len(b)-1
    zi = np.zeros( shape )
    
    #mirror/reflect left edge, upsample and filter
    pre = 2*np.take( x, [0], axis=axis ) - np.take( x, np.arange(2*L+1,0,-1), axis=axis )
    pre = upsample( pre, factor, axis=axis )
    
    pre, zi = sp.signal.lfilter( b, a, pre, axis=axis, zi=zi )
    
    #filter main data
    data, zi = sp.signal.lfilter( b, a, y, axis=axis, zi=zi )
    data = np.roll( data, -L*factor, axis=axis )
    
    #mirror/reflect right edge, upsample and filter
    post = 2*np.take( x, [-1,], axis=axis ) - np.take( x, np.arange(-2,-2*L-2,-1), axis=axis )
    post = upsample( post, factor, axis=axis )
    
    post, zi = sp.signal.lfilter( b, a, post, axis=axis, zi=zi )
    
    indices = [ slice(None), ]*len(shape)
    indices[axis] = slice(-L*factor,None)
    data[indices] = np.take( post, np.arange(L*factor), axis=axis )
    
    return data

def interp_time_vector( t, dt, factor ):
    """Interpolate time vector.
    
    Parameters
    ----------
    t : 1d array
        Time vector.
    dt : scalar
        Interval between time values.
    factor : int
        Upsampling factor
    
    Returns
    -------
    1d array
    
    """
    
    ts = np.arange(factor).reshape(1,factor) * dt/factor
    ts = ts + t.reshape( t.size, 1)
    ts = ts.flatten()
    return ts
