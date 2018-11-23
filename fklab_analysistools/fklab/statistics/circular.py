"""
======================================================
Circular statistics (:mod:`fklab.statistics.circular`)
======================================================

.. currentmodule:: fklab.statistics.circular

Statistical functions for circular variables.

Utilities
=========

.. autosummary::
    :toctree: generated/
    
    deg2rad
    rad2deg
    wrap
    diff
    rank
    uniformize
    inrange

Summary statistics
==================

.. autosummary::
    :toctree: generated/
    
    mean
    dispersion
    centralmoment
    moment
    median
    interval
    std

Statistical Tests
=================

.. autosummary::
    :toctree: generated/
    
    kuiper
    rayleigh
    
Circular Distributions
======================

.. autosummary::
    :toctree: generated/
    
    uniform
    vonmises

Circular Histogram and Density
==============================

.. autosummary::
    :toctree: generated/
    
    kde
    hist

"""

from __future__ import division

import numpy as np
import scipy as sp
import scipy.stats

from numpy import rad2deg, deg2rad
from scipy.stats import vonmises

from fklab.codetools import deprecated

__all__ = ['rad2deg', 'deg2rad', 'wrap', 'diff', 'mean', 'dispersion', 'centralmoment', 
           'moment', 'median', 'kuiper', 'rayleigh', 'rank', 'uniformize',
           'interval', 'std', 'inrange', 'uniform', 'vonmises', 'kde', 'hist']

def wrap(x,low=0.0,high=2*np.pi):
    """Wrap values to circular range.
    
    Parameters
    ----------
    x : ndarray
    low, high : float
        Low and high values that define the circular range.
    
    Returns
    -------
    ndarray
    
    """
    #y = np.mod( np.mod( x, extent ) + extent - offset, extent ) + offset;
    #y = np.mod( np.mod(x-low,high-low) + high-low - offset, high-low ) + offset + low
    x = np.asarray(x)
    y = np.mod( np.mod(x-low,high-low) + high-low, high-low ) + low
    return y

def diff(phi,theta=None,axis=0,low=0.0,high=2*np.pi,directed=False):
    """Computes circular difference.
    
    Parameters
    ----------
    phi : ndarray
        Array with angles (in radians).
    theta : ndarray
        If given, the function computes the pair-wise circular difference
        between phi and theta.
    axis : int
        Array axis along which to compute the circular difference.
    low, high : float
        Circular range.
    directed : bool
        Compute directed distance, which is negative if the shortest
        distance from one angle to another is counter-clockwise.
        
    Returns
    -------
    ndarray
    
    """
    
    #make sure phi is within range
    phi = wrap( phi, low=low, high=high )
    range_center = (low+high)/2.0
    
    if theta is None:
        d = np.diff(phi,1,axis=axis)
    else:
        theta = wrap( theta, low=low, high=high )
        d = theta-phi
    
    if directed:
        d = wrap( theta - phi, low=low-range_center, high=high-range_center )
    else:
        d = range_center - np.abs( range_center - np.abs(theta-phi) )
    
    return d

def mean(theta,axis=None,weights=None,keepdims=False, alpha=0.05):
    """Computes circular mean of angles.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    axis : int
        Array axis along which to compute circular mean.
    weights : ndarray
        Array of weights for a weighted circular mean.
    keepdims : bool
        Do not reduce the number of dimensions.
    alpha : float
        Alpha value for confidence interval.
    
    Returns
    -------
    mu : scalar or ndarray
        Mean angle
    rbar : scalar or ndarray
        Resultant vector length
    ci : [scalar, scalar ] or [ ndarray, ndarray ]
        Estimated low and high confidence intervals, based on circular
        dispersion
    
    """
    if weights is None:
        weights = 1
    
    theta = np.asarray(theta)
    weights = np.asarray(weights)
    
    S = np.nansum( weights * np.sin(theta), axis=axis, keepdims=keepdims )
    C = np.nansum( weights * np.cos(theta), axis=axis, keepdims=keepdims )
    
    m = np.arctan2(S,C)
    
    n = np.sum( weights * ~np.isnan(theta), axis=axis, keepdims=keepdims )
    rbar = np.sqrt(S**2 + C**2) / n
    
    d = np.sqrt( dispersion(theta,axis=axis,weights=weights,keepdims=keepdims)/n )
    d = sp.stats.norm.ppf( 1-0.5*alpha, loc=0, scale=1) * d
    d = np.minimum( d, 1.0 ) #if distribution is very spread out, then return the maximum confidence interval
    ci_mu = np.arcsin( d )
    ci_mu = [m-ci_mu,m+ci_mu]
    
    return (m, rbar, ci_mu)

def dispersion(theta,axis=None,weights=None,keepdims=False):
    """Computes circular dispersion.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    axis : int
        Array axis along which to compute circular dispersion.
    weights : ndarray
        Array of weights for a weighted circular dispersion.
    keepdims : bool
        Do not reduce the number of dimensions.
    
    Returns
    -------
    dispersion : ndarray
    
    """
    
    if weights is None:
        weights = 1
    
    theta = np.asarray(theta)
    weights = np.asarray(weights)
    
    r = np.abs( centralmoment( theta, k=1, axis=axis, weights=weights, keepdims=keepdims ) )
    p2 = np.abs( centralmoment( theta, k=2, axis=axis, weights=weights, keepdims=keepdims ) )
    
    d = (1-p2)/(2*r**2)
    
    return d

def centralmoment(theta,k=1,axis=None,weights=None,keepdims=False):
    """Computes circular central moment.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    k : int
    axis : int
        Array axis along which to compute circular central moment.
    weights : ndarray
        Array of weights for a weighted circular central moment.
    keepdims : bool
        Do not reduce the number of dimensions.
    
    Returns
    -------
    ndarray
    
    """
    
    if weights is None:
        weights = 1
    
    theta = np.asarray(theta)
    weights = np.asarray(weights)
    
    m1 = np.angle( moment( theta, k, axis=axis, weights=weights, keepdims=True ) )
    
    n = np.sum( weights * ~np.isnan(theta), axis=axis, keepdims=keepdims )
    S = np.nansum( weights * np.sin( k*(theta-m1)), axis=axis, keepdims=keepdims )/n
    C = np.nansum( weights * np.cos( k*(theta-m1)), axis=axis, keepdims=keepdims )/n
    
    m = C + S*1j
    
    return m

def moment(theta,k=1,axis=None,weights=None,keepdims=False):
    """Computes circular moment.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    k : int
    axis : int
        Array axis along which to compute circular moment.
    weights : ndarray
        Array of weights for a weighted circular moment.
    keepdims : bool
        Do not reduce the number of dimensions.
    
    Returns
    -------
    ndarray
    
    """
    
    if weights is None:
        weights=1
    
    theta = np.asarray(theta)
    weights = np.asarray(weights)
    
    n = np.sum( weights*~np.isnan(theta), axis=axis, keepdims=keepdims )
    S = np.nansum( weights * np.sin(k*theta), axis=axis, keepdims=keepdims )/n
    C = np.nansum( weights * np.cos(k*theta), axis=axis, keepdims=keepdims )/n
    
    m = C + S*1j
    
    return m

def median(theta):
    """Computes circular median.
    
    The circular median is computed as the angle for which the mean 
    circular difference to all angles in theta is minimized.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians). The median is computed over the
        whole array.
    
    Returns
    -------
    median : float
    mdev : float
        Mean circular difference of theta angles to median angle.
    mdiff : float
        Mean circular difference between all pairs of angles in theta.
    
    """
    
    theta = np.asarray(theta).ravel()
    theta = theta[~np.isnan(theta)]
    n = theta.size
    
    if n==0:
        return (np.NaN, np.NaN, np.NaN)
    
    if n==1:
        return (theta,0,0)
    
    theta = wrap(theta)
    
    mean_circ_diff = lambda a: np.sum( np.pi - np.abs( np.pi - np.abs( theta - a ) ) ) / n
    
    m, mdev = sp.optimize.fminbound( mean_circ_diff, 0, 2*np.pi, full_output=True )[0:2]
    
    mdiff=0
    for k in range(n):
        mdiff+=mean_circ_diff(theta[k])
    mdiff/=n
    
    return (m,mdev,mdiff)

def kuiper(theta,axis=None,keepdims=False):
    """Kuiper's one sample test of uniformity.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    axis : int
        Array axis along which to perform Kuiper's test.
    keepdims : bool
        Do not reduce the number of dimensions.
    
    Returns
    -------
    p : ndarray
        P-value.
    V : ndarray
        Test statistic.
    
    """
    
    theta=np.asarray(theta)
    
    if axis is None:
        theta = theta.ravel()
        axis=0
    
    n=np.sum( ~np.isnan(theta), axis=axis, keepdims=True )
    theta=np.sort( wrap( theta ), axis=axis )
    
    shape = [1] * theta.ndim
    shape[axis] = theta.shape[axis]
    U = theta/(2*np.pi) - np.arange(shape[axis]).reshape( shape )/n
    
    if not keepdims:
        n = n.squeeze(axis=axis)
    
    V = np.nanmax(U,axis=axis,keepdims=keepdims) - np.nanmin(U,axis=axis,keepdims=keepdims) + 1/n
    
    #modified according to Stephens
    V = V * (np.sqrt(n) + 0.155 + 0.24/np.sqrt(n))
    
    p = (8*V**2-2)*np.exp(-2*V**2)
    
    return (p,V)

def rayleigh(theta,axis=None,keepdims=False):
    """Rayleigh test for uniformity.
    
    Performs a Rayleigh test of uniformity, against the alternative
    hypothesis of a unimodal distribution.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    axis : int
        Array axis along which to perform Rayleigh test.
    keepdims : bool
        Do not reduce the number of dimensions.
    
    Returns
    -------
    p : ndarray
        P-value.
    V : ndarray
        Test statistic.
    
    """
    
    theta = np.asarray(theta)
    
    n = np.sum( ~np.isnan(theta), axis=axis, keepdims=keepdims )
    mu,rbar = mean( theta, axis=axis, keepdims=keepdims )[0:2]
    
    S = (1-1/(2*n))*2*n*rbar**2 + n*rbar**4/2
    p = 1 - sp.stats.chi2.cdf(S,2)
    
    return (p,S)

def rank(theta):
    """Computes circular rank order statistic.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians). The rank is computed over the
        whole array.
    
    Returns
    -------
    ndarray
    
    """
    
    theta = np.asarray(theta).ravel()
    valid = ~np.isnan(theta)
    
    n = np.sum(valid)
    r = np.empty( theta.shape ) * np.nan
    
    r[valid] = sp.stats.rankdata( wrap(theta[valid]), method='average' )
    g = 2*np.pi*r/n
    
    return g

def uniformize(population, sample=None,*args,**kwargs):
    """Uniformly distribute angles while preserving rank order.
    
    This function will transform angles according to the estimated CDF
    of the angles in population (i.e. the population distribution is made
    uniform).
    
    Parameters
    ----------
    population : callable or ndarray
        Either an array of angles that represent the population, or a 
        callable that takes the sample array as a first argument and
        computes the CDF of the population distribution at the sample
        angles.
    sample : ndarray
        Array of angles (in radians). If None, the circular rank of the
        population is computed.
    *args, **kwargs
        Extra arguments that are passed to population callable.
    
    Returns
    -------
    ndarray
    
    """
    
    if sample is not None:
        
        sample = np.asarray(sample)
        
        if callable(population):
            return 2*np.pi*population(sample,*args,**kwargs)
        else:
            
            population = np.asarray(population)
            poprank = rank(population)
            
            #remove duplicate values for correct interpolation
            population, bi = np.unique( population, return_index=True )
            poprank = poprank[bi]
            
            #compute ranks for samples
            sample = wrap(sample)
            r = sp.interpolate.interp1d(population,poprank,kind='linear')(sample)
            
            return r
            
    else:
        return rank(population)

def interval(theta,axis=None,keepdims=False):
    """Computes circular interval.
    
    This function will find the smallest arc that contains all angles in
    theta.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    axis : int
        Array axis over which interval is computed. If axis is None, then
        the interval is computed over the whole array.
    keepdims : bool
        Do not reduce the number of dimensions.
    
    Returns
    -------
    ndarray
    
    """
    
    theta = np.asarray(theta)
    
    if axis is None:
        theta = theta.ravel()
        axis=0
        
    theta = np.sort(theta,axis=axis)
    T = np.diff(theta,axis=axis)
    T = np.concatenate( (T,2*np.pi+np.diff(theta.take([-1,0],axis=axis),axis=axis)), axis=axis )
    
    #compute circular interval
    w = 2*np.pi - np.max( T, axis=axis, keepdims=keepdims )
    
    return w

def std(theta,axis=None,weights=None,keepdims=False):
    """Computes circular standard deviation.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    axis : int
        Array axis over which standard deviation is computed.
    weights : ndarray
        Array of weights for a weighted circular standard deviation.
    keepdims : bool
        Do not reduce the number of dimensions.
    
    Returns
    -------
    ndarray
    
    """
    
    if weights is None:
        weights=1
    
    rbar = np.abs( centralmoment( theta, moment=1, axis=axis, weights=weights, keepdims=keepdims ) )
    v = np.sqrt( -2*np.log(rbar) )
    
    return v

def inrange(theta,crange=[],low=0.0,high=2*np.pi):
    """Tests if angles are within a circular range.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles (in radians).
    crange : [lower, upper]
        Circular range to test for. If lower >= upper, then the wrapped
        around range from lower to upper is taken (e.g. if crange equals
        [1.5*pi, 0.5*pi], then angles larger than 1.5*pi and smaller
        than 0.5*pi are considered within the test range.)
    low, high : float
        Circular range of theta angles
    
    Returns
    -------
    ndarray
    
    """
    
    theta = np.asarray(theta)
    crange = np.asarray(crange)
    
    if crange.size==0:
        return np.zeros( theta.shape, dtype=np.bool )
    
    if crange[0]==crange[1]:
        return np.ones( theta.shape, dtype=np.bool )
    
    theta = wrap(theta,low=low,high=high)
    crange = wrap(crange,low=low,high=high)
    
    if crange[0]>=crange[1]:
        return np.logical_or( theta>=crange[0], theta<=crange[1] )
    else:
        return np.logical_and( theta>=crange[0], theta<=crange[1] )

@deprecated('Please use uniform.rvs instead.')
def uniformrnd(size=1):
    """Random sample from circular uniform distribution.
    
    Parameters
    ----------
    size : tuple of ints
    
    Returns
    -------
    ndarray
    
    """
    return np.random.uniform(low=0,high=2*np.pi,size=size)

@deprecated('Please use uniform.cdf instead.')
def uniformcdf(theta):
    """Circular uniform distribution function.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles at which the circular uniform cumulative
        distribution density is computed.
    
    Returns
    -------
    ndarray
    
    """
    
    return wrap(theta)/(2*np.pi)

@deprecated('Please use uniform.pdf instead.')
def uniformpdf(theta):
    """Circular uniform probability density function.
    
    Parameters
    ----------
    theta : ndarray
        Array of angles at which the circular uniform probability density
        is computed.
    
    Returns
    -------
    ndarray
    
    """
    
    theta=np.asarray(theta)
    return np.ones( theta.shape )/(2*np.pi)

class uniform_gen(scipy.stats.rv_continuous):
    """A uniform circular continuous random variable.

    This distribution is constant around the unit circle.

    %(before_notes)s

    %(example)s

    """
    
    def _rvs(self):
        return self._random_state.uniform(0.0, 2.*np.pi, self._size)

    def _pdf(self, x):
        return (x == x)/(2.*np.pi)

    def _cdf(self, x):
        return wrap(x)/(2.*np.pi)

    def _ppf(self, q):
        return wrap(q)/(2.*np.pi)

    def _stats_skip(self):
        return 0., None, 0., None

    def _entropy(self):
        return 0.0

uniform = uniform_gen()

def kde( data, theta=None, bandwidth=None, kernel='vonmises', weights=None, normalize=True ):
    """Kernel density estimate for circular data.
    
    Parameters
    ----------
    data : 1d array-like
    theta : int or 1d-array-like
    bandwidth : float
    kernel : {'vonmises', 'box'}
    weights : 1d array-like
    
    Returns
    -------
    density : 1d array
    theta : 1d array
    
    """
    
    data = np.array( data, copy=False )
    valid = np.logical_not( np.isnan( data ) )
    
    n = np.sum(valid)
    
    if theta is None:
        theta = 24
    
    if isinstance( theta, int ):
        theta = np.arange(theta, dtype=np.float)*2*np.pi/theta
    else:
        theta = np.array(theta, copy=False).ravel()
    
    if n==0:
        density = np.zeros( len(theta) )
        return density, theta
    
    if not kernel in ('vonmises', 'box'):
        raise ValueError('Invalid kernel.')
    
    if bandwidth is None:
        bandwidth = 5. if kernel=='vonmises' else np.pi/6.
    else:
        bandwidth = float(bandwidth)
    
    if weights is not None:
        weights = np.array(weights, copy=False).ravel()
        if len(weights)!=n:
            raise ValueError('Number of weights is not equal to the number of data points.')
        n = np.nansum( weights[valid] )
    
    if kernel == 'vonmises':
        density = vonmises.pdf( theta[:,None], bandwidth, loc=data[valid][None,:] )
    else:
        density = diff( theta[:,None], data[valid][None,:] ) <= 0.5*bandwidth
        density = density.astype(np.float)/bandwidth
    
    if weights is not None:
        density = density * weights[valid][None,:]
        
    density = np.nansum( density, axis=1 )
    
    if normalize:
        density = density / n
    
    return density, theta

def hist( data, bins=24 ):
    """Histogram of circular data.
    
    Parameters
    ----------
    data : 1d array-like
    bins : int
        Number of bins in histogram
        
    Returns
    -------
    count : 1d array
        
    theta : 1d array
        Bin centers
    width : float
        Bin width
    
    """
    
    width = 2*np.pi/int(bins)
    count, theta = kde( data, bins, bandwidth=width, kernel='box', normalize=False )
    count = (count * width).astype(np.int)
    
    return count, theta, width


