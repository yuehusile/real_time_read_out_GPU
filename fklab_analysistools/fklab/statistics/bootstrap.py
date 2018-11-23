"""
======================================================
Bootstrap (:mod:`fklab.statistics.bootstrap`)
======================================================

.. currentmodule:: fklab.statistics.bootstrap

Bootstrapping utilities.

.. autosummary::
    :toctree: generated/
    
    ci

"""
import numpy as np

def ci(data, nsamples=10000, statistic=None, alpha=0.05, axis=0):
    """Bootstrap estimate of 100.0*(1-alpha) confidence interval for statistic.
    
    Parameters
    ----------
    data : array or sequence of arrays
    nsamples : int, optional
        number of bootstrap samples
    statistic : function( data ) -> scalar or array, optional
    alpha : float, optional
    axis : int, optional
        axis of data samples
        
    Returns
    -------
    low,high : confidence interval
    
    """
    
    if not isinstance(data, (list,tuple) ):
        data = (data,)
    
    if statistic is None:
        statistic = lambda x : np.average( x, axis=axis )
    
    bootindexes = bootstrap_indexes( data[0].shape[axis], nsamples )
    stat = np.array([statistic(*(x.take(indexes,axis=axis) for x in data)) for indexes in bootindexes])
    #stat.sort(axis=0)
    
    p = np.percentile( stat, [100.*alpha/2., 100.*(1-alpha/2.)], axis=0 )
    
    return p[0], p[1]


def bootstrap_indexes(n, nsamples=10000):
    """Generator for bootstrap indices.
    """
    for _ in xrange(nsamples):
        yield np.random.randint(n, size=(n,))
