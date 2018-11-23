"""
======================================================
Information (:mod:`fklab.statistics.information`)
======================================================

.. currentmodule:: fklab.statistics.information

Information theoretic functions.

.. autosummary::
    :toctree: generated/
    
    spatial_information

"""

import numpy as np

def spatial_information( spike_rate, occupancy, units='bits/spike' ):
    """Compute spatial information measure.
    
    Parameters
    ----------
    spike_rate : array or sequence of arrays
    occupancy : array
    units : {'bits/spike','bits/second'}
    
    Returns
    -------
    scalar or list of scalars
    
    """
    
    list_input = False
    
    if not isinstance( spike_rate, (list, tuple) ):
        spike_rate = [ spike_rate, ]
        list_input = True
        
    
    spike_rate = [ np.array( x, copy=False ) for x in spike_rate ]
    
    occupancy = np.array( occupancy, copy=False )
    
    if not all( [x.shape==occupancy.shape for x in spike_rate] ):
        raise ValueError( 'Arrays do not have same shape' )
    
    if units not in ( 'bits/spike', 'bits/second' ):
        raise ValueError( 'Invalid units' )
    
    nspikes = [ x * occupancy for x in spike_rate ]
    totalspikes = [ np.nansum(x) for x in nspikes ]
    totaltime = np.nansum( occupancy )        
    
    print nspikes, totalspikes, totaltime
    
    si = [ np.nansum( (y/z) * np.log2( x/(z/totaltime) ) ) for x,y,z in zip( spike_rate, nspikes, totalspikes ) ]
    
    if units=='bits/second':
        si = [ x * y/totaltime for x,y in zip( si, totalspikes ) ]
    
    return si
