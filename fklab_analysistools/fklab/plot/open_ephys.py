"""
=================================================================
Open Ephys (:mod:`fklab.plot.open_ephys`)
=================================================================

.. currentmodule:: fklab.plot.open_ephys

Convenience functions for plotting open ephys data

.. autosummary::
    :toctree: generated/
    
    plot_recording

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import fklab.plot
import fklab.plot.interaction

__all__ = ['plot_recording']

def plot_recording( filename, start_time=0., channels=None, axes=None, spacing=0, xlim=None, ylim=None, labels=None, **kwargs ):
    """Plot signals in Open Ephys data file.
    
    Parameters
    ----------
    filename : str
    start_time : float, optional
    channels : int array
        Indices of channels to plot.
    axes : matplotlib Axes
    spacing : float
        Spacing between traces.
    xlim : (xmin, xmax) sequence
    ylim : (ymin, ymax) sequence
        The y-axis limits around a single trace.
    labels : [str, ...]
    **kwargs : extra keyword arguments for `fklab.plot.plot_signals` function.
    
    Returns
    -------
    LinearOffsetCollection and ScrollPanZoom objects
    
    """
    
    f = h5py.File(filename, mode='r')
    
    if channels is None:
        channels = np.arange( f['recordings/0/data'].shape[1], dtype=np.int )
    
    fs = f['recordings/0'].attrs['sample_rate']
    bitvolts = f['recordings/0/application_data'].attrs['channel_bit_volts'][channels]
    
    if labels is None:
        labels = [ 'channel {}'.format(k) for k in range(len(channels)) ]
        
    if axes is None:
        fig = plt.figure()
        axes = plt.axes()
    
    t = fklab.plot.utilities.RangeVector( f['recordings/0/data'].shape[0], start_time, 1./fs )
    
    h = fklab.plot.plot_signals( t, [ fklab.plot.utilities.ColumnView(f['recordings/0/data'],k, lambda x, scale=0.001*bitvolts[k]: x*scale ) for k in channels ], spacing=spacing, axes=axes, **kwargs )
    
    v = fklab.plot.interaction.ScrollPanZoom(axes)
    v.enable()
    
    if xlim is None:
        axes.set_xlim( t[0], t[0]+1 )
    else:
        axes.set_xlim(xlim)
    
    if ylim is None:
        ylim = [-1,1]
    
    maxoffset = (len(channels)-1)*spacing
    ymin = min( ylim[0], maxoffset + ylim[0] )
    ymax = max( ylim[1], maxoffset + ylim[1] )
    
    axes.set_ylim( ymin, ymax )
    
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    #axes.yaxis.set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    
    scalebar = fklab.plot.artists.StaticScaleBar( location='right', size=[0.,0.1] , label='{value:.2f} mV', linewidth=2, offset=5 )
    axes.add_artist(scalebar)
    
    axes.set_yticks( np.arange(len(channels)) * spacing )
    axes.set_yticklabels( labels )
    axes.grid(axis='both')
    
    axes.set_xlabel('time [s]')
    
    if len(filename)>50:
        axes.set_title( '...' + filename[-50:] )
    else:
        axes.set_title( filename )
    
    axes.figure.canvas.draw()
    
    return h, v
