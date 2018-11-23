"""
=================================================================
Neuralynx (:mod:`fklab.plot.neuralynx`)
=================================================================

.. currentmodule:: fklab.plot.neuralynx

Convenience functions for plotting neuralynx data

.. autosummary::
    :toctree: generated/
    
    plot_csc
    plot_csc_spectrogram
    plot_nev

"""

import numpy as np
import fklab.io.neuralynx as nlx
from .interaction import ScrollPanZoom
from .plotting import plot_signals, plot_events, plot_spectrogram
import matplotlib.pyplot as plt
import fklab.events
import fklab.plot.artists

__all__ = ['plot_nev','plot_csc', 'plot_csc_spectrogram']

def plot_nev( filename, events=None, axes=None, **kwargs ):
    """Plot events from neuralynx nev file.
    
    Parameters
    ----------
    filename : str
    events : str or list of str
    axes : matplotlib Axes object
    kwargs : option for plot_events function
    
    Returns
    -------
    collection of events
    
    """
    
    fid = nlx.NlxOpen( filename )
    data = fid.data.default[:]
    data = fklab.events.split_eventstrings( data.time, data.eventstring  )
    
    if events is None:
        events = data.keys()
    elif not isinstance(events,(list,tuple)):
        events = [events,]
    
    data = [ data[k] for k in events ]
    
    if axes is None:
        fig = plt.figure()
        axes = plt.axes()
    
    h = plot_events( data, fullheight=True, axes=axes, **kwargs )
    
    return h

#def inspect_csc( filename, xlim=None, **kwargs ):
    
    # open file
#    fid = nlx.NlxOpen( filename )
    
    # create two subplots with linked x-axes
#    fig = plt.figure()
    
    # plot_csc( fid, axes=, xlim=xlim, ... )
    
    # plot_csc_spectrogram( fid, axes=, xlim=xlim, ... )
    
    # return plot objects

def plot_csc_spectrogram( filename, axes=None, xlim=None, **kwargs ):
    """Plot spectrogram of CSC file.
    
    Parameters
    ----------
    files : str or list of str
    axes : matplotlib Axes object, optional 
    xlim : 2-element sequence
        initial x-axis limits
    kwargs : options for plot_spectrogram
    
    Returns
    -------
    FastSpectrogram object
    ScrollPanZoom interaction object
    
    """
    
    if isinstance(filename, str):
        fid = nlx.NlxOpen( filename )
    else:
        fid = filename
    
    fid.units='mV'
    
    kwargs['fs'] = fid.header['SamplingFrequency']
    kwargs['start_time'] = fid.starttime
    kwargs['freq_range'] = [0.,250.]
    kwargs['decimate'] = True
    
    if axes is None:
        fig = plt.figure()
        axes = plt.axes()
    
    h = plot_spectrogram( fid.data.signal_by_sample, axes=axes, **kwargs )
    
    v = ScrollPanZoom(axes)
    v.enable()
    
    if xlim is None:
        axes.set_xlim( fid.starttime, fid.endtime )
    else:
        axes.set_xlim(xlim)
    
    axes.set_ylim(h._freq_range)
    
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    
    #TODO: set units correctly based on options passed to FastSpectrogram
    cbar = plt.colorbar(h,ax=axes)
    cbar.set_label('power spectral density [mV*mV/Hz] in db')
    
    axes.set_ylabel('frequency [Hz]')
    axes.set_xlabel('time [s]')
    h.set_clim(-150,5)
    
    axes.figure.canvas.draw()
    
    return h, v
    
def plot_csc( files, axes=None, spacing=0, xlim=None, ylim=[-2,2], labels=None, **kwargs ):
    """Plot signals from neuralynx csc files.
    
    Parameters
    ----------
    files : str or list of str
    axes : matplotlib Axes object, optional
    spacing : float
        spacing between traces
    xlim : 2-element sequence
        initial x-axis limits
    ylim : 2-element sequence
        single trace y-axis limits. Used to properly set ylim of axes.
    labels : sequence of str
        trace labels
    kwargs : options for plot_signals
    
    Returns
    -------
    collection of lines
    ScrollPanZoom interaction object
    
    """
    
    x = []
    y = []
    
    xmin = np.inf
    xmax = -np.inf
    
    if not isinstance(files, (tuple,list)):
        files = [files,]
    
    if labels is None:
        labels = [ 'trace {}'.format(k) for k in range(len(files)) ]
    
    for f in files:
        if isinstance(f,str):
            fid = nlx.NlxOpen( f )
        else:
            fid = f
        fid.units = 'mV'
        x.append( fid.data.time_by_sample )
        y.append( fid.data.signal_by_sample )
        
        xmin = min( fid.starttime, xmin )
        xmax = max( fid.endtime, xmax )
    
    if axes is None:
        fig = plt.figure()
        axes = plt.axes()
    
    h = plot_signals( x, y, spacing=spacing, axes=axes, **kwargs )
    
    v = ScrollPanZoom(axes)
    v.enable()
    
    if xlim is None:
        axes.set_xlim( xmin, xmax )
    else:
        axes.set_xlim(xlim)
    
    maxoffset = (len(files)-1)*spacing
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
    
    axes.set_yticks( np.arange(len(files)) * spacing )
    axes.set_yticklabels( labels )
    axes.grid(axis='both')
    
    axes.set_xlabel('time [s]')
    
    axes.figure.canvas.draw()
    
    return h, v
