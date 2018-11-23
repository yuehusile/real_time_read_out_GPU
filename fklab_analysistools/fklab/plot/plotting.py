"""
=======================================================
General plotting functions (:mod:`fklab.plot.plotting`)
=======================================================

.. currentmodule:: fklab.plot.plotting

Function for plotting multiple signals, events, segments and rasters.

plotting functions
==================

.. autosummary::
    :toctree: generated/
    
    plot_signals
    plot_events
    plot_segments
    plot_raster
    plot_spectrogram
    add_scalebar
    add_static_scalebar

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.transforms
import matplotlib.cm

from artists import FastLine, FastRaster, FastSpectrogram, StaticScaleBar, AnchoredScaleBar, AxesMessage
from utilities import LinearOffsetCollection, RangeVector, ColumnView
from interaction import ScrollPanZoom

import fklab.segments

__all__ = ['plot_signals','plot_events','plot_segments','plot_raster',
           'plot_spectrogram', 'add_static_scalebar', 'add_scalebar']


def plot_signals(*args,**kwargs):
    """Plot multiple time series.
    
    Parameters
    ----------
    x : 1D array-like or list of 1D-array like, optional
    y : array-like or list of array like
    spacing : scalar , optional
    origin : scalar, optional
    lineclass : {Line2D, FastLine}, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : line properties
    
    Returns
    -------
    collection of lines
    
    """

    if len(args)<1 or len(args)>2:
            raise ValueError('Expecting one or two positional data arguments')
    
    if len(args)==1:
        x = None
        y, = args
    else:
        x,y = args
    
    # TODO: check x and y inputs
    # how many arrays? how many signals? how many samples?
    if not isinstance( y, list ):
        y = [ y, ]
    
    if not all( [ len(p.shape)==1 or len(p.shape)==2 for p in y ] ):
        raise ValueError('Invalid data')
    
    if not isinstance( x, list ):
        x = [x, ] * len(y)
    elif len(x)!=len(y):
        raise ValueError('Invalid time vectors')
    
    if not all ( [p is None or len(p.shape)==1 for p in x] ):
        raise ValueError('Invalid time vectors')
    
    data_shapes = np.array( [ [p.shape[0], 1 if len(p.shape)==1 else p.shape[1] ] for p in y ] )
    time_shapes = np.array( [ -1 if p is None else p.shape[0] for p in x ] )
        
    if np.any( np.logical_and( data_shapes[:,0]!=time_shapes, time_shapes!=-1 ) ):
        raise ValueError('Invalid data')
    
    nsignals = np.sum( data_shapes[:,1] )
    
    # get LinearOffsetCollection keywords
    spacing = kwargs.pop( 'spacing', 0. )
    origin = kwargs.pop( 'origin', 0. )
    
    cm = kwargs.pop( 'colormap', None )
    
    if cm is not None:
        alpha = kwargs.pop( 'alpha', 1.0 )
        if isinstance(cm, str):
            cm = matplotlib.cm.get_cmap(cm, lut=nsignals)
    
    color = kwargs.pop('color',None)
    
    if cm is not None and color is not None:
        raise ValueError("Specify either color or colormap, but not both")
    
    lineclass = kwargs.pop( 'lineclass', FastLine )
    
    lines = []
    
    ax = kwargs.pop( 'axes', None )
        
    if ax is None:
        ax = plt.gca()
    
    def local_color(k):
        if cm is not None:
            return cm(k % cm.N, alpha=alpha)
        else:
            return color
    
    # create lines
    signal_index = 0
    for xdata,ydata in zip(x,y):
        
        #if cm is not None:
        #    color = cm(signal_index % cm.N, alpha=alpha)
        #    signal_index += 1
        
        if xdata is None:
            if ydata.ndim==1:
                lines.append( lineclass( RangeVector(len(ydata)), ydata, color=local_color(signal_index), **kwargs ) )
                signal_index += 1
            else:
                for idx in range(ydata.shape[1]):
                    lines.append( lineclass( RangeVector(ydata.shape[0]), ColumnView(ydata,idx), color=local_color(signal_index), **kwargs ) )
                    signal_index += 1
        else:
            if ydata.ndim==1:
                lines.append( lineclass( xdata, ydata, color=local_color(signal_index), **kwargs ) )
                signal_index += 1
            else:
                for idx in range(ydata.shape[1]):
                    lines.append( lineclass( xdata, ColumnView(ydata,idx), color=local_color(signal_index), **kwargs ) )
                    signal_index += 1
    
    for l in lines:
        ax.add_line( l )
    
    collection = LinearOffsetCollection( lines, spacing=spacing, origin=origin, direction='vertical' )
    
    plt.draw()
    
    return collection

def plot_events( events, **kwargs ):
    """Plot multiple event series.
    
    Parameters
    ----------
    events : 1D array-like or list of 1D-array like
    kind : {'event', 'raster'}, optional
    spacing : scalar , optional
    origin : scalar, optional
    fullheight : bool, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : EventCollection or FastRaster options
    
    Returns
    -------
    collection of events
    
    """
    
    if not isinstance( events, list ):
        events = [ events, ]
    
    if not all( [ p.ndim==1 for p in events ] ):
        raise ValueError('Invalid data')
    
    kind = kwargs.pop( 'kind', 'event' )
    
    # get LinearOffsetCollection keywords
    spacing = kwargs.pop( 'spacing', 0. )
    origin = kwargs.pop( 'origin', 0. )
    
    fullheight = kwargs.pop( 'fullheight', False )

    cm = kwargs.pop( 'colormap', None )
    
    alpha = kwargs.pop( 'alpha', 1.0 )
    
    if cm is not None:
        if isinstance(cm, str):
            cm = matplotlib.cm.get_cmap(cm, lut=len(events))
    
    color = kwargs.pop('color',None)
    
    if cm is not None and color is not None:
        raise ValueError("Specify either color or colormap, but not both")
        
    ax = kwargs.pop( 'axes', None )
        
    if ax is None:
        ax = plt.gca()

    items = []
    
    if kind=='event':
        for k in range( len(events) ):
            if cm is not None:
                color = cm(k % cm.N, alpha=alpha)
            items.append( matplotlib.collections.EventCollection( events[k], color=color, **kwargs ) )
    elif kind=='raster':
        for k in range( len(events) ):
            if cm is not None:
                color = cm(k % cm.N)
            items.append( FastRaster( events[k], foreground_color=color, foreground_alpha=alpha, **kwargs ) )
    else:
        raise ValueError('Unknown value for parameter kind')
    
    for item in items:
        if kind=='event': ax.add_collection( item )
        else: ax.add_image( item )
        
        if fullheight:
                item.set_transform( matplotlib.transforms.blended_transform_factory( ax.transData, ax.transAxes ) )
    
    if not fullheight:
        items = LinearOffsetCollection( items, spacing=spacing, origin=origin, direction='vertical' )
    
    plt.draw()
    
    return items

def plot_segments( s, **kwargs ):
    """Plot multiple segment series.
    
    Parameters
    ----------
    segments : Nx2 array-like or list of Nx2 array-like
    spacing : scalar , optional
    origin : scalar, optional
    fullheight : bool, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : BrokenBarHCollection options
    
    Returns
    -------
    collection of BrokenBarHCollection
    
    """
    
    if not isinstance( s, list ):
        s = [ s, ]
    
    s = [ fklab.segments.check_segments(k, copy=True) for k in s ]
    
    # get LinearOffsetCollection keywords
    spacing = kwargs.pop( 'spacing', 0. )
    origin = kwargs.pop( 'origin', 0. )
    
    fullheight = kwargs.pop( 'fullheight', False )
    
    cm = kwargs.pop( 'colormap', None )
    
    if cm is not None:
        alpha = kwargs.pop( 'alpha', 1.0 )
        if isinstance(cm, str):
            cm = matplotlib.cm.get_cmap(cm, lut=len(s))
    
    color = kwargs.pop('color',None)
    
    if cm is not None and color is not None:
        raise ValueError("Specify either color or colormap, but not both")
    
    ax = kwargs.pop( 'axes', None )
        
    if ax is None:
        ax = plt.gca()
    
    items = []
    
    for k in range( len(s) ):
        if cm is not None:
                color = cm(k % cm.N, alpha=alpha)
        s[k][:,1] = s[k][:,1]-s[k][:,0]
        items.append( matplotlib.collections.BrokenBarHCollection( s[k],  [0,1], facecolors=color, **kwargs ) )
    
    for item in items:
        ax.add_collection( item )
    
    if fullheight:
        for item in items:
            item.set_transform( matplotlib.transforms.blended_transform_factory( ax.transData, ax.transAxes ) )
    else:
        items = LinearOffsetCollection( items, spacing=spacing, origin=origin, direction='vertical' )
    
    plt.draw()
    
    return items

def plot_raster(events, **kwargs):
    """Plot multiple event series.
    
    Parameters
    ----------
    events : 1D array-like or list of 1D-array like
    spacing : scalar , optional
    origin : scalar, optional
    fullheight : bool, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : FastRaster options
    
    Returns
    -------
    collection of rasters
    
    """
    
    kwargs['kind']='raster'
    return plot_events( events, **kwargs )

def plot_spectrogram( signal, **kwargs):
    """Plot spectrogram of signal.
    
    Parameters
    ----------
    signal : 1D array-like
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : FastSpectrogram options
        
    Returns
    -------
    spectrogram image
    
    """
    
    if not signal.ndim==1:
        raise ValueError('Invalid signal vector')
    
    ax = kwargs.pop( 'axes', None )
        
    if ax is None:
        ax = plt.gca()
    
    item = FastSpectrogram( signal, **kwargs )
    ax.add_image( item )
    plt.sci(item)
    
    plt.draw()
    
    return item


def add_static_scalebar(ax, hidex=True, hidey=True, **kwargs):
    """Add static scalebar to axes
    
    Parameters
    ----------
    ax : axis
    **kwargs : additional arguments passed to StaticScaleBar
    
    Returns
    -------
    StaticScaleBar
    
    """
    
    sb = StaticScaleBar(**kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    return sb

def add_axes_message(ax, **kwargs):
    """Add message artist to axes
    
    Parameters
    ----------
    ax : Axes
    **kwargs : additional arguments passed to AxesMessage
    
    Returns
    -------
    AxesMessage
    
    """
    
    item = AxesMessage(**kwargs)
    ax.add_artist(item)
    
    return item

def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """Add scalebars to axes
    
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    
    Parameters
    ----------
    ax : the axis to attach ticks to
    matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    hidex,hidey : if True, hide x-axis and y-axis of parent
    **kwargs : additional arguments passed to AnchoredScaleBars
    
    Returns
    -------
    AnchoredScaleBar
    
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    return sb


# WIP 
def signals_plot(*args,**kwargs):
    
    labels = kwargs.pop('labels',None)
    interaction = kwargs.pop('interaction',True)
    xlabel = kwargs.pop('xlabel','')
    ylabel = kwargs.pop('ylabel','')
    
    ax = kwargs.pop('axes',None)
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()
        
    p = plot_signals(*args, axes=ax, **kwargs)
    
    ax.set_xlabel( xlabel )
    yl = ax.set_ylabel( ylabel )
    
    ax.set_yticks( np.array([-1,0,1]) * p._spacing[1] )
    ax.spines["left"].set_bounds( -p._spacing[1], p._spacing[1] )
    
    yl.set_transform( matplotlib.transforms.blended_transform_factory( matplotlib.transforms.IdentityTransform(), ax.transData ) )
    yl.set_y( 0. )
    
    t = matplotlib.transforms.blended_transform_factory( ax.transAxes, ax.transData )
    
    labels = ['a',] * len(p)
    
    for idx in range(len(p)):
        ax.text( 1, idx*p._spacing[1], labels[idx], transform = t + matplotlib.transforms.Affine2D().translate(10,0) )
    
    ax.set_ylim( -p._spacing[1], len(p) * p._spacing[1] )
    plt.draw()
    
    if interaction:
        ax.scrollpanzoom = ScrollPanZoom(ax)
    
    #draw_scale_annotation
    
    return
    
def events_plot():
    # plot events with offset
    # set up ylim, ytick, yticklabels
    # set up ylabel, xlabel
    # set up zoom/pan
    pass

def segments_plot():
    pass

def raster_plot():
    pass

def noline_legend( leg ):
    '''
        Remove the lines and the frame from a legend and color each label with
        the corresponding color
    '''
    for item in leg.legendHandles:
            item.set_visible(False)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color( line.get_color() ) 
    
    return