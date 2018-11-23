"""
=================================================================
Artists (:mod:`fklab.plot.artists`)
=================================================================

.. currentmodule:: fklab.plot.artists

Custom matplotlib artists

.. autosummary::
    :toctree: generated/
    
    FastRaster
    FastLine
    FastSpectrogram
    StaticScaleBar
    AnchoredScaleBar
    AxesMessage
    PositionTimeStrip

"""

__all__ = ['FastRaster','FastLine','FastSpectrogram',
           'StaticScaleBar', 'AnchoredScaleBar', 'AxesMessage',
           'PositionTimeStrip']

import bisect
import numpy as np
import scipy.signal

from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.text import Text
import matplotlib.artist
import matplotlib.transforms
from matplotlib.offsetbox import AnchoredOffsetbox

import matplotlib.colors

import matplotlib.pyplot as plt

import fklab.segments
import fklab.events
import fklab.signals.multitaper

import scipy.interpolate

class FastSpectrogram(AxesImage):
    """
    Matplotlib image artist that represents a spectrogram.
    
    Parameters
    ----------
    x : 1d array like
        Signal for which to compute the spectrogram.
    fs : float, optional
        Sampling frequency of the signal
    start_time : float, optional
        Start time of the signal
    window_size : float, optional
        Size of the time window for the spectrogram.
    overlap : float, in range [0., 1.), optional
        Overlap of time windows.
    bandwidth : float
        Bandiwdth for multi-taper spectral analysis.
    power_scale : {'linear', 'db', 'freq'}
    freq_range : upper or [lower, upper]
        Frequency range of the spectrogram to display.
    colormap : Matplotlib colormap
    decimate : bool
        Whether to decimate to data before computing the spectrogram.
    resolution : float
        Scaling factor that reduces image resolution.
    
    """
    def __init__(self, x, fs=1000., start_time=0., window_size=1.,
                 overlap=0., bandwidth=10., power_scale='db',
                 freq_range=None, colormap=None, decimate=False,
                 resolution=1):
        
        self.set_data(x)
        self._fs = float(fs)
        self.set_start_time(start_time)
        self.set_window_size(window_size)
        self.set_overlap(overlap)
        self.set_bandwidth(bandwidth)
        self.set_power_scale(power_scale)
        self.set_freq_range(freq_range)
        self.set_decimate(decimate)
        self.set_resolution(resolution)
        
        super(FastSpectrogram, self).__init__(None, interpolation='none', clip_on=True, cmap=colormap)
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self,ax):
        super(FastSpectrogram, FastSpectrogram).axes.__set__(self,ax)
        self._prepare_data()
        
    def set_start_time(self, val):
        self._start_time = float(val)
    
    def set_window_size(self, val):
        val = float(val)
        if val<=0:
            raise ValueError('Window size should be larger than zero.')
        self._window_size = val
    
    def set_overlap(self, val):
        val = float(val)
        if val<0 or val>=1.:
            raise ValueError('Overlap should be equal or larger than 0 and smaller than 1.')
        self._overlap = val
    
    def set_bandwidth(self, val):
        val = float(val)
        if val<=0:
            raise ValueError('Bandwidth should be larger than 0.')
        self._bandwidth = val
    
    def set_power_scale(self,val):
        if not val in ('linear', 'db', 'freq'):
            raise ValueError("Power scale should be one of 'linear', 'db' or 'freq'.")
        self._power_scale = val
    
    def set_freq_range(self,val):
        if not val is None:
            val = np.array(val).ravel()
            if len(val)==1 and val[0]>0:
                val = [0., val[0]]
            elif len(val)>2 or val[-1]<=val[0] or val[0]<0 or val[0]>self._fs/2.0:
                raise ValueError('Invalid frequency range')
            
            val[1] = np.minimum(self._fs/2.0, val[1])
        else:
            val = [0., self._fs/2.0]
        
        self._freq_range = val
        
    def set_decimate(self,val):
        self._decimate=bool(val)
    
    def set_resolution(self,val):
        val = int(val)
        if val<1:
            raise ValuError('Invalid resolution')
        self._resolution = val
    
    def set_data(self, x):
        self._data = x
        
        # are these line necessary??
        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None
    
    def get_data(self):
        return self._data    

    def _prepare_data(self):
        
        if self.axes is None:
            return
        
        ax = self.axes
        xlow, xhigh = ax.get_xlim()
        
        # subdivide x range into bins
        b = fklab.segments.segment_split( [xlow, xhigh], size=self._window_size, overlap=self._overlap )
        
        # axes width in pixels
        w = np.diff( ax.transData.transform(ax.viewLim), axis=0 )[0,0]
        w = w/self._resolution
        
        # select bins
        if len(b)>w:
            b = b[ np.round( np.linspace(0,len(b),w,endpoint=False) ).astype(np.int64) ]
        
        # convert bin start times to indices
        b = np.round( (b[:,0] - self._start_time) * self._fs ).astype(np.int64)
        
        # convert window size to samples
        winsz = np.int(np.round( self._window_size * self._fs ))
        
        # remove invalid bins
        b = b[ np.logical_and( b>=0, (b + winsz)<=len(self._data) ) ]
        
        indices = np.arange(winsz)[:,None] + b[None,:]
        indices = indices.ravel() #work-around, since memmapped neuralynx
                                  #files do not support multidimensional indices yet
        
        data = self._data[ indices ].reshape( winsz, len(b) )
        
        ## CACHE DATA - needs refresh if window_size, xlim, pixel_width change
        
        decimate_factor = 1
        if self._decimate:
            decimate_factor = int(self._fs / (self._freq_range[1]*4.0) )
            if decimate_factor>1:
                data = scipy.signal.decimate( data, decimate_factor, axis=0 )
            elif decimate_factor<1:
                decimate_factor = 1
        
        ## CACHE DECIMATED DATA - needs refresh if DATA, freq_range, decimate change
        
        if len(b)==0 or len(data)==0:
            self._A = np.zeros((0,0))
            self._extent = [xlow, xhigh, self._freq_range[0], self._freq_range[1]]
        else:
            try:
                data,f,_,_ = fklab.signals.multitaper._mtspectrum_single( data, bandwidth=self._bandwidth, fs=self._fs/decimate_factor, fpass=self._freq_range);
                
                ## CACHE SPECTROGRAM - needs refresh if DECIMATED DATA, bandwidth changes
                ## CACHE OPTIONS/TAPERS - needs refresh if data.shape[0], bandwidth, fs change
                
                if self._power_scale=='db':
                    data = 20.*np.log10(data)
                elif self._power_scale=='freq':
                    data =  data * f[:,None]
                    
                self._A = np.flipud(data)
                
                self._extent = [self._start_time + b[0]/self._fs, self._start_time + b[-1]/self._fs + self._window_size, self._freq_range[0], self._freq_range[1] ]
                
                
            except ValueError, e:
                
                self._A = np.zeros((10,10),dtype=np.float64)
                self._A[ np.arange(10), np.arange(10) ] = 1.
                self._A[ np.arange(10), np.arange(10)[::-1] ] = 1.
                self._extent = [ self._start_time + b[0]/self._fs, self._start_time + b[-1]/self._fs + self._window_size, self._freq_range[0], self._freq_range[1] ]
            
        self.changed()
    
    def draw(self, renderer, *args, **kwargs):
        self._prepare_data()
        super(FastSpectrogram, self).draw(renderer, *args, **kwargs)    

class FastRaster(AxesImage):
    """
    Matplotlib image artist that represents an event raster plot.
    
    Parameters
    ----------
    x : 1d array like
        Vector of event times
    linewidth : float
        Line width in pixels
    foreground_color : Matplotlib color
    background_color : Matplotlib color
    foreground_alpha : float
    background_alpha : float
    
    """
    
    def __init__(self, x, linewidth=1, foreground_color='black', background_color='white', foreground_alpha=1., background_alpha=0.):
        
        self.set_data(x)
        self._raster_colors = np.array( [[1.,1.,1.,0.],[0.,0.,0.,1.]] ).reshape( (1,2,4) )
        
        self.set_foreground_color(foreground_color)
        self.set_foreground_alpha(foreground_alpha)
        self.set_background_color(background_color)
        self.set_background_alpha(background_alpha)
        self.set_linewidth(linewidth)
        
        super(FastRaster, self).__init__(None, interpolation='none', clip_on=True)
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self,ax):
        super(FastRaster, FastRaster).axes.__set__(self,ax)
        self._prepare_data()
    
    def set_linewidth(self,val):
        val = float(val)
        if val<0:
            raise ValueError("Line width should be equal or larger than 0.")
        self._linewidth = val
    
    def set_foreground_color(self,col):
        col = matplotlib.colors.colorConverter.to_rgb( col )
        self._raster_colors[0,1,0:3] = col
    
    def set_background_color(self,col):
        col = matplotlib.colors.colorConverter.to_rgb( col )
        self._raster_colors[0,0,0:3] = col
    
    def set_foreground_alpha(self,alpha):
        self._raster_colors[0,1,3] = float(alpha)
    
    def set_background_alpha(self,alpha):
        self._raster_colors[0,0,3] = float(alpha)
    
    def set_data(self, x):
        self._data = x
        
        # are these line necessary??
        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None

    def get_data(self):
        return self._data

    def _prepare_data(self):
        
        if self.axes is None:
            return
        
        ax = self.axes
        xlow, xhigh = ax.get_xlim()
        
        if self._linewidth>0:
            w = np.diff( ax.transData.transform(ax.viewLim), axis=0 )[0,0]
            w = w/self._linewidth
           
            bins = fklab.segments.segment_split( [xlow, xhigh], (xhigh-xlow)/w )
            data = fklab.events.event_bin( self._data, bins, kind='binary' )
        
            self._A = self._raster_colors[ :, data.ravel(), : ]
        else:
            self._A = self._raster_colors[:,0:1,:]
        
        self._extent = [xlow, xhigh, 0, 1]
        
        self.changed()

    def draw(self, renderer, *args, **kwargs):
        self._prepare_data()
        super(FastRaster, self).draw(renderer, *args, **kwargs)

class FastLine(Line2D):
    """
    Line class that supports automatic downsampling.
    
    Parameters
    ----------
    factor : float
        Number of samples per pixel
    random : bool
        Perform random downsampling of signal
    *args, **kwargs : Matplotlib Line2D arguments
    
    """
    def __init__(self,*args,**kwargs):
        self._full_xdata = None
        self._full_ydata = None
        self._factor = kwargs.pop('factor',1)
        self._random = kwargs.pop('random', False)
        super(FastLine,self).__init__(*args,**kwargs)
    
    def set_xdata(self, x):
        self._full_xdata = x
    
    def set_ydata(self, y):
        self._full_ydata = y
    
    def get_xdata(self,orig=True):
        if orig:
            return self._full_xdata
        
        return super(FastLine,self).get_xdata(orig)
    
    def get_ydata(self,orig=True):
        if orig:
            return self._full_ydata
        
        return super(FastLine,self).get_ydata(orig)
    
    def get_factor(self):
        return self._factor
    
    def set_factor(self,val):
        val = float(val)
        if val<=0:
            raise ValueError('Factor should be larger than zero.')
        self._factor = val
    
    def get_random(self):
        return self._random
    
    def set_random(self, val):
        val = bool(val)
        self._random=val
    
    def draw(self, renderer, *args, **kwargs):
        self._prepare_data()
        super(FastLine, self).draw(renderer, *args, **kwargs)
    
    def _prepare_data(self):
        
        ax = self.axes
        
        w = self._factor * np.diff( ax.transData.transform(ax.viewLim), axis=0 )[0,0]
        
        xlow, xhigh = ax.get_xlim()
        
        idxstart = np.maximum( 0, bisect.bisect_left( self._full_xdata, xlow ) )
        idxend = np.minimum( len(self._full_xdata), bisect.bisect_right( self._full_xdata, xhigh ) )
        
        #downsample
        n = idxend-idxstart+1
        if n>w:
            if self._random:
                idx = np.sort(np.random.random_integers(low=idxstart,high=idxend,size=w))
            else:
                f = int(n/w)
                idx = slice(idxstart,idxend,f)
        else:
            idx = slice(idxstart,idxend)
        
        self._xorig = self._full_xdata[idx]
        self._yorig = self._full_ydata[idx]
        
        self._invalidx = True
        self._invalidy = True
        self.stale = True

class StaticScaleBar(matplotlib.artist.Artist):
    """
    Fixed length scale bar with label that adapts the axes limits.
    
    Parameters
    ----------
    location : {'left', 'right', 'top', 'bottom'}
        The scale bar is plotted at the specified side of the axes.
    size : float or 2-element sequence
        Relative size and location of scale bar. 
    offset : float
        Offset of scale bar relative to axes.
    label : string
        Scale bar label. Any reference to {label} in the string will be
        replaced by the size of the scale bar in axes coordinates.
    label_offset : float
        Offset of label relative the scale bar.
    color : Matplotlib color
    linewidth : float
    lineprop : dict
        Extra properties for the scale bar line.
    textprop : dict
        Extra properties for the scale bar label.
    *args, **kwargs : extra arguments for Artist
    
    """
    
    def __init__(self, location='right', size=0.1, offset=0, label='{value}', label_offset=5, color='black', linewidth=2, lineprop=dict(), textprop=dict(), *args, **kwargs):
        
        self.scalebar = Line2D([],[],color=color,linewidth=linewidth,**lineprop)
        self.scalebar_label = Text(color=color,**textprop)
        
        self._scalebar_location = 'left'
        self._scalebar_size=[0,0]
        self._scalebar_label=''
        self._scalebar_label_offset=0
        self._scalebar_offset=0
        
        self.set_location(location)
        self.set_size(size)
        self.set_label(label)
        self.set_label_offset(label_offset)
        self.set_offset(offset)
        
        super(StaticScaleBar, self).__init__(*args, **kwargs)
        
    def _update_location(self):
        
        if self._scalebar_location in ('left','right'):
            if self._scalebar_location == 'left':
                self.scalebar.set_xdata([0.,0.])
                self.scalebar_label.set_x(0.)
                self.scalebar_label.set_ha('right')
            else:
                self.scalebar.set_xdata([1.,1.])
                self.scalebar_label.set_x(1.)
                self.scalebar_label.set_ha('left')
            
            self.scalebar.set_ydata(self._scalebar_size)
            self.scalebar_label.set_y(0.5*sum(self._scalebar_size))
            self.scalebar_label.set_rotation(90)
            self.scalebar_label.set_va('center')
            
        elif self._scalebar_location in ('top','bottom'):
            if self._scalebar_location == 'bottom':
                self.scalebar.set_ydata([0.,0.])
                self.scalebar_label.set_y(0.)
                self.scalebar_label.set_va('top')
            else:
                self.scalebar.set_ydata([1.,1.])
                self.scalebar_label.set_y(1.)
                self.scalebar_label.set_va('bottom')
            
            self.scalebar.set_xdata(self._scalebar_size)
            self.scalebar_label.set_x(0.5*sum(self._scalebar_size))
            self.scalebar_label.set_rotation(0)
            self.scalebar_label.set_ha('center')
        
        self._update_transforms()
    
    def _update_transforms(self):
        
        if not hasattr(self,'axes') or not self.axes:
            return
        
        if self._scalebar_location == 'left':
            _x,_y=-self._scalebar_offset,0
        elif self._scalebar_location == 'right':
            _x,_y=self._scalebar_offset,0
        elif self._scalebar_location == 'top':
            _x,_y=0,self._scalebar_offset
        elif self._scalebar_location == 'bottom':
            _x,_y=0,-self._scalebar_offset
        
        self.scalebar.set_transform( matplotlib.transforms.offset_copy(self.axes.transAxes,x=_x,y=_y, fig=self.axes.figure, units='points' ) )
        
        if self._scalebar_location == 'left':
            _x=_x -self._scalebar_label_offset
        elif self._scalebar_location == 'right':
            _x=_x+self._scalebar_label_offset
        elif self._scalebar_location == 'top':
            _y=_y+self._scalebar_label_offset
        elif self._scalebar_location == 'bottom':
            _y=_y-self._scalebar_label_offset
            
        self.scalebar_label.set_transform( matplotlib.transforms.offset_copy(self.axes.transAxes,x=_x,y=_y, fig=self.axes.figure, units='points' ))
    
    def set_figure(self, fig):
         matplotlib.artist.Artist.set_figure(self, fig)
         for c in (self.scalebar, self.scalebar_label):
             c.set_figure(fig)
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self,ax):
        super(StaticScaleBar, StaticScaleBar).axes.__set__(self,ax)
        self.scalebar.axes = ax
        self.scalebar_label.axes = ax
        self._update_transforms()
    
    def contains(self, mouseevent):
        for c in (self.scalebar, self.scalebar_label):
            a, b = c.contains(mouseevent)
            if a:
                return a, b
        return False, {}
    
    def draw(self, renderer, *args, **kwargs):
        
        #update label
        if self._scalebar_location in ('left','right'):
            ylow, yhigh = self.axes.get_ylim()
            val = (yhigh-ylow) * (self._scalebar_size[1]-self._scalebar_size[0])
        else:
            xlow, xhigh = self.axes.get_xlim()
            val = (xhigh-xlow) * (self._scalebar_size[1]-self._scalebar_size[0])

        label = self._scalebar_label.format(value=val)
        self.scalebar_label.set_text( label )
        
        self.scalebar.draw(renderer,*args,**kwargs)
        self.scalebar_label.draw(renderer,*args,**kwargs)

    def get_location(self):
        return self._scalebar_location
    
    def set_location(self,val):
        val = str(val)
        if not val in ('left','right','top','bottom'):
            raise ValueError('Invalid location')
        self._scalebar_location = val
        self._update_location()
    
    def get_size(self):
        return self._scalebar_size
    
    def set_size(self,val):
        if isinstance(val,(list,tuple)):
            val = [float(val[0]),float(val[-1])]
        else:
            val = [0., float(val)]
        
        if val[1]<=val[0]:
            raise ValueError('Invalid size')
            
        self._scalebar_size = val
        
        self._update_location()
    
    def get_label(self):
        return self._scalebar_label
    
    def set_label(self,val):
        val = str(val)
        self._scalebar_label = val
    
    def get_label_offset(self):
        return self._scalebar_label_offset
    
    def set_label_offset(self,val):
        val = float(val)
        self._scalebar_label_offset = val
        self._update_transforms()
    
    def get_offset(self):
        return self._scalebar_offset
    
    def set_offset(self,val):
        val = float(val)
        self._scalebar_offset=val
        self._update_transforms()

class AnchoredScaleBar(AnchoredOffsetbox):
    """
    Set of scale bars that match the size of the ticks of the plot.
    
    Draws a horizontal and/or vertical bar with the size in data coordinates
    of the give axes. A label will be drawn underneath (center-aligned).
    
    Parameters
    ----------
    transform : Matplotlib Transform
        The coordinate frame (typically axes.transData)
    sizex, sizey : float
        Width of x,y bar, in data units. 0 to omit.
    labelx, labely : string
        Labels for x,y bars; None to omit
    loc : string or integer
        Position in containing axes.
    pad, borderpad : float
        Padding, in fraction of the legend font size (or prop).
    sep : float
        Separation between labels and bars in points.
    prop : Matplotlib FontProperties
        Font property.
    **kwargs : additional arguments passed to base class constructor
    
    """
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
        
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        
        bars = AuxTransformBox(transform)
        
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none"))
        
        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)
        
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)

class AxesMessage(matplotlib.artist.Artist):
    """
    Display message in Matplotlib axes.
    
    Parameters
    ----------
    location : {'c', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'}
    alpha : float
    color : Matplotlib color
    textprop : dict
        Extra properties for message text.
    *args, **kwargs : extra arguments for Artist.
    
    Methods
    -------
    show(message, duration=0)
        Display a message for a given duration, after which it will fade
        out. If `duration` is 0, then the message will remain indefinitely.
    hide()
        Hide message.
    
    """
    
    def __init__(self, location='C', alpha=0.75, color='red', textprop={'fontsize':20, 'fontweight':'bold'}, *args, **kwargs):
        
        self.figure = None
        self._text_alpha = alpha
        self._text = Text(visible=False,color=color,alpha=alpha,zorder=100,**textprop)
        self.set_location( location )
        
        self._timer = None
        self._fadeout_timer = None
        self._fadeout_alpha = 0
        
        super(AxesMessage, self).__init__(*args, **kwargs)
        
    def set_figure(self, fig):
         matplotlib.artist.Artist.set_figure(self, fig)
         self._text.set_figure(fig)
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self, ax):
        super(AxesMessage, AxesMessage).axes.__set__(self,ax)
        self._text.axes = ax
        if ax is not None:
            self._text.set_transform( ax.transAxes )
            self.set_zorder(100)
    
    def draw(self, renderer, *args, **kwargs):
        self._text.draw(renderer,*args,**kwargs)
    
    def set_location(self, L):
        
        L = str(L).lower()
        
        if L =='c': #center
            halign = 'center'
            valign = 'center'
            x = 0.5
            y = 0.5
        
        if L in ['n','nw','ne']:
            valign = 'top'
            y = 1.
        
        if L in ['s','sw','se']:
            valign = 'bottom'
            y = 0.
        
        if L in ['w','e']:
            valign = 'center'
            y = 0.5
        
        if L in ['e','ne','se']:
            halign = 'right'
            x = 1.
        
        if L in ['w','sw','nw']:
            halign = 'left'
            x = 0.
        
        if L in ['n','s']:
            halign = 'center'
            x = 0.5
        
        #plt.setp( self._text, horizontalalignment=halign, verticalalignment=valign, x=x, y=y )
        self._text.set_horizontalalignment(halign)
        self._text.set_verticalalignment(valign)
        self._text.set_x(x)
        self._text.set_y(y)
        
        if not self.figure is None:
            self.figure.canvas.draw()
    
    def show(self, message, duration=0):
        #plt.setp( self._text, visible=True, alpha=self._text_alpha, text=message )
        self._text.set_visible(True)
        self._text.set_alpha(self._text_alpha)
        self._text.set_text(message)
        
        if not self.figure is None:
            self.figure.canvas.draw()
        
        if not self.figure is None and duration>0:
            self._fadeout_alpha = self._text_alpha
            self._timer = self.figure.canvas.new_timer( duration*1000 )
            self._timer.add_callback( self.hide )
            self._timer.single_shot = True
            self._timer.start()
        
    def hide(self):
        
        if not self._timer is None:
            self._timer.stop()
            self._timer = None
        
        if not self.figure is None and self._fadeout_alpha>0:
            self._fadeout_timer = self.figure.canvas.new_timer( 100 )
            self._fadeout_timer.add_callback( self._fadeout )
            self._fadeout_timer.start()
        else:
            self._text.set_visible(False)
            
            if not self.figure is None:
                self.figure.canvas.draw()
        
    def _fadeout(self):
        self._fadeout_alpha = self._fadeout_alpha * 0.75
        #plt.setp( self._text, alpha=self._fadeout_alpha )
        self._text.set_alpha(self._fadeout_alpha)
        
        if self._fadeout_alpha < 0.05:
            self._text.set_visible(False)
            self._fadeout_alpha = 0
            self._fadeout_timer.stop()
            self._fadeout_timer = None
        
        if not self.figure is None:
            self.figure.canvas.draw()

class PositionTimeStrip(matplotlib.artist.Artist):
    """Matplotlib artist that plots a time strip of 2D positions.
    
    Parameters
    ----------
    t : 1d array
        Vector of times.
    xy : (n,2) array
        Array of x,y locations.
    roi : (2,2) array-like
        Region of interest in space. Should be in form: [[xmin, xmax], [ymin, ymax]]
    n : int
        Number of plots in strip
    *args, **kwargs : extra arguments to Artist base class
    
    
    """
    
    def __init__(self, t, xy, roi, n, *args, **kwargs):
        
        self.figure = None
        self._t = t
        self._xy = xy # (n,2) array
        self._roi = roi # [ [xmin, xmax], [ymin, ymax] ]
        self._n = n # integer
        
        # construct artists for current position, future and past paths
        self._current = Line2D( [], [], marker='o', color='black', linestyle='none', markersize=10, clip_on=True )
        self._future = [ Line2D( [], [], marker='.', color='red', linestyle='none', clip_on=True ) for x in range(self._n+1) ]
        self._past = [ Line2D( [], [], marker='.', color='blue', linestyle='none', clip_on=True ) for x in range(self._n+1) ]
        
        # construct artists for background
        img, _, _ = np.histogram2d( xy[:,0], xy[:,1], bins=[100,100], range=roi, normed=True )
        img[img>0.] = 0.2
        
        self._all = [ AxesImage( None, data=img.T, cmap='gray_r', norm=matplotlib.colors.Normalize(vmin=0, vmax=1.) ) for  x in range(self._n+1) ]
        
        super(PositionTimeStrip, self).__init__(*args, **kwargs)
        
    def set_figure(self, fig):
        matplotlib.artist.Artist.set_figure(self, fig)
        
        for k in range(self._n+1):
            self._all[k].set_figure(fig)
            self._future[k].set_figure(fig)
            self._past[k].set_figure(fig)
        
        self._current.set_figure(fig)
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self, ax):
        super(PositionTimeStrip, PositionTimeStrip).axes.__set__(self,ax)
        
        for k in range(self._n+1):
            ax.add_artist( self._all[k] )
            ax.add_artist( self._future[k] )
            ax.add_artist( self._past[k] )

        ax.add_artist( self._current )
    
    def draw(self, renderer, *args, **kwargs):
        self._prepare_data()
        self._current.draw(renderer,*args,**kwargs)
        for k in range(self._n+1):
            self._future[k].draw(renderer,*args,**kwargs)
            self._past[k].draw(renderer,*args,**kwargs)
            self._all[k].draw(renderer,*args,**kwargs)
    
    def _prepare_data(self):
        
        ax = self.axes
        
        xlow, xhigh = ax.get_xlim()
        
        dt = (xhigh-xlow)/self._n
        
        t = np.arange( self._n + 1 ) * dt + np.round( xlow / dt ) * dt
        
        
        interp = scipy.interpolate.interp1d( self._t, self._xy, kind='nearest', axis=0, copy=False, bounds_error=False, fill_value=np.nan, assume_sorted=True)
        xy = interp(t)
        
        xy[:,0] = xy[:,0] * dt / (self._roi[0][1]-self._roi[0][0])
        xy[:,0] += t - 0.5*dt
        
        self._current.set_xdata( xy[:,0] )
        self._current.set_ydata( xy[:,1] )
        
        for k in range(self._n+1):
            
            xy = interp( t[k] + np.linspace(0.,5.,25) )
            xy[:,0] = xy[:,0] * dt / (self._roi[0][1]-self._roi[0][0])
            xy[:,0] += t[k] - 0.5*dt
        
            self._future[k].set_xdata( xy[:,0] )
            self._future[k].set_ydata( xy[:,1] )
            
            xy = interp( t[k] - np.linspace(0.,5.,25) )
            xy[:,0] = xy[:,0] * dt / (self._roi[0][1]-self._roi[0][0])
            xy[:,0] += t[k] - 0.5*dt
        
            self._past[k].set_xdata( xy[:,0] )
            self._past[k].set_ydata( xy[:,1] )
            
            self._all[k].set_extent( [ t[k]-0.5*dt, t[k]+0.5*dt, self._roi[1][0], self._roi[1][1] ] )
        
        
