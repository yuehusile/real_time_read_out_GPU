"""
=================================================================
Interaction (:mod:`fklab.plot.interaction`)
=================================================================

.. currentmodule:: fklab.plot.interaction

Utilities for interaction with matplotlib figures

.. autosummary::
    :toctree: generated/
    
    ScrollPanZoom
    interactive_figure
    create_square
    create_rectangle
    create_circle
    create_ellipse
    create_polyline
    create_polygon
    iEllipse
    iRectangle
    iPolyline
    iPolygon

"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.lines import Line2D

import itertools
import functools

from fklab.plot.artists import AxesMessage
    
__all__ = ['interactive_figure','iEllipse','iRectangle','iPolyline','iPolygon',
           'create_square', 'create_rectangle','create_circle','create_ellipse',
           'create_polyline','create_polygon', 'ScrollPanZoom']


class ScrollPanZoom:
    """
    Adds pan/zoom mouse interaction to axes.
    
    Zooming and panning is performed using the mouse wheel.
    To zoom, press Shift key. Use x and y keys to switch between horizontal
    and vertical zoom/scroll. 
    
    Parameters
    ----------
    ax : Axes
    scale : float
        Scaling step for zooming in/out.
    shift : float
        Offset step for panning.
        
    Methods
    -------
    enable()
        Enables panning/zooming
    disable()
        Disables panning/zooming
    
    """
    def __init__(self, ax, scale=1.5, shift=0.25):
        self._axes = ax
        self._scale = float(scale)
        self._shift = float(shift)
        self._callbacks = ()
        self._shift_is_held = False
        self._horizontal = True
        
        self._message = AxesMessage()
        ax.add_artist( self._message )
        
        self.enable()
        
    def enable(self):
        fig = self._axes.get_figure() # get the figure of interest
        self._callbacks = ( fig.canvas.mpl_connect('scroll_event', self.panzoom), 
                            fig.canvas.mpl_connect('key_press_event', self.on_key_press), 
                            fig.canvas.mpl_connect('key_release_event', self.on_key_release)
                        )
    
    def disable(self):
        fig = self._axes.get_figure() # get the figure of interest
        for c in self._callbacks:
            fig.canvas.mpl_disconnect(c)
        
        self._callbacks = ()
    
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self,value):
        self._scale = abs( float( value ) )
    
    @property
    def shift(self):
        return self._shift
    
    @shift.setter
    def shift(self,value):
        self._shift = float( value )

    def panzoom(self, event):
        
        if event.inaxes!=self._axes:
            return
        
        if self._horizontal:
            curlim = self._axes.get_xlim()
            cursor_loc = event.xdata
        else:
            curlim = self._axes.get_ylim()
            cursor_loc = event.ydata
        
        if self._shift_is_held:
            newlim = self._zoom( curlim, event.button, cursor_loc )
        else:
            newlim = self._pan( curlim, event.button, cursor_loc )
        
        if self._horizontal:
            self._axes.set_xlim( newlim )
        else:
            self._axes.set_ylim( newlim )
        
        self._axes.figure.canvas.draw()
    
    def _zoom(self, curlim, btn, x):
        scale_factor = self._scale
        if btn == 'down': # zoom in
            scale_factor = 1. / scale_factor
        
        new_width = (curlim[1] - curlim[0]) * scale_factor
        relx = (curlim[1] - x)/(curlim[1] - curlim[0])
        
        newxlim = ([x - new_width * (1-relx), x + new_width * (relx)])
        
        return newxlim
            
    def _pan(self, curlim, btn, x):
        shift = self._shift * (curlim[1]-curlim[0])
        
        if btn == 'up':
            shift = -shift
        
        newxlim = curlim[0] + shift, curlim[1] + shift
        
        return newxlim
    
    def on_key_press(self, event):
        if event.inaxes==self._axes and event.key == 'shift':
            self._shift_is_held = True
            self._message.show('zoom', duration=1)
    
    def on_key_release(self, event):
        if not event.inaxes==self._axes:
            return
            
        if event.key == 'shift':
            self._shift_is_held = False
            self._message.show('pan', duration=1)
        elif event.key == 'x':
            self._horizontal = True
            self._message.show('x-axis', duration=1)
        elif event.key == 'y':
            self._horizontal = False
            self._message.show('y-axis', duration=1)


def interactive_figure(figure=None):
    """Create interactive figure.
    
    Adds a mouse button press handler to the figure that forwards the 
    click event to the artist under the pointer. Responsive artists should
    implement a _interaction_start( event, extra ) method that returns an
    interaction context that is a tuple with handlers for move and release events.
    
    Parameters
    ----------
    figure : Matplotlib Figure, optional
        If no figure is provided, then a new figure is created.
    
    Returns
    -------
    figure
    
    """
    if figure is None:
        figure = plt.figure()
    
    cid_press = figure.canvas.mpl_connect('button_press_event',_on_press)

    figure._interactive = True
    figure._interactive_callback = cid_press
    
    return figure

def _hitlist(ax,event):
    L = []
    for c in itertools.chain( reversed(ax.lines), reversed(ax.patches) ):
        try:
            hascursor, info = c.contains(event)
            if hascursor:
                L.append( (c,info) )
        except:
            pass
    #sort by z-order
    L.sort(key=lambda x:x[0].zorder,reverse=True)
    return L

def _on_press(event):
    
    ax = event.inaxes
    if ax is None:
        return

    hit = _hitlist(ax,event)
    if not hit:
        return

    shape,extra = hit[0]
    
    try:
        context = shape._interaction_start(event,extra)
    except AttributeError:
        return
    
    if not context:
        return
    
    ax.figure._interactive_context = context
    
    #prepare blitting
    canvas = ax.figure.canvas
    shape.set_animated(True)
    canvas.draw()
    ax.figure._interactive_background = canvas.copy_from_bbox(ax.bbox)
    # now redraw just the rectangle
    ax.draw_artist(shape)
    # and blit just the redrawn area
    canvas.blit(ax.bbox)
    
    #then set up button_release and motion_notify events
    cid_release = ax.figure.canvas.mpl_connect('button_release_event',functools.partial(_on_release, shape=shape))
    cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', functools.partial(_on_move, shape=shape) )
    ax.figure._interactive_callback2 = (cid_release,cid_motion)

def _on_move(event,shape=None):
    
    canvas = shape.figure.canvas
    axes = shape.axes
    
    if axes.figure._interactive_context[0] is not None and event.xdata is not None:
        axes.figure._interactive_context[0](event)
    
    # restore the background region
    canvas.restore_region(axes.figure._interactive_background)

    # redraw just the current rectangle
    axes.draw_artist(shape)

    # blit just the redrawn area
    canvas.blit(axes.bbox)

def _on_release(event,shape=None):
    
    fig = shape.axes.figure
    
    if fig._interactive_context[1] is not None:
        fig._interactive_context[1](event)
    
    fig.canvas.mpl_disconnect(fig._interactive_callback2[1])
    fig.canvas.mpl_disconnect(fig._interactive_callback2[0])
    fig._interactive_callback2=None
    shape.set_animated(False)
    shape.axes.figure._interactive_background = None
    shape.figure.canvas.draw()

class iEllipse(Ellipse):
    """Interactive ellipse.
    
    Creates an ellipse that can be moved and resized through interaction
    handles. Rotation of the ellipse is done by pressing the Ctrl-key and
    moving the interaction handles.
    
    Note: the figure that contains the ellipse needs to support interaction
    (see `interactive_figure`)
    
    Parameters
    ----------
    center : 2-element sequence
    size : scalar or 2-element sequence
        Size of the two axes of the ellipse
    orientation : scalar
        Orientation of the ellipse in radians
    **kwargs : extra arguments for matplotlib.patches.Ellipse
    
    Attributes
    ----------
    center : [x, y]
    size : [width, height]
    orientation : ellipse orientation in radians
    
    """
    def __init__(self,center, size, orientation,**kwargs):
        self._interaction_handles = Line2D( [], [], markersize=8, color='k', marker='s', linestyle='' )
        
        kwargs['angle'] = orientation * 180 / np.pi
        size = np.array(size).ravel()
        kwargs['width'] = size[0]
        kwargs['height'] = size[-1]
        kwargs['xy'] = center
        
        super(iEllipse,self).__init__(**kwargs)

        self._update_interaction_handles()
        self._interactive = True
        
    @matplotlib.artist.Artist.axes.setter
    def axes(self, ax):
        super(iEllipse,iEllipse).axes.__set__(self,ax)

        self._interaction_handles.set_axes(ax)
        
        if ax is None:
            return
        
        ax._set_artist_props(self._interaction_handles)
        if self._interaction_handles.get_clip_path() is None:
            self._interaction_handles.set_clip_path(ax.patch)
        ax._update_line_limits(self._interaction_handles)
    
    def _update_interaction_handles(self):
        xc,yc = self.center
        w,h = self.width/2.0, self.height/2.0
        x = np.array([0,-w,0,w,w,w,0,-w,-w])
        y = np.array([0,-h,-h,-h,0,h,h,h,0])
        a = self.angle*np.pi/180
        self._interaction_handles.set_xdata( np.cos(a)*x - np.sin(a)*y + xc)
        self._interaction_handles.set_ydata( np.sin(a)*x + np.cos(a)*y + yc)
        self._interaction_handles.set_marker( (4,0,self.angle-45) )
    
    @property
    def orientation(self):
        return self.angle * np.pi / 180.
    
    @orientation.setter
    def orientation(self,value):
        self.angle = value*180./np.pi
    
    @property
    def size(self):
        return [self.width, self.height]
    
    @size.setter
    def size(self, value):
        value = np.array(value)
        if value.size<1:
            raise ValueError('Invalid size')
        self.width = value[0]
        self.height = value[-1]
    
    @property
    def interactive(self):
        return self._interactive
    
    @interactive.setter
    def interactive(self,value):
        self._interactive = bool(value)
        self._interaction_handles.set_visible(bool(value))
    
    def _interaction_start(self,event,extra):
        
        if not self._interactive:
            return False
        
        if not extra or int(extra['ind'][0])==0:
            #save current center and mouse cursor location relative to center
            self._origin = np.asarray(self.center)
            self._click = np.array([event.xdata, event.ydata])
            context = (self._itranslate, None)
        elif event.key=='control':
            #save current angle and mouse cursor location relative to center
            self._origin_angle = np.asarray(self.angle) * np.pi/180
            self._click_angle = np.arctan2( event.ydata - self.center[1], event.xdata - self.center[0] )
            context = (self._irotate, None)
        elif event.key!='shift':
            #save current center, current angle, and mouse cursor location relative to center
            self._resize_point = int(extra['ind'][0])
            self._origin = np.asarray(self.center)
            self._origin_angle = np.asarray(self.angle) * np.pi/180
            self._origin_size = self.width, self.height
            xm = np.cos(-self._origin_angle)*(event.xdata - self._origin[0]) - np.sin(-self._origin_angle)*(event.ydata - self._origin[1])
            ym = np.sin(-self._origin_angle)*(event.xdata - self._origin[0]) + np.cos(-self._origin_angle)*(event.ydata - self._origin[1])
            self._click = np.array([xm,ym])
            context = (self._iresize, None)
        else:
            return False
        
        return context
    
    def _itranslate(self,event):
        self.center = self._origin + np.array([event.xdata,event.ydata]) - self._click
        self._update_interaction_handles()
    
    def _irotate(self,event):
        a = np.arctan2( event.ydata - self.center[1], event.xdata - self.center[0] )
        self.angle = (self._origin_angle + a - self._click_angle)*180/np.pi
        self._update_interaction_handles()
    
    def _iresize(self,event):
        #un-rotate mouse cursor location
        xm = np.cos(-self._origin_angle)*(event.xdata - self._origin[0]) - np.sin(-self._origin_angle)*(event.ydata - self._origin[1])
        ym = np.sin(-self._origin_angle)*(event.xdata - self._origin[0]) + np.cos(-self._origin_angle)*(event.ydata - self._origin[1])
        
        dx = xm-self._click[0]
        dy = ym-self._click[1]
        
        dcx = 0
        dcy = 0
        
        if self._resize_point in [1,2,3]: #bottom
            h = np.maximum( 0, self._origin_size[1]-dy )
            self.height = h
            dcy = (self._origin_size[1]-h)/2.0
            
        if self._resize_point in [3,4,5]: #right
            w = np.maximum(0, self._origin_size[0]+dx)
            self.width = w
            dcx = (w - self._origin_size[0])/2.0
            
        if self._resize_point in [5,6,7]: #top
            h = np.maximum(0, self._origin_size[1]+dy)
            self.height = h
            dcy = (h - self._origin_size[1])/2.0
            
        if self._resize_point in [7,8,1]: #left
            w = np.maximum( 0, self._origin_size[0]-dx )
            self.width = w
            dcx = (self._origin_size[0]-w)/2.0
        
        self.center = ( self._origin[0] + np.cos(self._origin_angle)*dcx - np.sin(self._origin_angle)*dcy, self._origin[1] + np.sin(self._origin_angle)*dcx + np.cos(self._origin_angle)*dcy )
        
        self._update_interaction_handles()
    
    def contains(self,event):
        b,extra = self._interaction_handles.contains(event)
        if not b:
            b,extra = super(iEllipse,self).contains(event)
        
        return (b,extra)
    
    def set_animated(self,*args,**kwargs):
        super(iEllipse,self).set_animated(*args,**kwargs)
        self._interaction_handles.set_animated(*args,**kwargs)
    
    def draw(self,*args,**kwargs):
        super(iEllipse,self).draw(*args,**kwargs)
        self._interaction_handles.draw(*args,**kwargs)

class iRectangle(Rectangle):
    """Interactive rectangle.
    
    Creates an rectangle that can be moved and resized through interaction
    handles. Rotation of the rectangle is done by pressing the Ctrl-key and
    moving the interaction handles.
    
    Note: the figure that contains the rectangle needs to support interaction
    (see `interactive_figure`)
    
    Parameters
    ----------
    center : 2-element sequence
    size : scalar or 2-element sequence
       Width and height of the rectangle
    orientation : scalar
        Orientation of the rectangle in radians
    **kwargs : extra arguments for matplotlib.patches.Rectangle
    
    Attributes
    ----------
    center : [x, y]
    size : [width, height]
    orientation : rectangle orientation in radians
    
    """
    def __init__(self, center, size, orientation=0. ,**kwargs):
        self._interaction_handles = Line2D( [], [], markersize=8, color='k', marker='s', linestyle='' )
        
        kwargs['angle'] = orientation * 180 / np.pi
        
        size = np.array(size).ravel()
        kwargs['width'] = size[0]
        kwargs['height'] = size[-1]
        
        
        kwargs['xy'] = center - np.array([ 0.5*size[0]*np.cos(orientation) - 0.5*size[-1]*np.sin(orientation), 
                                           0.5*size[-1]*np.cos(orientation) + 0.5*size[0]*np.sin(orientation) ])
        
        super(iRectangle,self).__init__(**kwargs)
        
        self._update_interaction_handles()
        self._interactive = True
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self, ax):
        super(iRectangle,iRectangle).axes.__set__(self,ax)

        self._interaction_handles.set_axes(ax)
        
        if ax is None:
            return
        
        ax._set_artist_props(self._interaction_handles)
        if self._interaction_handles.get_clip_path() is None:
            self._interaction_handles.set_clip_path(ax.patch)
        ax._update_line_limits(self._interaction_handles)
    
    def _update_interaction_handles(self):
        xc,yc = self.center
        w,h = self.width/2.0, self.height/2.0
        x = np.array([0,-w,0,w,w,w,0,-w,-w])
        y = np.array([0,-h,-h,-h,0,h,h,h,0])
        a = self.orientation
        self._interaction_handles.set_xdata( np.cos(a)*x - np.sin(a)*y + xc)
        self._interaction_handles.set_ydata( np.sin(a)*x + np.cos(a)*y + yc)
        self._interaction_handles.set_marker( (4,0, (a*180./np.pi)-45) )
    
    @property
    def orientation(self):
        return self._angle * np.pi / 180.
    
    @orientation.setter
    def orientation(self,value):
        self._angle = value * 180. / np.pi
    
    @property
    def size(self):
        return [self.width, self.height]
    
    @size.setter
    def size(self, value):
        value = np.array(value)
        if value.size<1:
            raise ValueError('Invalid size')
        self.width = value[0]
        self.height = value[-1]
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self,value):
        self._height = value
        
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self,value):
        self._width = value
    
    @property
    def center(self):
        x0 = self.get_x()
        y0 = self.get_y()
        w = self.width
        h = self.height
        a = self.orientation
        xc = np.cos(a)*w/2.0 - np.sin(a)*h/2.0 + x0
        yc = np.sin(a)*w/2.0 + np.cos(a)*h/2.0 + y0
        return np.array([xc,yc])
    
    @center.setter
    def center(self,c):
        c = np.asarray(c)
        w = self.width
        h = self.height
        a = self.orientation
        x0 = c[0] - np.cos(a)*w/2.0 + np.sin(a)*h/2.0
        y0 = c[1] - np.sin(a)*w/2.0 - np.cos(a)*h/2.0
        self.set_x(x0)
        self.set_y(y0)
    
    @property
    def interactive(self):
        return self._interactive
    
    @interactive.setter
    def interactive(self,value):
        self._interactive = bool(value)
        self._interaction_handles.set_visible(bool(value))
    
    def _interaction_start(self,event,extra):
        
        if not self._interactive:
            return False
        
        if not extra or int(extra['ind'][0])==0:
            #save current center and mouse cursor location relative to center
            self._origin = np.asarray(self.center)
            self._click = np.array([event.xdata, event.ydata])
            context = (self._itranslate, None)
        elif event.key=='control':
            #save current angle and mouse cursor location relative to center
            self._origin = np.asarray(self.center)
            self._origin_angle = np.asarray(self.angle) * np.pi/180
            self._click_angle = np.arctan2( event.ydata - self.center[1], event.xdata - self.center[0] )
            context = (self._irotate, None)
        elif event.key!='shift':
            #save current center, current angle, and mouse cursor location relative to center
            self._resize_point = int(extra['ind'][0])
            self._origin = np.asarray(self.center)
            self._origin_angle = np.asarray(self.angle) * np.pi/180
            self._origin_size = self.width, self.height
            xm = np.cos(-self._origin_angle)*(event.xdata - self._origin[0]) - np.sin(-self._origin_angle)*(event.ydata - self._origin[1])
            ym = np.sin(-self._origin_angle)*(event.xdata - self._origin[0]) + np.cos(-self._origin_angle)*(event.ydata - self._origin[1])
            self._click = np.array([xm,ym])
            context = (self._iresize, None)
        return context
    
    def _itranslate(self,event):
        self.center = self._origin + np.array([event.xdata,event.ydata]) - self._click
        self._update_interaction_handles()
    
    def _irotate(self,event):
        a = np.arctan2( event.ydata - self._origin[1], event.xdata - self._origin[0] )
        self.angle = (self._origin_angle + a - self._click_angle)*180/np.pi
        self.center = self._origin #because we want rotation around center and not lower-left corner
        self._update_interaction_handles()
    
    def _iresize(self,event):
        #un-rotate mouse cursor location
        xm = np.cos(-self._origin_angle)*(event.xdata - self._origin[0]) - np.sin(-self._origin_angle)*(event.ydata - self._origin[1])
        ym = np.sin(-self._origin_angle)*(event.xdata - self._origin[0]) + np.cos(-self._origin_angle)*(event.ydata - self._origin[1])
        
        dx = xm-self._click[0]
        dy = ym-self._click[1]
        
        dcx = 0
        dcy = 0
        
        if self._resize_point in [1,2,3]: #bottom
            h = np.maximum( 0, self._origin_size[1]-dy )
            self.height = h
            dcy = (self._origin_size[1]-h)/2.0
            
        if self._resize_point in [3,4,5]: #right
            w = np.maximum(0, self._origin_size[0]+dx)
            self.width = w
            dcx = (w - self._origin_size[0])/2.0
            
        if self._resize_point in [5,6,7]: #top
            h = np.maximum(0, self._origin_size[1]+dy)
            self.height = h
            dcy = (h - self._origin_size[1])/2.0
            
        if self._resize_point in [7,8,1]: #left
            w = np.maximum( 0, self._origin_size[0]-dx )
            self.width = w
            dcx = (self._origin_size[0]-w)/2.0
        
        self.center = ( self._origin[0] + np.cos(self._origin_angle)*dcx - np.sin(self._origin_angle)*dcy, self._origin[1] + np.sin(self._origin_angle)*dcx + np.cos(self._origin_angle)*dcy )
        
        self._update_interaction_handles()
    
    def contains(self,event):
        b,extra = self._interaction_handles.contains(event)
        if not b:
            b,extra = super(iRectangle,self).contains(event)
        
        return (b,extra)
    
    def set_animated(self,*args,**kwargs):
        super(iRectangle,self).set_animated(*args,**kwargs)
        self._interaction_handles.set_animated(*args,**kwargs)
    
    def draw(self,*args,**kwargs):
        super(iRectangle,self).draw(*args,**kwargs)
        self._interaction_handles.draw(*args,**kwargs)

class iPolyline(Line2D):
    """Interactive polyline.
    
    Creates a polyline that can be manipulated through interaction
    handles. Rotation of the polyline is done by pressing the Ctrl-key
    and moving the interaction handles.
    
    Note: the figure that contains the polyline needs to support interaction
    (see `interactive_figure`)
    
    Parameters
    ----------
    xy : (N,2) array
    **kwargs : extra arguments for matplotlib.lines.Line2D
    
    """
    def __init__(self,xy,**kwargs):
        self._interaction_handles = Line2D( [], [], markersize=8, color='k', marker='s', linestyle='' )
        
        super(iPolyline,self).__init__(xy[:,0], xy[:,1],**kwargs)
        
        self._update_interaction_handles()
        self._interactive = True
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self, ax):
        super(iPolyline,iPolyline).axes.__set__(self,ax)
        self._interaction_handles.axes = ax
        
        if ax is None:
            return
        
        ax._set_artist_props(self._interaction_handles)
        if self._interaction_handles.get_clip_path() is None:
            self._interaction_handles.set_clip_path(ax.patch)
        ax._update_line_limits(self._interaction_handles)
    
    def _update_interaction_handles(self):
        self._interaction_handles.set_xdata( self.get_xdata() )
        self._interaction_handles.set_ydata( self.get_ydata() )
    
    @property
    def interactive(self):
        return self._interactive
    
    @interactive.setter
    def interactive(self,value):
        self._interactive = bool(value)
        self._interaction_handles.set_visible(bool(value))
    
    def _interaction_start(self,event,extra):
        
        if not self._interactive or not extra:
            return False
        
        if extra.has_key('ind'):
            self._click_index = int(extra['ind'][0])
            if event.button==1 and event.key=='control': #left + control: rotate
                self._center = [np.mean( np.asarray(self.get_xdata()) ), np.mean(np.asarray(self.get_ydata())) ]
                self._click_angle = np.arctan2( event.ydata - self._center[1], event.xdata - self._center[0] )
                context = (self._rotate_curve,None)
            elif event.button==1: #left
                context = (self._move_point,None)
            elif event.button==3: #right
                context = (None,self._remove_point)
        elif extra.has_key('seg'):
            if event.button==1: #left: insert point
                self._click_index = int(extra['seg'][0])+1
                x = np.asarray(self.get_xdata(),dtype=np.float64)
                y = np.asarray(self.get_ydata(),dtype=np.float64)
                x = np.insert(x,self._click_index,event.xdata)
                y = np.insert(y,self._click_index,event.ydata)
                self.set_xdata(x)
                self.set_ydata(y)
                self._update_interaction_handles()
                context = (self._move_point,None)
            elif event.button==3: #right: move curve
                self._cursor_point = (event.xdata, event.ydata)
                context = (self._move_curve,None)
        else:
            return False
        
        return context
    
    def _rotate_curve(self,event):
        x = np.asarray(self.get_xdata(),dtype=np.float64) - self._center[0]
        y = np.asarray(self.get_ydata(),dtype=np.float64) - self._center[1]
        a = np.arctan2( event.ydata - self._center[1], event.xdata - self._center[0] ) - self._click_angle
        self._click_angle += a
        self.set_xdata( np.cos(a)*x - np.sin(a)*y + self._center[0] )
        self.set_ydata( np.sin(a)*x + np.cos(a)*y + self._center[1] )
        self._update_interaction_handles()
    
    def _move_curve(self,event):
        dx = event.xdata - self._cursor_point[0]
        dy = event.ydata - self._cursor_point[1]
        x = np.asarray(self.get_xdata(),dtype=np.float64)
        y = np.asarray(self.get_ydata(),dtype=np.float64)
        x+=dx
        y+=dy
        self.set_xdata(x)
        self.set_ydata(y)
        self._cursor_point=(event.xdata,event.ydata)
        self._update_interaction_handles()
    
    def _move_point(self,event):
        x = np.asarray(self.get_xdata(),dtype=np.float64)
        y = np.asarray(self.get_ydata(),dtype=np.float64)
        x[self._click_index] = event.xdata
        y[self._click_index] = event.ydata
        self.set_xdata(x)
        self.set_ydata(y)
        self._update_interaction_handles()
    
    def _remove_point(self,event):
        x = np.asarray(self.get_xdata(),dtype=np.float64)
        y = np.asarray(self.get_ydata(),dtype=np.float64)
        x = np.delete(x,self._click_index)
        y = np.delete(y,self._click_index)
        self.set_xdata(x)
        self.set_ydata(y)
        self._update_interaction_handles()
    
    def contains(self,event):
        b,extra = self._interaction_handles.contains(event)
        if not b:
            b,extra = super(iPolyline,self).contains(event)
            if extra.has_key('ind'):
                extra['seg']=extra['ind']
                extra.pop('ind')
        
        return (b,extra)
    
    def set_animated(self,*args,**kwargs):
        super(iPolyline,self).set_animated(*args,**kwargs)
        self._interaction_handles.set_animated(*args,**kwargs)
    
    def draw(self,*args,**kwargs):
        super(iPolyline,self).draw(*args,**kwargs)
        self._interaction_handles.draw(*args,**kwargs)
    
class iPolygon(Polygon):
    """Interactive polygon.
    
    Creates a polygon that can be manipulated through interaction
    handles. Rotation of the polygon is done by pressing the Ctrl-key
    and moving the interaction handles.
    
    Note: the figure that contains the polygon needs to support interaction
    (see `interactive_figure`)
    
    Parameters
    ----------
    xy : (N,2) array
    highlight : bool
    highlight_color : matplotlib color spec
    **kwargs : extra arguments for matplotlib.patches.Polygon
    
    """
    def __init__(self,*args,**kwargs):
        self._highlight = bool( kwargs.pop('highlight', False) )
        self._highlight_color = kwargs.pop('highlight_color', 'red')
        self._handle_color = kwargs.get('color', 'black')
        
        self._interaction_handles = Line2D( [], [], markersize=8, color=self._handle_color, marker='s', linestyle='' )
        self._interaction_handles2 = Line2D( [], [], color=self._handle_color, linestyle='-' )
        
        super(iPolygon,self).__init__(*args,edgecolor=self._handle_color,**kwargs)
        
        self._update_interaction_handles()
        self._interactive = True
    
    def get_highlight(self):
        return self._highlight
    
    def set_highlight(self,val=True):
        val = bool(val)
        self._highlight = val
        col = self._highlight_color if val else self._handle_color
        self._interaction_handles.set_color(col)
        self._interaction_handles2.set_color(col)
        self.set_edgecolor(col)
    
    def toggle_highlight(self):
        self.set_highlight( not self._highlight )
    
    @matplotlib.artist.Artist.axes.setter
    def axes(self, ax):
        super(iPolygon,iPolygon).axes.__set__(self,ax)

        self._interaction_handles.set_axes(ax)
        
        if ax is None:
            return
        
        ax._set_artist_props(self._interaction_handles)
        if self._interaction_handles.get_clip_path() is None:
            self._interaction_handles.set_clip_path(ax.patch)
        ax._update_line_limits(self._interaction_handles)
        
        self._interaction_handles2.set_axes(ax)
        ax._set_artist_props(self._interaction_handles2)
        if self._interaction_handles2.get_clip_path() is None:
            self._interaction_handles2.set_clip_path(ax.patch)
        ax._update_line_limits(self._interaction_handles2)
    
    def _update_interaction_handles(self):
        xy = self.get_xy()
        self._interaction_handles.set_xdata( xy[:-1,0] )
        self._interaction_handles.set_ydata( xy[:-1,1] )
        self._interaction_handles2.set_xdata( xy[:,0] )
        self._interaction_handles2.set_ydata( xy[:,1] )
    
    @property
    def interactive(self):
        return self._interactive
    
    @interactive.setter
    def interactive(self,value):
        self._interactive = bool(value)
        self._interaction_handles.set_visible(bool(value))
        self._interaction_handles2.set_visible(bool(value))
    
    def _interaction_start(self,event,extra):
        if not self._interactive:
            return False
        
        if extra.has_key('ind'):
            self._click_index = int(extra['ind'][0])
            if event.button==1 and event.key=='control': #left + control: rotate
                self._center = np.mean( np.asarray(self.get_xy()), axis=0,keepdims=True )
                self._click_angle = np.arctan2( event.ydata - self._center[0,1], event.xdata - self._center[0,0] )
                context = (self._rotate_curve,None)
            elif event.button==1: #left: movepoint
                context = (self._move_point,None)
            elif event.button==3: #right: remove point
                context = (None,self._remove_point)
        elif extra.has_key('seg'):
            if event.button==1: #left: insert point
                self._click_index = int(extra['seg'][0])+1
                xy = np.array(self.get_xy(),dtype=np.float64)
                xy = np.insert(xy,self._click_index,np.array([[event.xdata,event.ydata]]),axis=0)
                self.set_xy(xy)
                self._update_interaction_handles()
                context = (self._move_point,None)
            elif event.button==3: #right: move curve
                self._cursor_point = (event.xdata, event.ydata)
                context = (self._move_curve,None)
        else:
            self._cursor_point = (event.xdata, event.ydata)
            context = (self._move_curve,None)
        
        return context
    
    def _rotate_curve(self,event):
        xy = np.array(self.get_xy(),dtype=np.float64)
        a = np.arctan2( event.ydata - self._center[0,1], event.xdata - self._center[0,0] ) - self._click_angle
        self._click_angle += a
        xy = np.dot( xy - self._center, np.array( [[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]] ).T ) + self._center
        self.set_xy(xy)
        self._update_interaction_handles()
    
    def _move_curve(self,event):
        dx = event.xdata - self._cursor_point[0]
        dy = event.ydata - self._cursor_point[1]
        xy = np.array(self.get_xy(),dtype=np.float64)
        xy[:,0]+=dx
        xy[:,1]+=dy
        self.set_xy(xy)
        self._cursor_point=(event.xdata,event.ydata)
        self._update_interaction_handles()
        
    def _move_point(self,event):
        xy = np.array(self.get_xy(),dtype=np.float64)[:-1,:]
        xy[self._click_index,:] = np.array([event.xdata,event.ydata])
        self.set_xy(xy)
        self._update_interaction_handles()
    
    def _remove_point(self,event):
        xy = np.array(self.get_xy(),dtype=np.float64)[:-1,:]
        xy = np.delete(xy,self._click_index,axis=0)
        self.set_xy(xy)
        self._update_interaction_handles()
    
    def contains(self,event):
        b,extra = self._interaction_handles.contains(event)
        if not b:
            b,extra = self._interaction_handles2.contains(event)
            if b:
                extra['seg']=extra['ind']
                extra.pop('ind')
            else:
                b,extra = super(iPolygon,self).contains(event)
        
        return (b,extra)
    
    def set_animated(self,*args,**kwargs):
        super(iPolygon,self).set_animated(*args,**kwargs)
        self._interaction_handles.set_animated(*args,**kwargs)
        self._interaction_handles2.set_animated(*args,**kwargs)
        
    def draw(self,*args,**kwargs):
        super(iPolygon,self).draw(*args,**kwargs)
        self._interaction_handles.draw(*args,**kwargs)
        self._interaction_handles2.draw(*args,**kwargs)

def _method_axes( points, pointer ):
    
    if len(points) == 0 or (len(points)==1 and pointer is None):
        center = [np.NaN, np.NaN]
        width = 0.
        height = 0.
        angle = 0.
    elif len(points)==1:
        center = 0.5*( points[0] + pointer )
        width = np.sqrt( np.sum( (pointer-points[0])**2 ) )
        height = 0.
        v = pointer-points[0]
        angle = np.arctan2( v[1], v[0] )
    else:
        v = points[1]-points[0]
        angle = np.arctan2( v[1], v[0] )
        xp = np.sqrt( np.sum( v**2 ) )
        if len(points)==2:
            if pointer is None:
                yp = 0.
            else:
                yp = np.sin(-angle)*(pointer[0]-points[0,0]) + np.cos(-angle)*(pointer[1]-points[0,1])
        else:
            yp = np.sin(-angle)*(points[2,0]-points[0,0]) + np.cos(-angle)*(points[2,1]-points[0,1])
        
        center = (points[0]+points[1])/2.0
        width = xp
        height = yp*2.0
    
    return center, [width, height], angle 

def _method_diagonal( points, pointer ):
    
    angle = 0.
    
    if len(points) == 0 or (len(points)==1 and pointer is None):
        center = [np.NaN, np.NaN]
        width = 0.
        height = 0.
    elif len(points) == 1:
        center = 0.5*( points[0] + pointer )
        width = np.abs( pointer[0] - points[0,0] )
        height = np.abs( pointer[1] - points[0,1] )
    else:
        center = 0.5*( points[0] + points[1] )
        width = np.abs( points[1,0] - points[0,0] )
        height = np.abs( points[1,1] - points[0,1] )
    
    return center, [width, height], angle

def _method_center_size( points, pointer ):

    if len(points)==0:
        center = [np.NaN, np.NaN]
        width = height = 0.
        angle = 0.
    elif len(points)==1:
        center = points[0]
        if pointer is None:
            width = height = 0.
            angle = 0.
        else:
            width = height = 2. * np.sqrt( np.sum( (pointer-points[0])**2 ) )
            angle = np.arctan2( pointer[1]-points[0,1], pointer[0]-points[0,0] )
    else:
        center = points[0]
        width = height = 2. * np.sqrt( np.sum( (points[1]-points[0])**2 ) )
        angle = np.arctan2( points[1,1]-points[0,1], points[1,0]-points[0,0] )
    
    return center, width, angle

def _method_center_axes( points, pointer ):
    
    if len(points)==0:
        center = [np.NaN, np.NaN]
        width = height = 0.
        angle = 0.
    elif len(points)==1:
        center = points[0]
        if pointer is None:
            width = height = 0.
            angle = 0.
        else:
            width = 2. * np.sqrt( np.sum( (pointer-points[0])**2 ) )
            height = 0.
            angle = np.arctan2( pointer[1]-points[0,1], pointer[0]-points[0,0] )
    else:
        center = points[0]
        width = 2. * np.sqrt( np.sum( (points[1]-points[0])**2 ) )
        angle = np.arctan2( points[1,1]-points[0,1], points[1,0]-points[0,0] )
        
        if len(points)==2:
            if pointer is None: height = 0.
            else:
                height = np.sin(-angle)*(pointer[0]-center[0]) + np.cos(-angle)*(pointer[1]-center[1])
        else:
            height = np.sin(-angle)*(points[2,0]-center[0]) + np.cos(-angle)*(points[2,1]-center[1])
        
        height = height * 2.
    
    return center, [width, height], angle
    
boxed_shape_method = {
    'center-size' : (2, _method_center_size, "Set center and size."),
    'diagonal' : (2, _method_diagonal, "Draw diagonal."),
    'axes' : (3, _method_axes, "Set left-right, and height."),
    'center-axes' : (3, _method_center_axes, "Set center, width and height."),
    }

def create_boxed_shape(ax, method='center-size', shape='rectangle'):
    """User-drawn box-like shape.
    
    Parameters
    ----------
    ax : matplotlib axes
    method : {'center-size', 'diagonal', 'axes', 'center-axis'}
        Method for defining the box-like shape
    shape : {'rectangle', 'ellipse'}
    
    Returns
    -------
    center : [x, y]
    size : [width, height]
    orientation : in radians
    
    """
    
    _method = boxed_shape_method[method][1]
    msg_string = boxed_shape_method[method][2]
    N = boxed_shape_method[method][0]
    
    hL = Line2D([],[],marker='o',linestyle='-')
    ax.add_line(hL)
    hL.set_animated(True)
    
    if shape=='rectangle':
        hShape = Rectangle( [np.nan,np.nan], 0, 0, facecolor='b', alpha=0.3)
    else:
        hShape = Ellipse([np.nan,np.nan],0,0, linestyle=None, facecolor='b',alpha=0.3)

    ax.add_patch(hShape)
    hShape.set_animated(True)
    
    msg = AxesMessage()
    ax.add_artist( msg )
    msg.show( msg_string )
    
    def update_shape(ax,data):
        return True
    
    def cleanup_shape(ax):
        ax.lines.remove(hL)
        ax.patches.remove(hShape)
        return True
    
    def update_animation(ax,data,pointer):
        ctr, sz, angle = _method( data, pointer )
        
        sz = np.array(sz).ravel()
        
        if shape=='rectangle':
            hShape.set_xy( ctr - np.array([ 0.5*sz[0]*np.cos(angle) - 0.5*sz[-1]*np.sin(angle), 0.5*sz[-1]*np.cos(angle) + 0.5*sz[0]*np.sin(angle) ]) )
            hShape._angle = angle*180/np.pi
            hShape.set_width(sz[0])
            hShape.set_height(sz[-1])
        else:
            hShape.center = ctr
            hShape.width = sz[0]
            hShape.height = sz[-1]
            hShape.angle = angle * 180 / np.pi
        
        x = [
            ctr[0] - 0.5*sz[0] * np.cos( angle ),
            ctr[0] + 0.5*sz[0] * np.cos( angle ),
            np.NaN,
            ctr[0] - 0.5*sz[-1] * np.cos( angle + 0.5*np.pi ),
            ctr[0] + 0.5*sz[-1] * np.cos( angle + 0.5*np.pi),
            ]
        
        y = [
            ctr[1] - 0.5*sz[0] * np.sin( angle ),
            ctr[1] + 0.5*sz[0] * np.sin( angle ),
            np.NaN,
            ctr[1] - 0.5*sz[-1] * np.sin( angle + 0.5*np.pi),
            ctr[1] + 0.5*sz[-1] * np.sin( angle + 0.5*np.pi),
            ]
        
        hL.set_xdata( x )
        hL.set_ydata( y )
        
        return [hL, hShape]
    
    b = BlockingCreateShape(ax)
    data = b( n=N, update_shape=update_shape, cleanup_shape=cleanup_shape, update_animation=update_animation)
    
    msg.hide()
    msg.remove()
    
    if data is not None:
        data = _method(data,None)
    
    return data

def create_circle(ax):
    """User-drawn circle.
    
    Parameters
    ----------
    ax : matplotlib axes
    
    Returns
    -------
    center : [x, y]
    diameter
    orientation : in radians
    
    """
    return create_boxed_shape(ax, method='center-size', shape='ellipse')

def create_ellipse(ax, method='center-axes'):
    """User-drawn ellipse.
    
    Parameters
    ----------
    ax : matplotlib axes
    method : {'center-size', 'diagonal', 'axes', 'center-axis'}
        Method for defining ellipse
        
    Returns
    -------
    center : [x, y]
    size : [width, height]
    orientation : in radians
    
    """
    
    return create_boxed_shape(ax, method, shape='ellipse')

def create_square(ax):
    """User-drawn square.
    
    Parameters
    ----------
    ax : matplotlib axes
    
    Returns
    -------
    center : [x, y]
    size
    orientation : in radians
    
    """
    
    return create_boxed_shape(ax, method='center-size', shape='rectangle')

def create_rectangle(ax, method='center-axes'):
    """User-drawn rectangle.
    
    Parameters
    ----------
    ax : matplotlib axes
    method : {'center-size', 'diagonal', 'axes', 'center-axis'}
        Method for defining rectangle
        
    Returns
    -------
    center : [x, y]
    size : [width, height]
    orientation : in radians
    
    """
    
    return create_boxed_shape(ax, method, shape='rectangle')

def create_polyline(ax):
    """User-drawn polyline.
    
    Parameters
    ----------
    ax : matplotlib axes
        
    Returns
    -------
    nodes : (N,2) array
    
    """
    
    hRubber = Line2D([],[],linestyle='dotted')
    ax.add_line(hRubber)
    hRubber.set_animated(True)
    
    hL = Line2D([],[],marker='o',linestyle='-')
    ax.add_line(hL)
    
    msg = AxesMessage()
    ax.add_artist( msg )
    msg.show( "Draw polyline.\nLeft button = draw node.\nRight button = stop.\nEsc = cancel" )
    
    def update_shape(ax,data):
        hL.set_xdata(data[:,0])
        hL.set_ydata(data[:,1])
        return True
    
    def cleanup_shape(ax):
        ax.lines.remove(hRubber)
        ax.lines.remove(hL)
        return True
    
    def update_animation(ax,data,pointer):
        if len(data)>0:
            hRubber.set_xdata([data[-1,0],pointer[0]])
            hRubber.set_ydata([data[-1,1],pointer[1]])
        else:
            hRubber.set_xdata([])
            hRubber.set_ydata([])
            
        return [hRubber]
        
    b = BlockingCreateShape(ax)
    data = b( n=0, update_shape=update_shape, cleanup_shape=cleanup_shape, update_animation=update_animation)
    
    msg.hide()
    msg.remove()
    
    return data

def create_polygon(ax):
    """User-drawn polygon.
    
    Parameters
    ----------
    ax : matplotlib axes
        
    Returns
    -------
    nodes : (N,2) array
    
    """
    
    hRubber = Line2D([],[],linestyle='dotted')
    ax.add_line(hRubber)
    hRubber.set_animated(True)
    
    hL = Line2D([],[],marker='o',linestyle='-')
    ax.add_line(hL)
    
    hP = Polygon( [[np.nan,np.nan]], facecolor='b', alpha=0.3)
    ax.add_patch(hP)
    hP.set_animated(True)
    
    msg = AxesMessage()
    ax.add_artist( msg )
    msg.show( "Draw polygon.\nLeft button = draw node.\nRight button = stop.\nEsc = cancel" )
    
    def update_shape(ax,data):
        hL.set_xdata(data[:,0])
        hL.set_ydata(data[:,1])
        return True
    
    def cleanup_shape(ax):
        ax.lines.remove(hRubber)
        ax.lines.remove(hL)
        ax.patches.remove(hP)
        return True
    
    def update_animation(ax,data,pointer):
        if len(data)>0:
            hRubber.set_xdata([data[-1,0],pointer[0]])
            hRubber.set_ydata([data[-1,1],pointer[1]])
        else:
            hRubber.set_xdata([])
            hRubber.set_ydata([])
        
        if len(data)>2:
            hP.set_xy( np.append( data, pointer.reshape((1,2)), axis=0 ) )
        else:
            hP.set_xy( [[np.nan,np.nan]] )
        
        return [hRubber,hP]
        
    b = BlockingCreateShape(ax)
    data = b( n=0, update_shape=update_shape, cleanup_shape=cleanup_shape, update_animation=update_animation)
    
    msg.hide()
    msg.remove()
    
    return data


from matplotlib.blocking_input import BlockingInput

class BlockingCreateShape(BlockingInput):

    def __init__(self, ax):
        
        self._axes = ax
        fig = ax.figure
        if not fig:
            raise ValueError()
        
        BlockingInput.__init__(self, fig=fig,
                               eventslist=('button_press_event',
                                           'key_press_event'))
       
    def post_event(self):
        """
        This will be called to process events
        """
        assert len(self.events) > 0, "No events yet"

        if self.events[-1].name == 'key_press_event':
            self.key_event()
        else:
            self.mouse_event()

    def mouse_event(self):
        '''Process a mouse click event'''
        
        event = self.events[-1]
        
        if event.inaxes != self._axes:
            BlockingInput.pop(self, -1) #remove event
            return
        
        button = event.button
        key = event.key
        
        if button == 1:
            if key == 'control':
                self.mouse_event_pop(event)
            elif key is None:
                self.mouse_event_add(event)
        elif button == 3 and self._manual_stop:
            self.mouse_event_stop(event)
        elif button == 2:
            self.mouse_event_cancel(event)
        else:
            BlockingInput.pop(self, -1) #remove event

    def key_event(self):
        '''
        Process a key click event.  This maps certain keys to appropriate
        mouse click events.
        '''

        event = self.events[-1]
        if event.key is None:
            # at least in mac os X gtk backend some key returns None.
            return

        key = event.key.lower()
        
        if key in ['x']:
            self.mouse_event_pop(event)
        elif key in ['z'] and self._manual_stop:
            self.mouse_event_stop(event)
        elif key in ['escape']:
            # on windows XP and wxAgg, the enter key doesn't seem to register
            self.mouse_event_cancel(event)
        elif key == ' ':
            self.mouse_event_add(event)
        else:
            BlockingInput.pop(self, -1) #remove event

    def mouse_event_add(self, event):
        self._data = np.append( self._data, [[event.xdata,event.ydata]], axis=0 )
        #data_updated callback
        if not self._update_shape( self._axes, self._data ):
            self.fig.canvas.stop_event_loop()
        else:
            self._axes.figure.canvas.draw()
            self._background = self._axes.figure.canvas.copy_from_bbox(self._axes.bbox)
            # and blit just the redrawn area
            #self._axes.figure.canvas.blit(self._axes.bbox)
            self.mouse_event_move(event)
    
    def mouse_event_stop(self,event):
        BlockingInput.pop(self, -1)
        self.fig.canvas.stop_event_loop()
    
    def mouse_event_cancel(self, event):
        self._data = None
        self.mouse_event_stop(event)
    
    def mouse_event_pop(self, event):
        
        # Remove this last event
        BlockingInput.pop(self, -1)

        # Now remove any existing clicks if possible
        if len(self.events) > 0:
            self.pop(event, -1)
            self._update_shape(self._axes,self._data)
            self._axes.figure.canvas.draw()
            self._background = self._axes.figure.canvas.copy_from_bbox(self._axes.bbox)
            # and blit just the redrawn area
            #self._axes.figure.canvas.blit(self._axes.bbox)
            self.mouse_event_move(event)

    def mouse_event_move(self, event):
        #return self._move_pointer( self._axes, self._data, np.array([[event.xdata,event.ydata]]) )
        
        if event.xdata is None or event.ydata is None:
            pointer = None
        else:
            pointer = np.array([event.xdata,event.ydata])
        
        hAnim = self._update_animation( self._axes, self._data, pointer )
        
        # restore the background region
        self._axes.figure.canvas.restore_region(self._background)
        # redraw just the current rectangle
        if hAnim:
            for k in hAnim:
                self._axes.draw_artist(k)
        # blit just the redrawn area
        self._axes.figure.canvas.blit(self._axes.bbox)
        
    def pop_click(self, event, index=-1):
        self._data = np.delete(self._data,-1,axis=0)

    def pop(self, event, index=-1):
        self.pop_click(event, index)
        BlockingInput.pop(self, index)

    def cleanup(self, event=None):
        # clean up figure
        self._cleanup_shape(self._axes)
        # Call base class to remove callbacks
        BlockingInput.cleanup(self)

    def __call__(self, n=0, update_shape = None, cleanup_shape = None, update_animation=None):
        """
        Blocking call to retrieve n coordinate pairs through mouse
        clicks.
        """
        if n==0:
            self._manual_stop = True
        else:
            self._manual_stop = False
        
        self._update_shape = update_shape
        self._cleanup_shape = cleanup_shape
        self._update_animation = update_animation
        
        self._data = np.zeros( (0,2) )
        
        #prepare blitting
        canvas = self._axes.figure.canvas
        canvas.draw()
        self._background = canvas.copy_from_bbox(self._axes.bbox)
        # and blit just the redrawn area
        canvas.blit(self._axes.bbox)
        
        cid = self._axes.figure.canvas.mpl_connect('motion_notify_event',self.mouse_event_move)
        
        try:
            #BlockingInput.__call__(self, n=n, timeout=0)
            
            self.n = n
            
            self.events = []
            self.callbacks = []
            
            # connect the events to the on_event function call
            for n in self.eventslist:
                self.callbacks.append(
                    self.fig.canvas.mpl_connect(n, self.on_event))

            try:
                # Start event loop
                self.fig.canvas.start_event_loop(timeout=0)
            finally:  # Run even on exception like ctrl-c
                # Disconnect the callbacks
                self.cleanup()

            
        finally:
            self._axes.figure.canvas.mpl_disconnect(cid)
            self._axes.figure.canvas.draw()
        
        return self._data


