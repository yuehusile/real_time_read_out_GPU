import sys
import math

from collections import OrderedDict
import itertools

from PyQt5 import QtGui, QtCore, QtWidgets #, QtOpenGL

import numpy as np

from fklab.geometry.shapes import _sample_spline
import fklab.geometry.shapes

class Handle( QtWidgets.QGraphicsItem ):
    def __init__(self, movefunc=None, clickfun=None, extra = None, x=0, y=0, size=6.0, color=[100,100,255], parent=0 ):
        QtWidgets.QGraphicsItem.__init__( self, parent=parent )
        
        self.setAcceptHoverEvents(True)
        
        self._move_function = movefunc
        self._click_function = clickfun
        self._extra = extra
        
        self._x = float(x)
        self._y = float(y)
        
        self._size = size
        
        self._brush = QtGui.QBrush( QtGui.QColor( *color ) )
        self._pen = QtGui.QPen( QtCore.Qt.NoPen )
        self._pen.setWidthF( 0 )
        
        self._active_pen = self._pen
        self._active_brush = self._brush
        
        self._dragging = False
    
    def setLocation(self, x, y ):
        self.prepareGeometryChange()
        self._x = float(x)
        self._y = float(y)
    
    def hoverEnterEvent(self,event):
        self._active_brush = QtGui.QBrush( self._brush )
        c = self._active_brush.color()
        c.setHsv( (c.hue() + 60) % 360, c.saturation(), c.value() )
        self._active_brush.setColor( c )
        self.update()
    
    def hoverLeaveEvent(self,event):
        self._active_brush = self._brush
        self.update()
    
    def mousePressEvent(self,event):
        #save modifier state
        self._modifiers = event.modifiers()
        self._clicked_pos = event.pos()
        self._clicked_scene_pos = event.scenePos()
        
        #only for left button events, ignore others
        if event.button() != QtCore.Qt.LeftButton:
            return
        
        self._dragging = True
        
        #grab mouse
        self.grabMouse()
    
    def mouseMoveEvent(self,event):
        
        if not self._dragging:
            return
        
        #tell owner that handle has moved
        pos = event.pos()
        scene_pos = event.scenePos()
        
        if self._move_function is not None:
            self._move_function( self._clicked_scene_pos, self._clicked_pos, scene_pos, pos, self._modifiers, self._extra )
    
    def mouseReleaseEvent(self,event):
        self._modifiers = None
        self._clicked_pos = None
        self._clicked_scene_pos = None
        
        if self._click_function is not None:
            self._click_function( event, self._extra )
        
        if self._dragging:
            self._dragging = False
            
        #ungrab mouse
        self.ungrabMouse()
    
    def _view(self):
        assert (not self.scene() is None)
            
        return self.scene().views()[0]
    
    def shape(self):
        view = self._view()
        dt = self.deviceTransform(view.viewportTransform())
        
        if dt is None:
            path = QtGui.QPainterPath()
            path.addRect( QtCore.QRectF( -0.5,-0.5, 1, 1) )
            return path
        
        #handle center in device coordinates
        p = dt.map( QtCore.QPointF(self._x, self._y ) ) 
        
        #normalized vectors in device coordinates
        vx = dt.map( QtCore.QPointF( self._x + 1, self._y ) ) - p
        vy = dt.map( QtCore.QPointF( self._x, self._y + 1 ) ) - p
        
        vx = vx / math.sqrt( vx.x()**2 + vx.y()**2 )
        vy = vy / math.sqrt( vy.x()**2 + vy.y()**2 )
        
        #construct handle in device coordinates
        coords = []
        coords.append( p + 0.5*self._size*(-vx-vy) )
        coords.append( p + 0.5*self._size*(vx-vy) )
        coords.append( p + 0.5*self._size*(vx+vy) )
        coords.append( p + 0.5*self._size*(-vx+vy) )
        coords = QtGui.QPolygonF( coords )
        
        #transform back to local coordinates
        dti = dt.inverted()[0]
        path = QtGui.QPainterPath()
        path.addPolygon( dti.map( coords ) )
        
        return path
        
    def boundingRect(self):
        return self.shape().boundingRect()
        
    def paint(self, painter, options, widget ):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush( self._active_brush )
        painter.setPen( self._active_pen )
        painter.drawPath( self.shape() )

class HandleContainer( QtWidgets.QGraphicsItemGroup ):
    def __init__(self, parent):
        
        QtGui.QGraphicsItemGroup.__init__(self, parent=parent)
        
        self.setVisible(False)
        self.setFiltersChildEvents(True)
        
        self._handles = set()
        
    def addHandle( self, handle ):
        
        assert( isinstance(handle, Handle) )
        if not handle in self._handles:
            self._handles.add( handle )
            self.scene().addItem( handle )
    
    def removeHandle( self, handle ):
        
        assert( isinstance( handle, Handle ) )
        
        if handle in self._handles:
            self.scene().removeItem( handle )
            self._handles.remove( handle )
    
    def sceneEventFilter(self, obj, event ):
        return True


class Resizer( QtWidgets.QGraphicsItemGroup ):
    def __init__(self, parent):
        
        QtGui.QGraphicsItemGroup.__init__(self, parent=parent)
        self.setHandlesChildEvents(False)
        self.setVisible(False)
        
        self._create_handles()
        self.updateHandles()
    
    def updateHandles(self):
        
        b = self.parentItem().extent()
        top, bottom, left, right = b.top(), b.bottom(), b.left(), b.right()
        xc, yc = 0.5*(left+right), 0.5*(top+bottom)
        
        self._topleft.setLocation( left, top )
        self._topright.setLocation( right, top )
        self._bottomright.setLocation( right, bottom )
        self._bottomleft.setLocation( left, bottom )
        self._top.setLocation( xc, top )
        self._right.setLocation( right, yc )
        self._bottom.setLocation( xc, bottom )
        self._left.setLocation( left, yc )
        
    def _create_handles(self):
        self._topleft = Handle( clickfun=self._click, movefunc = self._resize, extra = ['top','left'], parent=self)
        self._topright = Handle(clickfun=self._click,movefunc = self._resize, extra = ['top','right'], parent=self)
        self._bottomright = Handle(clickfun=self._click,movefunc = self._resize, extra = ['bottom','right'], parent=self)
        self._bottomleft = Handle(clickfun=self._click,movefunc = self._resize, extra = ['bottom','left'], parent=self)
        self._top = Handle(clickfun=self._click,movefunc = self._resize, extra = ['top',], parent=self)
        self._right = Handle(clickfun=self._click,movefunc = self._resize, extra = ['right',], parent=self)
        self._bottom = Handle(clickfun=self._click,movefunc = self._resize, extra = ['bottom',], parent=self)
        self._left = Handle(clickfun=self._click,movefunc = self._resize, extra = ['left'], parent=self)
    
    def _click(self, event, extra):
        if event.button() == QtCore.Qt.LeftButton:
            self.parentItem().notify_change()
    
    def _resize(self, clicked_scene_pos, clicked_pos, scene_pos, pos, modifiers, extra):
        
        #0. get extent of parent
        b = self.parentItem().extent()
        center = b.center()
        width, height = b.width(), b.height()
        
        #1. handle rotation
        if modifiers & QtCore.Qt.ControlModifier:
            xc, yc = center.x(), center.y()
            y = self.parentItem()._scale_transform.yScale()
            x = self.parentItem()._scale_transform.xScale()
            r = ( np.arctan2( y*(clicked_pos.y()-yc), x*(clicked_pos.x()-xc) ) - np.arctan2(y*(pos.y()-yc),x*(pos.x()-xc)) ) * 180/np.pi
            self.parentItem()._rotation_transform.setAngle( self.parentItem()._rotation_transform.angle() - r )
            self.parentItem().has_changed=True
            return
        
        #2. get translation
        pos = self.parentItem().pos()
        shift = center + pos
        
        #3. construct transformation matrix: scene to local without scaling
        tform = QtGui.QTransform()
        tform.rotate( -self.parentItem()._rotation_transform.angle() )
        tform.translate( -shift.x(), -shift.y() )
        
        #4. transform scene mouse position
        scene_pos = tform.map( scene_pos )
        
        #5. compute scaling factors
        xscale = self.parentItem()._scale_transform.xScale()
        yscale = self.parentItem()._scale_transform.yScale()
        w, h = width * xscale, height * yscale
        dx, dy = 0., 0.
        if modifiers & QtCore.Qt.ShiftModifier:
            if 'top' in extra:
                h = -2.*scene_pos.y()
            elif 'bottom' in extra:
                h = 2.*scene_pos.y()
            
            if 'left' in extra:
                w = -2.*scene_pos.x()
            elif 'right' in extra:
                w = 2.*scene_pos.x()
        else:
            if 'top' in extra:
                h = -scene_pos.y() + 0.5*h
                dy = 0.5*(scene_pos.y() + 0.5*height*yscale)
            elif 'bottom' in extra:
                h = scene_pos.y() + 0.5*h
                dy = 0.5*(scene_pos.y() - 0.5*height*yscale)
            
            if 'left' in extra:
                w = -scene_pos.x() + 0.5*w
                dx = 0.5*(scene_pos.x() + 0.5*width*xscale)
            elif 'right' in extra:
                w = scene_pos.x() + 0.5*w
                dx = 0.5*(scene_pos.x() - 0.5*width*xscale)
        
        #6. apply scaling and translation
        delta = tform.inverted()[0].map( QtCore.QPointF( dx, dy ) )
        self.parentItem().setPos( delta - center)
        
        xscale = w / width
        yscale = h / height
        self.parentItem()._scale_transform.setXScale( xscale )
        self.parentItem()._scale_transform.setYScale( yscale )
        
        self.parentItem().has_changed=True



class EditModeManager( QtCore.QObject ):
    
    finished = QtCore.pyqtSignal(bool)
    mode_changed = QtCore.pyqtSignal(str)
    instructions = QtCore.pyqtSignal(str)
    
    def __init__(self, target=None, modes={}):
        QtCore.QObject.__init__(self)
        
        self._target = target
        
        if not isinstance( modes, dict ):
            raise ValueError
        
        self._modes = OrderedDict( modes )
        
        self._active = False
        self._active_mode = None
        self._active_mode_object = None
        
        self._allowed_events = [6,7] + range(155,168)
    
    @property
    def isActive(self):
        return self._active
    
    @property
    def numModes(self):
        return len(self._modes)
    
    @property
    def currentMode(self):
        return self._active_mode
    
    def start(self, mode=None ):
        
        if self._active or len(self._modes)==0:
            return False
        
        self._active = True
        if mode is None:
            self._active_mode = self._modes.keys()[0]
        elif not mode in self._modes.keys():
            raise ValueError
        else:
            self._active_mode = mode
        
        self._active_mode_object = self._modes[self._active_mode]( self._target )
        self.instructions.emit( self._active_mode_object.instructions() )
        
        self._target.scene().installEventFilter( self )
        
        return True
    
    def stop(self, cancel=False):
        
        if not self._active:
            return
        
        self._target.scene().removeEventFilter( self )
        
        self._active_mode_object.stop()
        
        self._active_mode = None
        self._active = False
        
        self.instructions.emit( "" )
        
        self.finished.emit(not cancel)
    
    def cycleMode(self, mode=None):
        
        if not self._active:
            return
        
        if mode is None:
            idx = self._modes.keys().index( self._active_mode )
            idx = (idx+1) % len(self._modes)
            mode = self._modes.keys()[idx]
        elif not mode in self._modes.keys():
            raise ValueError
        
        self._active_mode_object.stop()
        
        self._active_mode = mode
        self._active_mode_object = self._modes[self._active_mode]( self._target )
        
        self.instructions.emit( self._active_mode_object.instructions() )
        
        self.mode_changed.emit( mode )
        
    def eventFilter(self, obj, event ):
        
        event_type = event.type()
        
        if obj is self._target.scene():
            
            if event_type == 6: #keyPressEvent
                
                key = event.key()
                
                if key == QtCore.Qt.Key_Q or key == QtCore.Qt.Key_Escape:
                    self.stop(key == QtCore.Qt.Key_Escape)
                    return True
                elif event.key() == QtCore.Qt.Key_Space:
                    self.cycleMode()
                    return True
        
        #delegate to edit mode eventFilter
        if event_type in self._allowed_events:
            self._active_mode_object.eventFilter( obj, event )
            return True
        else:
            return False

class EditMode( QtCore.QObject ):
    
    finished = QtCore.pyqtSignal()
    
    def __init__(self, target=None ):
        
        QtCore.QObject.__init__(self)
        
        self._target = target
        
    def eventFilter(self, obj, event):
        return False
        if obj is self._target.scene():
            
            event_type = event.type()
            
            if event_type == 6:
                
                if event.key() == QtCore.Qt.Key_Q:
                    self._graph.scene().finishEditShape()
                    return True
                elif event.key() == QtCore.Qt.Key_M:
                    self._target.switch_edit_mode()
                    return True
            
        return False
            
    
    def stop(self):
        self.finished.emit()
    
    def instructions(self):
        return "No instructions. Please figure it out yourself."
    
class GraphEditor( EditMode ):
    
    def __init__(self, target):
        
        EditMode.__init__(self, target)
        
        assert( isinstance( self._target, iGraph ) )
        
        self._graph = self._target
        
        self._container = QtGui.QGraphicsItemGroup( parent=self._graph )
        
        self._nodes = self._create_nodes()
        self._handles = self._create_handles()
        
        self._dragging = False
        self._drag_data = []
        
    def stop(self):
        
        #remove all handles
        self._clear_nodes_and_handles()
        
        #call base class implementation
        EditMode.stop(self)
    
    def _clear_nodes_and_handles(self):
        
        self._graph.scene().removeItem( self._container )
        self._nodes.clear()
        self._handles = []
    
    def _reset_nodes_and_handles(self):
        
        self._clear_nodes_and_handles()
        self._container = QtGui.QGraphicsItemGroup( parent=self._graph )
        self._nodes = self._create_nodes()
        self._handles = self._create_handles()
    
    def _create_nodes(self):
        
        nodes = self._graph._nodes
        
        h = {}
        for (n,(x,y)) in nodes:
            h[n] = Handle( x=x, y=y, extra = n, size=10.0, color=[255,0,0], parent=self._container )
        
        return h
    
    def _create_handles(self):
        
        edges = self._graph._edges
        
        h = []
        for idx,(start_node,points,end_node) in enumerate(edges):
            hh = []
            for k,(x,y) in enumerate(points):
                hh.append( Handle( x=x, y=y, extra = [idx,k], size=6.0, parent=self._container ) )
            h.append(hh)
        
        return h
    
    def eventFilter(self, obj, event):
        
        done = False
        skip = False
        
        if obj is self._graph.scene():
            
            event_type = event.type()
            
            if event_type<154 or event_type>158:
                skip = True
            
            if not done and not skip and self._dragging:
                if event_type == 155: #mouseMoveEvent
                    pos = self._graph.mapFromScene( event.scenePos() )
                    if self._drag_data[0] == 'node':
                        self._graph.moveNodeTo( self._drag_data[1], [pos.x(), pos.y()] ) 
                        self._nodes[self._drag_data[1]].setLocation( pos.x(), pos.y() )
                    else: #handle
                        self._graph.moveEdgePointTo( self._drag_data[1][0], self._drag_data[1][1], [pos.x(), pos.y()] )
                        self._handles[self._drag_data[1][0]][self._drag_data[1][1]].setLocation( pos.x(), pos.y() )
                elif event_type == 157 and event.button() == QtCore.Qt.LeftButton: #mouseReleaseEvent
                    self._dragging = False
                
                done = True
            
            if not done and not skip:
                #find first item under mouse
                items = self._graph.scene().items( event.scenePos() )
                item_type = None
                for z in items:
                    if z in self._nodes.values():
                        item_type = 'node'
                        item = z
                    elif z in itertools.chain( *self._handles ):
                        item_type = 'handle'
                        item = z
                    elif z is self._graph:
                        item_type = 'graph'
                        item = z
                    else:
                        continue
                    break
                
                if item_type is not None and event_type == 156: #mousePressEvent
                    
                    if item_type == 'node':
                        if event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier: #left: start drag
                            self._dragging = True
                            self._drag_data = ['node', item._extra]
                            
                        elif event.button() == QtCore.Qt.RightButton and not self._dragging and event.modifiers() == QtCore.Qt.NoModifier: #right: remove
                            self._graph.removeNodeByName( item._extra )
                            self._reset_nodes_and_handles()
                            
                        done = True
                    
                    elif item_type == 'handle':
                        if event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier: #left: start drag
                            self._dragging = True
                            self._drag_data = ['handle',item._extra]
                        
                        elif event.button() == QtCore.Qt.RightButton and not self._dragging and event.modifiers() == QtCore.Qt.NoModifier: #right: remove
                            self._graph.removeEdgePointByIndex( *item._extra )
                            self._reset_nodes_and_handles()
                        
                        done = True
                    
                    elif item_type == 'graph':
                        if event.button() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.NoModifier:
                            pos = self._graph.mapFromScene( event.scenePos() )
                            edge = self._graph.findNearestEdge( [pos.x(), pos.y()] )
                            self._graph.removeEdgeByIndex( edge[0] )
                            self._reset_nodes_and_handles()
                            
                        elif event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.ControlModifier:
                            pos = self._graph.mapFromScene( event.scenePos() )
                            self._graph.addEdgePointNear( [pos.x(), pos.y()] )
                            self._reset_nodes_and_handles()
                        
                        done = True
        
        return done
    
    def instructions(self):
        return "Edit graph. left-click-drag=move node or edge vertex; ctrl-left-click edge=add vertex; right-click=remove node, edge or vertex; q or esc=finish; space bar=extend mode."
    
class GraphExtender( EditMode ):
    
    def __init__(self,target):
        
        EditMode.__init__(self, target)
        
        assert( isinstance( self._target, iGraph ) )
        
        self._graph = self._target
        self._mode = 'new'
        self._edge = [None,[],None]
        
        self._graph.node_changed.connect( self._node_changed )
        
        #create node handles
        self._handles = self._create_node_handles()
        
        #create trace item and rubber item
        self._pen = create_pen( width=2, cosmetic=True, color=[0,0,0], style=1 )
        self._rubber_pen = create_pen( width=1, cosmetic=True, color=[255,0,0], style=2 )
        
        self._current_mouse_position = None
        
        self._trace_item = QtGui.QGraphicsPathItem(parent=self._graph)
        self._trace_item.setPen(self._pen)
        
        self._rubber_item = QtGui.QGraphicsPathItem(parent=self._graph)
        self._rubber_item.setPen(self._rubber_pen)
    
    def stop(self):
        
        #clear handles
        self._clear_node_handles()
        
        self._graph.scene().removeItem(self._trace_item )
        self._graph.scene().removeItem(self._rubber_item )
        
        #call base class implementation
        EditMode.stop(self)
    
    def _node_changed(self, action, node_name):
        if action=='add':
            node_name = str(node_name )
            xy = self._graph._get_node( node_name )
            self._handles.append( Handle( x=xy[0], y=xy[1], extra = node_name, size=10.0, color=[255,130,0], parent=self._graph ) )
        else:
            self._reset_node_handles()
    
    def update_trace(self):
        #TODO: include start node!!
        
        path = QtGui.QPainterPath()
        
        if self._edge[0] is None:
            pass
        elif isinstance(self._edge[0],str):
            xy = self._graph._get_node( self._edge[0] )
            path.moveTo( *xy )
        else:
            path.moveTo( *self._edge[0] )
        
        if len(self._edge[1])>0:
            for p in self._edge[1]:
                path.lineTo( *p )
            
        self._trace_item.setPath( path )
    
    def update_rubber(self, current_pos):
        
        path = QtGui.QPainterPath()
        
        npoints = len(self._edge[1])
        if npoints==0 and self._edge[0] is not None:
            if isinstance(self._edge[0], str):
                xy = self._graph._get_node( self._edge[0] )
            else:
                xy = self._edge[0]
            
            path.moveTo( *xy )
            path.lineTo( current_pos )
        elif npoints>0:
            path.moveTo( *self._edge[1][-1] )
            path.lineTo( current_pos )
        else:
            pass
        
        self._rubber_item.setPath( path )
    
    def eventFilter(self, obj, event):
         
        if obj is self._graph.scene():
             
            event_type = event.type()
            
            if event_type == 156: #mousePressEvent
                
                button = event.button()
                
                if self._mode=='new' and button == QtCore.Qt.LeftButton:
                    
                    items = self._graph.scene().items( event.scenePos() )
                    items = [ x for x in items if x in self._handles ]
                    if len(items)>0:
                        self._edge[0] = items[0]._extra
                    else:
                        pos = self._graph.mapFromScene( event.scenePos() )
                        self._edge[0] = [pos.x(), pos.y()]
                    
                    self._mode = 'add'
                    
                    return True
                    
                elif self._mode=='add' and button == QtCore.Qt.RightButton:
                    del self._edge[1][-1]
                    self.update_trace()
                    self.update_rubber( self._graph.mapFromScene( event.scenePos() ) )
                    return True
                
                elif self._mode =='add' and button == QtCore.Qt.LeftButton:
                    
                    items = self._graph.scene().items( event.scenePos() )
                    items = [ x for x in items if x in self._handles ]
                    if len(items)>0:
                        self._edge[2] = items[0]._extra
                        self._graph.addEdge(start = self._edge[0], points = self._edge[1], end = self._edge[2])
                        self._mode = 'new'
                        self._edge = [None, [], None]
                        self.update_trace()
                        self.update_rubber( None )
                    else:
                        pos = self._graph.mapFromScene( event.scenePos() )
                        self._edge[1].append( [pos.x(),pos.y()] )
                        self.update_trace()
                        self.update_rubber( self._graph.mapFromScene( event.scenePos() ) )
                    return True
                
                return False
            
            elif event_type == 155: #mouse move
                if self._mode=='add':
                    self.update_rubber( self._graph.mapFromScene( event.scenePos() ) )
                    return False
                
                return False
            
            elif event_type == 6: #key press
                key = event.key()
                
                if self._mode == 'add' and key == QtCore.Qt.Key_F:
                    #do we have at least one point?
                    if len( self._edge[1] )==0:
                        return True
                    self._graph.addEdge(start = self._edge[0], points = self._edge[1], end = self._edge[2])
                    self._mode = 'new'
                    self._edge = [None, [], None]
                    self.update_trace()
                    self.update_rubber( None )
                    return True
                
                elif self._mode == 'add' and key == QtCore.Qt.Key_Escape:
                    self._mode = 'new'
                    self._edge = [None, [], None]
                    self.update_trace()
                    self.update_rubber( None )
                    return True
                    
        return False
    
    def _clear_node_handles(self):
        for k in self._handles:
            self._graph.scene().removeItem( k )
        self._handles = []
            
    def _reset_node_handles(self):
        self._clear_node_handles()
        
        self._handles = self._create_node_handles()
    
    def _create_node_handles(self):
        
        nodes = self._graph._nodes
        
        h = []
        for (n,(x,y)) in nodes:
            h.append( Handle( x=x, y=y, extra = n, size=10.0, color=[255,130,0], parent=self._graph ) )
        
        return h
    
    def instructions(self):
        return "Extend graph. left-click=add new edge vertex; f=finish edge, start new one; q or esc=finish; space bar=edit mode."
    
class ShapeEditor( EditMode ):
    
    def __init__(self, target):
        
        EditMode.__init__(self, target)
        
        assert( isinstance(self._target,iShape) )
        
        self._shape = self._target
        
        self._container = QtGui.QGraphicsItemGroup( parent=self._shape )
        
        self._handles = self._create_handles( len(self._shape._vertices) )
        self.updateHandles()
        
        self._dragging = False
        self._drag_data = []
    
    def stop(self):
        
        #clear handles
        self._shape.scene().removeItem( self._container )
        
        #call base class implementation
        EditMode.stop(self)
    
    def updateHandles(self):
        
        nhandles = len(self._handles)
        nvertices = len(self._shape._vertices)
        
        if nhandles>nvertices:
            #remove handles
            for k in range(nvertices,nhandles):
                self._shape.scene().removeItem( self._handles[k] )
            self._handles = self._handles[:nvertices]
        elif nvertices>nhandles:
            self._handles = self._handles + self._create_handles( nvertices-nhandles, nhandles )
        
        vertices = self._shape._vertices
        
        for index,k in enumerate( self._handles ):
            k.setLocation( vertices[index,0], vertices[index,1] )
    
    def _create_handles(self,n,start_index=0):
        
        h = []
        for k in range(n):
            h.append( Handle( extra = start_index, parent=self._container ) )
            start_index += 1
        
        return h
    
    def eventFilter(self, obj, event):
        
        done = False
        skip = False
        
        if obj is self._shape.scene():
            
            event_type = event.type()
            
            if event_type == 6: #keyPressEvent
                key = event.key()
                
                if key == QtCore.Qt.Key_C:
                    self._shape.closed = not self._shape.closed
                elif key == QtCore.Qt.Key_S:
                    try:
                        self._shape.spline = not self._shape.spline
                    except:
                        pass
                
                return True
            
            if event_type<154 or event_type>158:
                skip = True
            
            if not done and not skip and self._dragging:
                if event_type == 155: #mouseMoveEvent
                    pos = self._shape.mapFromScene( event.scenePos() )
                    self._shape.moveNodeTo( self._drag_data, [pos.x(), pos.y()] )
                    self._handles[self._drag_data].setLocation( pos.x(), pos.y() )
                elif event_type == 157 and event.button() == QtCore.Qt.LeftButton: #mouseReleaseEvent
                    self._dragging = False
                
                done = True
            
            if not done and not skip:
                #find first item under mouse
                items = self._shape.scene().items( event.scenePos() )
                item_type = None
                for z in items:
                    if z in self._handles:
                        item_type = 'node'
                        item = z
                    elif z is self._shape:
                        item_type = 'shape'
                        item = z
                    else:
                        continue
                    break
                
                if item_type is not None and event_type == 156: #mousePressEvent
                    
                    if item_type == 'node':
                        if event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier: #left: start drag
                            self._dragging = True
                            self._drag_data = item._extra
                            
                        elif event.button() == QtCore.Qt.RightButton and not self._dragging and event.modifiers() == QtCore.Qt.NoModifier: #right: remove
                            self._shape.removeNodeByIndex( item._extra )
                            self.updateHandles()
                        
                        done = True
                    
                    elif item_type == 'shape':
                        if event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.ControlModifier:
                            pos = self._shape.mapFromScene( event.scenePos() )
                            self._shape.addNodeNear( [pos.x(), pos.y()] )
                            self.updateHandles()
                        
                        done = True
        
        return done
    
    def instructions(self):
        return "Edit polyline. left-drag=move vertex; left-double-click edge=create vertex; right-click=remove vertex; c=toggle close; s=toggle spline; q or esc=finish."

    
class CustomPath( QtWidgets.QGraphicsPathItem ):
    def paint(self,painter,*args):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        QtWidgets.QGraphicsPathItem.paint( self, painter, *args )

class iGraphicsItem( QtWidgets.QGraphicsObject ):
    
    _type = 'graphicsitem'
    
    changed = QtCore.pyqtSignal(int)
    instructions = QtCore.pyqtSignal(str)
    
    def notify_change(self, changed=None):
        if changed is None:
            changed = self._has_changed
        
        if changed:
            self.changed.emit(self._id)
            
            self.has_changed=False
    
    def itemChange(self, change, value ):
        if (change==QtGui.QGraphicsItem.ItemPositionChange or
            change==QtGui.QGraphicsItem.ItemTransformHasChanged or
            change==QtGui.QGraphicsItem.ItemRotationHasChanged or
            change==QtGui.QGraphicsItem.ItemScaleHasChanged or
            change==QtGui.QGraphicsItem.ItemTransformOriginPointHasChanged):
                
            self._has_changed=True
        
        return QtGui.QGraphicsObject.itemChange(self, change, value)
    
    def __init__(self, position=(0.0,0.0), size=(1.0,1.0), rotation=0.0, parent=None, data={}, edit_modes = {}, editable=True, resizable=True, movable=True, identifier=0 ):
        
        QtGui.QGraphicsObject.__init__(self, parent=parent )
        
        self._id = int(identifier)
        self._has_changed = False
        
        #set up transformations
        self._rotation_transform = QtGui.QGraphicsRotation()
        self._rotation_transform.setOrigin( QtGui.QVector3D(0,0,0) )
        self._rotation_transform.setAngle( rotation )
        
        self._scale_transform = QtGui.QGraphicsScale()
        self._scale_transform.setXScale( size[0] )
        self._scale_transform.setYScale( size[1] )
        self._scale_transform.setOrigin( QtGui.QVector3D(0,0,0) )
        
        self._path = None
        
        #create pens
        self._pen = create_pen(width=3, cosmetic=True, color=[0,0,130] )
        
        self._highlight_pen = create_pen(width=5, cosmetic=True, color=[50,200,50])
        self._select_pen = create_pen(width=3, cosmetic=True, color=[170,80,240])
        
        self.setTransformations( [self._rotation_transform, self._scale_transform] )
        
        self.setPos( *position )
        
        #create edit mode manager
        self._editmodemanager = EditModeManager( target=self, modes=edit_modes )
        self._editmodemanager.finished.connect( self.finished_editing )
        self._editmodemanager.instructions.connect( self.instructions )
        
        self._edit_finish_callback = None
        
        self._selected = False
        self._data = data
        
        self._editable = editable
        self._resizable = resizable
        self.movable = movable
        
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, True)
        
        self.setAcceptHoverEvents(True)
        
        self._has_changed = False
        
        #create child path
        self._path = CustomPath( parent=self )
        
        self._path.setPen(self._pen)
        self._path.setBrush( QtGui.QBrush( QtCore.Qt.NoBrush ) )
        
        #create resizer object
        self._resizer = Resizer( self )
    
    @property
    def identifier(self):
        return self._id
    
    @property
    def has_changed(self):
        return self._has_changed
    
    @has_changed.setter
    def has_changed(self,val):
        self._has_changed = bool(val)
    
    @property
    def canEdit(self):
        return self._editmodemanager.numModes>0
    
    @property
    def movable(self):
        return self._movable
    
    @movable.setter
    def movable(self,value):
        self._movable = bool(value)
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, self._movable)
    
    @property
    def selected(self):
        return self._selected
    
    @property
    def resizable(self):
        return self._resizable
    
    @resizable.setter
    def resizable(self, value):
        self._resizable = bool(value)
        if self._resizable:
            self._resizer.setVisible(self._selected)
        else:
            self._resizer.setVisible(False)
    
    @property
    def editable(self):
        return self._editable
    
    @editable.setter
    def editable(self,value):
        self._editable = bool(value)
        
        if not self._editable:
            self.stopEdit()
    
    def getData(self):
        return self._data
    
    def todict(self):
        d = OrderedDict()
        d['type'] = self._type
        d['position'] = [self.pos().x(),self.pos().y()]
        d['size'] = list(self.size)
        d['rotation'] = self._rotation_transform.angle()
        d['data'] = self._data
        return d
    
    def hoverEnterEvent(self,event):
        self._path.setPen( self._highlight_pen )
    
    def hoverLeaveEvent(self,event):
        if self._selected:
            self._path.setPen( self._select_pen )
        else:
            self._path.setPen( self._pen )
    
    @property
    def size(self):
        return (self._scale_transform.xScale(),self._scale_transform.yScale())
    
    @size.setter
    def size(self,value):
        self._scale_transform.setXScale( value[0] )
        self._scale_transform.setYScale( value[1] )
    
    @property
    def angle(self):
        return self._rotation_transform.angle()
    
    @angle.setter
    def angle(self,value):
        self._rotation_transform.setAngle( float(value) )
        
    @property
    def position(self):
        return self.pos()
    
    @position.setter
    def position(self,value):
        self.setPos( float(value[0]), float(value[1]) )
        
    def shape(self):
        path = self._path.path()
        
        if self.scene() is None:
            return path
        
        stroker = QtGui.QPainterPathStroker()
        
        pen = self._path.pen()
        width = pen.width()
        
        if pen.isCosmetic():
            width = max(1,pen.width())
            #transform from view to local coordinates
            view = self.scene().views()[0]
            dti = self.deviceTransform(view.viewportTransform()).inverted()[0]
            p = dti.map( QtCore.QPointF(width,width) ) - dti.map( QtCore.QPointF(0.0,0.0) )
            width = math.sqrt( p.x()**2 + p.y()**2 )
        
        stroker.setWidth( width )
        
        path = stroker.createStroke( path )
        return path
    
    def boundingRect(self):
        return self.shape().boundingRect()
    
    def paint(self, *args):
        pass
    
    def extent(self):
        return self._path.path().boundingRect()
        
    def select(self,value):
        self._selected = bool(value)
        
        if self._selected:
            self._path.setPen( self._select_pen )
        else:
            self._path.setPen( self._pen )
        
        if self._resizable:
            self._resizer.setVisible(self._selected)
    
    def edit(self, mode=None, callback=None):
        
        self._edit_finish_callback = callback
        
        if self._editable and not self._editmodemanager.isActive:
            self._resizer.setVisible(False)
            if not self._editmodemanager.start( mode ):
                self.finished_editing()
                return False
        
        return True
    
    def stopEdit(self):
        
        if self._editmodemanager.isActive:
            self._editmodemanager.stop()
    
    def finished_editing(self):
        self._resizer.updateHandles()
        if self._resizable:
            self._resizer.setVisible( self._selected )
        
        if not self._edit_finish_callback is None:
            self._edit_finish_callback()
            self._edit_finish_callback = None
        
        self.notify_change()
    
    def mousePressEvent(self,event):
        if (event.button() & QtCore.Qt.LeftButton ):
            control_key = event.modifiers() & QtCore.Qt.ControlModifier
            self.scene().selectShape( self, value=not (self._selected and control_key), add=control_key )
        elif (not (self.flags() & QtGui.QGraphicsItem.ItemIsMovable) ): event.ignore()
        else: event.ignore()
        
    def mouseReleaseEvent(self,event):
        self.notify_change()
        QtGui.QGraphicsItem.mouseReleaseEvent(self, event)
    
    def mouseDoubleClickEvent(self, event):
        if self._editable:
            self.scene().editShape(self)
        else:
            QtGui.QGraphicsObject.mouseDoubleClickEvent(self,event)

class iShape(iGraphicsItem):
    
    _type = 'polyline'
    
    def __init__(self, vertices, closed=True, **kwargs):
        
        self._vertices = np.array( vertices )
        
        iGraphicsItem.__init__(self, **kwargs)
        
        self._closed = bool(closed)
        
        minval = np.min( self._vertices, axis=0 )
        maxval = np.max( self._vertices, axis=0 )
        center = 0.5*(minval+maxval)
        
        self._rotation_transform.setOrigin( QtGui.QVector3D(center[0],center[1],0.0) )
        self._scale_transform.setOrigin( QtGui.QVector3D(center[0],center[1],0.0) )
        
        self._update_path()
    
    def addNodeNear(self, pos):
        
        #project to nearest edge
        from fklab.geometry.utilities import point2polyline
        
        pos = np.array([pos])
        vertices = self._vertices
        if self._closed:
            vertices = np.concatenate( [vertices, vertices[0:1]], axis=0 )
        (dist_to_path, point_on_path, edge, dist_along_edge, dist_along_path) = point2polyline(vertices, pos)
        
        #insert point
        edge = int(edge)+1
        self._vertices = np.concatenate( [self._vertices[:edge], pos, self._vertices[edge:]], axis=0 )
        
        self.has_changed=True
        self._update_path()
    
    def removeNodeByIndex(self, index):
        index = int(index)
        assert(index>=0 and index<len(self._vertices))
        
        self._vertices = np.delete(self._vertices, [index], axis=0 )
        
        self.has_changed=True
        self._update_path()
    
    def moveNodeTo(self,index,pos):
        
        index = int(index)
        pos = [float(x) for x in pos]
        assert(len(pos)==2)
        
        self._vertices[index] = pos
        
        self.has_changed=True
        self._update_path()
        
    @property
    def closed(self):
        return self._closed
    
    @closed.setter
    def closed(self,value):
        self._closed = bool(value)
        self.has_changed=True
        self._update_path()
        
    def todict(self):
        d = iGraphicsItem.todict(self)
        if self._closed and d['type']=='polyline':
            d['type'] = 'polygon'
        d['vertices'] = self._vertices.tolist()
        d['closed'] = self._closed
        return d
    
    @classmethod
    def fromdict(cls, d):
        return cls( **d)
    
    def _update_path(self):
        
        vertices = self._vertices
        
        if len(vertices)<1:
            self._path.setPath( QtGui.QPainterPath() )
            return
        
        path = QtGui.QPainterPath()
        path.moveTo( *vertices[0] )
        for p in vertices[1:]:
            path.lineTo( *p )
        
        if self._closed:
            path.closeSubpath()
        
        self._path.setPath( path )
    
    def extent(self):
        minval = np.min( self._vertices, axis=0 )
        maxval = np.max( self._vertices, axis=0 )
        return QtCore.QRectF( minval[0], minval[1], maxval[0]-minval[0], maxval[1]-minval[1] )
    
class iRectangle(iShape):
    
    _type = 'rectangle'
    
    def __init__(self,x=0.0,y=0.0, width=1.0, height=1.0, **kwargs):
        vertices=np.array( [[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]] )
        kwargs['closed']=True
        kwargs['position']=[x,y]
        kwargs['size']=[width,height]
        iShape.__init__(self, vertices, **kwargs)
    
    @classmethod
    def create_instruction(cls, **kwargs):
        method = kwargs.get('method','3-points')
        s = ""
        if method=='3-points':
            s = "Select 3 corners to create new rectangle."
        elif method=='2-points':
            s = "Select 2 opposite corners to create new axes-aligned rectangle."
        elif method=='center square':
            s = "Select center and one corner to create new square."
            
        s += "Backspace=delete last point; Esc=cancel."
        
        return s
    
    def todict(self):
        d = iShape.todict(self)
        del d['vertices']
        del d['closed']
        return d
    
    @classmethod
    def fromdict(cls, d):
        position = d.pop('position',[0,0])
        scale = d.pop('size',[1.0,1.0])
        return cls( x=position[0], y=position[1], width=scale[0], height=scale[1], **d)
    
    def toshape(self):
        d = self.todict()
        return fklab.geometry.shapes.rectangle( d['position'], d['size'], np.pi*d['rotation']/180. )
    
    @classmethod
    def fromshape(self, shape, **kwargs):
        if not isinstance(shape, fklab.geometry.shapes.boxed):
            raise TypeError('Can only create iRectangle from boxed shape')
        
        return iRectangle( x=shape.center[0], y=shape.center[1], width=shape.size[0], 
            height=shape.size[-1], rotation=180.*shape.orientation/np.pi, **kwargs )
    
    @classmethod
    def create( cls, scene, method='3-points', **kwargs ):
        
        p = RectangleGenerator(ellipse=False, method=method)
        (x,y),width,height,rotation = p.generate(scene, instruction=cls.create_instruction( method=method ))
        
        if width is None:
            return None
        else:
            shape = iRectangle( x, y, width, height, rotation=rotation, **kwargs )
            return shape

class iEllipse(iGraphicsItem):
    
    _type = 'ellipse'
    
    def __init__(self, x=0, y=0, width=1.0, height=1.0, **kwargs):
        kwargs['position']=[x,y]
        kwargs['size']=[width,height]
        iGraphicsItem.__init__(self, **kwargs)
        
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        
        path = QtGui.QPainterPath()
        path.addEllipse(-0.5,-0.5,1.0,1.0)
        self._path.setPath( path )
    
    @classmethod
    def create_instruction(cls, **kwargs):
        method = kwargs.get('method','3-points')
        s = ""
        if method=='3-points':
            s = "Select 3 corners to create new ellipse."
        elif method=='2-points':
            s = "Select 2 opposite corners to create new axes-aligned ellipse."
        elif method=='center square':
            s = "Select center and one corner to create new circle."
            
        s += "Backspace=delete last point; Esc=cancel."
        
        return s
    
    @classmethod
    def fromdict(cls, d):
        position = d.pop('position',[0,0])
        scale = d.pop('size',[1.0,1.0])
        return cls( x=position[0], y=position[1], width=scale[0], height=scale[1], **d )
    
    def toshape(self):
        d = self.todict()
        return fklab.geometry.shapes.ellipse( d['position'], d['size'], np.pi*d['rotation']/180. )
    
    @classmethod
    def fromshape(self, shape, **kwargs):
        if not isinstance(shape, fklab.geometry.shapes.boxed):
            raise TypeError('Can only create iRectangle from boxed shape')
        
        return iEllipse( x=shape.center[0], y=shape.center[1], width=shape.size[0], 
            height=shape.size[-1], rotation=180.*shape.orientation/np.pi, **kwargs )
    
    
    def extent(self):
        #needed to overload, because ellipse path is not yet set when Resizer is created in GraphicsItem constructor
        return QtCore.QRectF(-0.5,-0.5,1.0,1.0)
        
    @classmethod
    def create( cls, scene, method='3-points', **kwargs ):
        
        p = RectangleGenerator(ellipse=True, method=method)
        (x,y),width,height,rotation = p.generate(scene, instruction=cls.create_instruction( method=method ))
        
        if width is None:
            return None
        else:
            shape = iEllipse( x, y, width, height, rotation=rotation, **kwargs )
            return shape
        
class iNGon(iShape):
    
    def __init__(self,n=3,x=0.0, y=0.0, radius=1.0, **kwargs):
        
        angle = np.arange(n)*2*np.pi/n
        vertices = 0.5*np.vstack( (np.cos(angle),np.sin(angle)) ).T
        
        kwargs['closed']=True
        kwargs['position']=[x,y]
        kwargs['size']=[radius,radius]
        
        iShape.__init__(self, vertices, **kwargs)

class iPolyline(iShape):
    
    def __init__(self, vertices=None, closed=False, spline=False, **kwargs):
        if vertices is None:
            vertices = np.zeros( (0,2) )
        
        kwargs['closed']=closed
        kwargs['edit_modes'] = OrderedDict( [['edit',ShapeEditor]] )
        iShape.__init__(self, vertices, **kwargs)
        
        self._spline = bool(spline)
        
        self._spline_pen = create_pen( width=1, cosmetic=True, color=[255,0,130], style=1 )
        self._spline_item = QtGui.QGraphicsPathItem(parent=self)
        self._spline_item.setPen(self._spline_pen)
        
        self._update_spline()
    
    @property
    def spline(self):
        return self._spline
    
    @spline.setter
    def spline(self, value):
        self._spline = bool(value)
        self.has_changed=True
        self._update_spline()
    
    @classmethod
    def create_instruction(cls, **kwargs):
        
        s = 'Draw polyline. left-click=new vertex; right-click or backspace=remove last vertex; c=toggle close; s=toggle spline; q=finish; esc=cancel.'
        return s
    
    def _update_path(self):
        
        iShape._update_path(self)
        try: #needed because during object construction, spline has not been created yet when _update_path is called for first time
            self._update_spline()
        except:
            pass
        
    def _update_spline(self):
        
        path = QtGui.QPainterPath()
        if self._spline and ( len(self._vertices) + int(self._closed) )>3:
            
            vertices = self._vertices
            
            if self._closed:
                vertices = np.concatenate( [vertices, vertices[0:1]], axis=0 )
            
            try:
                spline, spline_param = _sample_spline( vertices, oversampling=20, closed=self._closed, openpath=False)
                path.moveTo( *spline[0][0] )
                for p in spline[0][1:]:
                    path.lineTo( *p )
            except:
                print("spline calculation error")
        
        self._spline_item.setPath( path )
    
    def todict(self):
        d = iShape.todict(self)
        d['spline']=self._spline
        return d
    
    @classmethod
    def fromdict(cls, d):
        return cls(**d)
    
    def toshape(self):
        d = self.todict()
        if d['closed']:
            cls = fklab.geometry.shapes.polygon
        else:
            cls = fklab.geometry.shapes.polyline
        
        # Do we need to transform vertices?
        return cls( vertices=d['vertices'], spline=d['spline'] )
    
    @classmethod
    def fromshape(self, shape, **kwargs):
        if not isinstance(shape, fklab.geometry.shapes.polyline):
            raise TypeError('Can only create iRectangle from polyline shape')
        
        return iPolyline( vertices=shape.vertices, closed=shape.issolid, spline=shape.isspline, **kwargs )
    
    
    @classmethod
    def create( cls, scene, closed=False, spline=False, **kwargs ):
        
        p = PolylineGenerator(closed=closed, spline=spline)
        points, closed, spline = p.generate(scene, instruction=cls.create_instruction())
        
        if len(points)>0:
            shape = iPolyline( vertices=np.array(points), closed=closed, spline=spline, **kwargs )
            return shape
        else:
            return None

class iGraph(iGraphicsItem):
    
    _type = 'graph'
    
    node_changed = QtCore.pyqtSignal(str, str) # add/remove/move, node name
    
    def __init__(self, nodes = [], edges = [], **kwargs):
        
        kwargs['edit_modes'] = OrderedDict( [ ['edit',GraphEditor], ['extend',GraphExtender] ] )
        iGraphicsItem.__init__(self, **kwargs)
        
        nodes, edges = iGraph._check_graph( nodes, edges )
        
        self._nodes = nodes
        self._edges = edges
        
        self._update_path()
        self._set_center()
        
        self._resizer.updateHandles()
        
    
    def todict(self):
        d = iGraphicsItem.todict(self)
        d['nodes']=self._nodes
        d['edges'] = self._edges
        return d
    
    @classmethod
    def fromdict(cls, d,):
        return cls( **d )
    
    def toshape(self):
        
        nodes = np.array( [ xy for _,xy in self._nodes ] )
        
        edges = []
        for edge in self._edges:
            edges.append( [self._get_node( edge[0] )] + edge[1] + 
                [self._get_node( edge[2] )] )
            
        edges = [ fklab.geometry.shapes.polyline( vertices=np.array(x) ) for x in edges ]
            
        return fklab.geometry.shapes.graph( polylines=edges, nodes=nodes )
    
    @classmethod
    def fromshape(self, shape, **kwargs):
        if not isinstance( shape, fklab.geometry.shapes.graph ):
            raise TypeError('Can only create iGraph from graph shape')
        
        nodes = [ [str(k),xy.tolist()] for k, xy in enumerate( shape._nodes ) ]
        edges = []
        for p, (e1,e2) in zip(shape._polylines, shape._edges):
            edges.append( [ nodes[e1][0] ] + [p._vertices[1:-1].tolist()] + [ nodes[e2][0] ] )
        
        return iGraph(nodes=nodes, edges=edges, **kwargs)
        
    def _set_center(self):
        r = self._path.path().boundingRect()
        center = r.center()
        
        #TODO: adjust position!
        #compute vector difference between old center and new center
        #apply difference to item position
        
        old_origin = self._rotation_transform.origin()
        
        #convert to scene coordinates with old transform
        #scene = self.scene()
        #if scene is not None:
        p1 = self.mapToScene( old_origin.x(), old_origin.y() )
        
        #set new origin
        new_origin = QtGui.QVector3D(center.x(),center.y(),0.0)
        self._rotation_transform.setOrigin( new_origin )
        self._scale_transform.setOrigin( new_origin )
        
        #if scene is not None:
        #convert to scene coordinates with new transform
        p2 = self.mapToScene( old_origin.x(), old_origin.y() )
        
        #adjust position
        self.setPos( self.pos() - p2 + p1 )
    
    @classmethod
    def _check_graph( cls, nodes, edges ):
        
        if not isinstance( nodes, list ) or not isinstance( edges, list ):
            raise ValueError
        
        try:
            nodes = [[str(n),[float(x),float(y)]] for (n,(x,y)) in nodes]
        except:
            raise ValueError
        
        node_names = [n for (n,xy) in nodes]
        if len(node_names)>len(set(node_names)):
            #non-unique names
            raise ValueError
        
        try:
            edges = [[str(a),list(b),str(c)] for (a,b,c) in edges]
            for k,(start_node, mid_nodes, end_node) in enumerate(edges):
                assert( start_node in node_names and end_node in node_names )
                edges[k][1] = [ [float(x),float(y)] for (x,y) in mid_nodes ]
        except:
            raise ValueError
        
        return nodes, edges
    
    def _get_node(self, name):
        node = [ xy for (n,xy) in self._nodes if n==name ]
        assert( len(node)==1 )
        return node[0]
    
    def _update_path(self, fake=False):
        
        self.prepareGeometryChange()
        
        path = QtGui.QPainterPath()

        for e in self._edges:
            start_node = self._get_node( e[0] )
            path.moveTo( *start_node )
            for l in e[1]:
                path.lineTo( *l )
            end_node = self._get_node( e[2] )
            path.lineTo( *end_node )
        
        self._path.setPath( path )
        
        self._set_center()
    
    @classmethod
    def create(cls, scene, **kwargs):
        return generate_graph(scene, **kwargs)
    
    def removeOrphanNodes(self):
        
        start_nodes = [ a for a,b,c in self._edges ]
        end_nodes = [ c for a,b,c in self._edges ]
        
        nodes_to_remove = [ n for n,xy in self._nodes if (n not in start_nodes and n not in end_nodes) ]
        
        for k in nodes_to_remove:
            self.removeNodeByName( k )
    
    def removeNodeByName(self, name):
        
        name = str(name)
        idx = [ k for k,(n,xy) in enumerate(self._nodes) if n==name ]
        assert( len(idx)==1 )
        self.removeNodeByIndex( idx[0] )
        
    def removeNodeByIndex(self,idx):
        
        idx = int(idx)
        assert( idx>=0 and idx<len(self._nodes) )
        
        node_name = self._nodes[idx][0]
        
        #delete edges that connect to this node
        self._edges = [ [start_node,mid_nodes,end_node] for (start_node,mid_nodes,end_node) in self._edges if (start_node!=node_name and end_node!=node_name) ]
        
        #delete node itself
        del self._nodes[idx]
        
        self.has_changed=True
        self._update_path()
        
        self.node_changed.emit( 'remove', node_name )
    
    def removeEdgePointByIndex(self, edge, idx):
        
        edge = int(edge)
        assert( edge>=0 and edge<len(self._edges) )
        
        idx = int(idx)
        assert( idx>=0 and idx<=len(self._edges[edge][1]) )
        
        del self._edges[edge][1][idx]
        
        self.has_changed=True
        self._update_path()
        
    
    def removeEdgeByIndex(self,idx):
        idx = int(idx)
        assert( idx>=0 and idx<len(self._edges) )
        
        del self._edges[idx]
        
        self.has_changed=True
        self._update_path()
        
        #self.edge_changed.emit( 'remove', idx )
        
        self.removeOrphanNodes()
    
    def findNearestEdge(self, pos ):
        
        from fklab.geometry.utilities import point2polyline
        
        pos = np.array([pos])
        
        min_distance = np.Inf
        min_edge = None
        edge_segment = None
        
        for k,v in enumerate(self._edges):
            #create edge point array
            points = np.array( self._get_node( v[0] ) ).reshape( (1,2) )
            points = np.concatenate( (points, np.array( v[1] ).reshape( (len(v[1]),2) ) ), axis=0 )
            points = np.concatenate( (points, np.array( self._get_node( v[2] ) ).reshape( (1,2) ) ), axis=0 )
            
            (dist_to_path, point_on_path, edge, dist_along_edge, dist_along_path) = point2polyline(points, pos)
            dist_to_path = abs(dist_to_path)
            if dist_to_path<min_distance:
                min_distance = dist_to_path
                min_edge = k
                edge_segment = edge
        
        return min_edge, edge_segment
    
    def addEdgePointNear(self, pos):

        edge, seg = self.findNearestEdge( pos )
        self.addEdgePoint( edge, seg, pos )
    
    def addEdgePoint(self, edge, insert_index, pos ):
        edge = int(edge)
        insert_index = int(insert_index)
        self._edges[edge][1].insert( insert_index, pos )
        self.has_changed=True
        self._update_path()
    
    def removeEdgeNear(self, pos):
        
        edge, seg = self._findNearestEdge( pos )
        self.removeEdgeByIndex( edge )
    
    def addNode( self, xy, name=None ):
        n = self.addNodes( [xy], [name] )
        return n[0]
    
    def addNodes(self, xy, name=None):
        
        #xy should be a list of [x,y] lists
        #name should be a list of strings or None
        
        import random
        
        if len(xy)==0:
            return []
        
        if name is None:
            name = [ None ] * len(xy)
        
        def _local_check_name( n, nlist ):
            
            #Generate unique name. Only try 5 times.
            if n is None:
                done = False
                for k in range(5): #5 tries
                    n = 'node' + str(random.randint(100000,999999))
                    if n not in nlist:
                        done = True
                        break
                
                if not done:
                    raise InternalError
            
            else:
                n = str(n)
                if n in node_names:
                    raise ValueError
            
            return n
        
        node_names = [n for (n,z) in self._nodes]
        
        name = [ _local_check_name(n,node_names) for n in name ]
        
        xy = [ [float(x),float(y)] for x,y in xy ]
        
        self._nodes.extend( map( list, zip(name, xy) ) )
        
        self.has_changed=True
        
        self._update_path()
        
        for n in name:
            self.node_changed.emit('add', n)
        
        return name
            
    def addEdge(self, start=None, points=[], end=None):
        
        if start is None:
            return
        
        if len(points)==0 and end is None:
            return
        
        new_nodes = []
        start_new, end_new = False, False
        
        node_names = [n for (n,xy) in self._nodes]
        
        if isinstance(start,str):
            if start not in node_names:
                raise ValueError
        else:
            new_nodes.append( start )
            start_new = True
        
        if end is None:
            end = points.pop()
        
        if isinstance(end,str):
            if end not in node_names:
                raise ValueError
        else:
            new_nodes.append( end )
            end_new = True
        
        points = [ [float(x),float(y)] for x,y in points ]
        
        new_names = self.addNodes( new_nodes )
        
        if start_new:
            start = new_names[0]
        
        if end_new:
            end = new_names[-1]
        
        self._edges.append( [start, points, end] )
        
        self.has_changed=True
        
        self._update_path()
        
    def moveNodeTo(self, node, pos):
        node = str(node)
        pos = [ float(x) for x in pos ]
        assert(len(pos)==2)
        
        for k in range(len(self._nodes)):
            if self._nodes[k][0] == node:
                self._nodes[k][1] = pos
        
        self.has_changed=True
        self._update_path()
        
        self.node_changed.emit('move', node )
    
    def moveEdgePointTo(self, edge, index, pos ):
        
        edge = int(edge)
        index = int(index)
        pos = [float(x) for x in pos]
        assert(len(pos)==2)
        
        self._edges[edge][1][index] = pos
        
        self.has_changed=True
        self._update_path()
        

SHAPES = {'rectangle':iRectangle, 'ellipse':iEllipse, 'square':iRectangle, 'circle':iEllipse, 'polyline':iPolyline, 'polygon':iPolyline, 'graph':iGraph}


class MouseClickCollector(QtWidgets.QGraphicsObject):
    finished = QtCore.pyqtSignal()
    def __init__(self, nclicks=0, buttons=[QtCore.Qt.LeftButton], escape=True, **kwargs):
        
        QtWidgets.QGraphicsObject.__init__(self,**kwargs)

        self._nclicks = nclicks
        self._buttons = buttons
        self._escape = escape
        
        self.clicked_points = []
        
        self._scene = None
        
        self.setFlag(QtGui.QGraphicsItem.ItemIsFocusable, True)
        self.setVisible(True)
    
    def finish(self):
        self.finished.emit()
        
    def prepare_scene(self, scene):
        scene.addItem(self)
        self.setFocus()
    
    def cleanup_scene(self, scene):
        scene.removeItem(self)
    
    def post_process(self, points):
        return points
    
    def eventFilter( self, obj, event ):
        
        #catch mouse press events
        #and key press events
        if obj is self._scene:
            event_type = event.type()
            if event_type == 156: #button press
                self.mousePressEvent(event)
                return True
            elif event_type == 155: #mouse move
                self.mouseMoveEvent(event)
                return True
            elif event_type == 6: #key press
                self.keyPressEvent(event)
                return True
        
        return False
    
    def generate(self, scene, timeout=0, instruction=""):
        
        self._scene = scene
        scene.setInstructions( instruction )
        
        self.prepare_scene(scene)
        
        scene.installEventFilter( self ) 
        
        #wait for the signal
        sw = SignalWait( self.finished )
        timedout = sw.wait(0)
        
        scene.removeEventFilter( self )
        
        self.cleanup_scene(scene)
        
        scene.setInstructions( "" )
        
        self._scene = None
        
        return self.post_process( self.clicked_points )
    
    def add_click(self,pos):
        self.clicked_points.append( [pos.x(), pos.y()])
        if self._nclicks>0 and len(self.clicked_points)>=self._nclicks:
            self.finish()
    
    def mousePressEvent(self,event):
        
        if (event.button() in self._buttons):
            self.add_click(event.scenePos())
            event.accept()
    
    def mouseMoveEvent(self, event):
        pass
    
    def keyPressEvent(self,event):

        if self._escape and event.key() == QtCore.Qt.Key_Escape:
            self.clicked_points = []
            event.accept()
            self.finished.emit()
            return
    
    def paint(self,*args,**kwargs):
        return
    
    def boundingRect(self):
        return QtCore.QRectF(0,0,0,0)

class MouseTracer( MouseClickCollector ):
    def __init__(self, **kwargs):
        MouseClickCollector.__init__(self, **kwargs)
        
        self._pen = create_pen( width=2, cosmetic=True, color=[0,0,0], style=1 )
        self._rubber_pen = create_pen( width=1, cosmetic=True, color=[255,0,0], style=2 )
        
        self._current_mouse_position = None
    
    def prepare_scene(self, scene):
    
        MouseClickCollector.prepare_scene(self, scene)
        
        self._trace_item = QtGui.QGraphicsPathItem(parent=self)
        self._trace_item.setPen(self._pen)
        
        self._rubber_item = QtGui.QGraphicsPathItem(parent=self)
        self._rubber_item.setPen(self._rubber_pen)
    
    def update_trace(self):
        if len(self.clicked_points)>1:
            path = QtGui.QPainterPath()
            path.moveTo( *self.clicked_points[0] )
            for p in self.clicked_points[1:]:
                path.lineTo( *p )
        else:
            path = QtGui.QPainterPath()
            
        self._trace_item.setPath( path )
    
    def update_rubber(self):
        if len(self.clicked_points)>0:
            path = QtGui.QPainterPath()
            path.moveTo( *self.clicked_points[-1] )
            path.lineTo( self._current_mouse_position )
        else:
            path = QtGui.QPainterPath()
        
        self._rubber_item.setPath( path )
    
    def mousePressEvent(self, event):
        MouseClickCollector.mousePressEvent(self,event)
        if event.isAccepted():
            self._current_mouse_position = event.scenePos()
            self.update_trace()
            self.update_rubber()
    
    def mouseMoveEvent(self, event):
        MouseClickCollector.mouseMoveEvent(self,event)
        event.accept()
        self._current_mouse_position = event.scenePos()
        self.update_rubber()

class PolylineGenerator( MouseTracer ):
    def __init__(self, npoints=0, closed=False, spline=False, **kwargs):
        kwargs['buttons'] = [QtCore.Qt.LeftButton]
        MouseTracer.__init__( self, nclicks = npoints, **kwargs )
        
        self.closed=closed
        self.spline=spline
        
        self._spline_pen = create_pen( width=1, cosmetic=True, color=[255,0,130], style=1 )
    
    def prepare_scene(self,scene):
        
        MouseTracer.prepare_scene(self,scene)
        
        self._close_item = QtGui.QGraphicsPathItem(parent=self)
        self._close_item.setPen(self._rubber_pen)
        
        self._spline_item = QtGui.QGraphicsPathItem(parent=self)
        self._spline_item.setPen(self._spline_pen)
    
    def post_process(self, points):
        return (points, self.closed, self.spline)
    
    def update_close(self):
        
        path = QtGui.QPainterPath()
        if self.closed and len(self.clicked_points)>1:
            path.moveTo( self._current_mouse_position )
            path.lineTo( *self.clicked_points[0] )
        
        self._close_item.setPath(path)
    
    def update_spline(self):
        path = QtGui.QPainterPath()
        if self.spline and len(self.clicked_points)>3:
            
            vertices = self.clicked_points
            current_point = [self._current_mouse_position.x(),self._current_mouse_position.y()]
            if current_point[0]!=self.clicked_points[-1][0] or current_point[1]!=self.clicked_points[-1][1]:
                vertices = vertices + [current_point]
            
            if self.closed:
                vertices = vertices + [self.clicked_points[0]]
            
            vertices = np.array(vertices)
            
            try:
                spline, spline_param = _sample_spline( vertices, oversampling=20, closed=self.closed, openpath=False)
                path.moveTo( *spline[0][0] )
                for p in spline[0][1:]:
                    path.lineTo( *p )
            except:
                print("spline calculation error")
        
        self._spline_item.setPath( path )
    
    def mousePressEvent(self, event):
        
        if (event.button() == QtCore.Qt.RightButton):
            if len(self.clicked_points)>0: del self.clicked_points[-1]
            event.accept()
        else:
            MouseTracer.mousePressEvent(self,event)
        
        if event.isAccepted():
            self.update_trace()
            self.update_rubber()
            self.update_close()
            self.update_spline()
    
    def mouseMoveEvent(self, event):
        MouseTracer.mouseMoveEvent(self,event)
        if event.isAccepted():
            self.update_close()
            self.update_spline()
    
    def keyPressEvent(self, event):
        key = event.key()
        
        if key == QtCore.Qt.Key_Q:
            event.accept()
            self.finish()
        elif key == QtCore.Qt.Key_C:
            event.accept()
            self.closed = not self.closed
        elif key == QtCore.Qt.Key_S:
            event.accept()
            self.spline = not self.spline
        elif key == QtCore.Qt.Key_Backspace:
            event.accept()
            if len(self.clicked_points)>0: del self.clicked_points[-1]
        else:
            MouseClickCollector.keyPressEvent(self,event)
        
        if event.isAccepted():
            self.update_trace()
            self.update_rubber()
            self.update_close()
            self.update_spline()

class RectangleGenerator( MouseClickCollector ):
    def __init__(self, ellipse=False, method='3-points', **kwargs):
        
        method = str(method)
        if not method in ['3-points', '2-points', 'center square']:
            raise NotImplementedError
        
        if method == '3-points':
            nclicks = 3
        else:
            nclicks = 2
        
        kwargs['buttons'] = [QtCore.Qt.LeftButton]
        MouseClickCollector.__init__(self, nclicks=nclicks, **kwargs)
        
        self._ellipse = bool(ellipse)
        self._method = method
        
        self._pen = create_pen( width=2, cosmetic=True, color=[0,0,0], style=1 )
        
        self._current_mouse_position = None
    
    def prepare_scene(self, scene):
        
        MouseClickCollector.prepare_scene(self, scene)
        
        if self._ellipse:
            self._rect_item = QtGui.QGraphicsEllipseItem(parent=self)
        else:
            self._rect_item = QtGui.QGraphicsRectItem(parent=self)
        
        self._rect_item.setPen(self._pen)
    
    def post_process(self,points):
        
        if len(points)==0:
            return ([None,None],None,None,None)
        
        (x0,y0),width,height,alpha = self._compute_rect( points ) 
        
        #convert x,y to center of rectangle
        phi = alpha*np.pi/180
        x = 0.5*(width*np.cos(phi) - height*np.sin(phi)) + x0
        y = 0.5*(width*np.sin(phi) + height*np.cos(phi)) + y0
        
        return ([x,y],width,height,alpha)
    
    def _compute_rect(self, points):
        
        if self._method == '3-points':
            
            width = math.sqrt( (points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2 )
            alpha = np.arctan2( points[1][1]-points[0][1], points[1][0]-points[0][0] )
            
            #rotate x3,y3 by -alpha around x1,x2
            y = (points[2][0]-points[0][0])*np.sin( -alpha ) + (points[2][1]-points[0][1])*np.cos( -alpha )
            height = np.abs(y)
            
            alpha = alpha*180/np.pi
                
            if y < 0:
                width, height = height, width
                alpha -= 90
            
            x,y = points[0][0], points[0][1]
            
        elif self._method == '2-points':
            width = abs(points[1][0] - points[0][0])
            height = abs(points[1][1] - points[0][1])
            alpha = 0.0
            x,y = min( points[0][0], points[1][0] ), min( points[0][1], points[1][1] )
            
        else: #'center square'
            width = height = 2*math.sqrt( 0.5* ( (points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2 ) )
            alpha = np.arctan2( points[1][1]-points[0][1], points[1][0]-points[0][0] ) * 180/np.pi + 135
            x,y = points[1][0], points[1][1]
            
        return ((x,y), width, height, alpha)
    
    def update_rect(self):
        npoints = len(self.clicked_points)
        if npoints==0:
            self._rect_item.setVisible(False)
            return
        
        points = []
        points.extend( self.clicked_points )
        points.append( (self._current_mouse_position.x(), self._current_mouse_position.y()) )
        
        if self._method == '3-points' and npoints==1:
            points.append( points[-1] )
        
        (x1,y1),width,height,alpha = self._compute_rect( points )
        
        self._rect_item.setRect( x1, y1, width, height )
        self._rect_item.setTransformOriginPoint( x1, y1)
        self._rect_item.setRotation( alpha )
        self._rect_item.setVisible(True)
    
    def mousePressEvent(self, event):
        
        if (event.button() == QtCore.Qt.RightButton):
            if len(self.clicked_points)>0: del self.clicked_points[-1]
            event.accepted()
        
        self._current_mouse_position = event.scenePos()
        
        MouseClickCollector.mousePressEvent(self,event)
        
        if event.isAccepted():
            self.update_rect()
    
    def mouseMoveEvent(self, event):
        event.accept()
        self._current_mouse_position = event.scenePos()
        MouseClickCollector.mouseMoveEvent(self,event)
        self.update_rect()
    
    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Backspace:
            if len(self.clicked_points)>0: del self.clicked_points[-1]
            event.accept()
        else:
            MouseClickCollector.keyPressEvent(self,event)
        
        if event.isAccepted():
            self.update_rect()

def generate_graph(scene, **kwargs):
    
    shape = iGraph( nodes = [], edges = [], **kwargs )
    shape.instructions.connect( scene.instructions )
    scene.addItem( shape )
    
    #set graph to extend edit mode
    shape.edit(mode='extend')
    
    #wait for finish edit signal
    sw = SignalWait( shape._editmodemanager.finished )
    timedout, accepted = sw.wait(0)
    
    scene.removeItem(shape)
    
    #if graph is empty -> remove, return None
    if len(shape._nodes)==0 or not accepted:
        return None
    else:
        return shape

def iTriangle( x=0, y=0, radius=1.0, rotation=0.0, **kwargs ):
    return iNGon( 3, x=x, y=y, radius=radius, rotation=rotation, **kwargs)

def iDiamond( *args, **kwargs ):
    return iNGon( 4, *args, **kwargs)

def iPentagon( *args, **kwargs ):
    return iNGon( 5, *args, **kwargs)


class GraphicsScene( QtWidgets.QGraphicsScene ):
    
    instructions = QtCore.pyqtSignal(str)
    
    mode_changed = QtCore.pyqtSignal(str)
    selection_changed = QtCore.pyqtSignal()
    shape_edited = QtCore.pyqtSignal(int)
    
    def setInstructions(self, s=""):
        
        if len(s) == 0:
            if len(self._shapes)>0:
                s = "Click shape to select. Ctrl-click to multi-select. Left-click-drag shape or handle to move/scale. Ctrl-left-click-drag handle to rotate."
            else:
                s = "Add shapes."
        
        self.instructions.emit( s )
    
    def __init__(self,*args,**kwargs):
        QtGui.QGraphicsScene.__init__(self,*args,**kwargs)
        
        self._shape_identifier = int(0)
        
        self._shapes = []
        self._selection = []
        
        self._shape_counter = 0
        
        self._mode = 'default'
        
        self._edit_shape = None
        
        self._tracker_data_pen = create_pen( width=1, cosmetic=True, color=[130,130,130], style=1, alpha=0.15 )
        
        self._tracker_data = QtGui.QGraphicsPathItem()
        self._tracker_data.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self._tracker_data.setAcceptHoverEvents(False)
        self._tracker_data.setZValue(-1)
        self._tracker_data.setCacheMode( QtGui.QGraphicsItem.DeviceCoordinateCache )
        self._tracker_data.setEnabled(False)
        self._tracker_data.setPen( self._tracker_data_pen )
        self.addItem(self._tracker_data)
        
        self._video_data = QtGui.QGraphicsPixmapItem()
        self._video_data.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self._video_data.setAcceptHoverEvents(False)
        self._video_data.setZValue(-2)
        self._video_data.setEnabled(False)
        self.addItem(self._video_data)
        
    
    def editShape(self,shape,*args,**kwargs):
        
        if not shape.canEdit:
            return
        
        #1. signal that mode changed
        self._mode = "edit"
        self._edit_shape = shape
        self.mode_changed.emit(self._mode)
        
        #2. disable all other shapes
        for _,s in self._shapes:
            if s is not shape:
                s.setEnabled(False)
        
        #3. call shape to be edited
        shape.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
        
        if not shape.edit(*args,callback=self.finishEditShape,**kwargs):
            self.finishEditShape()
        
    def finishEditShape(self):
        if self._mode != 'edit':
            return
        
        #1. call shape that was edited
        shape = self._edit_shape
        shape.stopEdit()
        shape.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        
        #2. enable all other shapes
        identifier = -1
        for shape_id,s in self._shapes:
            if s is not shape:
                s.setEnabled(True)
            else:
                identifier = shape_id
        
        #3. signal that mode changed
        self._mode = "default"
        self._edit_shape = None
        self.mode_changed.emit(self._mode)
    
    def unselectAll(self, but=[]):
        new_selection = []
        for k in self._selection:
            if k not in but:
                k.select(False)
            else:
                new_selection.append( k )
        
        self._selection = new_selection
        
        self.selection_changed.emit()
                    
    def selectShape(self,shapes,value=True,add=False):
        if shapes is None:
            return
        
        if not isinstance(shapes,list):
            shapes = [shapes]
        
        selection_changed = False
        
        for shape in shapes:
        
            if shape in self._selection:
                if value and not add:
                    self.unselectAll(but=[shape])
                    add = True #so that other shapes in the list are added
                    selection_changed = True
                elif not value:
                    #unselect shape
                    shape.select(False)
                    self._selection.remove(shape)
                    selection_changed = True
            elif value:
                if add:
                    #select shape
                    shape.select(True)
                    self._selection.append( shape )
                    selection_changed = True
                else:
                    #unselect selection
                    #select shape
                    self.unselectAll()
                    shape.select(True)
                    self._selection.append( shape )
                    add = True
                    selection_changed = True
        
        if selection_changed: self.selection_changed.emit()
    
    def selectByIndex(self, idx, value=True, add=False):
        
        if isinstance(idx,int): idx=[idx]
        
        shapes = [ s for k,(_,s) in enumerate(self._shapes) if k in idx  ]
        
        if len(shapes)>0:
            self.selectShape( shapes, value=value, add=add )
    
    def getSelection(self):
        return self._selection
    
    def getSelectionIndex(self):
        k = [ idx for idx,(_,s) in enumerate(self._shapes) if s in self._selection ]
        return k
    
    def selectByID(self, identifiers, value=True, add=False):
        
        if isinstance(identifiers,int): identifiers=[identifiers]
        
        shapes = [ s for k,(ID,s) in enumerate(self._shapes) if ID in identifiers  ]
        
        if len(shapes)>0:
            self.selectShape( shapes, value=value, add=add )
    
    def getSelectionID(self):
        k = [ ID for idx,(ID,s) in enumerate(self._shapes) if s in self._selection ]
        return k
    
    def mousePressEvent(self,event):
        
        QtGui.QGraphicsScene.mousePressEvent(self,event)
        if not event.isAccepted():
            if self._mode=='default':
                self.unselectAll()
    
    def keyPressEvent(self,event):
        key=event.key()
        
        if self._mode == 'default':
            if key == QtCore.Qt.Key_Backspace or key == QtCore.Qt.Key_Delete:
                #remove selected shapes
                #self.removeSelectedShapes()
                pass
            elif key == QtCore.Qt.Key_A:
                #toggle selection of all/none
                if len(self._selection)==len(self._shapes):
                    #deselect all
                    self.unselectAll()
                else:
                    #select all
                    shapes = [s for _,s in self._shapes]
                    self.selectShape( shapes )
            else:
                QtGui.QGraphicsScene.keyPressEvent(self,event)
        else:
            QtGui.QGraphicsScene.keyPressEvent(self,event)
    
    def todict(self, selection=None):
        
        if selection is None:
            selection = range(len(self._shapes))
        
        d = OrderedDict()
        for idx in selection:
            d[self._shapes[idx][0]] = self._shapes[idx][1].todict()
        
        return d
    
    def createShape(self, shapetype, **kwargs):
        
        if self._mode != 'default':
            raise ValueError 
        
        mode = self._mode
        self._mode = 'create'
        self.mode_changed.emit(self._mode)
        
        #disable all other shapes
        for _,s in self._shapes:
            s.setEnabled(False)
        
        shape_class = SHAPES[shapetype]
        
        shape = shape_class.create(self, **kwargs)
        
        #enable all other shapes
        for _,s in self._shapes:
            s.setEnabled(True)
        
        self._mode = mode
        self.mode_changed.emit(self._mode)
        
        if not shape is None:
            shape = shape.toshape()
        
        return shape
    
    def getShape(self,ID):
        shape = [s for (_id,s) in self._shapes if _id==ID ]
        if len(shape)==1:
            return shape[0]
        elif len(shape)==0:
            return None
        else:
            raise KeyError
        
    def getShapeByIndex(self,idx):
        return self._shapes[idx][1]
    
    def new_identifier(self):
        val = self._shape_identifier
        self._shape_identifier = self._shape_identifier + 1
        return val
        
    def addShape(self, shape):
        
        if shape is None:
            return
        
        _id = self.new_identifier()
        
        if isinstance(shape, fklab.geometry.shapes.rectangle):
            shape = iRectangle.fromshape( shape, identifier=_id )
        elif isinstance(shape, fklab.geometry.shapes.ellipse):
            shape = iEllipse.fromshape( shape, identifier=_id )
        elif isinstance(shape, fklab.geometry.shapes.polyline):
            shape = iPolyline.fromshape( shape, identifier=_id )
        elif isinstance(shape, fklab.geometry.shapes.graph):
            shape = iGraph.fromshape( shape, identifier=_id )
        else:
            raise TypeError('Unknown shape type')
        
        self.addItem(shape)
        self._shapes.append( (_id, shape) )
        
        shape.instructions.connect( self.setInstructions )
        shape.changed.connect( self.shapeChanged )
        
        self.setInstructions()
        
        return _id
    
    def shapeChanged(self, identifier):
        self.shape_edited.emit(identifier)
    
    def addShapesFromDict(self, d, replace=False):
        return
        for (k,v) in d.iteritems():
            shapetype = v.pop('type')
            shapeclass = SHAPES[shapetype]
            self.addShape( shapeclass.fromdict( v ), name=k, replace=replace )
    
    def createAddShape(self, shapetype, **kwargs):
        shape = self.createShape( shapetype, **kwargs )
        return self.addShape( shape )
    
    def removeSelectedShapes(self):
        identifiers = [ _id for k,(_id,s) in enumerate(self._shapes) if s in self._selection ]
        for k in identifiers:
            self.removeShapeByID( k )
        
    def removeShape(self,shape):
        idx = [ k for k,(n,s) in enumerate(self._shapes) if s is shape ]
        if len(idx)==0:
            return False
            
        assert( len(idx)==1 )
        
        return self.removeShapeByIndex( idx[0] )
        
    def removeShapeByID(self, ID):
                
        idx = [ k for k,(_id,s) in enumerate(self._shapes) if _id==ID ]
        if len(idx)==0:
            return False
            
        assert( len(idx)==1 )
        
        return self.removeShapeByIndex( idx[0] )
    
    def removeShapeByIndex(self, idx):
        idx = int(idx)
        if idx<0 or idx>=len(self._shapes):
            return False
        
        ID, shape = self._shapes[idx]
        self.removeItem( shape )
        
        del self._shapes[idx]
        
        if shape in self._selection:
            self._selection.remove( shape )
        
        self.setInstructions()
        
        return True
    
    def removeAllShapes(self):
        
        for ID, shape in self._shapes:
            self.removeItem( shape )
        
        self._shapes = []
        self._selection = []
        
        self.setInstructions()
    
    def setTrackerData(self, data=None, size=None):
        
        if data is None or len(data)<1:
            self._tracker_data.setPath( QtGui.QPainterPath() )
            return
        
        path = QtWidgets.QPainterPath()
        path.moveTo( *data[0] )
        for p in data[1:]:
            path.lineTo( *p )
        
        self._tracker_data.setPath( path )
        
        if size is None:
            #TODO: calculate min and max of data
            pass
    
    def setTrackerImage(self, image=None):
        if image is None:
            self._video_data.setPixmap( QtGui.QPixmap() )
        else:
            self._video_data.setPixmap( QtGui.QPixmap( image ) )


class GraphicsCanvas( QtWidgets.QGraphicsView ):
    
    coordinates_changed = QtCore.pyqtSignal(float,float)
    
    def __init__(self, *args, **kwargs):
        
        QtWidgets.QGraphicsView.__init__(self, *args, **kwargs )
        
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        self.setViewportUpdateMode( 0 )
        
        self._scene = GraphicsScene(parent=self)
        self.setScene( self._scene )
        
        self.setSceneRect( QtCore.QRectF() )
        
        self._zoom_speed = 1.15
        self._zoom_factor = 1.0
        
        self._create_mode = False
    
    def leaveEvent(self,event):
        self.coordinates_changed.emit( 0, 0 )
        QtGui.QGraphicsView.leaveEvent(self,event)
    
    def mouseMoveEvent(self,event):
        pos = self.mapToScene( event.pos() )
        x,y = pos.x(), pos.y()
        self.coordinates_changed.emit( x, y )
        QtGui.QGraphicsView.mouseMoveEvent(self,event)
        
    @property
    def zoom_speed(self):
        return self._zoom_speed
    
    @zoom_speed.setter
    def zoom_speed(self,value):
        value = float(value)
        if value<=1:
            raise ValueError
        self._zoom_speed = value
    
    def wheelEvent(self, event):
        
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        
        numsteps = event.delta() / 120.0
        
        if numsteps==0: return
        
        # compute scaling factor
        scaleFactor = self._zoom_speed**numsteps
        
        # don't allow zoom out more than original
        new_zoom_factor = self._zoom_factor * scaleFactor
        scaleFactor = new_zoom_factor / self._zoom_factor
        
        # scale the view / do the zoom
        self.scale( scaleFactor, scaleFactor )
        
        self._zoom_factor = new_zoom_factor
        
        # Don't call superclass handler here
        # as wheel is normally used for moving scrollbars
        
        return
    

class ScrollStackWidget( QtWidgets.QStackedWidget ):
    def wheelEvent(self,event):
        count = self.count()
        numsteps = -1 if event.delta() > 0 else 1
        self.setCurrentIndex( (self.currentIndex() + numsteps) % count )

class DiodeWidget( QtWidgets.QWidget ):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self._diode_colors = ['red', 'blue']
        self._orientation = 0
    
        self._color_map = dict( red=QtCore.Qt.red,
                                green=QtCore.Qt.green,
                                blue=QtCore.Qt.blue,
                                intensity=QtCore.Qt.yellow )
    
    def setOrientation(self,val):
        self._orientation = float(val)
        self.update()
    
    def setDiodeColors(self,val):
        if not isinstance(val, (tuple,list)):
            val = [val,]
        
        if not all( x in ('red','green','blue', 'intensity') for x in val ):
            raise ValueError('Invalid color')
        
        if len(val)>2:
            raise ValueError('Only support for 0-2 diodes')
        
        self._diode_colors = val
        self.update()
    
    def paintEvent(self, event):
        
        paint = QtGui.QPainter()
        
        paint.begin(self)
        
        paint.setPen( QtGui.QPen( QtCore.Qt.NoPen) )
        paint.setBrush(QtCore.Qt.white)
        paint.drawRect( event.rect() )
        
        w = event.rect().width()
        h = event.rect().height()
        
        xc = w/2
        yc = h/2
        
        diode_size = 5
        center_size = 2
        
        triangle_size = min(w,h)/3
        p = QtGui.QPainterPath()
        p.moveTo( xc, yc-triangle_size )
        p.lineTo( xc + np.cos( np.pi/3 + 0.5*np.pi ) * triangle_size, yc + np.sin(np.pi/3 + 0.5*np.pi) * triangle_size )
        p.lineTo( xc - np.cos( np.pi/3 + 0.5*np.pi ) * triangle_size, yc + np.sin(np.pi/3 + 0.5*np.pi) * triangle_size )
        p.lineTo( xc, yc-triangle_size )
        
        paint.setPen( QtGui.QPen( QtCore.Qt.NoPen) )
        paint.fillPath( p, QtGui.QBrush( QtCore.Qt.gray ) )
        
        paint.setPen( QtCore.Qt.black )
        paint.setBrush( QtCore.Qt.black )
        paint.drawEllipse( QtCore.QPoint(xc,yc), center_size, center_size )
        
        if len(self._diode_colors)==0:
            pass
        elif len(self._diode_colors)==1:
            
            paint.setPen( self._color_map[self._diode_colors[0]] )
            paint.setBrush( self._color_map[self._diode_colors[0]] )
            paint.drawEllipse( QtCore.QPoint(xc,yc), diode_size, diode_size )
        
        else:
            
            radius = min(w,h)*3./8.
            
            x1 = xc + np.cos(self._orientation + 0.5*np.pi) * radius
            y1 = yc - np.sin(self._orientation + 0.5*np.pi) * radius
            
            x2 = xc - np.cos(self._orientation + 0.5*np.pi) * radius
            y2 = yc + np.sin(self._orientation + 0.5*np.pi) * radius
            
            paint.setPen( QtCore.Qt.black )
            paint.drawLine( x1,y1,x2,y2)
            
            paint.setPen( self._color_map[self._diode_colors[0]] )
            paint.setBrush( self._color_map[self._diode_colors[0]] )
            paint.drawEllipse( QtCore.QPoint(x1,y1), diode_size, diode_size )
            
            paint.setPen( self._color_map[self._diode_colors[1]] )
            paint.setBrush( self._color_map[self._diode_colors[1]] )
            paint.drawEllipse( QtCore.QPoint(x2,y2), diode_size, diode_size )
        
        paint.end()


class SignalWait(QtCore.QObject):
    def __init__(self,signal):
        
        QtCore.QObject.__init__(self)
        
        signal.connect( self.finish )
        self._done = False
        self.result = None
    
    def finish(self, *args):
        self.result = args
        self._done = True
        
    def wait(self, timeout=0):
        app = QtCore.QCoreApplication.instance()
        
        if timeout>0:
            timer = QtCore.QTime(0,0,0,timeout)
            timer.start();
        
        while( not self._done and (timeout==0 or timer.elapsed()<timeout) ):
            app.processEvents()
        
        return (self._done,) + self.result


def create_pen( width=1, cosmetic=True, color=[255,0,0], style=1, alpha=1.0 ):
    
    pen = QtGui.QPen()
    pen.setCosmetic(cosmetic)
    pen.setWidth(width)
    col = QtGui.QColor(*color)
    col.setAlphaF( alpha )
    pen.setColor( col )
    pen.setStyle( style )
    return pen 


def main(app):
    view = GraphicsCanvas()
    
    view.scene().setSceneRect(QtCore.QRectF(-100, -100, 500, 500))
    
    items=[]
    items.append(iRectangle( x=0, y=0, width=200, height=100 ))
    items.append(iTriangle( x=200, y=200, radius=100 ))
    items.append(iPentagon( x=100, y=100, radius=100 ))
    items.append(iDiamond( x=200, y=300, radius=150 ))
    items.append(iEllipse( x=200, y=200, width=200, height=100 ))
    items.append(iShape( [[100,100.0],[150,200],[200,50],[100,25]], closed=False ) )
    items.append(iGraph( nodes = [ ['a',[100,250]],['b',[100,150]], ['c',[50,50]], ['d',[150,50]] ], edges = [ ['a',[[80,220],[120,180]],'b'], ['b',[],'c'], ['b',[[110,120],[140,80]],'d'] ] ))
    
    for k in items:
        view.scene().addItem(k)
        
    view.show()
    
    return app.exec_()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    sys.exit(main(app))
