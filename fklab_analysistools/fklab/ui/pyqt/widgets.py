import numpy as np

from fklab.ui.pyqt.core import *

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


