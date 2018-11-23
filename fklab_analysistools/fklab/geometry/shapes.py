"""
=================================================
Geometrical shapes (:mod:`fklab.geometry.shapes`)
=================================================

.. currentmodule:: fklab.geometry.shapes

Classes for working with geometrical shapes, including methods for
projecting 2D points to the shape outline (e.g. as used for behavioral
track linearization).

Shape classes
=============

.. autosummary::
    :toctree: generated/
    
    ellipse
    rectangle
    polyline
    polygon
    graph
    multishape

Polygon creation
================

.. autosummary::
    :toctree: generated/
    
    ngon
    triangle
    pentagon
    hexagon
    
"""

from __future__ import division

import abc
import collections

import numpy as np
import scipy as sp
import scipy.special
import scipy.interpolate
import scipy.spatial

from . import transforms as tf
from . import utilities as util

import fklab.statistics.circular as circ
import fklab.utilities.yaml as yaml

__all__ = ['ellipse','rectangle','polyline','polygon','graph','multishape','ngon','triangle','pentagon','hexagon']

#class hierarchy
# shape <- path <- solid <- boxed <- (ellipse,rectangle)
# shape <- path <- (polyline, graph)
# shape <- path <- (polyline,solid) <- polygon
# shape <- multishape

# we need to create a single common meta base class
class meta(yaml.YAMLObjectMetaclass, abc.ABCMeta):
    pass

class shape(yaml.YAMLObject):
    __metaclass__ = meta
    
    def __init__(self,name=''):
        self.name = name
    
    @abc.abstractmethod
    def __repr__(self):
        return 'abstract shape'
    
    @property
    def name(self):
        """Name of shape."""
        return self._name
    
    @name.setter
    def name(self,value):
        self._name = str(value)
    
    @abc.abstractproperty
    def center(self):
        """Returns center location of shape."""
        return np.zeros(2)
    
    @abc.abstractmethod
    def scale(self,factor):
        """Scale shape.
        
        Parameters:
            factor : scalar or [factor_x, factor_y]
        
        """
        pass
    
    @abc.abstractmethod
    def rotate(self,angle):
        """Rotate shape around its center.
        
        Parameters
        ----------
        angle : scalar
        
        """
        pass
    
    @abc.abstractmethod
    def translate(self,offset):
        """Translate shape.
        
        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]
        
        """
        pass
    
    @abc.abstractmethod
    def transform(self,tform):
        """Transform shape.
        
        Parameters
        ----------
        tform : Transform
        
        """
        pass
    
    @abc.abstractproperty
    def boundingbox(self):
        """Bounding box of shape."""
        pass
    
    @property
    def ispath(self):
        """Test if shape is a path."""
        return isinstance(self,path)
    
    @property
    def issolid(self):
        """Test if shape is a solid."""
        return isinstance(self,solid)
        
class path(shape):
    @abc.abstractproperty
    def pathlength(self):
        """Return the path length or circumference of shape."""
        return 0

    @abc.abstractmethod
    def point2path(self,points):
        """Project points to shape path."""
        pass
    
    @abc.abstractmethod
    def path2point(self,points):
        """Unproject points on shape path."""
        pass
    
    @abc.abstractmethod
    def path2edge(self,x):
        """Convert between path and edge representations."""
        pass
    
    @abc.abstractmethod
    def edge2path(self,x):
        """Convert between edge and path representations."""
        pass
    
    @abc.abstractmethod
    def tangent(self,x):
        """Compute path tangent at given locations along path."""
        pass
    
    def normal(self,x):
        """Compute path normal at given locations along path."""
        return self.tangent(x) - 0.5*np.pi

    def gradient(self,x,dx=1):
        """Compute gradient at given locations along path."""
        #convert to dist_along_path if given as (edge,dist_along_edge) tuple
        if isinstance(x,tuple):
            x = self.edge2path(x)
        
        return gradient(x,dx)
    
    @abc.abstractmethod
    def random_on(self,n):
        """Draw random coordinates on path."""
        pass
    
    def random_along(self,n=1):
        """Draw uniform random locations along path."""
        return np.random.uniform(low=0,high=self.pathlength,size=n)
    
    def distance(self,x,y):
        """Compute distance along path.
        
        Parameters
        ----------
        x,y : array or tuple
            distance along path or (edge index, distance along edge) tuple
        
        Returns
        -------
        ndarray
            element-wise distance between `x` and `y`
        
        """
        if isinstance(x,tuple):
            x = self.edge2path(x)
        if isinstance(y,tuple):
            y = self.edge2path(y)
        
        return y-x
        
    @abc.abstractmethod
    def samplepath(self,oversampling=None):
        """Densely sample points along path."""
        pass

class solid(path):
    @abc.abstractproperty
    def area(self):
        """Returns area of closed shape."""
        return 0
    
    @abc.abstractmethod
    def contains(self,points):
        """Tests if points are contained within shape."""
        return False
    
    @abc.abstractmethod
    def random_in(self,n):
        """Draw random points inside shape."""
        pass
    
    def gradient(self,x,dx=1):
        """Compute gradient at given locations along path.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        dx : scalar
        
        Returns
        -------
        ndarray
        
        """
        #convert linear position to (pseudo) angle and unwrap
        f = 2*np.pi/self.pathlength
        x = np.unwrap( f*x ) / f
        #compute gradient
        g = np.gradient(x, dx)
        return g
    
    def distance(self,x,y):
        """Compute distance along path.
        
        Parameters
        ----------
        x,y : array or tuple
              distance along path or (edge index, distance along edge) tuple
        
        Returns
        -------
        ndarray
            element-wise distance between `x` and `y`
        
        """
        return circ.diff(x,y,low=0.0,high=self.pathlength,directed=True)

class boxed(solid):
    def __init__(self, center=[0,0], size=1, orientation=0, **kwargs):
        super(boxed,self).__init__(**kwargs)
        self.center = center
        self.size = size
        self.orientation = orientation
    
    @property
    def center(self):
        """Center of shape."""
        return self._center
    
    @center.setter
    def center(self,value):
        try:
            value = np.array(value,dtype=np.float64)
            value = value.reshape( (2,) )
        except:
            raise TypeError
        
        self._center = value
    
    @property
    def size(self):
        """Size of shape."""
        return self._size
    
    @size.setter
    def size(self,value):
        try:
            value = np.array(value,dtype=np.float64)
            value = value.reshape( (value.size,) )
        except:
            raise TypeError
        
        if value.size<1 or value.size>2:
            raise TypeError
        
        if value.size==2 and value[0]==value[1]:
            value = value[0:1]
        
        self._size = value
    
    @property
    def orientation(self):
        """Orientation of shape."""
        return self._orientation
    
    @orientation.setter
    def orientation(self,value):
        try:
            value = np.array(value,dtype=np.float64)
            value = value.reshape( (1,) )
        except:
            raise TypeError
        
        value = np.mod(value,2*np.pi)
        
        self._orientation = value
    
    def scale(self,factor):
        """Scale shape.
        
        Parameters:
            factor : scalar or [factor_x, factor_y]
        
        """
        factor = np.array(factor, dtype=np.float64)
        self.size = self._size*factor
    
    def rotate(self,angle):
        """Rotate shape around its center.
        
        Parameters
        ----------
        angle : scalar
        
        """
        angle = np.array(angle, dtype=np.float64)
        self.orientation = self._orientation + angle
    
    def translate(self,offset):
        """Translate shape.
        
        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]
        
        """
        offset = np.array(offset, dtype=np.float64)
        self.center = self._center + offset
    
    def boxvertices(self):
        """Vertices of enclosing box."""
        v = np.array( [[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]] )
        tform = tf.Scale(self.size) + tf.Rotate(self.orientation) + tf.Translate(self.center)
        v = tform.transform(v)
        return v

class ellipse(boxed):
    """Ellipse shape.
    
    Parameters
    ----------
    center : (2,) array_like
        x and y coordinates of ellipse center
    size : scalar or (2,) array_like
        length of ellipse major and minor axes
    orientation : float, optional
        orientation of the ellipse in radians (default is 0)
    name : str, optional
        name of the shape (default is "")
    
    Attributes
    ----------
    name
    center
    size
    orientation
    boundingbox
    ispath 
    issolid
    iscircle
    pathlength
    area
    eccentricity
    
    Notes
    -----
    An ellipse is represented by the equations (canonical form):
    
    .. math::
    
        x = a*cos(t) \\
        y = b*sin(t)
    
    The starting point of an ellipse is at (a,0). The angles increase
    counter-clockwise. At the starting point, the tangent angle is
    1.5 :math:`\pi` and the normal is 0.
    
    """
    _inv_elliptic = None #cached elleptic integral interpolator
    
    yaml_tag=u'!ellipse_shape'
    
    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent ellipse as YAML."""
        d = collections.OrderedDict()
        d['center'] = data.center.tolist()
        d['size'] = data.size.tolist()
        d['orientation'] = float(data.orientation)
        node = dumper.represent_mapping(cls.yaml_tag, d.iteritems())
        return node
    
    @classmethod
    def from_yaml(cls, loader, node):
        """Construct ellipse from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(center=d['center'], size=d['size'], orientation=d['orientation'])
    
    def __repr__(self):
        if self.iscircle:
            return 'circle (radius={radius[0]}, center=[{center[0]},{center[1]}])'.format( center=self.center, radius=self.size )
        else:
            return 'ellipse (size=[{size[0]},{size[1]}], center=[{center[0]},{center[1]}], orientation={orientation[0]} rad)'.format( size = self.size, center=self.center, orientation=self.orientation )
    
    def _get_inv_elliptic(self):
        if self._inv_elliptic is None or np.any(self._inv_elliptic[0]!=self._size):
            #update cached inverse elliptic integral interpolator
            self._inv_elliptic = ( self._size.copy(), _inv_ellipeinc(self._size[0], self._size[1]) )
        
        return self._inv_elliptic[1]
        
    @property
    def iscircle(self):
        """True if shape is a circle (i.e. major and minor axes are equal).
        """
        return self._size.size==1
    
    @property
    def eccentricity(self):
        """Returns eccentricity of ellipse.
        
        The eccentricity can be thought of as a measure of how much an
        ellipse deviates from being circular. A circle has eccentricity
        of 0 and an ellipse has eccentricity between 0 and 1.
        """
        if self.iscircle:
            return 0
        else:
            a = np.max(self._size)
            b = np.min(self._size)
            return np.sqrt( 1 - ( b/a )**2 )
    
    def transform(self,tform):
        """Applies transformation to ellipse.
        
        The ellipse is first approximated by a polygon before the
        transformation is applied.
        
        Parameters
        ----------
        tform : Transform
            the transformation to apply
        
        Returns
        -------
        polygon
        
        """
        p = self.aspolygon()
        p.transform(tform)
        return p
    
    @property
    def boundingbox(self):
        """Axis aligned bounding box of ellipse.
        """
        if self.iscircle:
            return rectangle( center=self._center, size=np.ones(2)*2*self._size )
        else:
            #parameterized equations for ellipse:
            #x = center[0] + size[0]*cos(t)*cos(orientation) - size[1]*sin(t)*sin(orientation)
            #y = center[1] + size[1]*sin(t)*cos(orientation) + size[0]*cos(t)*sin(orientation)
            #then solve dx/dt=0 and dy/dt=0
            #resulting in (for x): t = atan( -size[1]*tan(orientation)/size[0] ) + n*PI
            #and (for y): t = atan( size[1]*cot(orientation)/size[0] ) + n*PI
            #plugging in `t` into parameterized equations will give the extreme
            #x and y coordinates of the ellipse (and thus of the bounding box)
            t = np.arctan( -self._size[1] * np.tan( self._orientation ) / self._size[0] )
            x = self._center[0] + self._size[0]*np.cos([t,t-np.pi])*np.cos(self._orientation) - self._size[1]*np.sin([t,t-np.pi])*np.sin(self._orientation)
            if self.orientation == 0:
                t = np.pi
            else:
                t = np.arctan(  self._size[1] / ( np.tan( self._orientation ) * self._size[0] ) )
            y = self._center[1] + self._size[1]*np.sin([t,t-np.pi])*np.cos(self._orientation) + self._size[0]*np.cos([t,t-np.pi])*np.sin(self._orientation)
            return rectangle( center=[(x[0]+x[1])/2,(y[0]+y[1])/2], size=np.abs([x[1]-x[0],y[1]-y[0]]), orientation=0 )
    
    @property
    def pathlength(self):
        """Circumference of ellipse.
        """
        if self.iscircle:
            return 2*np.pi*self._size[0]
        else:
            return 4*np.max(self._size)*sp.special.ellipe( self.eccentricity**2 )
    
    def point2path(self,points): #OK
        """Project points to ellipse circumference.
        
        Parameters
        ----------
        points : (n,2) array
        
        Returns
        -------
        dist_along_path : array
            distance along the circumference to the projected point
        dist_to_path: array
            distance between the original point and the project point on
            the circumference
        point_on_path: array
            coordinates of projected point on circumference
        edge: tuple
            representation of projected point as edge index and distance
            along edge. An ellipse consists of only a single edge with
            index 0.
        
        Notes
        -----
        Ellipses (but not circles) are first approximated by a polygon.
        
        """
        if self.iscircle:
            [theta,rho] = util.cart2pol( points[:,0]-self._center[0], points[:,1]-self._center[1] )
            dist_along_path = circ.wrap(theta - self.orientation)*self._size
            dist_to_path = rho - self._size
            point_on_path = np.vstack((self._size*np.cos(theta) + self.center[0],self._size*np.sin(theta) + self.center[1])).T
        else:
            #approximate for ellipse
            p = self.aspolygon(oversampling=100)
            (dist_along_path,dist_to_path,point_on_path) = p.point2path(points)[0:3]
            #correct linear distance
            dist_along_path = dist_along_path * self.pathlength / p.pathlength
        
        return (dist_along_path,dist_to_path,point_on_path,(np.zeros(dist_along_path.shape,dtype=np.int),dist_along_path))
    
    def path2point(self,x,distance=0,_method=1): #OK
        """Converts points along circumference to 2d coordinates.
        
        Parameters
        ----------
        x : array or tuple
            distance along circumference or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from circumference (default is 0)
        
        Returns
        -------
        xy : (n,2) array
            x,y coordinates
        
        """
        #convert to distance along path, if given as (edge,dist_along_edge) tuple
        if isinstance(x,tuple):
            x = x[1]
        
        if self.iscircle:
            [x,y] = util.pol2cart( x/self._size + self.orientation, self._size + distance )
            xy = np.vstack( (x+self.center[0],y+self.center[1]) ).T
            return xy
        elif _method==1:
            #transform x to ellipse parameter t
            fcn = self._get_inv_elliptic()
            t = fcn( x )
            #compute x and y coordinates
            xy = np.vstack( ((self._size[0]+distance) * np.cos(t), (self._size[1]+distance) * np.sin(t)) ).T
            tform = tf.Rotate( self.orientation ) + tf.Translate( self.center )
            xy = tform.transform(xy)
            return xy
        else:
            p = self.aspolygon(oversampling=100)
            x = x * p.pathlength / self.pathlength
            return p.path2point(x,distance)
    
    def path2edge(self,x):
        """Converts path to edge representation.
        
        Parameters
        ----------
        x : array
            distance along ellipse circumference
        
        Returns
        -------
        tuple
            representation of point on circumference as edge index and 
            distance along edge. An ellipse consists of only a single
            edge with index 0.
            
        """
        return (np.zeros(x.shape,dtype=np.int),x)
    
    def edge2path(self,x):
        """Converts edge to path representation.
        
        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            ellipse circumference
        
        Returns
        -------
        array
            distance along ellipse circumference
        
        """
        if not isinstance(x,tuple):
            raise ValueError()
        return x[1]
    
    def tangent(self,x): #OK
        """Computes tangential angle at points along path.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        
        Returns
        -------
        array
            tangential angle in radians
        
        """
        
        #convert to distance along path, if given as (edge,dist_along_edge) tuple
        if isinstance(x,tuple):
            x = x[1]
        
        if self.iscircle:
            d = circ.wrap( x / self._size + 0.5*np.pi + self.orientation )
        else:
            #transform to ellipse parameter t
            fcn = self._get_inv_elliptic()
            t = fcn( x )
            #compute tangent angle
            d = circ.wrap( np.arctan2( self._size[1] * np.cos(t) , -np.sin(t) * self._size[0] ) + self.orientation )
        
        return d
    
    def samplepath(self,oversampling=20,openpath=False): #OK
        """Regular sampling of points on circumference.
        
        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along circumference. If oversampling is 1, four points are
            sampled. In general, 4*oversampling points are sampled 
            (default is 20)
        openpath : bool, optional
            Whether or not to leave sampled path open. If False, the
            first and last sampled point will be identical
            (default is False).
        
        Returns
        -------
        array
            x,y coordinates of sampled points
        
        """
        npoints = oversampling*4
        
        if not openpath:
            npoints+=1
        
        vertices = np.empty( (npoints,2), dtype=np.float64 )
        vertices[:,0], vertices[:,1] = util.pol2cart( np.pi * 2 * np.linspace(0,1,num=npoints,endpoint=not openpath), 1.0 )
        
        t = tf.Scale( self._size ) + tf.Rotate( self._orientation ) + tf.Translate( self._center )
        vertices = t.transform( vertices )
        
        return vertices
    
    def random_on(self,n): #OK
        """Samples random points on circumference.
        
        Parameters
        ----------
        n : int
            number of randomly sampled points
        
        Returns
        -------
        array
            x,y coordinates of randomly sampled points
        
        """
        if self.iscircle:
            #generate random points on circle
            x,y = util.pol2cart( np.random.uniform( low=0, high=2*np.pi, size=n), self._size[0] );
            xy = np.vstack( (x, y) ).T
        else:
            #generate uniform random arc lengths
            L = np.random.uniform(low=0,high=self.pathlength,size=n)
            #transform to ellipse parameter t
            fcn = self._get_inv_elliptic()
            t = fcn( L )
            #compute x and y coordinates
            xy = np.vstack( (self._size[0] * np.cos(t), self._size[1] * np.sin(t)) ).T
        
        #transform points
        tform = tf.Rotate( self._orientation ) + tf.Translate( self._center )
        xy = tform.transform( xy )
            
        return xy
    
    @property
    def area(self):
        """Surface area of ellipse.
        """
        return np.pi*self._size[0]*self._size[-1]
    
    def contains(self,points):
        """Test if points are contained within ellipse.
        
        Parameters
        ----------
        points : array_like
            x,y coordinates of points to be tested.
        
        Returns
        -------
        bool array
            True if point is contained within ellipse.
        
        """
        
        points = np.array( points )
        if points.ndim==1 and points.size==2:
            points = points.reshape((1,2))

        assert points.ndim==2 and points.shape[1]==2
        
        if self.iscircle:
            return np.sum( (points - self._center)**2, axis=1 ) <= self._size**2
        else:
            #transform points: translate(-center), rotate(-orientation), scale(1/size)
            #test if distance of point to (0,0) <= 1
            t = tf.Translate( -self._center ) + tf.Rotate( -self._orientation ) + tf.Scale( 1/self._size )
            points = t.transform(points)
            return np.sum( points**2, axis=1 ) <= 1
    
    def random_in(self,n):
        """Samples random points inside ellipse.
        
        Parameters
        ----------
        n : int
            number of randomly sampled points
        
        Returns
        -------
        array
            x,y coordinates of randomly sampled points
        
        """
        #generate random points in unit circle
        x,y = util.pol2cart( np.random.uniform( low=0, high=2*np.pi, size=n), np.sqrt( np.random.uniform(low=0,high=1,size=n) ) );
        #transform points
        t = tf.Scale( self._size ) + tf.Rotate( self._orientation ) + tf.Translate( self._center )
        xy = t.transform( np.vstack( (x, y) ).T )
        return xy
        
    def aspolygon(self,oversampling=20):
        """Convert ellipse to polygon.
        
        Parameters
        ----------
        oversampling : int
        
        Returns
        -------
        polygon
        
        """
        vertices = self.samplepath(oversampling=oversampling,openpath=True)
        return polygon( vertices=vertices, spline=False )

class rectangle(boxed):
    """Rectangle shape.
    
    Parameters
    ----------
    center : (2,) array_like
        x and y coordinates of rectangle center
    size : scalar or (2,) array_like
        major and minor axes of rectangle
    orientation : float, optional
        orientation of the rectangle in radians (default is 0)
    name : str, optional
        name of the shape (default is "")
    
    Attributes
    ----------
    name
    center
    size
    orientation
    boundingbox
    ispath 
    issolid
    issquare
    pathlength
    area
    
    """
    
    yaml_tag=u'!rectangle_shape'
    
    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent rectangle as YAML."""
        d = collections.OrderedDict()
        d['center'] = data.center.tolist()
        d['size'] = data.size.tolist()
        d['orientation'] = float(data.orientation)
        node = dumper.represent_mapping(cls.yaml_tag, d.iteritems())
        return node
    
    @classmethod
    def from_yaml(cls, loader, node):
        """Construct rectangle from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(center=d['center'], size=d['size'], orientation=d['orientation'])
    
    def __repr__(self):
        if self.issquare:
            return 'square (size={size[0]}, center=[{center[0]},{center[1]}], orientation={orientation[0]} rad)'.format(size=self.size,center=self.center,orientation=self.orientation)
        else:
            return 'rectangle (size=[{size[0]},{size[1]}], center=[{center[0]},{center[1]}], orientation={orientation[0]} rad)'.format(size=self.size,center=self.center,orientation=self.orientation)
    
    @property
    def issquare(self):
        """True if shape is a square (i.e. major and minor axes are equal).
        """
        return len(self._size)==1

    def transform(self,tform):
        """Applies transformation to rectangle.
        
        The rectangle is first approximated by a polygon before the
        transformation is applied.
        
        Parameters
        ----------
        tform : Transform
            the transformation to apply
        
        Returns
        -------
        polygon
        
        """
        p = self.aspolygon()
        p.transform(tform)
        return p

    @property
    def boundingbox(self):
        """Axis aligned bounding box of rectangle.
        """
        vertices = self.boxvertices
        maxval = np.max( vertices, axis=0 )
        minval = np.min( vertices, axis=0 )
        return rectangle( center=(maxval+minval)/2.0, size=maxval-minval, orientation=0 )
    
    @property
    def pathlength(self):
        """Perimeter of rectangle.
        """
        return 2*(self._size[0] + self._size[-1])
    
    def point2path(self,*args,**kwargs):
        """Project points to rectangle perimeter.
        
        Parameters
        ----------
        points : (n,2) array
        
        Returns
        -------
        dist_along_path : array
            distance along the perimeter to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the perimeter
        point_on_path: array
            coordinates of projected point on perimeter
        edge: tuple
            representation of projected point as edge index and distance
            along edge. A rectangle consists of four edges.
        
        """
        return self.aspolygon().point2path(*args,**kwargs)
    
    def path2point(self,*args,**kwargs):
        """Converts points along perimeter to 2d coordinates.
        
        Parameters
        ----------
        x : array or tuple
            distance along perimeter or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from perimeter (default is 0)
        
        Returns
        -------
        xy : (n,2) array
            x,y coordinates
        
        """
        return self.aspolygon().path2point(*args,**kwargs)
    
    def edge2path(self,*args,**kwargs):
        """Converts edge to path representation.
        
        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            rectangle perimeter
        
        Returns
        -------
        array
            distance along rectangle perimeter
        
        """
        return self.aspolygon().edge2path(*args,**kwargs)
    
    def path2edge(self,*args,**kwargs):
        """Converts path to edge representation.
        
        Parameters
        ----------
        x : array
            distance along rectangle perimeter
        
        Returns
        -------
        tuple
            representation of point on perimeter as edge index and 
            distance along edge.
            
        """
        return self.aspolygon().path2edge(*args,**kwargs)
    
    def tangent(self,*args,**kwargs):
        """Computes tangential angle at points along path.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        
        Returns
        -------
        array
            tangential angle in radians
        
        """
        return self.aspolygon().tangent(*args,**kwargs)
    
    def samplepath(self,oversampling=1,openpath=False):
        """Regular sampling of points on perimeter.
        
        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along circumference. If oversampling is 1, four points are
            sampled. In general, 4*oversampling points are sampled 
            (default is 1)
        openpath : bool, optional
            Whether or not to leave sampled path open. If False, the
            first and last sampled point will be identical
            (default is False).
        
        Returns
        -------
        array
            x,y coordinates of sampled points
        
        """
        
        npoints = np.floor(oversampling)*4
        
        if not openpath:
            npoints+=1
        
        #construct vertices, the first edge is the right-most edge
        vertices = np.array( [[0.5,-0.5],[0.5,0.5],[-0.5,0.5],[-0.5,-0.5],[0.5,-0.5]] )
        fcn = sp.interpolate.interp1d( np.arange(5), vertices, kind='linear', axis=0, bounds_error = False )
        
        x = np.linspace(0,4,num=npoints,endpoint=not openpath)
        
        vertices = fcn(x)
        
        t = tf.Scale( self._size ) + tf.Rotate( self._orientation ) + tf.Translate( self._center )
        vertices = t.transform( vertices )
        
        return vertices
    
    def random_on(self,n):
        """Samples random points on perimeter.
        
        Parameters
        ----------
        n : int
            number of randomly sampled points
        
        Returns
        -------
        array
            x,y coordinates of randomly sampled points
        
        """
        
        ratio = self._size[0] / np.sum(self._size)
        nx = np.round( ratio * n )
        
        xy = np.random.uniform(low=-0.5,high=0.5,size=(n,2))

        xy[0:nx,1] = np.random.randint(2,size=nx  )-0.5
        xy[nx:,0]  = np.random.randint(2,size=n-nx)-0.5
        
        t = tf.Scale(self._size) + tf.Rotate(self._orientation) + tf.Translate(self._center)
        xy = t.transform(xy)
        
        return xy
    
    @property
    def area(self):
        """Surface area of rectangle.
        """
        return self._size[0]*self._size[-1]
    
    def contains(self,*args,**kwargs):
        """Test if points are contained within rectangle.
        
        Parameters
        ----------
        points : array_like
            x,y coordinates of points to be tested.
        
        Returns
        -------
        bool array
            True if point is contained within rectangle.
        
        """
        return self.aspolygon().contains(*args,**kwargs)
    
    def random_in(self,n):
        """Samples random points inside rectangle.
        
        Parameters
        ----------
        n : int
            number of randomly sampled points
        
        Returns
        -------
        array
            x,y coordinates of randomly sampled points
        
        """
        #create random points in unit square
        xy = np.random.uniform(low=-0.5,high=0.5,size=(n,2))
        t = tf.Scale(self._size) + tf.Rotate(self._orientation) + tf.Translate(self._center)
        xy = t.transform(xy)
        return xy
    
    def aspolygon(self,oversampling=1):
        """Convert rectangle to polygon.
        
        Parameters
        ----------
        oversampling : int
        
        Returns
        -------
        polygon
        
        """
        vertices = self.samplepath(oversampling=oversampling,openpath=True)
        return polygon( vertices=vertices, spline=False )

class polyline(path):
    """Polyline shape.
    
    Parameters
    ----------
    vertices : (n,2) array
    spline : bool
        Apply spline interpolation.
    name : str, optional
        name of the shape (default is "")
    
    Attributes
    ----------
    name
    center
    boundingbox
    numvertices
    vertices
    isspline
    ispath 
    issolid
    pathlength
    edgelengths
        
    """
    
    _length = None #cached path length
    _isspline = False #True if interpolated spline
    _vertices = [] #(n,2) array of vertices
    _edge_lengths = []
    _path_integral = []
    
    yaml_tag=u'!polyline_shape'
    
    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent polyline as YAML."""
        d = collections.OrderedDict()
        d['vertices'] = data.vertices.tolist()
        d['spline'] = data.isspline
        node = dumper.represent_mapping(cls.yaml_tag, d.iteritems())
        return node
    
    @classmethod
    def from_yaml(cls, loader, node):
        """Construct polyline from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(vertices=d['vertices'], spline=d['spline'] )
    
    def __init__(self,vertices=[],spline=False,**kwargs):
        super(polyline,self).__init__(**kwargs)
        self.vertices = vertices
        self.isspline = spline
    
    def __repr__(self):
        return '{shapetype} {klass} with {n} vertices'.format(shapetype='spline' if self.isspline else 'straight', klass=self.__class__.__name__, n=self.numvertices) 
    
    @property
    def numvertices(self):
        """Number of polyline vertices."""
        return self._vertices.shape[0]
    
    @property
    def center(self):
        """Center of shape."""
        return np.mean( self._vertices, axis=0 )
    
    def _get_expanded_vertices(self,spline=False):
        if spline:
            return self._sampled_spline[0]
        elif self.issolid:
            return np.concatenate( (self._vertices, self._vertices[0:1,:]), axis=0 )
        else:
            return self._vertices
        
    def _update_cached_values(self):
        
        vertices = self._get_expanded_vertices(spline=False)
        
        if self.isspline:
            self._sampled_spline, self._spline = _sample_spline(vertices,oversampling=50,closed=self.issolid,openpath=False)
            L = np.sqrt( np.sum( np.diff( self._sampled_spline[0], n=1, axis=0 )**2, axis=1 ) )
            pathlength = np.sum(L)
        else:
            try:
                del self._sampled_spline
                del self._spline
            except:
                pass
             
            if self.numvertices<2:
                pathlength = 0
                edge_lengths = np.array([])
            else:
                L = np.sqrt( np.sum( np.diff( vertices, n=1, axis=0 )**2, axis=1 ) )
                pathlength = np.sum( L )
        
        self._length = pathlength
        self._path_integral = np.cumsum( np.concatenate( ([0], L ) ) )
        self._path_integral[-1] = pathlength
        
        if self.isspline:
            self._edge_lengths = np.diff( sp.interpolate.interp1d( self._sampled_spline[1], self._path_integral, kind='linear' )(self._spline[1]) )
        else:
            self._edge_lengths = L
        
    @property
    def vertices(self):
        """Polyline vertices."""
        return self._vertices
    
    @vertices.setter
    def vertices(self,value):
        value = np.array(value,dtype=np.float64)
        
        if value.size == 0:
            value = np.zeros( (0,2) )
        elif value.ndim!=2 or value.shape[1]!=2:
            raise TypeError()
        
        self._vertices = value
        self._update_cached_values()
    
    @property
    def isspline(self):
        """Whether spline interpolation is enabled."""
        return self._isspline and self.numvertices>3
    
    @isspline.setter
    def isspline(self,value):
        self._isspline = bool(value)
        self._update_cached_values()

    def scale(self,factor):
        """Scale shape.
        
        Parameters:
            factor : scalar or [factor_x, factor_y]
        
        """
        t = tf.Scale(factor=factor,origin=self.center)
        self._vertices = t.transform(self._vertices)
    
    def rotate(self,angle):
        """Rotate shape around its center.
        
        Parameters
        ----------
        angle : scalar
        
        """
        t = tf.Rotate(angle=angle,origin=self.center)
        self._vertices = t.transform(self._vertices)
    
    def translate(self,offset):
        """Translate shape.
        
        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]
        
        """
        t = tf.Translate(offset=offset)
        self._vertices = t.transform(self._vertices)
    
    def transform(self,tform):
        """Transform shape.
        
        Parameters
        ----------
        tform : Transform
        
        """
        self._vertices = tform.transform(self._vertices)
    
    @property
    def boundingbox(self):
        """Axis-aligned bounding box of polyline."""
        vertices = self._get_expanded_vertices(self.isspline)
        maxval = np.max( vertices, axis=0 )
        minval = np.min( vertices, axis=0 )
        return rectangle( center=(maxval+minval)/2.0, size=maxval-minval, orientation=0 )
    
    @property
    def pathlength(self):
        """Path length of polyline."""
        return self._length
       
    def point2path(self,points,clip=('normal','normal')): #OK
        """Project points to polyline.
        
        Parameters
        ----------
        points : (n,2) array
        clip : 2-element tuple
            Clipping behavior for the two polyline end points.
            'normal' = points beyond line segment are projected to the line end points
            'blunt' = points beyond line segment are excluded
            'full' = points are projected to the full line that passes through the end points
        
        Returns
        -------
        dist_along_path : array
            distance along the path to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the path
        point_on_path: array
            coordinates of projected point on path
        edge: tuple
            representation of projected point as edge index and distance
            along edge.
        
        """
        vertices = self._get_expanded_vertices(self.isspline)
        (dist_to_path, point_on_path, edge, dist_along_edge, dist_along_path) = util.point2polyline(vertices, points, clip)
        return (dist_along_path, dist_to_path, point_on_path, (edge,dist_along_edge))
    
    def path2point(self,x,distance=0): #OK
        """Converts points along path to 2d coordinates.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from path (default is 0)
        
        Returns
        -------
        xy : (n,2) array
            x,y coordinates
        
        """
        
        if isinstance(x,tuple):
            x = self.edge2path(x)
        
        vertices = self._get_expanded_vertices(spline=self.isspline)
        L = self._path_integral
        
        xy = scipy.interpolate.interp1d( L, vertices, kind='linear',axis=0 )(x) #missing: extrapolation!
        
        distance = np.asarray(distance)
        
        if np.any(distance!=0):
            normal = self.normal(x)
            xy[:,0] += distance * np.cos(normal)
            xy[:,1] += distance * np.sin(normal)
        
        return xy
    
    def path2edge(self,x):
        """Converts path to edge representation.
        
        Parameters
        ----------
        x : array
            distance along path
        
        Returns
        -------
        tuple
            representation of point on path as edge index and 
            distance along edge.
            
        """
        return _path2edge(x,self.edgelengths,self.pathlength)
    
    def edge2path(self,x):
        """Converts edge to path representation.
        
        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            path
        
        Returns
        -------
        array
            distance along path
        
        """
        return _edge2path(x,self.edgelengths,self.pathlength)
    
    @property
    def edgelengths(self):
        """Lengths of polyline edges."""
        return self._edge_lengths
    
    def tangent(self,x): #OK?
        """Computes tangential angle at points along path.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        
        Returns
        -------
        array
            tangential angle in radians
        
        """
        if self.isspline:
            if isinstance(x,tuple):
                x = self.edge2path(x)
            dx,dy = sp.interpolate.splev( sp.interpolate.interp1d(self._path_integral,self._sampled_spline[1] ,kind='linear',axis=0)(x), self._spline[0], der=1 )
            d = np.arctan2( dy, dx )
        else:
            if not isinstance(x,tuple):
                x = self.path2edge(x)[0]
            vertices = self._get_expanded_vertices()
            d = np.arctan2( vertices[x+1,1]-vertices[x,1], vertices[x+1,0]-vertices[x,0] )
        
        return d
    
    def samplepath(self,oversampling=None,openpath=False): #OK
        """Regular sampling of points on path.
        
        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along path (default is 1; with spline interpolation default is 20)
        openpath : bool, optional
            Whether or not to leave sampled path open. If False, the
            first and last sampled point will be identical
            (default is False).
        
        Returns
        -------
        array
            x,y coordinates of sampled points
        
        """
        
        if oversampling is None:
            oversampling = 20 if self.isspline else 1
        
        oversampling = np.floor(oversampling)
        vertices = self._get_expanded_vertices()
        
        if oversampling==1:
            return vertices.copy()
        
        if self.isspline:
            sampled_spline,spline  = _sample_spline( vertices, oversampling=oversampling, closed=self.issolid, openpath=openpath )
            vertices = sampled_spline[0]
        else:
            vertices = _sample_polyline( vertices, oversampling=oversampling, closed=self.issolid, openpath=openpath )
        
        return vertices
        
    def random_on(self,n): #OK?
        """Samples random points on path.
        
        Parameters
        ----------
        n : int
            number of randomly sampled points
        
        Returns
        -------
        array
            x,y coordinates of randomly sampled points
        
        """
        #draw points uniformly from [0,L), where L is length of polyline
        #map points back to 2D
        
        p = self.random_along(n)
        xy = self.path2point(p)
        
        return xy
    
    def aspolygon(self):
        """Convert polyline to closed polygon."""
        return polygon( vertices = self._vertices, spline=self.isspline, name=self.name )

class polygon(polyline,solid):
    """Polygon shape.
    
    Parameters
    ----------
    vertices : 
    spline : bool
        Apply spline interpolation.
    name : str, optional
        name of the shape (default is "")
    
    Attributes
    ----------
    numvertices
    area
    center
    vertices
    isspline
    boundingbox
    pathlength
    edgelengths
    
    """
    
    yaml_tag=u'!polygon_shape'
    
    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent polygon as YAML."""
        d = collections.OrderedDict()
        d['vertices'] = data.vertices.tolist()
        d['spline'] = data.isspline
        node = dumper.represent_mapping(cls.yaml_tag, d.iteritems())
        return node
    
    @classmethod
    def from_yaml(cls, loader, node):
        """Construct polygon from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(vertices=d['vertices'], spline=d['spline'])
    
    def __init__(self,vertices=[],spline=False,**kwargs):
        super(polyline,self).__init__(**kwargs)
        self.vertices = vertices
        self.isspline = spline
    
    @property
    def area(self):
        """Polygon area."""
        vertices = self._get_expanded_vertices(self.isspline)
        return np.abs(util.polygonarea(vertices))
    
    def contains(self,points):
        """Test if points are contained within polygon.
        
        Parameters
        ----------
        points : array_like
            x,y coordinates of points to be tested.
        
        Returns
        -------
        bool array
            True if point is contained within polygon.
        
        """
        #TODO: check points (either here or in inpoly)
        vertices = self._get_expanded_vertices(self.isspline)
        return util.inpoly( vertices, points )
    
    def random_in(self,n=1):
        """Samples random points inside polygon.
        
        Parameters
        ----------
        n : int
            number of randomly sampled points
        
        Returns
        -------
        array
            x,y coordinates of randomly sampled points
        
        """
        
        #use rejection sampling
        #first get the bounding box
        #compute fraction of bounding box area occupied by polygon (won't work for complex polygons (i.e. with self intersections), which should not be allowed anyays)
        #randomly sample points from bounding box
        #reject all points that are not withon polygon
        #continue sampling until required number of points have been found
        
        if not self.isclosed:
            raise TypeError()
        
        #a rejection sampling approach is used to generate random points
        #first let's get the bounding box (used to generate random points)
        bb = self.boundingbox()
        #based on the area of the polygon, relative to its bounding box,
        #we determine the oversampling fraction
        f = 1.02 * bb.area / self.area
        
        #pre-allocate the return array
        xy = np.zeros( (n,2), dtype=np.float64 )
        
        npoints = 0 #number of random points inside polygon that have been generated so far
        niteration = 0 #number of iterations
        MAXITER = 10 #maximum number of iterations - we should never reach this
        
        while niteration<MAXITER:
            #determine the number of random points to generate
            nsamples = np.ceil( (n-npoints) * f )
            #draw random points from bounding box
            samples = bb.randompoints(nsamples)
            #test if points are inside polygon
            b = self.contains( samples )
            nvalid = np.sum(b)
            
            if (npoints+nvalid)>n: #we have generated more valid random points than we needed
                xy[ npoints:n, : ] = samples[b][0:(n-npoints)]
                npoints = n
                break
            else: #we have generated fewer random points than we needed, so let's save them and continue
                xy[ npoints:(npoints+nvalid), : ] = samples[b]
            
            npoints = npoints + nvalid #update the number of generated random points
            niteration+=1
            
        assert npoints==n
        
        return xy
    
    def aspolyline(self):
        """Convert polygon to open polyline."""
        return polyline( vertices=self._vertices, spline=self.isspline, name=self.name)

class graph(path):
    """Graph of nodes connected by polylines.
    
    Parameters
    ----------
    polylines : sequence of polylines
    nodes : (n,2) array
    name : str, optional
        name of the shape (default is "")
    
    Attributes
    ----------
    name
    center
    ispath
    issolid
    pathlength
    edgelengths
    boundingbox
    numnodes
    numedges
    
    """
    
    yaml_tag=u'!graph_shape'
    
    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent graph as YAML."""
        d = collections.OrderedDict()
        d['nodes'] = data._nodes.tolist()
        d['polylines'] = data._polylines
        node = dumper.represent_mapping(cls.yaml_tag, d.iteritems())
        return node
    
    @classmethod
    def from_yaml(cls, loader, node):
        """Construct graph from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(polylines=d['polylines'], nodes=d['nodes'])
    
    def __init__(self, polylines=[], nodes=None, labels=None, **kwargs):
        super(graph,self).__init__(**kwargs)
        
        nodes, polylines, edges = _check_graph( nodes, polylines )
        
        #TODO: labels
        
        numnodes = nodes.shape[0]
        
        #make sure all polylines and nodes are connected
        if np.any( edges<0 ) or not np.all( np.in1d( edges, np.arange( numnodes ) ) ):
            raise TypeError()
        
        if np.any( edges[:,0]==edges[:,1] ):
            raise TypeError('polylines cannot start and end at the same vertex')
        
        if numnodes<2 or len(polylines)==0:
            raise TypeError('Need at least two vertices and 1 polyline')
        
        self._graph = np.zeros( (numnodes,numnodes), dtype=np.float64 ) + np.inf
        self._graph[ np.diag_indices(numnodes) ] = 0 
        
        for index,p in enumerate( polylines ):
            self._graph[edges[index,0],edges[index,1]] = np.minimum( self._graph[edges[index,0],edges[index,1]], p.pathlength )
            self._graph[edges[index,1],edges[index,0]] = self._graph[edges[index,0],edges[index,1]]
        
        self._edges = edges
        self._nodes = nodes
        self._polylines = polylines
        
        self._pathlengths, _ = util.floyd_warshall( self._graph );
            
    def __repr__(self):
        return 'graph with {n} nodes and {ne} edges'.format(n=self.numnodes,ne=self.numedges)
    
    @property
    def center(self): 
        """Center of graph."""      
        #center of mass of all nodes
        return np.mean( self._nodes, axis=0 )
    
    @property
    def pathlength(self):
        """Total length of all paths in graph."""
        return np.sum( self.edgelengths )
    
    @property
    def edgelengths(self):
        """Path lengths of graph edges."""
        return np.array([p.pathlength for p in self._polylines])
    
    @property
    def numnodes(self):
        """Number of nodes in graph."""
        return self._nodes.shape[0]
    
    @property
    def numedges(self):
        """Number of edges in graph."""
        return len(self._polylines)
    
    def scale(self,factor):
        """Scale shape.
        
        Parameters:
            factor : scalar or [factor_x, factor_y]
        
        """
        t = tf.Scale(factor=factor,origin=self.center)
        self._nodes = t.transform(self._nodes)
        for p in self._polylines:
            p.transform(t)
    
    def rotate(self,angle):
        """Rotate shape around its center.
        
        Parameters
        ----------
        angle : scalar
        
        """
        t = tf.Rotate(angle=angle,origin=self.center)
        self._nodes = t.transform(self._nodes)
        for p in self._polylines:
            p.transform(t)
    
    def translate(self,offset):
        """Translate shape.
        
        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]
        
        """
        t = tf.Translate(offset=offset)
        self._nodes = t.transform(self._nodes)
        for p in self._polylines:
            p.transform(t)
        
    def transform(self,tform):
        """Transform shape.
        
        Parameters
        ----------
        tform : Transform
        
        """
        self._nodes = tform.transform(self._nodes)
        for p in self._polylines:
            p.transform(t)
    
    def path2edge(self,x):
        """Converts path to edge representation.
        
        Parameters
        ----------
        x : array
            distance along path
        
        Returns
        -------
        tuple
            representation of point on path as edge index and 
            distance along edge.
            
        """
        return _path2edge(x,self.edgelengths,self.pathlength)
    
    def edge2path(self,x):
        """Converts edge to path representation.
        
        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            path
        
        Returns
        -------
        array
            distance along path
        
        """
        return _edge2path(x,self.edgelengths,self.pathlength)
        
    def point2path(self,points): #OK?
        """Project points to graph.
        
        Parameters
        ----------
        points : (n,2) array

        Returns
        -------
        dist_along_path : array
            distance along the path to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the path
        point_on_path: array
            coordinates of projected point on path
        edge: tuple
            representation of projected point as edge index and distance
            along edge.
        
        """
        points = np.asarray(points)
        
        edge = -np.ones( points.shape[0], dtype=np.int )
        dist_along_edge = np.zeros( points.shape[0], dtype=np.float64 )
        dist_to_path = np.zeros( points.shape[0], dtype=np.float64 ) + np.inf
        point_on_path = np.zeros( points.shape, dtype=np.float64 )
        
        #loop though all edges
        for k,p in enumerate( self._polylines ):
            #compute point2path for edge
            (ld, d, pp)=p.point2path( points )[0:3]
            #test if dist_to_path is smallest so far
            idx = np.abs(d) < np.abs(dist_to_path)
            edge[idx]=k
            dist_along_edge[idx]= np.minimum( ld[idx], self.edgelengths[k]-0.0001 ) #exclude endpoints of segments
            dist_to_path[idx] = d[idx]
            point_on_path[idx] = pp[idx]
            
        valid = ~np.isnan( dist_along_edge )
        
        L = np.cumsum( np.concatenate( ([0],self.edgelengths) ) )
        L[-1] = self.pathlength
        
        dist_along_path = np.empty( dist_along_edge.shape ) * np.nan
        dist_along_path[valid] = dist_along_edge[valid] + L[ edge[valid] ]
        
        return (dist_along_path,dist_to_path,point_on_path,(edge,dist_along_edge))
        
    def path2point(self,x,distance=0): #OK?
        """Converts points along path to 2d coordinates.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from path (default is 0)
        
        Returns
        -------
        xy : (n,2) array
            x,y coordinates
        
        """
        
        if not isinstance(x,tuple):
            edge,dist_along_edge = self.path2edge(x)
        else:
            edge,dist_along_edge = x
        
        xy = np.empty( (edge.size,2) )*np.nan
        
        for k,p in enumerate( self._polylines):
            #find points on this edge
            idx = edge==k
            if np.any(idx):
                xy[idx] = p.path2point(dist_along_edge[idx],distance) 
        
        return xy
    
    def tangent(self,x): #OK?
        """Computes tangential angle at points along path.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        
        Returns
        -------
        array
            tangential angle in radians
        
        """
        if not isinstance(x,tuple):
            edge,dist_along_edge = self.path2edge(x)
        else:
            edge,dist_along_edge = x
        
        d = np.empty( (edge.size) ) * np.nan
        
        for k,p in enumerate(self._polylines):
            idx = edge==k
            if np.any(idx):
                d[idx] = p.tangent( dist_along_edge[idx] )
        
        return d
    
    def gradient(self,x,dx=1): #OK?
        """Compute gradient at given locations along path.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        dx : scalar
        
        Returns
        -------
        ndarray
        
        """
        
        if isinstance(x,tuple):
            x = self.edge2path(x)
        
        g = np.empty( x.size ) * np.nan
        g[0]  = self.distance( x[0] , x[1]  )/dx
        g[-1] = self.distance( x[-2], x[-1] )/dx
        g[1:-1] = ( self.distance( x[1:-1],x[2:] ) - self.distance( x[1:-1], x[0:-2] ) ) / (2*dx)
        
        return g
    
    def distance(self,x,y): #OK? - element-wise distance only
        """Compute distance along path.
        
        Parameters
        ----------
        x,y : array or tuple
            distance along path or (edge index, distance along edge) tuple
        
        Returns
        -------
        ndarray
            element-wise distance between `x` and `y`
        
        """
        
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        if x.shape != y.shape:
            raise ValueError('x and y arrays need to have the same shape')
        
        if x.ndim>1:
            original_shape = x.shape
            x = x.ravel()
            y = y.ravel()
        else:
            original_shape = None
        
        if not isinstance(x,tuple):
            x_edge,x_dist_along_path = self.path2edge(x)
        else:
            x_edge,x_dist_along_path = x
        
        if not isinstance(y,tuple):
            y_edge,y_dist_along_path = self.path2edge(y)
        else:
            y_edge,y_dist_along_path = y
        
        L = self.edgelengths
        
        d = np.full( x_edge.size, np.inf )
        
        #this short-cut does not produce correct results, because it assumes
        #that the shortest path between two points on the same edge is along that edge
        idx = x_edge==y_edge
        d[idx] = y_dist_along_path[idx] - x_dist_along_path[idx]
        #idx = np.nonzero(~idx)
        
        #TODO: deal with NaNs
        
        tmp = np.vstack( (
        -(self._pathlengths[ self._edges[x_edge,0], self._edges[y_edge,0] ] + x_dist_along_path + y_dist_along_path),
        -(self._pathlengths[ self._edges[x_edge,0], self._edges[y_edge,1] ] + x_dist_along_path + L[y_edge] - y_dist_along_path),
          self._pathlengths[ self._edges[x_edge,1], self._edges[y_edge,0] ] + L[x_edge] - x_dist_along_path + y_dist_along_path,
          self._pathlengths[ self._edges[x_edge,1], self._edges[y_edge,1] ] + L[x_edge] - x_dist_along_path + L[y_edge] - y_dist_along_path,
          d,
        ) )
        
        mi = np.argmin( np.abs( tmp ), axis=0 )
        d = tmp[mi,np.arange(tmp.shape[1])]
        
        if not original_shape is None:
            d=d.reshape( original_shape )
        
        return d
    
    def random_on(self,n): #OK?
        """Samples random points on path.
        
        Parameters
        ----------
        n : int
            number of randomly sampled points
        
        Returns
        -------
        array
            x,y coordinates of randomly sampled points
        
        """
        
        #draw points uniformly from [0,L), where L is pathlength
        #map points back to 2D
        
        p = self.random_along(n)
        xy = self.path2point(p)
        
        return xy
    
    @property
    def boundingbox(self): #OK?
        """Axis-aligned bounding box of graph."""
        xymax = np.zeros(2)-np.inf
        xymin = np.zeros(2)+np.inf
        
        for p in self._polylines:
            bb = p.boundingbox.boxvertices()
            xymax = np.maximum( np.max( bb, axis=0 ), xymax )
            xymin = np.minimum( np.min( bb, axis=0 ), xymin )
        
        return rectangle( center=(xymax+xymin)/2.0, size=xymax-xymin, orientation=0 )
    
    def samplepath(self,oversampling=None): #TODO
        """Regular sampling of points on path.
        
        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along path.
            
        Returns
        -------
        array
            x,y coordinates of sampled points
        
        """
        return np.concatenate( [p.samplepath(oversampling=oversampling) for p in self._polylines], axis=0 )

class multishape(shape):
    """Collection of shapes.
    
    Parameters
    ----------
    *args : shape objects
    
    Attributes
    ----------
    name
    ispath
    issolid
    numshapes
    center
    boundingbox
    pathlength
    shapelengths
    
    """
    
    yaml_tag=u'!multi_shape'
    
    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent multishape as YAML."""
        d = dict( shapes=data._shapes )
        node = dumper.represent_mapping(cls.yaml_tag, d.iteritems())
        return node
    
    @classmethod
    def from_yaml(cls, loader, node):
        """Construct multishape from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(*d['shapes'])
    
    def __init__(self,*args):
        if any( [not isinstance(x,shape) for x in args] ):
            raise ValueError()
        
        self._shapes = list(args)
    
    def __repr__(self):
        return 'multishape with {n} shapes'.format(n=len(self._shapes))
    
    @property
    def numshapes(self):
        """Number of shapes in collection."""
        return len(self._shapes)
    
    @property
    def center(self):
        """Averaged center of shapes in collection."""
        if self.numshapes==0:
            return np.empty(2)*np.nan
        else:
            return np.mean( np.vstack( [x.center for x in self._shapes] ).T, axis=0 )
    
    def scale(self,factor):
        """Scaling transform is not implemented."""
        raise NotImplementedError()
    
    def rotate(self,angle):
        """Rotation transform is not implemented."""
        raise NotImplementedError()
    
    def translate(self,offset):
        """Translate transform is not implemented."""
        raise NotImplementedError()
    
    def transform(self,tform):
        """Transformations are not implemented."""
        raise NotImplementedError()
    
    @property
    def boundingbox(self):
        """Axis-aligned bounding box of shapes in collection."""
        xymax = np.zeros(2)-np.inf
        xymin = np.zeros(2)+np.inf
        
        for x in self._shapes:
            bb = x.boundingbox.boxvertices()
            xymax = np.maximum( np.max( bb, axis=0 ), xymax )
            xymin = np.minimum( np.min( bb, axis=0 ), xymin )
        
        return rectangle( center=(xymax+xymin)/2.0, size=xymax-xymin, orientation=0 )
    
    @property
    def ispath(self):
        """True if any shape in collection is a path."""
        return any( [x.ispath for x in self._shapes] )
        
    @property
    def issolid(self):
        """True if any shape in collection is a solid."""
        return any( [x.issolid for x in self._shapes] )
    
    def point2path(self,points,context=None):
        """Project points to shape collection.
        
        Parameters
        ----------
        points : (n,2) array
        context : shape
            Restrict projection to one of the shapes.

        Returns
        -------
        dist_along_path : array
            distance along the path to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the path
        point_on_path: array
            coordinates of projected point on path
        shape: tuple
            representation of projected point as shape index and distance
            along shape.
        
        """
        #TODO: if context is None, then for each point find nearest shape
        if context is None:
                raise NotImplementedError("Need a context.")
                
        #TODO: check correctness of code below. Is dist_along_path_expanded correct??
        
        #find index of shape context in self._shapes
        idx = self._shapes.index(context)
        dist_along_path,dist_to_path,point_on_path,dist_along_shape = self._shapes[idx].point2path(points)
        dist_along_path_expanded = ( np.zeros(dist_along_path.shape)+idx, dist_along_path )
        dist_along_path = dist_along_path + np.sum( self.shapelengths[0:idx] )
        return (dist_along_path, dist_to_path, point_on_path, dist_along_path_expanded)
    
    def path2point(self,x):
        """Converts points along path to 2d coordinates.
        
        Parameters
        ----------
        x : array or tuple
            distance along path or ( shape index, distance along shape )
            representation
        distance : float, optional
            distance to offset point from path (default is 0)
        
        Returns
        -------
        xy : (n,2) array
            x,y coordinates
        
        """
        #find index of shape context in self._shapes
        #if x is not a tuple, then convert to (shape, dist_along_path)
        if not isinstance(x,tuple):
            x = self.path2shape(x)
        
        if isinstance( x[1], tuple ):
            if np.unique( x[0].ravel() ).shape[0] != 1:
                raise ValueError()
            xy = self._shapes[x[0][0]].path2point( x[1] )
        else:
            xy = np.empty( (x[1].size, 2) ) * np.nan
            for k,p in enumerate(self._shapes):
                idx = x[0]==k
                xy[idx] = self._shapes[k].path2point( x[1][idx] )
        
        return xy
    
    @property
    def pathlength(self):
        """Total path length of all shapes combined."""
        return np.sum( self.shapelengths )
    
    @property
    def shapelengths(self):
        """Path length of each shape in collection."""
        return np.array( [x.pathlength for x in self._shapes] )
    
    def path2shape(self,x):
        """Converts path to shape representation.
        
        Parameters
        ----------
        x : array
            distance along path
        
        Returns
        -------
        tuple
            representation of point on path as shape index and 
            distance along shape.
            
        """
        return _path2edge(x,self.shapelengths,self.pathlength)
    
    def shape2path(self,x):
        """Converts shape to path representation.
        
        Parameters
        ----------
        x : tuple
            (shapeindex, distance along shape) representation of points on
            path
        
        Returns
        -------
        array
            distance along path
        
        """
        return _edge2path(x,self.shapelengths,self.pathlength)
    
    def samplepath(self,oversampling=None):
        """Regular sampling of points on path.
        
        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along path.
            
        Returns
        -------
        array
            x,y coordinates of sampled points
        
        """
        #xy = np.zeros( (0,2) )
        xy = []
        for p in self._shapes:
            #xy = np.concatenate( (xy,p.samplepath( oversampling=oversampling )) )
            xy.append( p.samplepath( oversampling=oversampling ) )
        return xy
            
#convience functions
def ngon(size=1,n=3):
    """General n-sided polygon.
    
    Builds an n-sided polygon centered on (0,0) with the spoke lengths
    set by `size`.
    
    Parameters
    ----------
    size : scalar
    n : int
    
    Returns
    -------
    polygon
    
    """
    angle = np.arange(n)*2*np.pi/n
    vertices = size*np.vstack( (np.cos(angle),np.sin(angle)) ).T
    return polygon(vertices=vertices)

def triangle(*args,**kwargs):
    """Equal-sided triangle.
    
    Parameters
    ----------
    size : scalar
    
    Returns
    -------
    polygon
    
    """
    kwargs['n']=3
    return ngon(*args,**kwargs)

def pentagon(*args,**kwargs):
    """Equal-sided pentagon.
    
    Parameters
    ----------
    size : scalar
    
    Returns
    -------
    polygon
    
    """
    kwargs['n']=5
    return ngon(*args,**kwargs)

def hexagon(*args,**kwargs):
    """Equal-sided hexagon.
    
    Parameters
    ----------
    size : scalar
    
    Returns
    -------
    polygon
    
    """
    kwargs['n']=6
    return ngon(*args,**kwargs)

# helper functions
def _construct_spline(vertices,closed=False):
    s = sp.interpolate.splprep( (vertices[:,0], vertices[:,1]), s=0.0, per=closed )
    #sder = sp.interpolate.splder( s[0] )
    return s[0:2]

def _sample_spline(vertices,oversampling=20,closed=False,openpath=False):
    
    oversampling = np.floor(oversampling)
    if oversampling==1:
        return vertices.copy()
    
    nvertices = vertices.shape[0]
    npoints = nvertices + (oversampling-1)*(nvertices-1) - int(openpath)
    
    spline_param, spline_u = _construct_spline(vertices,closed)
    
    fcn = sp.interpolate.interp1d( np.arange(len(spline_u)), spline_u, kind='linear', axis=0, bounds_error = False )
    x = np.linspace(0,len(spline_u)-1,num=npoints,endpoint=not (closed and openpath))
    sampled_u = fcn(x)

    sampled_spline = np.empty( (sampled_u.size, 2) )
    sampled_spline[:,0],sampled_spline[:,1] = sp.interpolate.splev( sampled_u, spline_param )
    
    #xyder = np.empty( (u.size, 2) )
    #xyder[:,0],xyder[:,1] = sp.interpolate.splev( u, spline_param, der=1 )
    
    return ((sampled_spline, sampled_u), (spline_param, spline_u))

def _sample_polyline(vertices,oversampling=1,closed=False,openpath=False):
    oversampling = np.floor(oversampling)
    if oversampling==1:
        return vertices.copy()
    
    nvertices = vertices.shape[0]
    
    npoints = nvertices + (oversampling-1)*(nvertices-1) - int(openpath)
    
    fcn = sp.interpolate.interp1d( np.arange(nvertices), vertices, kind='linear', axis=0, bounds_error = False )
    x = np.linspace(0,nvertices-1,num=npoints,endpoint=not (closed and openpath))
    xy = fcn(x)
    
    return xy

def _inv_ellipeinc(a,b,n=100):
    
    if a>b:
        t_offset = 1.5*np.pi
    else:
        t_offset = np.pi
        a,b=b,a
    
    eccentricity = 1-(b/a)**2
    t = np.linspace(0,1,num=n)*2*np.pi
        
    offset = a*( sp.special.ellipeinc( 2*np.pi, eccentricity ) - sp.special.ellipeinc( t_offset, eccentricity ) )
    s = np.mod(a*sp.special.ellipeinc( circ.wrap( t+t_offset ), eccentricity ) + offset, a*sp.special.ellipeinc( 2*np.pi, eccentricity ) )
    s[-1] = a*sp.special.ellipeinc( 2*np.pi, eccentricity )
        
    fcn = sp.interpolate.interp1d( s, t, bounds_error=False)
    return fcn

def _check_graph(nodes,polylines,tol=0.001,correct=False):
    
    nodes = util.aspoints(nodes)
    
    if isinstance(polylines,polyline):
        polylines = [polylines]
    elif not all( [isinstance(x,polyline) for x in polylines] ):
        raise TypeError()
    
    #check for duplicate nodes
    if nodes.shape[0]>1:
        d = scipy.spatial.distance.pdist( nodes )
        d = scipy.spatial.distance.squareform( d )
        if np.min( d[np.tril_indices_from(d,k=-1)]  ) < tol:
            raise ValueError
    
    npolylines = len(polylines)
    
    edges = np.zeros( (npolylines,2), dtype=np.int ) -1
    
    for index,p in enumerate( polylines ):
        #find closest node for polyline start vertex
        d = scipy.spatial.distance.cdist( nodes, p.vertices[0:1] )
        dmin = np.argmin( d )
        if d[dmin]<=tol:
            edges[index,0] = dmin
            if correct and d[dmin]>0:
                p.vertices[0] = nodes[dmin]
        
        #find closest node for polyline end vertex
        d = scipy.spatial.distance.cdist( nodes, p.vertices[-1:] )
        dmin = np.argmin( d )
        if d[dmin]<=tol:
            edges[index,1] = dmin
            if correct and d[dmin]>0:
                p.vertices[-1] = nodes[dmin]
    
    return (nodes, polylines, edges)

def _path2edge(x,edgelengths,pathlength):
    n = len(edgelengths)+1
    L = np.cumsum( np.concatenate( ([0],edgelengths) ) )
    L[-1] = pathlength
    edge_index = np.concatenate( (np.arange(n-1),[n-2]) )
    edge = np.floor( sp.interpolate.interp1d( L, edge_index, kind='linear', axis=0)(x) ).astype(np.int)
    return (edge, x-L[edge])

def _edge2path(x,edgelengths,pathlength):
    if not isinstance(x,tuple):
        raise ValueError()
    L = np.cumsum( np.concatenate( ([0],edgelengths) ) )
    L[-1] = pathlength
    return L[x[0]] + x[1]
