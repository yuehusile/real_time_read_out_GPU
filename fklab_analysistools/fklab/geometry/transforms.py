"""
=====================================================
2D Transformations (:mod:`fklab.geometry.transforms`)
=====================================================

.. currentmodule:: fklab.geometry.transforms

Classes to facilitate working with 2D transformations.

.. autosummary::
    :toctree: generated/
    
    Transform
    Scale
    Translate
    Rotate
    TransformStack
    
"""

import numpy as np

__all__ = ['Transform','Scale','Translate','Rotate','TransformStack']

def _scale_matrix( sxy ):
    return np.matrix( [[sxy[0],0,0],[0,sxy[-1],0],[0,0,1]] )

def _rotation_matrix( phi ):
    phi = np.asscalar(phi)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    return np.matrix( [[cosphi,-sinphi,0],[sinphi,cosphi,0],[0,0,1]] )

def _translation_matrix( txy ):
    return np.matrix( [[1,0,txy[0]],[0,1,txy[-1]],[0,0,1]] )

def _identity_matrix():
    return np.matrix( np.diag( np.ones(3) ) )

class Transform(object):
    """Base class for 2D transformations objects.
    
    Attributes
    ----------
    matrix : transformation matrix
    invmatrix : inverse transformation matrix
    
    Methods
    -------
    transform(data)
    invtransform(data)
    
    """
    
    _indent = 2
    _indent_str = '-'
    
    def __init__(self):
        pass
    
    def copy(self):
        return Transform()

    def __repr__(self):
        return self._get_str()
    
    def _get_indent(self,indent=0):
        return self._indent_str*(self._indent*indent)
    
    def _get_str(self,indent=0):
        s = self._get_indent(indent) + 'Identity'
        return s
    
    def transform(self,data):
        """Apply transformation to data.
        
        Parameters:
        data : (N,2) array
        
        Returns:
        (N,2) array
        
        """
        
        M = self._get_matrix()
        return np.array(data * M[0:2,0:2].T + M[0:2,2].T)
    
    def invtransform(self,data):
        """Apply inverse transformation to data.
        
        Parameters:
        data : (N,2) array
        
        Returns:
        (N,2) array
        
        """
        
        M = self._get_invmatrix()
        return np.array(data * M[0:2,0:2].T + M[0:2,2].T)
    
    def _get_matrix(self):
        return _identity_matrix()
    
    def _get_invmatrix(self):
        return _identity_matrix()
    
    @property
    def matrix(self):
        return self._get_matrix()
    
    @property
    def invmatrix(self):
        return self._get_invmatrix()

    def __iadd__(self,other):
        raise NotImplementedError

    def __add__(self,other):
        if isinstance(other,Transform):
            t = TransformStack() + self + other
        else:
            raise TypeError
        
        return t

class Scale(Transform):
    """Scaling transformation.
    
    Parameters
    ----------
    factor : scalar or [scale_x, scale_y]
    origin : [x, y]
    
    """
    
    _factor = None
    _origin = None
    
    def __init__(self,factor=[1,1],origin=[0,0]):
        self.factor = factor
        self.origin = origin
    
    def copy(self):
        return Scale( factor=self._factor, origin=self._origin)
    
    def __repr__(self):
        return self._get_str()
    
    def _get_str(self,indent=0):
        s = self._get_indent(indent) + "Scale by %(xfactor)f, %(yfactor)f with origin (%(xorigin)f,%(yorigin)f)" % {'xfactor':self._factor[0],'yfactor':self._factor[-1],'xorigin':self._origin[0],'yorigin':self._origin[1]}
        return s
        
    def _get_matrix(self):
        return _translation_matrix( self._origin ) * _scale_matrix( self._factor ) * _translation_matrix( -self._origin )
    
    def _get_invmatrix(self):
        return _translation_matrix( self._origin ) * _scale_matrix( 1.0/self._factor ) * _translation_matrix( -self._origin )
        
    @property
    def factor(self):
        return self._factor
    
    @factor.setter
    def factor(self,value):
        value = np.array(value).ravel()
        if len(value)<1 or len(value)>2:
            raise ValueError()
        
        self._factor = value
    
    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self,value):
        value = np.array(value).ravel()
        if len(value)!=2:
            raise ValueError()
            
        self._origin = value
    
class Rotate(Transform):
    """Rotation transformation.
    
    Parameters
    ----------
    angle : scalar
    origin : [x, y]
    
    """
    
    _angle = None
    _origin = None
    
    def __init__(self,angle=0,origin=[0,0]):
        self.angle = angle
        self.origin = origin
    
    def copy(self):
        return Rotate(angle=self._angle, origin=self._origin)
    
    def __repr__(self):
        return self._get_str()
    
    def _get_str(self,indent=0):
        s = self._get_indent(indent) + "Rotate by %(angle)f radians around origin (%(xorigin)f,%(yorigin)f)" % {'angle':self._angle,'xorigin':self._origin[0],'yorigin':self._origin[1]}
        return s
    
    def _get_matrix(self):
        return _translation_matrix( self._origin ) * _rotation_matrix( self._angle ) * _translation_matrix( -self._origin )
    
    def _get_invmatrix(self):
        return _translation_matrix( self._origin ) * _rotation_matrix( -self._angle ) * _translation_matrix( -self._origin )
        
    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self,value):
        value = np.array(value).ravel()
        if len(value)!=1:
            raise ValueError()
            
        self._angle = np.mod(value,np.pi*2)
    
    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self,value):
        value = np.array(value).ravel()
        if len(value)!=2:
            raise ValueError()
            
        self._origin = value

class Translate(Transform):
    """Translate transformation.
    
    Parameters
    ----------
    offset : scalar or [offset_x, offset_y]
    
    """
    
    _offset = None
    
    def __init__(self,offset=[0,0]):
        self.offset = offset
    
    def copy(self):
        return Translate(offset=self._offset)
    
    def __repr__(self):
        return self._get_str()
    
    def _get_str(self,indent=0):
        s = self._get_indent(indent) + "Translate %(xoffset)f, %(yoffset)f" % {'xoffset':self._offset[0],'yoffset':self._offset[-1]}
        return s
    
    def _get_matrix(self):
        return _translation_matrix( self._offset )
    
    def _get_invmatrix(self):
        return _translation_matrix( -self._offset )
        
    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self,value):
        value = np.array(value).ravel()
        if len(value)<1 or len(value)>2:
            raise ValueError()
            
        self._offset = value

class TransformStack(Transform):
    """Combination of multiple transformations.
    
    Methods
    -------
    scale( factor, origin )
    rotate( angle, origin )
    translate( offset )
    pop()
    copy()
    append( Transform )
    extend( TransformStack )
    
    """
    
    def __init__(self):
        self._transforms = []
    
    def copy(self):
        t = TransformStack()
        t._transforms = self._transforms
        return t
    
    def __repr__(self):
        return self._get_str()
    
    def _get_str(self,indent=0):
        
        s = []
        for k in self._transforms:
            if isinstance(k,TransformStack):
                s.append( k._get_str(indent=indent+1) )
            else:
                s.append( k._get_str(indent=indent) )
        
        return '\n'.join(s)
    
    def scale(self,factor=np.array([1,1]),origin=np.array([0,0])):
        """Add scaling to transformation stack.
        
        Parameters
        ----------
        factor : scalar or [scale_x, scale_y]
        origin : [x, y]
    
        """
        self._transforms.append( Scale( factor, origin ) )
    
    def rotate(self,angle=np.array([0]),origin=np.array([0,0])):
        """Add rotation to transformation stack.
        
        Parameters
        ----------
        angle : scalar
        origin : [x, y]
        
        """
        self._transforms.append( Rotate( angle, origin ) )
    
    def translate(self,offset=np.array([0,0])):
        """Add translation to transformation stack.
        
        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]
        
        """
        
        self._transforms.append( Translate( offset ) )
    
    def pop(self):
        """Pop and return transformation from stack."""
        return self._transforms.pop()
    
    def __getitem__(self,key):
        return self._transforms.__getitem__(key)
    
    def __setitem__(self,key,value):
        if isinstance(value,TransformBase):
            pass
        elif isinstance(value,(list,tuple)) and all( [isinstance(x,TransformBase) for x in value] ):
            pass
        self._transforms.__setitem__(key,value)
    
    def append(self,value):
        """Append transformation object to stack."""
        if not isinstance(value,Transform):
            raise TypeError
        self._transforms.append(value)
    
    def extend(self,value):
        """Extend stack with transformation on other stack."""
        if not isinstance(value,TransformStack):
            raise TypeError
        self._transforms.extend(value._transforms)
    
    def _get_matrix(self):
        m = _identity_matrix()
        for k in self._transforms:
            m = k._get_matrix() * m
        return m
    
    def _get_invmatrix(self):
        m = _identity_matrix()
        for k in reversed(self._transforms):
            m = k._get_invmatrix() * m
        return m
    
    def __iadd__(self,other):
        if isinstance(other,TransformStack):
            self.extend( other )
        else:
            self.append( other )
        return self
        
    def __add__(self,other):
        t = self.copy()
        t.__iadd__(other)
            
        return t

# -- functions below are not used ---

def _matrix_to_RST(m):
    #m should be 3x3 matrix
    m = np.array(m)
    t = Translate( offset = m[0:2,2] )
    a = np.arctan( m[1,0]/m[0,0] )
    print( a, np.arctan( -m[0,1]/m[1,1] ) )
    r = Rotate( angle = a )
    #sx = np.sign(m[0,0])*np.sqrt( np.sum( m[0,0:2]**2 ) )
    #sy = np.sign(m[1,1])*np.sqrt( np.sum( m[1,0:2]**2 ) )
    sx = m[0,0]/np.cos(a)
    sy = m[1,1]/np.cos(a)
    s = Scale( factor = np.array( [sx,sy] ) )
    return s+r+t
