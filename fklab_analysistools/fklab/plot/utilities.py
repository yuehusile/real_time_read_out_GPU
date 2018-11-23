"""
=================================================================
Utilities (:mod:`fklab.plot.utilities`)
=================================================================

.. currentmodule:: fklab.plot.utilities

Plotting utilities

.. autosummary::
    :toctree: generated/
    
    LinearOffsetCollection
    RangeVector
    ColumnView

"""

import numpy as np
import matplotlib as mpl
import matplotlib.transforms

__all__ = ['LinearOffsetCollection', 'RangeVector', 'ColumnView']

class LinearOffsetCollection:
    """
    Helper class to layout artists in axes.
    
    Parameters
    ----------
    children : list
        List of artists.
    spacing : float
        Offsets between children (can be negative).
    origin : float
        Base offset.
    direction : {'vertical', 'horizontal'}
        Apply offsets in horizontal or vertical direction.
    
    """
    def __init__(self, children=[], spacing=0., origin=0., direction='vertical'):
        self._children = children
        self._spacing = float(spacing)
        self._origin = float(origin)
        
        self._vertical = True if direction in ('vertical','v','vert') else False
        
        self._apply_offset()
        
    def set_origin(self, origin):
        self._origin = float(origin)
        self._apply_offset()
    
    def set_spacing(self, spacing):
        self._spacing = float(spacing)
        self._apply_offset()
        
    def set_vertical(self,val=True):
        self._vertical = bool(val)
        self._apply_offset()
        
    def set_horizontal(self,val=True):
        self._vertical = not bool(val)
        self._apply_offset()
        
    def set_direction(self,val):
        self._vertical = True if val in ('vertical','v','vert') else False
        self._apply_offset()
    
    def __len__(self):
        return self._children.__len__()
    
    def __getitem__(self,key):
        return self._children.__getitem__(key)
    
    def __setitem__(self,key,value):
        self._children.__setitem__(key,value)
        self._apply_offset()
    
    def __delitem__(self,key):
        self._children.__delitem__(key)
        self._apply_offset()
    
    def append(self,value):
        self._children.append(value)
        self._apply_offset()
    
    def extend(self,values):
        self._children.extend(values)
        self._apply_offset()
    
    def insert(self,index,value):
        self._children.insert(index,value)
        self._apply_offset()
    
    def sort(self, cmp=None, key=None, reverse=False):
        self._children.sort( cmp=cmp, key=key, reverse=reverse)
        self._apply_offset()
    
    def update(self):
        self._apply_offset()
    
    def _apply_offset(self):
        
        if self._vertical:
            translation = [0, self._origin]
            index = 1
        else:
            translation = [self._origin, 0]
            index = 0
        
        for child in self._children:
            t = mpl.transforms.Affine2D()
            t.translate( *translation )
            child.set_transform( t + child.axes.transData )
            translation[index] += self._spacing
    
class RangeVector:
    """
    A lazily computed range of values.
    
    Parameters
    ----------
    n : int
        Number of values.
    start : float, optional
        Start value.
    delta : float, optional
        Spacing between values.
    
    """
    def __init__(self,n, start=0, delta=1):
        self._n = int(n)
        self._start = float(start)
        self._delta = float(delta)
    
    def __getitem__(self,key):
        if isinstance(key,slice):
            return np.arange( *key.indices(self._n) )*self._delta + self._start
        elif isinstance(key,int):
            if key<0: key+=self._n
            if key<0 or key>=self._n:
                raise KeyError("Key out of bounds")
            return key * self._delta + self._start
        else:
            raise NotImplementedError
        
    def __setitem__(self,key,value):
        raise NotImplementedError
    
    def __len__(self):
        return (self._n)
    
    @property
    def shape(self):
        return [self._n,]
    
    @property
    def ndim(self):
        return 1

class ColumnView(object):
    """A column view on a numpy 2d array
    
    A ColumnView object represents a read-only column in a 2d numpy array.
    This class is needed for HDF5 data arrays that do not provide views,
    but rather load the data when indexed. Optionally, a function can be
    applied to the indexed data.
    
    Parameters
    ----------
    source : 2d array
    col : int, optional
        Column index.
    fcn : callable, optional
        Function that is called when data is requested. The function should
        work element-wise and should not change the shape of the array.
    
    """
    
    def __init__(self, source, col=0, fcn=None):
        
        if not source.ndim==2:
            raise TypeError('Expecting 2-d array')
        
        self._source = source
        self._col = int(col)
        self._fcn = fcn
        
    def __len__(self):
        return len(self._source)
    
    @property
    def shape(self):
        return (self._source.shape[0],)
    
    @property
    def ndim(self):
        return 1
    
    def __getitem__(self, key):
        
        if self._fcn is None:
            return self._source[ key, self._col ]
        else:
            return self._fcn( self._source[ key, self._col ] )
