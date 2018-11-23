# -*- coding: utf-8 -*-

#array_axis class
#a vector that transforms values into indices (and v.v.)
#axis values have units (from quantities)
#indices and values need to be monotonically increasing, but not necessary linearly??
#values can be categorical?, ordinal? etc
#

#array class
#each dimension has an associated array_axis object
#array values have units
#slicing -> slices array_axis
#concatenation -> check units, how to extend/combine array_axis objects?
#operations on values in the array will update units
 


import numpy as np

class array_axis(object):
    def __init__(self, values, indices=None):
        self.values = values
        if indices is None:
            self.indices = np.arange(0,len(self.values))
    
    def value2index(self,values):
        return np.interp(values,self.values,self.indices)
    
    def index2value(self,indices):
        return np.interp(indices,self.indices,self.values)
    
    @property
    def value_support(self):
        return (self.values.min(),self.values.max())
    
    @property
    def index_support(self):
        return (self.indices.min(),self.indices.max())

class RealisticInfoArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)
    
    def __getitem__(self,items):
        print(items)
        #print(type(items))
        return super(RealisticInfoArray,self).__getitem__(items)