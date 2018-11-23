"""
================================================
Binary file reading (:mod:`fklab.io.binary_new`)
================================================

.. currentmodule:: fklab.io.binary_new

General class for reading binary files.

.. autosummary::
    :toctree: generated/
    
    BinaryFileReader
    accessor
    field_accessor
    
"""

__all__ = ['BinaryFileReader', 'accessor', 'field_accessor']

import os
import sys
from collections import namedtuple, Mapping

import numpy as np
import scipy

import time

class accessor(object):
    def __init__(self, source=None, dtype=None, fcn=None, concatenate=False, cache=False):
        
        self._source = None
        self._dtype_request = self._dtype = dtype
        self._fcn = fcn
        self._concatenate = concatenate
        self._cache=bool(cache)
        self._cached_key = None
        self._cached_data = None
        
        self.source = source #must be 1d array
    
    @property
    def ndim(self):
        if self._dtype is None:
            raise TypeError("No dtype.")
        
        n = len(self._dtype.shape)
        
        if not self._concatenate:
            n += 1
        
        return n
    
    @property
    def shape(self):
        if self._dtype is None or self._source is None:
            raise TypeError("No source/dtype.")
        
        l = len(self._source)
        s = self._dtype.shape
        
        if len(s)==0:
            s = (l,)
        elif self._concatenate:
            s = (s[0]*l,) + s[1:]
        else:
            s = (l,) + s
        
        return s
    
    def __len__(self):
        return self.shape[0]
    
    @property
    def source(self):
        return self._source
    
    @source.setter
    def source(self, value):
        
        if self._dtype_request is None:
            if value is None:
                self._dtype = None
            else:
                self._dtype = value.dtype
        
        self._source = value
    
    @property
    def dtype(self):
        return self._dtype.base
    
    def compute_indices(self, keys):
        
        if not self._concatenate or len(self._dtype.shape)==0 or self._dtype.shape[0]==1:
            record_key = keys[0]
            sub_key = (slice(None),) + keys[1:]
        else:
            # split first key into record index + remainder index
            if isinstance(keys[0],slice):
                start, stop, step = keys[0].indices( len(self._source) )
                k = np.arange( start, stop, step, dtype=np.int64 )
            else:
                k = np.array( keys[0], dtype=np.int64 )
                
            record_key = k//self._dtype.shape[0]
            sub_key = k - self._dtype.shape[0]*record_key
            record_key, ur = np.unique( record_key, return_inverse=True )
            sub_key += self._dtype.shape[0]*ur
            
            sub_key = (sub_key,) + keys[1:]
        
        return record_key, sub_key
    
    def _compute_key_hash(self, key):
        
        if isinstance(key,slice):
            h = hash( str(key) )
        elif isinstance(key,int):
            h = hash( key )
        else:
            # numpy array
            flag = key.flags.writeable
            key.flags.writeable = False
            h = hash( key.data )
            key.flags.writeable = flag
        
        return h
    
    def __getitem__(self, keys):
        
        # keys can be a slice, integer, integer 1d array or tuple of these
        if not isinstance(keys,tuple):
            keys = (keys,)
        
        if len(keys)==0:
            raise KeyError
        
        record_key, sub_key = self.compute_indices( keys )
        
        if self._cache:
            h = self._compute_key_hash(record_key)
        
        if self._cache and self._cached_key == h:
            out = self._cached_data
        else:
            if isinstance(record_key,slice):
                nn = str(record_key)
            else:
                nn = len(record_key)
                
            print("Loading new data: {0} records".format(nn))
            t = time.time()
            
            data = np.array(np.atleast_1d( self._source[record_key] ))
            
            dt = 1000*(time.time()-t)
            print("Loaded new data: {0} ms".format( dt ))
            print("Data type: {0}".format(type(data)))
            
            if self._fcn is not None:
                out = np.empty( len(data), dtype=self._dtype )
                self._fcn( data, out )
            else:
                out = data
            
            dt = 1000*(time.time()-t)-dt
            print("Processed new data: {0} ms".format( dt ))
            
            if self._concatenate and len(self._dtype.shape)>0:
                out = out.reshape( (out.shape[0] * out.shape[1],) + out.shape[2:] )
            
            if self._cache:
                # store record_key and out
                self._cached_key = h
                self._cached_data = out
            
        out = out.__getitem__( sub_key )
        
        return out

class field_accessor(accessor):
    def __init__(self, field, source=None):
        
        super(field_accessor,self).__init__(dtype=None, fcn=self._get_field )
        
        self._field = str(field)
        self.source = source
    
    @property
    def field(self):
        return self._field
    
    @accessor.source.setter
    def source(self,value):
        print(type(value))
        if value is None:
            self._dtype = None
            self._source = None
        else:
            
            if len(value.dtype.fields)==0:
                raise TypeError('Source dtype has no fields.')
            
            if self._field not in value.dtype.fields:
                raise TypeError('Source has no field "{0}".'.format(self._field))
            
            self._dtype = value.dtype.fields[self._field][0]
            self._source = value
    
    def _get_field(self, data, out):
        
        out[:] = data[self._field]


class BinaryFileReader(object):
    """Class for reading generic binary files.
    
    Parameters
    ----------
    filename : str
    dtype : numpy dtype
        Numpy type description of the binary data.
    offset : int
        Byte-offset of start binary data in file.
    accessors : dict
        Dictionary of accessor objects that provide views on the data by
        selecting and transforming data fields.
    
    """
    def __init__(self,filename,dtype,offset=0,accessors=None):
        
        #save file information
        filesize = os.path.getsize( filename )
        recordsize = dtype.itemsize
        nrecords = (filesize - offset)//recordsize
        self._fileinfo = {'filename':filename,'dtype':dtype,'offset':offset,'recordsize':recordsize,'nrecords':nrecords}
        
        if not sys.maxsize > 2**32:
            raise NotImplementedError("New style Neuralynx IO is only supported for 64 bit systems. Use the legacy module if you are running 32 bit system.")
        
        # because memmap in read mode does not allow the offset to be beyond the file size
        # we need to separately handle the case when nrecords==0
        if nrecords==0:
            self._memorymap = np.memmap(filename, mode = 'r', dtype=dtype, offset=0, shape=(nrecords,) )
        else:
            self._memorymap = np.memmap(filename, mode = 'r', dtype=dtype, offset=offset, shape=(nrecords,) )
        
        if accessors is None:
            accessors = {}
        elif not isinstance(accessors, Mapping):
            raise TypeError('Expecting accessor map.')
        
        for fld in self._fileinfo['dtype'].fields:
            if not fld in accessors:
                accessors[fld] = field_accessor( fld )
        
        #set up generic access to file records
        self.raw = self._memorymap
        
        self.add_accessors( accessors )
    
    def add_accessors(self, accessors):

        for k,v in accessors.iteritems():
            v.source = self._memorymap
            self.__setattr__(k, v)
    
    def __getitem__(self,key):
        try:
            return self.default.__getitem__(key)
        except AttributeError:
            return self.raw.__getitem__(key)
        
    @property
    def shape(self):
        try:
            return self.default.shape
        except AttributeError:
            return self.raw.shape
        
    def __len__(self):
        try:
            return len(self.default)
        except AttributeError:
            return len(self.raw)
