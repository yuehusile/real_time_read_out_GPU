"""
============================================
Binary file reading (:mod:`fklab.io.binary`)
============================================

.. currentmodule:: fklab.io.binary

General class for reading binary files.

.. autosummary::
    :toctree: generated/
    
    BinaryFileReader
    
"""

__all__ = ['BinaryFileReader']

import os
import sys
from collections import namedtuple

import numpy as np
import scipy

class default_sample_indexer:
    def __init__(self, n, nrecords):
        self._n = int(n)
        self._nrecords = int(nrecords)
    def __call__(self, indices):
        record_indices = np.int64( np.floor( indices / self._n ) )
        sub_indices = indices - record_indices*self._n
        return record_indices, sub_indices
    def nsamples(self):
        return self._nrecords * self._n
    def nsamples_per_record(self):
        return self._n
    
class signal_indexer:
    def __init__(self, n, nrecords, invalid_records=None):
        self._n = n
        self._nrecords = nrecords
        
        if invalid_records is None or len(invalid_records)==0:
            self._invalid=False
        else:
            self._invalid=True
        
            r = [0,]
            v = [0,]
            last = 0
            for k in invalid_records:
                r.extend([k[0],k[0]+1])
                v.extend([(k[0]-last)*n,k[1]])
                last = k[0]+1
            
            r.append(nrecords)
            v.append((nrecords - invalid_records[-1,0] - 1) * n)
            
            self._r = np.array(r)
            self._v = np.cumsum( np.array(v) )
    def nsamples(self):
        if self._invalid:
            return self._v[-1]
        else:
            return self._nrecords * self._n
    def __call__(self, indices):
        
        if self._invalid:
            record_indices = np.floor( scipy.interp( indices, self._v, self._r ) )
            sub_indices = indices - scipy.interp( record_indices, self._r, self._v )
        else:
            record_indices = np.int64( np.floor( indices / self._n ) )
            sub_indices = indices - record_indices*self._n
        
        return np.int64(record_indices), np.int64(sub_indices)
        
    def nsamples_per_record(self):
        return self._n
    
class accessor:
    def __init__(self, dtype, fields, label=None, record_shape=None, sample_dim=None, record_processor=None, sample_processor=None):
        if not isinstance(dtype, np.dtype) or dtype.fields is None:
            raise ValueError('Expected numpy.dtype with fields')
        
        if isinstance(fields, str):
            fields = [fields,]
        
        fields = list( fields )
        
        if not fields or not all( [ x in dtype.fields for x in fields ] ):
            raise ValueError('Invalid source fields')
        
        self._dtypes = [ dtype.fields[x][0] for x in fields ]
        self._fields = fields
        
        if record_processor is not None and not callable(record_processor):
            raise ValueError('Invalid record level processing function')
        self._record_processor = record_processor
        
        if self._record_processor is None and len(self._fields)>1:
            raise ValueError('Only one source field allowed if no record level processing function is specified')
        
        if sample_processor is not None and not callable(sample_processor):
            raise ValueError('Invalid sample level processing function')
        self._sample_processor = sample_processor
        
        if label is None:
            label= self._fields[0]
        
        if not isinstance(label, str):
            raise ValueError('Invalid label')
        
        self._label = label
        
        if record_shape is None:
            if not self._label in self._fields:
                raise ValueError('Please specify a record shape')
            self._record_shape = dtype.fields[ self._label ][0].shape
        else:
            self._record_shape = record_shape
            if not isinstance(self._record_shape,tuple) and not all( [isinstance(x,int) for x in self._record_shape] ):
                raise ValueError('Invalid record shape')
                
        if sample_dim is not None and ( not isinstance(sample_dim,int) or sample_dim<0 or sample_dim>=len(self._record_shape) ):
            raise ValueError('Invalid sample dimension')
        
        self._sample_dim = sample_dim
    
    def dtype(self):
        return self._dtypes
    
    def record_shape(self):
        return self._record_shape
        
    def fields(self):
        return self._fields
    
    def has_samples(self):
        return self._sample_dim is not None
    
    def nsamples(self):
        if self._sample_dim is None:
            return None
        
        return self._record_shape[ self._sample_dim ]
    
    def sample_dim(self):
        return self._sample_dim
    
    def process_records( self, **data ):
        if self._record_processor is None:
            return data.values()[0]
        else:
            return self._record_processor( **data )
    
    def process_samples( self, data ):
        if self._sample_processor is None:
            return data
        else:
            return self._sample_processor( data )
    
    def label(self):
        return self._label
    

class BinaryFileReader(object):
    """Clas for reading generic binary files.
    
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
    def __init__(self,filename,dtype,offset=0,accessors={}):
        
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
        
        #save accessors dictionary
        self._accessors = accessors
        
        #add record based access to any existing field in the file that is not listed in accessors
        for fld in self._fileinfo['dtype'].fields:
            if not fld in self._accessors:
                self._accessors[fld] = ( accessor( dtype, fld ), None )
        
        self._source = self._memorymap
        self._klass = _DataProxy_MemoryMap
        
        #set up generic access to file records
        self.raw = self._klass(self._source, None)
        
        self.add_accessors( accessors )
    
    def add_accessors(self, accessors={}):
        #set up access to individual fields
        for k,v in accessors.iteritems():
            self.__setattr__(k, self._klass(self._source, *v))
        
        self._accessors.update( accessors )
    
    def __getitem__(self,key):
        if 'default' in self._accessors:
            return self.default.__getitem__(key)
        else:
            return self.raw.__getitem__(key)
        
    @property
    def shape(self):
        if 'default' in self._accessors:
            return self.default.shape
        else:
            return self.raw.shape
        
    def __len__(self):
        if 'default' in self._accessors:
            return len(self.default)
        else:
            return (self.raw)


class _DataProxy_MemoryMap(object):
    def __init__(self, memorymap, accessors=None, indexer=None):
        
        self._memorymap = memorymap
        
        self._accessors = accessors
        
        if not isinstance(self._accessors, list) and not self._accessors is None:
            self._accessors = [ self._accessors ]
        
        self._indexer = indexer
        
        if self._indexer is not None and self._accessors is not None:
            if not all( [ c.nsamples()==self._indexer.nsamples_per_record() for c in self._accessors] ):
                raise ValueError('Accessors have incompatible number of samples')
        
        self._nrecords = self._memorymap.__len__()
        self.ndim = 1
        self.shape = ( self._nrecords, )
        
        self.dtype = memorymap.dtype
        
        if self._accessors is not None:
            if self._indexer is None:
                self.shape = [ (self._nrecords,) + c.record_shape() for c in self._accessors ]
            else:
                self.shape = [ (self._indexer.nsamples(),) + tuple([ x for (idx,x) in enumerate(c.record_shape()) if idx!=c.sample_dim() ]) for c in self._accessors ]
            
            self.dtype = [ np.float64, ] * len(self._accessors)
            
            if len(self.shape)==1:
                self.shape = self.shape[0]
                self.dtype = self.dtype[0]
        
        
        if not self._accessors is None and len(self._accessors)>1:
            self._return_factory = namedtuple( 'data',  [ c.label() for c in self._accessors ] )
        
    def __len__(self):
        if self._indexer is None:
            return self._nrecords
        else:
            return self._indexer.nsamples()
        
    def __getitem__(self,key):
        
        if isinstance(key,tuple):
            if len(key)!=1:
                raise KeyError('too many indices for array')
            else:
                key = key[0]
        
        # handle case of reading complete records
        if self._memorymap.dtype.fields is None or self._accessors is None:
            return np.array(self._memorymap.__getitem__(key))
        
        # determine fields to read
        fields_to_read = set()
        for c in self._accessors:
            fields_to_read.update( c.fields() )
        
        result = []
        
        if self._indexer is None: # load by record
            
            if isinstance(key,int): #make sure that loading data below keeps the first records axis
                key = slice(key,key+1)
            
            in_data = { fld:np.array(self._memorymap[fld][key]) for fld in fields_to_read }
            # perform record level processing for each accessor
            for c in self._accessors:
                selected_data = { k:in_data[k] for k in c.fields() }
                result.append( c.process_records( **selected_data ) )
        
        else:
            
            if False: #all( [ c['record_func'] is None for c in self._accessors ] ):
                # construct unraveled indices and directly index memorymap
                # perform index level processing
                # prepare outputs
                pass
            else:
                # convert key to records index, sub index
                if isinstance(key,slice):
                    start,stop,step = key.indices( len(self) )
                    indices = np.arange(start,stop,step,dtype=np.int64)
                else:
                    indices = np.array( key, dtype=np.int64 )
                
                record_indices, sub_indices = self._indexer( indices )
                
                unique_record_indices, unique_inverse = np.unique( record_indices, return_inverse=True )
                
                # load records
                in_data = { fld:np.array(self._memorymap[fld][unique_record_indices]) for fld in fields_to_read }
                
                for c in self._accessors:
                    
                    selected_data = { k:in_data[k] for k in c.fields() }
                    tmp = c.process_records( **selected_data )
                    
                    # perform indexing
                    t = [slice(None)]*( tmp.ndim )
                    t[0] = unique_inverse
                    t[c.sample_dim()+1] = sub_indices
                    
                    tmp = tmp[t]
                    
                    result.append( c.process_samples( tmp ) )
        
        if len(result)==1:
            return result[0]
        else:
            return self._return_factory._make( result )


